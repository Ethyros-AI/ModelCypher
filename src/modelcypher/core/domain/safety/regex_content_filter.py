"""
Regex Content Filter.

Fast local pattern filter for dangerous content (destructive commands, prompt injection, PII)
that runs before LLM moderation.

Ported 1:1 from the reference Swift implementation.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Set, Pattern, Union

logger = logging.getLogger(__name__)


class SafetyCategory(str, Enum):
    DANGEROUS_CODE = "dangerous_code"
    PROMPT_INJECTION = "prompt_injection"
    PII = "pii"
    SELF_HARM = "self_harm"
    VIOLENCE = "violence"
    HATE_SPEECH = "hate_speech"
    SEXUAL = "sexual"
    HARASSMENT = "harassment"


class SafetyStatus(str, Enum):
    REJECTED = "rejected"
    FLAGGED_FOR_REVIEW = "flagged_for_review"


class DatasetPurpose(str, Enum):
    GENERAL = "general"
    CODE_TRAINING = "code_training"
    MEDICAL = "medical"
    LEGAL = "legal"
    
    @property
    def whitelisted_rule_ids(self) -> Set[str]:
        if self == DatasetPurpose.CODE_TRAINING:
            return {"shell_commands", "code_execution", "rm_root"}
        return set()


@dataclass(frozen=True)
class ContentFilterResult:
    status: SafetyStatus
    reason: str
    category: Optional[SafetyCategory]
    rule_id: str
    matched_text: str


class RuleAction(str, Enum):
    REJECT = "reject"
    FLAG = "flag"


@dataclass(frozen=True)
class FilterRule:
    id: str
    expression: Pattern
    category: Optional[SafetyCategory]
    action: RuleAction
    reason: str


class RegexContentFilter:
    """
    Fast local pattern filter for dangerous content.
    """
    
    _pii_email_whitelist = {
        "example.com", "example.net", "example.org", "test.com", "localhost"
    }

    def __init__(self, rules: List[FilterRule]):
        self.rules = rules

    def check(
        self,
        text: str,
        purpose: DatasetPurpose = DatasetPurpose.GENERAL,
        custom_whitelist: Optional[Set[str]] = None,
    ) -> Optional[ContentFilterResult]:
        """Scans text for unsafe patterns."""
        if not text:
            return None
            
        custom_whitelist = custom_whitelist or set()
        whitelist = purpose.whitelisted_rule_ids.union(custom_whitelist)
        
        for rule in self.rules:
            if rule.id in whitelist:
                continue
                
            match = rule.expression.search(text)
            if match:
                matched_text = match.group(0)
                
                # Special handling for email whitelist
                if rule.id == "pii_email":
                    domain = self._domain_from_email(matched_text)
                    if domain and domain.lower() in self._pii_email_whitelist:
                        continue
                        
                status = SafetyStatus.REJECTED if rule.action == RuleAction.REJECT else SafetyStatus.FLAGGED_FOR_REVIEW
                
                return ContentFilterResult(
                    status=status,
                    reason=rule.reason,
                    category=rule.category,
                    rule_id=rule.id,
                    matched_text=matched_text
                )
        
        return None

    @staticmethod
    def _domain_from_email(email: str) -> Optional[str]:
        try:
            parts = email.split('@')
            if len(parts) == 2:
                return parts[1]
        except:
            pass
        return None

    @classmethod
    def default(cls) -> "RegexContentFilter":
        return cls(RegexContentFilter._build_default_rules())

    @staticmethod
    def _build_default_rules() -> List[FilterRule]:
        def make_rule(
            id: str,
            pattern: str,
            action: RuleAction,
            reason: str,
            category: Optional[SafetyCategory] = None,
            case_insensitive: bool = True
        ) -> FilterRule:
            flags = re.IGNORECASE if case_insensitive else 0
            try:
                # Use multiline to match start/end of lines correctly if needed
                regex = re.compile(pattern, flags | re.MULTILINE)
            except re.error as e:
                logger.error("Failed to compile regex rule %s: %s", id, e)
                regex = re.compile(r"a^") # Fail-safe (matches nothing)
                
            return FilterRule(id, regex, category, action, reason)

        return [
            # Dangerous Code
            make_rule(
                "rm_root", r"rm\s+-rf\s+\/",
                RuleAction.REJECT, "Destructive shell command", SafetyCategory.DANGEROUS_CODE
            ),
            make_rule(
                "format_disk", r"diskutil\s+eraseDisk",
                RuleAction.FLAG, "Disk erasure command", SafetyCategory.DANGEROUS_CODE
            ),
            make_rule(
                "shell_commands", r"(?:sudo\s+)?(?:rm|chmod|chown)\b",
                RuleAction.FLAG, "Shell command detected", SafetyCategory.DANGEROUS_CODE
            ),
            make_rule(
                "fork_bomb", r":\(\)\s*\{\s*:\|:&\s*\}\s*;",
                RuleAction.REJECT, "Fork bomb detected", SafetyCategory.DANGEROUS_CODE
            ),
            
            # Prompt Injection
            make_rule(
                "sql_injection", r"(?:union\s+select|drop\s+table|;--)",
                RuleAction.FLAG, "Potential SQL injection pattern", SafetyCategory.PROMPT_INJECTION
            ),
            make_rule(
                "prompt_override", r"ignore\s+the\s+previous\s+instructions",
                RuleAction.FLAG, "Prompt injection attempt", SafetyCategory.PROMPT_INJECTION
            ),
             make_rule(
                "prompt_system_override", r"(?:system\s*:\s*|<\|system\|>|<system>|###\s*system)",
                RuleAction.FLAG, "System prompt override attempt", SafetyCategory.PROMPT_INJECTION
            ),
            make_rule(
                "prompt_jailbreak", r"(?:DAN\s+mode|jailbreak|act\s+as\s+(?:DAN|uncensored|unfiltered)|ignore\s+(?:all\s+)?(?:safety|ethical)\s+(?:guidelines|rules))",
                RuleAction.REJECT, "Jailbreak attempt detected", SafetyCategory.PROMPT_INJECTION
            ),

            # PII
            make_rule(
                "pii_ssn", r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b",
                RuleAction.REJECT, "Possible SSN detected", SafetyCategory.PII
            ),
            make_rule(
                "pii_credit_card", r"\b(?:\d{4}[- ]?){3}\d{4}\b",
                RuleAction.FLAG, "Possible credit card number", SafetyCategory.PII
            ),
            make_rule(
                "pii_email", r"\b[\w\.-]+@[\w\.-]+\.\w{2,}\b",
                RuleAction.FLAG, "Email address detected", SafetyCategory.PII
            ),
             make_rule(
                "pii_phone_us", r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b",
                RuleAction.FLAG, "US phone number detected", SafetyCategory.PII
            ),
            make_rule(
                "pii_ipv4", r"\b(?:(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)\.){3}(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)\b",
                RuleAction.FLAG, "IPv4 address detected", SafetyCategory.PII
            ),
            make_rule(
                "pii_aws_key", r"\b(?:AKIA|ABIA|ACCA|ASIA)[A-Z0-9]{16}\b",
                RuleAction.REJECT, "AWS access key detected", SafetyCategory.PII, case_insensitive=False
            ),
            make_rule(
                "pii_aws_secret", r"\b[A-Za-z0-9+/]{40}\b",
                RuleAction.FLAG, "Possible AWS secret key", SafetyCategory.PII, case_insensitive=False
            ),

            # Toxicity
            make_rule(
                "toxicity_self_harm", r"(?:how\s+to\s+(?:kill|hurt)\s+(?:myself|yourself)|(?:want|going)\s+to\s+(?:die|end\s+it)|suicide\s+(?:method|instruction))",
                RuleAction.REJECT, "Self-harm content detected", SafetyCategory.SELF_HARM
            ),
            make_rule(
                "toxicity_violence", r"(?:(?:kill|murder|attack|bomb|shoot)\s+(?:people|person|everyone|them)|(?:mass|school)\s+shooting|(?:terrorist|terror)\s+attack)",
                RuleAction.REJECT, "Violence/threat content detected", SafetyCategory.VIOLENCE
            ),
            make_rule(
                "toxicity_hate_slurs", r"\b(?:n[i1]gg[ae3]r|f[a4]gg[o0]t|k[i1]ke|sp[i1]c|ch[i1]nk|wetback|towelhead)\b",
                RuleAction.REJECT, "Hate speech/slur detected", SafetyCategory.HATE_SPEECH
            ),
            make_rule(
                "toxicity_dehumanize", r"(?:(?:all|those)\s+\w+\s+(?:should|must|need\s+to)\s+(?:die|be\s+killed|be\s+eliminated))",
                RuleAction.REJECT, "Dehumanizing content detected", SafetyCategory.HATE_SPEECH
            ),
            make_rule(
                "toxicity_csam_indicators", r"(?:(?:child|minor|kid|underage)\s+(?:porn|sex|nude|naked)|(?:sexual|intimate)\s+(?:with|involving)\s+(?:child|minor|kid))",
                RuleAction.REJECT, "CSAM-related content detected", SafetyCategory.SEXUAL
            ),
            make_rule(
                "toxicity_harassment", r"(?:(?:dox|doxx)(?:ing|ed)?|(?:share|post|leak)\s+(?:their|his|her)\s+(?:address|phone|personal))",
                RuleAction.REJECT, "Harassment/doxxing content detected", SafetyCategory.HARASSMENT
            ),
        ]
