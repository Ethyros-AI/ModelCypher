"""Chat template library for major LLM families.

Provides formatting for instruction tuning and multi-turn conversations.
"""

from __future__ import annotations

from enum import Enum


from modelcypher.core.domain.dataset.chat_message import ChatMessage


class ChatTemplate(str, Enum):
    """Chat templates for major LLM families."""

    # Llama Family
    LLAMA3 = "llama3"
    LLAMA2 = "llama2"
    LLAMA3_VISION = "llama3_vision"

    # Qwen Family
    QWEN2 = "qwen2"
    QWEN3 = "qwen3"
    QWEN2_VL = "qwen2vl"

    # Gemma Family
    GEMMA2 = "gemma2"
    GEMMA3 = "gemma3"
    GEMMA3N = "gemma3n"

    # Mistral Family
    MISTRAL = "mistral"
    MISTRAL_V3 = "mistral_v3"
    PIXTRAL = "pixtral"

    # Phi Family
    PHI3 = "phi3"
    PHI4 = "phi4"

    # Cohere Family
    COMMAND_R = "command_r"
    COMMAND_R_V2 = "command_r_v2"

    # DeepSeek Family
    DEEPSEEK = "deepseek"
    DEEPSEEK_R1 = "deepseek_r1"

    # Granite Family
    GRANITE = "granite"

    # Other
    GPT_OSS = "gpt_oss"

    # Classic Instruction Formats
    ALPACA = "alpaca"
    VICUNA = "vicuna"
    ZEPHYR = "zephyr"
    CHATML = "chatml"

    # --- Instruction Tuning Format ---

    def format_instruction(
        self,
        instruction: str,
        output: str,
        input_text: str | None = None,
        system_prompt: str = "[Environment context.]",
    ) -> str:
        """Format a conversation turn for instruction tuning.

        Args:
            instruction: User input/question.
            output: Assistant response.
            input_text: Optional additional context (Alpaca-style).
            system_prompt: Optional system prompt.

        Returns:
            Formatted text ready for tokenization.
        """
        if self in (ChatTemplate.LLAMA3, ChatTemplate.LLAMA3_VISION):
            result = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>"
            result += f"<|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|>"
            result += f"<|start_header_id|>assistant<|end_header_id|>\n\n{output}<|eot_id|>"
            return result

        if self == ChatTemplate.LLAMA2:
            return f"<s>[INST] {instruction} [/INST] {output} </s>"

        if self in (ChatTemplate.QWEN2, ChatTemplate.QWEN3, ChatTemplate.QWEN2_VL):
            result = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            result += f"<|im_start|>user\n{instruction}<|im_end|>\n"
            result += f"<|im_start|>assistant\n{output}<|im_end|>"
            return result

        if self in (ChatTemplate.GEMMA2, ChatTemplate.GEMMA3, ChatTemplate.GEMMA3N):
            result = f"<bos><start_of_turn>user\n{instruction}<end_of_turn>\n"
            result += f"<start_of_turn>model\n{output}<end_of_turn>"
            return result

        if self in (ChatTemplate.MISTRAL, ChatTemplate.MISTRAL_V3, ChatTemplate.PIXTRAL):
            return f"<s>[INST] {instruction} [/INST] {output}</s>"

        if self in (ChatTemplate.PHI3, ChatTemplate.PHI4):
            result = f"<|user|>\n{instruction}<|end|>\n"
            result += f"<|assistant|>\n{output}<|end|>"
            return result

        if self in (ChatTemplate.COMMAND_R, ChatTemplate.COMMAND_R_V2):
            result = f"<|START_OF_TURN_TOKEN|><|USER_TOKEN|>{instruction}<|END_OF_TURN_TOKEN|>"
            result += f"<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>{output}<|END_OF_TURN_TOKEN|>"
            return result

        if self in (ChatTemplate.DEEPSEEK, ChatTemplate.DEEPSEEK_R1):
            return f"User: {instruction}\n\nAssistant: {output}"

        if self == ChatTemplate.GRANITE:
            return f"<|user|>\n{instruction}\n<|assistant|>\n{output}"

        if self == ChatTemplate.GPT_OSS:
            result = f"<|im_start|>user\n{instruction}<|im_end|>\n"
            result += f"<|im_start|>assistant\n{output}<|im_end|>"
            return result

        if self == ChatTemplate.ALPACA:
            result = f"### Instruction:\n{instruction}\n\n"
            if input_text:
                result += f"### Input:\n{input_text}\n\n"
            result += f"### Response:\n{output}"
            return result

        if self == ChatTemplate.VICUNA:
            return f"USER: {instruction}\nASSISTANT: {output}"

        if self == ChatTemplate.ZEPHYR:
            result = f"<|system|>\n{system_prompt}</s>\n"
            result += f"<|user|>\n{instruction}</s>\n"
            result += f"<|assistant|>\n{output}</s>"
            return result

        if self == ChatTemplate.CHATML:
            result = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            result += f"<|im_start|>user\n{instruction}<|im_end|>\n"
            result += f"<|im_start|>assistant\n{output}<|im_end|>"
            return result

        # Default to ChatML
        return self._format_chatml_instruction(instruction, output, system_prompt)

    def _format_chatml_instruction(
        self, instruction: str, output: str, system_prompt: str
    ) -> str:
        """Default ChatML formatting."""
        result = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        result += f"<|im_start|>user\n{instruction}<|im_end|>\n"
        result += f"<|im_start|>assistant\n{output}<|im_end|>"
        return result

    # --- Multi-Turn Chat Format ---

    def format_messages(self, messages: list[ChatMessage]) -> str:
        """Format a multi-turn conversation.

        Args:
            messages: Array of chat messages with role and content.

        Returns:
            Formatted text ready for tokenization.
        """
        if self in (ChatTemplate.MISTRAL, ChatTemplate.MISTRAL_V3, ChatTemplate.PIXTRAL):
            return self._format_mistral(messages)

        if self in (ChatTemplate.LLAMA3, ChatTemplate.LLAMA3_VISION):
            return self._format_llama3(messages)

        if self == ChatTemplate.LLAMA2:
            return self._format_llama2(messages)

        if self in (ChatTemplate.PHI3, ChatTemplate.PHI4):
            return self._format_phi(messages)

        if self in (ChatTemplate.QWEN2, ChatTemplate.QWEN3, ChatTemplate.QWEN2_VL):
            return self._format_qwen(messages)

        if self in (ChatTemplate.GEMMA2, ChatTemplate.GEMMA3, ChatTemplate.GEMMA3N):
            return self._format_gemma(messages)

        if self in (ChatTemplate.COMMAND_R, ChatTemplate.COMMAND_R_V2):
            return self._format_command_r(messages)

        if self in (ChatTemplate.DEEPSEEK, ChatTemplate.DEEPSEEK_R1):
            return self._format_deepseek(messages)

        if self == ChatTemplate.GRANITE:
            return self._format_granite(messages)

        if self == ChatTemplate.GPT_OSS:
            return self._format_gpt_oss(messages)

        if self == ChatTemplate.ALPACA:
            return self._format_alpaca(messages)

        if self == ChatTemplate.VICUNA:
            return self._format_vicuna(messages)

        if self == ChatTemplate.ZEPHYR:
            return self._format_zephyr(messages)

        if self == ChatTemplate.CHATML:
            return self._format_chatml(messages)

        # Default to ChatML
        return self._format_chatml(messages)

    def _format_mistral(self, messages: list[ChatMessage]) -> str:
        """Format for Mistral/Mixtral."""
        result = ""
        system_prompt: str | None = None

        if messages and messages[0].role == "system":
            system_prompt = messages[0].content

        conversation = [m for m in messages if m.role != "system"]

        for i, message in enumerate(conversation):
            if message.role == "user":
                user_content = message.content
                if i == 0 and system_prompt:
                    user_content = f"{system_prompt}\n\n{message.content}"
                result += f"<s>[INST] {user_content} [/INST]"
            elif message.role == "assistant":
                result += f" {message.content}</s>"

        return result

    def _format_llama3(self, messages: list[ChatMessage]) -> str:
        """Format for Llama 3."""
        result = "<|begin_of_text|>"
        for message in messages:
            result += f"<|start_header_id|>{message.role}<|end_header_id|>\n\n{message.content}<|eot_id|>"
        return result

    def _format_llama2(self, messages: list[ChatMessage]) -> str:
        """Format for Llama 2."""
        result = ""
        for message in messages:
            if message.role == "user":
                result += f"<s>[INST] {message.content} [/INST]"
            elif message.role == "assistant":
                result += f" {message.content}</s>"
        return result

    def _format_phi(self, messages: list[ChatMessage]) -> str:
        """Format for Phi."""
        result = ""
        for message in messages:
            result += f"<|{message.role}|>\n{message.content}<|end|>\n"
        return result

    def _format_qwen(self, messages: list[ChatMessage]) -> str:
        """Format for Qwen."""
        result = ""
        for message in messages:
            result += f"<|im_start|>{message.role}\n{message.content}<|im_end|>\n"
        return result

    def _format_gemma(self, messages: list[ChatMessage]) -> str:
        """Format for Gemma."""
        result = ""
        for message in messages:
            if message.role == "user":
                result += f"<start_of_turn>user\n{message.content}<end_of_turn>\n"
            elif message.role == "assistant":
                result += f"<start_of_turn>model\n{message.content}<end_of_turn>\n"
        return result

    def _format_command_r(self, messages: list[ChatMessage]) -> str:
        """Format for Command-R (Cohere)."""
        result = ""
        for message in messages:
            result += f"<|START_OF_TURN_TOKEN|><|{message.role.upper()}_TOKEN|>{message.content}<|END_OF_TURN_TOKEN|>"
        return result

    def _format_deepseek(self, messages: list[ChatMessage]) -> str:
        """Format for DeepSeek."""
        return "\n\n".join(
            f"{m.role.capitalize()}: {m.content}" for m in messages
        )

    def _format_granite(self, messages: list[ChatMessage]) -> str:
        """Format for Granite."""
        return "\n".join(f"<|{m.role}|>\n{m.content}" for m in messages)

    def _format_gpt_oss(self, messages: list[ChatMessage]) -> str:
        """Format for GPT-OSS."""
        result = ""
        for message in messages:
            result += f"<|im_start|>{message.role}\n{message.content}<|im_end|>\n"
        return result

    def _format_alpaca(self, messages: list[ChatMessage]) -> str:
        """Format for Alpaca."""
        return "\n\n".join(
            f"### {m.role.capitalize()}:\n{m.content}" for m in messages
        )

    def _format_vicuna(self, messages: list[ChatMessage]) -> str:
        """Format for Vicuna."""
        return "\n".join(f"{m.role.upper()}: {m.content}" for m in messages)

    def _format_zephyr(self, messages: list[ChatMessage]) -> str:
        """Format for Zephyr."""
        result = ""
        for message in messages:
            result += f"<|{message.role}|>\n{message.content}</s>\n"
        return result

    def _format_chatml(self, messages: list[ChatMessage]) -> str:
        """Format for ChatML."""
        result = ""
        for message in messages:
            result += f"<|im_start|>{message.role}\n{message.content}<|im_end|>\n"
        return result

    # --- Auto-Detection ---

    @classmethod
    def detect(cls, model_name: str) -> ChatTemplate:
        """Auto-detect template from model name.

        Args:
            model_name: Model identifier (e.g., "meta-llama/Llama-3.1-8B").

        Returns:
            Most appropriate template for the model.
        """
        name = model_name.lower()

        # Llama
        if "llama-3" in name or "llama3" in name:
            if "vision" in name or "vlm" in name:
                return cls.LLAMA3_VISION
            return cls.LLAMA3
        if "llama-2" in name or "llama2" in name:
            return cls.LLAMA2

        # Qwen
        if "qwen2vl" in name or "qwen-2-vl" in name:
            return cls.QWEN2_VL
        if "qwen3" in name or "qwen-3" in name:
            return cls.QWEN3
        if "qwen2" in name or "qwen-2" in name:
            return cls.QWEN2

        # Gemma
        if "gemma-3n" in name or "gemma3n" in name:
            return cls.GEMMA3N
        if "gemma-3" in name or "gemma3" in name:
            return cls.GEMMA3
        if "gemma-2" in name or "gemma2" in name or "gemma" in name:
            return cls.GEMMA2

        # Mistral
        if "pixtral" in name:
            return cls.PIXTRAL
        if "mistral" in name and "v3" in name:
            return cls.MISTRAL_V3
        if "mistral" in name or "mixtral" in name:
            return cls.MISTRAL

        # Phi
        if "phi-4" in name or "phi4" in name:
            return cls.PHI4
        if "phi-3" in name or "phi3" in name:
            return cls.PHI3

        # Cohere
        if "command-r" in name and "v2" in name:
            return cls.COMMAND_R_V2
        if "command-r" in name or "command_r" in name:
            return cls.COMMAND_R

        # DeepSeek
        if "deepseek-r1" in name or "deepseek_r1" in name:
            return cls.DEEPSEEK_R1
        if "deepseek" in name:
            return cls.DEEPSEEK

        # Granite
        if "granite" in name:
            return cls.GRANITE

        # GPT-OSS
        if "gpt-oss" in name:
            return cls.GPT_OSS

        # Classic formats
        if "alpaca" in name:
            return cls.ALPACA
        if "vicuna" in name:
            return cls.VICUNA
        if "zephyr" in name:
            return cls.ZEPHYR

        # Default to ChatML for unknown models
        return cls.CHATML

    # --- Display Properties ---

    @property
    def display_name(self) -> str:
        """Human-readable display name."""
        names = {
            ChatTemplate.LLAMA3: "Llama 3",
            ChatTemplate.LLAMA2: "Llama 2",
            ChatTemplate.LLAMA3_VISION: "Llama 3 Vision",
            ChatTemplate.QWEN2: "Qwen 2",
            ChatTemplate.QWEN3: "Qwen 3",
            ChatTemplate.QWEN2_VL: "Qwen 2 VL",
            ChatTemplate.GEMMA2: "Gemma 2",
            ChatTemplate.GEMMA3: "Gemma 3",
            ChatTemplate.GEMMA3N: "Gemma 3n",
            ChatTemplate.MISTRAL: "Mistral",
            ChatTemplate.MISTRAL_V3: "Mistral v3",
            ChatTemplate.PIXTRAL: "Pixtral",
            ChatTemplate.PHI3: "Phi-3",
            ChatTemplate.PHI4: "Phi-4",
            ChatTemplate.COMMAND_R: "Command R",
            ChatTemplate.COMMAND_R_V2: "Command R v2",
            ChatTemplate.DEEPSEEK: "DeepSeek",
            ChatTemplate.DEEPSEEK_R1: "DeepSeek R1",
            ChatTemplate.GRANITE: "Granite",
            ChatTemplate.GPT_OSS: "GPT-OSS",
            ChatTemplate.ALPACA: "Alpaca",
            ChatTemplate.VICUNA: "Vicuna",
            ChatTemplate.ZEPHYR: "Zephyr",
            ChatTemplate.CHATML: "ChatML (Universal)",
        }
        return names.get(self, self.value)

    @property
    def supports_vision(self) -> bool:
        """Whether this template supports vision/multimodal inputs."""
        return self in (
            ChatTemplate.LLAMA3_VISION,
            ChatTemplate.QWEN2_VL,
            ChatTemplate.PIXTRAL,
        )

    @property
    def description(self) -> str:
        """Human-readable description of the template."""
        descriptions = {
            ChatTemplate.LLAMA3: "Llama 3 format with header tags",
            ChatTemplate.LLAMA3_VISION: "Llama 3 format with header tags",
            ChatTemplate.LLAMA2: "Llama 2 format with [INST] tags",
            ChatTemplate.QWEN2: "Qwen ChatML format with im_start/im_end",
            ChatTemplate.QWEN3: "Qwen ChatML format with im_start/im_end",
            ChatTemplate.QWEN2_VL: "Qwen ChatML format with im_start/im_end",
            ChatTemplate.GEMMA2: "Gemma turn-based format",
            ChatTemplate.GEMMA3: "Gemma turn-based format",
            ChatTemplate.GEMMA3N: "Gemma turn-based format",
            ChatTemplate.MISTRAL: "Mistral/Mixtral format with [INST] tags",
            ChatTemplate.MISTRAL_V3: "Mistral/Mixtral format with [INST] tags",
            ChatTemplate.PIXTRAL: "Mistral/Mixtral format with [INST] tags",
            ChatTemplate.PHI3: "Phi format with role markers",
            ChatTemplate.PHI4: "Phi format with role markers",
            ChatTemplate.COMMAND_R: "Command-R (Cohere) format",
            ChatTemplate.COMMAND_R_V2: "Command-R (Cohere) format",
            ChatTemplate.DEEPSEEK: "DeepSeek format",
            ChatTemplate.DEEPSEEK_R1: "DeepSeek format",
            ChatTemplate.GRANITE: "Granite (IBM) format",
            ChatTemplate.GPT_OSS: "GPT-OSS ChatML variant",
            ChatTemplate.ALPACA: "Alpaca instruction format",
            ChatTemplate.VICUNA: "Vicuna conversation format",
            ChatTemplate.ZEPHYR: "Zephyr format with system prompts",
            ChatTemplate.CHATML: "ChatML universal format",
        }
        return descriptions.get(self, "Unknown format")

    @property
    def family(self) -> str:
        """Model family grouping for compatibility checking."""
        families = {
            ChatTemplate.LLAMA3: "llama",
            ChatTemplate.LLAMA2: "llama",
            ChatTemplate.LLAMA3_VISION: "llama",
            ChatTemplate.QWEN2: "qwen",
            ChatTemplate.QWEN3: "qwen",
            ChatTemplate.QWEN2_VL: "qwen",
            ChatTemplate.GEMMA2: "gemma",
            ChatTemplate.GEMMA3: "gemma",
            ChatTemplate.GEMMA3N: "gemma",
            ChatTemplate.MISTRAL: "mistral",
            ChatTemplate.MISTRAL_V3: "mistral",
            ChatTemplate.PIXTRAL: "mistral",
            ChatTemplate.PHI3: "phi",
            ChatTemplate.PHI4: "phi",
            ChatTemplate.COMMAND_R: "cohere",
            ChatTemplate.COMMAND_R_V2: "cohere",
            ChatTemplate.DEEPSEEK: "deepseek",
            ChatTemplate.DEEPSEEK_R1: "deepseek",
            ChatTemplate.GRANITE: "granite",
            ChatTemplate.GPT_OSS: "gpt",
            ChatTemplate.ALPACA: "classic",
            ChatTemplate.VICUNA: "classic",
            ChatTemplate.ZEPHYR: "classic",
            ChatTemplate.CHATML: "classic",
        }
        return families.get(self, "unknown")

    @property
    def example(self) -> str:
        """Example output for this template."""
        sample_messages = [
            ChatMessage(role="user", content="Hello!"),
            ChatMessage(role="assistant", content="Hi there!"),
        ]
        return self.format_messages(sample_messages)
