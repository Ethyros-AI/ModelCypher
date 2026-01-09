#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import json
import re
import shlex
import subprocess
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Issue:
    kind: str
    file: str
    line: int
    message: str
    context: str | None = None


@dataclass(frozen=True)
class AuditSummary:
    files_scanned: int
    issues_found: int
    issues_by_kind: dict[str, int]


@dataclass(frozen=True)
class AuditResult:
    summary: AuditSummary
    issues: list[Issue]
    arxiv: dict[str, object]
    cli: dict[str, object]


def _run(cmd: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=False,
    )


def _git_tracked_markdown_files(*, repo_root: Path) -> list[Path]:
    proc = _run(["git", "ls-files", "*.md"], cwd=repo_root)
    if proc.returncode != 0:
        raise RuntimeError(f"git ls-files failed: {proc.stderr.strip()}")
    files = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        files.append(repo_root / line)
    return sorted(files)


def _all_markdown_files_on_disk(*, repo_root: Path) -> list[Path]:
    ignored_parts = {".git", ".venv", "node_modules", ".pytest_cache"}
    files: list[Path] = []
    for path in repo_root.rglob("*.md"):
        if any(part in ignored_parts for part in path.parts):
            continue
        files.append(path)
    return sorted(files)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


ABSOLUTE_PATH_PATTERNS = [
    re.compile(r"(?<!\w)/Users/[^ \n\t]+"),
    re.compile(r"(?<!\w)/home/[^ \n\t]+"),
    re.compile(r"(?<!\w)/Volumes/[^ \n\t]+"),
    re.compile(r"(?<!\w)~/(?:[^ \n\t]+)"),
    re.compile(r"\b[A-Za-z]:\\[^ \n\t]+"),
]

HTTP_URL_RE = re.compile(r"\bhttp://[^\s)>\"]+")
TODO_RE = re.compile(r"\b(TODO|TBD|FIXME|XXX)\b")
PIP_INSTALL_RE = re.compile(r"\b(pip3?\s+install|python\s+-m\s+pip\s+install|conda\s+install)\b")
HF_CLI_RE = re.compile(r"\bhuggingface-cli\b")

# Markdown links, excluding images. This is intentionally simple and best-effort.
MD_LINK_RE = re.compile(r"(?<!\!)\[[^\]]+\]\(([^)]+)\)")

# A conservative "command-like token" heuristic for CLI path validation.
COMMAND_TOKEN_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9\-]*$")


def _is_external_link(target: str) -> bool:
    return target.startswith(("http://", "https://"))


def _strip_link_target(target: str) -> str:
    # Remove optional title: (path "title") or (path 'title')
    # Keep only the first whitespace-separated token.
    return target.strip().split()[0]


def _split_link_target(target: str) -> tuple[str, str | None]:
    if "#" not in target:
        return target, None
    base, frag = target.split("#", 1)
    return base, frag or None


def _normalize_heading_text(text: str) -> str:
    # Remove inline code markers and lightweight formatting.
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    return text.strip()


def _slug_candidates(heading_text: str) -> list[str]:
    """
    Generate a small set of anchor candidates, because different renderers vary
    on how they treat diacritics/unicode. We accept any candidate match.
    """
    raw = _normalize_heading_text(heading_text).lower()

    candidates: list[str] = []

    # Candidate 1: NFKD → ASCII (diacritics folded).
    folded = unicodedata.normalize("NFKD", raw).encode("ascii", "ignore").decode("ascii")
    candidates.append(folded)

    # Candidate 2: strip non-ascii without folding (drops diacritics entirely).
    dropped = raw.encode("ascii", "ignore").decode("ascii")
    candidates.append(dropped)

    # Candidate 3: keep unicode, but remove most punctuation.
    candidates.append(raw)

    slugs: list[str] = []
    for c in candidates:
        c = c.strip()
        c = re.sub(r"[^\w\s\-]", "", c, flags=re.UNICODE)
        c = re.sub(r"\s+", "-", c)
        c = re.sub(r"-+", "-", c)
        c = c.strip("-")
        if c and c not in slugs:
            slugs.append(c)
    return slugs


def _extract_heading_anchors(markdown_text: str) -> set[str]:
    anchors: set[str] = set()
    counts: Counter[str] = Counter()

    for line in markdown_text.splitlines():
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if not m:
            continue
        heading_text = m.group(2).strip()
        heading_text = heading_text.split("{#")[0].strip()  # ignore explicit anchors, if any
        for base in _slug_candidates(heading_text):
            slug = base
            counts[slug] += 1
            if counts[slug] == 1:
                anchors.add(slug)
            else:
                anchors.add(f"{slug}-{counts[slug]-1}")
    return anchors


def _resolve_internal_path(*, source_file: Path, target: str, repo_root: Path) -> Path | None:
    target = target.strip()
    if not target or target.startswith("#"):
        return source_file
    if target.startswith(("mailto:", "tel:")):
        return None
    if _is_external_link(target):
        return None
    # Preserve querystring? Usually irrelevant for local paths; drop it.
    target = target.split("?", 1)[0]
    base, _frag = _split_link_target(target)
    if not base:
        return source_file
    resolved = (source_file.parent / base).resolve()
    try:
        resolved.relative_to(repo_root.resolve())
    except Exception:
        # Link escapes repo root: treat as non-local for purposes of this audit.
        return None
    return resolved


def _iter_markdown_links(markdown_text: str) -> Iterable[str]:
    for match in MD_LINK_RE.finditer(markdown_text):
        yield match.group(1)


def _find_line_number(haystack_lines: list[str], needle: str) -> int:
    for idx, line in enumerate(haystack_lines, start=1):
        if needle in line:
            return idx
    return 1


def _audit_markdown_file(
    *,
    path: Path,
    repo_root: Path,
    anchor_cache: dict[Path, set[str]],
    pdf_files: set[str],
    pdf_root: Path,
    arxiv_ids_in_script: set[str],
    stale_arxiv_pdf_count: int,
) -> list[Issue]:
    rel = str(path.relative_to(repo_root))
    text = _read_text(path)
    lines = text.splitlines()
    issues: list[Issue] = []

    # Absolute paths (leaks machine-specific details)
    for pat in ABSOLUTE_PATH_PATTERNS:
        for match in pat.finditer(text):
            needle = match.group(0)
            line_no = _find_line_number(lines, needle)
            issues.append(
                Issue(
                    kind="absolute_path",
                    file=rel,
                    line=line_no,
                    message=f"Machine-specific path: {needle}",
                    context=lines[line_no - 1].strip() if 1 <= line_no <= len(lines) else None,
                )
            )

    # http:// links
    for match in HTTP_URL_RE.finditer(text):
        url = match.group(0)
        line_no = _find_line_number(lines, url)
        issues.append(
            Issue(
                kind="http_url",
                file=rel,
                line=line_no,
                message=f"Non-HTTPS URL: {url}",
                context=lines[line_no - 1].strip() if 1 <= line_no <= len(lines) else None,
            )
        )

    # TODO/TBD markers
    for idx, line in enumerate(lines, start=1):
        if TODO_RE.search(line):
            issues.append(
                Issue(
                    kind="todo_tbd",
                    file=rel,
                    line=idx,
                    message="Contains TODO/TBD/FIXME marker",
                    context=line.strip(),
                )
            )

    # Poetry vs pip
    for idx, line in enumerate(lines, start=1):
        if PIP_INSTALL_RE.search(line):
            issues.append(
                Issue(
                    kind="non_poetry_install",
                    file=rel,
                    line=idx,
                    message="Mentions pip/conda install; prefer Poetry commands in this repo",
                    context=line.strip(),
                )
            )

    # huggingface-cli usage (recommend mc model fetch for consistency)
    for idx, line in enumerate(lines, start=1):
        if HF_CLI_RE.search(line):
            issues.append(
                Issue(
                    kind="hf_cli_usage",
                    file=rel,
                    line=idx,
                    message="Uses huggingface-cli; consider mc model fetch for repo-native workflow",
                    context=line.strip(),
                )
            )

    # Check for stale claims about downloaded arXiv PDF count
    m = re.search(r"\b(\d+)\s+downloaded\s+arXiv\s+PDFs\b", text, flags=re.IGNORECASE)
    if m:
        claimed = int(m.group(1))
        if claimed != stale_arxiv_pdf_count:
            line_no = _find_line_number(lines, m.group(0))
            issues.append(
                Issue(
                    kind="stale_count",
                    file=rel,
                    line=line_no,
                    message=f"Stale arXiv PDF count claim: {claimed} (actual: {stale_arxiv_pdf_count})",
                    context=lines[line_no - 1].strip() if 1 <= line_no <= len(lines) else None,
                )
            )

    # Links: file existence + anchors (best-effort)
    for raw_target in _iter_markdown_links(text):
        stripped = _strip_link_target(raw_target)
        if _is_external_link(stripped) or stripped.startswith(("mailto:", "tel:")):
            continue

        base, frag = _split_link_target(stripped)
        resolved = _resolve_internal_path(source_file=path, target=stripped, repo_root=repo_root)
        if resolved is None:
            continue

        # Base target existence
        if base and not resolved.exists():
            line_no = _find_line_number(lines, raw_target)
            issues.append(
                Issue(
                    kind="broken_internal_link",
                    file=rel,
                    line=line_no,
                    message=f"Broken internal link target: {stripped}",
                    context=lines[line_no - 1].strip() if 1 <= line_no <= len(lines) else None,
                )
            )
            continue

        # PDF link sanity: ensure file exists in repo
        if base.lower().endswith(".pdf"):
            pdf_rel = None
            try:
                pdf_rel = str(resolved.relative_to(repo_root))
            except Exception:
                pass
            pdf_name = resolved.name
            if pdf_name not in pdf_files:
                line_no = _find_line_number(lines, raw_target)
                issues.append(
                    Issue(
                        kind="missing_pdf_file",
                        file=rel,
                        line=line_no,
                        message=f"PDF link points to missing file: {pdf_rel or base}",
                        context=lines[line_no - 1].strip() if 1 <= line_no <= len(lines) else None,
                    )
                )

        # Anchor existence (best-effort)
        if frag:
            target_file = resolved if resolved.is_file() else None
            if target_file is None:
                continue
            if target_file not in anchor_cache:
                anchor_cache[target_file] = _extract_heading_anchors(_read_text(target_file))
            anchors = anchor_cache[target_file]
            if frag not in anchors:
                line_no = _find_line_number(lines, raw_target)
                issues.append(
                    Issue(
                        kind="broken_anchor",
                        file=rel,
                        line=line_no,
                        message=f"Anchor not found in target: #{frag} -> {stripped}",
                        context=lines[line_no - 1].strip() if 1 <= line_no <= len(lines) else None,
                    )
                )

    # arXiv ids referenced but not in download script (proxy for “not downloaded”)
    seen_missing_arxiv: set[str] = set()
    for match in re.finditer(
        r"(?:arXiv\s*:\s*|https?://arxiv\.org/(?:abs|pdf)/)([0-9]{4}\.[0-9]{4,5}|[a-z\-]+/[0-9]{7})",
        text,
        flags=re.IGNORECASE,
    ):
        arxiv_id = match.group(1)
        if arxiv_id in seen_missing_arxiv:
            continue
        if arxiv_id not in arxiv_ids_in_script:
            needle = match.group(0)
            line_no = _find_line_number(lines, needle)
            issues.append(
                Issue(
                    kind="arxiv_missing_source",
                    file=rel,
                    line=line_no,
                    message=f"arXiv id referenced but not in docs/references/download_arxiv.sh: {arxiv_id}",
                    context=lines[line_no - 1].strip() if 1 <= line_no <= len(lines) else None,
                )
            )
            seen_missing_arxiv.add(arxiv_id)

    return issues


class _CliHelpIndex:
    def __init__(self, *, repo_root: Path) -> None:
        self._repo_root = repo_root
        self._commands_cache: dict[tuple[str, ...], set[str]] = {}
        self._errors: list[str] = []

    @property
    def errors(self) -> list[str]:
        return list(self._errors)

    def commands_for(self, prefix: tuple[str, ...]) -> set[str]:
        if prefix in self._commands_cache:
            return self._commands_cache[prefix]

        cmd = ["poetry", "run", "mc", *prefix, "--help"]
        proc = _run(cmd, cwd=self._repo_root)
        if proc.returncode != 0:
            self._errors.append(f"{' '.join(cmd)} failed: {proc.stderr.strip() or proc.stdout.strip()}")
            self._commands_cache[prefix] = set()
            return set()

        commands = _parse_typer_commands(proc.stdout)
        self._commands_cache[prefix] = commands
        return commands


def _parse_typer_commands(help_text: str) -> set[str]:
    commands: set[str] = set()
    in_commands = False
    for line in help_text.splitlines():
        if "╭─ Commands" in line:
            in_commands = True
            continue
        if in_commands and line.startswith("╰"):
            break
        if not in_commands:
            continue
        m = re.match(r"^\s*│\s+([a-zA-Z0-9\-]+)\s{2,}", line)
        if m:
            commands.add(m.group(1))
    return commands


def _extract_shell_commands(markdown_text: str) -> list[tuple[int, str]]:
    """
    Extract candidate shell lines from fenced code blocks.
    Returns (line_number, line_text).
    """
    results: list[tuple[int, str]] = []
    lines = markdown_text.splitlines()
    in_fence = False
    fence_lang: str | None = None
    fence_delim: str | None = None

    for idx, line in enumerate(lines, start=1):
        m = re.match(r"^(```+)\s*([A-Za-z0-9_-]+)?\s*$", line)
        if m:
            if not in_fence:
                in_fence = True
                fence_delim = m.group(1)
                fence_lang = (m.group(2) or "").lower() or None
            else:
                if fence_delim and line.startswith(fence_delim):
                    in_fence = False
                    fence_lang = None
                    fence_delim = None
            continue

        if not in_fence:
            continue

        if fence_lang and fence_lang not in {"bash", "sh", "shell", "zsh"}:
            continue

        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "mc " not in stripped and not stripped.startswith("mc"):
            continue
        results.append((idx, stripped))
    return results


def _parse_mc_invocation(command_line: str) -> list[str] | None:
    """
    Return tokens after the `mc` executable, or None if this line doesn't invoke mc.
    Best-effort, handles `poetry run mc`, env var prefixes, and leading `$`.
    """
    line = command_line.strip()
    if line.startswith("$"):
        line = line[1:].strip()

    try:
        tokens = shlex.split(line)
    except ValueError:
        return None

    # Drop leading env assignments (FOO=bar) until we hit an executable.
    i = 0
    while i < len(tokens) and "=" in tokens[i] and not tokens[i].startswith("-"):
        if tokens[i].count("=") == 1 and not tokens[i].startswith(("http://", "https://")):
            i += 1
            continue
        break

    tokens = tokens[i:]
    if not tokens:
        return None

    # Handle `poetry run mc ...`
    if len(tokens) >= 3 and tokens[0] == "poetry" and tokens[1] == "run" and tokens[2] == "mc":
        return tokens[3:]

    # Direct `mc ...`
    if tokens[0] == "mc":
        return tokens[1:]

    return None


def _looks_like_path(token: str) -> bool:
    return "/" in token or token.startswith(".") or token.endswith((".md", ".pdf", ".json", ".yaml", ".yml"))


def _audit_cli_commands_in_markdown(
    *,
    path: Path,
    repo_root: Path,
    cli_index: _CliHelpIndex,
) -> list[Issue]:
    rel = str(path.relative_to(repo_root))
    text = _read_text(path)
    issues: list[Issue] = []

    for line_no, line_text in _extract_shell_commands(text):
        argv = _parse_mc_invocation(line_text)
        if argv is None:
            continue

        prefix: list[str] = []
        commands = cli_index.commands_for(tuple(prefix))
        for token in argv:
            if token.startswith("-"):
                break
            if token in commands:
                prefix.append(token)
                commands = cli_index.commands_for(tuple(prefix))
                continue

            # If this level has known subcommands and this token looks like a subcommand,
            # treat it as likely wrong. If it looks like a path/arg, assume positional arg.
            if commands and COMMAND_TOKEN_RE.match(token) and not _looks_like_path(token):
                issues.append(
                    Issue(
                        kind="unknown_cli_subcommand",
                        file=rel,
                        line=line_no,
                        message=f"Unknown subcommand at {' '.join(['mc', *prefix])}: {token}",
                        context=line_text,
                    )
                )
            break

    return issues


def _parse_download_arxiv_script(*, repo_root: Path) -> tuple[set[str], dict[str, str]]:
    script_path = repo_root / "docs" / "references" / "download_arxiv.sh"
    if not script_path.exists():
        return set(), {}
    text = _read_text(script_path)
    entries = re.findall(r'^\s*"([^"|]+)\|([^"]+)"', text, flags=re.MULTILINE)
    ids = {arxiv_id for arxiv_id, _ in entries}
    mapping = {arxiv_id: f"{filename}.pdf" for arxiv_id, filename in entries}
    return ids, mapping


def _format_markdown_report(result: AuditResult) -> str:
    lines: list[str] = []
    lines.append("# Docs Audit: Fix List (Auto-Generated)")
    lines.append("")
    cmd = "poetry run python scripts/audit_docs.py --format markdown --write-fix-list docs/DOCS-FIX-LIST.md"
    if result.cli.get("enabled"):
        cmd = f"{cmd} --check-cli"
    lines.append(f"This file is generated by `{cmd}`.")
    lines.append("")
    lines.append("Notes:")
    lines.append("- `broken_anchor` is a best-effort check and may include false positives depending on the Markdown renderer/slug rules.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Files scanned: {result.summary.files_scanned}")
    lines.append(f"- Issues found: {result.summary.issues_found}")
    lines.append("")
    lines.append("### Issues by kind")
    lines.append("")
    for kind, count in sorted(result.summary.issues_by_kind.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"- `{kind}`: {count}")

    lines.append("")
    lines.append("## ArXiv coverage")
    lines.append("")
    arxiv = result.arxiv
    lines.append(f"- Referenced arXiv IDs in markdown: {arxiv.get('ids_referenced', 0)}")
    lines.append(f"- IDs in `docs/references/download_arxiv.sh`: {arxiv.get('ids_in_download_script', 0)}")
    lines.append(f"- PDFs in `docs/references/arxiv/`: {arxiv.get('pdfs_present', 0)}")
    missing = arxiv.get("ids_referenced_missing_from_download_script", [])
    if missing:
        lines.append("")
        lines.append("### Referenced but not in download script")
        lines.append("")
        for arxiv_id in missing:
            lines.append(f"- `{arxiv_id}`")

    lines.append("")
    lines.append("## CLI audit")
    lines.append("")
    if result.cli.get("enabled"):
        errors = result.cli.get("errors", [])
        if errors:
            lines.append("- CLI indexing errors (audit continued, but CLI checks may be incomplete):")
            for e in errors:
                lines.append(f"  - {e}")
        else:
            lines.append("- CLI help indexing succeeded.")
    else:
        lines.append("- CLI checks disabled for this run.")

    lines.append("")
    lines.append("## Per-file issues")
    lines.append("")

    issues_by_file: dict[str, list[Issue]] = defaultdict(list)
    for issue in result.issues:
        issues_by_file[issue.file].append(issue)

    for file in sorted(issues_by_file.keys()):
        lines.append(f"### `{file}`")
        lines.append("")
        for issue in sorted(issues_by_file[file], key=lambda i: (i.kind, i.line, i.message)):
            loc = f"{issue.file}:{issue.line}"
            lines.append(f"- `{issue.kind}` at `{loc}`: {issue.message}")
            if issue.context:
                lines.append(f"  - `{issue.context}`")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _format_text_summary(result: AuditResult) -> str:
    lines: list[str] = []
    lines.append(f"Files scanned: {result.summary.files_scanned}")
    lines.append(f"Issues found: {result.summary.issues_found}")
    for kind, count in sorted(result.summary.issues_by_kind.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"- {kind}: {count}")
    return "\n".join(lines).rstrip() + "\n"


def _write_tracker_file(*, repo_root: Path, tracker_path: Path, files: list[Path]) -> None:
    rows: list[tuple[str, str, str]] = []
    for path in files:
        rel = str(path.relative_to(repo_root))
        audience = _guess_audience(rel)
        tier = _guess_tier(rel)
        rows.append((rel, audience, tier))

    tracker_lines: list[str] = []
    tracker_lines.append("# Docs Review Tracker")
    tracker_lines.append("")
    tracker_lines.append("Single source of truth for doc review status across all tracked `*.md` files.")
    tracker_lines.append("")
    tracker_lines.append("Legend:")
    tracker_lines.append("- `Review`: `todo` | `wip` | `done` | `archive`")
    tracker_lines.append("- `Links`: `todo` | `fix` | `ok`")
    tracker_lines.append("- `Citations`: `todo` | `fix` | `ok` | `n/a`")
    tracker_lines.append("- `CLI`: `todo` | `fix` | `ok` | `n/a`")
    tracker_lines.append("")
    try:
        tracker_rel = tracker_path.relative_to(repo_root).as_posix()
    except Exception:
        tracker_rel = tracker_path.name
    tracker_lines.append(f"Generated by `poetry run python scripts/audit_docs.py --init-tracker {tracker_rel}`.")
    tracker_lines.append("")
    tracker_lines.append("| File | Audience | Tier | Review | Links | Citations | CLI | Notes |")
    tracker_lines.append("|------|----------|------|--------|-------|-----------|-----|-------|")

    for rel, audience, tier in rows:
        tracker_lines.append(f"| `{rel}` | {audience} | {tier} | todo | todo | todo | todo | |")

    tracker_lines.append("")

    tracker_path.parent.mkdir(parents=True, exist_ok=True)
    tracker_path.write_text("\n".join(tracker_lines), encoding="utf-8")


def _guess_audience(rel_path: str) -> str:
    name = Path(rel_path).name.lower()
    if rel_path.startswith("docs/research/") or rel_path.startswith("papers/"):
        return "research"
    if rel_path.startswith("docs/geometry/"):
        return "research"
    if rel_path.startswith("docs/references/"):
        return "internal"
    if "audit" in name or "repo-audit" in name or "deepdive" in name:
        return "internal"
    if name in {"code_of_conduct.md", "security.md", "disclaimer.md"}:
        return "policy"
    if name in {"contributing.md", "agents.md", "claude.md"}:
        return "developer"
    if rel_path.startswith("docs/"):
        return "public"
    return "public"


def _guess_tier(rel_path: str) -> str:
    name = Path(rel_path).name.lower()
    if name in {"readme.md", "start-here.md", "getting_started.md", "cli-reference.md"}:
        return "P0"
    if name in {"mcp.md", "glossary.md", "geometry-guide.md", "architecture.md"}:
        return "P1"
    if rel_path.startswith(("docs/research/", "papers/")):
        return "P2"
    if "audit" in name or "repo-audit" in name or "deepdive" in name:
        return "A"
    return "P1"


def _build_audit_result(
    *,
    repo_root: Path,
    files: list[Path],
    check_cli: bool,
) -> AuditResult:
    anchor_cache: dict[Path, set[str]] = {}

    pdf_root = repo_root / "docs" / "references" / "arxiv"
    pdf_files = {p.name for p in pdf_root.glob("*.pdf")} if pdf_root.exists() else set()

    arxiv_ids_in_script, arxiv_id_to_pdf = _parse_download_arxiv_script(repo_root=repo_root)

    issues: list[Issue] = []

    cli_index = _CliHelpIndex(repo_root=repo_root)
    cli_enabled = check_cli

    for path in files:
        issues.extend(
            _audit_markdown_file(
                path=path,
                repo_root=repo_root,
                anchor_cache=anchor_cache,
                pdf_files=pdf_files,
                pdf_root=pdf_root,
                arxiv_ids_in_script=arxiv_ids_in_script,
                stale_arxiv_pdf_count=len(pdf_files),
            )
        )
        if check_cli:
            issues.extend(_audit_cli_commands_in_markdown(path=path, repo_root=repo_root, cli_index=cli_index))

    issues_by_kind: Counter[str] = Counter(i.kind for i in issues)
    summary = AuditSummary(
        files_scanned=len(files),
        issues_found=len(issues),
        issues_by_kind=dict(sorted(issues_by_kind.items())),
    )

    # arXiv coverage stats
    referenced_ids: set[str] = set()
    arxiv_ref_re = re.compile(
        r"(?:arXiv\s*:\s*|https?://arxiv\.org/(?:abs|pdf)/)([0-9]{4}\.[0-9]{4,5}|[a-z\-]+/[0-9]{7})",
        flags=re.IGNORECASE,
    )
    for path in files:
        referenced_ids.update(arxiv_ref_re.findall(_read_text(path)))

    arxiv = {
        "ids_referenced": len(referenced_ids),
        "ids_in_download_script": len(arxiv_ids_in_script),
        "pdfs_present": len(pdf_files),
        "ids_referenced_missing_from_download_script": sorted(referenced_ids - arxiv_ids_in_script),
        "download_script_mapping_count": len(arxiv_id_to_pdf),
    }

    cli = {
        "enabled": cli_enabled,
        "errors": cli_index.errors if cli_enabled else [],
    }

    return AuditResult(summary=summary, issues=issues, arxiv=arxiv, cli=cli)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit repo Markdown docs for common issues.")
    parser.add_argument(
        "--include-untracked",
        action="store_true",
        help="Also scan untracked/ignored .md files on disk (default: only git-tracked).",
    )
    parser.add_argument(
        "--check-cli",
        action="store_true",
        help="Validate `mc ...` command paths found in docs by parsing `poetry run mc ... --help` output.",
    )
    parser.add_argument(
        "--format",
        choices=("json", "markdown", "text"),
        default="text",
        help="Output format for stdout (default: text).",
    )
    parser.add_argument(
        "--write-fix-list",
        type=str,
        default=None,
        help="Write a markdown fix list to this path.",
    )
    parser.add_argument(
        "--init-tracker",
        type=str,
        default=None,
        help="Create/overwrite a docs review tracker markdown file at this path.",
    )
    args = parser.parse_args(argv)

    if args.include_untracked:
        files = _all_markdown_files_on_disk(repo_root=REPO_ROOT)
    else:
        files = _git_tracked_markdown_files(repo_root=REPO_ROOT)

    if args.init_tracker:
        tracker_path = (REPO_ROOT / args.init_tracker).resolve()
        _write_tracker_file(repo_root=REPO_ROOT, tracker_path=tracker_path, files=_git_tracked_markdown_files(repo_root=REPO_ROOT))

    result = _build_audit_result(repo_root=REPO_ROOT, files=files, check_cli=args.check_cli)

    if args.write_fix_list:
        out_path = (REPO_ROOT / args.write_fix_list).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(_format_markdown_report(result), encoding="utf-8")

    if args.format == "json":
        payload = dataclasses.asdict(result)
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif args.format == "markdown":
        print(_format_markdown_report(result))
    else:
        print(_format_text_summary(result))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
