"""Layer 1: Static analysis — deterministic pattern matching, no LLM.

Scans SKILL.md and script files for known prompt injection patterns,
suspicious encodings, and structural anomalies. This is the bedrock layer:
what it catches, it catches with certainty.

This module has no network dependencies and no LLM calls.
"""

from __future__ import annotations

import base64
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from skillsecops.models import AnalysisVerdict, StaticAnalysisResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pattern definitions
# ---------------------------------------------------------------------------

@dataclass
class Pattern:
    """A single detection pattern."""
    name: str
    description: str
    regex: re.Pattern
    severity: str  # "critical" or "warning"


# Role-switching / instruction override attempts
_ROLE_SWITCH_PATTERNS = [
    Pattern(
        name="system_role_override",
        description="Attempts to inject a system/instruction role boundary",
        regex=re.compile(
            r"(\[SYSTEM\]|\[INST\]|<\|im_start\|>|<\|im_end\|>|### Instruction:|### System:|"
            r"<\|system\|>|<\|user\|>|<\|assistant\|>|<<SYS>>|<</SYS>>)",
            re.IGNORECASE,
        ),
        severity="critical",
    ),
    Pattern(
        name="ignore_previous",
        description="Direct instruction to disregard prior context",
        regex=re.compile(
            r"(ignore\s+(all\s+)?previous\s+instructions|"
            r"disregard\s+(all\s+)?(prior|previous|above)\s+(instructions|context|rules)|"
            r"forget\s+(everything|all)\s+(above|before|prior)|"
            r"override\s+(all\s+)?(previous|prior)\s+(instructions|rules))",
            re.IGNORECASE,
        ),
        severity="critical",
    ),
    Pattern(
        name="role_reassignment",
        description="Attempts to redefine the agent's identity or role",
        regex=re.compile(
            r"(you\s+are\s+now\s+(a|an|the)\b|"
            r"from\s+now\s+on\s+you\s+(are|will|must|should)\b|"
            r"your\s+new\s+(role|purpose|objective|mission)\s+is\b|"
            r"act\s+as\s+(if\s+you\s+are|a|an|the)\b)",
            re.IGNORECASE,
        ),
        severity="critical",
    ),
]

# Tool/action invocation attempts
_TOOL_INVOCATION_PATTERNS = [
    Pattern(
        name="tool_call_syntax",
        description="Raw tool-call or function-call formatting in skill content",
        regex=re.compile(
            r"(<tool_call>|<function_call>|<\|plugin\|>|"
            r'\{"(function|tool|name)":\s*"[^"]+"|'
            r"<invoke|<tool_use>)",
            re.IGNORECASE,
        ),
        severity="critical",
    ),
    Pattern(
        name="shell_execution_directive",
        description="Direct shell command execution instructions outside code blocks",
        regex=re.compile(
            r"(run\s+this\s+command|execute\s+the\s+following|"
            r"curl\s+https?://\S+\s*\|.*sh|"
            r"wget\s+\S+\s*&&\s*chmod\s*\+x|"
            r"eval\s*\(\s*['\"])",
            re.IGNORECASE,
        ),
        severity="warning",
    ),
]

# Encoding / obfuscation
_OBFUSCATION_PATTERNS = [
    Pattern(
        name="base64_blob",
        description="Large base64-encoded block that could hide instructions",
        regex=re.compile(
            # 128+ chars of contiguous base64 not inside a code fence
            r"(?<!\`)[A-Za-z0-9+/]{128,}={0,2}(?!\`)"
        ),
        severity="warning",
    ),
    Pattern(
        name="unicode_direction_override",
        description="Unicode bidirectional override characters (text direction attacks)",
        regex=re.compile(
            r"[\u200e\u200f\u202a-\u202e\u2066-\u2069]"
        ),
        severity="critical",
    ),
    Pattern(
        name="homoglyph_substitution",
        description="Cyrillic or confusable characters mixed with Latin text",
        regex=re.compile(
            # Common confusables: Cyrillic а(0430) е(0435) о(043e) р(0440) с(0441) etc
            r"[\u0400-\u04ff]"
        ),
        severity="warning",
    ),
    Pattern(
        name="zero_width_characters",
        description="Zero-width joiners/non-joiners that could hide content",
        regex=re.compile(
            r"[\u200b\u200c\u200d\ufeff]"
        ),
        severity="warning",
    ),
]

# Exfiltration / credential access
_EXFILTRATION_PATTERNS = [
    Pattern(
        name="credential_reference",
        description="References to API keys, tokens, or secrets in instruction context",
        regex=re.compile(
            r"(API_KEY|OPENAI_KEY|ANTHROPIC_KEY|SECRET_KEY|GITHUB_TOKEN|"
            r"access.token|private.key|credentials?\s*[\[\(]|"
            r"env\s*\[\s*['\"].*(?:KEY|TOKEN|SECRET|PASSWORD))",
            re.IGNORECASE,
        ),
        severity="warning",
    ),
    Pattern(
        name="url_exfiltration",
        description="URLs with query parameters that could exfiltrate data",
        regex=re.compile(
            r"https?://[^\s]+\?[^\s]*\{[^\s]*\}",
            re.IGNORECASE,
        ),
        severity="critical",
    ),
]

ALL_PATTERNS = (
    _ROLE_SWITCH_PATTERNS
    + _TOOL_INVOCATION_PATTERNS
    + _OBFUSCATION_PATTERNS
    + _EXFILTRATION_PATTERNS
)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def _is_inside_code_fence(content: str, match_start: int) -> bool:
    """Check if a match position falls inside a markdown fenced code block.

    Content inside code fences is expected to contain tool syntax, shell
    commands, etc. — these are illustrative, not injected instructions.
    """
    lines = content[:match_start].split("\n")
    fence_depth = 0
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            fence_depth = 1 - fence_depth
    return fence_depth > 0


def analyze_file(file_path: Path, content: str) -> list[tuple[Pattern, int]]:
    """Scan one file's content against all patterns.

    Returns list of (pattern, line_number) for matches outside code fences.
    """
    hits: list[tuple[Pattern, int]] = []

    for pattern in ALL_PATTERNS:
        for match in pattern.regex.finditer(content):
            if _is_inside_code_fence(content, match.start()):
                continue
            line_num = content[:match.start()].count("\n") + 1
            hits.append((pattern, line_num))

    return hits


def analyze_skill(skill_dir: Path) -> StaticAnalysisResult:
    """Run Layer 1 static analysis on an entire skill directory.

    Scans all text files for known injection patterns.
    Returns a deterministic, reproducible result.
    """
    all_hits: list[str] = []

    text_extensions = {".md", ".txt", ".py", ".sh", ".yaml", ".yml", ".json", ".toml"}

    for file_path in sorted(skill_dir.rglob("*")):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in text_extensions:
            continue

        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except OSError as e:
            logger.warning("Cannot read %s: %s", file_path, e)
            continue

        hits = analyze_file(file_path, content)
        rel_path = file_path.relative_to(skill_dir)

        for pattern, line_num in hits:
            description = f"[{pattern.severity}] {pattern.name}: {rel_path}:{line_num} — {pattern.description}"
            all_hits.append(description)
            logger.info("Static hit: %s", description)

    critical_hits = [h for h in all_hits if h.startswith("[critical]")]

    verdict = AnalysisVerdict.FAIL if critical_hits else AnalysisVerdict.PASS

    return StaticAnalysisResult(
        verdict=verdict,
        patterns_checked=len(ALL_PATTERNS),
        patterns_matched=all_hits,
        details=f"{len(all_hits)} pattern matches ({len(critical_hits)} critical)"
        if all_hits
        else "No known injection patterns detected",
    )
