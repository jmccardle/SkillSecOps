"""Layer 3: Cross-reference — compare summaries against declared metadata.

This layer NEVER sees raw skill content. It compares Layer 2's chunk
summaries against the skill's declared metadata (from YAML frontmatter)
to detect mismatches: undeclared capabilities, tools the skill claims
it doesn't use, or descriptions that don't match actual behavior.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Optional

from skillsecops.models import (
    AnalysisVerdict,
    ChunkSummary,
    CrossRefResult,
    SummarizationResult,
)

logger = logging.getLogger(__name__)

# Security-relevant capability keywords to check for in summaries
_SECURITY_CAPABILITIES = {
    "network": {"network", "http", "https", "api", "url", "request", "fetch",
                "download", "upload", "webhook", "socket", "dns"},
    "file_system": {"file", "read_file", "write_file", "directory", "path",
                    "filesystem", "disk", "folder"},
    "credential": {"credential", "password", "secret", "token", "api_key",
                   "api key", "apikey", "auth", "env", ".env", "environment variable"},
    "execution": {"execute", "eval", "exec", "subprocess", "shell", "command",
                  "run", "script", "code execution"},
    "exfiltration": {"exfiltrate", "send data", "leak", "transmit", "report to",
                     "phone home", "collect"},
}

# Common English stopwords for keyword overlap
_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "can", "shall", "it", "its",
    "this", "that", "these", "those", "i", "you", "he", "she", "we",
    "they", "me", "him", "her", "us", "them", "my", "your", "his",
    "our", "their", "what", "which", "who", "whom", "when", "where",
    "why", "how", "not", "no", "nor", "if", "then", "than", "so",
    "as", "just", "about", "into", "through", "during", "before",
    "after", "above", "below", "up", "down", "out", "off", "over",
    "under", "again", "further", "once", "all", "each", "every",
    "both", "few", "more", "most", "other", "some", "such", "only",
    "own", "same", "too", "very", "also",
})


# ---------------------------------------------------------------------------
# YAML frontmatter parsing
# ---------------------------------------------------------------------------

def _parse_yaml_frontmatter(skill_dir: Path) -> dict[str, str]:
    """Extract YAML frontmatter from SKILL.md.

    Uses PyYAML if available, falls back to a minimal line parser.
    Raises ValueError if no frontmatter found.
    """
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.exists():
        raise FileNotFoundError(f"No SKILL.md found in {skill_dir}")

    content = skill_md.read_text(encoding="utf-8", errors="ignore")

    if not content.startswith("---"):
        raise ValueError(f"No YAML frontmatter found in {skill_md}")

    end_marker = content.find("---", 3)
    if end_marker == -1:
        raise ValueError(f"Unterminated YAML frontmatter in {skill_md}")

    yaml_block = content[3:end_marker].strip()

    # Try PyYAML first
    try:
        import yaml
        parsed = yaml.safe_load(yaml_block)
        if isinstance(parsed, dict):
            return parsed
    except ImportError:
        pass
    except Exception:
        pass

    # Minimal fallback: parse key: value lines
    result: dict[str, str] = {}
    for line in yaml_block.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(r"^([\w][\w-]*)\s*:\s*(.+)$", line)
        if match:
            key = match.group(1)
            value = match.group(2).strip().strip('"').strip("'")
            result[key] = value

    if not result:
        raise ValueError(f"Could not parse frontmatter in {skill_md}")

    return result


def _extract_declared_tools(frontmatter: dict) -> set[str]:
    """Extract tool names the skill declares it needs."""
    tools: set[str] = set()

    for key in ("tools", "requires_tools", "tool_invocations"):
        val = frontmatter.get(key)
        if isinstance(val, list):
            tools.update(str(t).lower() for t in val)
        elif isinstance(val, str):
            tools.update(t.strip().lower() for t in val.split(","))

    return tools


# ---------------------------------------------------------------------------
# Keyword overlap
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> set[str]:
    """Tokenize text into lowercase content words, stopwords removed."""
    words = re.findall(r"[a-z][a-z0-9_-]*", text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}


def _compute_keyword_overlap(
    description: str,
    summaries: list[ChunkSummary],
) -> float:
    """Compute Jaccard similarity between description and summary text."""
    desc_tokens = _tokenize(description)
    summary_text = " ".join(c.summary for c in summaries)
    summary_tokens = _tokenize(summary_text)

    if not desc_tokens or not summary_tokens:
        return 0.0

    intersection = desc_tokens & summary_tokens
    union = desc_tokens | summary_tokens

    return len(intersection) / len(union) if union else 0.0


# ---------------------------------------------------------------------------
# Undeclared capability detection
# ---------------------------------------------------------------------------

def _find_undeclared_capabilities(
    summaries: list[ChunkSummary],
    declared_description: str,
    declared_tools: set[str],
) -> list[str]:
    """Find capabilities in summaries that the skill doesn't declare.

    Returns list of mismatch descriptions.
    """
    mismatches: list[str] = []
    desc_lower = declared_description.lower()

    # Check tools referenced in summaries vs declared tools.
    # Only flag SECURITY-SENSITIVE tools — network access, shell execution,
    # credential access tools. Implementation libraries (pandas, openpyxl)
    # are not security-relevant even if undeclared.
    _SECURITY_SENSITIVE_TOOLS = {
        "send_http", "curl", "wget", "requests", "fetch", "urllib",
        "execute_code", "eval", "exec", "subprocess", "os.system",
        "get_env_variable", "os.environ",
        "write_file", "open", "shutil",
    }

    summary_tools: set[str] = set()
    for chunk in summaries:
        summary_tools.update(t.lower() for t in chunk.declared_tools)

    undeclared_sensitive = (summary_tools - declared_tools) & _SECURITY_SENSITIVE_TOOLS
    if undeclared_sensitive:
        mismatches.append(
            f"Security-sensitive tools referenced but not declared: {undeclared_sensitive}"
        )

    # Check security-relevant capabilities in summaries vs description
    all_summary_text = " ".join(
        f"{c.summary} {' '.join(c.declared_capabilities)}"
        for c in summaries
    ).lower()

    for capability_name, keywords in _SECURITY_CAPABILITIES.items():
        # Is this capability mentioned in summaries?
        summary_mentions = any(kw in all_summary_text for kw in keywords)
        # Is it plausibly covered by the description?
        desc_mentions = any(kw in desc_lower for kw in keywords)

        if summary_mentions and not desc_mentions:
            matched_keywords = [kw for kw in keywords if kw in all_summary_text]
            mismatches.append(
                f"Undeclared {capability_name} capability: "
                f"summaries mention {matched_keywords} "
                f"but description does not cover this"
            )

    # Check for instructions_to_agent that go BEYOND the declared scope.
    # A skill about spreadsheets legitimately contains instructions like
    # "use Excel formulas" — these are consistent with its purpose.
    # Only flag instructions that reference security-relevant actions
    # not covered by the description.
    _SUSPICIOUS_INSTRUCTION_KEYWORDS = {
        "ignore previous", "ignore all", "ignore instructions",  # prompt injection (narrowed)
        "override previous", "override instructions", "override system",
        "disregard", "forget everything",
        "api_key", "api key", "secret key", "password", "credential",  # credential access
        "exfiltrate", "phone home", "report to",  # data exfiltration
        ".env file", "environment variable",  # credential file access
    }

    for chunk in summaries:
        for instruction in chunk.instructions_to_agent:
            instruction_lower = instruction.lower()
            suspicious_matches = [
                kw for kw in _SUSPICIOUS_INSTRUCTION_KEYWORDS
                if kw in instruction_lower
            ]
            if suspicious_matches:
                mismatches.append(
                    f"Chunk {chunk.chunk_index} contains suspicious instruction "
                    f"(matches: {suspicious_matches}): {instruction}"
                )

    return mismatches


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def crossref_skill(
    skill_dir: Path,
    summarization_result: SummarizationResult,
    *,
    use_llm: bool = False,
    client: Any = None,
    model: str = "gpt-4o-mini",
) -> CrossRefResult:
    """Layer 3: Cross-reference summaries against declared metadata.

    Reads only YAML frontmatter from SKILL.md (not the full content).
    Compares against Layer 2 chunk summaries.

    Default mode is deterministic. Set use_llm=True for semantic comparison
    (not yet implemented — reserved for future enhancement).

    Args:
        skill_dir: Path to skill directory.
        summarization_result: Output from Layer 2.
        use_llm: Whether to use LLM for semantic comparison.
        client: OpenAI client (only needed if use_llm=True).
        model: Model name for LLM comparison.

    Returns:
        CrossRefResult with verdict, mismatches, and undeclared capabilities.
    """
    try:
        frontmatter = _parse_yaml_frontmatter(skill_dir)
    except (ValueError, FileNotFoundError) as e:
        return CrossRefResult(
            verdict=AnalysisVerdict.ERROR,
            declared_description="",
            mismatches=[f"Frontmatter error: {e}"],
        )

    description = frontmatter.get("description", "")
    if isinstance(description, str):
        # Handle multiline YAML descriptions (joined with spaces)
        description = " ".join(description.split())
    else:
        description = str(description)

    declared_tools = _extract_declared_tools(frontmatter)
    chunks = summarization_result.chunks

    # Keyword overlap check
    overlap = _compute_keyword_overlap(description, chunks)
    mismatches: list[str] = []

    if overlap < 0.05 and chunks:
        mismatches.append(
            f"Very low keyword overlap ({overlap:.3f}) between declared "
            f"description and actual content summaries — skill may do "
            f"something unrelated to its description"
        )

    # Undeclared capability detection
    undeclared = _find_undeclared_capabilities(chunks, description, declared_tools)

    # Determine verdict
    if undeclared or mismatches:
        verdict = AnalysisVerdict.FAIL
    else:
        verdict = AnalysisVerdict.PASS

    return CrossRefResult(
        verdict=verdict,
        declared_description=description,
        mismatches=mismatches,
        undeclared_capabilities=undeclared,
    )
