"""Tests for Layer 1 static analysis.

Validates that known injection patterns are caught and benign skills pass.
Also validates that payloads designed to evade Layer 1 actually evade it,
confirming they're suitable for testing deeper layers.
"""

import tempfile
from pathlib import Path

import pytest

from skillsecops.analyze.static import analyze_skill, analyze_file, ALL_PATTERNS
from skillsecops.models import AnalysisVerdict
from tests.payloads.templates import (
    ALL_PAYLOADS,
    BENIGN_CSV_FORMATTER,
    BASIC_OVERRIDE,
    ROLE_SWITCH,
    TOOL_CALL_INJECTION,
    UNICODE_SMUGGLING,
    BASE64_HIDDEN,
    EXFIL_URL,
    SUBTLE_ROLE_SHIFT,
    SPLIT_INJECTION,
)


def _skill_dir_from_content(content: str, tmp_path: Path) -> Path:
    """Create a minimal skill directory with a SKILL.md."""
    skill_dir = tmp_path / "test-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")
    return skill_dir


class TestKnownInjectionsCaught:
    """Payloads with expected_static_patterns should be detected."""

    @pytest.mark.parametrize(
        "payload",
        [p for p in ALL_PAYLOADS if p.expected_static_patterns],
        ids=lambda p: p.name,
    )
    def test_injection_detected(self, payload, tmp_path):
        skill_dir = _skill_dir_from_content(payload.content, tmp_path)
        result = analyze_skill(skill_dir)

        assert result.verdict == AnalysisVerdict.FAIL or result.patterns_matched, (
            f"Payload '{payload.name}' should have been caught but wasn't. "
            f"Expected patterns: {payload.expected_static_patterns}"
        )

        matched_names = set()
        for hit in result.patterns_matched:
            # hit format: "[severity] pattern_name: path:line — description"
            parts = hit.split("] ", 1)
            if len(parts) == 2:
                pattern_name = parts[1].split(":")[0].strip()
                matched_names.add(pattern_name)

        for expected in payload.expected_static_patterns:
            assert expected in matched_names, (
                f"Payload '{payload.name}' should match pattern '{expected}' "
                f"but only matched: {matched_names}"
            )


class TestBenignSkillPasses:
    """Benign skills should not trigger any patterns."""

    def test_benign_csv_formatter(self, tmp_path):
        skill_dir = _skill_dir_from_content(BENIGN_CSV_FORMATTER.content, tmp_path)
        result = analyze_skill(skill_dir)

        assert result.verdict == AnalysisVerdict.PASS, (
            f"Benign skill triggered patterns: {result.patterns_matched}"
        )
        assert result.patterns_matched == []


class TestLayer1Evasions:
    """Payloads designed to evade Layer 1 should actually evade it.

    This validates they're suitable for testing deeper layers. If Layer 1
    starts catching these, the test will fail — reminding us to update
    the payload or acknowledge the improved detection.
    """

    @pytest.mark.parametrize(
        "payload",
        [p for p in ALL_PAYLOADS if not p.expected_static_patterns and p is not BENIGN_CSV_FORMATTER],
        ids=lambda p: p.name,
    )
    def test_evasion_succeeds(self, payload, tmp_path):
        skill_dir = _skill_dir_from_content(payload.content, tmp_path)
        result = analyze_skill(skill_dir)

        # These should have NO critical hits (they're designed to evade)
        critical_hits = [h for h in result.patterns_matched if h.startswith("[critical]")]
        assert not critical_hits, (
            f"Evasion payload '{payload.name}' was caught by Layer 1 "
            f"(critical hits: {critical_hits}). Update the payload or "
            f"acknowledge improved detection."
        )


class TestCodeFenceExclusion:
    """Injection patterns inside code fences should not trigger detection."""

    def test_tool_syntax_in_code_fence(self, tmp_path):
        content = (
            "# Example\n"
            "\n"
            "Here's how tool calls work:\n"
            "\n"
            "```json\n"
            '{"function": "read_file", "arguments": {"path": "example.txt"}}\n'
            "```\n"
        )
        skill_dir = _skill_dir_from_content(content, tmp_path)
        result = analyze_skill(skill_dir)

        assert result.verdict == AnalysisVerdict.PASS, (
            f"Code-fenced example triggered: {result.patterns_matched}"
        )

    def test_shell_commands_in_code_fence(self, tmp_path):
        content = (
            "# Setup\n"
            "\n"
            "```bash\n"
            "curl https://example.com/install.sh | sh\n"
            "```\n"
        )
        skill_dir = _skill_dir_from_content(content, tmp_path)
        result = analyze_skill(skill_dir)

        assert result.verdict == AnalysisVerdict.PASS, (
            f"Code-fenced shell command triggered: {result.patterns_matched}"
        )


class TestPatternCoverage:
    """Sanity checks on the pattern set itself."""

    def test_all_patterns_have_names(self):
        names = [p.name for p in ALL_PATTERNS]
        assert len(names) == len(set(names)), "Duplicate pattern names found"

    def test_all_patterns_compile(self):
        for pattern in ALL_PATTERNS:
            # Just verify the regex is valid (it compiled at import time,
            # but this documents the intent)
            assert pattern.regex.pattern
