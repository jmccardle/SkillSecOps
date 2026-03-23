"""Tests for the Claude CLI LLM wrapper."""

from __future__ import annotations

import json

from skillsecops.llm import (
    _build_prompt_parts,
    _format_tools,
    _parse_tool_calls,
    MODEL_MAP,
)


class TestBuildPromptParts:
    def test_system_separated_from_user(self):
        messages = [
            {"role": "system", "content": "You are a helper."},
            {"role": "user", "content": "Hello."},
        ]
        system, user = _build_prompt_parts(messages)
        assert system == "You are a helper."
        assert "Hello." in user
        # System should NOT appear in user prompt
        assert "You are a helper." not in user

    def test_tool_messages(self):
        messages = [
            {"role": "user", "content": "Do something."},
            {"role": "tool", "tool_call_id": "call_1", "content": "Tool result here."},
        ]
        system, user = _build_prompt_parts(messages)
        assert system is None
        assert "Tool result for call_1" in user
        assert "Tool result here." in user

    def test_tools_appended(self):
        messages = [{"role": "user", "content": "Hi."}]
        tools = [{
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                },
            },
        }]
        system, user = _build_prompt_parts(messages, tools)
        assert "read_file" in user
        assert "Available tools" in user
        assert "JSON array" in user

    def test_no_system_message(self):
        messages = [{"role": "user", "content": "Just a question."}]
        system, user = _build_prompt_parts(messages)
        assert system is None
        assert "Just a question." in user


class TestParseToolCalls:
    def test_valid_tool_calls(self):
        text = json.dumps([
            {"id": "call_1", "name": "read_file", "arguments": {"path": "/etc/hosts"}},
        ])
        result = _parse_tool_calls(text)
        assert result is not None
        assert len(result) == 1
        assert result[0].function.name == "read_file"
        assert "hosts" in result[0].function.arguments

    def test_plain_text_returns_none(self):
        assert _parse_tool_calls("Just a normal response.") is None

    def test_invalid_json_returns_none(self):
        assert _parse_tool_calls("[not valid json") is None

    def test_empty_array_returns_none(self):
        assert _parse_tool_calls("[]") is None

    def test_non_tool_array_returns_none(self):
        assert _parse_tool_calls('[{"foo": "bar"}]') is None


class TestModelMap:
    def test_sonnet_aliases(self):
        assert MODEL_MAP["sonnet"] == "sonnet"
        assert MODEL_MAP["claude-sonnet"] == "sonnet"
        assert MODEL_MAP["gpt-4o-mini"] == "sonnet"

    def test_opus_aliases(self):
        assert MODEL_MAP["opus"] == "opus"
        assert MODEL_MAP["gpt-4o"] == "opus"
