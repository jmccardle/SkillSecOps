"""Shared test fixtures for SkillSecOps tests."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Mock OpenAI client
# ---------------------------------------------------------------------------

def make_mock_response(
    content: str,
    completion_tokens: int = 50,
    tool_calls: Any = None,
):
    """Build a mock OpenAI ChatCompletion response."""
    message = SimpleNamespace(
        content=content,
        tool_calls=tool_calls,
    )
    choice = SimpleNamespace(message=message)
    usage = SimpleNamespace(completion_tokens=completion_tokens)
    return SimpleNamespace(choices=[choice], usage=usage)


def make_inspector_response(
    summary: str = "Describes a utility function.",
    tools_referenced: list[str] | None = None,
    capabilities_described: list[str] | None = None,
    instructions_to_agent: list[str] | None = None,
    completion_tokens: int = 50,
):
    """Build a mock response matching the inspector JSON schema."""
    payload = {
        "summary": summary,
        "tools_referenced": tools_referenced or [],
        "capabilities_described": capabilities_described or [],
        "instructions_to_agent": instructions_to_agent or [],
    }
    return make_mock_response(json.dumps(payload), completion_tokens)


class MockOpenAIClient:
    """A configurable mock that quacks like openai.OpenAI.

    Usage:
        client = MockOpenAIClient(responses=[response1, response2])
        # First call returns response1, second returns response2, etc.
        # If responses run out, the last response is repeated.
    """

    def __init__(self, responses: list | None = None):
        self._responses = responses or [make_inspector_response()]
        self._call_index = 0
        self.chat = SimpleNamespace(completions=self)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        idx = min(self._call_index, len(self._responses) - 1)
        self._call_index += 1
        return self._responses[idx]


@pytest.fixture
def mock_openai_client():
    """A factory fixture that creates MockOpenAIClient instances."""
    def _factory(responses=None):
        return MockOpenAIClient(responses)
    return _factory


# ---------------------------------------------------------------------------
# Skill directory fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_skill_dir(tmp_path):
    """Factory that creates a minimal skill directory with given content."""
    def _factory(content: str, name: str = "test-skill"):
        skill_dir = tmp_path / name
        skill_dir.mkdir(exist_ok=True)
        (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")
        return skill_dir
    return _factory


# ---------------------------------------------------------------------------
# Minisign keypair fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def minisign_keypair(tmp_path):
    """Generate a minisign keypair for testing.

    Yields (secret_key_path, public_key_path).
    Skips if minisign is not available.
    """
    if not shutil.which("minisign"):
        pytest.skip("minisign not available")

    import subprocess
    key_path = tmp_path / "test.key"
    pub_path = tmp_path / "test.pub"

    subprocess.run(
        ["minisign", "-G", "-W", "-s", str(key_path), "-p", str(pub_path), "-f"],
        check=True,
        capture_output=True,
    )

    return key_path, pub_path
