"""LLM backend abstraction — wraps `claude -p` for Anthropic subscription accounts.

Instead of the OpenAI SDK, this module shells out to the `claude` CLI with `-p`
(print mode) and `--output-format json`. This lets SkillSecOps use Anthropic
subscription accounts directly without API keys.

The wrapper implements just enough of the OpenAI client interface that
summarize.py and sandbox.py can use it as a drop-in replacement.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Any, Literal

logger = logging.getLogger(__name__)


def require_claude_cli() -> None:
    """Verify the claude CLI is on PATH."""
    if not shutil.which("claude"):
        raise RuntimeError(
            "claude CLI is not installed or not on PATH. "
            "Install from https://claude.ai/code"
        )


# ---------------------------------------------------------------------------
# Response types that match the OpenAI interface shape
# ---------------------------------------------------------------------------

@dataclass
class FunctionCall:
    name: str
    arguments: str


@dataclass
class ToolCall:
    id: str
    function: FunctionCall


@dataclass
class Usage:
    completion_tokens: int
    prompt_tokens: int = 0
    total_tokens: int = 0


@dataclass
class Message:
    content: str | None
    tool_calls: list[ToolCall] | None = None


@dataclass
class Choice:
    message: Message


@dataclass
class ClaudeResponse:
    choices: list[Choice]
    usage: Usage | None = None
    cost_usd: float = 0.0
    duration_ms: int = 0


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

# Model aliases for convenience
MODEL_MAP = {
    "sonnet": "sonnet",
    "opus": "opus",
    "haiku": "haiku",
    "claude-sonnet": "sonnet",
    "claude-opus": "opus",
    "claude-haiku": "haiku",
    # OpenAI-style names fall back to sonnet
    "gpt-4o-mini": "sonnet",
    "gpt-4o": "opus",
}


class ClaudeClient:
    """LLM client that wraps `claude -p` for Anthropic subscription accounts.

    Implements the subset of the OpenAI client interface used by
    summarize.py and sandbox.py:

        client.chat.completions.create(
            model=..., messages=..., max_tokens=..., tools=...
        )

    Usage:
        client = ClaudeClient(default_model="sonnet")
        # Use exactly like MockOpenAIClient or openai.OpenAI()
    """

    def __init__(self, default_model: str = "sonnet"):
        require_claude_cli()
        self.default_model = default_model
        self.chat = _ChatNamespace(self)
        self.total_cost_usd: float = 0.0
        self.call_count: int = 0

    def _resolve_model(self, model: str) -> str:
        return MODEL_MAP.get(model, model)


class _ChatNamespace:
    def __init__(self, client: ClaudeClient):
        self.completions = _CompletionsNamespace(client)


class _CompletionsNamespace:
    def __init__(self, client: ClaudeClient):
        self._client = client

    def create(
        self,
        *,
        model: str | None = None,
        messages: list[dict] | None = None,
        max_tokens: int = 1024,
        tools: list[dict] | None = None,
        **kwargs,
    ) -> ClaudeResponse:
        """Call claude -p with the given messages.

        Flattens the messages array into a single prompt string.
        If tools are provided, they are described in the prompt
        (claude -p does not support function calling natively, so
        tool calls are detected by parsing JSON from the response).
        """
        resolved_model = self._client._resolve_model(
            model or self._client.default_model
        )

        system_prompt, user_prompt = _build_prompt_parts(messages or [], tools)

        result_text, cost, duration, output_tokens = _call_claude(
            user_prompt, resolved_model, max_tokens, system_prompt=system_prompt
        )

        self._client.total_cost_usd += cost
        self._client.call_count += 1

        # Check if the response contains tool calls (for sandbox layer)
        tool_calls = None
        if tools:
            tool_calls = _parse_tool_calls(result_text)

        message = Message(
            content=result_text if not tool_calls else None,
            tool_calls=tool_calls,
        )

        return ClaudeResponse(
            choices=[Choice(message=message)],
            usage=Usage(completion_tokens=output_tokens),
            cost_usd=cost,
            duration_ms=duration,
        )


def _build_prompt_parts(
    messages: list[dict],
    tools: list[dict] | None = None,
) -> tuple[str | None, str]:
    """Split OpenAI-style messages into (system_prompt, user_prompt).

    The system message is extracted separately so it can be passed via
    claude's --system-prompt flag (not embedded in the user prompt where
    it looks like a role-switch injection).

    Returns (system_prompt_or_None, user_prompt).
    """
    system_parts: list[str] = []
    user_parts: list[str] = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "system":
            system_parts.append(content)
        elif role == "user":
            user_parts.append(content)
        elif role == "assistant":
            user_parts.append(f"[Previous assistant response]\n{content}")
        elif role == "tool":
            tool_id = msg.get("tool_call_id", "unknown")
            user_parts.append(f"[Tool result for {tool_id}]\n{content}")

    if tools:
        tool_descriptions = _format_tools(tools)
        user_parts.append(
            f"\n[Available tools]\n{tool_descriptions}\n\n"
            f"If you need to use a tool, respond with a JSON array of tool calls "
            f"in this exact format:\n"
            f'[{{"id": "call_1", "name": "tool_name", "arguments": {{...}}}}]\n'
            f"If you do not need any tools, respond normally with text."
        )

    system = "\n\n".join(system_parts) if system_parts else None
    user = "\n\n".join(user_parts)
    return system, user


def _format_tools(tools: list[dict]) -> str:
    """Format tool definitions into a readable string."""
    lines = []
    for tool in tools:
        func = tool.get("function", {})
        name = func.get("name", "unknown")
        desc = func.get("description", "")
        params = func.get("parameters", {})
        props = params.get("properties", {})
        param_list = ", ".join(
            f"{k}: {v.get('type', 'any')}" for k, v in props.items()
        )
        lines.append(f"- {name}({param_list}): {desc}")
    return "\n".join(lines)


def _call_claude(
    prompt: str,
    model: str,
    max_tokens: int,
    system_prompt: str | None = None,
) -> tuple[str, float, int, int]:
    """Shell out to claude -p. Returns (text, cost_usd, duration_ms, output_tokens).

    Uses --setting-sources user to skip project/local CLAUDE.md discovery
    (the inspector must not be influenced by project-level instructions)
    while preserving authentication. Uses --system-prompt to pass system
    messages via the proper channel instead of embedding them in the user
    prompt (which looks like a role-switch injection). Uses --allowedTools ""
    to ensure the model has no tool access.
    """
    cmd = [
        "claude", "-p", prompt,
        "--setting-sources", "user",
        "--output-format", "json",
        "--model", model,
        "--max-turns", "1",
        "--allowedTools", "",
    ]
    if system_prompt:
        cmd.extend(["--system-prompt", system_prompt])

    logger.debug("Calling claude -p (model=%s, prompt length=%d)", model, len(prompt))

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=300,
    )

    # Parse the JSON output — even on non-zero exit, the JSON may contain
    # error details in the result event
    stdout = result.stdout.strip()
    if not stdout:
        stderr = result.stderr.strip() if result.stderr else ""
        raise RuntimeError(f"claude -p failed (exit {result.returncode}): {stderr}")

    try:
        events = json.loads(stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Failed to parse claude -p output: {e}\n"
            f"Raw output (first 500 chars): {stdout[:500]}"
        ) from e

    text = ""
    cost = 0.0
    duration = 0
    output_tokens = 0

    for event in events:
        if event.get("type") == "result":
            if event.get("is_error"):
                raise RuntimeError(f"claude -p error: {event.get('result', 'unknown')}")
            text = event.get("result", "")
            cost = event.get("total_cost_usd", 0.0)
            duration = event.get("duration_ms", 0)
            usage = event.get("usage", {})
            output_tokens = usage.get("output_tokens", 0)

    return text, cost, duration, output_tokens


# ---------------------------------------------------------------------------
# Claude Code context decontamination
# ---------------------------------------------------------------------------

# Tools and capabilities that are Claude Code's own system context,
# not from the skill being analyzed. The inspector sees these because
# --system-prompt appends to (not replaces) the Claude Code system prompt.
# We strip these from inspector responses to avoid false positives.

CLAUDE_CODE_TOOLS = frozenset({
    # Core tools
    "agent", "askuserquestion", "bash", "config", "croncreate", "crondelete",
    "cronlist", "edit", "enterplanmode", "enterworktree", "exitplanmode",
    "exitworktree", "glob", "grep", "notebookedit", "read", "remotetrigger",
    "skill", "taskoutput", "taskstop", "todowrite", "toolsearch", "webfetch",
    "websearch", "write", "gh",
    # Skills (loaded as system context)
    "update-config", "debug", "simplify", "batch", "loop", "schedule",
    "claude-api", "textual-app-scaffold", "textual-custom-widgets",
    "textual-data-display", "textual-workers-async", "textual-forms-input",
    "textual-layout-css", "textual-screens-navigation", "textual-testing",
    "straight-talk", "video-outline", "video-script", "gadfly", "minesweeper",
    "roadmap", "req-architect", "expand-requirements", "frontend-design",
    "keybindings-help", "compact", "context", "cost", "heapdump", "init",
    "pr-comments", "release-notes", "review", "security-review", "insights",
    # SDK/config references from Claude Code system context
    "anthropic", "@anthropic-ai/sdk", "claude_agent_sdk",
    "settings.json", "settings.local.json", "claude.md",
    "~/.claude/keybindings.json",
})

CLAUDE_CODE_INSTRUCTION_MARKERS = [
    "ALWAYS use Grep for search tasks",
    "NEVER update the git config",
    "NEVER run destructive git commands",
    "NEVER skip hooks",
    "Always create NEW commits",
    "NEVER commit changes unless",
    "DO NOT push to the remote",
    "invoke the Skill tool BEFORE",
    "NEVER mention a skill without",
    "Always quote file paths",
    "Never use git commands with the -i flag",
    "Always pass git commit messages via a HEREDOC",
    "Do not add any fallback behavior",
    "TRIGGER the claude-api skill",
    "DO NOT TRIGGER claude-api",
    "claude-code-guide agent",
    "check if a claude-code-guide",
]


def decontaminate_inspector_response(parsed: dict) -> dict:
    """Remove Claude Code system context artifacts from an inspector response.

    The inspector's tools_referenced, capabilities_described, and
    instructions_to_agent may include items from Claude Code's own system
    prompt that leaked through. This strips known contaminants.
    """
    if not parsed:
        return parsed

    # Filter tools
    if "tools_referenced" in parsed:
        parsed["tools_referenced"] = [
            t for t in parsed["tools_referenced"]
            if t.lower() not in CLAUDE_CODE_TOOLS
        ]

    # Filter instructions that match Claude Code's behavioral rules
    if "instructions_to_agent" in parsed:
        cleaned = []
        for instruction in parsed["instructions_to_agent"]:
            is_claude_code = any(
                marker.lower() in instruction.lower()
                for marker in CLAUDE_CODE_INSTRUCTION_MARKERS
            )
            if not is_claude_code:
                cleaned.append(instruction)
        parsed["instructions_to_agent"] = cleaned

    return parsed


def _parse_tool_calls(text: str) -> list[ToolCall] | None:
    """Try to parse tool calls from the response text.

    The sandbox prompt instructs the model to respond with a JSON array
    of tool calls if it wants to use tools. This parser detects that format.
    """
    stripped = text.strip()

    # Try to find a JSON array in the response
    if not stripped.startswith("["):
        return None

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return None

    if not isinstance(parsed, list):
        return None

    tool_calls = []
    for item in parsed:
        if isinstance(item, dict) and "name" in item:
            tc = ToolCall(
                id=item.get("id", f"call_{len(tool_calls)}"),
                function=FunctionCall(
                    name=item["name"],
                    arguments=json.dumps(item.get("arguments", {})),
                ),
            )
            tool_calls.append(tc)

    return tool_calls if tool_calls else None
