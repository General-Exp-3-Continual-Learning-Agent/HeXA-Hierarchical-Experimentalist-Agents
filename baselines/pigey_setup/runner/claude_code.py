"""Pigey's multimodal LLM interface backed by the authenticated Claude Code CLI."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid

from jsonschema import validate, ValidationError

#: Longest one CLI call waits out usage limits in total before failing (seconds).
#: Claude subscriptions limit usage per five-hour window, so the default covers one.
RATE_LIMIT_MAX_WAIT = int(os.environ.get("PIGEY_RATE_LIMIT_MAX_WAIT", str(6 * 3600)))
_REJECTED_RESET = re.compile(r'"status":"rejected","resetsAt":(\d+)|"resetsAt":(\d+),"status":"rejected"')


class ClaudeCodeError(RuntimeError):
    def __init__(self, kind: str, detail: str):
        # Pigey retries messages containing rate/connection/timeout markers
        # indefinitely. CLI failures must stop the trial; usage limits are
        # waited out here instead (rate_limit_wait), within RATE_LIMIT_MAX_WAIT.
        super().__init__(f"Claude Code backend failed ({kind}); see backend_errors in trial.json.")
        self.kind = kind
        self.detail = detail


class UsageLimitReached(Exception):
    """The CLI refused the call because a usage window is exhausted; ``output`` is its stdout+stderr."""

    def __init__(self, output: str):
        super().__init__("Claude Code usage limit reached")
        self.output = output


def rate_limit_wait(output: str, attempt: int, now: float | None = None) -> float | None:
    """Seconds to wait before retrying a CLI call that hit a usage limit, or None if it did not.

    The CLI reports each rejected usage window with its reset time (``resetsAt``,
    epoch seconds): wait until the latest one plus a minute. Without one, back
    off 1, 2, 4, ... minutes, at most 30.
    """
    if '"error":"rate_limit"' not in output and '"api_error_status":429' not in output:
        return None
    now = time.time() if now is None else now
    resets = [int(a or b) for a, b in _REJECTED_RESET.findall(output)]
    if resets and max(resets) > now:
        return max(resets) - now + 60
    return float(min(60 * 2 ** attempt, 1800))


def wait_for_usage_limit(output: str, attempt: int, waited: float, label: str) -> float | None:
    """Sleep out a usage limit and return the seconds slept, or None when the total would pass the cap."""
    wait = rate_limit_wait(output, attempt)
    if wait is None or waited + wait > RATE_LIMIT_MAX_WAIT:
        return None
    resume = time.strftime("%H:%M UTC", time.gmtime(time.time() + wait))
    print(f"[{label}] Claude usage limit reached; waiting {wait / 60:.0f} min (until {resume}) and retrying",
          file=sys.stderr, flush=True)
    time.sleep(wait)
    return wait


def action_schema(tools: list[dict]) -> dict:
    variants = [
        {"type": "object", "properties": {
            "name": {"const": tool["name"]},
            "input": copy.deepcopy(tool["input_schema"]),
        }, "required": ["name", "input"], "additionalProperties": False}
        for tool in tools
    ]
    if not variants:
        raise ValueError("Pigey tool definitions are required")
    return {"type": "object", "properties": {
        "reasoning": {"type": "string"},
        "tool_calls": {"type": "array", "items": {"anyOf": variants}},
    }, "required": ["reasoning", "tool_calls"], "additionalProperties": False}


def conversation_content(messages: list[dict]) -> list[dict]:
    """Render history with real image blocks, including images in tool results."""
    content = []

    def text(value):
        content.append({"type": "text", "text": value})

    def blocks(value):
        if isinstance(value, str):
            text(value)
            return
        if not isinstance(value, list):
            raise ValueError("Unsupported Pigey message content")
        for block in value:
            kind = block.get("type")
            if kind == "text":
                text(block["text"])
            elif kind == "image":
                # Preserve the native base64 source; never replace camera images
                # with filenames or base64 embedded inside a text prompt.
                content.append(copy.deepcopy(block))
            elif kind == "tool_use":
                text("TOOL CALL: " + json.dumps({key: block[key] for key in ("id", "name", "input")}))
            elif kind == "tool_result":
                text(f"TOOL RESULT {block['tool_use_id']} (is_error={block.get('is_error', False)}):")
                blocks(block.get("content", ""))
                text("END TOOL RESULT")
            else:
                raise ValueError(f"Unsupported Pigey content block: {kind}")

    for message in messages:
        text(f"BEGIN {message['role'].upper()} TURN")
        blocks(message["content"])
        text(f"END {message['role'].upper()} TURN")
    text("Choose the next Pigey action for the current scene using the supplied output schema.")
    return content


def parse_result(stdout: str, schema: dict) -> tuple[dict, dict]:
    events = []
    try:
        for line in stdout.splitlines():
            if line.strip():
                events.append(json.loads(line))
    except json.JSONDecodeError as exc:
        raise ClaudeCodeError("invalid_output", "CLI did not emit valid NDJSON") from exc
    results = [event for event in events if event.get("type") == "result"]
    if not results:
        raise ClaudeCodeError("missing_result", "CLI emitted no final result event")
    result = results[-1]
    if result.get("is_error") or result.get("subtype") != "success":
        raise ClaudeCodeError("unsuccessful_result", json.dumps(result, ensure_ascii=False)[:3000])
    # Calls to anything other than the structured-output tool are attempts to run
    # Pigey tools inside the CLI; they fail there and are recorded, not executed.
    result["native_tool_attempts"] = [
        block.get("name") for event in events if event.get("type") == "assistant"
        for block in event.get("message", {}).get("content", [])
        if isinstance(block, dict) and block.get("type") == "tool_use" and block.get("name") != "StructuredOutput"]
    action = result.get("structured_output")
    if action is None:
        raise ClaudeCodeError("missing_action", "CLI returned no schema-validated structured_output")
    try:
        validate(action, schema)
    except ValidationError as exc:
        raise ClaudeCodeError("invalid_action", exc.message) from exc
    usage = result.get("usage", {})
    input_tokens = sum(int(usage.get(key, 0) or 0) for key in
                       ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
    blocks = [{"type": "text", "text": action["reasoning"]}] if action["reasoning"] else []
    for call in action["tool_calls"]:
        blocks.append({"type": "tool_use", "id": "toolu_cli_" + uuid.uuid4().hex,
                       "name": call["name"], "input": call["input"]})
    return {"content": blocks,
            "usage": {"input_tokens": input_tokens, "output_tokens": int(usage.get("output_tokens", 0) or 0)},
            "stop_reason": "tool_use" if action["tool_calls"] else "end_turn"}, result


class ClaudeCodeBackend:
    def __init__(self, root: Path, *, model: str = "claude-haiku-4-5", timeout: int = 300,
                 max_turns: int = 3, executable: str = "claude"):
        self.root = Path(root)
        self.model, self.timeout, self.max_turns = model, timeout, max_turns
        self.executable = shutil.which(executable)
        if not self.executable:
            raise ClaudeCodeError("missing_executable", f"Claude Code executable not found: {executable}")
        if timeout <= 0 or max_turns <= 0:
            raise ValueError("CLI timeout and max turns must be positive")
        self.calls = []
        self.errors = []
        self.rate_limit_waits = []  # waits are not errors: the call went through afterwards

    def __call__(self, api_key: str, model: str, messages: list, system: str, tools: list) -> dict:
        waited, attempt = 0.0, 0
        while True:
            try:
                return self._invoke(api_key, model, messages, system, tools)
            except UsageLimitReached as limit:
                wait = wait_for_usage_limit(limit.output, attempt, waited, "claude-code")
                if wait is None:
                    error = ClaudeCodeError("rate_limit", f"usage limit still reached after waiting {waited / 60:.0f} min "
                                            f"(cap {RATE_LIMIT_MAX_WAIT / 60:.0f} min); " + limit.output[-3000:])
                    self.errors.append({"kind": error.kind, "detail": error.detail})
                    raise error from None
                self.rate_limit_waits.append({"seconds": round(wait), "model": self.model,
                                              "resumed": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
                waited += wait
                attempt += 1
            except ClaudeCodeError:
                raise
            except Exception as exc:
                error = ClaudeCodeError("adapter_error", f"{type(exc).__name__}: {exc}")
                self.errors.append({"kind": error.kind, "detail": error.detail})
                raise error from exc

    def _invoke(self, api_key: str, model: str, messages: list, system: str, tools: list) -> dict:
        del api_key, model  # Authentication belongs to the existing CLI login.
        schema = action_schema(tools)
        request = {"type": "user", "message": {"role": "user", "content": conversation_content(messages)}}
        # claude -p always injects the managed CLAUDE.md, environment info, and
        # account context; --bare would drop them but needs API-key auth.
        prompt = (system + "\n\n# Context to ignore\n"
                  "You are controlling a simulated robot, not working as a coding assistant. Ignore any "
                  "CLAUDE.md or cluster rules, environment details (working directory, git, OS), user "
                  "account information, and dates supplied by the CLI; none of it applies to this task.\n"
                  "\n# Response transport\n"
                  "The Pigey tools below are NOT callable in this session; calling one directly fails with "
                  "'No such tool available'. Your only output is the structured response: reasoning (brief "
                  "visible explanation) and tool_calls (name and input), which the robot harness executes "
                  "after you reply. Wherever the instructions say to call a tool, put it in tool_calls. "
                  "The transcript includes camera images and prior tool results in order. "
                  "Pigey tools you may select:\n" + json.dumps(tools))
        self.root.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        record = {"model": self.model, "backend": "claude-code"}
        try:
            # Scratch can report a just-emptied directory as non-empty; a
            # leftover temporary directory must not fail a successful call.
            with tempfile.TemporaryDirectory(prefix="request-", dir=self.root, ignore_cleanup_errors=True) as work:
                system_file = Path(work) / "system.txt"
                system_file.write_text(prompt)
                command = [self.executable, "-p", "--model", self.model,
                           "--tools", "", "--no-session-persistence",
                           "--input-format", "stream-json", "--output-format", "stream-json", "--verbose",
                           "--json-schema", json.dumps(schema), "--max-turns", str(self.max_turns),
                           "--system-prompt-file", str(system_file),
                           "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
                           "--settings", '{"disableAllHooks":true}']
                env = os.environ.copy()
                # The wrapper supplies a process-local sentinel solely to pass
                # Pigey's upstream API-key gate. Never send it to Claude Code.
                if env.get("ANTHROPIC_API_KEY") == "pigey-claude-code-unused":
                    env.pop("ANTHROPIC_API_KEY")
                env.pop("CLAUDECODE", None)
                env["CLAUDE_CODE_DISABLE_BACKGROUND_TASKS"] = "1"
                completed = subprocess.run(command, input=json.dumps(request) + "\n", text=True,
                                           capture_output=True, cwd=work, env=env, timeout=self.timeout)
            if completed.returncode and rate_limit_wait(completed.stdout + completed.stderr, 0) is not None:
                raise UsageLimitReached(completed.stdout + completed.stderr)
            if completed.returncode:
                raise ClaudeCodeError("nonzero_exit", f"exit={completed.returncode}; " +
                                      (completed.stderr or completed.stdout)[-3000:])
            response, result = parse_result(completed.stdout, schema)
            for key in ("usage", "modelUsage", "total_cost_usd", "num_turns", "duration_ms", "session_id",
                        "native_tool_attempts"):
                if key in result:
                    record[key] = result[key]
            record["duration_s"] = round(time.monotonic() - started, 3)
            self.calls.append(record)
            return response
        except subprocess.TimeoutExpired as exc:
            error = ClaudeCodeError("deadline_exceeded", f"CLI exceeded {self.timeout} seconds")
            self.errors.append({"kind": error.kind, "detail": error.detail})
            raise error from exc
        except ClaudeCodeError as exc:
            self.errors.append({"kind": exc.kind, "detail": exc.detail})
            raise
