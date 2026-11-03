"""ReasoningBank's memory-extraction calls.

Two call types, matching the paper's Appendix A.1 (Figure 9):

  success   extract <=3 generalizable *strategy* items from a trajectory the
            environment verified as solved;
  failure   reflect on a failed trajectory and extract <=3 *lesson* items that
            would prevent the failure next time.

Both prompts are ports of the paper's web-navigation instructions, reworded for
2D physics placement puzzles (as `baselines/memory/summarize.py` does for
MemRL). What must not change with the task is the *structure* — reason first,
then at most 3 non-overlapping items in the title/description/content schema, at
extraction temperature 1.0 — because that is the method.

Two backends, mirroring `baselines/memory/summarize.py`:

  `make_rb_extractor`     any local `model_fn(messages, temp, max_tokens) -> str`
                          (e.g. Qwen2.5-7B-Instruct via `load_qwen_model`). The
                          paper sets the extractor to the *same backbone* as the
                          agent, so one Qwen on the GPU serves both roles.
  `make_cli_extractor`    `claude -p` subprocess, for a Claude actor. Note
                          `claude -p` exposes no temperature flag, so the paper's
                          T=1.0 is not settable on this backend.
"""

from __future__ import annotations

import re
import time
from typing import Callable, Optional

from baselines.memory.summarize import claude_cli_call

# ── Prompts (ports of ReasoningBank Figure 9, reworded for InterPhyre) ────────

SUCCESSFUL_SI = """You are an expert in 2D physics placement puzzles. You will be given a task (level description and scene geometry) and a trajectory showing **how an agent successfully solved it**.

## Guidelines
Extract and summarize useful insights as memory items based on the successful trajectory.
The goal of the memory items is to be helpful and generalizable for future similar puzzle instances (different seeds, shifted geometry).

## Important notes
  - First think about WHY the trajectory succeeded, then summarize the insights.
  - You can extract *at most 3* memory items from the trajectory.
  - Do not repeat similar or overlapping items.
  - Prefer concrete, actionable physics reasoning (name objects, state the mechanism, give approximate offsets or coordinate ranges). Do NOT hard-code the exact winning coordinates or seed-specific numbers; capture the transferable strategy instead.

## Output Format
Your output must strictly follow the Markdown format shown below:

```
# Memory Item 1
## Title <a short title for the memory item>
## Description <one sentence describing when (or when NOT) to use this item>
## Content <1-3 sentences describing the strategy learned for solving similar instances>
```
"""

FAILED_SI = """You are an expert in 2D physics placement puzzles. You will be given a task (level description and scene geometry) and a trajectory showing **how an agent attempted the puzzle but failed**.

## Guidelines
Extract and summarize useful insights as memory items based on the failed trajectory.
The goal of the memory items is to be helpful and generalizable for future similar puzzle instances (different seeds, shifted geometry).

## Important notes
  - First reflect on WHY the trajectory failed (wrong physics assumption, wrong placement region, wasted budget, ...), then summarize the lessons or preventative strategies.
  - You can extract *at most 3* memory items from the trajectory.
  - Do not repeat similar or overlapping items.
  - Prefer concrete, actionable recovery procedures (what to check, what region or offset to try instead, which assumption to drop). Do NOT hard-code exact coordinates or seed-specific numbers; capture the transferable lesson instead.

## Output Format
Your output must strictly follow the Markdown format shown below:

```
# Memory Item 1
## Title <a short title for the memory item>
## Description <one sentence describing when (or when NOT) to use this item>
## Content <1-3 sentences describing the lesson learned to avoid this failure on similar instances>
```
"""

_SYSTEMS = {"success": SUCCESSFUL_SI, "failure": FAILED_SI}


# ── Parsing ───────────────────────────────────────────────────────────────

_ITEM_RE = re.compile(r"^\s*#\s+Memory\s+Item\b", re.IGNORECASE)
_FIELD_RE = re.compile(r"^\s*#{1,4}\s*(Title|Description|Content)\b\s*[:\-.]?\s*(.*)$", re.IGNORECASE)


def _strip_fences(text: str) -> str:
    """Drop surrounding ``` fences the model sometimes wraps the output in."""
    lines = []
    for line in text.splitlines():
        if line.strip().startswith("```"):
            continue
        lines.append(line)
    return "\n".join(lines)


def parse_memory_items(text: str, *, max_items: int = 3) -> list[dict]:
    """Parse the extractor's Markdown into a list of {title, description, content}.

    Tolerant of the common variations: the field value may sit on the header
    line (`## Title Foo`) or on the following lines, field headers may use any of
    `#`..`####`, and the whole block may be fenced. Items with no usable text are
    dropped. A non-empty blob that matches no item/field header at all is kept as
    a single untitled item so an off-format extraction is never silently lost.
    """
    if not text or not text.strip():
        return []

    body = _strip_fences(text)
    lines = body.splitlines()

    items: list[dict] = []
    current: Optional[dict] = None
    field: Optional[str] = None

    def _flush() -> None:
        nonlocal current
        if current is None:
            return
        for key in ("title", "description", "content"):
            current[key] = current[key].strip()
        if current["title"] or current["content"] or current["description"]:
            items.append(current)
        current = None

    for line in lines:
        if _ITEM_RE.match(line):
            _flush()
            current = {"title": "", "description": "", "content": ""}
            field = None
            continue

        m = _FIELD_RE.match(line)
        if m:
            if current is None:
                current = {"title": "", "description": "", "content": ""}
            field = m.group(1).lower()
            inline = m.group(2).strip()
            if inline:
                current[field] = (current[field] + " " + inline).strip()
            continue

        if current is not None and field is not None and line.strip():
            current[field] = (current[field] + " " + line.strip()).strip()

    _flush()

    if not items:
        # Off-format but non-empty: keep it rather than swallow it.
        blob = body.strip()
        if blob:
            items = [{"title": "", "description": "", "content": blob}]

    return items[:max_items]


# ── Extractor backends ──────────────────────────────────────────────────────


def make_rb_extractor(
    model_fn,
    *,
    temperature: float = 1.0,
    max_new_tokens: int = 1024,
    max_items: int = 3,
    verbose: bool = False,
    systems: Optional[dict] = None,
) -> Callable[[bool, str, str], list[dict]]:
    """Return `extract(success, intent, trajectory_text) -> list[dict]`.

    Uses a local `model_fn(messages, temperature, max_tokens) -> str | {content}`,
    the same callable the actor loaded, so no second model is held in memory. The
    paper's extraction temperature (1.0) is honoured here (the `claude -p`
    backend below cannot set it).
    """
    prompts = systems or _SYSTEMS

    def extract(success: bool, intent: str, trajectory_text: str) -> list[dict]:
        kind = "success" if success else "failure"
        system = prompts[kind]
        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": f"Task:\n{intent}\n\nTrajectory:\n{trajectory_text}",
            },
        ]
        start = time.time()
        try:
            out = model_fn(messages, temperature, max_new_tokens)
        except Exception as exc:  # noqa: BLE001 — never kill a run over one episode
            print(f"[ReasoningBank extractor] {type(exc).__name__}: {exc}", flush=True)
            return []
        if isinstance(out, dict):  # OSS-style model_fns return {thinking, content}
            out = out.get("content") or ""
        text = str(out).strip()
        items = parse_memory_items(text, max_items=max_items)
        if not items:
            print(
                f"  [ReasoningBank {kind}] WARNING: extractor produced no usable items",
                flush=True,
            )
        elif verbose:
            print(
                f"  [ReasoningBank {kind} @ T={temperature}] {len(items)} item(s) "
                f"in {time.time() - start:.1f}s",
                flush=True,
            )
        return items

    return extract


def make_cli_extractor(
    *,
    model: str = "claude-sonnet-4-6",
    max_items: int = 3,
    verbose: bool = False,
    timeout: int = 900,
) -> Callable[[bool, str, str], list[dict]]:
    """Return an extractor backed by the `claude -p` CLI (for a Claude actor).

    `claude -p` has no temperature flag, so the paper's T=1.0 is not reproducible
    on this backend — same limitation noted for MemRL's summarizer.
    """

    def extract(success: bool, intent: str, trajectory_text: str) -> list[dict]:
        kind = "success" if success else "failure"
        system = _SYSTEMS[kind]
        prompt = (
            f"SYSTEM:\n{system}\n\n"
            f"USER:\nTask:\n{intent}\n\nTrajectory:\n{trajectory_text}\n\n"
            f"ASSISTANT:\n"
        )
        start = time.time()
        text = claude_cli_call(prompt, model=model, timeout=timeout).strip()
        items = parse_memory_items(text, max_items=max_items)
        if not items:
            print(
                f"  [ReasoningBank {kind}] WARNING: extractor produced no usable items",
                flush=True,
            )
        elif verbose:
            print(
                f"  [ReasoningBank {kind}] {len(items)} item(s) in "
                f"{time.time() - start:.1f}s",
                flush=True,
            )
        return items

    return extract


def mock_extractor(success: bool, intent: str, trajectory_text: str) -> list[dict]:
    """Deterministic stand-in for smoke tests (no LLM call)."""
    del intent
    tag = "strategy" if success else "lesson"
    return [
        {
            "title": f"mock {tag}",
            "description": f"when the scene resembles this {'solved' if success else 'failed'} case",
            "content": f"[mock {tag}] distilled from {len(trajectory_text)} chars of trajectory.",
        }
    ]
