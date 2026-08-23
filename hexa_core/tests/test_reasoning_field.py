"""Regression suite for the reasoning-channel field name.

This exists because of a bug that every other suite passed straight through.

CORE's ``NvidiaClient`` reads ``reasoning_content`` (`nvidia_client.py:137`), which
is correct for NVIDIA NIM. Our client copied that, and the parity tests copied it
too -- their fake responses carried ``reasoning_content``, so they confirmed our
assumption instead of testing it.

But vLLM 0.26.0 renamed the **response** field to ``reasoning``. Verified in the
installed package:

* ``ChatMessage.reasoning``  -- ``entrypoints/openai/chat_completion/protocol.py:65``
* ``DeltaMessage.reasoning`` -- ``entrypoints/openai/engine/protocol.py:396``
* ``GptOssReasoningParser.extract_reasoning`` raises ``NotImplementedError``
  ("only provides boundary detection. Use HarmonyParser for output parsing"), and
  ``HarmonyParser.parse`` returns ``(reasoning, content, tool_calls)``
  -- ``parser/harmony.py:210-212``
* the ``reasoning_content`` -> ``reasoning`` shim at
  ``chat_completion/protocol.py:503-505`` runs on **inbound request** messages only.
  There is no alias on the response.

Reading only ``reasoning_content`` therefore yields ``None`` against vLLM >=0.26,
``raw_text`` collapses to the final channel, and the chain-of-thought vanishes from
the string the verifier parses. Nothing raises. Accuracy simply drops -- worst on
Tower of Hanoi, whose ``_extract_moves_from_text`` takes the *last* parseable move
list, so which candidate wins depends on the reasoning being present at all.

The final test cross-checks ``REASONING_FIELDS`` against the *installed* vLLM
protocol classes, so a future rename is caught here rather than in a 900-call gate.
It self-skips where vLLM is absent (the system interpreter), which is exactly why
it cannot be the only defence -- hence the fakes above it.
"""

from __future__ import annotations

import importlib.util
from types import SimpleNamespace
from typing import Any

from hexa_core.llm.base import REASONING_FIELDS, Generation, _pick_reasoning
from hexa_core.tests import check, check_eq, run_module

VLLM_SHAPE = "reasoning"
NIM_SHAPE = "reasoning_content"


def _message(**fields: Any) -> SimpleNamespace:
    """A non-streaming ``choices[0].message`` carrying only the given fields."""
    return SimpleNamespace(**fields)


def _chunk(**delta_fields: Any) -> SimpleNamespace:
    """A streaming chunk whose delta carries only the given fields."""
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(**delta_fields), finish_reason=None)]
    )


# ---------------------------------------------------------------- field selection


def test_picks_vllm_field() -> None:
    """vLLM >=0.26 shape: only ``reasoning`` is present."""
    text, field = _pick_reasoning(_message(reasoning="THINKING", content="ANSWER"))
    check_eq("vllm reasoning text", text, "THINKING")
    check_eq("vllm reasoning field", field, VLLM_SHAPE)


def test_picks_nim_field() -> None:
    """NIM / older-vLLM shape: only ``reasoning_content`` is present.

    Back-compatibility is not optional -- ``nim_client`` is the reference provider
    path used to quantify the offset against CORE's published numbers.
    """
    text, field = _pick_reasoning(_message(reasoning_content="THINKING", content="ANSWER"))
    check_eq("nim reasoning text", text, "THINKING")
    check_eq("nim reasoning field", field, NIM_SHAPE)


def test_absent_reasoning_is_reported_not_guessed() -> None:
    """Neither field present -> no text, no field name. This is the misconfig case."""
    text, field = _pick_reasoning(_message(content="ANSWER"))
    check("absent reasoning -> falsy text", not text, f"got {text!r}")
    check_eq("absent reasoning -> no field", field, None)


def test_empty_field_is_not_a_match() -> None:
    """A present-but-empty field must not be reported as the live one.

    An empty analysis channel and an absent one look identical downstream, and the
    absent case is the one that means the server is misconfigured. Claiming the
    field "fired" would put a reassuring name in the manifest for a broken run.
    """
    _, field = _pick_reasoning(_message(reasoning="", content="ANSWER"))
    check_eq("empty reasoning -> no field claimed", field, None)


def test_newest_field_wins_when_both_present() -> None:
    """A server emitting both must resolve to the modern name, deterministically."""
    text, field = _pick_reasoning(
        _message(reasoning="NEW", reasoning_content="OLD", content="ANSWER")
    )
    check_eq("both present -> newest text", text, "NEW")
    check_eq("both present -> newest field", field, VLLM_SHAPE)
    check_eq("REASONING_FIELDS order", REASONING_FIELDS, (VLLM_SHAPE, NIM_SHAPE))


# ---------------------------------------------------------------- consequence


def test_dropping_reasoning_changes_the_scored_answer() -> None:
    """The concrete harm, on the task where it bites hardest.

    A Hanoi response that reasons its way past a wrong candidate to a correct one.
    CORE parses ``reasoning + final``; its ``_extract_moves_from_text`` takes the
    LAST parseable move list. Include the reasoning and the correct list is last;
    drop it and the *wrong* list is all that a naive reader would have seen.
    """
    reasoning = "First try moves = [[1,0,1]] -- illegal, discard."
    final = "moves = [[1,0,2],[2,0,1],[1,2,1]]"

    with_reasoning = Generation.from_parts(reasoning, final, "stop", None).raw_text
    without = Generation.from_parts("", final, "stop", None).raw_text

    check_eq("no separator inserted", with_reasoning, reasoning + final)
    check("dropping reasoning changes the string", with_reasoning != without)
    check(
        "the discarded candidate is only visible with reasoning",
        "[[1,0,1]]" in with_reasoning and "[[1,0,1]]" not in without,
    )

    from hexa_core.tests import core_module

    hanoi = core_module("tasks.single_turn.tower_of_hanoi_task")
    check_eq(
        "CORE parses the LAST list from the full text",
        hanoi._extract_moves_from_text(with_reasoning),
        [[1, 0, 2], [2, 0, 1], [1, 2, 1]],
    )


def test_streaming_and_nonstreaming_agree() -> None:
    """Both transports must reach the same ``raw_text`` for the same content.

    The two paths read the field independently, so a fix applied to one and not the
    other would show up only as a streaming-vs-not accuracy gap -- untraceable.
    """
    reasoning, final = "THINK", "ANSWER"

    streamed_r, streamed_f = [], []
    for chunk in (_chunk(reasoning=reasoning), _chunk(content=final)):
        delta = chunk.choices[0].delta
        text, _ = _pick_reasoning(delta)
        if text:
            streamed_r.append(text)
        if getattr(delta, "content", None):
            streamed_f.append(delta.content)

    streamed = Generation.from_parts("".join(streamed_r), "".join(streamed_f), None, None)
    single = Generation.from_parts(
        *(_pick_reasoning(_message(reasoning=reasoning, content=final))[0], final), None, None
    )
    check_eq("streaming == non-streaming", streamed.raw_text, single.raw_text)


# ---------------------------------------------------------------- live cross-check


def test_matches_installed_vllm_protocol() -> None:
    """Assert ``REASONING_FIELDS`` still covers what the installed vLLM declares.

    Skips where vLLM is absent. That is the common case for this suite's usual
    interpreter, which is precisely why the fake-based tests above exist -- this
    check is a bonus that fires in the serving env, not the primary defence.
    """
    if importlib.util.find_spec("vllm") is None:
        print("    SKIP test_matches_installed_vllm_protocol (vllm not importable here)")
        return

    from vllm.entrypoints.openai.chat_completion.protocol import ChatMessage
    from vllm.entrypoints.openai.engine.protocol import DeltaMessage

    for cls in (ChatMessage, DeltaMessage):
        declared = {
            name
            for name in cls.model_fields
            if "reason" in name.lower()
        }
        check(
            f"{cls.__name__} reasoning fields are all known",
            declared and declared <= set(REASONING_FIELDS),
            f"{cls.__name__} declares {sorted(declared)}; "
            f"REASONING_FIELDS={REASONING_FIELDS}. A new name here means the "
            f"chain-of-thought would be silently dropped -- add it.",
        )


if __name__ == "__main__":
    raise SystemExit(run_module(__import__(__name__, fromlist=["_"])))
