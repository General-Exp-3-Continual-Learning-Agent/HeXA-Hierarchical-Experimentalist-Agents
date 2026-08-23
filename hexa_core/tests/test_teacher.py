"""Tests for the vLLM teacher branch (`PHASE3_DESIGN.md` §6, build-order step 1).

The central assertion is the channel choice. The actor is scored on
``reasoning + final`` for CORE parity; the teacher must read ``final`` alone, because
its callers parse JSON with a greedy pattern. Getting that backwards does not raise --
it returns an empty skill list with a printed warning, which reads as "the method
found nothing". So it is asserted here against `distill`'s *real* parsers, not against
a restatement of them.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any

from hexa_core.llm.base import Generation
from hexa_core.llm.teacher import (
    VLLM_PREFIX,
    VLLMTeacher,
    resolve_teacher_model,
    teacher_base_url,
)
from hexa_core.tests import check, check_eq, run_module

# A response whose analysis channel contains a *discarded draft* in the same JSON
# shape as the answer. This is the realistic case, not a contrived one: gpt-oss drafts
# structured output in the analysis channel routinely.
DRAFT_REASONING = 'Let me draft. Maybe [{"title": "DRAFT-WRONG"}] -- no, reconsider.'
REAL_FINAL = '[{"title": "REAL-ANSWER", "principle": "p", "when_to_apply": "w"}]'


class _FakeClient:
    """Stands in for VLLMClient, recording what it was asked for."""

    def __init__(self, reasoning: str, final: str, finish_reason: str | None = "stop") -> None:
        self._generation = Generation.from_parts(reasoning, final, finish_reason, None)
        self.base_url = "http://fake:8000/v1"
        self.reasoning_field = "reasoning"
        self.calls: list[dict[str, Any]] = []

    def generate_structured(self, **kwargs: Any) -> Generation:
        self.calls.append(kwargs)
        return self._generation


def _teacher(reasoning: str, final: str, finish_reason: str | None = "stop") -> VLLMTeacher:
    """A VLLMTeacher with its client swapped out, bypassing __init__'s network work."""
    teacher = VLLMTeacher.__new__(VLLMTeacher)
    teacher.model = "openai/gpt-oss-120b"
    teacher.temperature = 0.3
    teacher.max_tokens = 8192
    teacher.seed = None
    teacher.client = _FakeClient(reasoning, final, finish_reason)
    teacher.calls = 0
    teacher.truncated = 0
    teacher.empty_final = 0
    teacher.last_reasoning = ""
    teacher.last_finish_reason = None
    return teacher


# ------------------------------------------------------------------ the channel rule


def test_returns_final_channel_only() -> None:
    """The teacher must not return the actor's raw_text concatenation."""
    teacher = _teacher(DRAFT_REASONING, REAL_FINAL)
    out = teacher("distil these trajectories")
    check_eq("teacher returns final only", out, REAL_FINAL)
    check("draft reasoning is excluded", "DRAFT-WRONG" not in out)


def test_real_distill_parser_needs_final_only() -> None:
    """Assert against `distill`'s ACTUAL parsers, not a copy of them.

    This is the test that justifies the whole module. If someone later "fixes" the
    teacher to return raw_text for consistency with the actor, this fails loudly
    instead of silently producing zero skills.
    """
    from skillrl.distillation.distill import _parse_json_array

    final_only = _parse_json_array(REAL_FINAL)
    check_eq("final-only parses to one skill", len(final_only), 1)
    check_eq("and it is the real answer", final_only[0]["title"], "REAL-ANSWER")

    concatenated = _parse_json_array(DRAFT_REASONING + REAL_FINAL)
    # The greedy fallback spans draft-open-bracket .. final-close-bracket. Whatever it
    # yields, it must NOT be the correct single-skill answer -- that is the bug.
    check(
        "concatenated channels do NOT yield the right answer",
        concatenated != final_only,
        f"greedy parse returned {concatenated!r}; if this now equals the "
        f"final-only result the hazard has changed and the docs need revisiting",
    )

    teacher = _teacher(DRAFT_REASONING, REAL_FINAL)
    check_eq(
        "teacher output parses correctly through the real parser",
        _parse_json_array(teacher("p")),
        final_only,
    )


# ------------------------------------------------------------------ generation config


def test_uses_teacher_temperature_not_actor_temperature() -> None:
    """0.3 (HF-teacher parity), not 0.6 (CORE's scored-path parity)."""
    teacher = _teacher("", REAL_FINAL)
    teacher("p")
    sent = teacher.client.calls[0]
    check_eq("teacher temperature", sent["temperature"], 0.3)


def test_accepts_message_list_like_hf_branch() -> None:
    """`_get_hf_teacher` returns a callable taking messages; stay drop-in."""
    teacher = _teacher("", REAL_FINAL)
    out = teacher([{"role": "user", "content": "distil this"}])
    check_eq("message list accepted", out, REAL_FINAL)
    check("content reached the client", "distil this" in teacher.client.calls[0]["prompt"])


# ------------------------------------------------------------------ diagnostics


def test_truncation_is_counted() -> None:
    """finish_reason == 'length' is the usual cause of unparseable teacher JSON."""
    teacher = _teacher("thinking", '[{"title": "cut off"', finish_reason="length")
    teacher("p")
    check_eq("truncation counted", teacher.truncated, 1)
    check_eq("stats reports it", teacher.stats()["truncated"], 1)


def test_empty_final_is_counted_separately_from_failure() -> None:
    """All budget spent reasoning is not the same as a failed call."""
    teacher = _teacher("thought for a very long time", "", finish_reason="length")
    out = teacher("p")
    check_eq("empty final returns empty string", out, "")
    check_eq("empty_final counted", teacher.empty_final, 1)


def test_stats_records_the_channel() -> None:
    """The manifest must say which channel was distilled from."""
    teacher = _teacher("r", REAL_FINAL)
    teacher("p")
    stats = teacher.stats()
    check_eq("channel recorded", stats["channel"], "final_text")
    check_eq("calls counted", stats["calls"], 1)


# ------------------------------------------------------------------ dispatch


def test_explicit_prefix_selects_vllm() -> None:
    check_eq(
        "vllm: prefix strips to served name",
        resolve_teacher_model(VLLM_PREFIX + "openai/gpt-oss-120b"),
        "openai/gpt-oss-120b",
    )


def test_claude_name_is_not_claimed() -> None:
    """The existing claude- branch must keep working untouched."""
    check_eq("claude not routed to vllm", resolve_teacher_model("claude-sonnet-4-6"), None)


def test_gpt_oss_autodetected_only_with_endpoint() -> None:
    """Auto-detection is a guard against the HF fallback loading a 117B checkpoint.

    Without an endpoint there is nothing to auto-detect *to*, so it must decline and
    let the normal dispatch happen.
    """
    saved = {
        name: os.environ.pop(name, None)
        for name in (
            "HEXA_TEACHER_BASE_URL",
            "HEXA_VLLM_BASE_URL",
            "HEXA_VLLM_URL",
            "VLLM_BASE_URL",
        )
    }
    try:
        check_eq("no endpoint -> declines", resolve_teacher_model("openai/gpt-oss-120b"), None)
        check_eq("no endpoint -> teacher_base_url None", teacher_base_url(), None)

        os.environ["HEXA_TEACHER_BASE_URL"] = "http://gpu020:8000/v1"
        check_eq(
            "endpoint set -> claims gpt-oss",
            resolve_teacher_model("openai/gpt-oss-120b"),
            "openai/gpt-oss-120b",
        )
        check_eq(
            "endpoint set -> still declines a qwen name",
            resolve_teacher_model("Qwen/Qwen2.5-7B-Instruct"),
            None,
        )
    finally:
        os.environ.pop("HEXA_TEACHER_BASE_URL", None)
        for name, value in saved.items():
            if value is not None:
                os.environ[name] = value


def test_call_teacher_routes_to_vllm(monkeypatch: Any = None) -> None:
    """End-to-end through the real `call_teacher`, with the network stubbed."""
    import hexa_core.llm.teacher as teacher_module
    from skillrl.distillation import distill

    sentinel = _teacher(DRAFT_REASONING, REAL_FINAL)
    original = teacher_module.get_vllm_teacher
    teacher_module.get_vllm_teacher = lambda model, base_url=None: sentinel
    try:
        out = distill.call_teacher("distil", model=VLLM_PREFIX + "openai/gpt-oss-120b")
        check_eq("call_teacher returned final channel", out, REAL_FINAL)
        check_eq("the teacher was actually invoked", sentinel.calls, 1)
    finally:
        teacher_module.get_vllm_teacher = original


if __name__ == "__main__":
    raise SystemExit(run_module(__import__(__name__, fromlist=["_"])))
