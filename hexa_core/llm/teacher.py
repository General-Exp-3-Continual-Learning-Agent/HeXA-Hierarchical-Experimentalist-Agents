"""The teacher/evolver backend: gpt-oss-120b on local vLLM, no API key.

`skillrl.distillation.distill.call_teacher` ships two branches -- a `claude-` CLI
subprocess and a local-HuggingFace fallback. Neither can serve the configuration
this project committed to (gpt-oss-120b, open weights, local vLLM, no API keys), so
the evolver could not run at all. `PHASE3_DESIGN.md` §6 lists this as the hard
blocker ahead of everything else in the loop.

Why this is a separate module rather than inline in `distill.py`
---------------------------------------------------------------
The actor and the teacher want *opposite* things from the same server, and the
difference is not cosmetic.

**The actor** must score ``reasoning + final`` concatenated, because that is what
CORE does (`nvidia_client.py:141`, `run_core.py:433`) and the comparison rests on
byte-parity with it. `Generation.from_parts` transcribes that rule.

**The teacher** must read the **final channel only**. Its callers parse JSON out of
the response with a *greedy* pattern -- `distill._parse_json_array` falls back to
``re.search(r"\\[.*\\]", text, re.DOTALL)``. Greedy ``.*`` spans from the first ``[``
anywhere in the text to the last ``]`` anywhere in it. gpt-oss's analysis channel
routinely contains draft JSON, so concatenating the channels makes that span run
from a *draft* bracket to the *final* closing bracket. Verified:

    reasoning = 'Let me draft. Maybe [{"title": "DRAFT-WRONG"}] no, reconsider.'
    final     = '[{"title": "REAL-ANSWER"}]'

    _parse_json_array(final)             -> [{'title': 'REAL-ANSWER'}]
    _parse_json_array(reasoning + final) -> parse failure

and a parse failure returns ``[]`` with a printed warning, not an exception. So the
whole distillation pass would yield **zero skills** while looking like "the teacher
didn't find anything" -- a silent null result attributable to the method rather than
to a channel bug. Hence `final_text`, and hence a named module explaining why, so
nobody later "fixes" this to use `raw_text` for consistency with the actor.

Temperature
-----------
0.3, matching the existing HF teacher branch (`distill._get_hf_teacher`) rather than
the actor's CORE-mandated 0.6. Distillation is a structured-output task where
determinism is worth more than diversity, and the 0.6 figure is a parity constraint
on the *scored* path only -- it carries no authority here.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from hexa_core.llm.vllm_client import VLLMClient

# Matches `distill._get_hf_teacher`, not the actor's PAPER_TEMPERATURE.
TEACHER_TEMPERATURE = 0.3

# MEASURED, not guessed. This budget must cover *reasoning + JSON*, because on gpt-oss
# the reasoning channel is generated against the same max_tokens and is not returned.
#
# 8192 was too small and failed in the worst possible way. Round 0 of the Stage 3a
# smoke pilot evolved fine (empty bank -> short prompt, short answer). Round 1 has to
# restate an 8-skill, 5-mistake bank, ran ~7k tokens, hit the cap mid-JSON, and
# surfaced only as "[Warning] Teacher response could not be parsed. Returning previous
# bank unchanged." The bank then froze for the rest of the run while the loop carried
# on reporting normally. At full scale -- 10 rounds, a bank that only grows -- every
# round after the first would fail this way.
#
# 32768 is chosen against the server's 65536 max_model_len: the evolution prompt runs
# a few thousand tokens, so this leaves the output room to be the binding constraint
# only if something has genuinely gone wrong. `__call__` now warns loudly on
# truncation, because "could not be parsed" points at the wrong culprit.
TEACHER_MAX_TOKENS = 32768

# Model names that mean "our local vLLM endpoint". Used only for auto-detection when
# the caller did not say `vllm:` explicitly; see `resolve_teacher_model`.
VLLM_MODEL_HINTS = ("gpt-oss",)

VLLM_PREFIX = "vllm:"


def teacher_base_url() -> Optional[str]:
    """The endpoint the teacher should use, or None if none is configured.

    Checked in the order most-specific-first so a run can point the teacher at a
    different server from the actor (useful if the evolver is ever moved to a
    smaller model) without having to override both.
    """
    for variable in (
        "HEXA_TEACHER_BASE_URL",
        "HEXA_VLLM_BASE_URL",
        "HEXA_VLLM_URL",
        "VLLM_BASE_URL",
    ):
        value = os.getenv(variable)
        if value:
            return value
    return None


def resolve_teacher_model(model: str) -> Optional[str]:
    """Return the served model name if `model` should go to vLLM, else None.

    Two ways to select the vLLM branch:

    * **Explicit** -- ``vllm:openai/gpt-oss-120b``. Always wins, needs no endpoint
      configured at resolve time, and is the form to use in scripts.
    * **Auto-detected** -- a bare name containing a `VLLM_MODEL_HINTS` entry *and* an
      endpoint configured.

    The auto-detection is not convenience, it is a guard. `call_teacher`'s final
    branch is ``_get_hf_teacher`` -> ``load_qwen_model``, so a bare
    ``openai/gpt-oss-120b`` that fell through would try to pull a 117B checkpoint
    into the *driver* process -- an OOM or a very long download, reported as a
    teacher error rather than as a misrouted model.
    """
    if model.startswith(VLLM_PREFIX):
        return model[len(VLLM_PREFIX):] or None
    if teacher_base_url() and any(hint in model for hint in VLLM_MODEL_HINTS):
        return model
    return None


class VLLMTeacher:
    """A callable teacher: prompt in, **final-channel** text out.

    Interface-compatible with what `call_teacher` needs (``str -> str``) and with the
    HF branch's ``generate(messages)`` shape via `__call__`.
    """

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        *,
        temperature: float = TEACHER_TEMPERATURE,
        max_tokens: int = TEACHER_MAX_TOKENS,
        seed: int | None = None,
        reasoning_effort: str | None = "high",
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        # A teacher call is not a scored rollout, so it is deliberately NOT charged
        # to RolloutBudget here; the loop decides that (PHASE3_DESIGN.md §2b/§5c
        # distinguish baseline / admission / train-solve charging).
        self.client = VLLMClient(
            model=model,
            base_url=base_url or teacher_base_url(),
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            seed=seed,
        )
        # Diagnostics for the run log. `truncated` is the one that matters: a cut-off
        # response usually means unparseable JSON, which downstream turns into an
        # empty skill list rather than an error.
        self.calls = 0
        self.truncated = 0
        self.empty_final = 0
        self.last_reasoning: str = ""
        self.last_finish_reason: Optional[str] = None

    def __call__(self, prompt: Any) -> str:
        """Return the final channel only. See the module docstring for why.

        Accepts either a plain prompt string or the HF branch's
        ``[{"role": ..., "content": ...}]`` message list, so this can stand in for
        `_get_hf_teacher`'s return value unchanged.
        """
        text = _as_prompt_text(prompt)
        generation = self.client.generate_structured(
            prompt=text,
            temperature=self.temperature,
            max_new_tokens=self.max_tokens,
            seed=self.seed,
        )
        self.calls += 1
        if generation is None:
            return ""

        self.last_reasoning = generation.reasoning_text
        self.last_finish_reason = generation.finish_reason
        if generation.finish_reason == "length":
            self.truncated += 1
            # Say so here. Downstream this appears as `_parse_json_array` returning []
            # and the evolver reporting "could not be parsed", which reads like a
            # malformed model response rather than a budget we set too low.
            print(
                f"  [Teacher TRUNCATED] hit max_tokens={self.max_tokens} on call "
                f"{self.calls}; the JSON is cut off and will not parse. The bank will "
                f"be returned UNCHANGED. Raise TEACHER_MAX_TOKENS."
            )
        final = (generation.final_text or "").strip()
        if not final:
            # Reasoning present but no final channel: the model spent the whole
            # budget thinking. Distinct from a failed call, and worth counting --
            # it is the signature of too small a max_tokens, and it presents
            # downstream as "the teacher proposed no skills".
            self.empty_final += 1
        return final

    def stats(self) -> dict[str, Any]:
        """Teacher-side counters for the run manifest."""
        return {
            "model": self.model,
            "base_url": self.client.base_url,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "calls": self.calls,
            "truncated": self.truncated,
            "empty_final": self.empty_final,
            "channel": "final_text",
            "reasoning_field": self.client.reasoning_field,
        }


def _as_prompt_text(prompt: Any) -> str:
    """Accept a string or an OpenAI-style message list."""
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, (list, tuple)):
        return "\n\n".join(
            str(part.get("content", "")) if isinstance(part, dict) else str(part)
            for part in prompt
        )
    return str(prompt)


_TEACHER_CACHE: dict[tuple[str, str], VLLMTeacher] = {}


def get_vllm_teacher(model: str, base_url: str | None = None) -> VLLMTeacher:
    """Cached per (model, endpoint), mirroring `distill._HF_TEACHER_CACHE`.

    Caching keeps the counters in `stats()` cumulative across a run, which is the
    point -- a per-call teacher would reset `truncated` and hide the problem it
    exists to surface.
    """
    url = base_url or teacher_base_url() or ""
    key = (model, url)
    if key not in _TEACHER_CACHE:
        _TEACHER_CACHE[key] = VLLMTeacher(model, base_url=base_url)
    return _TEACHER_CACHE[key]
