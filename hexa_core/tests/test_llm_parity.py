"""Scored-string parity: ``raw_text == reasoning + final``, and why the order matters.

Run with::

    PYTHONDONTWRITEBYTECODE=1 python -m hexa_core.tests.test_llm_parity

Zero API calls, zero GPU, and -- asserted, not hoped -- zero ``openai`` import. A
fake OpenAI-shaped transport is injected in place of
:meth:`hexa_core.llm.vllm_client.VLLMClient._make_client`, reusing the fakes the
LLM suite already publishes (``stream_chunks``, ``non_stream_response``, ``wire``).

Three layers, each checking something the layer above cannot:

1. **Our client** composes ``raw_text = reasoning_content + content``, in that
   order, with no separator, identically in the streaming and non-streaming paths.
2. **CORE agrees.** The two lines that define the rule are located in
   ``core-reasoning/src/model/nvidia_client.py`` by regex (not by line number --
   those drift), and then *executed* over a table of edge-case pairs so the check
   is behavioural rather than a string comparison of source code.
3. **The consequence.** Tower-of-Hanoi's extractor returns ``candidates[-1]``
   (``tower_of_hanoi_task.py:349``), so which text is concatenated -- and in which
   order -- decides which candidate move list gets scored. Two fixtures, run
   against CORE's real verifier on a real eval item, show that the correct
   composition scores 1 where final-text-only or a reversed concatenation score 0.
"""

from __future__ import annotations

import re
import sys
import textwrap
from types import SimpleNamespace
from typing import Any

from hexa_core.adapter import Problem
from hexa_core.core_bridge import core_repo_root
from hexa_core.llm.base import Generation
from hexa_core.llm.vllm_client import VLLMClient
from hexa_core.tasks import make_task
from hexa_core.tests import check, check_eq
from hexa_core.tests.test_llm_clients import new_client, non_stream_response, stream_chunks, wire

#: The two lines in CORE that define what gets scored. Matched by shape rather than
#: at ``nvidia_client.py:143-144`` so a harmless edit above does not break the
#: check and a rewrite *of the rule* does.
_CORE_JOIN_RE = re.compile(
    r"^[ \t]*parts\s*=\s*\[part for part in \((\w+),\s*(\w+)\) if part\]\s*\n"
    r"[ \t]*raw_text\s*=\s*\"\"\.join\(parts\)\s*$",
    re.M,
)

#: Pairs chosen to separate the real rule from plausible near-misses.
#: ``(" ", "F")`` is the important one: CORE's ``if part`` filter drops ``""`` but
#: keeps whitespace, so CORE emits ``" F"``. An implementation that "tidies up"
#: with ``.strip()`` matches the source text and fails here.
_PARTS_MATRIX: tuple[tuple[str, str], ...] = (
    ("R", "F"),
    ("", "F"),
    ("R", ""),
    ("", ""),
    (" ", "F"),
    ("a", " "),
    ("reasoning ends without punctuation", "and the answer begins"),
)


def _bounded_client(**kwargs: Any) -> VLLMClient:
    """A client pointed at a closed port, so a bypassed fake fails fast.

    Without this, a regression that reached the real transport would sit on the
    1800-second default timeout instead of erroring.
    """
    kwargs.setdefault("base_url", "http://127.0.0.1:1/v1")
    kwargs.setdefault("max_retries", 1)
    return new_client(**kwargs)


def _generate(reasoning: str, final: str, *, stream: bool, **kwargs: Any) -> Generation:
    """One fake completion through the real client code path."""
    client = _bounded_client(stream=stream, **kwargs)
    response = (
        stream_chunks(reasoning, final) if stream else non_stream_response(reasoning, final)
    )
    wire(client, [response])
    generation = client.generate_structured(prompt="p")
    assert generation is not None, "fake transport produced no generation"
    return generation


def _chunk(
    reasoning: str | None = None,
    content: str | None = None,
    finish_reason: str | None = None,
) -> Any:
    """A single streaming chunk, for cases ``stream_chunks`` cannot express."""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(reasoning_content=reasoning, content=content),
                finish_reason=finish_reason,
            )
        ],
        usage=None,
    )


def test_no_openai_import() -> None:
    """The 'zero API calls' claim, checked structurally.

    ``openai`` is not installed here, so an eager import would be an ImportError
    rather than a silent network client -- but this suite is also meant to keep
    passing on a machine where it *is* installed.
    """
    assert "hexa_core.llm.vllm_client" in sys.modules
    check("importing the vLLM client does not import openai", "openai" not in sys.modules)


def test_non_streaming_composes_reasoning_then_final() -> None:
    reasoning, final = "<R>", "<F>"
    generation = _generate(reasoning, final, stream=False)

    check_eq("non-streaming raw_text is reasoning + final", generation.raw_text, "<R><F>")
    assert generation.raw_text == generation.reasoning_text + generation.final_text
    assert generation.reasoning_text == reasoning and generation.final_text == final
    check(
        "reasoning precedes final in raw_text",
        generation.raw_text.index("<R>") < generation.raw_text.index("<F>"),
    )
    # No separator: the character at the seam is the first character of the final
    # text, not a newline or a space.
    check(
        "no separator at the seam",
        generation.raw_text[len(generation.reasoning_text)] == final[0]
        and len(generation.raw_text) == len(reasoning) + len(final),
        repr(generation.raw_text),
    )


def test_streaming_composes_identically() -> None:
    """A stream-only reordering would be invisible in the non-streaming test.

    ``stream=True`` is CORE's default (`nvidia_client.py:76`) and therefore the
    configuration the paper numbers were produced under, so this is the path that
    actually has to be right.
    """
    reasoning = "Let me think. 8 - 1 = 7.\nCheck: yes.\n"
    final = "The answer is 7."

    streamed = _generate(reasoning, final, stream=True)
    direct = _generate(reasoning, final, stream=False)

    check_eq("streamed raw_text == non-streamed raw_text", streamed.raw_text, direct.raw_text)
    assert streamed.raw_text == reasoning + final
    assert streamed.reasoning_text == direct.reasoning_text == reasoning
    assert streamed.final_text == direct.final_text == final

    # And through the plain `generate()` entry point, which is what returns the
    # scored string in CORE's driver.
    client = _bounded_client(stream=True)
    wire(client, [stream_chunks(reasoning, final)])
    check_eq(
        "generate() returns the composed scored string",
        client.generate("p"),
        reasoning + final,
    )


def test_composition_matches_core_nvidia_client_exactly() -> None:
    """Cross-check ordering and join against CORE's own source, then execute it."""
    source = (core_repo_root() / "src" / "model" / "nvidia_client.py").read_text(encoding="utf-8")
    match = _CORE_JOIN_RE.search(source)
    assert match is not None, (
        "could not find CORE's `parts = [...]` / `raw_text = \"\".join(parts)` pair in "
        "nvidia_client.py. If the concatenation was rewritten upstream, every number "
        "this package produces has to be re-derived before anything is run."
    )

    # 1-2: reasoning comes first in CORE's own tuple.
    check_eq("CORE joins reasoning_text first", match.group(1), "reasoning_text")
    check_eq("CORE joins final_text second", match.group(2), "final_text")
    # 3: no separator argument to join.
    check("CORE's join takes no separator", '"".join(parts)' in match.group(0), match.group(0))

    # 4: behavioural equality, over pairs a source-text check cannot distinguish.
    snippet = textwrap.dedent(match.group(0))
    for reasoning, final in _PARTS_MATRIX:
        namespace: dict[str, Any] = {"reasoning_text": reasoning, "final_text": final}
        exec(snippet, namespace)  # noqa: S102 - two transcribed lines from CORE, no imports
        core_raw = namespace["raw_text"]

        assert core_raw == reasoning + final, (reasoning, final, core_raw)
        assert Generation.from_parts(reasoning, final).raw_text == core_raw

        for stream in (True, False):
            ours = _generate(reasoning, final, stream=stream)
            assert ours.raw_text == core_raw, (stream, reasoning, final, ours.raw_text)

    check(
        f"CORE's own join, executed over {len(_PARTS_MATRIX)} pairs, agrees with our "
        "client in both stream modes",
        True,
    )

    whitespace = dict(zip(("reasoning_text", "final_text"), (" ", "F")))
    exec(snippet, whitespace)  # noqa: S102
    check_eq(
        "whitespace-only reasoning survives CORE's `if part` filter",
        whitespace["raw_text"],
        " F",
    )


def test_finish_reason_is_surfaced() -> None:
    stopped = _generate("r", "f", stream=False)
    check(
        "finish_reason surfaces as the plain string 'stop'",
        stopped.finish_reason == "stop" and type(stopped.finish_reason) is str,
        repr(stopped.finish_reason),
    )
    assert stopped.truncated is False

    client = _bounded_client(stream=False)
    wire(client, [non_stream_response("r", "f", finish_reason=None)])
    absent = client.generate_structured(prompt="p")
    assert absent is not None
    check(
        "a missing finish_reason stays None rather than becoming ''",
        absent.finish_reason is None,
        repr(absent.finish_reason),
    )
    assert absent.truncated is False

    # Streaming: the value arrives on the last chunk only, and CORE keeps the last
    # non-None one it saw (`nvidia_client.py:120-122`). A stream that reports
    # "stop" mid-flight and "length" at the end is truncated.
    client = _bounded_client(stream=True)
    wire(
        client,
        [
            [
                _chunk(reasoning="think "),
                _chunk(content="answer", finish_reason="stop"),
                _chunk(content=" more", finish_reason=None),
                _chunk(finish_reason="length"),
            ]
        ],
    )
    generation = client.generate_structured(prompt="p")
    assert generation is not None
    assert generation.raw_text == "think answer more"
    check_eq(
        "streaming keeps the LAST non-None finish_reason",
        generation.finish_reason,
        "length",
    )
    assert generation.truncated is True


def test_truncation_is_detectable() -> None:
    """Hanoi n=6 runs to 63 moves; hitting 32768 must not look like a wrong answer."""
    client = _bounded_client(stream=True)
    usage = {"completion_tokens": 32768, "completion_tokens_details": {"reasoning_tokens": 32700}}
    wire(client, [stream_chunks("very long reasoning", "", finish_reason="length", usage=usage)])
    generation = client.generate_structured(prompt="63-move hanoi")
    assert generation is not None

    check(
        "finish_reason='length' is detectable as Generation.truncated",
        generation.truncated is True,
    )
    assert generation.raw_text == generation.reasoning_text == "very long reasoning"
    assert generation.final_text == ""
    # Reasoning present but no final answer is the realistic truncation shape: the
    # generation is *not* empty, so a consumer keying off emptiness would miss it.
    check(
        "a truncated generation with reasoning but no answer is not 'empty'",
        generation.is_empty is False,
    )

    record = generation.to_dict()
    # `logging_schema`'s primary_length_count diagnostic reads the raw label, so
    # to_dict must carry the string and not only the derived boolean.
    check(
        "to_dict carries both finish_reason='length' and truncated=True",
        record["truncated"] is True and record["finish_reason"] == "length",
        repr(record["finish_reason"]),
    )
    check_eq("the usage chunk (empty choices) survived", record["usage"], usage)
    assert client.stats["truncated"] == 1


# ---------------------------------------------------------- the consequence

#: A short, legal-looking but wrong move list. Illegal-as-a-solution, parseable as
#: a move list, which is exactly the shape a mid-reasoning false start has.
_WRONG_MOVES = "moves = [[1, 0, 2], [2, 0, 1], [1, 2, 1]]"


def _hanoi_problem() -> tuple[Problem, str]:
    task = make_task("tower-of-hanoi", 5)
    problem = task.eval_problems()[0]
    reference = task.core_task.format_reference_answer_for_prompt(problem.all_valid_outputs)
    assert reference and reference.startswith("moves ="), reference
    return problem, reference


def test_hanoi_last_list_consequence() -> None:
    """Two fixtures, because no single one can fail *both* counterfactuals.

    If the final text contains the correct list, then final-text-only also parses
    it correctly; if it contains no list at all, then reversing the concatenation
    still ends on the correct list. So the claim "the right composition scores 1
    while final-only and the wrong order do not" needs one fixture per clause.
    Both are real shapes: A is a model that false-starts and then answers, B is the
    observed n=6 near-truncation where the work is in the reasoning and the final
    turn only defers to it.

    The whole matrix is asserted against CORE's real verifier on a real eval item,
    because ``_extract_moves_from_text`` returns ``candidates[-1]``
    (`tower_of_hanoi_task.py:349`) -- position in the string, not correctness, is
    what selects the answer.
    """
    problem, reference = _hanoi_problem()
    assert problem.all_valid_outputs["optimal_moves"] == 63, problem.all_valid_outputs

    fixture_a = (
        "Start with the obvious plan.\n" + _WRONG_MOVES + "\nThat is wrong, let me redo it.\n",
        "Final answer:\n```python\n" + reference + "\n```\n",
    )
    fixture_b = (
        "First attempt: " + _WRONG_MOVES + "\nNo. Recomputing properly:\n" + reference + "\n",
        "As derived above, that sequence solves it.",
    )

    #                           raw = R+F, final-only, reversed F+R
    expected = {"A": (1, 1, 0), "B": (1, 0, 1)}

    for label, (reasoning, final) in (("A", fixture_a), ("B", fixture_b)):
        scores = (
            problem.verify(reasoning + final),
            problem.verify(final),
            problem.verify(final + reasoning),
        )
        check_eq(
            f"fixture {label}: (raw=R+F, final-only, reversed F+R) verifier scores",
            scores,
            expected[label],
        )

        # The correct composition must come out of the client, not out of `+` in
        # this test: the claim is "what our client returns verifies correctly".
        for stream in (True, False):
            generation = _generate(reasoning, final, stream=stream)
            assert generation.raw_text == reasoning + final
            assert problem.verify(generation.raw_text) == 1, (label, stream)
            assert problem.parse_answer(generation.raw_text) == problem.parse_answer(reference)
        check(
            f"fixture {label}: the client's own raw_text scores 1 in both stream modes",
            True,
        )

    # Why each counterfactual fails, so a future failure is diagnosable.
    reasoning_a, final_a = fixture_a
    check_eq(
        "fixture A reversed picks the false start (wrong candidate, not a parse failure)",
        problem.parse_answer(final_a + reasoning_a),
        problem.parse_answer(_WRONG_MOVES),
    )
    reasoning_b, final_b = fixture_b
    check_eq("fixture B's final text alone parses to nothing", problem.parse_answer(final_b), "")
    print(f"    .... verified against CORE's own verifier on {problem.id} (63 optimal moves)")


def test_still_no_openai_import() -> None:
    """Nothing above -- including building a CORE task -- may reach the SDK."""
    check("no check above imported openai", "openai" not in sys.modules)


if __name__ == "__main__":
    import sys

    from hexa_core.tests import run_module

    sys.exit(run_module(sys.modules[__name__]))
