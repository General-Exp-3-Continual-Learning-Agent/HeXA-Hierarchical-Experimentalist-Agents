"""Verifier fidelity: CORE's own reference answers must score 1 under our wrapper.

We do not reimplement CORE's acceptance rule (hard rule 2), so the strongest check
available is that CORE's own reference answer for an item scores 1 when routed
through :meth:`hexa_core.adapter.Problem.verify`, on *every* eval problem -- and
that mismatched answers score 0, which is what makes the first claim non-trivial.

If a reference answer ever fails to verify, that is a fact about CORE's data and
must be reported, not softened: the failing ids are collected and printed in full
rather than aborting on the first one.

Run with::

    PYTHONDONTWRITEBYTECODE=1 python -m hexa_core.tests.test_verifiers

No network, no GPU.
"""

from __future__ import annotations

import json

from hexa_core.adapter import Problem
from hexa_core.tasks import TASK_NAMES
from hexa_core.tests import cached_task, check, check_eq

SAMPLE_REGIME = 10

#: Answer text that must never be accepted by any task's verifier.
GARBAGE = (
    "",
    "   ",
    "\n\n",
    "I have no idea.",
    "moves = []",
    "Correct version of the equation: BANANA",
    '{"solution": {"House 1": {"Name": "X"}}}',
    "So, this is my final answer: nothing",
)

#: Offsets used by the cross-problem negative control. Every (ref_i, target_j) pair
#: with j = (i + off) % 100 must score 0; without this, a verifier that always
#: returned 1 would pass the reference-answer test.
MISMATCH_OFFSETS = (1, 17, 50)


def _reference(problem: Problem, task_name: str, regime: int = SAMPLE_REGIME) -> str:
    core_task = cached_task(task_name, regime).core_task
    reference = core_task.format_reference_answer_for_prompt(problem.all_valid_outputs)
    if reference is None:
        raise AssertionError(f"{task_name} {problem.id}: CORE returned no reference answer")
    return str(reference)


def test_core_reference_answer_scores_one_on_eval() -> None:
    for name in TASK_NAMES:
        problems = cached_task(name, SAMPLE_REGIME).eval_problems()
        missing: list[str] = []
        failures: list[str] = []
        for problem in problems:
            core_task = cached_task(name, SAMPLE_REGIME).core_task
            reference = core_task.format_reference_answer_for_prompt(problem.all_valid_outputs)
            if reference is None:
                missing.append(problem.id)
                continue
            if problem.verify(str(reference)) != 1:
                failures.append(problem.id)
        check(f"{name}: CORE produced a reference answer for all {len(problems)} eval items",
              not missing, f"{len(missing)} missing: {missing[:10]}")
        check(f"{name}: CORE's reference answer scores 1 on all {len(problems)} eval items",
              not failures,
              f"{len(failures)} of {len(problems)} FAILED TO VERIFY -- this is a finding "
              f"about CORE's data, not a test to weaken: {failures[:20]}")


def test_core_reference_answer_scores_one_on_train() -> None:
    """The train split too: this is the reference-answer source for distillation."""
    for name in TASK_NAMES:
        problems = cached_task(name, SAMPLE_REGIME).train_problems()
        failures = [p.id for p in problems if p.verify(_reference(p, name)) != 1]
        check(f"{name}: CORE's reference answer scores 1 on all {len(problems)} train items",
              not failures, f"{len(failures)} failed: {failures[:20]}")


def test_garbage_scores_zero() -> None:
    for name in TASK_NAMES:
        problems = cached_task(name, SAMPLE_REGIME).eval_problems()[:5]
        accepted = [
            (p.id, text)
            for p in problems
            for text in GARBAGE
            if p.verify(text) != 0
        ]
        check(f"{name}: all {len(GARBAGE)} garbage answers score 0", not accepted, f"{accepted[:3]}")


def test_scores_are_ints_not_bools() -> None:
    """``verify`` is typed ``-> int``. A stray ``bool`` sums silently and breaks
    strict JSON consumers."""
    for name in TASK_NAMES:
        problem = cached_task(name, SAMPLE_REGIME).eval_problems()[0]
        for label, text in (("reference", _reference(problem, name)), ("garbage", "nope")):
            score = problem.verify(text)
            check(f"{name}: verify({label}) is exactly int, not bool",
                  type(score) is int and score in (0, 1), f"{type(score).__name__} {score!r}")


def test_verify_equals_verify_parsed_of_parse_answer() -> None:
    """``verify`` must be exactly parse-then-evaluate, including on CoT-wrapped text.

    CORE scores ``reasoning_text + final_text`` concatenated, so the shape that
    matters is prose followed by an answer.
    """
    for name in TASK_NAMES:
        for problem in cached_task(name, SAMPLE_REGIME).eval_problems()[:5]:
            reference = _reference(problem, name)
            cases = (
                reference,
                "",
                "I have no idea.",
                f"Let me work through this step by step, constraint by constraint.\n\n{reference}",
            )
            for text in cases:
                if problem.verify(text) != problem.verify_parsed(problem.parse_answer(text)):
                    check(f"{name} {problem.id}: verify == verify_parsed(parse_answer(x))", False,
                          f"on {text[:60]!r}")
        check(f"{name}: verify == verify_parsed(parse_answer(x)) on all sampled cases", True)


def test_cot_wrapped_reference_still_scores_one() -> None:
    """Prose before the answer must not change the verdict."""
    prose = (
        "Let me reason about this carefully. I will consider each possibility in turn, "
        "rule out the ones that break a constraint, and then state the answer.\n\n"
    )
    for name in TASK_NAMES:
        problems = cached_task(name, SAMPLE_REGIME).eval_problems()[:20]
        failures = [p.id for p in problems if p.verify(prose + _reference(p, name)) != 1]
        check(f"{name}: reference answer preceded by chain-of-thought still scores 1",
              not failures, f"{failures[:5]}")


def test_zebralogic_stored_solution_json_scores_one() -> None:
    """The stored ``solution_json`` path, independent of the reference formatter."""
    problems = cached_task("zebralogic", SAMPLE_REGIME).eval_problems()
    failures = [p.id for p in problems if p.verify(p.all_valid_outputs[0]) != 1]
    check(f"zebralogic: stored solution_json scores 1 on all {len(problems)} eval items",
          not failures, f"{failures[:10]}")

    prefixed = [
        p.id
        for p in problems
        if p.verify(f"So, this is my final answer: {p.all_valid_outputs[0]}") != 1
    ]
    check("zebralogic: solution_json behind CORE's FINAL_ANSWER_PREFIX also scores 1",
          not prefixed, f"{prefixed[:10]}")


#: Tasks whose answer space is small enough that two different problems can share
#: an answer. MathGAP's answers are integers and every one in this split lies in
#: 10..30, so a wrong-but-in-range guess is scored correct with probability equal to
#: the target distribution's collision rate -- measured at 0.061 on the 100 eval
#: items (19 distinct answers, modal answer 12 at 11%). This is a property of the
#: benchmark, not of our adapter: it puts a **non-zero guess floor of roughly 5-6%
#: under MathGAP accuracy**, which matters when reading a small HeXA-vs-baseline
#: delta on this task. The other three have effectively unbounded answer spaces
#: (equations, grids, move lists) and must show no collisions at all.
_COLLIDING_ANSWER_SPACE = frozenset({"mathgap"})


def test_verifier_is_discriminating() -> None:
    """Cross-problem negative control: another item's reference answer scores 0.

    A cross-problem reference that scores 1 is only excusable when the two problems
    *genuinely share an answer* -- no verifier can separate those, and pretending
    otherwise would mean asserting something false about MathGAP. So collisions are
    partitioned: a scored-1 pair whose parsed answers are equal is counted as a
    shared answer, and anything else is a real false positive and fails. That keeps
    the assertion exact for all four tasks instead of relaxing it to a tolerance.
    """
    for name in TASK_NAMES:
        problems = cached_task(name, SAMPLE_REGIME).eval_problems()
        references = [_reference(p, name) for p in problems]
        false_positives: list[tuple[str, str]] = []
        shared_answer: list[tuple[str, str]] = []
        pairs = 0
        for offset in MISMATCH_OFFSETS:
            for i, reference in enumerate(references):
                target = problems[(i + offset) % len(problems)]
                if target.id == problems[i].id:
                    continue
                pairs += 1
                if target.verify(reference) == 0:
                    continue
                own = target.parse_answer(_reference(target, name))
                if target.parse_answer(reference) == own:
                    shared_answer.append((problems[i].id, target.id))
                else:
                    false_positives.append((problems[i].id, target.id))

        check(
            f"{name}: {pairs} mismatched (ref_i, target_j) pairs score 0 unless the "
            f"two problems share an answer",
            not false_positives,
            f"{len(false_positives)} unexplained false positives, e.g. {false_positives[:5]}",
        )
        if name in _COLLIDING_ANSWER_SPACE:
            check(
                f"{name}: shares answers across problems as expected "
                f"({len(shared_answer)}/{pairs} = {len(shared_answer) / pairs:.3f} guess floor)",
                bool(shared_answer),
                "no collisions at all -- the small-answer-space note is now stale",
            )
        else:
            check(
                f"{name}: no two problems share an answer ({pairs} pairs)",
                not shared_answer,
                f"{len(shared_answer)} shared, e.g. {shared_answer[:5]}",
            )


def test_hanoi_last_answer_in_the_text_wins() -> None:
    """Hanoi grades the LAST parseable move list, so ordering inside the text matters.

    This is the parity detail behind ``raw_text = reasoning + final``: a discarded
    candidate earlier in the chain of thought must not win, and must not be ignored
    if it comes last.
    """
    task = cached_task("tower-of-hanoi", SAMPLE_REGIME)
    for problem in task.eval_problems()[:10]:
        reference = _reference(problem, "tower-of-hanoi")
        wrong = "moves = [[1, 0, 1]]"
        if problem.verify(wrong) != 0:
            check(f"hanoi {problem.id}: the decoy move list is genuinely wrong", False)
        if problem.verify(f"First attempt: {wrong}\nOn reflection: {reference}") != 1:
            check(f"hanoi {problem.id}: decoy first, reference last -> 1", False)
        if problem.verify(f"{reference}\nActually, scrap that: {wrong}") != 0:
            check(f"hanoi {problem.id}: reference first, decoy last -> 0", False)
    check("hanoi: the last parseable move list in the text is the one graded", True)


def test_hanoi_suboptimal_but_legal_solution_scores_one() -> None:
    """``enforce_optimal=False`` is the paper config: a legal detour must still score 1.

    Guards against a future default flip silently deflating the Hanoi number.
    """
    task = cached_task("tower-of-hanoi", SAMPLE_REGIME)
    check_eq("hanoi: enforce_optimal is False (paper config)", task.core_task.enforce_optimal, False)

    for problem in task.eval_problems()[:10]:
        moves = json.loads(problem.parse_answer(_reference(problem, "tower-of-hanoi")))
        optimal_moves = int(problem.all_valid_outputs["optimal_moves"])
        if len(moves) != optimal_moves:
            check(f"hanoi {problem.id}: reference solution is optimal-length", False,
                  f"{len(moves)} moves vs optimal_moves={optimal_moves}")

        # Disk 1 is the smallest, so moving it off its goal peg and back is always
        # legal and lands on the goal state again -- two moves longer than optimal.
        peg = int(problem.all_valid_outputs["goal_state"][0])
        detour = moves + [[1, peg, (peg + 1) % 3], [1, (peg + 1) % 3, peg]]
        text = "moves = " + json.dumps(detour)
        if problem.verify(text) != 1:
            check(f"hanoi {problem.id}: legal detour ({len(detour)} > {optimal_moves} moves) scores 1",
                  False)
    check("hanoi: suboptimal-but-legal solutions score 1 under enforce_optimal=False", True)


if __name__ == "__main__":
    import sys

    from hexa_core.tests import run_module

    sys.exit(run_module(sys.modules[__name__]))
