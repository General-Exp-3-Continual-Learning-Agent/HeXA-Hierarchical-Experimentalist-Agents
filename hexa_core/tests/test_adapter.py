"""Adapter fidelity: prompts, the guidelines slot, Problem identity, contamination.

Everything here defends one claim: :class:`hexa_core.adapter.Problem` is a *view*
of a CORE task item, never a reimplementation of one. Hard rule 3 says prompts must
be byte-identical to CORE's, so the central test compares every prompt we hand out
against a direct ``core_task.build_solve_prompt(problem, None)`` call on the same
bound instance.

Run with::

    PYTHONDONTWRITEBYTECODE=1 python -m hexa_core.tests.test_adapter

No network, no GPU.
"""

from __future__ import annotations

import copy
import random

from hexa_core.adapter import EvalContamination, Problem, assert_no_eval_contamination
from hexa_core.tasks import TASK_NAMES, TASK_SPECS
from hexa_core.tests import cached_task, check, check_eq, check_raises, core_module, fresh_task

#: Regime used wherever the test wants breadth rather than a specific regime:
#: 10 train + 100 eval per task = 330 problems, a few seconds of pure string work.
SAMPLE_REGIME = 10

#: First words of each CORE task's ``output_instruction``. Pinned so that an edit
#: on CORE's side shows up here as a diff rather than as a silent prompt change.
OUTPUT_INSTRUCTION_PREFIXES = {
    "matchstick": "Please think step by step. At the end of your response, write your final answer",
    "zebralogic": "Present your reasoning and solution in the requested paper-style JSON format",
    "tower-of-hanoi": "Ensure your final answer includes the complete list of moves in the format",
}

GUIDELINES = "SOME GUIDELINES"

#: A guidelines block with braces in it. CORE's templates are `str.format`-ed with
#: the guidelines as a *value*, so this is safe -- but a naive re-templating
#: implementation (which hard rule 3 forbids) would raise KeyError on `{peg}`.
BRACY_GUIDELINES = """- Track the state as {peg: [disks]} before every move.
- Prefer the move that does not undo the previous one.
- Format: {"answer": ...} is NOT what this task wants."""


def _all_problems(task_name: str, regime: int = SAMPLE_REGIME) -> list[Problem]:
    task = cached_task(task_name, regime)
    return task.train_problems() + task.eval_problems()


def test_prompt_is_byte_identical_to_core() -> None:
    """``Problem.prompt`` == a direct CORE call, for every problem in every task."""
    for name in TASK_NAMES:
        task = cached_task(name, SAMPLE_REGIME)
        build = task.core_task.build_solve_prompt
        for problem in _all_problems(name):
            expected = build(problem.problem_text, None)
            if problem.prompt != expected:
                check_eq(f"{name} {problem.id}: prompt == core build_solve_prompt(problem, None)",
                         problem.prompt, expected)
        check(f"{name}: all {len(_all_problems(name))} prompts byte-identical to CORE", True)


def test_build_prompt_none_matches_prompt() -> None:
    for name in TASK_NAMES:
        for problem in _all_problems(name):
            for label, produced in (
                ("build_prompt(None)", problem.build_prompt(None)),
                ("build_prompt()", problem.build_prompt()),
                # Every CORE template gates on truthiness (`if guidelines:`), so an
                # empty block means "no block". The runner relies on this.
                ('build_prompt("")', problem.build_prompt("")),
            ):
                if produced != problem.prompt:
                    check_eq(f"{name} {problem.id}: {label} == prompt", produced, problem.prompt)
        check(f"{name}: build_prompt(None) / () / ('') all reproduce .prompt", True)


def test_build_prompt_with_guidelines() -> None:
    for name in TASK_NAMES:
        for problem in _all_problems(name):
            augmented = problem.build_prompt(GUIDELINES)
            if augmented == problem.prompt or len(augmented) <= len(problem.prompt):
                check(f"{name} {problem.id}: guidelines change and lengthen the prompt", False,
                      f"len {len(augmented)} vs {len(problem.prompt)}")
            if GUIDELINES not in augmented:
                check(f"{name} {problem.id}: guidelines string appears in the prompt", False)
        check(f"{name}: guidelines slot is live for every problem", True)

        problem = _all_problems(name)[0]
        bracy = problem.build_prompt(BRACY_GUIDELINES)
        check(f"{name}: a guidelines block containing {{braces}} survives verbatim",
              BRACY_GUIDELINES in bracy)


def test_format_instructions_matches_core() -> None:
    """`Problem.format_instructions` is CORE's `output_instruction`, verbatim.

    MathGAP is the one task that has none: its answer-format line is part of each
    problem's text upstream, so `output_instruction` is the base class's `""`. That
    is declared on its `TaskSpec` and asserted here in the opposite direction, since
    a non-empty prefix check cannot be written for a string that must stay empty.
    """
    for name in TASK_NAMES:
        task = cached_task(name, SAMPLE_REGIME)
        expected = task.core_task.output_instruction
        mismatched = [p.id for p in _all_problems(name) if p.format_instructions != expected]
        check(f"{name}: format_instructions == core_task.output_instruction",
              not mismatched, f"{mismatched[:3]}")

        if TASK_SPECS[name].output_instruction_empty:
            check(f"{name}: CORE output_instruction is empty, as declared",
                  expected == "", f"got {expected[:90]!r}")
            # The format line has to be somewhere, or the actor is never told how to
            # answer and every attempt fails on parsing rather than on reasoning.
            missing = [p.id for p in _all_problems(name) if "final answer is" not in p.prompt]
            check(f"{name}: every prompt still carries its own answer-format line",
                  not missing, f"{len(missing)} without one, e.g. {missing[:3]}")
        else:
            check(f"{name}: CORE output_instruction is non-empty", bool(expected))
            check(f"{name}: output_instruction still starts with the pinned text",
                  expected.startswith(OUTPUT_INSTRUCTION_PREFIXES[name]),
                  f"got {expected[:90]!r}")


def test_problem_hashable_and_data_comparable() -> None:
    """Identity is the six string fields; metadata, tools and bound callables are not.

    ``frozen=True`` derives ``__hash__`` from every comparison field, so leaving
    ``metadata`` (a dict) or Hanoi's ``all_valid_outputs`` (a dict) comparable makes
    ``hash(problem)`` raise -- and Problems are used as dict keys for per-problem
    baselines ``b_q``, which would fail far away from the cause.
    """
    for name in TASK_NAMES:
        problem = cached_task(name, SAMPLE_REGIME).train_problems()[0]
        check(f"{name}: hash(problem) works", isinstance(hash(problem), int))
        check(f"{name}: usable as a dict key", {problem: 1}[problem] == 1)

        # Same item, independently constructed task: different bound instances,
        # different metadata dict object, same data -> must compare equal.
        twin = next(p for p in fresh_task(name, SAMPLE_REGIME).train_problems() if p.id == problem.id)
        check(f"{name}: twin is a distinct object", twin is not problem)
        check(f"{name}: twin from an independent construction compares equal", twin == problem)
        check(f"{name}: twin hashes equal", hash(twin) == hash(problem))
        check(f"{name}: bound callables really do differ",
              twin.build_prompt_fn is not problem.build_prompt_fn)

        other = cached_task(name, SAMPLE_REGIME).eval_problems()[0]
        check(f"{name}: a different item compares unequal", other != problem)

        problems = _all_problems(name)
        check_eq(f"{name}: all {len(problems)} problems are distinct in a set",
                 len(set(problems)), len(problems))


def test_zebralogic_prompt_stateless_under_interleaving() -> None:
    """The stateful-prompt regression test.

    ``ZebraLogicTask.build_solve_prompt`` reads ``self._problem_output_schema_json``
    and writes ``self._last_output_schema_json`` (zebralogic_task.py:437-438), so a
    prompt built out of order -- or on the wrong instance -- could carry another
    problem's output schema. Measured: all 100 eval problems have *distinct*
    schemas, so this test is fully discriminating.
    """
    zl = core_module("tasks.single_turn.zebralogic_task")
    task = cached_task("zebralogic", SAMPLE_REGIME)
    problems = task.eval_problems()

    schemas = {
        p.id: zl._build_blank_solution_schema_from_solution_json(p.all_valid_outputs[0])
        for p in problems
    }
    check_eq("zebralogic: eval output schemas are all distinct",
             len(set(schemas.values())), len(problems))

    snapshot = {p.id: p.build_prompt(None) for p in problems}

    order = list(problems) * 2
    random.Random(7).shuffle(order)
    for problem in order:
        rebuilt = problem.build_prompt(None)
        if rebuilt != snapshot[problem.id]:
            check_eq(f"zebralogic {problem.id}: prompt stable under interleaving",
                     rebuilt, snapshot[problem.id])
    check(f"zebralogic: {len(order)} interleaved builds all reproduced their own prompt", True)

    missing = [pid for pid, prompt in snapshot.items() if schemas[pid] not in prompt]
    check("zebralogic: every prompt embeds its own output schema", not missing, f"{missing[:3]}")
    placeholder = [pid for pid, prompt in snapshot.items() if zl.EXAMPLE_OUTPUT_SCHEMA_JSON in prompt]
    check("zebralogic: no prompt carries the EXAMPLE_OUTPUT_SCHEMA_JSON placeholder",
          not placeholder, f"{placeholder[:3]}")

    # Negative control: the failure this guards against is silent. A task instance
    # configured for a different size group has no entry for an xlarge problem and
    # falls through to `.get(problem, EXAMPLE_OUTPUT_SCHEMA_JSON)` -- emitting a
    # 3-house Name/Drink schema for a 6x5 puzzle, with no error or warning.
    wrong_instance = zl.ZebraLogicTask(num_train_items=2, num_eval_items=2, size_group="small")
    victim = problems[0]
    wrong_prompt = wrong_instance.build_solve_prompt(victim.problem_text, None)
    check("zebralogic negative control: a wrong-size_group instance yields a different prompt",
          wrong_prompt != snapshot[victim.id])
    check("zebralogic negative control: and it silently emits the placeholder schema",
          zl.EXAMPLE_OUTPUT_SCHEMA_JSON in wrong_prompt)


def test_assert_no_eval_contamination() -> None:
    tasks = [cached_task(name, SAMPLE_REGIME) for name in TASK_NAMES]

    for task in tasks:
        assert_no_eval_contamination(task, [p.id for p in task.train_problems()])
    check("clean: train ids of every task pass the contamination guard", True)
    assert_no_eval_contamination(tasks, [])
    assert_no_eval_contamination(tasks, [p.id for t in tasks for p in t.train_problems()])
    check("clean: empty attempt set and all train ids across tasks pass", True)

    task = tasks[0]
    leaked = task.eval_problems()[7].id
    exc = check_raises(
        "leaked eval id raises EvalContamination",
        EvalContamination,
        assert_no_eval_contamination,
        task,
        [p.id for p in task.train_problems()] + [leaked],
    )
    check("the raised message names the leaked id", leaked in str(exc), str(exc))

    # Multi-task iteration must not short-circuit: the leak belongs to the last task.
    tail_leak = tasks[-1].eval_problems()[0].id
    exc = check_raises(
        "a leak in the last of several tasks is still caught",
        EvalContamination,
        assert_no_eval_contamination,
        tasks,
        {tail_leak},
    )
    check("and that message names the leaked id too", tail_leak in str(exc), str(exc))


def test_verify_delegates_to_bound_instance() -> None:
    """A Problem detached from its task must still work.

    The bound-method fields keep the originating CORE instance alive, which is what
    lets a Problem be passed around (and shallow-copied into a log record) without
    carrying the task object explicitly.
    """
    for name in TASK_NAMES:
        problem = copy.copy(cached_task(name, SAMPLE_REGIME).eval_problems()[0])
        reference = cached_task(name, SAMPLE_REGIME).core_task.format_reference_answer_for_prompt(
            problem.all_valid_outputs
        )
        check(f"{name}: detached copy still builds its prompt",
              problem.build_prompt(None) == problem.prompt)
        check(f"{name}: detached copy still verifies CORE's reference answer",
              problem.verify(str(reference)) == 1)
        check(f"{name}: build_prompt_fn is bound to a CORE task instance",
              getattr(problem.build_prompt_fn, "__self__", None) is not None)


if __name__ == "__main__":
    import sys

    from hexa_core.tests import run_module

    sys.exit(run_module(sys.modules[__name__]))
