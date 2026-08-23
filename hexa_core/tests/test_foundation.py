"""Foundation checks: task construction, split invariants, prompt parity, budget, tokens.

Run with::

    PYTHONDONTWRITEBYTECODE=1 python -m hexa_core.tests.test_foundation

pytest is not installed in this environment, so this is a plain script: bare
``assert``, printed progress, non-zero exit on the first failure. It imports CORE
and reads CORE's data files, but makes no network calls and needs no GPU.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from hexa_core.adapter import (
    CoreTask,
    EvalContamination,
    Problem,
    Task,
    assert_no_eval_contamination,
    eval_ids_digest,
)
from hexa_core.budget import BudgetExhausted, RolloutBudget
from hexa_core.tasks import REGIMES, TASK_NAMES, TASK_SPECS, make_task
from hexa_core.tokens import added_tokens, count_tokens, counter_name


_LAZY_PROBE = """
import sys
import hexa_core
assert "hexa_core.adapter" not in sys.modules, "hexa_core eagerly loaded adapter"
assert "hexa_core.tasks" not in sys.modules, "hexa_core eagerly loaded tasks"
before = list(sys.path)
assert hexa_core.make_task.__name__ == "make_task"
assert hexa_core.make_task is hexa_core.tasks.make_task, "lazy re-export is not the real object"
assert sys.path == before or any("core-reasoning" in p for p in sys.path)
assert hexa_core.Problem.__module__ == "hexa_core.adapter"
try:
    hexa_core.nope
except AttributeError:
    pass
else:
    raise AssertionError("__getattr__ did not raise for an unknown name")
"""


def check_lazy_package_import() -> None:
    """`import hexa_core` must not drag CORE onto sys.path or load data files.

    Run in a fresh interpreter: this test module imports the submodules at the top,
    so in-process the check could only ever be faked.
    """
    proc = subprocess.run(
        [sys.executable, "-c", _LAZY_PROBE],
        cwd=str(Path(__file__).resolve().parents[2]),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"lazy-import probe failed:\n{proc.stderr}"
    print("[ok] hexa_core imports lazily; adapter/tasks not loaded at package import")


def build_all() -> dict[tuple[str, int], CoreTask]:
    tasks: dict[tuple[str, int], CoreTask] = {}
    for name in TASK_NAMES:
        for n in REGIMES:
            task = make_task(name, n)
            assert isinstance(task, Task), f"{name}/{n} does not satisfy the Task protocol"
            tasks[(name, n)] = task
            m = task.manifest()
            print(
                f"[ok] {name:<15} n={n:<3} core={m['core_task_name']!r:<24} "
                f"train={m['num_train_items']:<3} eval={m['num_eval_items']}"
            )
    return tasks


def check_eval_identical_across_regimes(tasks: dict[tuple[str, int], CoreTask]) -> None:
    """Eval sets are regime-invariant wherever `TaskSpec` declares them to be.

    MathGAP declares otherwise -- `train = data[:n]`, `eval = data[n : n + 100]` out
    of one shuffled pool -- so its window genuinely slides. Both directions are
    asserted: declared-invariant tasks must not move, and MathGAP must. See
    `tests/test_splits.py::test_mathgap_eval_window_slides`.
    """
    for name in TASK_NAMES:
        blobs = set()
        digests = set()
        for n in REGIMES:
            ids = [p.id for p in tasks[(name, n)].eval_problems()]
            blobs.add("\n".join(ids).encode("utf-8"))
            digests.add(tasks[(name, n)].manifest()["eval_ids_sha256"])
        if TASK_SPECS[name].eval_split_regime_invariant:
            assert len(blobs) == 1, f"{name}: eval id lists differ across regimes {REGIMES}"
            assert len(digests) == 1, f"{name}: eval_ids_sha256 differs across regimes"
            print(
                f"[ok] {name:<15} eval ids byte-identical across {REGIMES}"
                f"  sha={digests.pop()[:12]}"
            )
        else:
            assert len(blobs) == len(REGIMES), (
                f"{name}: declared regime-dependent but its eval id list did not move "
                f"across {REGIMES} -- the declaration is now wrong"
            )
            assert len(digests) == len(REGIMES), f"{name}: eval_ids_sha256 collided"
            print(f"[ok] {name:<15} eval window slides with the regime, as declared")


def check_no_overlap(tasks: dict[tuple[str, int], CoreTask]) -> None:
    for (name, n), task in tasks.items():
        train = {p.id for p in task.train_problems()}
        evl = {p.id for p in task.eval_problems()}
        assert not (train & evl), f"{name}/{n}: train/eval overlap {sorted(train & evl)[:5]}"
        assert len(train) == n, f"{name}/{n}: got {len(train)} train ids"
    print(f"[ok] train/eval disjoint by id for all {len(tasks)} task x regime combinations")


#: Tasks whose prompt template interpolates `output_instruction` literally.
#: ZebraLogic is deliberately absent: its `output_instruction` is a prose summary
#: ("... top-level keys \"reasoning\" and \"solution\"") while the prompt embeds a
#: per-problem JSON schema instead, so the string is not a substring of the prompt.
#: MathGAP is absent for the opposite reason: it has no `output_instruction` at all
#: (`TaskSpec.output_instruction_empty`), so the substring assertion below would
#: pass *vacuously* on the empty string. It is checked for emptiness instead.
_EMBEDS_OUTPUT_INSTRUCTION = frozenset({"matchstick", "tower-of-hanoi"})


def check_prompt_parity(tasks: dict[tuple[str, int], CoreTask]) -> None:
    checked = 0
    for (name, n), task in tasks.items():
        core_instruction = task.core_task.output_instruction
        embeds = name in _EMBEDS_OUTPUT_INSTRUCTION
        if TASK_SPECS[name].output_instruction_empty:
            assert core_instruction == "", (
                f"{name}/{n}: expected an empty output_instruction but CORE now "
                f"supplies {core_instruction!r} -- the answer-format line is no "
                f"longer only in the problem text, so prompt parity must be rechecked"
            )
        else:
            assert core_instruction, f"{name}/{n}: unexpectedly empty output_instruction"
        for problem in task.train_problems() + task.eval_problems():
            assert problem.prompt == problem.build_prompt(None), (
                f"{name}/{n} {problem.id}: prompt != build_prompt(None)"
            )
            assert problem.build_prompt("GUIDELINES") != problem.prompt, (
                f"{name}/{n} {problem.id}: guidelines slot is dead"
            )
            assert problem.format_instructions == core_instruction
            if embeds:
                assert problem.format_instructions in problem.prompt, (
                    f"{name}/{n} {problem.id}: output_instruction missing from prompt"
                )
            assert problem.task_name == name
            checked += 1
    print(f"[ok] prompt == build_prompt(None) and guidelines slot live for {checked} problems")


def check_contamination_guard(tasks: dict[tuple[str, int], CoreTask]) -> None:
    task = tasks[("matchstick", 10)]
    assert_no_eval_contamination(task, [p.id for p in task.train_problems()])
    assert_no_eval_contamination(list(tasks.values()), [])

    leaked = task.eval_problems()[3].id
    try:
        assert_no_eval_contamination(task, [leaked])
    except EvalContamination as exc:
        assert leaked in str(exc)
    else:
        raise AssertionError("assert_no_eval_contamination did not flag a leaked eval id")

    try:
        assert_no_eval_contamination(list(tasks.values()), {leaked})
    except EvalContamination:
        pass
    else:
        raise AssertionError("contamination guard failed over an iterable of tasks")
    print("[ok] assert_no_eval_contamination passes on train ids, raises on a leaked eval id")


def check_problem_identity(tasks: dict[tuple[str, int], CoreTask]) -> None:
    problem = tasks[("zebralogic", 5)].train_problems()[0]
    other_task = tasks[("zebralogic", 10)]
    assert {problem: 0.5}[problem] == 0.5, "Problem is not usable as a dict key"

    twin = Problem(
        id=problem.id,
        prompt=problem.prompt,
        problem_text=problem.problem_text,
        format_instructions=problem.format_instructions,
        task_name=problem.task_name,
        split=problem.split,
        metadata={"different": True},
        tools=None,
        build_prompt_fn=other_task.core_task.build_solve_prompt,
        parse_answer_fn=other_task.core_task.parse_answer,
        evaluate_fn=other_task.core_task.evaluate,
        all_valid_outputs=problem.all_valid_outputs,
    )
    assert twin == problem, "equality should ignore metadata and the bound callables"
    assert hash(twin) == hash(problem)
    print("[ok] Problem is hashable; equality ignores metadata/tools/bound callables")


def check_per_instance_binding(tasks: dict[tuple[str, int], CoreTask]) -> None:
    """A problem's prompt builder must be the instance the problem came from."""
    problem = tasks[("zebralogic", 5)].train_problems()[0]
    assert problem.build_prompt_fn.__self__ is tasks[("zebralogic", 5)].core_task
    print("[ok] Problem.build_prompt_fn is bound to its originating CORE task instance")


def check_verifier_roundtrip(tasks: dict[tuple[str, int], CoreTask]) -> None:
    cases: list[tuple[str, str]] = []

    matchstick = tasks[("matchstick", 5)].train_problems()[0]
    cases.append(
        (
            "matchstick",
            f"reasoning noise ...\nCorrect version of the equation: {matchstick.all_valid_outputs[0]}",
        )
    )

    zebra = tasks[("zebralogic", 5)].train_problems()[0]
    cases.append(("zebralogic", f"reasoning noise ...\nSo, this is my final answer: {zebra.all_valid_outputs[0]}"))

    hanoi_task = tasks[("tower-of-hanoi", 5)]
    hanoi = hanoi_task.train_problems()[0]
    reference = hanoi_task.core_task.format_reference_answer_for_prompt(hanoi.all_valid_outputs)
    assert reference, "Hanoi reference answer formatter returned nothing"
    cases.append(("tower-of-hanoi", f"reasoning noise ...\n{reference}"))

    mathgap = tasks[("mathgap", 5)].train_problems()[0]
    target = mathgap.all_valid_outputs[0]
    cases.append(
        ("mathgap", f"reasoning noise ...\nThe final answer is $\\boxed{{{target}}}$.")
    )

    problems = {
        "matchstick": matchstick,
        "mathgap": mathgap,
        "zebralogic": zebra,
        "tower-of-hanoi": hanoi,
    }
    for name, raw_text in cases:
        problem = problems[name]
        assert problem.verify(raw_text) == 1, f"{name}: known-correct answer scored 0"
        assert problem.verify("no answer here at all") == 0, f"{name}: garbage scored non-zero"

    # MathGAP's parser keeps whichever recognised pattern begins *latest* in the
    # text, not the first, and falls back to scanning lines bottom-up for a bare
    # integer when no pattern matches. Both behaviours decide which candidate gets
    # graded when a chain of thought contains several numbers, so they are pinned
    # rather than left to the generic round-trip above.
    wrong = str(int(target) + 1)
    boxed = "The final answer is $\\boxed{%s}$."
    assert mathgap.verify(f"{boxed % wrong}\nOn reflection, {boxed % target}") == 1, (
        "mathgap: a later corrected answer must win over an earlier wrong one"
    )
    assert mathgap.verify(f"{boxed % target}\nOn reflection, {boxed % wrong}") == 0, (
        "mathgap: the last answer must win even when an earlier one was right"
    )
    assert mathgap.parse_answer(f"scratch work {wrong}\n{target}") == target, (
        "mathgap: bare-integer fallback should read the last line, not the first number"
    )
    print(
        "[ok] verify() scores CORE reference answers 1 and garbage 0 on all four tasks; "
        "mathgap last-match-wins and bare-integer fallback pinned"
    )


def check_subsample_ordering(tasks: dict[tuple[str, int], CoreTask]) -> None:
    task = tasks[("matchstick", 100)]
    full = [p.id for p in task.train_problems()]
    sub = [p.id for p in task.train_problems(n=7, seed=1)]
    assert len(sub) == 7
    assert sub == [i for i in full if i in set(sub)], "sub-sample reordered the CORE order"
    assert sub == [p.id for p in task.train_problems(n=7, seed=1)], "sub-sample is not deterministic"
    try:
        task.train_problems(n=101)
    except ValueError:
        pass
    else:
        raise AssertionError("train_problems accepted n > len(train)")
    print("[ok] train_problems(n) is deterministic, keeps CORE order, rejects n > |train|")


def check_budget() -> None:
    b = RolloutBudget(max_train=2000)
    b.charge_train_solve(1900)
    b.charge_admission(100)
    assert b.training_rollouts == 2000 and b.train_remaining == 0
    assert not b.can_afford_train(1)

    # baseline is charged outside the gated budget, per run_core.py:2487-2493
    b.charge_baseline(10 * 5)
    assert b.training_rollouts == 2000, "baseline must not count toward max_train"

    try:
        b.charge_train_solve(1)
    except BudgetExhausted as exc:
        assert "max_train=2000" in str(exc)
    else:
        raise AssertionError("charge_train_solve did not raise past max_train")
    assert (b.train_solve, b.admission) == (1900, 100), "counters mutated despite the raise"

    report = b.report()
    assert report["training_rollouts"] == 2000
    assert report["all_in_total"] == 2050, report
    assert report["eval"] == 0

    b.charge_eval(100)
    assert b.report()["all_in_total"] == 2150
    capped = RolloutBudget(max_train=10, max_eval=5)
    capped.charge_eval(5)
    try:
        capped.charge_eval(1)
    except BudgetExhausted:
        pass
    else:
        raise AssertionError("max_eval was not enforced")
    print("[ok] RolloutBudget: max_train gates solve+admission only; all_in_total=2050 at N=5")


def check_tokens(tasks: dict[tuple[str, int], CoreTask]) -> None:
    name = counter_name()
    print(f"[ok] token counter in use: {name}")
    assert count_tokens("") == 0
    assert added_tokens("a long prompt with many words in it", "short") == 0, "not clamped at 0"

    problem = tasks[("tower-of-hanoi", 5)].eval_problems()[0]
    guidelines = "Always move the smallest disk that does not undo the previous move."
    delta = added_tokens(problem.prompt, problem.build_prompt(guidelines))
    assert delta > 0, "injecting guidelines did not increase the token count"
    print(f"[ok] added_tokens on a real CORE prompt with guidelines: +{delta} tokens")


def main() -> int:
    check_lazy_package_import()
    tasks = build_all()
    assert len(tasks) == len(TASK_NAMES) * len(REGIMES)
    check_eval_identical_across_regimes(tasks)
    check_no_overlap(tasks)
    check_prompt_parity(tasks)
    check_contamination_guard(tasks)
    check_problem_identity(tasks)
    check_per_instance_binding(tasks)
    check_verifier_roundtrip(tasks)
    check_subsample_ordering(tasks)
    check_budget()
    check_tokens(tasks)
    digest = eval_ids_digest(p.id for p in tasks[("matchstick", 5)].eval_problems())
    print(f"\nall foundation checks passed (matchstick eval sha256 {digest[:16]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
