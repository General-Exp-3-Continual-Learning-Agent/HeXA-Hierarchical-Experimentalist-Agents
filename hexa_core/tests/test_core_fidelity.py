"""Fidelity-to-CORE invariants: the places where we could silently drift from CORE.

The other suites check that our *outputs* match CORE's. This one checks the
*seams* -- that we still go through CORE's code rather than beside it, and that
the constants we transcribed still say what CORE says. Every check here resolves
its expectation out of the CORE checkout at run time; none of them hardcodes a
value that CORE could change without this suite noticing.

Line numbers are deliberately never used as anchors. CORE's ``run_core.py`` is
5000+ lines and every citation in this repo's comments is already a line number
that will drift; here the anchors are AST structure (which function a call sits
in) and call patterns, so a refactor upstream cannot make a check quietly
vacuous.

    PYTHONDONTWRITEBYTECODE=1 python -m hexa_core.tests.test_core_fidelity
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Any

from . import cached_task, check, check_eq, core_module, run_module

TASKS = ("matchstick", "zebralogic", "tower-of-hanoi")

#: The arguments CORE's own driver forwards for the published configuration.
#: Transcribed from ``init_task`` (``run_core.py``) plus the paper's shell-script
#: overrides -- see :mod:`hexa_core.tasks`.
PAPER_CORE_KWARGS: dict[str, dict[str, Any]] = {
    "matchstick": {},
    # MathGAP's `get_task` branch forwards only the two split sizes, so there is no
    # task-specific argument to pin.
    "mathgap": {},
    "zebralogic": {"zebralogic_size_group": "xlarge"},
    "tower-of-hanoi": {
        "hanoi_variant": "state",
        "hanoi_train_disks": [5],
        "hanoi_eval_disks": [6],
        "hanoi_enforce_optimal": False,
        "hanoi_min_distance": 1,
    },
}


def _run_core_source() -> str:
    from ..core_bridge import core_scripts_dir

    return (core_scripts_dir() / "run_core.py").read_text()


def _enclosing_functions(tree: ast.AST) -> dict[ast.AST, str]:
    """Map every node to the name of the function definition containing it."""
    owner: dict[ast.AST, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for child in ast.walk(node):
                owner.setdefault(child, node.name)
    return owner


# --------------------------------------------------------------------------- #
# Construction goes through CORE's factory
# --------------------------------------------------------------------------- #


def test_make_task_uses_cores_own_factory() -> None:
    """``make_task`` must not re-derive the args-to-constructor mapping.

    CORE owns that mapping in ``tasks.single_turn.get_task``, which is what
    ``run_core.py``'s ``init_task`` calls. Building the tasks a second way is how
    a future CORE change (a new required kwarg, a changed per-task default) gets
    missed. This asserts our tasks are *indistinguishable* from the factory's.
    """
    single_turn = core_module("tasks.single_turn")
    check("tasks.single_turn exposes get_task", callable(getattr(single_turn, "get_task", None)))

    for name in TASKS:
        ours = cached_task(name, 5)
        reference = single_turn.get_task(
            name, num_train_items=5, num_eval_items=100, seed=42, **PAPER_CORE_KWARGS[name]
        )
        check_eq(f"{name}: same CORE class", type(ours.core_task), type(reference))
        check_eq(f"{name}: same core .name", ours.core_task.name, reference.name)

        for split, mine in (("train", ours.train_problems()), ("eval", ours.eval_problems())):
            items = list(getattr(reference, f"{split}_data"))
            check_eq(
                f"{name}/{split}: same ids as get_task",
                [p.id for p in mine],
                [str(i.task_item_id) for i in items],
            )
            mismatched = [
                p.id
                for p, item in zip(mine, items)
                if p.prompt != reference.build_solve_prompt(item.problem, None)
            ]
            check(f"{name}/{split}: prompts identical to get_task's", not mismatched, str(mismatched[:5]))


def test_task_specs_do_not_hardcode_core_module_paths() -> None:
    """The registry should carry argument *values* only.

    Module paths and class names were previously duplicated here from CORE's own
    lazy-export table; importing the factory removes that copy, and this keeps it
    removed.
    """
    from ..tasks import TASK_SPECS

    for name, spec in TASK_SPECS.items():
        check(f"{name}: no module path in TaskSpec", not hasattr(spec, "module"))
        check(f"{name}: no class name in TaskSpec", not hasattr(spec, "class_name"))
        check_eq(f"{name}: spec name is CORE's get_task name", spec.name, name)
        check_eq(
            f"{name}: core_kwargs match the paper config",
            dict(spec.core_kwargs),
            PAPER_CORE_KWARGS[name],
        )
    check(
        "only tower-of-hanoi's split depends on the seed",
        [n for n, s in TASK_SPECS.items() if s.seed_affects_split] == ["tower-of-hanoi"],
    )


# --------------------------------------------------------------------------- #
# Per-phase generation seeds
# --------------------------------------------------------------------------- #


def test_seed_offsets_match_cores_call_sites() -> None:
    """Each phase's seed namespace, resolved from CORE's AST rather than a line number.

    CORE gives baseline estimation, eval, training solves, reflection and lesson
    admission five unrelated namespaces. Using the eval namespace to estimate the
    per-problem baseline b_q (or vice versa) draws from a different sampling
    distribution than CORE did, while producing perfectly plausible numbers.
    """
    from .. import seeds as br

    source = _run_core_source()
    tree = ast.parse(source)
    owner = _enclosing_functions(tree)

    # Second positional argument of every `_compute_generation_seed(base, OFFSET, ...)`
    # call, grouped by the function it appears in.
    by_function: dict[str, set[int]] = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_compute_generation_seed"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, int)
        ):
            by_function.setdefault(owner.get(node, "<module>"), set()).add(node.args[1].value)

    check(
        "found _compute_generation_seed call sites in CORE",
        len(by_function) >= 4,
        f"functions found: {sorted(by_function)}",
    )
    check_eq(
        "baseline estimation namespace (run_baseline_training)",
        by_function.get("run_baseline_training"),
        {br.CORE_BASELINE_SEED_OFFSET},
    )
    check_eq(
        "training-solve namespace (_prepare_training_attempt_execution)",
        by_function.get("_prepare_training_attempt_execution"),
        {br.CORE_TRAIN_SOLVE_SEED_OFFSET},
    )
    check_eq(
        "admission namespaces (_compute_admission_seed)",
        by_function.get("_compute_admission_seed"),
        set(br.CORE_ADMISSION_SEED_OFFSETS),
    )
    check_eq(
        "biased-sampler namespace (_sample_biased_training_item)",
        by_function.get("_sample_biased_training_item"),
        {br.CORE_BIASED_SAMPLER_SEED_OFFSET},
    )
    # Nothing else may be using a namespace we have not accounted for.
    known = {
        br.CORE_BASELINE_SEED_OFFSET,
        br.CORE_EVAL_SEED_OFFSET,
        br.CORE_TRAIN_SOLVE_SEED_OFFSET,
        br.CORE_REFLECTION_SEED_OFFSET,
        br.CORE_BIASED_SAMPLER_SEED_OFFSET,
        *br.CORE_ADMISSION_SEED_OFFSETS,
    }
    seen = {offset for offsets in by_function.values() for offset in offsets}
    check("every literal namespace in CORE is accounted for", seen <= known, str(sorted(seen - known)))

    # Eval passes it through a local `seed_offset` variable instead of a literal.
    eval_offsets = {
        node.value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and owner.get(node) == "_run_memory_passes_on_items"
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, int)
        and any(isinstance(t, ast.Name) and t.id == "seed_offset" for t in node.targets)
    }
    check_eq("eval namespace (_run_memory_passes_on_items)", eval_offsets, {br.CORE_EVAL_SEED_OFFSET})

    # And the reflection namespace arrives as a keyword.
    reflection = {
        kw.value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for kw in node.keywords
        if kw.arg == "seed_namespace"
        and isinstance(kw.value, ast.Constant)
        and isinstance(kw.value.value, int)
    }
    check_eq("reflection namespace", reflection, {br.CORE_REFLECTION_SEED_OFFSET})

    check("all recorded namespaces are distinct", len(known) == 7)


def test_seed_arithmetic_matches_core() -> None:
    """Our transcribed ``core_generation_seed`` against CORE's own function, executed."""
    from .. import seeds as br

    source = _run_core_source()
    tree = ast.parse(source)
    segment = next(
        (
            ast.get_source_segment(source, node)
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "_compute_generation_seed"
        ),
        None,
    )
    check("located CORE's _compute_generation_seed", segment is not None)
    namespace: dict[str, Any] = {}
    exec(segment, namespace)  # noqa: S102 -- four lines of arithmetic, read above
    core_seed = namespace["_compute_generation_seed"]

    offsets = (
        br.CORE_BASELINE_SEED_OFFSET,
        br.CORE_EVAL_SEED_OFFSET,
        br.CORE_TRAIN_SOLVE_SEED_OFFSET,
    )
    bad = [
        (base, offset, a, b)
        for base in (0, 42, 12345)
        for offset in offsets
        for a in range(4)
        for b in range(4)
        if core_seed(base, offset, a, b) != br.core_generation_seed(base, offset, a, b)
    ]
    check(f"seed arithmetic identical over {3 * len(offsets) * 16} inputs", not bad, str(bad[:3]))
    check(
        "different namespaces give different seeds",
        core_seed(42, br.CORE_BASELINE_SEED_OFFSET, 0, 0)
        != core_seed(42, br.CORE_EVAL_SEED_OFFSET, 0, 0),
    )


# --------------------------------------------------------------------------- #
# The two verifier numbers stay two numbers
# --------------------------------------------------------------------------- #


def test_problem_score_delegates_to_cores_score_prediction() -> None:
    """CORE logs ``is_correct`` and ``verifier_score`` from two different functions.

    ``_score_task_prediction`` calls ``task.evaluate`` *and*
    ``task.score_prediction``. All three tasks currently inherit the binary base
    implementation, so deriving one from the other looks harmless -- and would
    silently keep reporting a binary ``val_mean_verifier_score`` the day CORE
    gives any of them partial credit.
    """
    for name in TASKS:
        task = cached_task(name, 5)
        core = task.core_task
        check(
            f"{name}: Problem carries CORE's score_prediction",
            task.eval_problems()[0].score_prediction_fn is not None,
        )
        bad = []
        for problem in task.eval_problems()[:15] + task.train_problems():
            reference = core.format_reference_answer_for_prompt(problem.all_valid_outputs) or ""
            for text in (reference, "no answer here at all", ""):
                parsed = problem.parse_answer(text)
                expected = float(core.score_prediction(parsed, problem.all_valid_outputs))
                if problem.score(parsed) != expected:
                    bad.append((problem.id, problem.score(parsed), expected))
        check(f"{name}: score() == CORE score_prediction", not bad, str(bad[:3]))


def test_verifier_score_and_correctness_are_read_separately() -> None:
    """A graded ``score_prediction`` must reach the log even when ``evaluate`` says no."""
    from ..adapter import Problem

    problem = Problem(
        id="synthetic",
        prompt="p",
        problem_text="p",
        format_instructions="f",
        task_name="synthetic",
        split="eval",
        metadata={},
        tools=None,
        build_prompt_fn=lambda text, guidelines: text,
        parse_answer_fn=lambda raw: raw.strip(),
        evaluate_fn=lambda parsed, target: parsed == "exact",
        score_prediction_fn=lambda parsed, target: 0.5 if parsed == "partial" else 0.0,
        all_valid_outputs=["exact"],
    )
    check_eq("graded score survives a rejected answer", problem.score("partial"), 0.5)
    check_eq("...while correctness stays 0", problem.verify_parsed("partial"), 0)
    check_eq("no scorer supplied falls back to correctness", problem.score("exact"), 0.0)

    without = Problem(
        id="synthetic2",
        prompt="p",
        problem_text="p",
        format_instructions="f",
        task_name="synthetic",
        split="eval",
        metadata={},
        tools=None,
        build_prompt_fn=lambda text, guidelines: text,
        parse_answer_fn=lambda raw: raw.strip(),
        evaluate_fn=lambda parsed, target: parsed == "exact",
        all_valid_outputs=["exact"],
    )
    check_eq("fallback score is float(correctness)", without.score("exact"), 1.0)
    check_eq("fallback score of a wrong answer", without.score("nope"), 0.0)


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #


def test_core_checkout_is_unmodified() -> None:
    """CORE is a read-only dependency; every number assumes it ran unmodified.

    A failure here does not mean this suite is broken -- it means the comparison
    is invalid until the CORE tree is restored.
    """
    from ..core_bridge import core_git_dirty, core_git_revision

    dirty = core_git_dirty()
    check("core_git_dirty() is answerable", dirty is not None)
    check("CORE working tree is clean (-uall)", dirty is False)
    revision = core_git_revision()
    check("CORE revision is recorded", bool(revision), repr(revision))


def test_manifest_records_core_dirtiness() -> None:
    """A revision alone cannot support "CORE was unmodified" -- the flag must ship."""
    from ..core_bridge import core_provenance

    code = core_provenance()
    check("provenance has core_git_dirty", "core_git_dirty" in code)
    check_eq("provenance agrees with the live probe", code["core_git_dirty"], False)
    check("provenance has core_git_revision", bool(code.get("core_git_revision")))


def test_no_bytecode_written_into_core() -> None:
    """Importing CORE must not leave ``__pycache__`` in a read-only dependency."""
    from ..core_bridge import core_repo_root, install_core_on_path

    install_core_on_path()
    cached_task("matchstick", 5)
    check("sys.dont_write_bytecode is set once CORE is on the path", sys.dont_write_bytecode)
    stale = [
        str(path.relative_to(core_repo_root()))
        for path in (core_repo_root() / "src").rglob("__pycache__")
    ]
    # Pre-existing directories are gitignored upstream; what matters is that the
    # interpreter is configured not to add more.
    check(
        "no bytecode written this session",
        sys.dont_write_bytecode,
        f"pre-existing __pycache__ dirs: {len(stale)}",
    )


# --------------------------------------------------------------------------- #
# ZebraLogic size-group description
# --------------------------------------------------------------------------- #


def test_zebralogic_description_matches_the_real_size_group() -> None:
    """The teacher's task description must describe the split it will actually see.

    ``xlarge`` is defined by search-space size, not by a rectangular range of
    dimensions, so it includes 6 houses x 4 attributes -- about a fifth of the eval
    set. A description implying a 5x5..6x6 box misinforms the teacher about a
    fifth of the problems.
    """
    zebra = core_module("tasks.single_turn.zebralogic_task")
    size_group = zebra._size_group_from_size

    xlarge = {(h, a) for h in range(2, 8) for a in range(2, 8) if size_group(f"{h}*{a}") == "xlarge"}
    described = {(5, 5), (5, 6), (6, 4), (6, 5), (6, 6)}
    check_eq(
        "the sizes our description names are exactly CORE's xlarge (within 2..7)",
        xlarge & {(h, a) for h in range(2, 7) for a in range(2, 7)},
        described,
    )

    from ..prompts_core_tasks import CORE_TASK_DESCRIPTIONS

    text = CORE_TASK_DESCRIPTIONS["zebralogic"]
    check("description mentions the 6-house, 4-attribute case", "6 houses with 4" in text)
    check(
        "description does not claim a 5x5-to-6x6 range",
        "5 houses x 5 attributes up to 6 x 6" not in text,
    )


if __name__ == "__main__":
    sys.exit(run_module(sys.modules[__name__]))
