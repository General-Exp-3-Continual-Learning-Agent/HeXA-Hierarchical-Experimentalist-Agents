"""Tests for the memorised-answer detector (`PHASE3_DESIGN.md` §5b).

Real `CoreTask`, real reference answers. The fixture that matters is the one taken
verbatim from the first completed Stage 3a smoke run, where a skill's ``example``
field carried two moves from its own source problem.
"""

from __future__ import annotations

from hexa_core.memorisation import (
    DEFAULT_MIN_RUN,
    find_memorised_answers,
    references_from_task,
    report,
)
from hexa_core.tasks import make_task
from hexa_core.tests import check, check_eq, run_module
from skillrl.core.skill_bank import Skill

TASK = "tower-of-hanoi"


def _task():
    return make_task(TASK, 4)


def _skill(skill_id: str, example: str, source_seeds: list) -> Skill:
    return Skill(
        skill_id=skill_id,
        title="Self-simulation verification",
        principle="Simulate the move list and check legality at each step.",
        when_to_apply="Before emitting the final answer.",
        example=example,
        source_level=TASK,
        source_type="contrastive",
        source_seeds=source_seeds,
        generation=1,
        confidence=1.0,
    )


def _first_moves(task, problem, n: int) -> str:
    """The first *n* moves of a problem's reference answer, as the teacher wrote them."""
    import re

    reference = str(
        task.core_task.format_reference_answer_for_prompt(problem.all_valid_outputs)
    )
    tuples = re.findall(r"\[\s*\d+(?:\s*,\s*\d+)+\s*\]", reference)
    return ",".join(t.replace(" ", "") for t in tuples[:n])


def test_catches_the_leak_from_the_live_run() -> None:
    """The exact shape observed: two verbatim moves inside `example`."""
    task = _task()
    problems = task.train_problems(4)
    parent = problems[2]
    leaked = _skill(
        "tow_ev_1_006",
        f"Iterate through the list [{_first_moves(task, parent, 2)}] and update state.",
        [parent.id],
    )

    leaks = find_memorised_answers([leaked], problems, references_from_task(task))
    check_eq("one leak found", len(leaks), 1)
    check_eq("attributed to the right problem", leaks[0].problem_id, parent.id)
    check("recognised as its own source problem", leaks[0].is_parent)
    check("report names the skill", "tow_ev_1_006" in report(leaks))


def test_method_illustrations_are_not_flagged() -> None:
    """A skill teaching method, not answers, must pass -- else the check is useless."""
    task = _task()
    problems = task.train_problems(4)
    clean = [
        _skill(
            "s1",
            "Represent state as {1:2, 2:1} meaning disk 1 on peg 2, disk 2 on peg 1.",
            [problems[0].id],
        ),
        _skill("s2", "Move the largest misplaced disk toward its goal peg first.", []),
        _skill("s3", "", [problems[1].id]),
    ]
    leaks = find_memorised_answers(clean, problems, references_from_task(task))
    check_eq("nothing flagged", len(leaks), 0)
    check_eq("report says clean", report(leaks).strip(), "no memorised answers detected (clean)")


def test_a_single_shared_move_is_not_enough() -> None:
    """`[1,0,2]` occurs by chance; only a *run* is evidence."""
    task = _task()
    problems = task.train_problems(4)
    one_move = _skill("s1", f"e.g. [{_first_moves(task, problems[0], 1)}]", [])
    check_eq(
        "single move ignored",
        len(find_memorised_answers([one_move], problems, references_from_task(task))),
        0,
    )
    check_eq("min_run is 2", DEFAULT_MIN_RUN, 2)


def test_leak_from_a_different_problem_is_flagged_and_labelled() -> None:
    """A skill carrying *another* problem's answer is worse, and must say so."""
    task = _task()
    problems = task.train_problems(4)
    stolen = _skill(
        "s1",
        f"Consider [{_first_moves(task, problems[1], 3)}].",
        [problems[0].id],  # distilled from problem 0, quoting problem 1
    )
    leaks = find_memorised_answers([stolen], problems, references_from_task(task))
    check("flagged", len(leaks) >= 1)
    match = next(leak for leak in leaks if leak.problem_id == problems[1].id)
    check("labelled as a different problem", not match.is_parent)
    check("describe() says so", "DIFFERENT problem" in match.describe())


def test_the_wrong_resolver_silently_misses_a_real_leak() -> None:
    """Why `reference_for` is a required argument rather than inferred.

    The first version of this module resolved the reference answer as
    ``problem.all_valid_outputs`` -- which is the problem *spec*
    (``{'num_disks': 5, 'initial_state': [...], ...}``), not the answer -- and
    reported "clean" on a bank that provably contained a memorised answer.

    Note the empty-answer `ValueError` does **not** save you here: the spec contains
    numeric lists of its own (``initial_state``), so the detector gets tuples, just
    the wrong ones. Nothing about the output distinguishes this from a genuine
    all-clear, which is exactly why the resolver must be passed explicitly.
    """
    task = _task()
    problems = task.train_problems(4)
    parent = problems[2]
    leaked = _skill(
        "s1", f"Iterate through [{_first_moves(task, parent, 2)}].", [parent.id]
    )

    correct = find_memorised_answers(
        [leaked], problems, references_from_task(task)
    )
    check_eq("the correct resolver finds it", len(correct), 1)

    wrong = find_memorised_answers(
        [leaked], problems, lambda problem: str(problem.all_valid_outputs)
    )
    check_eq("the spec-as-answer resolver misses it", len(wrong), 0)


def test_an_empty_reference_set_raises_rather_than_passing() -> None:
    """A resolver yielding nothing to compare against must not read as an all-clear."""
    task = _task()
    try:
        find_memorised_answers(
            [_skill("s1", "[[1,2,1],[5,0,2]]", [])],
            task.train_problems(4),
            lambda problem: "no numbers here at all",
        )
    except ValueError as exc:
        check("error points at the resolver", "reference_for" in str(exc))
    else:
        check("an undetectable check must raise", False, "returned instead")


def _zebra():
    """ZebraLogic, whose answer is a grid rather than a string or a move list."""
    return make_task("zebralogic", 4)


def _zebra_skill(skill_id: str, principle: str = "", example: str = "") -> Skill:
    return Skill(
        skill_id=skill_id,
        title="Constraint ordering",
        principle=principle or "Propagate the tightest clue first.",
        when_to_apply="When several attributes are still unassigned.",
        example=example,
        source_level="zebralogic",
        source_type="contrastive",
        source_seeds=[],
        generation=1,
        confidence=1.0,
    )


def _zebra_row(task, problem) -> list:
    import json

    reference = str(
        task.core_task.format_reference_answer_for_prompt(problem.all_valid_outputs)
    )
    return json.loads(reference)["rows"][0]


def test_zebralogic_prompt_vocabulary_is_not_a_leak() -> None:
    """The check must survive the task it nearly false-positived on.

    Every ZebraLogic answer *cell* is enumerated in the puzzle's own prompt, so naming
    columns, or one or two attribute values, is method talk and not memorisation.
    """
    task = _zebra()
    problems = task.train_problems(4)
    reference_for = references_from_task(task)
    row = _zebra_row(task, problems[0])

    honest = [
        _zebra_skill("z1", "Fix FavoriteSport before HairColor, then Color."),
        _zebra_skill("z2", f"Anchor on the {row[2]} constraint, then propagate."),
        _zebra_skill("z3", f"If the {row[2]} person eats {row[5]}, the rest are forced."),
    ]
    for skill in honest:
        found = find_memorised_answers([skill], problems, reference_for)
        check_eq(f"{skill.skill_id} is clean", len(found), 0)


def test_zebralogic_catches_a_recited_assignment() -> None:
    """Three cells of one row is the assignment, which the prompt cannot supply."""
    task = _zebra()
    problems = task.train_problems(4)
    row = _zebra_row(task, problems[0])
    leaked = _zebra_skill(
        "z_leak",
        example=(
            f"For example {row[1]} plays {row[2]}, lives in the {row[3]} house "
            f"and eats {row[5]}."
        ),
    )
    found = find_memorised_answers([leaked], problems, references_from_task(task))
    check("a recited row is reported", len(found) >= 1)
    check_eq("reported as an assignment", found[0].kind, "assignment")
    check("names the row's cells", row[1] in found[0].matched)
    check("one row reports once", len([f for f in found if f.skill_id == "z_leak"]), 1)


def test_a_short_cell_cannot_carry_a_match() -> None:
    """The house number is a single digit; it must not count towards a row."""
    task = _zebra()
    problems = task.train_problems(4)
    row = _zebra_row(task, problems[0])
    check_eq("row starts with the house number", row[0], "1")
    # Only the number plus one real value: below the 3-cell bar even if "1" counted.
    skill = _zebra_skill("z_short", f"Start at house {row[0]} and test {row[2]}.")
    found = find_memorised_answers([skill], problems, references_from_task(task))
    check_eq("digit + one value is not a leak", len(found), 0)


def test_matchstick_answers_are_still_caught_verbatim() -> None:
    """The prompt filter must not blunt the single-string path it was added beside.

    A Matchstick answer is *not* in its problem's prompt -- the prompt holds the broken
    equation -- so all of its candidates survive the filter.
    """
    task = make_task("matchstick", 8)
    problems = task.train_problems(8)
    reference_for = references_from_task(task)
    answer = _answer_of(task, problems[3])
    leaked = _zebra_skill("m_leak", example=f"e.g. {answer} works")
    found = find_memorised_answers([leaked], problems, reference_for)
    check("the verbatim equation is reported", len(found) >= 1)
    check_eq("reported as an answer", found[0].kind, "answer")


def _answer_of(task, problem) -> str:
    """One valid answer string for a Matchstick problem."""
    import re

    reference = str(
        task.core_task.format_reference_answer_for_prompt(problem.all_valid_outputs)
    )
    quoted = re.findall(r'"([^"]{8,120})"', reference)
    if not quoted:
        raise AssertionError("no answer string in the Matchstick reference")
    return quoted[0]


if __name__ == "__main__":
    raise SystemExit(run_module(__import__(__name__, fromlist=["_"])))
