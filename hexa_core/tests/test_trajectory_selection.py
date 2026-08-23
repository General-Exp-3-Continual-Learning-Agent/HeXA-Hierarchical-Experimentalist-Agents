"""Tests for evolver trajectory selection (`WEAKNESSES.md` §1)."""
from __future__ import annotations

from hexa_core.hexa_bridge import (
    TRAJECTORY_SELECTIONS,
    _select_balanced,
    _select_paired,
    _select_top,
    attempt_utility,
    make_trajectory_formatter,
)
from hexa_core.tests import check, check_eq, run_module


def _t(pid: str, ok: bool, b_q: float) -> dict:
    """One attempt. `baseline` makes attempt_utility grade it by difficulty."""
    return {"seed": pid, "success": ok, "baseline": b_q, "task_name": "t",
            "action": f"{pid}-{'S' if ok else 'F'}-{b_q}", "trajectory": []}


def _chosen(mode: str, trajs: list[dict], max_trajs: int = 5) -> list[dict]:
    """The trajectories a mode picks.

    Calls the selectors directly rather than parsing the rendered block: the renderer
    is `format_trajectory_for_teacher`, which is HeXA's InterPhyre version unless
    `patch()` has run, and it does not emit the fields a fixture could be recovered by.
    Selection is what is under test here; rendering is covered by test_hexa_seam.
    """
    reward = attempt_utility
    if mode == "top5":
        return _select_top(trajs, max_trajs, reward)
    if mode == "balanced":
        return _select_balanced(trajs, max_trajs, reward, 3)
    return _select_paired(trajs, max_trajs, reward, 3)


# A realistic round: 2 problems, both with mixed outcomes, 8 successes / 4 failures.
ROUND = (
    [_t("A", True, 0.2) for _ in range(4)]
    + [_t("A", False, 0.2) for _ in range(2)]
    + [_t("B", True, 0.8) for _ in range(4)]
    + [_t("B", False, 0.8) for _ in range(2)]
)


def test_top5_sends_no_failures_which_is_the_bug() -> None:
    """The baseline arm, and the behaviour every Hanoi round exhibited."""
    got = _chosen("top5", ROUND)
    check_eq("5 chosen", len(got), 5)
    check_eq("all successes", sum(1 for t in got if t["success"]), 5)
    check("no failure reached the evolver", not any(not t["success"] for t in got))


def test_balanced_enforces_a_quota() -> None:
    got = _chosen("balanced", ROUND)
    check_eq("3 successes", sum(1 for t in got if t["success"]), 3)
    check_eq("2 failures", sum(1 for t in got if not t["success"]), 2)


def test_balanced_picks_the_most_damning_failures() -> None:
    """Failures on problems the model usually solves (high b_q => reward near -1)."""
    got = _chosen("balanced", ROUND)
    fails = [t for t in got if not t["success"]]
    check("failures come from the EASY problem B", all(t["seed"] == "B" for t in fails),
          f"got {[t['seed'] for t in fails]}")


def test_paired_gives_same_problem_contrast() -> None:
    """The point of the mode: contrast that holds problem difficulty fixed."""
    got = _chosen("paired", ROUND)
    by_problem = {}
    for t in got:
        by_problem.setdefault(t["seed"], set()).add(t["success"])
    paired = [p for p, outcomes in by_problem.items() if len(outcomes) == 2]
    check_eq("both problems contributed a success AND a failure", sorted(paired), ["A", "B"])
    check("at least one failure", any(not t["success"] for t in got))


def test_paired_degrades_to_balanced_without_mixed_problems() -> None:
    """A round where no problem has both outcomes must not return nothing."""
    split = [_t("A", True, 0.5) for _ in range(4)] + [_t("B", False, 0.5) for _ in range(4)]
    got = _chosen("paired", split)
    check_eq("still fills the block", len(got), 5)
    check("has both classes", any(t["success"] for t in got) and any(not t["success"] for t in got))


def test_every_mode_backfills_when_a_class_is_missing() -> None:
    """`we just take what we have` -- an all-success round still sends 5."""
    all_ok = [_t("A", True, 0.5) for _ in range(9)]
    for mode in TRAJECTORY_SELECTIONS:
        check_eq(f"{mode} sends 5 from an all-success round", len(_chosen(mode, all_ok)), 5)


def test_unknown_mode_raises() -> None:
    try:
        make_trajectory_formatter("greedy")
    except ValueError as exc:
        check("names the valid modes", "top5" in str(exc))
    else:
        check("unknown mode must raise", False, "accepted")


def test_the_formatter_overrides_the_hardcoded_call_site_width() -> None:
    """`evolving_distill` passes `max_trajs=5`; the configured width must win.

    This is the whole mechanism of the §1 fix. If the call-site argument won instead,
    the run would silently keep showing the evolver 5 of its 600 trajectories while its
    provenance claimed 12.
    """
    import skillrl.distillation.teacher_prompts as tp

    trajectories = [_t(f"p{i}", i % 2 == 0, 0.5) for i in range(20)]
    original = tp.format_trajectory_for_teacher
    tp.format_trajectory_for_teacher = lambda t: f"<{t['action']}>"
    try:
        for width in (5, 12):
            block = make_trajectory_formatter("paired", max_trajs=width)(
                trajectories, max_trajs=5  # what the real call site passes
            )
            check_eq(f"width {width} honoured", block.count("<"), width)
    finally:
        tp.format_trajectory_for_teacher = original


def test_paired_honours_the_quota_at_every_width() -> None:
    """The pairs' own successes must count against the quota, not sit outside it.

    `_select_paired` places one pair per problem, then backfills. Handing the backfill
    the *whole* success quota double-counts the pairs' successes: at width 12 with 2
    problems per round the pairs fill 2S/2F and the backfill was then asked for 7 more
    successes out of 8 slots, sending 9S/3F for a nominal 7S/5F. Width 5 hid it -- two
    pairs plus a single backfill slot capped to 1 success is exactly 3S/2F.
    """
    from hexa_core.hexa_bridge import default_success_slots

    # Two problems, ten attempts each, evenly split -- the new schedule's round shape.
    trajectories = [
        _t(f"p{p}", ok, 0.5)
        for p in range(2)
        for ok in ([True] * 5 + [False] * 5)
    ]
    for width in (5, 8, 12):
        slots = default_success_slots(width)
        chosen = _select_paired(trajectories, width, attempt_utility, slots)
        successes = sum(1 for t in chosen if t["success"])
        check_eq(f"width {width} sends a full block", len(chosen), width)
        check_eq(f"width {width} honours the quota", successes, slots)


def test_the_success_share_scales_with_the_width() -> None:
    """Widening the block must widen both classes, not invert it.

    `DEFAULT_SUCCESS_SLOTS` is 3 *of 5*. Kept as an absolute count it would mean 3
    successes and 9 failures at width 12 -- a failure-dominated block nobody asked for.
    """
    from hexa_core.hexa_bridge import default_success_slots

    check_eq("width 5 unchanged", default_success_slots(5), 3)
    check_eq("width 12 scales", default_success_slots(12), 7)
    for width in (2, 5, 8, 12, 20):
        slots = default_success_slots(width)
        check(f"width {width} keeps a failure slot", 1 <= slots <= width - 1)

    explicit = make_trajectory_formatter("balanced", success_slots=2, max_trajs=5)
    check("an explicit share is still honoured", explicit is not None)
    try:
        make_trajectory_formatter("balanced", success_slots=9, max_trajs=5)
    except ValueError as exc:
        check("an out-of-range share raises", "success_slots" in str(exc))
    else:
        check("success_slots > max_trajs must raise", False, "accepted")


def test_a_bad_width_is_rejected_at_construction() -> None:
    try:
        make_trajectory_formatter("paired", max_trajs=0)
    except ValueError as exc:
        check("names max_trajs", "max_trajs" in str(exc))
    else:
        check("max_trajs=0 must raise", False, "accepted")


def test_a_long_thought_is_elided_from_the_middle_not_the_tail() -> None:
    """The tail of a reasoning trace is where a wrong answer's error surfaces.

    A tail cut would keep only the solver's setup and discard every mistake, which is
    the opposite of what the evolver is being shown trajectories for.
    """
    from hexa_core.hexa_bridge import _THOUGHT_MAX_CHARS, _elide_middle

    text = "HEAD" + ("x" * (_THOUGHT_MAX_CHARS * 2)) + "TAIL"
    out = _elide_middle(text, _THOUGHT_MAX_CHARS)
    check("head kept", out.startswith("HEAD"))
    check("tail kept", out.endswith("TAIL"))
    check("says what it dropped", "elided from the middle" in out)
    check("shorter than the input", len(out) < len(text))
    short = "already short"
    check_eq("under the cap is untouched", _elide_middle(short, 100), short)


if __name__ == "__main__":
    raise SystemExit(run_module(__import__(__name__, fromlist=["_"])))
