"""Aggregation tests -- built from synthetic runs, no env and no model."""

from __future__ import annotations

from hexa_orbit.actor import InstanceRun, StepRecord
from hexa_orbit.report import aggregate, arm_summary, episode_success_rates


def make_step(episode_index: int, action: str = "up", finish: str = "stop") -> StepRecord:
    return StepRecord(
        episode_index=episode_index,
        episode_step=1,
        total_step=1,
        observation="obs",
        raw_text=f"\\boxed{{{action}}}",
        reasoning_text="",
        final_text=f"\\boxed{{{action}}}",
        parsed_action=action,
        env_action=f"\\boxed{{{action}}}",
        reward=0.0,
        finish_reason=finish,
        episode_done=True,
    )


def make_run(
    successes: list[bool],
    *,
    leaked: bool | None = False,
    actions: list[str] | None = None,
    added_tokens: int = 0,
    error: str | None = None,
) -> InstanceRun:
    actions = actions or ["up"] * len(successes)
    steps = [make_step(i, a) for i, a in enumerate(actions)]
    return InstanceRun(
        instance_id=f"i{abs(hash(tuple(successes) + tuple(actions))) % 10000}",
        dataset_seed=1,
        resolved_seed=1,
        shortest_path_len=7,
        leaked=leaked,
        steps=steps,
        messages=[],
        metrics={},
        episode_successes=successes,
        episode_lengths=[7] * len(successes),
        success_count=sum(successes),
        added_tokens=added_tokens,
        token_counter="stub",
        error=error,
    )


def test_episode_rates_are_positional():
    runs = [make_run([True, False, False]), make_run([False, True, True])]
    assert episode_success_rates(runs) == {
        "episode_1": 0.5,
        "episode_2": 0.5,
        "episode_3": 0.5,
    }


def test_leakage_split_partitions_and_reports_gap():
    runs = [
        make_run([True, True, True], leaked=True),
        make_run([False, False, False], leaked=False),
        make_run([False, False, False], leaked=False),
    ]
    report = aggregate(runs)
    assert report["pooled"]["n"] == 3
    assert report["leaked"]["n"] == 1
    assert report["clean"]["n"] == 2
    assert report["leaked"]["episode_success_rate"]["episode_1"] == 1.0
    assert report["clean"]["episode_success_rate"]["episode_1"] == 0.0
    assert report["episode_1_leakage_gap"] == 1.0


def test_errored_instances_are_excluded_not_counted_as_failures():
    good = make_run([True, True, True])
    bad = make_run([], error="GenerationFailed")
    report = aggregate([good, bad], errored=[bad])
    assert report["pooled"]["n"] == 1
    assert report["pooled"]["any_success_rate"] == 1.0
    assert report["n_errored"] == 1


def test_no_adaptation_rate_covers_only_never_solved_instances():
    """An instance that solved something is not a no-adaptation case."""
    repeated = make_run([False] * 3, actions=["up", "up", "up"])
    varied = make_run([False] * 3, actions=["up", "down", "left"])
    solved = make_run([True, False, False], actions=["up", "up", "up"])
    summary = arm_summary([repeated, varied, solved])
    assert summary["n_never_solved"] == 2
    assert summary["no_adaptation_rate"] == 0.5


def test_diagnostic_rates_are_per_turn():
    runs = [make_run([False], actions=["not an action"], leaked=False)]
    summary = arm_summary(runs)
    assert summary["unparsed_rate"] == 0.0  # raw_text is boxed by make_step
    truncated = [make_run([False], actions=["up"])]
    truncated[0].steps[0].finish_reason = "length"
    assert arm_summary(truncated)["truncated_rate"] == 1.0


def test_empty_input_is_safe():
    assert arm_summary([])["n"] == 0
    assert aggregate([])["pooled"]["n"] == 0
