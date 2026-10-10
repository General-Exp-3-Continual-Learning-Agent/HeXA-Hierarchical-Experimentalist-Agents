"""Phase 3 items 1-4: trajectory dict, reward, teacher rendering, bank audit.

All pure -- no server, no model calls. The one test that touches ``skillrl`` is
:func:`test_install_rebinds_the_seams`, which is the point: the formatter and
prompts have to actually land on the modules ``distill`` reads them from, and a
docstring claiming so is not evidence.
"""

from __future__ import annotations

import json
import pathlib
import re

import pytest
from skillrl.core.skill_bank import Skill, SkillBank

from hexa_orbit import audit, oracle, reward, teacher, trajectory
from hexa_orbit.actor import MazeActor
from hexa_orbit.adapter import EPISODES_PER_INSTANCE, MazeTask
from hexa_orbit.tests.test_actor import StubClient, gen


@pytest.fixture(scope="module")
def task() -> MazeTask:
    return MazeTask()


@pytest.fixture(scope="module")
def spec(task: MazeTask):
    return task.eval_specs(1)[0]


@pytest.fixture(scope="module")
def solved_run(task: MazeTask, spec):
    """A real instance solved optimally in all three episodes."""
    probe = task.instance(spec)
    probe.reset()
    actions = oracle.instance_optimal_actions(probe)
    script = [gen(f"go {a} {oracle.boxed(a)}", reasoning="deliberating") for a in actions * 3]
    actor = MazeActor(client=StubClient(script))
    return actor.run_instance(task.instance(spec), leaked=False)


@pytest.fixture(scope="module")
def stuck_run(task: MazeTask, spec):
    """A real instance that repeats one move forever -- the -1.00 bin."""
    actor = MazeActor(client=StubClient(default=gen(oracle.boxed("up"))))
    return actor.run_instance(task.instance(spec), leaked=True)


# ── 1. trajectory dict ───────────────────────────────────────────────────

def test_trajectory_dict_satisfies_skillrl_contract(solved_run):
    traj = trajectory.instance_trajectory(solved_run)
    for key in ("seed", "success", "iterations", "trajectory", "action"):
        assert key in traj
    assert isinstance(traj["seed"], str)          # string ids round-trip
    assert traj["success"] is True
    assert traj["iterations"] == len(solved_run.steps)
    assert len(traj["trajectory"]) == len(solved_run.steps)
    assert set(traj["trajectory"][0]) >= {"thought", "action", "observation"}


def test_positions_and_outcomes_are_recovered(solved_run):
    traj = trajectory.instance_trajectory(solved_run)
    entries = traj["trajectory"]
    assert all(e["position"] is not None for e in entries), "position parse failed"
    # Optimal play never bumps a wall, and each episode ends at the goal.
    assert all(e["result"] in {"moved", "GOAL"} for e in entries)
    assert sum(1 for e in entries if e["result"] == "GOAL") == EPISODES_PER_INSTANCE
    # The first turn stands on the start cell.
    assert entries[0]["position"] == trajectory.position_in(solved_run.steps[0].observation)


def test_wall_bumps_are_labelled(stuck_run):
    traj = trajectory.instance_trajectory(stuck_run)
    results = {e["result"] for e in traj["trajectory"]}
    assert results & {"wall", "budget exhausted"}
    assert traj["success"] is False


def test_reasoning_is_elided_unless_requested(solved_run):
    plain = trajectory.instance_trajectory(solved_run)
    verbose = trajectory.instance_trajectory(solved_run, include_reasoning=True)
    assert all(e["thought"] == "" for e in plain["trajectory"])
    assert all(e["thought"] == "deliberating" for e in verbose["trajectory"])


def test_split_drops_errored_runs(solved_run, stuck_run):
    import dataclasses

    errored = dataclasses.replace(stuck_run, error="GenerationFailed: boom")
    successes, failures = trajectory.split_successes_failures(
        [solved_run, stuck_run, errored]
    )
    assert len(successes) == 1 and len(failures) == 1


def test_first_success_episode():
    assert trajectory.first_success_episode([False, True, False]) == 2
    assert trajectory.first_success_episode([True, True, True]) == 1
    assert trajectory.first_success_episode([False] * 3) is None
    assert trajectory.first_success_episode([]) is None


# ── 2. reward ────────────────────────────────────────────────────────────

def _r(**kwargs):
    base = dict(
        first_success_episode=None, episode_1_steps=9, shortest_path_len=7, adapted=True
    )
    return reward.reward_from_parts(**{**base, **kwargs})


def test_reward_bins_are_ordered_and_in_range():
    values = [
        _r(first_success_episode=1, episode_1_steps=7),
        _r(first_success_episode=1, episode_1_steps=9),
        _r(first_success_episode=2),
        _r(first_success_episode=3),
        _r(adapted=True),
        _r(adapted=False),
    ]
    assert values == sorted(values, reverse=True), values
    assert all(-1.0 <= v <= 1.0 for v in values)
    assert values[0] == 1.0 and values[-1] == -1.0


def test_episode_1_success_always_beats_episode_2():
    """The floor exists so a wasteful cold-start solve still outranks a slow one."""
    worst_fast = _r(first_success_episode=1, episode_1_steps=99)
    assert worst_fast == reward.FAST_SUCCESS_FLOOR > _r(first_success_episode=2)


def test_no_adaptation_is_the_floor_and_distinct_from_exploring():
    assert _r(adapted=False) < _r(adapted=True)
    assert _r(adapted=False) == -1.0


def test_reward_rejects_an_episode_beyond_the_cap():
    """A 4th-episode success means the D2 loop broke; fail loudly, don't absorb it."""
    with pytest.raises(ValueError):
        _r(first_success_episode=4)


def test_instance_reward_matches_the_trajectory_key(solved_run, stuck_run):
    for run in (solved_run, stuck_run):
        traj = trajectory.instance_trajectory(run)
        assert traj["reward"] == reward.instance_reward(run)
        assert reward.trajectory_reward(traj) == traj["reward"]


def test_trajectory_reward_recomputes_when_the_key_is_absent(solved_run):
    traj = trajectory.instance_trajectory(solved_run)
    stripped = {k: v for k, v in traj.items() if k != "reward"}
    assert reward.trajectory_reward(stripped) == traj["reward"]


def test_reward_bin_names(solved_run, stuck_run):
    assert reward.reward_bin(trajectory.instance_trajectory(solved_run)) == "fast success"
    assert reward.reward_bin(trajectory.instance_trajectory(stuck_run)) == "no adaptation"
    assert reward.reward_bin({"first_success_episode": 3}) == "slow success"
    assert (
        reward.reward_bin({"first_success_episode": None, "adapted_across_episodes": True})
        == "exploratory failure"
    )


# ── 3. teacher rendering ─────────────────────────────────────────────────

def test_render_shows_all_three_episodes_not_a_tail(solved_run):
    """Truncating from the end would drop episode 1 -- the episode under test."""
    text = teacher.format_trajectory_for_teacher(
        trajectory.instance_trajectory(solved_run)
    )
    for episode in (1, 2, 3):
        assert f"--- Episode {episode}:" in text
    assert "SOLVED in" in text
    assert "=== Instance" in text and solved_run.instance_id in text


def test_render_flags_non_adaptation_and_unparsed_turns(stuck_run):
    traj = trajectory.instance_trajectory(stuck_run)
    text = teacher.format_trajectory_for_teacher(traj)
    assert "did not use its own earlier episodes" in text
    assert "reward -1.00 (no adaptation)" in text

    unparsed = {**traj, "n_unparsed": 4}
    assert "4 turn(s) produced no parseable action" in teacher.format_trajectory_for_teacher(unparsed)


def test_render_is_compact(solved_run):
    """~30 lines per instance is what max_trajs=6 was sized against."""
    text = teacher.format_trajectory_for_teacher(
        trajectory.instance_trajectory(solved_run)
    )
    assert len(text.splitlines()) < 45
    assert "deliberating" not in text  # reasoning elided by default


def test_render_can_include_reasoning(solved_run):
    text = teacher.format_trajectory_for_teacher(
        trajectory.instance_trajectory(solved_run, include_reasoning=True),
        include_reasoning=True,
    )
    assert "reasoning: deliberating" in text


def test_per_episode_truncation_keeps_every_episode(stuck_run):
    lines = teacher.render_episodes(
        trajectory.instance_trajectory(stuck_run), max_turns_per_episode=2
    )
    text = "\n".join(lines)
    assert text.count("further turns omitted") == EPISODES_PER_INSTANCE
    for episode in (1, 2, 3):
        assert f"--- Episode {episode}:" in text


def test_task_description_leaks_no_strategy():
    """The four structural regularities are what we test the evolver on finding."""
    described = teacher.MAZE_TASK_DESCRIPTION.lower()
    for leak in ("farthest", "7-8", "shortest path is", "wall-bump"):
        assert leak not in described


def test_prompts_have_the_slots_their_callers_fill():
    for prompt, slots in (
        (
            teacher.CONTRASTIVE_DISTILLATION_PROMPT_MAZE,
            ("level_name", "level_description", "n_successes", "n_failures",
             "success_block", "failure_block"),
        ),
        (
            teacher.COMMON_MISTAKES_PROMPT_MAZE,
            ("level_name", "level_description", "failure_block"),
        ),
        (
            teacher.SKILL_BANK_EVOLUTION_PROMPT_MAZE,
            ("level_name", "level_description", "existing_skills_block", "n_successes",
             "n_failures", "new_trajectories_block", "max_skills", "max_mistakes"),
        ),
    ):
        for slot in slots:
            assert "{" + slot + "}" in prompt, slot
        # Every other brace must be doubled, or `.format` raises on the JSON block.
        prompt.format(**{slot: "x" for slot in slots})


def test_install_rebinds_the_seams():
    """The formatter and prompts must land where ``distill`` reads them from."""
    import skillrl.distillation.distill as _distill
    import skillrl.distillation.evolving_distill as _evolving
    import skillrl.distillation.teacher_prompts as _tp
    from skillrl.core import config as _config

    teacher.install()
    assert teacher.is_installed()
    assert _tp.compute_trajectory_reward is reward.trajectory_reward
    assert _distill.compute_trajectory_reward is reward.trajectory_reward
    assert _distill.CONTRASTIVE_DISTILLATION_PROMPT is teacher.CONTRASTIVE_DISTILLATION_PROMPT_MAZE
    assert _distill.COMMON_MISTAKES_PROMPT is teacher.COMMON_MISTAKES_PROMPT_MAZE
    assert _evolving.SKILL_BANK_EVOLUTION_PROMPT is teacher.SKILL_BANK_EVOLUTION_PROMPT_MAZE
    assert _config.LEVEL_DESCRIPTIONS["maze"] == teacher.MAZE_TASK_DESCRIPTION
    # The formatter override must survive patch(), which binds CORE's first.
    rendered = _tp.format_trajectory_for_teacher({"seed": "abc", "trajectory": []})
    assert "=== Instance abc" in rendered


# ── 4. audit ─────────────────────────────────────────────────────────────

def _skill(skill_id: str, **kwargs) -> Skill:
    base = dict(title="t", principle="p", when_to_apply="w", example="")
    return Skill(skill_id=skill_id, **{**base, **kwargs})


@pytest.mark.parametrize(
    "text",
    [
        "Go left, left, then up.",
        "Move up up right to escape.",
        "The route is down and down and left.",
    ],
)
def test_move_sequences_are_flagged(text: str):
    assert audit.flag_skill(_skill("s", principle=text)) is not None


@pytest.mark.parametrize(
    "text",
    [
        "Prefer up or left when both are open.",              # two, legitimate
        "Increase distance from the start cell.",
        "Treat a wall bump as evidence and turn right instead.",
    ],
)
def test_legitimate_guidance_is_not_flagged(text: str):
    assert audit.flag_skill(_skill("s", principle=text)) is None


def test_coordinate_literals_need_two_to_flag():
    assert audit.flag_skill(_skill("s", principle="the cell (3, 3) is a wall")) is None
    assert audit.flag_skill(_skill("s", principle="(3, 3) and (5, 2) are walls")) is not None


def test_audit_bank_reports_fraction_and_manual_check():
    bank = SkillBank()
    bank.level_skills["maze"] = [
        _skill("m1", principle="Go left, left, then up."),
        _skill("m2", principle="Expand toward unexplored frontier."),
    ]
    result = audit.audit_bank(bank, round_index=3)
    assert result["n_skills"] == 2
    assert result["n_flagged"] == 1
    assert result["flagged_fraction"] == 0.5
    assert result["round"] == 3
    assert len(result["manual_check"]["regularities"]) == 4
    assert "m1" in audit.format_audit(result)


def test_leakage_delta_needs_both_sides():
    both = {
        "leaked": {"n": 10, "episode_success_rate": {"episode_1": 0.4}},
        "clean": {"n": 40, "episode_success_rate": {"episode_1": 0.1}},
    }
    assert audit.leakage_delta(both)["gap"] == pytest.approx(0.3)
    assert audit.leakage_delta({"leaked": {"n": 0}, "clean": {"n": 5}}) is None


def test_audit_survives_an_empty_bank():
    result = audit.audit_bank(SkillBank())
    assert result["n_skills"] == 0 and result["flagged_fraction"] == 0.0


def test_position_is_carried_through_wall_bumps(stuck_run):
    """A wall bump replies with no coordinates, but the agent has not moved.

    Without carry-forward a failing episode renders as a column of "?" precisely
    where the teacher most needs to see where the agent was stuck -- and the
    carried value is a fact about the env (``_execute_action`` leaves
    ``current_position`` untouched on a wall bump), not an inference.
    """
    entries = trajectory.instance_trajectory(stuck_run)["trajectory"]
    assert all(e["position"] is not None for e in entries)
    # The stuck agent bumps the same wall every turn, so it never leaves the start.
    start = entries[0]["position"]
    assert {e["position"] for e in entries} == {start}
    assert entries[0]["position_carried"] is False
    assert any(e["position_carried"] for e in entries)


def test_carry_forward_does_not_leak_across_episodes(stuck_run):
    """Each episode's first turn reads its own stated start, not a carried value.

    The boundary observation is the previous episode's terminal text concatenated
    with the new episode's opening. Neither maze terminal message contains
    coordinates, so the first match is the new episode's start cell.
    """
    entries = trajectory.instance_trajectory(stuck_run)["trajectory"]
    firsts = [e for e in entries if e["turn"] == 1]
    assert len(firsts) == EPISODES_PER_INSTANCE
    assert all(e["position_carried"] is False for e in firsts)


# ── 4b. source_seeds provenance ──────────────────────────────────────────

_KNOWN = ["132dc55db9c9f536", "e784773bdf393bd5", "0cbf328ba1e55168"]


@pytest.mark.parametrize(
    ("cited", "expected"),
    [
        ("132dc55db9c9f536", "132dc55db9c9f536"),                       # exact
        ("132dc55d", "132dc55db9c9f536"),                               # truncated
        ("132dc55db9c9f536ff00", "132dc55db9c9f536"),                   # extended
        ("  e784773bdf393bd5  ", "e784773bdf393bd5"),                   # whitespace
        ("zzzz", None),                                                 # invented
        ("", None),
    ],
)
def test_resolve_citation(cited, expected):
    assert audit.resolve_citation(cited, _KNOWN) == expected


def test_ambiguous_prefix_is_never_guessed():
    """Guessing wrong attributes a skill to the wrong evidence -- worse than a miss."""
    assert audit.resolve_citation("1", ["1abc", "1def"]) is None


def test_validate_source_seeds_separates_fell_back_from_repaired():
    skills = [
        _skill("s1", title="exact"),
        _skill("s2", title="repaired"),
        _skill("s3", title="fell back"),
        _skill("s4", title="uncited"),
    ]
    skills[0].source_seeds = [_KNOWN[0]]
    skills[1].source_seeds = [_KNOWN[0][:8], _KNOWN[1]]
    skills[2].source_seeds = ["nope", "also-nope"]
    skills[3].source_seeds = []

    report = audit.validate_source_seeds(skills, _KNOWN)
    assert (report["n_exact"], report["n_repaired"], report["n_unknown"]) == (2, 1, 2)
    assert [s["title"] for s in report["fell_back"]] == ["fell back"]
    assert [s["title"] for s in report["uncited"]] == ["uncited"]
    assert report["miss_rate"] == pytest.approx(3 / 5)
    assert "NO citation resolved" in audit.format_source_seed_report(report)


def test_clean_provenance_report_is_quiet():
    skill = _skill("s1")
    skill.source_seeds = [_KNOWN[0]]
    report = audit.validate_source_seeds([skill], _KNOWN)
    assert audit.format_source_seed_report(report) == "source_seeds: 1 citations, all exact"


def test_audit_bank_includes_provenance_only_when_ids_are_given():
    bank = SkillBank()
    skill = _skill("m1")
    skill.source_seeds = ["nope"]
    bank.level_skills["maze"] = [skill]
    assert audit.audit_bank(bank)["source_seeds"] is None
    with_ids = audit.audit_bank(bank, known_ids=_KNOWN)["source_seeds"]
    assert with_ids["n_unknown"] == 1 and len(with_ids["fell_back"]) == 1


# ── the preventive half: the confidence wrapper ──────────────────────────

def test_confidence_wrapper_repairs_a_truncated_id():
    """The whole point: a truncated id must yield its own evidence, not the mean."""
    import skillrl.distillation.distill as _distill

    teacher.install()
    teacher.reset_confidence_stats()
    trajs = [{"seed": _KNOWN[0], "reward": 1.0}, {"seed": _KNOWN[1], "reward": -0.5}]

    exact = _distill._compute_initial_confidence([_KNOWN[0]], trajs)
    repaired = _distill._compute_initial_confidence([_KNOWN[0][:8]], trajs)
    round_mean = _distill._compute_initial_confidence(["invented"], trajs)

    assert repaired == exact == 1.0
    assert round_mean != exact           # stock fallback preserved, not "fixed"
    assert round_mean == pytest.approx(0.625)

    stats = teacher.confidence_stats()
    assert (stats["exact"], stats["repaired"], stats["unknown"]) == (1, 1, 1)
    assert stats["fell_back"] == 1


def test_confidence_wrapper_preserves_stock_behaviour_on_a_total_miss():
    """No number moves silently: an unresolvable citation still gets the round mean.

    The wrapper's job is to make that visible, not to invent a different value --
    changing the number would make old runs incomparable for no gain.
    """
    import skillrl.distillation.distill as _distill

    teacher.install()
    trajs = [{"seed": _KNOWN[0], "reward": 1.0}, {"seed": _KNOWN[1], "reward": -1.0}]
    stock = _distill._compute_initial_confidence.__wrapped_stock__
    assert _distill._compute_initial_confidence(["bogus"], trajs) == stock(["bogus"], trajs)


def test_confidence_wrapper_is_idempotent_across_reinstalls():
    """Re-installing must not wrap the wrapper, or stats would double-count."""
    import skillrl.distillation.distill as _distill

    teacher.install()
    first = _distill._compute_initial_confidence
    teacher.install()
    second = _distill._compute_initial_confidence
    assert first.__wrapped_stock__ is second.__wrapped_stock__

    teacher.reset_confidence_stats()
    _distill._compute_initial_confidence([_KNOWN[0]], [{"seed": _KNOWN[0], "reward": 1.0}])
    assert teacher.confidence_stats()["calls"] == 1


def test_uncited_skills_are_counted_not_repaired():
    import skillrl.distillation.distill as _distill

    teacher.install()
    teacher.reset_confidence_stats()
    _distill._compute_initial_confidence([], [{"seed": _KNOWN[0], "reward": 1.0}])
    stats = teacher.confidence_stats()
    assert stats["uncited"] == 1 and stats["citations"] == 0


# ── 5. persistence round trip ────────────────────────────────────────────
#
# The bug these guard: `StepRecord.to_dict` does not store `raw_text`, and
# `InstanceRun.n_unparsed` reads it. A rehydrated run whose `raw_text` came back
# empty would report zero unparsed turns no matter what the model actually emitted
# -- and `unparsed_rate` is the diagnostic that decides whether a success rate is
# meaningful at all. So the reconstruction has to be exact, not approximate.

def test_step_record_round_trip_reconstructs_raw_text():
    from hexa_orbit.actor import StepRecord

    step = StepRecord(
        episode_index=1, episode_step=3, total_step=12, observation="obs",
        raw_text="ANALYSISfinal \\boxed{up}", reasoning_text="ANALYSIS",
        final_text="final \\boxed{up}", parsed_action="up",
        env_action="\\boxed{up}", reward=1.0, finish_reason="stop",
        episode_done=True, usage={"total_tokens": 7},
    )
    revived = StepRecord.from_dict(step.to_dict())
    assert revived.raw_text == step.raw_text          # reasoning + final, exactly
    assert revived == step


def test_step_record_honours_a_stored_raw_text():
    """An older file that does carry raw_text is used as-is, not recomputed."""
    from hexa_orbit.actor import StepRecord

    data = {
        "episode_index": 0, "episode_step": 1, "total_step": 1, "observation": "o",
        "raw_text": "SOMETHING ELSE", "reasoning_text": "a", "final_text": "b",
        "parsed_action": "up", "env_action": "\\boxed{up}", "reward": 0.0,
        "finish_reason": "stop", "episode_done": False, "usage": None,
    }
    assert StepRecord.from_dict(data).raw_text == "SOMETHING ELSE"


def test_unparsed_count_survives_a_round_trip(task, spec):
    """Unparsed turns must not vanish on reload."""
    actor = MazeActor(client=StubClient(default=gen("I have no idea")))
    run = actor.run_instance(task.instance(spec))
    assert run.n_unparsed == len(run.steps) > 0

    revived = type(run).from_dict(run.to_dict())
    assert revived.n_unparsed == run.n_unparsed
    assert revived.n_truncated == run.n_truncated
    assert revived.adapted_across_episodes == run.adapted_across_episodes


def test_action_in_the_reasoning_channel_is_not_counted_as_unparsed_on_reload(task, spec):
    """The dangerous direction: a working run must not reload as a broken one.

    When the model spends its budget on reasoning and emits the action there, the
    final channel is empty but ``raw_text`` still carries the ``\boxed{}``. If the
    reconstruction were dropped, ``raw_text`` would come back empty and every one
    of these turns would be counted unparsed -- inflating ``unparsed_rate``, the
    very diagnostic used to decide whether a run's success rate is trustworthy.
    """
    actor = MazeActor(
        client=StubClient(default=gen("", reasoning=f"thinking {oracle.boxed('up')}"))
    )
    run = actor.run_instance(task.instance(spec))
    assert run.n_unparsed == 0, "the action is in raw_text, so nothing is unparsed"

    revived = type(run).from_dict(run.to_dict())
    assert revived.n_unparsed == 0
    assert all(s.raw_text for s in revived.steps)


def test_derived_fields_are_recomputed_not_trusted(task, spec):
    """A stale stored value must not survive; the steps are the source of truth."""
    actor = MazeActor(client=StubClient(default=gen("no action here")))
    run = actor.run_instance(task.instance(spec))
    data = run.to_dict()
    data["n_unparsed"] = 0            # lie about it
    data["adapted_across_episodes"] = True
    revived = type(run).from_dict(data)
    assert revived.n_unparsed == run.n_unparsed > 0
    assert revived.adapted_across_episodes is False


def test_guidelines_hash_records_the_bank_without_storing_it(task, spec):
    """Provenance without bloat: which bank produced these numbers?"""
    block = "LEARNED STRATEGIES\n- Expand away from the start."
    plain = MazeActor(client=StubClient(default=gen(oracle.boxed("up"))))
    banked = MazeActor(client=StubClient(default=gen(oracle.boxed("up"))), guidelines=block)
    other = MazeActor(
        client=StubClient(default=gen(oracle.boxed("up"))), guidelines=block + " More."
    )

    no_bank = plain.run_instance(task.instance(spec))
    with_bank = banked.run_instance(task.instance(spec))
    with_other = other.run_instance(task.instance(spec))

    assert no_bank.guidelines_hash is None
    assert with_bank.guidelines_hash and len(with_bank.guidelines_hash) == 16
    assert with_bank.guidelines_hash != with_other.guidelines_hash
    # The block text itself is never in the per-instance record -- it lives once in
    # the round's injection.json.
    assert block not in json.dumps(with_bank.to_dict())
    assert with_bank.to_dict()["guidelines_hash"] == with_bank.guidelines_hash
    # And it does not survive the round trip, deliberately: nothing downstream of a
    # completed run reads it.
    assert type(with_bank).from_dict(with_bank.to_dict()).guidelines is None


def test_write_then_load_gives_back_the_same_trajectories(task, solved_run, stuck_run, tmp_path):
    from hexa_orbit import trajectory as traj_mod

    traj_mod.write_runs(tmp_path / "runs.jsonl", [solved_run, stuck_run])
    successes, failures = traj_mod.load_instance_trajectories(tmp_path)

    assert len(successes) == 1 and len(failures) == 1
    assert successes[0]["seed"] == solved_run.instance_id
    assert successes[0]["reward"] == traj_mod.instance_trajectory(solved_run)["reward"]
    assert failures[0]["adapted_across_episodes"] is False


def test_errored_runs_are_not_loaded_as_evidence(task, spec, solved_run, tmp_path):
    """A transport failure is evidence of nothing and must never reach the teacher."""
    import dataclasses

    from hexa_orbit import trajectory as traj_mod

    errored = dataclasses.replace(solved_run, error="GenerationFailed: boom")
    traj_mod.write_runs(tmp_path / "runs.jsonl", [errored])
    assert traj_mod.load_instance_trajectories(tmp_path) == ([], [])


def test_task_name_filter_and_missing_directory(solved_run, tmp_path):
    from hexa_orbit import trajectory as traj_mod

    traj_mod.write_runs(tmp_path / "runs.jsonl", [solved_run])
    assert traj_mod.load_instance_trajectories(tmp_path, "maze")[0]
    assert traj_mod.load_instance_trajectories(tmp_path, "mastermind") == ([], [])
    assert traj_mod.load_instance_trajectories(tmp_path / "absent") == ([], [])


def test_partial_and_ordered_files_are_not_both_loaded(task, solved_run, stuck_run, tmp_path):
    """A round dir holds runs.jsonl AND runs.partial.jsonl with the same runs.

    Globbing `*.jsonl` loaded every instance twice, so the evolver's fixed number of
    slots could be spent on duplicate transcripts of the same maze.
    """
    from hexa_orbit import trajectory as traj_mod

    import dataclasses

    from hexa_orbit import trajectory as traj_mod

    # The two fixtures share a spec, so give one a distinct id -- otherwise
    # identical ids are legitimate and the uniqueness check proves nothing.
    other = dataclasses.replace(stuck_run, instance_id="distinct000000")
    traj_mod.write_runs(tmp_path / "runs.jsonl", [solved_run, other])
    traj_mod.write_runs(tmp_path / "runs.partial.jsonl", [solved_run, other])

    successes, failures = traj_mod.load_instance_trajectories(tmp_path)
    assert len(successes) + len(failures) == 2, "trajectories loaded more than once"
    ids = [t["seed"] for t in successes + failures]
    assert len(ids) == len(set(ids)) == 2


def test_partial_file_is_used_when_the_ordered_one_is_missing(task, solved_run, tmp_path):
    """A round killed before writing runs.jsonl should still be evolvable."""
    from hexa_orbit import trajectory as traj_mod

    traj_mod.write_runs(tmp_path / "runs.partial.jsonl", [solved_run])
    successes, failures = traj_mod.load_instance_trajectories(tmp_path)
    assert len(successes) + len(failures) == 1


# ── selection: the evolver must see cross-episode learners ────────────────

def _traj(first_success, reward_value, seed):
    return {"seed": seed, "first_success_episode": first_success,
            "reward": reward_value, "success": first_success is not None,
            "adapted_across_episodes": True, "trajectory": [],
            "episode_successes": [], "episode_lengths": [], "shortest_path_len": 7}


def test_reward_bin_selection_always_includes_slow_successes():
    """The class `balanced` excluded: failed episode 1, then solved.

    That is the cross-episode adaptation Design B distills, and sorting successes
    by reward descending buries it under every +1.00 fast success.
    """
    trajectories = (
        [_traj(1, 1.0, f"fast{i}") for i in range(10)]      # plenty of fast successes
        + [_traj(2, 0.6, "slow0"), _traj(3, 0.3, "slow1")]  # only two slow ones
        + [_traj(None, -0.5, f"fail{i}") for i in range(8)]
    )
    chosen = teacher.select_by_reward_bin(
        trajectories, 6, lambda t: t["reward"]
    )
    kinds = [reward.reward_bin(t) for t in chosen]
    assert len(chosen) == 6
    assert kinds.count("slow success") == 2, kinds     # both, despite lowest reward
    assert "fast success" in kinds and "exploratory failure" in kinds


def test_reward_bin_selection_redistributes_unfilled_quota():
    """A round with no slow successes still sends a full block."""
    trajectories = [_traj(1, 1.0, f"f{i}") for i in range(4)] + [
        _traj(None, -0.5, f"x{i}") for i in range(4)
    ]
    chosen = teacher.select_by_reward_bin(trajectories, 6, lambda t: t["reward"])
    assert len(chosen) == 6


def test_reward_bin_selection_handles_a_tiny_round():
    chosen = teacher.select_by_reward_bin(
        [_traj(1, 1.0, "a"), _traj(None, -0.5, "b")], 6, lambda t: t["reward"]
    )
    assert len(chosen) == 2


# ── iteration 2: the three fixes ─────────────────────────────────────────

def test_graded_failure_reward_is_opt_in_and_ordered():
    """Off by default: a run in flight must not have its rewards silently rescored."""
    from hexa_orbit import reward as R

    assert R.GRADED_FAILURES[0] is False
    values = [R.graded_failure_reward(p) for p in (0.0, 0.5, 1.0)]
    assert values == sorted(values)                    # more progress scores higher
    assert all(-1.0 <= v <= 1.0 for v in values)
    # Every failure stays strictly below the worst success, or the ordering that the
    # whole scale encodes would break.
    assert max(values) < R.FIRST_SUCCESS_REWARD[3]


def test_graded_failures_discriminate_where_bins_could_not(task, spec):
    """The measured problem: 46% of instances all scored exactly -0.5."""
    from hexa_orbit import reward as R

    near = dict(first_success_episode=None, episode_1_steps=9, shortest_path_len=7,
                adapted=True, best_progress=0.9)
    stuck = {**near, "best_progress": 0.0}
    assert R.reward_from_parts(**near) > R.reward_from_parts(**stuck)
    # Without progress supplied, both collapse to the old single value.
    old_near = {k: v for k, v in near.items() if k != "best_progress"}
    old_stuck = {k: v for k, v in stuck.items() if k != "best_progress"}
    assert R.reward_from_parts(**old_near) == R.reward_from_parts(**old_stuck) == -0.5


def test_progress_is_measured_per_episode(task, spec):
    """1.0 on success; a partial fraction on failure; never a signal the agent sees."""
    instance = task.instance(spec)
    instance.reset()
    for action in oracle.instance_optimal_actions(instance):
        _, _, done, _ = instance.step(oracle.boxed(action))
        if done:
            break
    assert instance.episodes[0].progress == 1.0

    stuck = task.instance(spec)
    stuck.reset()
    while not stuck.episodes:
        stuck.step(oracle.boxed("up"))          # bang the same wall
    assert stuck.episodes[0].progress == 0.0    # never left the start cell


def test_prompts_quote_the_actor_prompt_and_forbid_restating_it():
    """Three of five iteration-1 skills paraphrased the agent's own instructions."""
    from hexa_orbit.adapter import ORBIT_SYSTEM_PROMPT_MULTI

    for name in ("CONTRASTIVE_DISTILLATION_PROMPT_MAZE",
                 "COMMON_MISTAKES_PROMPT_MAZE",
                 "SKILL_BANK_EVOLUTION_PROMPT_MAZE"):
        prompt = getattr(teacher, name)
        rendered = prompt.format(**{
            k: "x" for k in
            ("level_name", "level_description", "n_successes", "n_failures",
             "success_block", "failure_block", "existing_skills_block",
             "new_trajectories_block", "max_skills", "max_mistakes")
            if "{" + k + "}" in prompt
        })
        assert "Leverage information gathered from earlier episodes" in rendered, name
        # Pinned to the *mechanism*, not to a sentence: the restatement guard was
        # a banned-phrase list, which v2 satisfied by rewording (five skills all
        # saying "use episode 1's exploration" differently). It is now a test the
        # teacher applies, so assert the test is stated and has its escape hatch.
        assert "already know to do this" in rendered, name
        assert "mechanically" in rendered, name
        # The quoted prompt must survive brace-escaping intact.
        assert "\\boxed{}" in rendered, name


def test_install_v2_turns_on_graded_failures_together():
    from hexa_orbit import reward as R

    teacher.install(reward_bin_selection=True)
    assert R.GRADED_FAILURES[0] is True
    teacher.install(reward_bin_selection=False)
    assert R.GRADED_FAILURES[0] is False


def test_seeding_path_gets_the_outcome_class_selector():
    """`distill` holds its own reference to the block; rebinding must reach it.

    ``skillrl.distillation.distill`` does a module-level ``from teacher_prompts
    import format_trajectories_block``, so ``install`` rebinding only ``_tp`` and
    ``_ev`` left round-0 seeding on the stock reward-descending selector -- the
    exact bias the v2 fix exists to remove, still live in the one call that builds
    the bank everything downstream refines.
    """
    import skillrl.distillation.distill as _distill
    import skillrl.distillation.evolving_distill as _ev
    import skillrl.distillation.teacher_prompts as _tp

    stock = _tp.format_trajectories_block
    teacher.install(reward_bin_selection=True)
    try:
        for module in (_distill, _ev, _tp):
            assert module.format_trajectories_block is not stock, module.__name__
    finally:
        teacher.install(reward_bin_selection=False)


def test_seeding_prefers_slow_successes_in_a_successes_only_call():
    """The seeding path calls the block with successes alone, then failures alone.

    Under fixed thirds a successes-only call reserved a slot for the `failed`
    class, which cannot appear, and redistribution handed it to a fast solve.
    Measured on v2 round 0: six cross-episode learners available, one sent.
    """
    successes = [_traj(1, 1.0, f"fast{i}") for i in range(4)] + [
        _traj(2, 0.6, "slow0"), _traj(3, 0.3, "slow1"), _traj(3, 0.3, "slow2"),
        _traj(3, 0.3, "slow3"), _traj(3, 0.3, "slow4"), _traj(3, 0.3, "slow5"),
    ]
    chosen = teacher.select_by_reward_bin(successes, 5, lambda t: t["reward"])
    kinds = [reward.reward_bin(t) for t in chosen]
    assert len(chosen) == 5
    # Weights are slow-heavy again (see `_PRIORITY`), so a successes-only call
    # renormalises to 3 slow / 2 fast. What the test guards is unchanged: the call
    # reserves nothing for the `failed` class, which cannot appear in it -- the
    # original bug, which sent one slow success out of six available.
    assert kinds.count("slow success") == 3, kinds
    assert kinds.count("fast success") == 2, kinds


def test_apportionment_sums_exactly_and_skips_empty_classes():
    from hexa_orbit.teacher import _apportion

    weights = {"slow": 0.5, "fast": 0.25, "failed": 0.25}
    all_three = {"slow": [1] * 9, "fast": [1] * 9, "failed": [1] * 9}
    for total in range(1, 13):
        share = _apportion(total, all_three, weights)
        assert sum(share.values()) == total, (total, share)
        # Every present class represented once there are slots to go round.
        if total >= 3:
            assert all(v >= 1 for v in share.values()), (total, share)

    # An empty class gets nothing and its weight is redistributed, not wasted.
    share = _apportion(5, {"slow": [1] * 9, "fast": [1] * 9, "failed": []}, weights)
    assert share == {"slow": 3, "fast": 2, "failed": 0}, share
    assert _apportion(4, {"slow": [], "fast": [], "failed": []}, weights) == {
        "slow": 0, "fast": 0, "failed": 0
    }


def test_evolution_block_keeps_six_while_seeding_honours_its_caller():
    """`evolving_distill` hardcodes max_trajs=5 at the call site.

    So the evolution binding must *override* the caller to reach the 6 of
    DESIGN_PHASE3_MAZE §12, while the seeding binding must *not* -- distill's 5+5
    is what its prompt budgets for. One shared closure cannot do both.
    """
    import skillrl.distillation.distill as _distill
    import skillrl.distillation.evolving_distill as _ev

    trajectories = (
        [_traj(1, 1.0, f"fast{i}") for i in range(6)]
        + [_traj(3, 0.3, f"slow{i}") for i in range(6)]
        + [_traj(None, -0.3, f"fail{i}") for i in range(6)]
    )
    teacher.install(reward_bin_selection=True)
    try:
        _ev.format_trajectories_block(trajectories, max_trajs=5)
        assert len(_tp_last()) == 6, "evolution must override the hardcoded 5"
        _distill.format_trajectories_block(trajectories, max_trajs=5)
        assert len(_tp_last()) == 5, "seeding must honour its caller"
    finally:
        teacher.install(reward_bin_selection=False)


def _tp_last():
    import skillrl.distillation.teacher_prompts as _tp

    return _tp.last_selection()


def test_seeding_log_reports_what_was_actually_sent():
    """distill used to re-derive the sort for its log line, so the log and the
    prompt disagreed the moment the block was rebound. It now reads the record."""
    import skillrl.distillation.distill as _distill
    import skillrl.distillation.teacher_prompts as _tp

    source = pathlib.Path(_distill.__file__).read_text()
    assert "last_selection()" in source
    assert "sorted(successes, key=lambda t: compute_trajectory_reward(t)" not in source

    chosen = [_traj(3, 0.3, "slow0"), _traj(1, 1.0, "fast0")]
    _tp.record_selection(chosen)
    assert [t["seed"] for t in _tp.last_selection()] == ["slow0", "fast0"]


def test_failures_are_not_starved_when_they_are_the_majority_class():
    """The v2 round-1 regression, pinned.

    Real pools were slow=2, fast=4, failed=14 -- 70% of the round unsolved -- and
    the block sent 2 slow / 3 fast / 1 failure. One failure out of fourteen, while
    the least informative class took half the block. Fixing v1's "never sees a slow
    success" had created "barely sees a failure".
    """
    pool = (
        [_traj(2, 0.6, f"slow{i}") for i in range(2)]
        + [_traj(1, 1.0, f"fast{i}") for i in range(4)]
        + [_traj(None, -0.3, f"fail{i}") for i in range(14)]
    )
    kinds = [reward.reward_bin(t) for t in
             teacher.select_by_reward_bin(pool, 6, lambda t: t["reward"])]
    # Slow-heavy weights: 2 slow (only 2 exist), 3 failures, 1 fast. The guard is
    # unchanged -- failures are the majority class here (14 of 20) and must not be
    # reduced to a token single example.
    assert kinds.count("exploratory failure") == 3, kinds
    assert kinds.count("slow success") == 2, kinds
    assert kinds.count("fast success") == 1, kinds


def test_unfilled_quota_goes_to_a_starved_class_not_the_top_reward():
    """Redistribution used to sort leftovers by reward descending, which hands
    every spare slot to a +1.00 fast success by construction."""
    pool = (
        [_traj(3, 0.3, "slow0")]                              # slow quota is 2, has 1
        + [_traj(1, 1.0, f"fast{i}") for i in range(9)]
        + [_traj(None, -0.3, f"fail{i}") for i in range(9)]
    )
    kinds = [reward.reward_bin(t) for t in
             teacher.select_by_reward_bin(pool, 6, lambda t: t["reward"])]
    assert kinds.count("slow success") == 1, kinds
    # The spare slot is handed out in `_PRIORITY` order, NOT by reward -- sorting
    # leftovers by reward gives every spare to a +1.00 fast success by
    # construction, which is the bug this pins. Failures must still be represented.
    assert kinds.count("exploratory failure") >= 2, kinds
    assert len(kinds) == 6, kinds


def test_selector_favours_the_class_that_drives_episode_three():
    """Slow-heavy, because episode 3 / solved-at-all is the endpoint.

    This constant has been set both ways and the reversal is the point. Fast-first
    (v5) was argued from episode 1 being the pre-registered claim; episode-1
    accuracy did not move under it, and rounds 1-2 of v3/v4/v5 produced identical
    episode-1 counts under three different banks. Slow-heavy is what v3 ran, and v3
    is the only run with a measured effect: ep3 +6.2, solved-at-all +9.5 (p=0.013),
    concentrated in the clean instances.
    """
    from hexa_orbit.teacher import _CLASS_WEIGHTS, _PRIORITY, _apportion

    assert _PRIORITY[0] == "slow"
    assert _CLASS_WEIGHTS["slow"] == _CLASS_WEIGHTS["failed"] > _CLASS_WEIGHTS["fast"]
    full = {"slow": [1] * 9, "fast": [1] * 9, "failed": [1] * 9}
    assert _apportion(6, full, _CLASS_WEIGHTS) == {"slow": 3, "fast": 1, "failed": 2}
    for total in range(3, 13):
        share = _apportion(total, full, _CLASS_WEIGHTS)
        # Failures are the majority class on this env and must never be starved
        # below the fast successes -- the v2 regression this originally caught.
        assert share["failed"] >= share["fast"], (total, share)
def test_evolver_reports_where_skills_stop_retracing():
    """Retargeted with rule 8: the reported field must match the one asked for.

    The log reported `first_contact` after the schema had moved to
    `departs_from_earlier_path`, so it silently reported nothing -- the same class
    of stale-key bug as reading `n_skills` off a `RetrievedBlock`.
    """
    src = pathlib.Path("skillrl/distillation/evolving_distill.py").read_text()
    assert "departs_from_earlier_path" in src
    assert "no trigger named" in src
    assert "first_contact" not in src


def test_contrastive_aggregate_splits_by_outcome_class():
    """Pooled numbers cannot answer "what separated the solves from the failures".

    The teacher was shown "84 of 416 cell-entries were repeats" and "31 turns
    produced no parseable move" every round and banked neither, writing instead
    about three-walled cells -- re-entered 5 times in 2605 turns. Episode-1 success
    is nearly determined by wasted turns (79% at none, 48% at one, 1% at two), and
    a pooled block hides that the classes differ on exactly those rows.
    """
    from hexa_orbit import observed

    def _rec(first, ever, revisits):
        traj = [{"episode": 1, "turn": i + 1, "action": "up",
                 "position": (1, 1 if i >= revisits else i + 2),
                 "result": "moved", "observation": ""} for i in range(5)]
        return {"episode_successes": [first, ever, ever], "trajectory": traj}

    records = [_rec(True, True, 0), _rec(True, True, 0),
               _rec(False, True, 2), _rec(False, False, 3)]
    by_class = observed.aggregate_by_class(records)
    assert set(by_class) == {"fast success", "slow success", "failure"}
    assert by_class["fast success"]["n_instances"] == 2

    text = observed.format_contrastive_for_teacher(records)
    assert "HOW THE OUTCOME CLASSES DIFFER" in text
    for row in ("cells re-entered per episode", "turns with no parseable move"):
        assert row in text, row
    # The pooled block is kept underneath, not replaced.
    assert "WHAT THE AGENT OBSERVED" in text


def test_contrastive_aggregate_falls_back_when_one_class_is_present():
    """A successes-only call has nothing to contrast; pooled is correct there."""
    from hexa_orbit import observed

    same = [{"episode_successes": [True, True, True],
             "trajectory": [{"episode": 1, "turn": 1, "action": "up",
                             "position": (1, 1), "result": "GOAL",
                             "observation": ""}]}] * 3
    text = observed.format_contrastive_for_teacher(same)
    assert "HOW THE OUTCOME CLASSES DIFFER" not in text
    assert "WHAT THE AGENT OBSERVED" in text


def test_contrastive_rates_are_per_episode_not_raw_counts():
    """Raw counts make the largest class the worst offender by construction."""
    from hexa_orbit import observed

    # Behavioural, not a docstring check: two classes with the same per-episode
    # rate but different sizes must report the same number. Asserting only the
    # comment let a raw-count mutation pass.
    def _rec(first, revisits):
        traj = [{"episode": 1, "turn": i + 1, "action": "up",
                 "position": (1, 1 if i >= revisits else i + 2),
                 "result": "moved", "observation": ""} for i in range(5)]
        return {"episode_successes": [first, first, first], "trajectory": traj}

    # One fast instance and three failures, all with the identical revisit pattern.
    records = [_rec(True, 2)] + [_rec(False, 2) for _ in range(3)]
    by_class = observed.aggregate_by_class(records)
    fast = observed._rate(by_class["fast success"], "revisits")
    fail = observed._rate(by_class["failure"], "revisits")
    assert fast == fail, (fast, fail)   # raw counts would give 3x here
    # Every compared row must come from the enumerated quantity list, so no single
    # statistic is promoted for being the interesting one.
    keys = {key for _, key, _ in observed._COMPARED}
    enumerated = {name for name, _ in observed.QUANTITIES}
    known = enumerated | {"revisits", "unparsed_turns", "blocked_moves",
                          "cells_visited", "net_displacement", "turns_per_episode"}
    assert keys <= known, keys - known


# ── restored: tests lost to an over-wide edit, rewritten ─────────────────

def test_addition_is_forbidden_only_when_the_bank_is_full():
    """Rule 3 must not ban addition outright.

    The prohibition it used to end on -- "If no existing skill is weaker than the
    candidate, add nothing." -- was written for a *full* bank but stated
    unconditionally. Measured cost on v2: three rounds of evolution added zero
    skills while the bank fell 9 -> 6 -> 5 against a cap of 10.
    """
    from hexa_orbit.teacher import SKILL_BANK_EVOLUTION_PROMPT_MAZE as prompt

    rule3 = prompt[prompt.index("3. **Add**"):prompt.index("4. **Remove**")]
    assert "add nothing" not in rule3.lower(), rule3
    at_cap = rule3[rule3.index("**At the cap**"):]
    assert "leave the bank as it is" in at_cap
    assert "leave the bank as it is" not in rule3[:rule3.index("**At the cap**")]
    below = rule3[rule3.index("**Below the cap"):rule3.index("**At the cap**")]
    assert "displaces nothing" in below
    assert rule3.index("**Below the cap") < rule3.index("**At the cap**")


def test_observed_never_reads_a_privileged_field():
    """A skill built on generator metadata is ours, not learned."""
    src = pathlib.Path("hexa_orbit/observed.py").read_text()
    body = re.sub(r'"""(?:.|\n)*?"""', "", src)
    for forbidden in ("shortest_path_len", "MazeGenerator", "maze_env_adapter"):
        assert forbidden not in body, forbidden


def test_teacher_transcript_no_longer_quotes_the_known_map_optimum():
    """`(shortest possible: N)` was a privileged channel to regularity #2."""
    src = pathlib.Path("hexa_orbit/teacher.py").read_text()
    assert "shortest possible: {optimum}" not in src
    assert 'trajectory.get("shortest_path_len")' not in src

    from hexa_orbit import teacher as t

    rendered = t.format_trajectory_for_teacher({
        "seed": "abc", "first_success_episode": 1, "success_count": 3,
        "episode_successes": [True], "episode_lengths": [8],
        "shortest_path_len": 7, "reward": 1.0,
        "trajectory": [{"episode": 1, "turn": 1, "action": "up",
                        "position": (1, 1), "result": "moved"}],
    })
    assert "shortest possible" not in rendered
    assert "SOLVED in 8 turns" in rendered


def test_observed_counts_what_the_agent_saw():
    from hexa_orbit import observed

    record = {"trajectory": [
        {"episode": 1, "turn": 1, "action": "up", "position": (1, 1),
         "result": "moved", "observation": "up leads to path, left leads to wall"},
        {"episode": 1, "turn": 2, "action": "left", "position": (1, 2),
         "result": "invalid", "observation": "left leads to wall"},
        {"episode": 1, "turn": 3, "action": "up", "position": (1, 2),
         "result": "GOAL", "observation": "up leads to goal"},
        {"episode": 2, "turn": 1, "action": "left", "position": (1, 1),
         "result": "moved", "observation": "left leads to path"},
    ]}
    a = observed.aggregate([record])
    assert a["n_instances"] == 1
    assert a["episodes_solved"] == 1 and a["episodes_total"] == 2
    assert a["blocked_moves"] == 1
    assert a["action_counts"] == {"up": 2, "left": 2}
    assert a["surroundings_seen"]["wall"] == 2
    assert a["goal_sighting_turn"][3] == 1
    assert a["first_divergence_turn"][1] == 1
    assert a["revisits"] == 1


def test_observed_ignores_reasoning_text_masquerading_as_an_action():
    """`parsed_action` holds the raw response on an unparseable turn."""
    from hexa_orbit import observed

    record = {"steps": [
        {"episode_index": 0, "episode_step": 1, "parsed_action": "up",
         "observation": "position (1, 1)", "reward": 0.0},
        {"episode_index": 0, "episode_step": 2,
         "parsed_action": "Okay, let me think about which way to go here...",
         "observation": "position (1, 2)", "reward": 0.0},
    ]}
    a = observed.aggregate([record])
    assert a["action_counts"] == {"up": 1}
    assert a["unparsed_turns"] == 1


def test_observed_filters_non_moves_on_the_normalised_path_too():
    """Two independent filters guard the histogram; pin each separately."""
    from hexa_orbit import observed

    record = {"trajectory": [
        {"episode": 1, "turn": 1, "action": "up", "position": (1, 1),
         "result": "moved", "observation": ""},
        {"episode": 1, "turn": 2,
         "action": "Okay, so the agent is at (1,1) and needs to decide...",
         "position": (1, 2), "result": "moved", "observation": ""},
    ]}
    a = observed.aggregate([record])
    assert a["action_counts"] == {"up": 1}, a["action_counts"]
    assert a["unparsed_turns"] == 1


def test_evolution_prompt_asks_for_invariants_and_gates_on_a_cited_turn():
    from hexa_orbit.teacher import SKILL_BANK_EVOLUTION_PROMPT_MAZE as prompt

    text = prompt.format(
        level_name="maze", level_description="d", existing_skills_block="B",
        n_successes=8, n_failures=12, new_trajectories_block="T",
        max_skills=10, max_mistakes=5,
    )
    assert text.index("what is invariant here") < text.index("Rules:")
    assert '"invariants"' in text
    assert "changes_action" in text
    assert "cannot be tied to a specific turn" in text


def test_already_known_is_a_test_with_an_escape_hatch_not_a_phrase_ban():
    """A phrase list is satisfiable by rewording, which is what v2 did."""
    from hexa_orbit.teacher import _ALREADY_KNOWN

    assert "do NOT bank skills whose content reduces to" not in _ALREADY_KNOWN
    assert "does an agent that has read the prompt above already know" in _ALREADY_KNOWN
    assert "admissible" in _ALREADY_KNOWN
    assert "mechanically" in _ALREADY_KNOWN
def test_evolver_reports_invariant_coverage():
    src = pathlib.Path("skillrl/distillation/evolving_distill.py").read_text()
    assert "acts_on_invariant" in src
    assert "NO SKILL" in src
    assert "key in c.lower()" in src


def test_reserve_pool_is_held_out_from_both_working_pools():
    from hexa_orbit import pools

    assert pools.RESERVE_GLOBAL_SEED != pools.DEFAULT_GLOBAL_SEED
    cfg = pools.published_maze_config()
    train = pools.build_pool(cfg, 512, is_train=True)
    ev = pools.build_pool(cfg, pools.EVAL_POOL_SIZE, is_train=False)
    reserve = pools.build_reserve_pool(cfg)
    assert len(reserve) == pools.EVAL_POOL_SIZE
    pools.assert_disjoint(train, reserve)
    pools.assert_disjoint(ev, reserve)


def test_train_pool_is_deduplicated_by_layout_and_eval_is_not():
    """512 seeds hold only 459 distinct mazes; v2 died on the tenth round."""
    from hexa_orbit import pools
    from hexa_orbit.adapter import MazeTask

    task = MazeTask()
    train = task.train_specs(512)
    assert len({s.maze_hash for s in train}) == len(train)
    raw = task.train_specs(512, dedup_layouts=False)
    assert len(raw) == 512 and len({s.maze_hash for s in raw}) < 512
    ev = task.eval_specs(256)
    assert len(ev) == 256
    assert len({s.maze_hash for s in ev}) < 256
    pools.assert_disjoint(train, ev)


def test_episode_one_only_sees_skills_it_can_act_on():
    from hexa_orbit import loop

    class _Item:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    reuse = _Item(principle="Use Episode 1 data to avoid dead ends",
                  when_to_apply="in later episodes", example="")
    fresh = _Item(principle="Record walls seen at each cell",
                  when_to_apply="on encountering a wall", example="")
    assert loop.requires_prior_episode(reuse) is True
    assert loop.requires_prior_episode(fresh) is False


def test_actor_swaps_the_block_when_the_episode_advances():
    from hexa_orbit.actor import MazeActor

    actor = MazeActor(guidelines={0: "EXPLORE BLOCK", 1: "REUSE BLOCK"})
    assert "EXPLORE BLOCK" in actor.system_prompt_for(0)
    assert "REUSE BLOCK" in actor.system_prompt_for(1)
    assert "REUSE BLOCK" in actor.system_prompt_for(2)

    plain = MazeActor(guidelines="ONE BLOCK")
    assert plain.system_prompt_for(0) == plain.system_prompt_for(2)
    assert plain.system_prompt == plain.system_prompt_for(0)

    bare = MazeActor()
    assert bare.system_prompt_for(0) == bare.base_system_prompt


def test_guidelines_hash_survives_a_per_episode_mapping():
    """A dict has no `.encode`; the un-fixed version raises once per instance."""
    from hexa_orbit.actor import InstanceRun

    def _hash(value):
        run = InstanceRun.__new__(InstanceRun)
        run.guidelines = value
        return InstanceRun.guidelines_hash.fget(run)

    assert _hash(None) is None
    assert _hash("ONE BLOCK") is not None
    mapped = _hash({0: "A", 1: "B"})
    assert mapped is not None and len(mapped) == 16
    assert _hash({1: "B", 0: "A"}) == mapped
    assert _hash({0: "A", 1: "C"}) != mapped


def test_split_block_is_smaller_in_episode_one_than_later():
    """Per-episode injection stays available even though v6 does not use it."""
    from hexa_core.tokens import count_tokens
    from skillrl.core.skill_bank import SkillBank
    from hexa_orbit import loop

    path = pathlib.Path("results/orbit_maze_hexa_v3/round9/bank.json")
    if not path.exists():  # pragma: no cover - depends on run artefacts
        pytest.skip("v3 bank not present")
    blocks = loop.render_blocks_by_episode(SkillBank.load(path))
    assert set(blocks) == {0, 1}
    assert count_tokens(blocks[0]) < count_tokens(blocks[1])
    assert "Episode 1 data" not in blocks[0]
    assert "prior exploration" not in blocks[0]


# ── admission gating, retargeted to the episode-3 endpoint (v7) ──────────

def test_admission_trials_run_whole_instances_not_single_episodes():
    """A single-episode trial cannot express a cross-episode skill.

    `run_trial` drives `instance.new_episode()`, which has no earlier episode, so
    "stop retracing your episode-1 route" scores like a null skill there. With
    episode 3 as the endpoint that gate would reject exactly the skills the
    measured driver points at, so trials must play the whole instance.
    """
    from hexa_orbit import admission

    assert hasattr(admission, "run_instance_trial")
    src = pathlib.Path("hexa_orbit/loop.py").read_text()
    assert "whole_instance=True" in src


def test_admission_compares_against_a_no_bank_control():
    """`passes > 0` at this base rate admits ~2/3 of candidates on luck."""
    from hexa_orbit import admission

    assert hasattr(admission, "control_mean_score")
    assert admission.DEFAULT_ADMISSION_MARGIN > 0, "a tie must not admit"
    src = pathlib.Path("hexa_orbit/admission.py").read_text()
    assert "control_mean + margin" in src


def test_loop_binds_the_admission_module_it_calls():
    """The enabled path is never exercised by default -- pin the binding.

    `admission_mod.control_mean_score` is only reached with `--admission`, so an
    unbound name here passes every test and fails on the first round of a gated
    run.
    """
    from hexa_orbit import loop

    assert hasattr(loop, "admission_mod")
    assert callable(loop.admission_mod.control_mean_score)


def test_optional_teacher_fields_reject_the_string_null():
    """The model writes "null" as a string; a truthy check flatters the report.

    Measured in v6 round 1: the log said 4/4 skills named a departure trigger when
    exactly 1 did, because `"null"` is a non-empty string. Any report over an
    optional field needs this guard.
    """
    src = pathlib.Path("skillrl/distillation/evolving_distill.py").read_text()
    assert 'text not in ("", "null", "none", "n/a", "na", "-")' in src
    assert "if any(_named(d) for _, d in departures)" in src
    assert "if _named(acts)" in src


def test_evolution_prompt_is_back_to_the_v3_shape():
    """v6 added rules 7-8 and a contrastive pairing section and lost 10 points.

    Measured over the banked rounds: v6 ep3 0.408 (n=98) against v3's 0.594
    (n=180) and a no-bank baseline of 0.506 -- v6 landed *below* the baseline. The
    teacher did read the extra evidence (it stated "agents re-enter cells 2.2 times
    per episode" as an invariant) and banked nothing from it, while the prompt grew
    from ~8.2k to ~11.5k characters. v7 isolates the admission gate against the one
    configuration that has ever beaten baseline, so the prompt must match v3's.
    """
    from hexa_orbit.teacher import SKILL_BANK_EVOLUTION_PROMPT_MAZE as prompt

    text = prompt.format(
        level_name="maze", level_description="d", existing_skills_block="B",
        n_successes=8, n_failures=12, new_trajectories_block="T",
        max_skills=10, max_mistakes=5,
    )
    # Kept -- all three were in v3 and v3 worked with them.
    assert "what is invariant here" in text
    assert "changes_action" in text
    assert "Below the cap" in text
    # Added after v3, and reverted.
    for gone in ("7. **Close the loop", "8. **A skill about a later episode",
                 "acts_on_invariant", "departs_from_earlier_path",
                 "what separated the solves"):
        assert gone not in text, gone
    # v7 adds rules 7-8 (two-way confidence, keep the mechanism) from the catapult
    # truncation postmortem. Both are evidenced in the banked maze rounds -- see
    # `test_evolution_prompt_carries_the_postmortem_rules` -- so the budget moves,
    # but only by what they cost. The guard remains: v6's failure was 3.4k chars of
    # instructions the teacher banked nothing from, and nothing may be added here
    # without an equivalent measurement.
    assert len(text) < 9800, f"prompt is {len(text)} chars; v3 was ~8135, v6 ~11500"


def test_evolution_prompt_carries_the_postmortem_rules():
    """The two fixes the catapult truncation postmortem left to the prompt.

    Maze banks show both failure modes. Confidence is a one-way ratchet:
    `orbit_maze_hexa_qwen8_16k/run0` holds three skills pinned at 0.85/0.82/0.78 for
    nine consecutive rounds, moving only upward (0.78 -> 0.82), while confidence is
    the retriever's ranking key. And principles are compressed on every rewrite:
    round 0 (seeded, no bank in the prompt) averages 168 chars; from round 1, the
    first round fed back through the teacher, every round averages 100-110 and the
    "record which side of which cell" mechanism thins to a restatement of the title.

    Rules 9-10 of the postmortem (ban unfalsifiable skills, ban commit-gating
    skills) are deliberately NOT ported: a scan of 287 unique skills across 80 maze
    banks found no "this maze may be unsolvable" hedge and no budget-burning gate,
    and `_ALREADY_KNOWN` already covers the "adapt to the situation" class.
    """
    from hexa_orbit.teacher import SKILL_BANK_EVOLUTION_PROMPT_MAZE as prompt

    text = prompt.format(
        level_name="maze", level_description="d", existing_skills_block="B",
        n_successes=8, n_failures=12, new_trajectories_block="T",
        max_skills=10, max_mistakes=5,
    )
    assert "Confidence moves in both directions" in text
    assert "least 0.1" in text  # wrapped: "lower it by at\n   least 0.1"
    assert "confidence_changes" in text
    assert "keeping its mechanism" in text
    # Rule 1 must not re-freeze what rule 7 unfreezes.
    assert "preserving their confidence" not in text


def test_evolver_block_uses_the_pooled_aggregate_not_the_contrastive_one():
    """The contrastive split stays in `observed` but is not wired for v7."""
    src = pathlib.Path("hexa_orbit/teacher.py").read_text()
    assert "_obs.format_for_teacher(trajectories)" in src
    assert "_obs.format_contrastive_for_teacher(trajectories)" not in src

    from hexa_orbit import observed
    assert hasattr(observed, "format_contrastive_for_teacher")


# ── citation verification ────────────────────────────────────────────────

def _traj_with(seed, actions):
    return {"seed": seed, "trajectory": [
        {"episode": 1, "turn": i + 1, "action": a} for i, a in enumerate(actions)
    ]}


def test_a_false_citation_is_rejected():
    """14 of 24 turn citations across all maze runs (58%) were fabricated.

    Rule 6 asks the teacher to name the turn a skill would have changed. It was an
    honour system: e.g. "Instance 49289a818872a592, turn 1: agent chose 'right'"
    when the agent chose `up`. The transcripts are in memory when the evolver runs,
    so the claim is checkable.
    """
    from hexa_orbit import citations

    trajs = [_traj_with("49289a818872a592", ["up", "left", "right"])]
    ok, reason = citations.verify(
        "Instance 49289a818872a592, turn 1: agent chose 'right'", trajs)
    assert ok is False and reason.startswith("FALSE"), reason

    ok, reason = citations.verify(
        "Instance 49289a818872a592, turn 1: agent chose 'up'", trajs)
    assert ok is True and reason.startswith("verified"), reason


def test_the_counterfactual_action_is_not_checked_against_the_transcript():
    """"agent chose 'X' vs skill would choose 'Y'" -- only X is a claim of fact.

    Y is by construction what the agent did *not* do, so matching it against the
    transcript would reject every correct citation.
    """
    from hexa_orbit import citations

    c = citations.parse_citation(
        "Instance abcdef0123456789, turn 2: agent chose 'left' vs skill would choose 'down'")
    assert c.action == "left"
    trajs = [_traj_with("abcdef0123456789", ["up", "left"])]
    ok, _ = citations.verify(
        "Instance abcdef0123456789, turn 2: agent chose 'left' vs skill would choose 'down'",
        trajs)
    assert ok is True


def test_unverifiable_citations_are_kept_not_dropped():
    """Only demonstrably false claims may delete a skill.

    Rejecting whatever cannot be parsed would empty the bank on any formatting
    change in the teacher's output, and would remove skills that may be good.
    """
    from hexa_orbit import citations

    trajs = [_traj_with("abcdef0123456789", ["up"])]
    for text in (None, "", "somewhere in episode 1", "Instance ffffffffffffffff, turn 1: agent chose 'up'"):
        ok, reason = citations.verify(text, trajs)
        assert ok is True, (text, reason)
        assert reason.startswith("unverifiable"), reason


def test_a_turn_beyond_the_episode_is_false_not_unverifiable():
    """Citing turn 9 of a 3-turn episode is a fabrication, not a parse failure."""
    from hexa_orbit import citations

    trajs = [_traj_with("abcdef0123456789", ["up", "left", "right"])]
    ok, reason = citations.verify(
        "Instance abcdef0123456789, turn 9: agent chose 'up'", trajs)
    assert ok is False and "no turn 9" in reason, reason


def test_filter_drops_only_the_fabricated_skill():
    from hexa_orbit import citations

    trajs = [_traj_with("abcdef0123456789", ["up", "left"])]
    skills = [
        {"title": "good", "changes_action": "Instance abcdef0123456789, turn 2: agent chose 'left'"},
        {"title": "bad", "changes_action": "Instance abcdef0123456789, turn 2: agent chose 'right'"},
        {"title": "uncited", "changes_action": None},
    ]
    kept, notes = citations.filter_fabricated(skills, trajs)
    assert [s["title"] for s in kept] == ["good", "uncited"]
    assert sum(1 for _, r in notes if r.startswith("FALSE")) == 1


def test_verifier_is_installed_and_wired_into_the_evolver():
    import skillrl.distillation.evolving_distill as ev

    teacher.install(reward_bin_selection=True)
    assert ev._CITATION_VERIFIER is not None
    src = pathlib.Path("skillrl/distillation/evolving_distill.py").read_text()
    assert "Dropped" in src and "fabricated evidence" in src


def test_admission_trials_run_concurrently():
    """The gate ran one call at a time while the round used a 12-way pool.

    Measured on v7: ~950 sequential calls for one round's gate (~2h15m) against
    ~35 minutes for the round itself, which would have blown the 6h walltime around
    round 2 with no eval produced.
    """
    from hexa_orbit import admission

    import time

    # Behavioural, not a source check: asserting "ThreadPoolExecutor" appears in
    # the file passes even when the parallel branch is dead, which is exactly what
    # a forced-serial mutation does. Time it instead, with a wide margin so this
    # is not flaky -- 12 quarter-second tasks take 3s serially and ~0.25s at
    # concurrency 12.
    def _slow(i):
        time.sleep(0.2)
        return i

    started = time.monotonic()
    out = admission._run_parallel(
        [lambda i=i: _slow(i) for i in range(12)], concurrency=12
    )
    elapsed = time.monotonic() - started
    assert elapsed < 1.2, f"trials ran serially: {elapsed:.2f}s for 12 x 0.2s"
    # Order must survive parallelism, or scores attach to the wrong trial.
    assert out == list(range(12))
    # Serial fallback stays available for a single trial.
    assert admission._run_parallel([lambda: "x"], concurrency=1) == ["x"]

    loop_src = pathlib.Path("hexa_orbit/loop.py").read_text()
    assert loop_src.count("concurrency=concurrency") >= 2, "loop must pass it through"
