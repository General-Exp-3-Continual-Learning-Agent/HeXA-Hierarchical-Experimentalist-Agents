"""Phase 3 items 1-4: trajectory dict, reward, teacher rendering, bank audit.

All pure -- no server, no model calls. The one test that touches ``skillrl`` is
:func:`test_install_rebinds_the_seams`, which is the point: the formatter and
prompts have to actually land on the modules ``distill`` reads them from, and a
docstring claiming so is not evidence.
"""

from __future__ import annotations

import json

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
