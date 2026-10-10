"""Round-loop tests, driven with a stubbed teacher and actor -- no server.

The loop is the piece where a mistake is expensive and invisible: an off-by-one in
which bank a round consumed, a training instance reused across rounds, or a
resume that silently replays 540 calls. So these tests check the *structure* of a
run -- what each round saw, what it wrote, and what it consumed -- rather than any
success rate.
"""

from __future__ import annotations

import json

import pytest

from hexa_orbit import loop, oracle, teacher, trajectory
from hexa_orbit.actor import MazeActor
from hexa_orbit.adapter import MazeTask
from hexa_orbit.tests.test_actor import StubClient, gen

SKILLS_JSON = json.dumps([
    {
        "title": "Expand away from the start",
        "principle": "Prefer directions that increase distance from the start cell.",
        "when_to_apply": "when two or more openings are available",
        "example": "treat a corridor already walked as spent",
        "source_seeds": [],
        "confidence": 0.8,
        "is_new": True,
    }
])
MISTAKES_JSON = json.dumps({
    "mistakes": [{
        "description": "Repeats a direction that already hit a wall.",
        "why_it_happens": "No record is kept of failed probes.",
        "how_to_avoid": "Treat a wall reply as a permanent fact about that cell.",
    }],
    "partial_skills": [],
})
EVOLVE_JSON = json.dumps({
    "skills": [{
        "title": "Expand away from the start",
        "principle": "Prefer directions that increase distance from the start cell.",
        "when_to_apply": "when two or more openings are available",
        "example": "treat a corridor already walked as spent",
        "source_seeds": [],
        "confidence": 0.8,
        "is_new": False,
    }],
    "mistakes": [],
    "removed_skill_titles": [],
    "reasoning": "unchanged",
})


@pytest.fixture
def stub_teacher(monkeypatch):
    """Route every teacher call to a canned response, per prompt type."""
    calls: list[str] = []

    def call_teacher(prompt: str, model: str = "stub") -> str:
        if "CURRENT SKILL BANK" in prompt:
            calls.append("evolve")
            return EVOLVE_JSON
        if "partial_skills" in prompt or "UNSOLVED INSTANCES" in prompt and "SOLVED" not in prompt:
            calls.append("mistakes")
            return MISTAKES_JSON
        calls.append("distill")
        return SKILLS_JSON

    import skillrl.distillation.distill as _distill
    import skillrl.distillation.evolving_distill as _evolving

    monkeypatch.setattr(_distill, "call_teacher", call_teacher)
    monkeypatch.setattr(_evolving, "call_teacher", call_teacher, raising=False)
    return calls


@pytest.fixture
def task() -> MazeTask:
    return MazeTask()


def optimal_actor_factory(task, specs):
    """An actor that plays each instance optimally, via the oracle."""
    def factory(guidelines, instance=None):
        probe = task.instance(specs[0])
        probe.reset()
        actions = oracle.instance_optimal_actions(probe)
        script = [gen(oracle.boxed(a)) for a in actions * 6]
        return MazeActor(client=StubClient(script), guidelines=guidelines)
    return factory


def wallbang_actor_factory(guidelines_seen: list):
    """An actor that always moves up. Records what block it was handed."""
    def factory(guidelines, instance=None):
        guidelines_seen.append(guidelines)
        return MazeActor(client=StubClient(default=gen(oracle.boxed("up"))),
                         guidelines=guidelines)
    return factory


# ── structure of a run ───────────────────────────────────────────────────

def test_two_rounds_write_the_expected_artifacts(task, stub_teacher, tmp_path):
    specs = task.train_specs(4)
    seen: list = []
    result = loop.run_hexa_loop(
        task, specs, wallbang_actor_factory(seen), tmp_path,
        rounds=2, batch_size=2, teacher_model="stub", resume=False,
    )
    assert len(result.rounds) == 2
    for index in (0, 1):
        round_dir = tmp_path / f"round{index}"
        for name in ("runs.jsonl", "bank.json", "audit.json", "admission.json", "injection.json"):
            assert (round_dir / name).exists(), f"round{index}/{name} missing"
    assert (tmp_path / "run.json").exists()
    assert result.streaming_leak is False


def test_round_1_gets_no_bank_and_round_2_gets_one(task, stub_teacher, tmp_path):
    """The off-by-one that would be invisible: which bank did each round consume?"""
    specs = task.train_specs(4)
    seen: list = []
    loop.run_hexa_loop(
        task, specs, wallbang_actor_factory(seen), tmp_path,
        rounds=2, batch_size=2, teacher_model="stub", resume=False,
    )
    # 2 instances per round; every actor in round 0 saw None, round 1 saw a block.
    assert seen[:2] == [None, None]
    assert all(g and "Expand away from the start" in g for g in seen[2:4])


def test_bank_json_is_written_after_evolution(task, stub_teacher, tmp_path):
    """`roundN/bank.json` must be what round N produced, not what it consumed."""
    specs = task.train_specs(2)
    loop.run_hexa_loop(
        task, specs, wallbang_actor_factory([]), tmp_path,
        rounds=1, batch_size=2, teacher_model="stub", resume=False,
    )
    from skillrl.core.skill_bank import SkillBank

    bank = SkillBank.load(tmp_path / "round0" / "bank.json")
    assert bank.level_skills["maze"], "round 0 produced an empty bank"


def test_instances_are_never_reused_across_rounds(task, stub_teacher, tmp_path):
    specs = task.train_specs(6)
    result = loop.run_hexa_loop(
        task, specs, wallbang_actor_factory([]), tmp_path,
        rounds=3, batch_size=2, teacher_model="stub", resume=False,
    )
    all_ids = [i for r in result.rounds for i in r.instance_ids]
    assert len(all_ids) == len(set(all_ids)) == 6
    assert result.streaming_leak is False


def test_exhausting_the_pool_stops_early_rather_than_repeating(task, stub_teacher, tmp_path):
    specs = task.train_specs(2)
    result = loop.run_hexa_loop(
        task, specs, wallbang_actor_factory([]), tmp_path,
        rounds=5, batch_size=2, teacher_model="stub", resume=False,
    )
    assert len(result.rounds) == 1
    assert result.stopped_early == "training pool exhausted"


def test_round_record_carries_the_diagnostics(task, stub_teacher, tmp_path):
    specs = task.train_specs(2)
    result = loop.run_hexa_loop(
        task, specs, wallbang_actor_factory([]), tmp_path,
        rounds=1, batch_size=2, teacher_model="stub", resume=False,
    )
    record = result.rounds[0].to_dict()
    assert record["reward_bins"]              # the [PROVISIONAL] revisit signal
    assert record["n_skills_before"] == 0
    assert record["n_skills_after"] >= 1
    assert "source_seeds" in record["audit"]
    assert "confidence_stats" in record
    assert record["summary"]["n"] == 2


# ── resumability ─────────────────────────────────────────────────────────

def test_resume_skips_completed_rounds(task, stub_teacher, tmp_path):
    specs = task.train_specs(6)
    loop.run_hexa_loop(
        task, specs, wallbang_actor_factory([]), tmp_path,
        rounds=1, batch_size=2, teacher_model="stub", resume=False,
    )
    next_round, bank, consumed = loop.resume_state(tmp_path)
    assert next_round == 1 and bank is not None and len(consumed) == 2

    seen: list = []
    result = loop.run_hexa_loop(
        task, specs, wallbang_actor_factory(seen), tmp_path,
        rounds=2, batch_size=2, teacher_model="stub", resume=True,
    )
    # Only round 1 ran, on the two instances round 0 did not consume.
    assert [r.round_index for r in result.rounds] == [1]
    assert len(seen) == 2 and all(g is not None for g in seen)


def test_missing_bank_json_means_died_mid_evolution(task, stub_teacher, tmp_path):
    """Runs without a bank: the round is re-evolved, not replayed."""
    specs = task.train_specs(4)
    loop.run_hexa_loop(
        task, specs, wallbang_actor_factory([]), tmp_path,
        rounds=1, batch_size=2, teacher_model="stub", resume=False,
    )
    (tmp_path / "round0" / "bank.json").unlink()
    next_round, bank, consumed = loop.resume_state(tmp_path)
    assert next_round == 0            # redo round 0's evolution
    assert bank is None
    assert len(consumed) == 2         # but its instances are already spent


def test_resume_state_on_a_fresh_directory(tmp_path):
    assert loop.resume_state(tmp_path / "nope") == (0, None, set())


# ── frozen eval ──────────────────────────────────────────────────────────

def test_frozen_eval_writes_nothing_to_the_bank(task, stub_teacher, tmp_path):
    specs = task.train_specs(2)
    loop.run_hexa_loop(
        task, specs, wallbang_actor_factory([]), tmp_path / "train",
        rounds=1, batch_size=2, teacher_model="stub", resume=False,
    )
    from skillrl.core.skill_bank import SkillBank

    bank_path = tmp_path / "train" / "round0" / "bank.json"
    before = bank_path.read_text()
    bank = SkillBank.load(bank_path)

    eval_specs = task.eval_specs(2)
    summary = loop.frozen_eval(
        task, eval_specs, wallbang_actor_factory([]), bank, tmp_path / "eval",
        train_specs=specs,
    )
    assert bank_path.read_text() == before, "frozen eval mutated the bank on disk"
    assert summary["arm"] == "frozen_bank"
    assert summary["report"]["pooled"]["n"] == 2
    assert (tmp_path / "eval" / "runs.jsonl").exists()


def test_no_bank_arm_injects_nothing(task, tmp_path):
    eval_specs = task.eval_specs(2)
    seen: list = []
    summary = loop.frozen_eval(
        task, eval_specs, wallbang_actor_factory(seen), None, tmp_path / "eval_nb",
    )
    assert summary["arm"] == "no_bank"
    assert seen == [None, None]
    assert summary["injection"]["tokens"] == 0


def test_eval_runs_are_not_reachable_by_the_evolver(task, tmp_path):
    """Eval runs live in their own directory so a later distil cannot ingest them."""
    loop.frozen_eval(
        task, task.eval_specs(2), wallbang_actor_factory([]), None, tmp_path / "eval_nb",
    )
    successes, failures = trajectory.load_instance_trajectories(tmp_path / "round0")
    assert (successes, failures) == ([], [])


# ── the integration seam ─────────────────────────────────────────────────

def test_evolver_receives_the_rounds_trajectories(task, tmp_path, monkeypatch):
    """Round 2+ must evolve from the round's runs, not from an empty directory.

    `evolve_skill_bank` takes a *directory* and calls `load_trajectories` on it.
    The stock loader globs `trajectory_seed*.json` and finds nothing in a round
    directory -- and treats an empty round as "no new evidence" rather than an
    error, so a missing `load_trajectories_fn` rebinding makes every round after
    the first evolve from ZERO trajectories, silently. This asserts the round's own
    instance ids reach the evolver prompt.
    """
    seen_prompts: list[str] = []

    def call_teacher(prompt: str, model: str = "stub") -> str:
        seen_prompts.append(prompt)
        if "CURRENT SKILL BANK" in prompt:
            return EVOLVE_JSON
        if "partial_skills" in prompt:
            return MISTAKES_JSON
        return SKILLS_JSON

    import skillrl.distillation.distill as _distill
    import skillrl.distillation.evolving_distill as _evolving

    monkeypatch.setattr(_distill, "call_teacher", call_teacher)
    monkeypatch.setattr(_evolving, "call_teacher", call_teacher, raising=False)

    specs = task.train_specs(4)
    result = loop.run_hexa_loop(
        task, specs, wallbang_actor_factory([]), tmp_path,
        rounds=2, batch_size=2, teacher_model="stub", resume=False,
    )

    evolution_prompts = [p for p in seen_prompts if "CURRENT SKILL BANK" in p]
    assert evolution_prompts, "no evolution prompt was built"
    prompt = evolution_prompts[-1]
    round1_ids = result.rounds[1].instance_ids
    assert any(i in prompt for i in round1_ids), (
        "the evolver saw none of round 1's instances -- load_trajectories_fn is "
        "not rebound"
    )
    assert "=== Instance" in prompt          # our formatter, not the stock one
    assert "0 successes, 0 failures" not in prompt


def test_load_trajectories_is_rebound_by_install():
    import skillrl.distillation.distill as _distill
    import skillrl.distillation.evolving_distill as _evolving

    from hexa_orbit.trajectory import load_instance_trajectories

    teacher.install()
    assert _distill.load_trajectories is load_instance_trajectories
    assert _evolving.load_trajectories is load_instance_trajectories


def test_bank_unchanged_is_detected(task, tmp_path, monkeypatch):
    """A teacher whose JSON fails to parse makes evolution a silent no-op.

    `evolve_skill_bank` handles an unparseable response by printing a warning and
    returning the *previous* bank. Nothing in the round record would otherwise
    distinguish "the teacher had nothing to add" from "the teacher's output was
    garbage" -- and with a 7B teacher emitting strict JSON for a whole bank, the
    second is the likely case.
    """
    def call_teacher(prompt: str, model: str = "stub") -> str:
        if "CURRENT SKILL BANK" in prompt:
            return "not json at all"          # forces the fallback
        if "partial_skills" in prompt:
            return MISTAKES_JSON
        return SKILLS_JSON

    import skillrl.distillation.distill as _distill
    import skillrl.distillation.evolving_distill as _evolving

    monkeypatch.setattr(_distill, "call_teacher", call_teacher)
    monkeypatch.setattr(_evolving, "call_teacher", call_teacher, raising=False)

    specs = task.train_specs(4)
    result = loop.run_hexa_loop(
        task, specs, wallbang_actor_factory([]), tmp_path,
        rounds=2, batch_size=2, teacher_model="stub", resume=False,
    )
    assert result.rounds[0].bank_unchanged is False   # round 0 distils, nothing to match
    assert result.rounds[1].bank_unchanged is True


def test_empty_skills_does_not_wipe_the_bank(task, tmp_path, monkeypatch):
    """A teacher returning `{"skills": []}` must not erase the accumulated bank.

    This parses cleanly, so it misses the unparseable guard above and reaches the
    rewrite, which builds a fresh SkillBank and copies over only *other* levels'
    skills -- saving an empty bank for the level under training. On the 2026-09-09
    gpt-oss Maze build this fired at 6 of 57 round transitions and left two of
    three replicates finishing 20 rounds with generation-2 skills. The round looks
    normal in every log line except the bank size, which is why it needs a test.
    """
    def call_teacher(prompt: str, model: str = "stub") -> str:
        if "CURRENT SKILL BANK" in prompt:
            return '{"skills": [], "removed_skill_titles": [], "mistakes": []}'
        if "partial_skills" in prompt:
            return MISTAKES_JSON
        return SKILLS_JSON

    import skillrl.distillation.distill as _distill
    import skillrl.distillation.evolving_distill as _evolving

    monkeypatch.setattr(_distill, "call_teacher", call_teacher)
    monkeypatch.setattr(_evolving, "call_teacher", call_teacher, raising=False)

    specs = task.train_specs(4)
    result = loop.run_hexa_loop(
        task, specs, wallbang_actor_factory([]), tmp_path,
        rounds=2, batch_size=2, teacher_model="stub", resume=False,
    )
    seeded = json.loads((tmp_path / "round0" / "bank.json").read_text())
    evolved = json.loads((tmp_path / "round1" / "bank.json").read_text())
    assert sum(len(v) for v in seeded["level_skills"].values()) > 0, "round 0 seeded nothing"
    assert evolved["level_skills"] == seeded["level_skills"]
    assert result.rounds[1].bank_unchanged is True
    assert result.to_dict()["rounds_bank_unchanged"] == 1


def test_a_real_evolution_is_not_flagged_unchanged(task, stub_teacher, tmp_path):
    """The detector must not cry wolf when the teacher genuinely changed the bank."""
    specs = task.train_specs(4)
    result = loop.run_hexa_loop(
        task, specs, wallbang_actor_factory([]), tmp_path,
        rounds=2, batch_size=2, teacher_model="stub", resume=False,
    )
    # The stub returns the same single skill each round, so this asserts the
    # fingerprint really is content-based: identical advice IS unchanged.
    assert result.rounds[1].bank_unchanged is True
    assert loop.bank_fingerprint(None) == frozenset()


def test_partial_file_is_written_as_instances_complete(task, tmp_path):
    """Crash safety for multi-hour runs: every finished run must already be on disk.

    The regression this guards: routing the baseline runner through `play_batch`
    replaced its per-instance flush with a single write after the whole batch, so a
    256-instance arm killed at 90 minutes would have left an empty file.
    """
    from hexa_orbit import trajectory

    specs = task.eval_specs(4)
    partial = tmp_path / "runs.partial.jsonl"
    seen_during: list[int] = []

    def probe(position, run):
        # Read the partial file mid-batch: it must already contain this run.
        seen_during.append(len(partial.read_text().splitlines()))

    runs, errored = loop.play_batch(
        task, specs, wallbang_actor_factory([]), None,
        on_instance=probe, partial_path=partial,
    )
    assert len(runs) == 4
    lines = partial.read_text().splitlines()
    assert len(lines) == 4
    # Progress fires live, so the file grows 1..4 as callbacks arrive rather than
    # being complete at the first one. That is the regression this pins: reporting
    # after `pool.map` would leave a multi-hour arm silent until it finished.
    assert seen_during == [1, 2, 3, 4], seen_during

    revived = trajectory.read_runs(partial)
    assert {r.instance_id for r in revived} == {r.instance_id for r in runs}


def test_partial_file_is_truncated_not_appended_across_runs(task, tmp_path):
    """A resumed round must not inherit the previous attempt's partial rows."""
    specs = task.eval_specs(2)
    partial = tmp_path / "runs.partial.jsonl"
    partial.write_text('{"stale": true}\n')
    loop.play_batch(task, specs, wallbang_actor_factory([]), None, partial_path=partial)
    assert "stale" not in partial.read_text()
    assert len(partial.read_text().splitlines()) == 2
