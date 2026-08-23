"""Checks for the HeXA<->CORE seam: attempt records, neutral prompts, patch, budget.

Run with::

    PYTHONDONTWRITEBYTECODE=1 python -m hexa_core.tests.test_hexa_seam

pytest is not installed here, so this is a plain script: bare ``assert``, printed
progress, non-zero exit on the first failure. It imports ``skillrl`` but never
CORE, makes no network calls and needs no GPU.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from string import Formatter

import skillrl.core.config as skill_config
import skillrl.distillation.distill as distill
import skillrl.distillation.evolving_distill as evolving_distill
import skillrl.distillation.skills_only_distill as skills_only_distill
import skillrl.distillation.teacher_prompts as teacher_prompts
from skillrl.core.retriever import retrieve_skills
from skillrl.core.skill_bank import Mistake, Skill, SkillBank
from skillrl.distillation.evolution_prompts import SKILL_BANK_EVOLUTION_PROMPT

from hexa_core import hexa_bridge
from hexa_core.tasks import TASK_NAMES
from hexa_core.hexa_bridge import (
    attempt_utility,
    binary_trajectory_reward,
    compute_initial_confidence_core,
    patch,
    retrieve_within_token_budget,
    unpatch,
)
from hexa_core.prompts_core_tasks import (
    CORE_TASK_DESCRIPTIONS,
    COMMON_MISTAKES_PROMPT_CORE,
    CONTRASTIVE_DISTILLATION_PROMPT_CORE,
    SKILL_BANK_EVOLUTION_PROMPT_CORE,
)
from hexa_core.tokens import counter_name
from hexa_core.trajectory import (
    Attempt,
    assign_rounds,
    attempt_to_trajectory,
    chunk_attempts,
    group_by_round,
    load_attempt_trajectories,
    read_attempts,
    round_summary,
    split_successes_failures,
    write_attempts,
)

TASK = "tower-of-hanoi"


def make_attempt(
    problem_id: str,
    correct: bool,
    *,
    round: int = 0,
    baseline: float | None = None,
    finish_reason: str = "stop",
) -> Attempt:
    answer = "moves = [[1, 0, 2], [2, 0, 1], [1, 2, 1]]"
    return Attempt(
        problem_id=problem_id,
        task_name=TASK,
        prompt="<CORE prompt>",
        raw_text=f"Reasoning for {problem_id}. " + answer,
        parsed_answer=answer,
        correct=correct,
        reward=1.0 if correct else 0.0,
        baseline=baseline,
        round=round,
        added_tokens=42,
        finish_reason=finish_reason,
    )


def check_trajectory_shape() -> None:
    a = make_attempt("hanoi_train_007", True)
    traj = attempt_to_trajectory(a)

    for key in ("seed", "success", "iterations", "trajectory", "action"):
        assert key in traj, f"missing HeXA contract key {key!r}"
    assert traj["seed"] == a.problem_id, "problem_id must stand in for HeXA's seed"
    assert traj["success"] is True
    assert traj["iterations"] == 1, "a CORE attempt is a one-step trajectory"
    assert len(traj["trajectory"]) == 1
    step = traj["trajectory"][0]
    assert set(step) == {"thought", "action", "observation"}, step
    assert step["thought"] == a.raw_text
    assert step["action"] == a.parsed_answer
    assert "CORRECT" in step["observation"] and "INCORRECT" not in step["observation"]
    assert traj["action"] == a.parsed_answer

    truncated = attempt_to_trajectory(make_attempt("p1", False, finish_reason="length"))
    assert truncated["truncated"] is True
    obs = truncated["trajectory"][0]["observation"]
    assert obs.startswith("INCORRECT") and "token limit" in obs, obs

    clipped = attempt_to_trajectory(make_attempt("p2", True), thought_max_chars=10)
    assert clipped["trajectory"][0]["thought"].endswith("[reasoning truncated]")
    print("[ok] attempt_to_trajectory emits HeXA's dict shape with one step")


def check_attempt_record() -> None:
    plain = make_attempt("p", True)
    assert not plain.has_baseline
    assert plain.utility == plain.reward, "utility must degrade to reward without b_q"

    with_baseline = make_attempt("p", True, baseline=0.18)
    assert with_baseline.has_baseline
    assert abs(with_baseline.utility - 0.82) < 1e-9, with_baseline.utility
    assert abs(make_attempt("p", False, baseline=0.68).utility + 0.68) < 1e-9

    assert Attempt.from_dict(with_baseline.to_dict()) == with_baseline
    assert Attempt.from_dict({**with_baseline.to_dict(), "future_field": 1}) == with_baseline

    attempts = [make_attempt(f"p{i}", i % 2 == 0) for i in range(6)]
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        path = Path(tmp) / "nested" / "round0.jsonl"
        write_attempts(path, attempts)
        assert read_attempts(path) == attempts, "jsonl round-trip changed the records"

        stamped = assign_rounds(attempts, 2, start_round=1)
        write_attempts(Path(tmp) / "nested" / "round1.jsonl", stamped)
        succ, fail = load_attempt_trajectories(path.parent, TASK)
        assert len(succ) == 6 and len(fail) == 6, (len(succ), len(fail))
        assert load_attempt_trajectories(path.parent, "matchstick") == ([], [])
        assert load_attempt_trajectories(Path(tmp) / "nope") == ([], [])

    assert [a.round for a in assign_rounds(attempts, 2)] == [0, 0, 1, 1, 2, 2]
    assert [len(c) for c in chunk_attempts(attempts, 4)] == [4, 2]
    assert list(group_by_round(assign_rounds(attempts, 2))) == [0, 1, 2]

    succ, fail = split_successes_failures(attempts)
    assert len(succ) == 3 and len(fail) == 3
    summary = round_summary(attempts)
    assert summary["n"] == 6 and summary["n_correct"] == 3
    assert summary["accuracy"] == 0.5 and summary["task_names"] == [TASK]
    assert round_summary([])["n"] == 0
    print("[ok] Attempt utility/round-trip and the M3 batch helpers")


def check_binary_reward() -> None:
    assert binary_trajectory_reward({"success": True}) == 1.0
    assert binary_trajectory_reward({"success": False}) == -1.0
    assert binary_trajectory_reward({}) == -1.0, "absent success must not read as True"
    # Iteration count is ignored — the stock reward would return 0.25 here.
    assert binary_trajectory_reward({"success": True, "iterations": 25}) == 1.0

    all_right = [attempt_to_trajectory(make_attempt(f"p{i}", True)) for i in range(3)]
    all_wrong = [attempt_to_trajectory(make_attempt(f"q{i}", False)) for i in range(3)]
    patch()
    try:
        hi = distill._compute_initial_confidence(["p0", "p1"], all_right)
        lo = distill._compute_initial_confidence(["q0", "q1"], all_wrong)
    finally:
        unpatch()
    assert hi == 1.0, hi
    assert abs(lo - 0.1) < 1e-9, lo
    # No `baseline` on these attempts, so M1 degrades to the binary reward -- which is
    # the pre-M1 behaviour on purpose, not a silent b_q = 0. See check_m1_confidence.
    print("[ok] binary reward: +/-1.0, and confidence maps to 1.0 / 0.1 through distill")


def check_m1_confidence() -> None:
    """M1: confidence must measure help-over-baseline, pooled over every attempt.

    Regression for two bugs that were live together. The stock function reads the
    binary outcome and keeps one trajectory per seed, so under CORE's k-attempts-per-
    problem regime a skill from a problem solved 3/10 and one from a problem solved
    1/10 both scored exactly 1.0 -- and confidence is the retrieval ranking key.
    """
    def pool(problem_id: str, n_correct: int, n_total: int, b_q: float) -> list[dict]:
        attempts = [
            make_attempt(problem_id, i < n_correct, baseline=b_q) for i in range(n_total)
        ]
        # Exactly how distill_contrastive orders them: successes first, then failures.
        trajectories = [attempt_to_trajectory(a) for a in attempts]
        return (
            [t for t in trajectories if t["success"]]
            + [t for t in trajectories if not t["success"]]
        )

    # -- the utility of one attempt ---------------------------------------------
    assert attempt_utility({"success": True, "baseline": 0.3}) == 0.7
    assert abs(attempt_utility({"success": False, "baseline": 0.3}) - (-0.3)) < 1e-9
    # Absent baseline degrades to binary, NOT to b_q = 0 (which would read as a
    # skill that beat a zero baseline).
    assert attempt_utility({"success": True}) == 1.0
    assert attempt_utility({"success": False}) == -1.0
    # The M1/M2 domain guard: never `reward - b_q`, which would give -1.5 here.
    for correct in (True, False):
        for b_q in (0.0, 0.5, 1.0):
            u = attempt_utility({"success": correct, "baseline": b_q})
            assert -1.0 <= u <= 1.0, f"utility {u} outside the map's domain"

    # -- pooling beats the first-occurrence collapse ----------------------------
    easy = compute_initial_confidence_core(["p"], pool("p", 3, 10, 0.3))
    hard = compute_initial_confidence_core(["p"], pool("p", 1, 10, 0.05))
    never = compute_initial_confidence_core(["p"], pool("p", 0, 10, 0.4))

    # Before the fix these were 1.0, 1.0, 0.1 -- the first two indistinguishable.
    assert easy != hard, f"confidence still collapsed: {easy} == {hard}"
    # Solved exactly at its own base rate => neutral 0.5, not 1.0.
    assert abs(easy - 0.5) < 1e-9, f"3/10 at b_q=0.3 should be neutral, got {easy}"
    # Solved above its base rate => above neutral, even though it looks "worse".
    assert hard > 0.5, f"1/10 at b_q=0.05 beat baseline, got {hard}"
    assert never < 0.5, f"0/10 at b_q=0.4 underperformed, got {never}"

    # -- base rates must not masquerade as skill quality ------------------------
    # Same help-over-baseline (+0.2) on tasks with very different base rates.
    matchstick = compute_initial_confidence_core(["m"], pool("m", 88, 100, 0.68))
    hanoi = compute_initial_confidence_core(["h"], pool("h", 38, 100, 0.18))
    assert abs(matchstick - hanoi) < 1e-9, (
        f"equal help scored unequally: matchstick {matchstick} vs hanoi {hanoi}"
    )

    # -- every attempt counts ---------------------------------------------------
    ten = pool("p", 5, 10, 0.5)
    assert len(ten) == 10, "fixture must actually hold 10 attempts"
    assert abs(compute_initial_confidence_core(["p"], ten) - 0.5) < 1e-9

    # An unnamed source falls back to the level-wide average, not to a constant.
    assert compute_initial_confidence_core(["absent"], ten) == (
        compute_initial_confidence_core(["p"], ten)
    )
    # Empty inputs return the caller's fallback, as the stock function does.
    assert compute_initial_confidence_core([], ten, fallback_confidence=0.42) == 0.42
    assert compute_initial_confidence_core(["p"], [], fallback_confidence=0.42) == 0.42

    print(
        "[ok] M1 confidence: pooled over all attempts, baseline-relative "
        f"(3/10@0.3 -> {easy}, 1/10@0.05 -> {hard:.3f}, 0/10@0.4 -> {never:.3f})"
    )


def check_prompt_parity() -> None:
    def fields(template: str) -> set[str]:
        return {name for _, name, _, _ in Formatter().parse(template) if name}

    pairs = [
        (teacher_prompts.CONTRASTIVE_DISTILLATION_PROMPT, CONTRASTIVE_DISTILLATION_PROMPT_CORE),
        (teacher_prompts.COMMON_MISTAKES_PROMPT, COMMON_MISTAKES_PROMPT_CORE),
        (SKILL_BANK_EVOLUTION_PROMPT, SKILL_BANK_EVOLUTION_PROMPT_CORE),
    ]
    for original, variant in pairs:
        assert fields(original) == fields(variant), (
            f"placeholder drift: {fields(variant) ^ fields(original)}"
        )

    rendered = [
        CONTRASTIVE_DISTILLATION_PROMPT_CORE.format(
            level_name=TASK,
            level_description=CORE_TASK_DESCRIPTIONS[TASK],
            n_successes=2,
            n_failures=3,
            success_block="S",
            failure_block="F",
        ),
        COMMON_MISTAKES_PROMPT_CORE.format(
            level_name=TASK,
            level_description=CORE_TASK_DESCRIPTIONS[TASK],
            failure_block="F",
        ),
        SKILL_BANK_EVOLUTION_PROMPT_CORE.format(
            level_name=TASK,
            level_description=CORE_TASK_DESCRIPTIONS[TASK],
            existing_skills_block="(none)",
            n_successes=2,
            n_failures=3,
            new_trajectories_block="T",
            max_skills=8,
            max_mistakes=4,
        ),
    ]
    for text in rendered:
        # .format() succeeding already proves the braces are escaped correctly;
        # this catches a placeholder that was renamed and so never substituted.
        assert "{level_name}" not in text and "{failure_block}" not in text
        assert "{" in text and "}" in text, "the JSON example block did not survive"
        lowered = text.lower()
        for banned in ("physics", "box2d", "gravity", "red ball", "iters"):
            assert banned not in lowered, f"leftover InterPhyre wording: {banned!r}"
        assert "source_seeds" in text

    assert '"mistakes"' in rendered[1] and '"partial_skills"' in rendered[1]
    for key in ('"is_new"', '"removed_skill_titles"', '"confidence"', '"reasoning"'):
        assert key in rendered[2], key

    # Tied to the registry rather than a literal set, so registering a task without
    # writing its description fails here instead of silently shipping a run whose
    # evolver was never told what the task is.
    assert set(CORE_TASK_DESCRIPTIONS) == set(TASK_NAMES), (
        f"description set != registry: missing {set(TASK_NAMES) - set(CORE_TASK_DESCRIPTIONS)}, "
        f"extra {set(CORE_TASK_DESCRIPTIONS) - set(TASK_NAMES)}"
    )
    collisions = set(CORE_TASK_DESCRIPTIONS) & set(skill_config.LEVEL_DESCRIPTIONS)
    assert not collisions, f"CORE task names collide with InterPhyre levels: {collisions}"
    print("[ok] neutral prompts: placeholder + schema parity, no InterPhyre wording")


def check_patch_roundtrip() -> None:
    originals = {
        (distill, "CONTRASTIVE_DISTILLATION_PROMPT"),
        (distill, "COMMON_MISTAKES_PROMPT"),
        (distill, "compute_trajectory_reward"),
        (distill, "load_trajectories"),
        (evolving_distill, "SKILL_BANK_EVOLUTION_PROMPT"),
        (evolving_distill, "load_trajectories"),
        (teacher_prompts, "compute_trajectory_reward"),
        (teacher_prompts, "format_trajectory_for_teacher"),
        (distill, "_compute_initial_confidence"),
        (evolving_distill, "_compute_initial_confidence"),
        (skills_only_distill, "_compute_initial_confidence"),
    }
    before = {(m, a): getattr(m, a) for m, a in originals}
    level_keys_before = set(skill_config.LEVEL_DESCRIPTIONS)

    unpatch()  # no-op when not patched
    assert not hexa_bridge.is_patched()

    patch(load_trajectories_fn=load_attempt_trajectories)
    patch()  # idempotent; must not clobber the snapshot with patched values
    assert hexa_bridge.is_patched()
    assert distill.CONTRASTIVE_DISTILLATION_PROMPT is CONTRASTIVE_DISTILLATION_PROMPT_CORE
    assert distill.COMMON_MISTAKES_PROMPT is COMMON_MISTAKES_PROMPT_CORE
    assert evolving_distill.SKILL_BANK_EVOLUTION_PROMPT is SKILL_BANK_EVOLUTION_PROMPT_CORE
    assert teacher_prompts.compute_trajectory_reward is binary_trajectory_reward
    assert distill.compute_trajectory_reward is binary_trajectory_reward
    # M1 must be bound in all three modules. `evolving_distill` and
    # `skills_only_distill` import the name directly, so a single rebinding of
    # `distill` would leave `evolve_skill_bank` on the stock function.
    for module in (distill, evolving_distill, skills_only_distill):
        assert module._compute_initial_confidence is compute_initial_confidence_core, (
            f"{module.__name__} still on the stock confidence function"
        )
    assert teacher_prompts.format_trajectory_for_teacher is (
        hexa_bridge.format_trajectory_for_teacher_core
    )
    assert distill.load_trajectories is load_attempt_trajectories
    assert evolving_distill.load_trajectories is load_attempt_trajectories
    for name, description in CORE_TASK_DESCRIPTIONS.items():
        assert skill_config.LEVEL_DESCRIPTIONS[name] == description

    unpatch()
    assert not hexa_bridge.is_patched()
    for (module, attr), value in before.items():
        assert getattr(module, attr) is value, f"{module.__name__}.{attr} not restored"
    assert set(skill_config.LEVEL_DESCRIPTIONS) == level_keys_before
    unpatch()  # still a no-op

    clash = {next(iter(level_keys_before)): "hijacked"}
    try:
        patch(task_descriptions=clash)
    except ValueError:
        pass
    else:
        unpatch()
        raise AssertionError("patch() overwrote an existing LEVEL_DESCRIPTIONS entry")
    assert not hexa_bridge.is_patched(), "a rejected patch left residue"
    print("[ok] patch/unpatch: idempotent, restores by identity, refuses key clashes")


def check_patched_formatting() -> None:
    trajs = [
        attempt_to_trajectory(make_attempt("hanoi_eval_001", False)),
        attempt_to_trajectory(make_attempt("hanoi_eval_002", True)),
    ]
    stock = teacher_prompts.format_trajectories_block(trajs)
    assert "x=m, y=o, r=v" in stock, (
        "expected the stock formatter to mangle a string action; if this ever stops "
        "being true the patch is no longer load-bearing"
    )

    patch()
    try:
        block = teacher_prompts.format_trajectories_block(trajs)
    finally:
        unpatch()

    assert "x=" not in block, block
    assert "Final answer: moves = [[1, 0, 2], [2, 0, 1], [1, 2, 1]]" in block
    assert block.index("hanoi_eval_002") < block.index("hanoi_eval_001"), (
        "successes must sort ahead of failures under the binary reward"
    )
    assert "Reward: +1.00" in block and "Reward: -1.00" in block
    assert f"Task: {TASK}" in block
    print("[ok] patched formatter prints the parsed answer and sorts by binary reward")


def build_bank(n_skills: int = 6, n_mistakes: int = 3) -> SkillBank:
    bank = SkillBank()
    for i in range(n_skills):
        bank.add_skill(Skill(
            skill_id=f"han_sk_{i:03d}",
            title=f"Strategy {i}",
            principle="Track the peg contents explicitly after every move so the "
                      "legality check never depends on remembering earlier moves. " * 2,
            when_to_apply="Whenever the goal configuration is not the classical form.",
            source_level=TASK,
            source_type="contrastive",
            source_seeds=[f"p{i}"],
            confidence=1.0 - 0.05 * i,
        ))
    for i in range(n_mistakes):
        bank.add_mistake(Mistake(
            mistake_id=f"han_err_{i:03d}",
            description=f"Mistake {i}: reports a move list that never reaches the goal.",
            why_it_happens="The model stops replaying the state partway through.",
            how_to_avoid="Replay the full list against the stated goal before answering.",
            source_level=TASK,
        ))
    return bank


def check_token_budget() -> None:
    bank = build_bank()

    generous = retrieve_within_token_budget(bank, TASK, 100_000)
    assert len(generous.skills) == 6 and len(generous.mistakes) == 3
    assert generous.tokens > 0
    retrieved_ids = {s.skill_id for s in retrieve_skills(bank, TASK, max_general=0, max_specific=8)}
    assert {s.skill_id for s in generous.skills} == retrieved_ids, (
        "membership must match the existing retriever, not a reimplementation"
    )
    assert generous.block_text == bank.format_skills_as_prompt(
        TASK, skills=generous.skills, mistakes=generous.mistakes
    ).replace("=== LEARNED PHYSICS SKILLS ===", hexa_bridge.SKILLS_BLOCK_HEADING).replace(
        "## Tower-Of-Hanoi-Specific Skills", f"## Task-Specific Strategies ({TASK})"
    )
    assert "PHYSICS" not in generous.block_text
    assert hexa_bridge.SKILLS_BLOCK_HEADING in generous.block_text
    assert not generous.budget_bound, "nothing was dropped, so this must be False"
    assert (generous.dropped_skills, generous.dropped_mistakes) == (0, 0)

    tight = retrieve_within_token_budget(bank, TASK, 200)
    assert 0 < tight.tokens <= 200, tight.tokens
    assert len(tight.skills) < 6, "a 200-token budget should not fit the whole bank"
    assert tight.block_text.count("Strategy ") == len(tight.skills), (
        "block_text and the returned skill list disagree"
    )
    # The truncation must be visible in the return value, not only inferable by
    # re-querying the bank: this is the one place a silent cap could hide.
    assert tight.budget_bound
    assert tight.dropped_skills == 6 - len(tight.skills)
    assert tight.dropped_mistakes == 3, (
        "the walk stopped on a skill, so every mistake was starved"
    )

    reserved = retrieve_within_token_budget(bank, TASK, 200, overhead_tokens=150)
    assert reserved.tokens + 150 <= 200
    assert len(reserved.skills) <= len(tight.skills)

    # One skill so long it alone blows the budget, ahead of everything else
    # because retrieval orders by confidence. Without skip_oversized the greedy
    # walk stops there and nothing at all is returned.
    blocked = build_bank(n_skills=1, n_mistakes=2)
    blocked.level_skills[TASK][0].principle *= 40
    starved = retrieve_within_token_budget(blocked, TASK, 300)
    forgiving = retrieve_within_token_budget(blocked, TASK, 300, skip_oversized=True)
    assert starved == ("", [], [], 0, 1, 2), starved
    assert not forgiving.skills and len(forgiving.mistakes) == 2, forgiving.mistakes
    assert 0 < forgiving.tokens <= 300
    assert (forgiving.dropped_skills, forgiving.dropped_mistakes) == (1, 0)

    unlabelled = retrieve_within_token_budget(bank, TASK, 100_000, relabel=False)
    assert "=== LEARNED PHYSICS SKILLS ===" in unlabelled.block_text

    # "the bank had nothing" and "the budget rejected everything" must not be the
    # same return value -- the first is normal, the second is a misconfiguration.
    assert retrieve_within_token_budget(SkillBank(), TASK, 1000) == ("", [], [], 0, 0, 0)
    assert retrieve_within_token_budget(None, TASK, 1000) == ("", [], [], 0, 0, 0)
    assert retrieve_within_token_budget(bank, TASK, 0) == ("", [], [], 0, 6, 3)
    assert retrieve_within_token_budget(bank, "matchstick", 1000) == ("", [], [], 0, 0, 0)
    print(
        f"[ok] retrieve_within_token_budget: {len(tight.skills)}/6 skills at 200 tokens, "
        f"{generous.tokens} tokens uncapped (counter={counter_name()})"
    )


def main() -> int:
    check_trajectory_shape()
    check_attempt_record()
    check_binary_reward()
    check_m1_confidence()
    check_prompt_parity()
    check_patch_roundtrip()
    check_patched_formatting()
    check_token_budget()
    print("\nall hexa seam checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
