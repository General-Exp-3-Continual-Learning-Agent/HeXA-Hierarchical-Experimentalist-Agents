"""The Design C source-task teacher path: rendering, loading, installation.

Covers the seam that killed a Maze run and a Mastermind run respectively: a
trajectory loader bound to the wrong level, and prompts that fail their own
``.format`` at the moment the evolver calls them (which surfaces as zero skills
and a printed warning, not an exception).
"""

from __future__ import annotations

import json

import pytest

from hexa_orbit import source_teacher
from hexa_orbit.actor import InstanceRun, StepRecord
from hexa_orbit.source_trajectory import (
    instance_trajectory,
    loader_for,
    minesweeper_outcome,
    wordle_outcome,
)

FORMAT_KWARGS = dict(
    level_name="L", level_description="D", n_successes=1, n_failures=2,
    success_block="S", failure_block="F", existing_skills_block="E",
    new_trajectories_block="N", max_skills=8, max_mistakes=4,
)


def _step(episode, step, total, observation, action):
    return StepRecord(
        episode_index=episode, episode_step=step, total_step=total,
        observation=observation, raw_text=f"\\boxed{{{action}}}", reasoning_text="",
        final_text=f"\\boxed{{{action}}}", parsed_action=action,
        env_action=f"\\boxed{{{action}}}", reward=0.0, finish_reason="stop",
        episode_done=False,
    )


def _wordle_run() -> InstanceRun:
    obs1 = "[Episode 1] New episode begins.\nYou are playing Wordle."
    obs2 = "At turn 1, you guessed CRANE\nFeedback:\nCRANE\nXYGXX"
    obs3 = "At turn 2, you guessed SLOTH\nFeedback:\nSLOTH\nXXGXX"
    return InstanceRun(
        instance_id="wd0", dataset_seed=1, resolved_seed=1, shortest_path_len=1,
        leaked=False,
        steps=[_step(0, 1, 1, obs1, "CRANE"), _step(0, 2, 2, obs2, "SLOTH")],
        messages=[{"role": "user", "content": obs3}],
        metrics={}, episode_successes=[False], episode_lengths=[2], success_count=0,
        added_tokens=0, token_counter="tiktoken",
        episode_progress=[0.2],
    )


# ── outcome extraction ─────────────────────────────────────────────────

def test_wordle_result_is_anchored_to_its_own_turn():
    """At an episode boundary the observation holds the finished episode's last
    feedback *and* the next episode's opening; matching blind attributes the wrong
    turn. Mastermind hit this exact bug."""
    boundary = ("At turn 9, you guessed SLOTH\nFeedback:\nSLOTH\nXXGXX\n\n"
                "[Episode 2] New episode begins.\n"
                "At turn 1, you guessed CRANE\nFeedback:\nCRANE\nGGGGG")
    assert wordle_outcome(boundary, 9) == "XXGXX"


def test_minesweeper_result_reports_how_much_the_reveal_opened():
    prev = "Here is the current board layout:\n. . . . .\n. . . . .\nEnter your guess."
    cur = ("At turn 3, you successfully revealed cell (1, 4).Here is the current "
           "board layout:\n1 . . . .\n. . . . .\nEnter your guess.")
    assert minesweeper_outcome(cur, 3, prev) == "safe, opened 1 cell(s), 9 still hidden"


@pytest.mark.parametrize("text,expected", [
    ("Game over! You hit a mine at (2, 3)", "HIT A MINE"),
    ("Congratulations! You have successfully cleared the Minesweeper board.", "CLEARED"),
    ("At turn 4, you chose to reveal cell (1, 1), which has already been revealed.",
     "wasted (already revealed)"),
    ("At turn 2, you did not provide a valid guess.", "unparsed -- episode ended"),
])
def test_minesweeper_terminal_outcomes(text, expected):
    assert minesweeper_outcome(text) == expected


@pytest.mark.parametrize("text,expected", [
    ("Congratulations! You guessed the secret word YOUNG in 3 turns.", "SOLVED"),
    ("At turn 3, you guessed CRANE, which has been already guessed before.",
     "wasted (repeat of an earlier guess)"),
])
def test_wordle_terminal_outcomes(text, expected):
    assert wordle_outcome(text) == expected


# ── trajectory dicts ───────────────────────────────────────────────────

def test_trajectory_has_the_shape_the_shared_distiller_expects():
    trajectory = instance_trajectory(_wordle_run(), "wordle")
    for key in ("seed", "success", "trajectory", "reward", "task_name",
                "episode_successes", "first_success_episode", "best_progress",
                "adapted_across_episodes", "num_episodes"):
        assert key in trajectory
    assert trajectory["task_name"] == "wordle"
    assert trajectory["success"] is False
    assert [t["result"] for t in trajectory["trajectory"]] == ["XYGXX", "XXGXX"]


def test_loader_reads_runs_jsonl_only(tmp_path):
    """A round directory also holds runs.partial.jsonl with the same runs;
    globbing both doubled every trajectory in an earlier Maze run."""
    round_dir = tmp_path / "round0"
    round_dir.mkdir()
    record = json.dumps(_wordle_run().to_dict())
    (round_dir / "runs.jsonl").write_text(record + "\n")
    (round_dir / "runs.partial.jsonl").write_text(record + "\n")

    successes, failures = loader_for("wordle")(round_dir)
    assert len(successes) + len(failures) == 1
    assert failures[0]["task_name"] == "wordle"


def test_loader_is_quiet_and_empty_when_the_round_has_no_runs(tmp_path):
    assert loader_for("minesweeper")(tmp_path) == ([], [])


# ── prompts ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("level", source_teacher.LEVELS)
def test_every_prompt_survives_the_format_call_the_distiller_makes(level):
    for prompt in (source_teacher.contrastive_prompt(level),
                   source_teacher.mistakes_prompt(level),
                   source_teacher.evolution_prompt(level)):
        rendered = prompt.format(**FORMAT_KWARGS)
        for placeholder in FORMAT_KWARGS:
            assert "{" + placeholder + "}" not in rendered


@pytest.mark.parametrize("level", source_teacher.LEVELS)
def test_prompts_carry_the_anti_memorisation_clause(level):
    """A source bank that memorises its instances does not merely fail to
    transfer -- Phase 3 re-grounds it into a fabricated target skill."""
    for prompt in (source_teacher.contrastive_prompt(level),
                   source_teacher.evolution_prompt(level)):
        assert "is not a skill" in prompt
        assert "injected into the prompts of *other* instances" in prompt


def test_formatter_renders_a_per_episode_table():
    trajectory = instance_trajectory(_wordle_run(), "wordle")
    rendered = source_teacher.formatter_for("wordle")(trajectory)
    assert "=== Instance wd0" in rendered
    assert "--- Episode 1: not solved" in rendered
    assert "CRANE" in rendered and "XYGXX" in rendered


def test_install_rebinds_every_module_that_holds_its_own_reference():
    import skillrl.distillation.distill as distill
    import skillrl.distillation.evolving_distill as evolving
    import skillrl.distillation.teacher_prompts as tp

    source_teacher.install("minesweeper", max_trajs=4)
    assert source_teacher.installed_level() == "minesweeper"
    assert "Minesweeper" in distill.CONTRASTIVE_DISTILLATION_PROMPT
    assert "Minesweeper" in evolving.SKILL_BANK_EVOLUTION_PROMPT
    # `distill` does a module-level `from teacher_prompts import ...`, so the
    # seeding path stays on the stock selector unless it is rebound too.
    assert distill.format_trajectories_block is not tp.format_trajectories_block
    assert evolving.format_trajectories_block is tp.format_trajectories_block


def test_install_rejects_an_unknown_level():
    with pytest.raises(ValueError, match="unknown source level"):
        source_teacher.install("hangman")


# ── a bank must not invoke an action the environment does not have ───────────
#
# Regression guard for the run2 collapse. The evolved Minesweeper bank contained
# a skill about a `terminate` action that does not exist in
# `game:Minesweeper-v0-only-reveal`. The actor appended `\boxed{terminate}` to
# its batched output, the parser reads only the LAST \boxed{}, and the env
# rejected it with "you did not provide a valid guess" -- which ENDS the episode.
# 461 of 2267 turns (20.3%) died that way and `any` fell from .55 to .05.
#
# Nothing caught it: `unparsed_rate` was 0.0000, *lower* than the two healthy
# runs, because `\boxed{terminate}` parses fine as a boxed expression. That
# metric answers "did we extract a \boxed{}", not "was the action accepted", so
# it is useless as a health check here.

VALID_MINESWEEPER_VERBS = {"reveal", "flag"}
VALID_WORDLE_VERBS: set[str] = set()  # wordle actions are bare words, not verbs

#: Verbs a teacher has been observed to invent, or might plausibly invent, none
#: of which the env accepts.
HALLUCINATED_VERBS = {"terminate", "quit", "stop", "done", "submit", "unflag",
                      "restart", "pass", "skip", "end"}


def bank_invokes_unknown_action(bank_text: str, level: str) -> list[str]:
    """Verbs a bank tells the agent to emit that the level does not accept."""
    import re
    valid = VALID_MINESWEEPER_VERBS if level == "minesweeper" else VALID_WORDLE_VERBS
    found = set()
    for m in re.finditer(r"\\boxed\{\s*(\w+)", bank_text):
        verb = m.group(1).lower()
        if valid and verb not in valid:
            found.add(verb)
    for verb in HALLUCINATED_VERBS:
        if re.search(rf"\b{verb}\b\s+(action|command)", bank_text, re.I):
            found.add(verb)
    return sorted(found)


def test_minesweeper_description_enumerates_the_action_set():
    """The teacher cannot avoid inventing an action it was never told about."""
    from hexa_orbit.source_teacher import TASK_DESCRIPTIONS

    desc = TASK_DESCRIPTIONS["minesweeper"]
    assert "reveal R C" in desc
    assert "flag R C" in desc
    assert "NO terminate" in desc
    # the batching trap that turned one bad skill into 20% dead turns
    assert "LAST" in desc


def test_minesweeper_description_states_the_turn_budget():
    """The 8-turn cap is the level's whole difficulty and the actor is never told it.

    The environment prompt names no limit, maximum or budget -- verified against a
    recorded observation. A teacher that does not know the cap writes skills for
    unlimited probing, which is a different game.
    """
    from hexa_orbit.source_teacher import TASK_DESCRIPTIONS

    desc = TASK_DESCRIPTIONS["minesweeper"]
    assert "8 turns" in desc
    assert "NEVER TOLD" in desc


def test_minesweeper_description_names_both_information_free_moves():
    """Flags and re-reveals each cost a turn and change nothing.

    The evolved Qwen bank promoted *Count-Driven Flagging* at confidence 0.85 --
    a skill for a move that cannot satisfy an `only-reveal` win condition. The
    teacher had no way to know that from the old description.
    """
    from hexa_orbit.source_teacher import TASK_DESCRIPTIONS

    desc = TASK_DESCRIPTIONS["minesweeper"]
    assert "only-reveal" in desc
    assert "Flagging never advances it" in desc
    assert "already revealed" in desc


def test_minesweeper_description_carries_no_cross_model_comparison():
    """Actor and teacher are the same model; the prompt must not smuggle in another.

    An earlier draft of this description quoted one actor's waste rate against
    another's. Aggregate or not, that is one model's behaviour entering a second
    model's teacher prompt, which is the thing Design C is supposed to isolate.
    """
    from hexa_orbit.source_teacher import TASK_DESCRIPTIONS

    desc = TASK_DESCRIPTIONS["minesweeper"].lower()
    for leak in ("stronger actor", "weaker actor", "gpt-oss", "qwen"):
        assert leak not in desc


def test_detects_a_bank_that_invents_terminate():
    bad = (r"[Strategy: Terminate only when board solved] The agent should only "
           r"issue the terminate action when all safe cells are revealed. "
           r"Example: \boxed{terminate}")
    assert bank_invokes_unknown_action(bad, "minesweeper") == ["terminate"]


def test_accepts_a_bank_using_only_real_actions():
    good = (r"[Strategy: Reveal deduced-safe cells] Example: \boxed{reveal 2 3}. "
            r"Flag a cell only when forced: \boxed{flag 0 1}.")
    assert bank_invokes_unknown_action(good, "minesweeper") == []


def test_a_runaway_turn_cannot_blow_up_the_teacher_prompt():
    """A turn that emits no `\\boxed{}` must not paste its reasoning into the prompt.

    `extract_last_boxed` returns the whole response when no box is present, and
    that string is recorded as the step's action. Unclamped, one runaway turn made
    a single Wordle trajectory 24,749 tokens against a ~1,800-token norm, which
    overflowed the 32k Qwen context on every round and silently froze the bank for
    a whole 13-round run.
    """
    runaway = "Okay, let me think about this. " * 1000
    rendered = source_teacher._render_action(runaway)
    assert len(rendered) < 200
    assert "unparsed" in rendered
    assert rendered.startswith("Okay, let me think about this.")


def test_a_normal_action_is_rendered_untouched():
    assert source_teacher._render_action(r"\boxed{CRANE}") == "CRANE"
    assert source_teacher._render_action(r"\boxed{reveal 2 3}") == "reveal 2 3"
    assert source_teacher._render_action("") == "(none)"
