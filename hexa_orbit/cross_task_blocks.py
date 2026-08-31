"""Factual task blocks for Design C cross-task synthesis on ORBIT.

:mod:`skillrl.distillation.cross_level_synthesis` was built for InterPhyre and
carries its Box2D scene registries inline. This module is the ORBIT counterpart:
the same three things (a shared-environment paragraph, per-source scene blocks,
per-target scene blocks) for turn-based text games, loaded via that module's
``--blocks-module`` flag.

**Every block here is strategy-free by construction.** They state what the agent
sees, what the actions are, what ends an episode, and nothing about how to
play — the strategy is carried by the *source skill banks*, which is the whole
claim being tested. A target block that hinted at approach would make a Phase 4
Episode-1 gain unattributable.

All facts are transcribed from the ORBIT checkout (see ``orbit_bridge``), with
the config values from ``configs/multi_task_multi_episode_config.yaml`` applied,
since ``MultiEpisodeEnv.from_dict`` forwards ``max_turns_per_episode`` into the
inner env's ``env_kwargs`` (``envs/multi_episode_env.py:733-742``) and thereby
overrides the registered defaults (Minesweeper's registered ``max_turns=25``
becomes 8). Two blocks deliberately do **not** describe ORBIT's published config,
and must not be "corrected" back to it:

- **Wordle runs at 5 turns, not 10** (D-C6). At 10 the task is saturated -- `any`
  = .994, zero failures in every training round.
- **Mastermind is 4 digits over 1-8, not 3 over 1-6** -- ``mastermind.TASK_CONFIG``,
  because 3-over-6 gives only 120 codes and the train and eval pools could not be
  disjoint at all.

A block that describes a config the bank was not evolved on, or that the eval does
not run, hands the teacher a false premise -- and with no target trajectories,
nothing downstream would catch it.
"""

from __future__ import annotations

#: Prompt template this domain needs; consumed by ``--blocks-module``.
PROMPT_VARIANT = "text-game"


DOMAIN_BLOCK = """\
**Environment (shared across all tasks below)**: turn-based text games. The agent
is a language model that reads a text observation and replies with one action per
turn, written inside `\\boxed{...}`. Anything outside the box is reasoning and is
not parsed as an action.

Each task instance is played for exactly 3 EPISODES against the same hidden
instance, under a fixed total step budget (3 x the per-episode turn limit). An
episode ends on success, on an illegal-format action, or when the per-episode
turn limit is reached; the environment then resets to the same hidden instance
and a new episode begins. The agent keeps its full transcript — every past
observation and its own past reasoning — across all three episodes, so anything
an earlier episode revealed about the hidden instance is still readable in the
context of a later one.

An episode succeeds only if the agent reaches the task's success condition within
that episode's turn limit.
"""


# ── Source tasks (ORBIT train split) ───────────────────────────────────

MINESWEEPER_BLOCK = """\
**Observation (factual):** A 5x5 grid, rendered each turn. Unrevealed cells show
`.`; a revealed cell shows the count of mines among its up-to-8 neighbours;
flagged cells show `F`. 5 of the 25 cells hold mines. Rows and columns are
0-indexed.

**Actions:** `\\boxed{reveal <row> <col>}` or `\\boxed{flag <row> <col>}`. Flagging
toggles a marker and reveals nothing.

**Mechanics:** Mines are placed *after* the first reveal of an episode, never in
the 3x3 zone around it, so that reveal can never lose and always cascades.
Because the first reveal is fixed at the board centre and played identically in
every episode, the mine layout is the same in all three: what an earlier episode
revealed about a cell still holds. Revealing a cell whose neighbour-count is 0
cascades, revealing its neighbours transitively. Revealing a mine ends the
episode immediately as a loss.

**Success condition:** every non-mine cell revealed. Flags are not required.

**Turn limit:** 8 turns per episode, 24 total across the 3 episodes. The first
turn of every episode is spent on the fixed opening reveal, leaving 7 for the
agent.
"""

WORDLE_BLOCK = """\
**Observation (factual):** The history of the agent's guesses this episode, each
with per-letter feedback against a hidden 5-letter secret word drawn from a
fixed English word list:
  - `G` — that letter is correct and in the correct position
  - `Y` — that letter occurs in the secret word but in a different position
  - `X` — that letter does not occur in the secret word

**Actions:** `\\boxed{WORD}` — one 5-letter word.

**Mechanics:** A guess of the wrong length, or a word already guessed in this
episode, is rejected as an invalid action; it consumes the turn and returns no
feedback. The secret word is fixed for the whole instance, so it is the same
across all 3 episodes.

**Success condition:** feedback of `G` in all five positions.

**Turn limit:** 5 turns per episode, 15 total across the 3 episodes.
"""


# ── Target tasks (ORBIT held-out split) ────────────────────────────────

MAZE_BLOCK = """\
**Observation (factual):** The agent's current position `(x, y)` in a hidden
rectangular maze (6x6, 6x7, 7x6 or 7x7), plus what lies immediately in each of
the four directions — each reported as `wall`, `path`, or `goal`. Out-of-bounds
renders as `wall`, so the border is indistinguishable from an interior wall.
Nothing beyond the four adjacent cells is observable.

**Actions:** `\\boxed{up}`, `\\boxed{down}`, `\\boxed{left}`, `\\boxed{right}`.

**Mechanics:** Moving into a wall leaves the position unchanged and consumes the
turn. The maze layout and the start position are fixed for the whole instance,
so all 3 episodes replay the same maze from the same start. The shortest path
from start to goal is 7 or 8 moves.

**Success condition:** reach the goal cell.

**Turn limit:** 9 turns per episode, 27 total across the 3 episodes.
"""

MASTERMIND_BLOCK = """\
**Observation (factual):** The history of the agent's guesses this episode
against a hidden 3-digit code. Digits are drawn from 1-6 and no digit repeats
within the code, so there are 120 possible codes. Each guess is scored with:
  - black pegs — the count of positions holding the correct digit
  - white pegs — the count of digits present in the code but in a wrong position

**Actions:** `\\boxed{d d d}` — three digits separated by spaces.

**Mechanics:** The scoring returns counts only; it never says *which* positions
were correct. The hidden code is fixed for the whole instance, so it is the same
across all 3 episodes.

**Success condition:** 3 black pegs.

**Turn limit:** 3 turns per episode, 9 total across the 3 episodes.
"""


SOURCE_BLOCKS: dict[str, str] = {
    "minesweeper": MINESWEEPER_BLOCK,
    "wordle": WORDLE_BLOCK,
}

TARGET_BLOCKS: dict[str, str] = {
    "maze": MAZE_BLOCK,
    "mastermind": MASTERMIND_BLOCK,
}
