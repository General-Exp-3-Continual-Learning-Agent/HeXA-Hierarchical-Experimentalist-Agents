# Confounded: banks synthesised against the wrong Mastermind task

`MASTERMIND_BLOCK` in `hexa_orbit/cross_task_blocks.py` described the target as
a **4-digit code, digits 1-8, 1680 possible codes**, success at 4 black pegs,
actions `\boxed{d d d d}`.

The task the eval actually runs (`hexa_orbit.mastermind.TASK_CONFIG`) is a
**3-digit code, digits 1-6, 120 possible codes**, success at 3 black pegs,
actions `\boxed{d d d}`.

So all three Mastermind banks here were synthesised against a task that does not
exist. Skills such as *Global Digit Set Completion* ("once four distinct digits
have been identified") and *Full-Digit Identification Shortcut* describe
arithmetic that is wrong for a 3-of-6 code.

Measured (n=40, no matched baseline at the time):

| arm | ep1 | ep2 | ep3 | any |
|---|---|---|---|---|
| near ← wordle | .125 | .500 | .900 | .900 |
| far ← minesweeper | .075 | .025 | .100 | .200 |
| both | .125 | .350 | .825 | .825 |

The far arm collapsed — ep2 (.025) below ep1 (.075) — which is what pointed at
the bug. Its bank leans hardest on the four-digit framing.

Kept for the record, excluded from every result. The block is now pinned to
TASK_CONFIG by `test_mastermind_block_matches_task_config`.

The Maze blocks were checked against real transcripts and are accurate, so the
Maze arms are unaffected and stand.

The `mastermind_rawdump_wordle` ablation injects the *source* bank verbatim and
never reads the target block, so it is unaffected and was not moved.
