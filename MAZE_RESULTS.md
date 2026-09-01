# Maze — consolidated results (gpt-oss-120b)

Actor and teacher are both `openai/gpt-oss-120b`.

Last updated 2026-09-01.

## Task configuration and splits

```
env_id: maze          shapes: 6x6, 7x6, 7x7, 6x7
max_turns_per_episode: 9      total_step_cap: 45
shortest_path: 7 or 8 moves   test_size: 256
```

Observation is the agent's absolute position `(x, y)` plus what lies in each of
the four directions, each reported as `wall`, `path` or `goal`. Out-of-bounds
renders as `wall`. Nothing beyond the four adjacent cells is visible.

| pool | n | how |
|---|---|---|
| eval | 256 requested, **243 paired** | `MazeTask.eval_specs()` |
| train | independent draw | `MazeTask.train_specs(n)` |

Unlike Mastermind (120 codes) and Wordle (192 words), Maze seeds come from a
~1e6 space, so train and eval are **independent draws** rather than a partition
— two draws effectively never collide. A repeated *layout* is tolerable and is
reported by a leakage split; a repeated *code* or *secret word* would be the
answer itself, which is why those two levels must partition instead.

243 of 256 are used throughout: 13 instances errored in one or more runs and are
excluded from the paired set rather than scored as losses.

## Results

n=243 paired. Deltas are the mean over all arm-run x baseline-run pairs; the
fraction is how many reach p<0.05 on a paired McNemar exact test.

| arm | runs | ep1 | ep2 | ep3 | `any` |
|---|---|---|---|---|---|
| baseline (no bank) | 3 | .244 (.239-.247) | .392 (.370-.416) | .610 (.593-.621) | .647 (.634-.658) |
| Design B (within-level) | **1** | .210 | .420 | .613 | .634 |
| Design C near ← minesweeper | 3 | .240 (.210-.263) | .432 (.399-.453) | .586 (.564-.601) | .610 (.588-.621) |
| Design C far ← wordle | 3 | .263 (.230-.300) | .465 (.444-.481) | .597 (.576-.630) | .671 (.646-.687) |
| **Design C both ← ms+wordle** | 3 | .263 (.251-.272) | **.499 (.477-.531)** | .630 (.613-.650) | .708 (.700-.724) |
| ablation rawdump ← minesweeper | 3 | .217 (.210-.222) | .451 (.436-.469) | .631 (.613-.663) | .668 (.642-.704) |

### Delta vs baseline

| arm | ep1 | ep2 | ep3 | `any` |
|---|---|---|---|---|
| Design B (within-level) | −3.4 (0/3) | +2.7 (0/3) | +0.3 (0/3) | −1.4 (0/3) |
| Design C near ← minesweeper | −0.4 (0/9) | +4.0 (1/9) | −2.5 (0/9) | −3.7 (0/9) |
| Design C far ← wordle | +1.9 (0/9) | +7.3 (3/9) | −1.4 (0/9) | +2.3 (0/9) |
| **Design C both ← ms+wordle** | +1.9 (0/9) | **+10.7 (7/9)** | +1.9 (0/9) | +6.0 (1/9) |
| ablation rawdump ← minesweeper | −2.7 (0/9) | +5.9 (3/9) | +2.1 (0/9) | +2.1 (0/9) |

## Reading

**Maze is a null result almost everywhere.** Of 20 arm x metric cells, exactly
one reaches broad significance: multi-source Design C on **ep2, +10.7, 7/9**.

**Within-level HExA does nothing on Maze** — every metric 0/3, ep1 and `any`
trending negative. This is the original Design B maze finding and it survives:
the bank cannot beat the model's implicit navigation policy.

**Cross-task transfer does not beat within-level here either.** The best Design C
arm gains ep2 where Design B does not, but neither moves ep1, ep3 or `any` in a
way that holds up.

**The ablation does NOT support re-grounding on Maze.** Raw-dumping the
Minesweeper bank is *better* than synthesising from it on three of four metrics:

| metric | rawdump | Design C near (synthesised) |
|---|---|---|
| ep1 | −2.7 | −0.4 |
| ep2 | +5.9 (3/9) | +4.0 (1/9) |
| ep3 | **+2.1** | −2.5 |
| `any` | **+2.1** | **−3.7** |

An earlier single-run reading (rawdump ep1 −3.7/−2.9 vs synthesised +1.6/+2.5)
suggested the opposite; at n=3 both are null on ep1 and the ordering reverses on
the late metrics. **Re-grounding is supported on Mastermind only.** Design C
near ← minesweeper is the only arm in either target trending negative on `any`.

## Why Maze resists a skill bank

Measured against baseline behaviour, the banks target errors the actor does not
make:

| baseline behaviour | rate | how much bank text addresses it |
|---|---|---|
| moves into a **known** wall | **0.9%** (55/6275) | the single most repeated instruction across all three Maze banks |
| adjacent cell reported `goal` | 1.5% of labels | 2 skills, conf .80-.85 |
| ep2 already optimal when ep1 succeeded | 57%, mean +0.70 steps over optimal | *Cross-Episode Path Reuse* |
| immediate reversal (move, then move back) | 9.8% | one skill |

The skills are correct; they are not binding.

**The actual bottleneck is one no bank addresses.** The per-episode cap is 9
steps against a mean optimal path of 7.46. All 83 never-solved instances run
9.0/9.0/9.0 — every episode capped — at mean `best_progress` 0.507, with **zero
improvement across episodes**. Winning them requires treating episode 1 as a
scouting run expected to fail, mapping aggressively, then exploiting in 2-3.
Neither source has the budget structure that makes that necessary: Minesweeper's
episode *ends* on a wrong probe, and Wordle gets 5 guesses against 192 words. The
premise that generates the right Maze strategy does not exist in either source,
so synthesis could not have produced it.

## Caveats

- **Design B Maze is n=1** (one bank, one eval), so its deltas are 1x3 not 3x3.
- **The injection budget truncated every arm.** All common-mistakes entries were
  dropped in all six arms, and skills are cut lowest-confidence-first — which is
  anti-correlated with utility here. `maze_near` lost *Cross-Episode Map
  Retention* (conf .65), the one skill aimed at the real bottleneck, while
  keeping the vacuous *Zero-Propagation Analogy* (conf .80). Only `maze_both` was
  injected whole (8/8), which is a plausible partial explanation of its win and
  should be controlled for before the multi-source result is leaned on.

| arm | tokens | skills kept | dropped |
|---|---|---|---|
| maze near | 854 | 5/8 | Cross-Episode Map Retention (.65), Avoid Redundant Moves (.68), Probabilistic Direction Guessing (.60) |
| maze far | 956 | 7/8 | Duplicate Observation Reconciliation (.60) |
| maze both | 972 | **8/8** | — |

## Provenance

| artifact | path |
|---|---|
| baseline | `results/design_c/maze_baseline/run{0,1,2}` |
| Design B bank + eval | `results/orbit_maze_hexa_gptoss_v4t12/` |
| Design C banks | `results/design_c/maze_from_{minesweeper,wordle,minesweeper_wordle}/run0` |
| Design C evals | `results/design_c/phase4/maze_{near,far,both,rawdump}_*/run{0,1,2}` |
