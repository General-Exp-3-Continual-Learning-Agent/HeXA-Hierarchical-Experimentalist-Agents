# HeXA configuration variants

HeXA is one cell in a 3×3 design space:

|              | **Static** (no rounds)         | **Iterative** (re-distill each round) | **Evolving** (incrementally evolve) |
|---           |---                             |---                                    |---                                  |
| **Offline**  | `run_hexa_offline.py`          | —                                     | —                                   |
| **Off2On**   | —                              | `run_hexa_iterative.py`               | **`run_hexa.py`** *(HeXA)*          |
| **Online**   | —                              | `run_hexa_online_iterative.py`        | `run_hexa_online_evolving.py`       |

Off2On = "offline-to-online": round 1 distills from initial offline trajectories, rounds 2+ collect online. **Online** = "pure online": round 1 runs the agent with no skill bank, rounds 2+ distill/evolve from the online rollouts.

## The 5 variants in detail

### Off2On / Evolving — `run_hexa.py` (the headline)
**Loop:** [skillrl/loops/evolving_loop.py](../skillrl/loops/evolving_loop.py) · `run_evolving_loop`
- Round 1: `run_distillation` from `--initial-traj-dir`
- Rounds 2+: `evolve_skill_bank(prev_bank, new_trajs)` — preserves confidence for retained skills, enforces `MAX_SKILLS_PER_LEVEL`
- Reported in the paper as **HeXA** with `--seeds-per-round 3`

### Off2On / Iterative — `run_hexa_iterative.py`
**Loop:** [skillrl/loops/iterative_loop.py](../skillrl/loops/iterative_loop.py) · `run_iterative_loop`
- Each round re-distills from scratch using the latest round's trajectories + best N carry-over successes
- The teacher does **not** see the prior bank
- Skills can accumulate without redundancy removal

### Offline / Static — `run_hexa_offline.py`
**Loop:** [skillrl/loops/offline_loop.py](../skillrl/loops/offline_loop.py) · `run_offline`
- One distillation pass on the initial trajectories
- Run all evaluation seeds against that one frozen bank — no rounds, no evolution
- Useful as a "what does distillation alone buy?" baseline

### Online / Evolving — `run_hexa_online_evolving.py`
**Loop:** [skillrl/loops/online_evolving_loop.py](../skillrl/loops/online_evolving_loop.py) · `run_online_evolving_loop`
- Round 1: agent runs with **no** skill bank (online baseline)
- Round 2: distill from round-1 trajectories
- Rounds 3+: evolve (same rule as HeXA)

### Online / Iterative — `run_hexa_online_iterative.py`
**Loop:** [skillrl/loops/online_iterative_loop.py](../skillrl/loops/online_iterative_loop.py) · `run_online_loop`
- Round 1: no skill bank
- Rounds 2+: re-distill from the latest round's trajectories + carry-over successes (no bank context to teacher)

## Defaults

All loops share a common set of defaults from [skillrl/core/config.py](../skillrl/core/config.py):

```python
TEACHER_MODEL          = "claude-sonnet-4-6"
DEFAULT_MAX_ITERATIONS = 25
DEFAULT_TEMPERATURE    = 0.3
DEFAULT_MAX_NEW_TOKENS = 800
MAX_SKILLS_PER_LEVEL   = 10
```

Override on the CLI: `--teacher-model`, `--max-iterations`, `--temperature`, `--max-skills`, etc. Run any wrapper with `--help` for the full set.
