# HeXA: Hierarchical Experimentalist Agents

**HeXA** is a skill-evolution framework that turns failed and successful agent rollouts into a curated bank of physics skills the agent reads at solve time. On our [InterPhyre](https://github.com/General-Exp-3-Continual-Learning-Agent/HeXA-Hierarchical-Experimentalist-Agents/tree/main/interphyre) suite of 2D physics puzzles, HeXA more than doubles single-shot ReAct's success rate using the same actor model. We also open-source [InterPhyre](https://github.com/General-Exp-3-Continual-Learning-Agent/HeXA-Hierarchical-Experimentalist-Agents/tree/main/interphyre) as one of the first experimentation centric benchmark as a Testbed for Learning through Experimentation and Hypothesis generations and refinement. Please see more details of our environment and benchmark [InterPhyre](https://github.com/sankaranv/interphyre) here.



This repository contains the HeXA implementation, **five configuration variants** (Init × Update rule), **two ablations** (reward, skills-only), **three ReAct-style baselines**, and a cross-level meta-transfer pipeline. Everything runs through the [Claude Code CLI](https://claude.ai/code) — no API keys, no Anthropic SDK.

See [docs/HEXA.md](docs/HEXA.md) for the method, [docs/CONFIGS.md](docs/CONFIGS.md) for the variant matrix, [docs/BASELINES.md](docs/BASELINES.md) for the baselines, and [docs/ABLATIONS.md](docs/ABLATIONS.md) for the ablations.

---

## Quickstart

```bash
# 1. Install Claude Code CLI (one-time): https://claude.ai/code
which claude && claude --version

# 2. Install Python deps
pip install -r requirements.txt
pip install ./interphyre  # the simulator (uses pre-built wheel for box2d-py)

# 3. Smoke test: HeXA on catapult (1 round, 3 seeds, ~15-30 min)
python scripts/run_hexa_catapult.py \
    --initial-traj-dir Initial_trajectories/catapult \
    --output-dir skillrl/data/hexa_catapult_smoke/ \
    --num-rounds 1 --seeds-per-round 3 --start-seed 6
```

For the full paper configuration on catapult, use `--num-rounds 17 --seeds-per-round 3`.
For other levels, use `python scripts/run_hexa.py --level <level> --initial-traj-dir ...`.

---

## What's where

```
HeXA/
├── README.md, INSTALL.md, requirements.txt, LICENSE
├── docs/                              deeper writeups (HeXA, configs, baselines, ablations, cross-level)
├── scripts/                           one wrapper per experiment type — see below
├── skillrl/                           HeXA core (code only; no shipped run artefacts)
│   ├── loops/                         the 7 loop variants (5 main configs + 2 ablations)
│   ├── distillation/                  evolver prompts, distill, evolve, cross-level synthesis
│   └── core/, runner/, analysis/      skill bank, retriever, agent runner, post-hoc analysis scripts
├── react_agent/                       ReAct agent + Reflexion subpackage
├── interphyre/                        physics simulator (Box2D); install with pip install ./interphyre
└── Initial_trajectories/              seed trajectories per level (only `catapult/` ships in this repo)
```

> **Note**: the public release ships only **code** and seed trajectories for `catapult`. Pre-computed run outputs (skill banks, per-round trajectories, cross-level results, baseline summaries) are not bundled — re-generate them by running the scripts in `scripts/`, or pull them from the companion data archive. For levels other than `catapult`, supply your own seed trajectories via `--initial-traj-dir`.

### One script per experiment

| Script | Underlying module | What it runs |
|---|---|---|
| `scripts/run_hexa.py` | `skillrl.loops.evolving_loop` | **HeXA** (offline→online evolving) — headline method |
| `scripts/run_hexa_catapult.py` | `skillrl.distillation.teacher_prompts_catapult` | Dedicated catapult entry point (equivalent to `run_hexa.py --level catapult`) |
| `scripts/run_hexa_iterative.py` | `skillrl.loops.iterative_loop` | Off2On with re-distillation each round |
| `scripts/run_hexa_offline.py` | `skillrl.loops.offline_loop` | Distill once, evaluate against fixed bank |
| `scripts/run_hexa_online_evolving.py` | `skillrl.loops.online_evolving_loop` | Online start, then evolve |
| `scripts/run_hexa_online_iterative.py` | `skillrl.loops.online_iterative_loop` | Online start, then re-distill each round |
| `scripts/run_hexa_contrastive.py` | `skillrl.loops.contrastive_only_loop` | **Reward ablation**: contrastive-only, no rewards, no confidence |
| `scripts/run_hexa_skills_only.py` | `skillrl.loops.skills_only_evolving_loop` | **Skills-only ablation**: success trajectories only, no mistakes in the bank |
| `scripts/run_cross_level.py` | composite | Synthesise target bank from source banks → run offline eval |
| `scripts/run_react.py` | `react_agent.run_react` | ReAct baseline (Qwen / vLLM / mock) |
| `scripts/run_react_claude.py` | `react_agent.run_react_claude` | ReAct baseline (Claude Sonnet via CLI) |
| `scripts/run_reflexion.py` | `react_agent.run_react_claude_reflexion` | Reflexion (K=2 trials, mid-seed resume) |
| `scripts/run_direct.py` | `react_agent.run_react_claude_direct` | Direct-answer (2-iteration zero-shot) |

Wrappers forward every flag to the underlying module — `python scripts/run_hexa.py --help` prints the same options as `python -m skillrl.loops.evolving_loop --help`.

---

## Documentation

- [docs/HEXA.md](docs/HEXA.md) — method walkthrough: distil → evolve cycle, bank capacity, evolution rule
- [docs/CONFIGS.md](docs/CONFIGS.md) — variant matrix: Init × Update rule, mapped to scripts and files
- [docs/BASELINES.md](docs/BASELINES.md) — ReAct, Reflexion, Direct
- [docs/CROSS_LEVEL.md](docs/CROSS_LEVEL.md) — meta-transfer (no target trajectories used)
- [docs/ABLATIONS.md](docs/ABLATIONS.md) — reward ablation (contrastive-only) and skills-only ablation

---

## License

MIT. See [LICENSE](LICENSE).

## Citation

```bibtex
@article{hexa2026,
  title  = {HeXA: Hierarchical Experimentalist Agents},
  author = {Abhranil Chandra and Sankaran Vaidyanathan and Utsav Dhanuka and Varun Gandhi and Scott Niekum},
  year   = {2026}
}
```
