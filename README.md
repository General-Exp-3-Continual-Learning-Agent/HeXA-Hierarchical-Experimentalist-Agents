# HeXA: Hierarchical Experimentalist Agents

**HeXA** is a in-context RL and skill-evolution framework for learning via experimentation, that turns failed and successful agent rollouts into a curated bank of physics skills the agent reads at solve time. On our [InterPhyre](https://github.com/General-Exp-3-Continual-Learning-Agent/HeXA-Hierarchical-Experimentalist-Agents/tree/main/interphyre) suite of 2D physics puzzles, HeXA more than doubles single-shot ReAct's success rate using the same actor model. We also open-source [InterPhyre](https://github.com/General-Exp-3-Continual-Learning-Agent/HeXA-Hierarchical-Experimentalist-Agents/tree/main/interphyre) as one of the first experimentation centric benchmark as a Testbed for Learning through Experimentation and Hypothesis generations and refinement. Please see more details of our environment and benchmark [InterPhyre](https://github.com/sankaranv/interphyre) here.



This repository contains the HeXA implementation, **five configuration variants** (Init × Update rule), **two ablations** (reward, skills-only), **three ReAct-style baselines**, and a cross-level meta-transfer pipeline. Everything runs through the [Claude Code CLI](https://claude.ai/code) — no API keys, no Anthropic SDK.

See [docs/HEXA.md](docs/HEXA.md) for the method, [docs/CONFIGS.md](docs/CONFIGS.md) for the variant matrix, and [docs/ABLATIONS.md](docs/ABLATIONS.md) for the ablations.

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

## Results at a glance

Rough headline numbers from the paper. Every arm differs only in what is added to
the actor's context, every memory is built on training instances and frozen
before evaluation, and all arms are paired on the same held-out instances.
Appendices carry the significance tests, token budgets and per-episode splits.

### InterPhyre (our suite) — held-out solve rate (%)

| Method | catapult | PtP | DtE (7B) | TBP (7B) | Tokens/ep. |
|---|---|---|---|---|---|
| Zero-shot | 0.0 | 0.0 | 7.0 | 0.0 | 0 |
| ReAct | 11.0 | 20.0 | 63.3 | 12.0 | 0 |
| Reflexion | 21.3 | 32.0 | 76.0 | 27.9 | 287 |
| Episodic RAG | 62.9 | 60.0 | 42.7 | 18.7 | 6,513 |
| MemRL | 49.0 | 46.0 | 36.0 | 23.3 | 10,149 |
| **HeXA** | **82.7** | **96.0** | **82.7** | **50.0** | **2,392** |

Claude Sonnet on catapult / pass-the-parcel, Qwen2.5-7B on down-to-earth /
two-body. HeXA wins on every level while injecting fewer tokens than either
memory baseline. Cross-level synthesis — a catapult bank written from other
levels' banks, with no catapult trajectory ever seen — reaches **42.7%**, against
11.0% for ReAct on the same level.

### Interactive games — success in at least one of three episodes

| Task | Actor | Zero-Shot | Episodic RAG | MemRL | HeXA |
|---|---|---|---|---|---|
| Mastermind | gpt-oss-120b | .683 | .650 | .690 | **.883** |
| Mastermind | Qwen3-8B | .367 | .492 | .392 | **.640** |
| Maze | gpt-oss-120b | .607 | .601 | **.667** | .649 |
| Maze | Qwen3-8B | .479 | .483 | .513 | **.564** |

Maze on gpt-oss-120b is the one setting of the ten external ones where HeXA is
not the best method.

Cross-task transfer holds up here too: a Mastermind bank synthesised from a
**Wordle** bank plus a statement of Mastermind's rules, having seen *zero*
Mastermind trajectories, scores **.933** — matching the within-task bank (.883)
built from 80 target trajectories.

### Reasoning and planning — accuracy on held-out items

| Task | Zero-Shot | Episodic RAG | MemRL | HeXA |
|---|---|---|---|---|
| MathGAP | .510 | .560 | .600 | **.863** |
| ZebraLogic | .560 | .720 | .650 | **.807** |
| Tower of Hanoi | .263 | .377 | .273 | **.520** |
| Matchstick | .737 | .767 | .703 | **.787** |

gpt-oss-120b throughout.

### DiscoverPhysics — explanation score from an independent judge

| Task | Zero-Shot | Episodic RAG | MemRL | HeXA |
|---|---|---|---|---|
| oscillator | .516 | .822 | .780 | **.853** |
| circle | .580 | .580 | .569 | **.916** |

Claude Sonnet actor, judged by an independent model.

### Robot manipulation — LIBERO-PRO success rate (%)

HeXA's bank is injected into the **Pigey** agent, whose Claude Sonnet
orchestrator drives a frozen pi-0.5 policy. The pi-0.5 row is the policy given
the task instruction directly, with no orchestrator.

| Method | Goal task | Goal swap | Object task | Object swap | Spatial task | Spatial swap |
|---|---|---|---|---|---|---|
| pi-0.5 | 24.4 | 36.4 | 10.4 | 16.8 | 56.4 | 47.6 |
| Pigey | 30.4 | 43.2 | 28.0 | 26.0 | 70.4 | 51.6 |
| **HeXA** | **56.8** | **73.6** | **53.6** | **35.6** | **77.2** | **63.2** |
| Δ over Pigey | +26.4 | +30.4 | +25.6 | +9.6 | +6.8 | +11.6 |

The bank helps on all six perturbation suites, +6.8 to +30.4 points over the same
orchestrator without it — about +44% relative on average. The smallest gain is on
the suite where the no-bank agent already does best.

Scripts for these: the `scripts/run_orbit_*` family (games and cross-task
transfer), `scripts/run_core_*` (reasoning and planning), and
`scripts/run_discoverphysics_memory.py`. The LIBERO-PRO robotics integration is
not on `main` — it lives on the `robotics_baseline` branch, since it depends on
the Pigey agent and a pi-0.5 checkpoint rather than on this repo's simulator.

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
| `scripts/run_hexa_off2on_iterative.py` | `skillrl.loops.iterative_loop` | Off2On with re-distillation each round |
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

### Small open-weight actors (Qwen2.5-3B)

HeXA uses the same actor model as its teacher, so a 3B run asks one small model to
both solve episodes and maintain the bank. The paper's Qwen2.5-3B runs therefore use
a different **configuration** from the 7B and Sonnet runs. The method is unchanged —
same loop, same retriever, same update rule, same frozen-bank evaluation — and the
three settings below exist because the defaults were tuned against larger teachers.

**1. Slim evolution prompt (automatic).** The default
`SKILL_BANK_EVOLUTION_PROMPT` is ~8.2k characters and is resent on every round from
round 2 on, on top of the bank and the trajectory block. A 3B teacher loses the task
in that much preamble: it returns flat confidences, stops merging, and emits
near-duplicate skills until the cap. `skillrl/distillation/teacher_prompts_qwen_slim.py`
keeps every evolution rule — complete-bank output, hard caps, `is_new`/`source_seeds`,
confidence moving in both directions, numeric constants copied through, duplicate and
dead-skill removal, contrastive mistakes, self-consistency — and cuts the worked
examples, the per-rule rationales, and the optional audit fields that
`evolving_distill` already reads with a default. Placeholders are identical, so it is
a drop-in rebind. `evolving_loop` applies it automatically when the teacher model name
looks like a 3B Qwen, so **no other model's runs are affected**: 7B, Qwen3-8B, gpt-oss
and Sonnet all keep the full template.

**2. Bank capacity 5 skills / 3 mistakes** (`--max-skills 5 --max-mistakes 3`, default
10/5). At the default the 3B teacher filled the bank with near-paraphrases that all
carried the same uninformative confidence, which makes the retriever's choice among
them arbitrary. A tighter cap forces the evolver to merge or drop rather than
accumulate. Keep the injection caps consistent with capacity — asking for 6 skills out
of a 5-skill bank just injects the whole bank.

**3. Six seeds per round** (`--seeds-per-round 6`, default 3). Evolution is
contrastive: it needs a success and a failure from the same round to say what the
success did differently. At a small actor's solve rate, 3 seeds/round frequently
produced a round with no success at all, so the evolver had nothing to contrast and
kept rewriting the same few ideas. Wider rounds, same number of rounds.

```bash
python scripts/run_hexa.py \
  --level two_body_problem \
  --initial-traj-dir Initial_trajectories/two_body_problem \
  --output-dir runs/tbp_qwen3b \
  --model openai:Qwen/Qwen2.5-3B-Instruct \
  --teacher-model openai:Qwen/Qwen2.5-3B-Instruct \
  --num-rounds 17 --seeds-per-round 6 --start-seed 6 \
  --max-skills 5 --max-mistakes 3
```

The actor prompt, tool set, retriever, update strategy and evaluation protocol
(frozen bank, `--no-evolver`, held-out seeds) are exactly those used for every other
model in the paper.

---

## Documentation

- [docs/HEXA.md](docs/HEXA.md) — method walkthrough: distil → evolve cycle, bank capacity, evolution rule
- [docs/CONFIGS.md](docs/CONFIGS.md) — variant matrix: Init × Update rule, mapped to scripts and files
- [docs/CROSS_LEVEL.md](docs/CROSS_LEVEL.md) — meta-transfer (no target trajectories used)
- [docs/ABLATIONS.md](docs/ABLATIONS.md) — every reported ablation: reward labels, initialization x update rule, within-level skill building, cross-level hierarchy, injection-time bank content

---

## License

MIT. See [LICENSE](LICENSE).

## Citation

```bibtex

{
@article{hexa2026,
  title  = {HeXA: Hierarchical Experimentalist Agents},
  author = {Abhranil Chandra and Sankaran Vaidyanathan and Utsav Dhanuka and Varun Gandhi and Scott Niekum},
  year   = {2026}
}
```
