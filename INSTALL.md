# Install

## 1. Claude Code CLI

All Claude-actor and teacher calls go through `claude -p` as a subprocess. There is no Anthropic SDK and no API key — auth piggybacks on whatever your local `claude` session uses.

```bash
# Install: https://claude.ai/code
which claude        # should print a path
claude --version    # should print a version
```

If you only plan to use the open-source / Qwen baselines (`scripts/run_react.py`), you can skip Claude.

## 2. Python dependencies

Requires Python 3.10+.

```bash
pip install -r requirements.txt
```

## 3. InterPhyre simulator

The physics environment is in `interphyre/`. It vendors a pre-built `box2d-py` Linux wheel (PyPI does not ship one, and building from source needs SWIG <4 — this is the canonical install path on the cluster the paper was run on):

```bash
pip install ./interphyre
```

If you are not on Linux x86_64, you will need to build `box2d-py` yourself; see `interphyre/pyproject.toml` for guidance.

## 4. Sanity check

```bash
python -c "from skillrl.core.skill_bank import SkillBank; print('skillrl OK')"
python -c "from react_agent.reflexion.runner import run_seed_with_reflexion; print('react_agent OK')"
python -c "import interphyre; print('interphyre OK')"
```

## 5. (Optional) Qwen / local vLLM

Only needed for the open-source ReAct baseline (`scripts/run_react.py`) and the reward ablation on Qwen-7B (`scripts/run_hexa_contrastive.py`). Install your preferred Qwen runtime — typically:

```bash
pip install vllm transformers
# Start a local OpenAI-compatible server, e.g.:
vllm serve "Qwen/Qwen2.5-7B-Instruct" --port 8000
# Then point the runner at it:
python scripts/run_react.py --model "Qwen2.5-7B-Instruct" --vllm-url http://localhost:8000/v1 ...
```

## 6. Smoke test

```bash
python scripts/run_hexa.py \
    --level catapult \
    --initial-traj-dir Initial_trajectories/catapult \
    --num-rounds 10 --seeds-per-round 3 --start-seed 6
```

Expected: completes in a few minutes, writes `skillrl/data/evolving/catapult/skill_bank_evolving_1.json`.
