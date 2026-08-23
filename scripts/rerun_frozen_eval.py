#!/usr/bin/env python3
"""Rerun only the frozen paired eval against an already-trained bank, at a new
`max_tokens`.

Exists because §8 of `hexa_core/WEAKNESSES.md` warns that a positive delta must be
checked against a larger token budget before it is believed -- and paying for a full
30-round retrain just to change one eval-time setting wastes the 600+300 calls of
training that already happened. `_frozen_eval` (`hexa_core/passive_loop.py:606`) is
self-contained: task + frozen bank + a generate function, no training state. This
script calls it twice (with_memory=True/False) exactly as `run_pilot.py` does, with the
same seeding (`llm_seed`, `run_index`) so the pairing is identical to the original run
-- only the client's `max_tokens` differs.

Usage
-----
    python scripts/rerun_frozen_eval.py \\
        --task tower-of-hanoi --n-train 60 \\
        --bank results/tower_of_hanoi_v3/run0/round29/bank.json \\
        --base-url http://gpu016:8000/v1 --model openai/gpt-oss-120b \\
        --max-tokens 60000 \\
        --output-dir results/tower_of_hanoi_v3/run0_eval60k
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hexa_core.budget import RolloutBudget  # noqa: E402
from hexa_core.llm.vllm_client import VLLMClient  # noqa: E402
from hexa_core.passive_loop import _frozen_eval  # noqa: E402
from hexa_core.tasks import make_task  # noqa: E402
from skillrl.core.skill_bank import SkillBank  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--n-train", type=int, required=True,
                         help="Must match the original run -- some tasks derive the "
                              "eval split independently of n_train, but pass the same "
                              "value used originally to be safe.")
    parser.add_argument("--bank", required=True, help="Path to the frozen bank.json.")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--injection-budget-tokens", type=int, default=1000)
    parser.add_argument("--hierarchy", default="task", choices=["task", "task:type"])
    parser.add_argument("--no-parent-exclusion", action="store_true")
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--run-index", type=int, default=0)
    parser.add_argument("--llm-seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    task = make_task(args.task, args.n_train)
    bank = SkillBank.load(args.bank)
    client = VLLMClient(model=args.model, base_url=args.base_url,
                         max_tokens=args.max_tokens)

    def generate_meta(prompt: str, seed: int | None = None) -> tuple[str, str]:
        generation = client.generate_structured(prompt=prompt, seed=seed)
        if generation is None:
            return "", "error"
        return (generation.raw_text or ""), (generation.finish_reason or "")

    budget = RolloutBudget(max_train=0)
    kwargs = dict(
        budget=budget,
        injection_budget_tokens=args.injection_budget_tokens,
        hierarchy=args.hierarchy,
        parent_exclusion=not args.no_parent_exclusion,
        concurrency=args.concurrency,
        generate_meta=generate_meta,
        run_index=args.run_index,
        llm_seed=args.llm_seed,
    )

    print(f"[eval] with-memory, max_tokens={args.max_tokens} ...")
    wm_acc, wm_n, wm_trunc = _frozen_eval(
        task, bank, client.generate, with_memory=True,
        record_path=out_dir / "eval_with_memory.jsonl", **kwargs,
    )
    print(f"  with-memory acc={wm_acc:.3f}  n={wm_n}  truncated={wm_trunc}")

    print(f"[eval] no-memory, max_tokens={args.max_tokens} ...")
    nm_acc, nm_n, nm_trunc = _frozen_eval(
        task, bank, client.generate, with_memory=False,
        record_path=out_dir / "eval_no_memory.jsonl", **kwargs,
    )
    print(f"  no-memory acc={nm_acc:.3f}  n={nm_n}  truncated={nm_trunc}")

    delta = wm_acc - nm_acc
    print("=" * 78)
    print(f"  HeXA (with bank)  : {wm_acc:.3f}  truncated={wm_trunc}/{wm_n}")
    print(f"  no memory         : {nm_acc:.3f}  truncated={nm_trunc}/{nm_n}")
    print(f"  delta             : {delta:+.3f}")
    print("=" * 78)

    (out_dir / "rerun_config.json").write_text(json.dumps({
        "task": args.task, "n_train": args.n_train, "bank": args.bank,
        "max_tokens": args.max_tokens, "run_index": args.run_index,
        "llm_seed": args.llm_seed, "with_memory_acc": wm_acc,
        "no_memory_acc": nm_acc, "delta": delta,
        "with_memory_truncated": wm_trunc, "no_memory_truncated": nm_trunc,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
