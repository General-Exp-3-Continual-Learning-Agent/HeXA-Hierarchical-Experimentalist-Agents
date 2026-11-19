"""Recover the judge critique for trajectories written before it was stored.

`evaluate_episode` used to keep only `expl_score` and discard
`expl_result["reasoning"]`. Re-judging recovers the critique WITHOUT re-running
any episode: the agent's explanation is already on disk, so this is judge calls
only. The recovered text is stored redacted, exactly as a fresh run now does.
"""

from __future__ import annotations

import argparse, json
from pathlib import Path

from hexa_discovery import _paths
_paths.bootstrap()

from hexa_discovery.config import WORLD_ENGINE
from hexa_discovery.traj_adapter import redact_text


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--worlds", nargs="+", required=True)
    p.add_argument("--judge-model", default="sonnet")
    p.add_argument("--bank-free-only", action="store_true", default=True)
    args = p.parse_args()

    from hexa_discovery.cli_client import patch_llm_client
    patch_llm_client(model_override=args.judge_model)
    from scienceagent.worlds import get_world
    from scienceagent.evaluator import ExplanationJudge

    judge = ExplanationJudge(judge_model=args.judge_model)
    n_done = n_skip = 0
    for w in args.worlds:
        for f in sorted(Path(f"hexa_discovery/data/offline/sonnet/{w}/discovery").glob("trajectory_seed*.json")):
            d = json.loads(f.read_text())
            if args.bank_free_only and (d.get("skills_injected") or d.get("skill_ids_injected")):
                continue
            if d.get("expl_reasoning"):
                n_skip += 1; continue
            wd = get_world(w, engine=WORLD_ENGINE.get(w, "nbody"),
                           noise_std=d.get("noise_std", 0.075), noise_seed=d["seed"])
            r = judge.score(agent_explanation=d.get("explanation"),
                            optimal_explanation=wd.get("optimal_explanation", ""),
                            rubric=wd.get("explanation_rubric", ""), verbose=False)
            d["expl_reasoning"] = redact_text(r.get("reasoning") or "")
            f.write_text(json.dumps(d, indent=2, default=str))
            n_done += 1
            print(f"  {w} seed {d['seed']}: critique stored ({len(d['expl_reasoning'])} chars)")
    print(f"\nbackfilled {n_done}, already had {n_skip}")


if __name__ == "__main__":
    main()
