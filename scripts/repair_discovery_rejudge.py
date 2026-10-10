#!/usr/bin/env python
"""Rebuild a complete re-judge JSON for an eval directory whose file was clobbered.

`hexa_discovery.rejudge` writes to `<run-dir>/rejudge_<judge>.json`, so running
it once per arm into the same directory leaves only the arm that ran last.  That
happened on `circle_eval` and `osc_eval`: the bank pass overwrote the baseline
rows, and 30 of 60 re-judged episodes survive only in `rejudge_base.log`.

The reported numbers were read off those logs and are correct -- this changes no
result.  What it fixes is that the JSON no longer disagrees with the claim in
`DOC_Results/DISCOVERY_RESULTS.md` §7 that it holds every evaluation episode.

The log carries the independent explanation score and reward but not
`norm_mse`, which the pass criterion needs; that comes from the episode JSON the
log names, so nothing is recomputed from scratch and nothing is re-judged.

    python scripts/repair_discovery_rejudge.py \
        --run-dir discovery-agents/hexa_discovery/data/runs/osc_eval
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "discovery-agents"))

from hexa_discovery.config import PASS_ERR_THRESHOLD  # noqa: E402
from hexa_discovery.rejudge import (  # noqa: E402
    CODE_SCORE_THRESHOLD,
    PAPER_SCORE_THRESHOLD,
)
from hexa_discovery.reward import blended_reward  # noqa: E402

#: "  oscillator baseline  seed 300: expl 0.2 -> 0.5   reward -0.400 -> -0.250"
ROW = re.compile(
    r"^\s*(?P<world>\S+)\s+(?P<arm>\S+)\s+seed\s+(?P<seed>\d+):\s+"
    r"expl\s+(?P<self>[\d.]+|None)\s*->\s*(?P<indep>[\d.]+)\s+"
    r"reward\s+(?P<rs>[-+][\d.]+)\s*->\s*(?P<ri>[-+][\d.]+)"
)

#: Where an arm's episode JSONs live, relative to the run directory.
ARM_DIRS = {
    "baseline": "baseline/discovery",
    "with_bank": "transfer/{world}/with_bank/discovery",
}


def episode_path(run: Path, arm: str, world: str, seed: int) -> Path | None:
    template = ARM_DIRS.get(arm)
    if template is None:
        return None
    path = run / template.format(world=world) / f"trajectory_seed{seed}_skillrl.json"
    return path if path.exists() else None


def rows_from_log(run: Path, log: Path) -> list[dict]:
    out = []
    for line in log.read_text().splitlines():
        m = ROW.match(line)
        if not m:
            continue
        world, arm, seed = m["world"], m["arm"], int(m["seed"])
        path = episode_path(run, arm, world, seed)
        if path is None:
            print(f"  ! no episode JSON for {arm} {world} seed {seed}; skipped")
            continue
        norm_mse = json.loads(path.read_text()).get("norm_mse")
        indep = float(m["indep"])
        out.append({
            "file": str(path), "world": world, "seed": seed, "arm": arm,
            "norm_mse": norm_mse,
            "expl_self": None if m["self"] == "None" else float(m["self"]),
            "expl_independent": indep,
            "reward_self": float(m["rs"]),
            "reward_independent": blended_reward(indep, norm_mse),
            "pass_indep_075": bool(norm_mse is not None
                                   and norm_mse < PASS_ERR_THRESHOLD
                                   and indep >= CODE_SCORE_THRESHOLD),
            "pass_indep_090": bool(norm_mse is not None
                                   and norm_mse < PASS_ERR_THRESHOLD
                                   and indep >= PAPER_SCORE_THRESHOLD),
            "recovered_from": log.name,
        })
        # The log's own reward is the same blend of the same two inputs; a
        # mismatch means the file pairing is wrong, not that a number changed.
        if abs(out[-1]["reward_independent"] - float(m["ri"])) > 5e-3:
            print(f"  ! reward mismatch for {arm} seed {seed}: "
                  f"log {m['ri']} vs recomputed "
                  f"{out[-1]['reward_independent']:+.3f}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run-dir", required=True)
    p.add_argument("--judge-model", default="claude-opus-4-6")
    p.add_argument("--write", action="store_true",
                   help="Without this the repair is only reported, not saved")
    args = p.parse_args()

    run = Path(args.run_dir)
    target = run / f"rejudge_{args.judge_model}.json"
    existing = json.loads(target.read_text())["rows"] if target.exists() else []
    have = {(r["arm"], r["world"], r["seed"]) for r in existing}
    print(f"{target.name}: {len(existing)} row(s), arms "
          f"{sorted({r['arm'] for r in existing})}")

    recovered = []
    for log in sorted(run.glob("rejudge_*.log")):
        for row in rows_from_log(run, log):
            key = (row["arm"], row["world"], row["seed"])
            if key not in have:
                recovered.append(row)
                have.add(key)

    merged = existing + recovered
    merged.sort(key=lambda r: (r["arm"], r["world"], r["seed"]))
    print(f"recovered {len(recovered)} row(s) from logs -> {len(merged)} total, "
          f"arms {sorted({r['arm'] for r in merged})}")

    if not args.write:
        print("\n(dry run; pass --write to save)")
        return
    backup = target.with_suffix(".json.clobbered")
    if target.exists() and not backup.exists():
        backup.write_bytes(target.read_bytes())
        print(f"kept the clobbered original at {backup.name}")
    target.write_text(json.dumps(
        {"judge_model": args.judge_model,
         "note": "baseline rows recovered from rejudge_*.log after both arms "
                 "wrote to this same path; see scripts/repair_discovery_rejudge.py",
         "rows": merged}, indent=2))
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
