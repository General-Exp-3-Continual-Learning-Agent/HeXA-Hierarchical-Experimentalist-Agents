"""Side-by-side comparison of finished (or in-flight) baseline runs.

Reads each run's `episodes.jsonl` plus `memory_store.json` and prints the table
the write-up needs: success rate, the two budget axes (simulator calls and added
context tokens), retrieval behaviour, and MemRL's learned utilities.

Works on partial runs, so it doubles as a progress check.

    python -m baselines.analysis.compare_runs baselines/data/*_catapult_qwen7b
    python -m baselines.analysis.compare_runs --json baselines/data/memrl_*
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path
from typing import Any, Optional


def load_run(path: Path) -> Optional[dict]:
    episodes = path / "episodes.jsonl"
    if not episodes.exists():
        return None
    rows = []
    for line in episodes.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue          # a run killed mid-write can leave a partial line
    if not rows:
        return None

    store_path = path / "memory_store.json"
    store = json.loads(store_path.read_text()) if store_path.exists() else {"records": []}
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    return {"name": path.name, "rows": rows, "store": store, "manifest": manifest}


def _mean(values) -> float:
    values = [v for v in values if isinstance(v, (int, float))]
    return statistics.fmean(values) if values else 0.0


def metrics(run: dict) -> dict[str, Any]:
    rows, store = run["rows"], run["store"]
    n = len(rows)
    succ = sum(1 for r in rows if r.get("success"))
    sims = [r.get("simulator_calls") or 0 for r in rows]
    toks = [r.get("added_context_tokens") or 0 for r in rows]
    injected = [len(r.get("retrieved_memory_ids") or []) for r in rows]
    qs = [rec.get("q", 0.0) for rec in store.get("records", [])]

    return {
        "seeds": f"{n}",
        "successes": f"{succ}",
        "success rate": f"{succ / n:.1%}" if n else "-",
        "avg iterations": f"{_mean(r.get('iterations') for r in rows):.1f}",
        "sims / seed": f"{_mean(sims):.1f}",
        "seeds w/ 0 sims": f"{sum(1 for s in sims if s == 0)}/{n}",
        "tokens / seed": f"{_mean(toks):,.0f}",
        "max tokens": f"{max(toks):,}" if toks else "-",
        "memories injected": f"{_mean(injected):.2f}",
        "episodes w/ none": f"{sum(1 for i in injected if i == 0)}/{n}",
        "store size": f"{len(store.get('records', []))}",
        "mean Q": f"{statistics.fmean(qs):+.3f}" if qs else "-",
        "Q range": (
            f"{min(qs):+.2f}..{max(qs):+.2f}" if qs and any(qs) else "-"
        ),
        "wall clock": f"{sum(r.get('elapsed_time') or 0 for r in rows) / 3600:.2f} h",
    }


ROW_ORDER = [
    "seeds", "successes", "success rate", "avg iterations",
    "sims / seed", "seeds w/ 0 sims",
    "tokens / seed", "max tokens",
    "memories injected", "episodes w/ none", "store size",
    "mean Q", "Q range", "wall clock",
]


def print_table(runs: list[dict], width: int = 22) -> None:
    cols = [metrics(r) for r in runs]
    names = [r["name"] for r in runs]

    header = " " * 20 + "".join(f"{n[:width]:>{width}}" for n in names)
    print(header)
    print("-" * len(header))
    for key in ROW_ORDER:
        line = f"{key:20}" + "".join(f"{c.get(key, '-'):>{width}}" for c in cols)
        print(line)

    for run in runs:
        rows = run["rows"]
        phases = collections.Counter(
            (r.get("retrieval_diagnostics") or {}).get("phase_a")
            for r in rows
            if (r.get("retrieval_diagnostics") or {}).get("phase_a")
        )
        modes = collections.Counter(
            (r.get("retrieval_diagnostics") or {}).get("mode")
            for r in rows
            if (r.get("retrieval_diagnostics") or {}).get("mode")
        )
        detail = dict(phases or modes)
        if detail:
            print(f"\n{run['name']} retrieval: {detail}")
        dev = run["manifest"].get("deviations")
        if dev:
            print(f"{run['name']} deviations:")
            for d in dev:
                print(f"   - {d}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare baseline runs")
    ap.add_argument("runs", nargs="+", help="run directories")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    args = ap.parse_args()

    loaded = []
    for p in args.runs:
        run = load_run(Path(p))
        if run is None:
            print(f"[skip] no episodes.jsonl in {p}")
            continue
        loaded.append(run)
    if not loaded:
        raise SystemExit("No runs with episodes found.")

    if args.json:
        print(json.dumps({r["name"]: metrics(r) for r in loaded}, indent=2))
    else:
        print_table(loaded)


if __name__ == "__main__":
    main()
