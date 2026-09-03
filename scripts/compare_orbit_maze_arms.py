#!/usr/bin/env python
"""Paired comparison of two Maze arms (DESIGN_PHASE3_MAZE §14).

    python scripts/compare_orbit_maze_arms.py \
        --a results/orbit_maze_baseline \
        --b results/orbit_maze_hexa/eval_frozen \
        --out results/comparison_hexa_vs_nobank.json

Primary endpoint: episode-1 success, arm B minus arm A, on `clean` instances,
McNemar exact on discordant pairs. Everything else is secondary and printed
without significance claims.

Refuses to compare arms that did not run the same instances -- McNemar assumes
pairing, and `hexa_core/WEAKNESSES.md:300` records this project shipping a version
that assumed it without checking. Pass --allow-unpaired to inspect a partial run;
never for a reported number.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hexa_orbit.compare import (  # noqa: E402
    Unpaired,
    compare_arms,
    format_comparison,
    load_arm,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", type=Path, required=True, help="baseline arm run dir")
    parser.add_argument("--b", type=Path, required=True, help="treatment arm run dir")
    parser.add_argument("--name-a", default="no-bank")
    parser.add_argument("--name-b", default="frozen-bank")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--allow-unpaired", action="store_true",
        help="intersect instead of refusing. For inspection only.",
    )
    args = parser.parse_args()

    arm_a, arm_b = load_arm(args.a), load_arm(args.b)
    print(f"loaded {len(arm_a)} runs from {args.a}, {len(arm_b)} from {args.b}")

    try:
        result = compare_arms(
            arm_a, arm_b,
            name_a=args.name_a, name_b=args.name_b,
            strict=not args.allow_unpaired,
        )
    except Unpaired as exc:
        print(f"\nERROR: {exc}")
        return 2

    if args.allow_unpaired:
        print("\n*** --allow-unpaired: intersected. NOT a reportable number. ***")

    print()
    print(format_comparison(result))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
