"""One-time migration: compact bundles to one entry per seed.

Bundles generated before the compact-format change store one entry per
variant tried (including intermediate impossible variants).  This script
collapses each bundle to a single entry per seed — the valid entry if one
exists, otherwise a single impossible marker at variant=0.

Usage:
    python -m interphyre.validation.compact_bundles

Safe to run multiple times (idempotent — already-compact bundles are left
unchanged and reported as "already compact").
"""

from __future__ import annotations

import json
import lzma
from pathlib import Path

_BUNDLE_DIR = Path(__file__).parent.parent / "data" / "levels"


def compact_bundle(bundle_path: Path) -> None:
    """Compact a single bundle file in-place."""
    with lzma.open(bundle_path, "rb") as fh:
        data = json.load(fh)

    entries = data["entries"]
    n_before = len(entries)

    # Collapse to one entry per seed: valid entry wins over impossible.
    seed_map: dict[int, dict] = {}
    for entry in entries:
        s = entry["seed"]
        if s not in seed_map or entry["status"] == "valid":
            seed_map[s] = entry

    compacted = sorted(seed_map.values(), key=lambda e: e["seed"])
    n_after = len(compacted)

    if n_before == n_after:
        print(f"  {bundle_path.name}: already compact ({n_after} entries)")
        return

    data["entries"] = compacted
    with lzma.open(bundle_path, "wt", encoding="utf-8") as fh:
        json.dump(data, fh)

    n_valid = sum(1 for e in compacted if e["status"] == "valid")
    print(
        f"  {bundle_path.name}: {n_before} → {n_after} entries "
        f"({n_valid} valid, {n_after - n_valid} impossible, "
        f"{n_before - n_after} removed)"
    )


def main() -> None:
    bundles = sorted(_BUNDLE_DIR.glob("*.json.lzma"))
    if not bundles:
        print("No bundles found in", _BUNDLE_DIR)
        return

    print(f"Compacting {len(bundles)} bundles in {_BUNDLE_DIR}...\n")
    for bundle_path in bundles:
        compact_bundle(bundle_path)
    print("\nDone.")


if __name__ == "__main__":
    main()
