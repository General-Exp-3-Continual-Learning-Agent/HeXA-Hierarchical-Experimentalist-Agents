"""Distil one bank K_l per world, from BANK-FREE trajectories only.

HeXA's cross-level transfer is

    K_l*  =  pi_ev( K_l1, ..., K_lS,  description(l*) )

which needs a SEPARATE bank per source world. Our cross-world loop instead grew
one shared bank mutated sequentially, and that made ordering decisive: measured
on `loo_fracfamily`, 7 of 11 final skills came from the last three rounds, and
the world order is deterministic (always ends ether -> hubble -> oscillator). The
bank handed to `circle` was therefore dominated by global-term worlds it has
nothing structurally in common with.

Per-world banks remove that failure mode *structurally* rather than mitigating
it: every source bank enters pi_ev as an equal input, so there is no "last world".

**Only bank-free episodes are used.** A trajectory produced with the shared bank
injected carries that bank's content -- and, worse, carries the bias of its own
position in the training order -- so distilling K_l from it would reintroduce
the ordering effect by another route while looking clean. `_is_bank_free` is the
gate; a world with too few clean episodes is skipped loudly rather than silently
distilled from contaminated data.

Several worlds are single-class (gravity 6/6 pass, coulomb_easy 0/5). Round-1
distillation already handles one-sided input, and for pi_ev it is acceptable:
each K_l is a separate input and the contrast moves to the synthesis step.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hexa_discovery import _paths

_paths.bootstrap()

from hexa_discovery.config import (BENCHMARK_WORLDS, DISCOVERY_LEVEL,
                                   world_facts_with_controls)
from hexa_discovery.cross_world_loop import _install_seams

import skillrl.core.config as _skcfg
from skillrl.distillation.distill import run_distillation

MIN_EPISODES = 3


def _is_bank_free(traj: dict) -> bool:
    return not (traj.get("skills_injected") or traj.get("skill_ids_injected"))


def clean_dir(world: str, staging: Path) -> tuple[Path | None, int, int]:
    """Stage a world's bank-free trajectories; return (dir, n_pass, n_fail)."""
    src = Path(f"hexa_discovery/data/offline/sonnet/{world}/discovery")
    if not src.exists():
        return None, 0, 0
    out = staging / world / DISCOVERY_LEVEL
    out.mkdir(parents=True, exist_ok=True)
    n_pass = n_fail = 0
    for f in sorted(src.glob("trajectory_seed*.json")):
        try:
            d = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if not _is_bank_free(d):
            continue
        (out / f.name).write_text(json.dumps(d, indent=2, default=str))
        n_pass += bool(d.get("success"))
        n_fail += not d.get("success")
    return out, n_pass, n_fail


def main() -> None:
    p = argparse.ArgumentParser(description="Distil one bank per world from clean trajectories")
    p.add_argument("--output-dir", default="hexa_discovery/data/banks_per_world")
    p.add_argument("--worlds", nargs="+", default=None)
    p.add_argument("--teacher-model", default="claude-sonnet-5")
    p.add_argument("--skip-existing", action="store_true", default=True)
    args = p.parse_args()

    _install_seams()
    out_root = Path(args.output_dir)
    staging = out_root / "_staged"
    worlds = args.worlds or BENCHMARK_WORLDS

    summary = {}
    for w in worlds:
        bank_path = out_root / f"skill_bank_{w}.json"
        if args.skip_existing and bank_path.exists():
            print(f"\n=== {w}: reusing {bank_path.name} ===")
            continue
        d, np_, nf = clean_dir(w, staging)
        if d is None or np_ + nf < MIN_EPISODES:
            print(f"\n=== {w}: SKIPPED — only {np_ + nf} bank-free episode(s), "
                  f"need >= {MIN_EPISODES} ===")
            summary[w] = {"skipped": True, "n": np_ + nf}
            continue
        print(f"\n=== {w}: {np_} pass / {nf} fail (bank-free) ===")
        # The teacher sees this world's own answer-free description.
        _skcfg.LEVEL_DESCRIPTIONS[DISCOVERY_LEVEL] = world_facts_with_controls(
            w, include_base_prompt=True)
        bank = run_distillation(
            traj_dirs={DISCOVERY_LEVEL: d},
            levels=[DISCOVERY_LEVEL],
            output_path=bank_path,
            teacher_model=args.teacher_model,
            skip_general=True,
        )
        n = len(bank.level_skills.get(DISCOVERY_LEVEL, []))
        summary[w] = {"skills": n, "pass": np_, "fail": nf, "bank": str(bank_path)}
        print(f"  -> {n} skills")

    (out_root / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== per-world banks ===")
    for w, s in summary.items():
        print(f"  {w:17s} " + ("SKIPPED" if s.get("skipped")
                                else f"{s['skills']:2d} skills  ({s['pass']}P/{s['fail']}F)"))


if __name__ == "__main__":
    main()
