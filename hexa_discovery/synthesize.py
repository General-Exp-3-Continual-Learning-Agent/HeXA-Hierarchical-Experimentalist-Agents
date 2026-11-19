"""Target-conditioned bank synthesis for a held-out world (HeXA's pi_ev step).

    K_l*  =  pi_ev( K_l1, ..., K_lS,  description(l*) )

This step was MISSING from the discovery arm. The cross-world loop grows one
shared bank under the synthetic level `discovery` and injects it **verbatim** at
transfer time; `config.WORLD_FACTS[held_out_world]` was never read. So the bank
that reached ether had never been re-grounded on ether's scene — it carried
world-agnostic method prose and nothing about "a heavy central anchor, ring
orbiters, and probes you control, where a global additive term may be present".

`skillrl.distillation.cross_level_synthesis.synthesize_target_bank` already
implements pi_ev for InterPhyre. Its prompt enforces what we lacked:
  - every emitted skill cites the source skills it came from
  - no invented coordinates (the teacher has not seen the target solved)
  - **target-side entities only** -- skills must be re-expressed in the target
    scene's objects, which is the re-grounding step
  - a `transfer_rationale` naming the bridging physical primitive
  - confidence >= 0.7 only when the primitive is well supported AND the target
    scene clearly invokes it

Our worlds are not in its InterPhyre `TARGET_LEVEL_BLOCKS` /
`SOURCE_LEVEL_BLOCKS` registries, so both blocks are supplied as overrides from
`config.WORLD_FACTS` -- the same answer-free factual text the teacher already
sees during training, never the hidden law.

S=1 here: our design produces a single shared bank rather than one bank per
world, so the synthesis re-grounds that one bank onto the target. The faithful
S>1 version needs a separate evolving run per source world.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from hexa_discovery import _paths

_paths.bootstrap()

from hexa_discovery.config import (DISCOVERY_LEVEL, WORLD_FACTS,
                                   leave_one_out_split, world_facts_with_controls)
from hexa_discovery.cross_world_loop import _install_seams
from hexa_discovery.bank_audit import audit_bank, report as report_audit


def main() -> None:
    p = argparse.ArgumentParser(description="Synthesise a target-conditioned bank")
    p.add_argument("--source-bank", default=None,
                   help="Single evolved bank to re-ground (S=1).")
    p.add_argument("--source-banks-dir", default=None,
                   help="Directory of per-world banks named skill_bank_<world>.json. "
                        "Every world except the target becomes a SEPARATE source "
                        "input to pi_ev (S>1) — the faithful form of the operator, "
                        "in which no source is privileged by training order.")
    p.add_argument("--target-world", required=True)
    p.add_argument("--source-worlds", nargs="+", default=None,
                   help="Worlds the source bank was grown on; used only to build "
                        "its answer-free scene block.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--teacher-model", default="claude-sonnet-5")
    p.add_argument("--prompt-variant", default="physics",
                   choices=("physics", "physics-slim", "text-game"))
    p.add_argument("--dry-run", action="store_true",
                   help="Build and dump the prompt without calling the teacher.")
    args = p.parse_args()

    if args.source_worlds and args.target_world in args.source_worlds:
        raise SystemExit(
            f"target {args.target_world!r} is also listed as a source world; "
            "the synthesised bank would be conditioned on its own answer"
        )

    # Seams first: the synthesis prompt carries whole banks plus scene blocks and
    # goes over the 128KiB argv limit, so the stdin teacher must be installed.
    _install_seams()
    from skillrl.distillation.cross_level_synthesis import synthesize_target_bank

    # The synthesised bank is what the agent reads at solve time, so it must
    # be constrained to controls that exist. Measured on circle: half a
    # synthesised bank named actions the world does not support.
    target_block = world_facts_with_controls(args.target_world)
    src_worlds = args.source_worlds or []
    source_block = (
        f"A set of {len(src_worlds)} simulated worlds, each with its own "
        "hidden governing law. In all of them the agent chooses particle initial "
        "positions, velocities and scalar properties, and the measurement times, "
        "then observes noisy positions over time.\n\n"
        + "\n\n".join(WORLD_FACTS[w] for w in src_worlds)
    )

    # Per-target leak filter. The law-form redaction controls the teacher's
    # INPUT; these banks were written by a teacher that can name a form from its
    # own physics prior, and measured on the per-world banks it did: K_yukawa
    # says "power-law vs screened/exponential", K_circle says "inverse-square",
    # K_coulomb_easy says "coulomb". Harmless while the naming world is a SOURCE.
    # Fatal if it names the TARGET's answer -- pi_ev would hand the synthesiser
    # the thing it is supposed to infer. Auditing after the fact is too late, so
    # offending skills are dropped from the source banks before synthesis.
    from hexa_discovery.bank_audit import attribute
    from hexa_discovery.traj_adapter import scan_for_law_forms
    from skillrl.core.skill_bank import SkillBank

    staging = Path(args.output_dir) / "_filtered_sources"
    staging.mkdir(parents=True, exist_ok=True)

    if args.source_banks_dir:
        found = sorted(Path(args.source_banks_dir).glob("skill_bank_*.json"))
        raw = {f.stem.replace("skill_bank_", ""): f for f in found}
        raw.pop(args.target_world, None)      # never source the target's own bank
        if not raw:
            raise SystemExit(f"no per-world banks in {args.source_banks_dir}")
    else:
        if not args.source_bank:
            raise SystemExit("pass --source-bank or --source-banks-dir")
        raw = {DISCOVERY_LEVEL: Path(args.source_bank)}

    source_bank_paths: dict[str, Path] = {}
    n_dropped = 0
    for name, path in raw.items():
        bank = SkillBank.load(path)
        # A per-world bank stores its skills under DISCOVERY_LEVEL; re-key to the
        # world name so pi_ev can cite which source each skill came from.
        skills = bank.level_skills.get(DISCOVERY_LEVEL, [])
        kept = []
        for sk in skills:
            terms = scan_for_law_forms(
                f"{sk.title} {sk.principle} {sk.when_to_apply} {sk.example}")
            if {w for t in terms for w in attribute(t)} & {args.target_world}:
                n_dropped += 1
                print(f"    leak: dropping [{', '.join(sorted(terms))}] "
                      f"{name}/{sk.title[:52]}")
                continue
            sk.source_level = name
            kept.append(sk)
        if not kept:
            print(f"    {name}: no skills survive the leak filter — skipping source")
            continue
        nb = SkillBank()
        for sk in kept:
            nb.add_skill(sk)
        out_p = staging / f"{name}.json"
        nb.save(out_p)
        source_bank_paths[name] = out_p
    print(f"  leak filter: {n_dropped} skill(s) dropped; "
          f"{len(source_bank_paths)} source bank(s) -> pi_ev (S={len(source_bank_paths)})")
    if not source_bank_paths:
        raise SystemExit("every source bank was emptied by the leak filter")

    # Coverage constraint for pi_ev.
    #
    # pi_ev selects ~8 skills from ~75 across the source banks, and it selects on
    # fit to the TARGET SCENE. That systematically discards lessons that are not
    # scene-specific -- above all the submission-time ones. Measured on
    # `fractional`: 5 of 10 independent source banks derived "do not declare an
    # already-measured constant as a free-fit parameter" (max confidence 1.00),
    # pi_ev kept none of them, and the refit penalty then fired on all 5 transfer
    # seeds (own constants 1e-08 -> refit 2.7e-04, up to 160,000x worse).
    #
    # This does not inject any specific skill -- that would be hand-writing the
    # answer. It requires the synthesised bank to SPAN the episode lifecycle, so
    # the teacher must pick something for each stage and cannot fill all eight
    # slots with experiment-design skills.
    coverage = (
        "COVERAGE REQUIREMENT. The agent's episode has four stages, and the bank "
        "is the only guidance it gets for all of them. Allocate the skills so "
        "every stage is represented -- do NOT spend every slot on experiment "
        "design:\n"
        "  1. EXPERIMENT DESIGN — what to probe and how to vary it.\n"
        "  2. DATA ANALYSIS — how to read noisy trajectories and rule a form in "
        "or out.\n"
        "  3. LAW SUBMISSION — how to express the final law, including which "
        "constants to declare as free/fittable and which to pin, and how the "
        "declaration is scored.\n"
        "  4. EXPLANATION — what the written explanation must contain to be "
        "judged correct.\n"
        "Stages 3 and 4 are where most episodes are actually lost. A lesson "
        "supported by several independent source banks should survive even when "
        "it is not specific to this target's scene."
    )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"synthesising a bank for TARGET={args.target_world}\n"
          f"  sources (S={len(source_bank_paths)}): "
          f"{', '.join(sorted(source_bank_paths))}")

    bank, audit = synthesize_target_bank(
        target_level=DISCOVERY_LEVEL,          # the actor retrieves under this key
        source_bank_paths=source_bank_paths,
        output_dir=out,
        teacher_model=args.teacher_model,
        target_block_override=target_block,
        source_block_overrides={n: (world_facts_with_controls(n)
                                    if n in WORLD_FACTS else source_block)
                                for n in source_bank_paths},
        structural_hint=coverage,
        prompt_variant=args.prompt_variant,
        dry_run=args.dry_run,
    )
    if bank is None:
        print("no bank produced (dry run, or the teacher returned nothing)")
        return

    skills = bank.level_skills.get(DISCOVERY_LEVEL, [])
    print(f"\nsynthesised {len(skills)} skill(s) for {args.target_world}:")
    for s in sorted(skills, key=lambda s: -s.confidence):
        print(f"  [{s.confidence:.2f}] {s.title}")

    # The synthesised bank is conditioned ON the target's description, so the
    # audit matters more here than anywhere else: this is the one step that could
    # legitimately be accused of handing the answer over.
    _, held = leave_one_out_split(args.target_world)
    print()
    report_audit(audit_bank(bank, DISCOVERY_LEVEL, held), where="synthesised bank")


if __name__ == "__main__":
    main()
