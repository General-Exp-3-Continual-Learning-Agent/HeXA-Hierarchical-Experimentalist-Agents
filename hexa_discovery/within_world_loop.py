"""HeXA's actual training phase, run WITHIN one world.

Everything in this project so far has iterated over WORLDS -- one batch of seeds
per world, evolve, move on. That is not how HeXA is trained. On InterPhyre a
single level accumulates many interactions (the `catapult` level has 221
trajectories on disk) via

    offline distil -> run x seeds -> evolve -> run x seeds -> evolve -> ...

so the teacher repeatedly sees fresh evidence about the SAME task and can refine
rather than restate. Our per-world banks were distilled from five episodes, and
33% of their skills merely restated the agent's own 9k-character system prompt.
This runs the real loop so the round-by-round learning curve is visible.

**Within-world gains are NOT a transfer claim.** A DiscoverPhysics seed only
reshuffles observation noise while the hidden law stays fixed, so a bank grown on
one world can legitimately encode that world's answer. This measures whether the
evolution machinery produces a learning curve at all, and whether richer evidence
yields more specific skills. Transfer must still be tested on a held-out world.
"""

from __future__ import annotations

import argparse, json
from pathlib import Path

from hexa_discovery import _paths
_paths.bootstrap()

from hexa_discovery import reward as _reward
from hexa_discovery.config import DISCOVERY_LEVEL, world_facts_with_controls
from hexa_discovery.bank_audit import audit_bank, report as report_audit
from hexa_discovery.cross_world_loop import _install_seams, _reuse_or, summarise
from hexa_discovery.discovery_runner import run_batch_discovery

import skillrl.core.config as _skcfg
from skillrl.distillation.distill import run_distillation
from skillrl.distillation.evolving_distill import evolve_skill_bank


def main() -> None:
    p = argparse.ArgumentParser(description="HeXA training phase within one world")
    p.add_argument("--world", required=True)
    p.add_argument("--offline-traj-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--rounds", type=int, default=8)
    p.add_argument("--seeds-per-round", type=int, default=3)
    p.add_argument("--start-seed", type=int, default=200)
    p.add_argument("--max-skills", type=int, default=12)
    p.add_argument("--max-mistakes", type=int, default=6)
    p.add_argument("--min-rounds", type=int, default=None,
                   help="Override ScienceAgent's MIN_ROUNDS (default 2). A run "
                        "using this is NOT comparable with one at the default.")
    p.add_argument("--cli-model", default="sonnet")
    p.add_argument("--teacher-model", default="claude-sonnet-5")
    p.add_argument("--allow-agent-derived", action="store_true",
                   help="WITHIN-WORLD ONLY. Do not mask law-form terms that the "
                        "AGENT ITSELF wrote in its own reasoning. Within-level "
                        "learning means accumulating task knowledge from your own "
                        "experiments -- InterPhyre's banks say 'drop the red ball "
                        "on the longer lever arm', which is that level's answer, "
                        "learned from experience. Masking it is why earlier "
                        "within-world runs produced only generic skills. ORACLE "
                        "feedback (the judge critique, which reads the ground-truth "
                        "explanation) is excluded regardless -- that is not the "
                        "agent's own discovery. A bank built this way is NOT "
                        "transferable and must never be used cross-world.")
    args = p.parse_args()

    w = args.world
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    _install_seams()
    if args.allow_agent_derived:
        # Stop the teacher-facing formatter from masking the agent's own words.
        # The judge critique is already excluded upstream, so what survives here
        # is only what the agent observed and wrote for itself.
        import hexa_discovery.teacher_prompts_discovery as _tp
        _tp.scrub_law_forms = lambda t: t or ""
        print("  [within-world] agent-derived law-form terms are NOT masked; "
              "this bank is NOT transferable")
    _skcfg.LEVEL_DESCRIPTIONS[DISCOVERY_LEVEL] = world_facts_with_controls(
        w, include_base_prompt=True)

    ek = dict(model=args.cli_model, judge_model=args.cli_model,
              use_cli=True, cli_model=args.cli_model)
    if args.min_rounds is not None:
        ek["min_rounds"] = args.min_rounds
        print(f"  [deviation] min_rounds={args.min_rounds} "
              f"(ScienceAgent default is 2); this arm is not comparable with "
              f"arms run at the default")

    # Round 0: distil from the world's own bank-free episodes.
    bank_path = out / "skill_bank_round00.json"
    print(f"\n=== Round 0: distil from OFFLINE {w} ===")
    bank = _reuse_or(bank_path) or run_distillation(
        traj_dirs={DISCOVERY_LEVEL: Path(args.offline_traj_dir)},
        levels=[DISCOVERY_LEVEL], output_path=bank_path,
        teacher_model=args.teacher_model, skip_general=True)

    progress = {"world": w, "rounds": [], "seeds_per_round": args.seeds_per_round}
    seed = args.start_seed
    for r in range(1, args.rounds + 1):
        seeds = list(range(seed, seed + args.seeds_per_round))
        seed += args.seeds_per_round
        print(f"\n=== Round {r}/{args.rounds}: {w} seeds {seeds} "
              f"({len(bank.level_skills.get(DISCOVERY_LEVEL, []))} skills in bank) ===")
        rd = out / f"round_{r:02d}"
        res = run_batch_discovery(w, seeds, bank, rd, **ek)
        s = summarise(res)
        print(f"  round {r}: pass {s['pass_rate']:.0%}  reward {s['mean_reward']:+.3f} "
              f" expl {s['mean_expl_score']}")

        prev = bank
        bank_path = out / f"skill_bank_round{r:02d}.json"
        cached = _reuse_or(bank_path)
        bank = cached or evolve_skill_bank(
            level_name=DISCOVERY_LEVEL, prev_bank=bank,
            new_trajs_dir=rd / DISCOVERY_LEVEL, output_path=bank_path,
            max_skills=args.max_skills, max_mistakes=args.max_mistakes,
            teacher_model=args.teacher_model)
        if cached is None:
            from hexa_discovery.bank_reground import reground_bank
            reground_bank(bank, prev, res, DISCOVERY_LEVEL)
            bank.save(bank_path)

        # Audit EVERY round. Within-world training has no held-out world, so the
        # cross-world leak filter never runs here -- and the teacher sees one
        # world's episodes repeatedly, so it can infer the hidden operator and
        # write it into a skill. Measured on the first circle run: by round 3 a
        # skill's `example` field named the fractional Laplacian outright, and
        # 3/3 of that round's agents then named it in their explanations. A
        # within-world gain built on that is memorisation, not learning.
        audit = audit_bank(bank, DISCOVERY_LEVEL, [w])
        report_audit(audit, where=f"round {r} bank")
        # Under --allow-agent-derived the world's own terms are permitted BY
        # DESIGN: within-level learning means accumulating task knowledge from
        # your own experiments, and the audit cannot tell an agent-derived term
        # from an oracle-derived one -- it only matches strings. Oracle feedback
        # (the judge critique) is excluded upstream regardless, so what reaches
        # the teacher here is only what the agent itself observed. The audit
        # stays on as a WARNING: the resulting bank is not transferable, and any
        # cross-world use must re-run it through the leak filter.
        if audit["n_critical"] and args.allow_agent_derived:
            print(f"  [note] round {r}'s bank names {w}'s own physics. Expected "
                  f"under --allow-agent-derived (agent-derived, not oracle). "
                  f"This bank is NOT transferable.")
        elif audit["n_critical"]:
            print(f"  [STOP] round {r}'s bank names {w}'s own answer; later rounds "
                  f"would be confounded. Halting rather than producing a result "
                  f"that cannot be reported.")
            (out / "progress.json").write_text(json.dumps(
                {**progress, "halted_at_round": r, "reason": "bank names the target's answer",
                 "law_form_audit": audit}, indent=2))
            return

        progress["rounds"].append({
            "round": r, "seeds": seeds, **s,
            "n_skills": len(bank.level_skills.get(DISCOVERY_LEVEL, [])),
            "bank": str(bank_path)})
        (out / "progress.json").write_text(json.dumps(progress, indent=2))

    print("\n=== LEARNING CURVE ===")
    print(f"  {'round':>5s} {'pass':>6s} {'reward':>8s} {'expl':>6s} {'skills':>7s}")
    for rd_ in progress["rounds"]:
        print(f"  {rd_['round']:5d} {rd_['pass_rate']:6.0%} {rd_['mean_reward']:+8.3f} "
              f"{rd_['mean_expl_score'] or 0:6.2f} {rd_['n_skills']:7d}")


if __name__ == "__main__":
    main()
