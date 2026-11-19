"""Axis B: cross-world skill evolution for HeXA on DiscoverPhysics.

One shared meta-bank (stored under the synthetic level `discovery`) is grown by
iterating over WORLDS (not seed-batches):

  round 1  : distill the bank from world[0]'s OFFLINE seed trajectories
  world w  : run the agent on w with the current bank (S noise-seeds), then
             evolve the bank by folding w's fresh trajectories in
  transfer : freeze the final bank; on each held-out world run
             (a) agent + frozen bank and (b) plain agent (no bank), same actor,
             and compare pass-rates.

The bank is CROSS-WORLD, so it must be genuinely evaluated on unseen worlds —
that leave-one-out / leave-one-group-out transfer number is the headline result.

Reuses HeXA's run_distillation / evolve_skill_bank unchanged; supplies the four
DiscoverPhysics seams (reward, prompts+formatter, runner, trajectory schema).
Also supports a POOLED baseline (distill once from all train worlds' offline
trajectories, no online evolve) via `mode="pooled"`.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional

from hexa_discovery import _paths

_paths.bootstrap()

from hexa_discovery import confidence as _confidence
from hexa_discovery import teacher_cli as _teacher_cli
from hexa_discovery import reward as _reward
from hexa_discovery import teacher_prompts_discovery as _prompts
from hexa_discovery.bank_audit import audit_bank, report as report_audit
from hexa_discovery.bank_reground import reground_bank
from hexa_discovery.replay import ReplayBuffer, describe as describe_batch
from hexa_discovery.config import (
    BENCHMARK_WORLDS,
    DISCOVERY_LEVEL,
    world_facts_with_controls,
    WORLD_FACTS,
    leave_one_out_split,
    leave_one_group_out_split,
    pooled_world_facts,
)
from hexa_discovery.discovery_runner import run_batch_discovery

import skillrl.core.config as _skcfg
from skillrl.core.skill_bank import SkillBank
from skillrl.distillation.distill import run_distillation
from skillrl.distillation.evolving_distill import evolve_skill_bank


#: Seed-number stride per world when pooling. Pooled trajectories from different
#: worlds reuse the same seed numbers (every world starts at `--start-seed`), and
#: the teacher cites skills by seed. Offsetting by world index makes each citation
#: name exactly one episode.
_POOL_SEED_STRIDE = 1000


def build_pooled_dir(
    traj_dirs: list[Path], output_dir: Path, level: str = DISCOVERY_LEVEL
) -> Path:
    """Merge per-world trajectory dirs into one directory `load_trajectories` can read.

    `skillrl`'s `load_trajectories` globs `trajectory_seed*.json` in a SINGLE
    directory, so a multi-world pooled distillation needs the files in one place.
    A naive copy is wrong twice over: filenames collide (every world has a
    `trajectory_seed0_skillrl.json`, so later worlds would overwrite earlier
    ones), and the `seed` field inside collides too, which makes the teacher's
    `source_seeds` citations ambiguous and — before the confidence seam — made
    all but the first trajectory per seed invisible to confidence.

    So each file is rewritten with a globally unique seed
    (`world_index * 1000 + seed`), keeping the original under `orig_seed` and the
    world name in `world` (which the trajectory header prints). Returns the
    merged directory.
    """
    merged = output_dir / "pooled_input" / level
    merged.mkdir(parents=True, exist_ok=True)

    n_written = 0
    for idx, traj_dir in enumerate(traj_dirs):
        traj_dir = Path(traj_dir)
        files = sorted(traj_dir.glob("trajectory_seed*.json"))
        if not files:
            print(f"  [Warning] no trajectories in {traj_dir}")
            continue
        for f in files:
            try:
                data = json.loads(f.read_text())
            except (json.JSONDecodeError, OSError):
                print(f"  [Warning] skipping unreadable {f}")
                continue
            orig = data.get("seed", 0)
            world = data.get("world", traj_dir.parent.name)
            data["orig_seed"] = orig
            data["seed"] = idx * _POOL_SEED_STRIDE + int(orig)
            data["world"] = world
            out = merged / f"trajectory_seed{data['seed']}_{world}_skillrl.json"
            out.write_text(json.dumps(data, indent=2, default=str))
            n_written += 1
        print(f"  pooled {len(files)} trajectories from {world} ({traj_dir})")

    print(f"  → {n_written} trajectories merged into {merged}")
    return merged


def _worlds_in_dir(traj_dir: Path) -> list[str]:
    """Distinct `world` values among a directory's trajectories, for labelling."""
    worlds: set[str] = set()
    for f in sorted(Path(traj_dir).glob("trajectory_seed*.json")):
        try:
            w = json.loads(f.read_text()).get("world")
        except (json.JSONDecodeError, OSError):
            continue
        if w:
            worlds.add(w)
    return sorted(worlds)


def _set_current_world_facts(world: str) -> None:
    """Inject the current world's answer-free facts as the teacher's
    level_description (in-place mutation is seen by distill + evolving_distill)."""
    _skcfg.LEVEL_DESCRIPTIONS[DISCOVERY_LEVEL] = world_facts_with_controls(
        world, include_base_prompt=True)


def _install_seams(evolve_max_trajs: int | None = None) -> None:
    """Install the DiscoverPhysics seams. Order matters.

    reward first: both of the others read the reward that it installs —
    `confidence` rescales from its range (and asserts the endpoints match),
    and the prompt formatter resolves it to sort and to print the header.
    """
    _reward.patch()
    _confidence.patch()
    _prompts.patch(evolve_max_trajs=evolve_max_trajs)
    # Independent of the other three: routes the teacher's (very large) prompt
    # through stdin instead of argv. See teacher_cli for why that is mandatory
    # rather than a tidy-up.
    _teacher_cli.patch()
    # Salvage truncated teacher replies instead of silently dropping the
    # whole pass; see hexa_discovery/json_salvage.py.
    from hexa_discovery import json_salvage as _json_salvage
    _json_salvage.patch()


def _reuse_or(bank_path: Path):
    """Load an already-written bank instead of rebuilding it, or None.

    Resume correctness, not just speed. The teacher is stochastic, so
    re-running `run_distillation` / `evolve_skill_bank` produces a DIFFERENT
    bank from the one on disk. A run interrupted during transfer (both runs so
    far died there on a CLI usage limit) would then evaluate its remaining
    held-out episodes against a bank the earlier episodes never saw — mixing two
    banks inside one transfer arm, silently, and invalidating the comparison it
    exists to make. Nine teacher calls saved is the lesser reason.
    """
    if not bank_path.exists():
        return None
    try:
        bank = SkillBank.load(bank_path)
    except Exception as exc:  # corrupt/partial write -> rebuild rather than die
        print(f"  [Warning] could not reuse {bank_path} ({exc}); rebuilding")
        return None
    n = len(bank.level_skills.get(DISCOVERY_LEVEL, []))
    print(f"  reusing existing bank {bank_path.name} ({n} skills) — resume")
    return bank


def _pass_rate(results: list[dict]) -> float:
    return sum(1 for r in results if r.get("success")) / len(results) if results else 0.0


def _mean(values: list[float]) -> Optional[float]:
    vals = [v for v in values if v is not None and v == v]  # drop None/NaN
    return sum(vals) / len(vals) if vals else None


def summarise(results: list[dict]) -> dict:
    """Pass-rate PLUS the graded reward and its two components.

    Pass-rate alone is close to useless as a transfer metric here, and that is
    measured rather than assumed -- and the two arms behave completely differently,
    which is the reason to record both numbers.

    The NO-BANK baseline is close to deterministic. Across the bootstrap every
    world came out unanimous (gravity 6/6, yukawa 6/6, fractional 0/2,
    three_species 0/2, ether 0/2) with reward spreads of 0.05-0.11, and
    three_species returned 0.15294939 / 0.15292719 on two different noise seeds --
    a relative spread under 0.02%.

    WITH the bank, variance explodes: reward spreads of 1.39 on oscillator, 1.29
    on ether, 1.05 on circle, 0.99 on fractional. The same bank on the same world
    scores +0.99 on one seed and -0.40 on another. The bank changes which
    experiments the agent designs, so it moves the outcome far more than
    observation noise ever does, in both directions.

    Hence pass-rate alone is a bad headline: flat in baseline, genuinely noisy in
    the bank arm, and blind to magnitude in both. The graded reward records how
    far a world moved, which is what distinguishes "the bank rescued this world"
    from "the bank happened to flip one seed". It also means n=3 seeds is thin
    against a spread above 1.0 -- report the per-seed values, not just the mean.

    The graded reward does vary where pass-rate does not: fractional's two
    failures scored +0.70 and +0.75, three_species' -0.11 and -0.16. Recording it
    means a bank that moves a held-out world from "badly wrong" to "just under the
    line" is visible instead of reading as a flat 0% either way.

    `expl_score` is reported separately because on the 8 FIT_WORLDS it is doing
    almost all the discriminating: a wrong functional form with fitted free
    parameters reached norm_mse 3.3e-05 on fractional -- 3000x inside the pass
    line -- and only the judge caught it.
    """
    return {
        "n": len(results),
        "pass_rate": _pass_rate(results),
        "mean_reward": _mean([r.get("reward") for r in results]),
        "mean_norm_mse": _mean([r.get("norm_mse") for r in results]),
        "mean_expl_score": _mean([r.get("expl_score") for r in results]),
        "rewards": [r.get("reward") for r in results],
    }


def run_cross_world_loop(
    train_worlds: list[str],
    held_out_worlds: list[str],
    offline_traj_dir: Path,
    output_dir: Path,
    *,
    seeds_per_world: int = 3,
    start_seed: int = 100,
    max_skills: int = 12,
    max_mistakes: int = 6,
    max_evolve_trajs: int = 6,
    cross_world_contrast: bool = False,
    teacher_model: str = "claude-sonnet-4-6",
    mode: str = "sequential",  # "sequential" (evolve) or "pooled"
    offline_traj_dirs: Optional[dict[str, Path]] = None,
    pooled_traj_dirs: Optional[list[Path]] = None,
    episode_kwargs: Optional[dict] = None,
) -> dict:
    """Grow a cross-world bank on train_worlds, then transfer-test on held_out_worlds."""
    if not teacher_model.startswith("claude-"):
        # skillrl.distillation.distill.call_teacher routes to the Claude CLI
        # ONLY when model.startswith("claude-"); anything else silently falls
        # through to an HF/Qwen loader (e.g. tries to download "Qwen/sonnet"),
        # which fails and makes distillation return zero skills with no error
        # surfaced here. Fail fast instead of burning agent episodes on a run
        # that can never produce a bank.
        raise ValueError(
            f"--teacher-model={teacher_model!r} must start with 'claude-' "
            "(e.g. 'claude-sonnet-4-6') to route through the Claude CLI. Bare "
            "aliases like 'sonnet' only work for --cli-model, not --teacher-model."
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    episode_kwargs = dict(episode_kwargs or {})
    _install_seams(evolve_max_trajs=max_evolve_trajs if cross_world_contrast else None)

    seeds = list(range(start_seed, start_seed + seeds_per_world))
    progress: dict = {"mode": mode, "train_worlds": train_worlds,
                      "held_out_worlds": held_out_worlds, "rounds": []}

    # ── Build the bank ──────────────────────────────────────────────────
    if mode == "pooled":
        # Distill once from every train world's offline trajectories pooled
        # under the single `discovery` level. No online evolve.
        if pooled_traj_dirs:
            pooled_dir = build_pooled_dir(
                [Path(d) for d in pooled_traj_dirs], output_dir
            )
            pooled_worlds = train_worlds
        else:
            # A single, already-pooled directory (the pre-existing behaviour).
            pooled_dir = offline_traj_dir
            pooled_worlds = _worlds_in_dir(Path(pooled_dir)) or train_worlds
        # The pooled set spans many worlds; labelling it with one world's setup
        # describes every other episode wrongly.
        _skcfg.LEVEL_DESCRIPTIONS[DISCOVERY_LEVEL] = pooled_world_facts(pooled_worlds)
        print(f"  pooled level_description covers {len(pooled_worlds)} world(s): "
              f"{', '.join(sorted(pooled_worlds))}")
        bank_path = output_dir / "skill_bank_pooled.json"
        bank = run_distillation(
            traj_dirs={DISCOVERY_LEVEL: Path(pooled_dir)},
            levels=[DISCOVERY_LEVEL],
            output_path=bank_path,
            teacher_model=teacher_model,
            skip_general=True,
        )
        progress["rounds"].append({"world": "POOLED", "bank": str(bank_path)})
    else:
        # Sequential: round 1 seeds from world[0] offline; then evolve per world.
        w0 = train_worlds[0]
        bank_path = output_dir / f"skill_bank_after_00_{w0}.json"
        w0_dir = (offline_traj_dirs or {}).get(w0, offline_traj_dir)

        # Round 1 is CONTRASTIVE, and no single world supplies both classes: at
        # sonnet every world is unanimous (gravity 6/6, three_species 0/2, ...),
        # so a single-world bootstrap gives the teacher successes with no
        # failures or the reverse. The contrast has to come from pooling across
        # TRAIN worlds -- which means the round-1 directory is often multi-world,
        # and labelling it with train_worlds[0]'s setup describes every other
        # episode in it wrongly. Same bug as the one fixed in `pooled` mode.
        boot_worlds = _worlds_in_dir(Path(w0_dir))
        if len(boot_worlds) > 1:
            leaked = [w for w in boot_worlds if w in held_out_worlds]
            if leaked:
                raise ValueError(
                    f"bootstrap dir {w0_dir} contains HELD-OUT world(s) {leaked}; "
                    "the round-1 bank would be distilled from the very worlds the "
                    "transfer test is supposed to be blind to."
                )
            _skcfg.LEVEL_DESCRIPTIONS[DISCOVERY_LEVEL] = pooled_world_facts(boot_worlds)
            print(f"\n=== Round 1: distill bank from POOLED offline "
                  f"({len(boot_worlds)} worlds: {', '.join(boot_worlds)}) ({w0_dir}) ===")
        else:
            _set_current_world_facts(w0)
            print(f"\n=== Round 1: distill bank from OFFLINE {w0} ({w0_dir}) ===")
        bank = _reuse_or(bank_path) or run_distillation(
            traj_dirs={DISCOVERY_LEVEL: Path(w0_dir)},
            levels=[DISCOVERY_LEVEL],
            output_path=bank_path,
            teacher_model=teacher_model,
            skip_general=True,
        )
        progress["rounds"].append({"world": w0, "phase": "offline_distill",
                                   "bank": str(bank_path)})

        # Cross-world contrast buffer: seeded with the bootstrap, then grown with
        # every completed round. Without it each evolve round sees one unanimous
        # world and has no contrast to reason from (6 of 8 rounds on the
        # 2026-09-12 run). Held-out worlds are refused, loudly.
        buffer = ReplayBuffer(held_out_worlds=held_out_worlds)
        buffer.add_dir(Path(w0_dir))
        print(f"  contrast buffer seeded with {len(buffer)} episode(s) "
              f"from {', '.join(buffer.worlds)}")

        for idx, w in enumerate(train_worlds[1:], start=1):
            print(f"\n=== World {idx}: {w} (run + evolve) ===")
            round_dir = output_dir / f"round_{idx:02d}_{w}"
            results = run_batch_discovery(
                w, seeds, bank, round_dir, **episode_kwargs
            )
            pr = _pass_rate(results)
            print(f"  {w} online pass-rate with current bank: {pr:.0%}")

            _set_current_world_facts(w)
            bank_path = output_dir / f"skill_bank_after_{idx:02d}_{w}.json"

            # Contrast batch = this round's episodes + backfill of the missing
            # class from earlier worlds. Written to its own directory so the
            # round's canonical episode record (which drives resume) stays clean.
            if cross_world_contrast:
                # ABLATION. Backfills the missing class from earlier worlds.
                # Measured to raise training-world reward and collapse per-seed
                # variance, but to DEGRADE transfer on ether (-0.264, -2.2 sigma),
                # because the only replayable successes come from central-force
                # worlds. Off by default: stock HeXA evolves on the round's own
                # episodes, and that is the method being reported.
                batch = buffer.select_contrast(
                    results, max_trajs=max_evolve_trajs,
                    success_slots=max(1, round(max_evolve_trajs * 0.5)),
                )
                print(f"  contrast batch: {describe_batch(batch)}")
                evolve_dir = buffer.write_evolve_dir(
                    batch, output_dir / f"evolve_input_{idx:02d}_{w}", DISCOVERY_LEVEL
                )
            else:
                batch = results
                evolve_dir = round_dir / DISCOVERY_LEVEL

            prev_bank = bank
            cached = _reuse_or(bank_path)
            bank = cached or evolve_skill_bank(
                level_name=DISCOVERY_LEVEL,
                prev_bank=bank,
                new_trajs_dir=evolve_dir,
                output_path=bank_path,
                max_skills=max_skills,
                max_mistakes=max_mistakes,
                teacher_model=teacher_model,
            )
            if cached is None:
                # Teacher-asserted confidence and erased lineage, both repaired
                # against the evidence actually in this batch.
                reground_bank(bank, prev_bank, batch, DISCOVERY_LEVEL)
                bank.save(bank_path)

            # Grow the buffer only AFTER selecting, so a round never contrasts
            # against itself.
            buffer.add(results)
            audit = audit_bank(bank, DISCOVERY_LEVEL, held_out_worlds)
            report_audit(audit, where=f"after {w}")
            progress["rounds"].append({
                "world": w, "phase": "run+evolve", "seeds": seeds,
                "online_pass_rate": pr, "bank": str(bank_path),
                "law_form_audit": audit,
            })
            (output_dir / "progress.json").write_text(json.dumps(progress, indent=2))

    final_bank_path = output_dir / "skill_bank_final.json"
    bank.save(final_bank_path)
    progress["final_bank"] = str(final_bank_path)

    # The frozen bank is what travels to the held-out worlds, so this is the
    # audit that decides whether the transfer number means anything.
    final_audit = audit_bank(bank, DISCOVERY_LEVEL, held_out_worlds)
    print("\n=== Law-form audit of the FINAL bank ===")
    report_audit(final_audit, where="final bank")
    progress["law_form_audit_final"] = final_audit
    # Flush before transfer starts. Transfer is the longest phase and the most
    # likely place to hit a CLI usage limit; without this the completed training
    # record and the final audit die with the process even though every bank file
    # is already on disk.
    (output_dir / "progress.json").write_text(json.dumps(progress, indent=2))

    # ── Transfer eval on held-out worlds: bank vs plain baseline ─────────
    transfer = {}
    for w in held_out_worlds:
        print(f"\n=== Transfer eval: {w} ===")
        eval_seeds = list(range(start_seed, start_seed + seeds_per_world))
        with_bank = run_batch_discovery(
            w, eval_seeds, bank,
            output_dir / "transfer" / w / "with_bank",
            **episode_kwargs,
        )
        baseline = run_batch_discovery(
            w, eval_seeds, None,
            output_dir / "transfer" / w / "baseline",
            **episode_kwargs,
        )
        s_bank, s_base = summarise(with_bank), summarise(baseline)
        transfer[w] = {
            # Kept under their original names so existing readers of
            # progress.json keep working.
            "with_bank_pass_rate": s_bank["pass_rate"],
            "baseline_pass_rate": s_base["pass_rate"],
            "with_bank": s_bank,
            "baseline": s_base,
            "delta_reward": (
                None if s_bank["mean_reward"] is None or s_base["mean_reward"] is None
                else s_bank["mean_reward"] - s_base["mean_reward"]
            ),
        }
        progress["transfer"] = transfer
        (output_dir / "progress.json").write_text(json.dumps(progress, indent=2))
        d = transfer[w]["delta_reward"]
        print(f"  {w}: pass {s_bank['pass_rate']:.0%} vs {s_base['pass_rate']:.0%}"
              f"   |   mean reward {s_bank['mean_reward']:+.3f} vs "
              f"{s_base['mean_reward']:+.3f}"
              + (f"  (delta {d:+.3f})" if d is not None else ""))

    # The unit of replication is the held-out WORLD (see summarise()), so the
    # across-world aggregate is the number to read, not any single world's.
    deltas = [t["delta_reward"] for t in transfer.values() if t["delta_reward"] is not None]
    progress["transfer"] = transfer
    progress["transfer_summary"] = {
        "n_worlds": len(transfer),
        "mean_delta_reward": _mean(deltas),
        "worlds_improved": sum(1 for d in deltas if d > 0),
        "worlds_worsened": sum(1 for d in deltas if d < 0),
        "mean_pass_rate_with_bank": _mean([t["with_bank_pass_rate"] for t in transfer.values()]),
        "mean_pass_rate_baseline": _mean([t["baseline_pass_rate"] for t in transfer.values()]),
    }
    (output_dir / "progress.json").write_text(json.dumps(progress, indent=2))
    print("\n=== DONE ===")
    print(json.dumps(transfer, indent=2))
    return progress


def main() -> None:
    p = argparse.ArgumentParser(description="HeXA cross-world skill evolution on DiscoverPhysics")
    p.add_argument("--offline-traj-dir", required=True,
                   help="Offline seed trajectories for the FIRST train world "
                        "(sequential) or the pooled set (pooled mode).")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--held-out-world", default=None,
                   help="Leave-one-world-out: this world is held out, the rest train.")
    p.add_argument("--held-out-worlds", nargs="+", default=None,
                   help="Explicit held-out set, for families the named groups do "
                        "not capture. Use when two worlds share a governing "
                        "operator and must be held out together — e.g. "
                        "'--held-out-worlds circle fractional', which share the "
                        "fractional Laplacian, so holding out only one trains on "
                        "the other's answer.")
    p.add_argument("--held-out-group", default=None,
                   help="Leave-one-group-out: this group is held out (overrides "
                        "--held-out-world).")
    p.add_argument("--train-worlds", nargs="+", default=None,
                   help="Explicit train-world list (overrides the auto split's "
                        "train set; held-out worlds are still removed). Use for a "
                        "cheap smoke run, e.g. --train-worlds gravity yukawa.")
    p.add_argument("--mode", choices=("sequential", "pooled"), default="sequential")
    p.add_argument("--pooled-traj-dirs", nargs="+", default=None,
                   help="Pooled mode only: several per-world trajectory dirs to "
                        "merge (e.g. .../gravity/discovery .../yukawa/discovery). "
                        "They are copied into <output-dir>/pooled_input/discovery "
                        "with globally unique seeds first, since skillrl's loader "
                        "reads ONE directory and seed numbers collide across "
                        "worlds. Omit to use --offline-traj-dir as an "
                        "already-pooled directory.")
    p.add_argument("--seeds-per-world", type=int, default=3)
    p.add_argument("--start-seed", type=int, default=100)
    p.add_argument("--noise-std", type=float, default=0.075)
    p.add_argument("--max-rounds", type=int, default=None)
    p.add_argument("--max-skills", type=int, default=12)
    p.add_argument("--max-mistakes", type=int, default=6)
    p.add_argument("--cross-world-contrast", action="store_true",
                   help="ABLATION: backfill each evolve round's batch with the "
                        "missing class from earlier train worlds. Off by default "
                        "— stock HeXA evolves on the round's own episodes. "
                        "Measured to degrade ether transfer by -2.2 sigma.")
    p.add_argument("--max-evolve-trajs", type=int, default=6,
                   help="Trajectories shown to the evolver each round: this "
                        "round's episodes plus cross-world backfill of whichever "
                        "class the round lacks. Half the slots are reserved for "
                        "successes.")
    p.add_argument("--model", default="claude-opus-4-6")
    p.add_argument("--judge-model", default="claude-opus-4-6")
    p.add_argument("--teacher-model", default="claude-sonnet-4-6",
                   help="Teacher model, passed to skillrl's own call_teacher "
                        "dispatcher (NOT our cli_client shim). MUST start with "
                        "'claude-' (e.g. 'claude-sonnet-4-6', 'claude-opus-4-6') "
                        "or it silently falls through to an HF/Qwen loader and "
                        "returns no skills. Bare CLI aliases like 'sonnet' do "
                        "NOT work here — that's only valid for --cli-model.")
    p.add_argument("--cli-model", default=None,
                   help="Force this model for the actor/judge/critic CLI calls "
                        "via our cli_client shim (bare aliases OK, e.g. 'sonnet', "
                        "'opus', 'haiku'). Unrelated to --teacher-model.")
    p.add_argument("--no-cli", action="store_true",
                   help="Use API providers instead of the Claude CLI.")
    p.add_argument("--use-critic", action="store_true")
    args = p.parse_args()

    if args.held_out_worlds:
        held = list(args.held_out_worlds)
        unknown = [w for w in held if w not in BENCHMARK_WORLDS]
        if unknown:
            raise SystemExit(f"unknown world(s): {unknown}")
        train = [w for w in BENCHMARK_WORLDS if w not in held]
    elif args.held_out_group:
        train, held = leave_one_group_out_split(args.held_out_group)
    elif args.held_out_world:
        train, held = leave_one_out_split(args.held_out_world)
    else:
        raise SystemExit("Specify --held-out-world, --held-out-worlds or --held-out-group")

    # Optional explicit train subset (cheap smoke runs); never train on a held-out world.
    if args.train_worlds:
        train = [w for w in args.train_worlds if w not in held]
        if not train:
            raise SystemExit("--train-worlds is empty after removing held-out worlds")

    episode_kwargs = dict(
        model=args.model, judge_model=args.judge_model,
        noise_std=args.noise_std, max_rounds=args.max_rounds,
        use_critic=args.use_critic,
        use_cli=not args.no_cli, cli_model=args.cli_model,
    )

    run_cross_world_loop(
        train_worlds=train,
        held_out_worlds=held,
        offline_traj_dir=Path(args.offline_traj_dir),
        output_dir=Path(args.output_dir),
        seeds_per_world=args.seeds_per_world,
        start_seed=args.start_seed,
        max_skills=args.max_skills,
        max_mistakes=args.max_mistakes,
        max_evolve_trajs=args.max_evolve_trajs,
        cross_world_contrast=args.cross_world_contrast,
        teacher_model=args.teacher_model,
        mode=args.mode,
        pooled_traj_dirs=args.pooled_traj_dirs,
        episode_kwargs=episode_kwargs,
    )


if __name__ == "__main__":
    main()
