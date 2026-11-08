"""HeXA's evolving loop over Pigey trials (Off2On when initial trials are given).

Round 1 starts from ``--initial-bank`` (a finished bank, copied in), or distills a
bank from ``--initial-trials``, or runs with no bank (the online start); every round runs each task on this round's init states
with the current bank appended to Pigey's system prompt, and the next round's bank
is evolved from those trials. Mirrors ``skillrl.loops.evolving_loop`` and
``online_evolving_loop``; the actor is ``run_claude.sh``, one subprocess per trial,
against servers the calling sbatch script already started.

Layout under --out-dir::

    initial/               symlinks to the initial trial.json files (staged once, --initial-trials)
    initial_bank_source.json  where bank_round_1.json was copied from (--initial-bank), with its sha256
    bank_round_<r>.json    the bank round r ran with (absent for an online round 1);
                           bank_round_<rounds+1>.json is the final bank, evolved from the last round
    round_<r>/             trial directories written by run_claude.py
    teacher/               every teacher prompt and response
    progress.json          one record per completed round

Resumable: an existing bank or completed trial is reused, not recomputed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

SETUP_ROOT = Path(__file__).resolve().parents[1]  # baselines/pigey_setup
if str(SETUP_ROOT) not in sys.path:
    sys.path.insert(0, str(SETUP_ROOT))

from hexa.trajectory import iter_trials, load_trial_trajectories, seed_of  # noqa: E402
from hexa import teacher as hexa_teacher  # noqa: E402

SETUP = SETUP_ROOT


def parse_tasks(spec: list[str]) -> list[int]:
    tasks = []
    for item in spec:
        if "-" in item:
            start, end = item.split("-")
            tasks.extend(range(int(start), int(end) + 1))
        else:
            tasks.append(int(item))
    return tasks


def stage_initial(dirs: list[str], staged: Path, suite: str) -> int:
    """Symlink every completed trial.json of ``suite`` under the given dirs into ``staged``."""
    staged.mkdir(parents=True, exist_ok=True)
    count = 0
    for source in dirs:
        for path, trial in iter_trials(source):
            if trial.get("suite") != suite or trial.get("status") == "backend_error":
                continue
            target = staged / f"{count:03d}_{seed_of(trial['task_id'], trial['episode'])}"
            target.mkdir(exist_ok=True)
            link = target / "trial.json"
            if not link.exists():
                link.symlink_to(path.resolve())
            count += 1
    return count


def seed_bank(source: Path, out: Path) -> Path:
    """Use a finished bank as round 1's bank: copy it to ``bank_round_1.json`` once.

    A resumed run must not silently continue from a different bank, so an existing
    ``bank_round_1.json`` has to match the source byte for byte.
    """
    target = out / "bank_round_1.json"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if target.exists():
        if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise SystemExit(f"{target} exists and differs from --initial-bank {source}; use another --out-dir")
        print(f"Initial bank already in place: {target}")
        return target
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    (out / "initial_bank_source.json").write_text(json.dumps(
        {"source": str(source.resolve()), "sha256": digest, "copied_at": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=2))
    print(f"Initial bank copied from {source} (sha256 {digest[:12]}...) to {target}")
    return target


def completed_trials(round_dir: Path, suite: str) -> dict[str, dict]:
    """``seed -> trial`` for trials in ``round_dir`` that finished without a backend error."""
    done = {}
    for _, trial in iter_trials(round_dir):
        if trial.get("suite") == suite and trial.get("status") != "backend_error" and trial.get("success") is not None:
            done[seed_of(trial["task_id"], trial["episode"])] = trial
    return done


def servers_alive(pid_files: list[str]) -> bool:
    for pid_file in pid_files:
        try:
            os.kill(int(Path(pid_file).read_text().strip()), 0)
        except (OSError, ValueError):
            print(f"[hexa] server behind {pid_file} is no longer running", file=sys.stderr)
            return False
    return True


def run_trial(args, task: int, episode: int, round_dir: Path, bank_path: Path | None, extra: list[str]) -> int:
    command = ["bash", str(SETUP / "run_claude.sh"), "--suite", args.suite, "--task", str(task), "--episode", str(episode),
               "--policy-host", args.policy_host, "--policy-port", str(args.policy_port),
               "--max-llm-steps", str(args.max_llm_steps), "--max-steps-per-rollout", str(args.max_steps_per_rollout),
               "--out-dir", str(round_dir)]
    if bank_path is not None:
        command += ["--skill-bank", str(bank_path), "--skill-level", args.suite,
                    "--skill-budget-tokens", str(args.injection_budget_tokens)]
    command += extra
    print(f"\n##### round trial task {task} episode {episode} ({time.strftime('%Y-%m-%d %H:%M:%S')})", flush=True)
    return subprocess.run(command).returncode


def trial_cost(trial: dict) -> float:
    return sum(call.get("total_cost_usd") or 0 for call in trial.get("claude_code_calls", []))


def build_bank(args, round_num: int, out: Path, teacher, suite: str) -> Path | None:
    """Create round ``round_num``'s bank, or return None for an online round 1."""
    from skillrl.core.skill_bank import SkillBank
    from skillrl.distillation.distill import run_distillation
    from skillrl.distillation.evolving_distill import evolve_skill_bank

    bank_path = out / f"bank_round_{round_num}.json"
    if bank_path.exists():
        print(f"\n  Phase 1: bank exists, reusing {bank_path}")
        return bank_path
    initial = out / "initial"
    has_initial = initial.exists() and any(initial.rglob("trial.json"))
    previous_bank = out / f"bank_round_{round_num - 1}.json"
    previous_round = out / f"round_{round_num - 1}"
    calls_before = len(teacher.calls)

    if round_num == 1:
        if not has_initial:
            print("\n  Phase 1: no initial trials; round 1 runs without a bank (online start)")
            return None
        print(f"\n  Phase 1: distilling the first bank from {initial}")
        run_distillation(traj_dirs={suite: initial}, levels=[suite], output_path=bank_path,
                         teacher_model=args.teacher_model, skip_general=True,
                         max_success_trajs=args.seed_trajs, max_failure_trajs=args.seed_trajs)
    elif not previous_bank.exists():
        # Online start: the first bank is distilled from round 1's own trials.
        print(f"\n  Phase 1: distilling the first bank from {previous_round}")
        run_distillation(traj_dirs={suite: previous_round}, levels=[suite], output_path=bank_path,
                         teacher_model=args.teacher_model, skip_general=True,
                         max_success_trajs=args.seed_trajs, max_failure_trajs=args.seed_trajs)
    else:
        print(f"\n  Phase 1: evolving {previous_bank} with trials from {previous_round}")
        evolve_skill_bank(level_name=suite, prev_bank=SkillBank.load(previous_bank), new_trajs_dir=previous_round,
                          output_path=bank_path, max_skills=args.max_skills, max_mistakes=args.max_mistakes,
                          teacher_model=args.teacher_model)
    if not bank_path.exists():
        raise RuntimeError(f"teacher produced no bank at {bank_path}; see {out / 'teacher'}")
    failed = [call for call in teacher.calls[calls_before:] if call.get("error")]
    if failed:
        # A failed teacher call leaves the previous bank in place under this round's
        # name; set it aside so a rerun builds this bank again instead of reusing it.
        aside = out / f"bank_round_{round_num}.teacher_error.json"
        bank_path.replace(aside)
        raise SystemExit(f"[hexa] stopping: {len(failed)} teacher call(s) failed while building round "
                         f"{round_num}'s bank ({failed[0]['error'][:200]}); kept the output as {aside.name}. "
                         "Rerunning rebuilds this bank.")
    return bank_path


def bank_summary(bank_path: Path | None, suite: str) -> dict:
    if bank_path is None:
        return {"skills": 0, "mistakes": 0}
    from skillrl.core.skill_bank import SkillBank

    bank = SkillBank.load(bank_path)
    block = hexa_teacher.render_block(bank, suite)
    return {"skills": len(bank.level_skills.get(suite, [])), "mistakes": len(bank.level_mistakes.get(suite, [])),
            "injected_skills": len(block.skills), "injected_mistakes": len(block.mistakes),
            "block_tokens": block.tokens, "titles": [skill.title for skill in block.skills]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog="Other options are forwarded to run_claude.sh (e.g. --model).")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--suite", default=hexa_teacher.DEFAULT_SUITE)
    parser.add_argument("--tasks", nargs="+", default=["0-9"], help="Task ids or ranges, e.g. 0-9 or 5 7")
    parser.add_argument("--episodes", nargs="+", type=int, default=list(range(10, 20)),
                        help="Pool of init states consumed in order, --episodes-per-round per round")
    parser.add_argument("--episodes-per-round", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--initial-bank", default=None,
                        help="A finished bank JSON used as round 1's bank (copied to bank_round_1.json)")
    parser.add_argument("--initial-trials", nargs="*", default=[],
                        help="Directories of finished trial.json files to distill round 1's bank from (Off2On); "
                             "with neither this nor --initial-bank, round 1 runs without a bank (online start)")
    parser.add_argument("--teacher-model", default=os.environ.get("PIGEY_TEACHER_MODEL", hexa_teacher.DEFAULT_TEACHER_MODEL))
    parser.add_argument("--max-skills", type=int, default=10)
    parser.add_argument("--max-mistakes", type=int, default=5)
    parser.add_argument("--max-trajs", type=int, default=hexa_teacher.DEFAULT_MAX_TRAJS, help="Trials shown to the evolver")
    parser.add_argument("--seed-trajs", type=int, default=6, help="Successes and failures each shown to the round-1 distiller")
    parser.add_argument("--injection-budget-tokens", type=int, default=hexa_teacher.DEFAULT_INJECTION_BUDGET,
                        help="Rendered-block cap; 4000 fits a full bank of 10 skills and 5 mistakes (about 3400 tokens)")
    parser.add_argument("--policy-host", default="localhost")
    parser.add_argument("--policy-port", type=int, default=int(os.environ.get("PIGEY_POLICY_PORT", "0")))
    parser.add_argument("--pid-files", nargs="*", default=[], help="Server pid files checked before every trial")
    parser.add_argument("--max-llm-steps", type=int, default=35)
    parser.add_argument("--max-steps-per-rollout", type=int, default=500)
    parser.add_argument("--distill-only", action="store_true", help="Build (or copy) round 1's bank and stop")
    parser.add_argument("--min-round-fraction", type=float, default=0.5,
                        help="Stop, without recording the round or evolving from it, when fewer than this share "
                             "of its trials finish (backend errors); rerunning resumes the round")
    args, extra = parser.parse_known_args(argv)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    suite = args.suite
    tasks = parse_tasks(args.tasks)
    teacher = hexa_teacher.install(level_name=suite, teacher_log_dir=out / "teacher", max_trajs=args.max_trajs,
                                   executable=os.environ.get("CLAUDE_BINARY", "claude"))

    if args.initial_bank and args.initial_trials:
        parser.error("--initial-bank replaces --initial-trials; give one of them")
    if args.initial_bank:
        seed_bank(Path(args.initial_bank), out)
    if args.initial_trials:
        staged = stage_initial(args.initial_trials, out / "initial", suite)
        successes, failures = load_trial_trajectories(out / "initial", suite)
        print(f"Initial trials staged: {staged} ({len(successes)} successes, {len(failures)} failures)")

    progress_path = out / "progress.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else []
    done_rounds = {record["round"] for record in progress}

    for round_num in range(1, args.rounds + 1):
        print(f"\n{'#' * 70}\n  ROUND {round_num}/{args.rounds}  ({suite}, tasks {tasks})\n{'#' * 70}")
        bank_path = build_bank(args, round_num, out, teacher, suite)
        print(f"  Bank: {json.dumps(bank_summary(bank_path, suite))}")
        if args.distill_only:
            print(f"\n--distill-only: bank written to {bank_path}; teacher calls: {json.dumps(teacher.calls, indent=1)}")
            return 0
        if round_num in done_rounds:
            print(f"  Round {round_num} already recorded in {progress_path}; skipping its trials")
            continue

        start = (round_num - 1) * args.episodes_per_round
        episodes = args.episodes[start:start + args.episodes_per_round]
        if not episodes:
            print(f"  Init-state pool exhausted before round {round_num}; stopping")
            break
        if not args.policy_port:
            parser.error(f"--policy-port (or PIGEY_POLICY_PORT) is required to run round {round_num}'s trials")
        round_dir = out / f"round_{round_num}"
        round_dir.mkdir(exist_ok=True)
        done = completed_trials(round_dir, suite)
        failed_runs = []
        round_start = time.time()
        for task in tasks:
            for episode in episodes:
                if seed_of(task, episode) in done:
                    print(f"  trial {seed_of(task, episode)} already complete; skipping")
                    continue
                if args.pid_files and not servers_alive(args.pid_files):
                    print("[hexa] stopping: a server died", file=sys.stderr)
                    return 1
                code = run_trial(args, task, episode, round_dir, bank_path, extra)
                if code:
                    failed_runs.append({"task": task, "episode": episode, "exit": code})

        trials = completed_trials(round_dir, suite)
        planned = len(tasks) * len(episodes)
        if len(trials) < args.min_round_fraction * planned:
            print(f"[hexa] stopping: round {round_num} finished only {len(trials)}/{planned} trials "
                  f"({len(failed_runs)} failed runs), below --min-round-fraction {args.min_round_fraction}; "
                  "the round is not recorded and no bank is evolved from it. Rerunning resumes its missing trials.",
                  file=sys.stderr)
            return 1
        wins = sum(1 for trial in trials.values() if trial.get("success"))
        record = {"round": round_num, "bank": str(bank_path) if bank_path else None, "episodes": episodes, "tasks": tasks,
                  "trials": len(trials), "successes": wins, "accuracy": wins / len(trials) if trials else 0.0,
                  "per_task": {str(task): [bool(trial.get("success")) for seed, trial in sorted(trials.items())
                                           if trial["task_id"] == task] for task in tasks},
                  "actor_cost_usd": round(sum(trial_cost(trial) for trial in trials.values()), 3),
                  "teacher_calls": teacher.calls[:], "failed_runs": failed_runs,
                  "elapsed_s": round(time.time() - round_start, 1)}
        teacher.calls.clear()
        progress.append(record)
        progress_path.write_text(json.dumps(progress, indent=2))
        print(f"\n  Round {round_num}: {wins}/{len(trials)} = {record['accuracy']:.0%}, actor ${record['actor_cost_usd']:.2f}, "
              f"{len(failed_runs)} failed run(s), {record['elapsed_s'] / 60:.1f} min")

    if args.rounds in {record["round"] for record in progress}:
        # The last round's trials feed one more evolution, so the bank handed to
        # the frozen eval has seen every training episode. Needs only the teacher.
        print(f"\n{'#' * 70}\n  FINAL EVOLUTION (bank for the frozen eval)\n{'#' * 70}")
        final_bank = build_bank(args, args.rounds + 1, out, teacher, suite)
        print(f"  Final bank: {final_bank}\n  {json.dumps(bank_summary(final_bank, suite))}")
        if teacher.calls:
            (out / "final_evolution.json").write_text(json.dumps(teacher.calls, indent=2))
            teacher.calls.clear()

    print(f"\n{'=' * 70}\n  HEXA LOOP COMPLETE\n{'=' * 70}")
    for record in progress:
        print(f"    Round {record['round']}: {record['successes']}/{record['trials']} = {record['accuracy']:.0%} "
              f"(bank: {record['bank'] or 'none'})")
    total = sum(record["trials"] for record in progress)
    wins = sum(record["successes"] for record in progress)
    if total:
        print(f"\n  Overall: {wins}/{total} = {wins / total:.0%}")
    print(f"  Banks and progress: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
