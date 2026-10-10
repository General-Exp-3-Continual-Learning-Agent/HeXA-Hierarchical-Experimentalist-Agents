#!/usr/bin/env python
"""Reflexion baseline on ORBIT's Maze and Mastermind.

Reflexion (Shinn et al., 2023) as a trial-level wrapper: play the whole
instance, and if it was not solved cleanly, have the actor write a short
self-reflection on its own transcript and replay the instance from scratch with
that reflection -- and nothing else -- in the system prompt. See
`baselines/orbit/reflexion.py` for why a trial is a whole instance here rather
than an episode.

    trial 1 -> reflect -> trial 2      (--k-trials 2, the paper's default)

Trial 1 injects nothing, so it is a no-bank baseline replicate produced inside
this run. Every reflexion claim is therefore a paired within-run contrast, not a
difference against a separately scheduled arm.

There is **no build pass**: Reflexion's memory is per-task and is discarded at
the instance boundary, so unlike Episodic RAG and MemRL there is nothing to
construct before evaluating.

    # server up first (Qwen3-8B needs all three overrides):
    #   MODEL_DIR=/datasets/ai/qwen3/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218 \
    #   SERVED_MODEL_NAME=Qwen/Qwen3-8B REASONING_PARSER=qwen3 \
    #     sbatch scripts/serve_qwen_vllm.sh

    python scripts/run_orbit_reflexion.py --task mastermind \
        --model Qwen/Qwen3-8B --base-url http://gpu013:8000/v1 \
        --max-response-tokens 16384 --out results_orbit/orbit_mastermind_reflexion_qwen8/run0

    # plumbing check -- no server, no GPU, oracle plays the maze optimally:
    python scripts/run_orbit_reflexion.py --task maze --n 3 --stub --out /tmp/smoke
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hexa_core.tokens import TOKENIZER_NOTE  # noqa: E402
from hexa_orbit import loop, oracle, report, trajectory  # noqa: E402
from hexa_orbit.actor import ActorConfig, InstanceRun, MazeActor  # noqa: E402

from baselines.memory.blocks import count_tokens  # noqa: E402
from baselines.orbit import reflexion as rfx  # noqa: E402
from baselines.orbit.rollout import format_verifier_feedback  # noqa: E402
from baselines.orbit.tasks import TASKS, build_task  # noqa: E402


class OracleStub:
    """Plays the known-map optimum. Maze only; exercises the plumbing sans GPU."""

    def __init__(self, instance):
        self.instance = instance
        self.queue: list[str] = []

    def generate_structured(self, *, messages=None, **_):
        from hexa_core.llm.base import Generation

        if not self.queue:
            self.queue = oracle.instance_optimal_actions(self.instance)
        if not self.queue:
            return Generation.from_parts("", oracle.boxed("up"), finish_reason="stop")
        return Generation.from_parts("", oracle.boxed(self.queue.pop(0)), finish_reason="stop")


class ReflectionStub:
    """A canned reflection, so --stub covers the retry path too."""

    def generate_structured(self, **_):
        from hexa_core.llm.base import Generation

        return Generation.from_parts("", "Stub lesson: try the other branch first.",
                                     finish_reason="stop")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=TASKS, required=True)
    ap.add_argument("--n", type=int, default=None,
                    help="eval instances. Default is the task's full pool: maze 256 "
                         "(243 distinct layouts), mastermind 40.")
    ap.add_argument(
        "--variant", default="episode-reset",
        choices=["episode-reset", "episode-keep", "trials"],
        help="where the Reflexion trial boundary sits. 'episode-reset' (the "
             "reported arm) reflects between ORBIT's own episodes and wipes the "
             "chat, so the lesson is the only thing that crosses -- 3 episodes = "
             "3 trials. 'episode-keep' is its ablation: same reflections, "
             "transcript retained, which isolates analysis from memory. 'trials' "
             "replays the whole instance K times instead. See "
             "baselines/orbit/reflexion.py.")
    ap.add_argument("--k-trials", type=int, default=2,
                    help="attempts per instance, --variant trials only. The "
                         "paper's headline uses 2. Ignored by the episode variants, "
                         "where the episode count is ORBIT's (3).")
    ap.add_argument(
        "--reflect-on", choices=["imperfect", "unsolved"], default="imperfect",
        help="which trials earn a reflection. 'imperfect' = R<3/3, the generous "
             "reading: an instance solved at episode 3 after two failures still has "
             "a lesson in it. 'unsolved' = R==0, the strict binary the paper uses. "
             "Recorded in the manifest either way -- it changes how many instances "
             "carry a reflection into trial 2, so it is not a free knob.")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--train-size", type=int, default=None,
                    help="pool used for the maze leakage split (default 512 maze / 80 mastermind)")

    ap.add_argument("--model", default=ActorConfig.model)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--temperature", type=float, default=ActorConfig.temperature)
    ap.add_argument("--top-p", type=float, default=ActorConfig.top_p)
    ap.add_argument("--max-response-tokens", type=int, default=ActorConfig.max_response_tokens,
                    help="Qwen3-8B arms in MAZE_RESULTS.md / MASTERMIND_RESULTS.md use "
                         "16384 on BOTH tasks; the 2k default truncates ~5%% of turns "
                         "and truncation_scoring=exclude then drops them silently.")
    ap.add_argument("--reflection-max-tokens", type=int,
                    default=rfx.DEFAULT_REFLECTION_MAX_TOKENS)
    ap.add_argument("--reasoning-effort", default=ActorConfig.reasoning_effort,
                    help="pass '' for models with no reasoning channel")
    ap.add_argument("--seed", type=int, default=None,
                    help="per-call sampling seed. Mastermind replicates use "
                         "0 / 100000 / 200000, NOT 0/1/2.")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--transcript-mode", choices=["final", "raw"], default="final")
    ap.add_argument("--trajectory-token-budget", type=int, default=None)
    ap.add_argument("--truncation-scoring", choices=["exclude", "orbit"], default="exclude")
    ap.add_argument("--maze-config", default=None,
                    help="maze only. Default is the historical cap-45 pool, which is "
                         "what the Qwen no-bank arms this is paired against used. The "
                         "cap-27 pool is a DIFFERENT eval set; never pool the two.")
    ap.add_argument("--stub", action="store_true", help="oracle actor, canned reflections")
    ap.add_argument("--resume", action="store_true",
                    help="reuse completed trials and reflections already on disk")
    return ap.parse_args()


def _slurm_job_alive(job_id: str) -> Optional[bool]:
    """Is that Slurm job still queued or running? ``None`` = cannot tell."""
    if not job_id:
        return None
    try:
        out = subprocess.run(["squeue", "-h", "-j", job_id, "-o", "%T"],
                             capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        # squeue exits non-zero for an unknown job, which is exactly the
        # "finished long ago" case -- but also for a dead controller, which is
        # not. Only the explicit "invalid job id" wording is safe to read as gone.
        return False if "Invalid job id" in (out.stderr or "") else None
    return bool(out.stdout.strip())


def _writer_lock(out: Path) -> Path:
    """Refuse to share an --out directory with another live process.

    Two writers silently interleave into one set of trial files and the arm has
    to be discarded; this happened once already on a MemRL build. A filesystem
    lock rather than a pgrep guard, because the collision was a login-node chain
    against a compute node and pgrep sees only the local node.

    The lock records its Slurm job id so a **walltime kill can be distinguished
    from a live writer**. SIGKILL runs no cleanup, so without this every
    resumable arm would come back to a lock nobody holds and refuse to start --
    which trains you to delete locks by reflex, and the one time it mattered you
    would delete a real one. A lock whose job is definitively gone is taken over
    and said so; anything undecidable still stops the run.
    """
    out.mkdir(parents=True, exist_ok=True)
    lock = out / ".writer.lock"
    job_id = os.environ.get("SLURM_JOB_ID", "")
    payload = json.dumps({
        "node": os.uname().nodename, "pid": os.getpid(), "slurm_job_id": job_id,
    })
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raw = lock.read_text().strip() if lock.exists() else ""
        try:
            held = json.loads(raw)
        except (ValueError, TypeError):
            held = {}
        alive = _slurm_job_alive(held.get("slurm_job_id", ""))
        if alive is False:
            print(f"[lock] taking over {lock}: Slurm job "
                  f"{held.get('slurm_job_id')} is gone (walltime kill or crash)",
                  flush=True)
            lock.write_text(payload)
            return lock
        raise SystemExit(
            f"FATAL: {lock} is held by {raw or 'an unreadable lock'}.\n"
            + ("Its Slurm job is still live -- another process is writing here.\n"
               if alive else
               "Could not determine whether that job is alive. Check it, then rm "
               "the lock only if you are sure.\n")
        )
    with os.fdopen(fd, "w") as handle:
        handle.write(payload)
    return lock


def main() -> int:  # noqa: PLR0915
    args = parse_args()
    if args.k_trials < 1:
        raise SystemExit("--k-trials must be >= 1")
    if args.task != "maze" and args.maze_config:
        raise SystemExit("--maze-config applies to --task maze only")
    if args.stub and args.task != "maze":
        # `oracle.instance_optimal_actions` reads a maze map; there is no
        # Mastermind equivalent, and a stub that silently played badly would
        # look like a working smoke test.
        raise SystemExit("--stub is maze-only; use baselines/tests/test_orbit_reflexion.py")

    config = ActorConfig(
        model=args.model,
        base_url=args.base_url,
        temperature=args.temperature,
        top_p=args.top_p,
        max_response_tokens=args.max_response_tokens,
        # '' from the CLI means "no reasoning channel"; it must become None or
        # VLLMClient sends reasoning_effort="".
        reasoning_effort=args.reasoning_effort or None,
        transcript_mode=args.transcript_mode,
        trajectory_token_budget=args.trajectory_token_budget,
        truncation_scoring=args.truncation_scoring,
        seed=args.seed,
    )

    task_kwargs = {}
    if args.task == "maze" and args.maze_config:
        from hexa_orbit.adapter import MazeTask  # noqa: PLC0415

        orbit_task = build_task("maze")
        orbit_task.task = MazeTask(config_name=args.maze_config)
    else:
        orbit_task = build_task(args.task, **task_kwargs)

    n = args.n if args.n is not None else (256 if args.task == "maze" else 40)
    train_size = args.train_size if args.train_size is not None else (512 if args.task == "maze" else 80)
    eval_specs = orbit_task.eval_specs(n)
    train_specs = orbit_task.train_specs(train_size)
    orbit_task.assert_disjoint(train_specs, eval_specs)
    leaked_ids = orbit_task.leaked_ids(eval_specs, train_specs)
    print(f"{len(eval_specs)} eval instances; {len(leaked_ids)} share content with the "
          f"{len(train_specs)}-instance train pool")

    if args.variant != "trials" and args.k_trials != 2:
        print(f"note: --k-trials {args.k_trials} ignored; --variant "
              f"{args.variant} uses ORBIT's own 3 episodes as its trials")

    lock = _writer_lock(args.out)
    try:
        if args.variant == "trials":
            return _run(args, config, orbit_task, eval_specs, leaked_ids)
        return _run_episode(args, config, orbit_task, eval_specs, leaked_ids)
    finally:
        lock.unlink(missing_ok=True)


def _reflection_client(args, config):
    """The reflector. Same model as the actor, always.

    HExA is self-improvement rather than distillation, and every arm in these
    tables runs actor == teacher; a Reflexion baseline that reflected with a
    stronger model would be measuring a different thing and would not be
    comparable to any of them.
    """
    if args.stub:
        return ReflectionStub()
    from hexa_core.llm.vllm_client import VLLMClient  # noqa: PLC0415

    return VLLMClient(
        model=config.model,
        base_url=config.base_url,
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=args.reflection_max_tokens,
        reasoning_effort=config.reasoning_effort,
        seed=config.seed,
        raise_on_exhausted=False,
    )


def _run_episode(args, config, orbit_task, eval_specs, leaked_ids) -> int:
    """The episode-level variants: one pass, reflection inside the instance.

    A single `play_batch` -- there is no outer trial loop, because the retry
    structure is ORBIT's own three episodes and the reflection happens at the
    boundary between them (`EpisodeReflexionActor.on_episode_boundary`). One
    eval pass, so this costs roughly what the no-bank arm costs plus two short
    reflection calls per instance.
    """
    started = time.monotonic()
    reflector = _reflection_client(args, config)
    reset_context = args.variant == "episode-reset"

    runs_path = args.out / "runs.jsonl"
    partial_path = args.out / "runs.partial.jsonl"

    # The actors are the only place the reflection log lives, and `play_batch`
    # does not hand them back, so keep a reference as each is built.
    actors: dict[str, Any] = {}
    actors_lock = threading.Lock()

    def actor_factory(_guidelines, instance=None):
        actor = rfx.EpisodeReflexionActor(
            config, reflector=reflector, task_name=orbit_task.name,
            reset_context=reset_context,
            reflection_max_tokens=args.reflection_max_tokens,
        )
        if args.stub and instance is not None:
            actor._client = OracleStub(instance)
        if instance is not None:
            with actors_lock:
                actors[instance.id] = actor
        return actor

    def progress(position: int, run: InstanceRun) -> None:
        tag = ("ERROR: " + str(run.error)) if run.error else (
            f"R={run.success_count}/3 eps={run.episode_successes} "
            f"tok={run.added_tokens}")
        print(f"[{position}/{len(eval_specs)}] {run.instance_id} {tag}", flush=True)

    done_runs: list[InstanceRun] = []
    specs_todo = list(eval_specs)
    carry = ""
    if args.resume and partial_path.exists():
        done_runs = [r for r in trajectory.read_runs(partial_path) if r.error is None]
        done_ids = {r.instance_id for r in done_runs}
        specs_todo = [s for s in eval_specs if s.instance_id not in done_ids]
        trajectory.write_runs(partial_path, done_runs)
        carry = partial_path.read_text()
        print(f"resuming: {len(done_runs)} done, {len(specs_todo)} remaining")

    print(f"\n=== {orbit_task.name} reflexion ({args.variant}), "
          f"{len(specs_todo)} instances ===", flush=True)
    runs, errored = loop.play_batch(
        orbit_task.task, specs_todo, actor_factory, None,
        leaked_ids=leaked_ids, on_instance=progress, concurrency=args.concurrency,
        partial_path=partial_path, carry_over=carry,
    )
    runs = done_runs + runs
    trajectory.write_runs(runs_path, runs + errored)

    # ── reflection accounting ───────────────────────────────────────────────
    # `n_empty` is the diagnostic to read before any success rate: an episode
    # whose reflection came back empty ran with no block, so it is the no-bank
    # condition rather than the method.
    logged = 0
    n_empty = 0
    tokens_seen: list[int] = []
    with (args.out / "reflections.jsonl").open("w", encoding="utf-8") as handle:
        for instance_id, actor in sorted(actors.items()):
            for entry in actor.reflection_log:
                logged += 1
                if entry["empty"]:
                    n_empty += 1
                else:
                    tok = count_tokens(entry["reflection"])
                    if tok:
                        tokens_seen.append(tok)
                handle.write(json.dumps({"instance_id": instance_id, **entry,
                                         "tokens": count_tokens(entry["reflection"])},
                                        ensure_ascii=False) + "\n")
    reflection_stats = {
        "n_reflection_points": logged,
        "n_empty": n_empty,
        "mean_tokens": round(sum(tokens_seen) / len(tokens_seen), 1) if tokens_seen else 0,
        "max_tokens": max(tokens_seen) if tokens_seen else 0,
        "mean_per_instance": round(logged / len(runs), 2) if runs else 0,
    }

    aggregated = report.aggregate(runs, errored=errored)
    summary = {
        "arm": "reflexion",
        "variant": args.variant,
        "task": orbit_task.name,
        "reset_context": reset_context,
        "actor": config.to_dict(),
        "reflector": {"model": config.model,
                      "max_tokens": args.reflection_max_tokens,
                      "note": "same model as the actor, always"},
        "stub": args.stub,
        "n_requested": len(eval_specs),
        "concurrency": args.concurrency,
        "wall_clock_s": round(time.monotonic() - started, 1),
        "report": aggregated,
        "reflection_stats": reflection_stats,
        "manifest": orbit_task.manifest(),
        "token_note": TOKENIZER_NOTE,
        "deviations": [
            "A Reflexion trial is one ORBIT episode. Episode 1 injects nothing "
            "and is byte-identical to the no-bank baseline.",
            ("episode-reset: the chat is truncated to the system message at each "
             "episode boundary, so the reflection is the only thing that crosses. "
             "This REMOVES the cumulative transcript the no-bank arm keeps, which "
             "is what makes it Reflexion rather than the paper's trajectory-only "
             "ablation -- and is why episode-keep exists to separate the two.")
            if reset_context else
            ("episode-keep: ORBIT's cumulative transcript is retained, so the "
             "reflection adds analysis rather than memory. This is the paper's "
             "trajectory-only ablation and is expected to move little; it is the "
             "control for episode-reset, not a headline arm."),
            "Reflection happens after a FAILED episode; an episode outcome is "
            "binary, so there is no reflect_on choice to make.",
        ],
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))

    print()
    print(report.format_summary(
        aggregated, title=f"{orbit_task.name} reflexion ({args.variant})"))
    print(f"\nreflections: {logged} points, {n_empty} empty, "
          f"mean {reflection_stats['mean_tokens']:.0f} tokens, "
          f"{reflection_stats['mean_per_instance']} per instance")
    if n_empty:
        print(f"WARNING: {n_empty}/{logged} reflections were EMPTY -- those "
              f"episodes ran as no-bank.")
    print(f"wrote {args.out / 'summary.json'}")
    return 0


def _run(args, config, orbit_task, eval_specs, leaked_ids) -> int:  # noqa: PLR0915
    started = time.monotonic()
    reflections: dict[str, list[str]] = {}
    reflections_path = args.out / "reflections.jsonl"
    if args.resume and reflections_path.exists():
        for line in reflections_path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            reflections.setdefault(rec["instance_id"], []).append(rec["reflection"])

    reflector = _reflection_client(args, config)
    trials: list[dict] = []
    # Per-trial reflection accounting. `n_empty` is the diagnostic that must be
    # read before any success rate: an instance whose reflection came back empty
    # ran the next trial as no-bank, so it belongs to the control, not the
    # method. Same role `retrieval.n_empty_block` serves for the memory
    # baselines (docs/MEMORY_BASELINES_ORBIT.md).
    reflection_stats: list[dict] = []
    all_runs: dict[int, list[InstanceRun]] = {}

    for trial in range(args.k_trials):
        trial_dir = args.out / f"trial{trial}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        runs_path = trial_dir / "runs.jsonl"
        partial_path = trial_dir / "runs.partial.jsonl"

        if args.resume and runs_path.exists():
            runs = [r for r in trajectory.read_runs(runs_path) if r.error is None]
            errored = [r for r in trajectory.read_runs(runs_path) if r.error is not None]
            print(f"\n=== trial {trial + 1}/{args.k_trials}: reusing {len(runs)} completed runs ===")
        else:
            # Per-instance guidelines: trial 1 gets None everywhere (byte-identical
            # to no-bank), later trials get that instance's own accumulated lessons.
            blocks = {
                spec.instance_id: rfx.reflexion_block(reflections.get(spec.instance_id, []))
                for spec in eval_specs
            }
            n_with = sum(1 for b in blocks.values() if b)
            print(f"\n=== trial {trial + 1}/{args.k_trials}: "
                  f"{n_with}/{len(eval_specs)} instances carry a reflection ===", flush=True)

            def actor_factory(_guidelines, instance=None, _blocks=blocks):
                block = _blocks.get(getattr(instance, "id", None)) if instance else None
                actor = MazeActor(config, guidelines=block)
                if args.stub and instance is not None:
                    actor._client = OracleStub(instance)
                return actor

            def progress(position: int, run: InstanceRun, _t=trial) -> None:
                tag = "ERROR: " + str(run.error) if run.error else (
                    f"R={run.success_count}/3 eps={run.episode_successes}")
                print(f"[t{_t + 1} {position}/{len(eval_specs)}] {run.instance_id} {tag}",
                      flush=True)

            done_runs: list[InstanceRun] = []
            specs_todo = list(eval_specs)
            carry = ""
            if args.resume and partial_path.exists():
                done_runs = [r for r in trajectory.read_runs(partial_path) if r.error is None]
                done_ids = {r.instance_id for r in done_runs}
                specs_todo = [s for s in eval_specs if s.instance_id not in done_ids]
                trajectory.write_runs(partial_path, done_runs)
                carry = partial_path.read_text()
                print(f"  resuming mid-trial: {len(done_runs)} done, {len(specs_todo)} left")

            runs, errored = loop.play_batch(
                orbit_task.task, specs_todo, actor_factory, None,
                leaked_ids=leaked_ids, on_instance=progress,
                concurrency=args.concurrency,
                partial_path=partial_path, carry_over=carry,
            )
            runs = done_runs + runs
            trajectory.write_runs(runs_path, runs + errored)

        all_runs[trial] = runs
        aggregated = report.aggregate(runs, errored=errored)
        trials.append({"trial": trial, "report": aggregated,
                       "n_with_reflection": sum(
                           1 for s in eval_specs if reflections.get(s.instance_id))})
        print()
        print(report.format_summary(
            aggregated, title=f"{orbit_task.name} reflexion trial {trial + 1}"))

        # ── reflect, unless this was the last trial ──────────────────────────
        if trial == args.k_trials - 1:
            break

        def needs_reflection(run: InstanceRun) -> bool:
            if args.reflect_on == "unsolved":
                return run.success_count == 0
            return run.success_count < 3

        todo = [r for r in runs if needs_reflection(r)
                and len(reflections.get(r.instance_id, [])) <= trial]
        print(f"\nreflecting on {len(todo)}/{len(runs)} instances "
              f"({args.reflect_on})...", flush=True)

        def do_reflect(run: InstanceRun) -> tuple[str, str]:
            result = dict(orbit_task.to_trajectory(run))
            feedback = format_verifier_feedback(result, orbit_task.name)
            text = rfx.reflect(
                reflector, orbit_task.name, run, feedback,
                reflections.get(run.instance_id, []),
                max_new_tokens=args.reflection_max_tokens,
            )
            return run.instance_id, text

        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            produced = list(pool.map(do_reflect, todo))

        n_empty = 0
        with reflections_path.open("a", encoding="utf-8") as handle:
            for instance_id, text in produced:
                if not text:
                    n_empty += 1
                reflections.setdefault(instance_id, []).append(text)
                handle.write(json.dumps({
                    "instance_id": instance_id, "after_trial": trial,
                    "reflection": text, "tokens": count_tokens(text),
                }, ensure_ascii=False) + "\n")
                handle.flush()
        kept = [t for _, t in produced if t]
        tok = [count_tokens(t) or 0 for t in kept]
        reflection_stats.append({
            "after_trial": trial,
            "n_eligible": len(todo),
            "n_reflections": len(kept),
            "n_empty": n_empty,
            "mean_tokens": round(sum(tok) / len(tok), 1) if tok else 0,
            "max_tokens": max(tok) if tok else 0,
        })
        print(f"  {len(kept)} reflections, {n_empty} empty "
              f"(those instances are the no-bank condition in trial {trial + 2})")
        if tok:
            print(f"  reflection tokens: mean {sum(tok) / len(tok):.0f}, max {max(tok)}")

    # ── cumulative best-of-K, the paper's headline curve ────────────────────
    by_instance: dict[str, list[InstanceRun]] = {}
    for trial_runs in all_runs.values():
        for run in trial_runs:
            by_instance.setdefault(run.instance_id, []).append(run)
    solved_any = sum(1 for rs in by_instance.values()
                     if any(any(r.episode_successes) for r in rs))
    cumulative = {
        "n_instances": len(by_instance),
        "any_success_over_trials": solved_any / len(by_instance) if by_instance else 0.0,
    }

    summary = {
        "arm": "reflexion",
        "variant": "trials",
        "task": orbit_task.name,
        "k_trials": args.k_trials,
        "reflect_on": args.reflect_on,
        "actor": config.to_dict(),
        "reflector": {"model": config.model,
                      "max_tokens": args.reflection_max_tokens,
                      "note": "same model as the actor, always"},
        "stub": args.stub,
        "n_requested": len(eval_specs),
        "concurrency": args.concurrency,
        "wall_clock_s": round(time.monotonic() - started, 1),
        "trials": trials,
        "reflection_stats": reflection_stats,
        "cumulative": cumulative,
        "manifest": orbit_task.manifest(),
        "token_note": TOKENIZER_NOTE,
        "deviations": [
            "A Reflexion trial is a whole ORBIT instance (3 episodes), not one "
            "episode: ORBIT never resets the chat between episodes, so an "
            "episode-level reflection would explain a transcript the model is "
            "already reading -- the paper's trajectory-only ablation, not the method.",
            f"reflect_on={args.reflect_on}: the paper's success signal is binary; an "
            "ORBIT instance is graded 0-3.",
            "Trial 1 injects nothing and is reported as a no-bank replicate.",
        ],
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 60)
    for entry in trials:
        pooled = entry["report"]["pooled"]
        rates = pooled.get("episode_success_rate", {})
        print(f"trial {entry['trial'] + 1}: n={pooled.get('n')} "
              f"ep1={rates.get('episode_1', 0):.3f} ep2={rates.get('episode_2', 0):.3f} "
              f"ep3={rates.get('episode_3', 0):.3f} any={pooled.get('any_success_rate', 0):.3f} "
              f"({entry['n_with_reflection']} carried a reflection)")
    for stat in reflection_stats:
        if stat["n_empty"]:
            print(f"WARNING: {stat['n_empty']}/{stat['n_eligible']} reflections after "
                  f"trial {stat['after_trial'] + 1} were EMPTY -- those instances ran "
                  f"the next trial as no-bank.")
    print(f"best-of-{args.k_trials} any: {cumulative['any_success_over_trials']:.3f}")
    print(f"\nwrote {args.out / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
