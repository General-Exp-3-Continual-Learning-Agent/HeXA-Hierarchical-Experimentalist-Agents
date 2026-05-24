"""Per-seed Reflexion trial loop.

Wraps `ReactAgent.solve()` with K trials and a self-reflection step between
trials. The reflection memory is injected into the system prompt for trial t+1
by monkey-patching `build_system_prompt` in the `react_agent.react_agent`
module namespace for the duration of the trial. This avoids any change to the
core agent loop or the level prompt files.
"""

import contextlib
import json
import os
import time

import cv2

from react_agent import react_agent as _ra_module
from react_agent.react_agent import ReactAgent
from react_agent.tools import InterphyreToolkit

from react_agent.reflexion.memory import ReflexionMemory
from react_agent.reflexion.reflect import reflect_on_trajectory


@contextlib.contextmanager
def _inject_reflections_into_system_prompt(memory: ReflexionMemory):
    """Monkey-patch `react_agent.react_agent.build_system_prompt` so the next
    `ReactAgent.solve()` call appends the Reflexion memory block to the system
    prompt. Restored on exit.
    """
    orig_build_system = _ra_module.build_system_prompt
    block = memory.format_as_block()

    def patched_build_system(level_name, is_oss=False):
        base = orig_build_system(level_name, is_oss=is_oss)
        if not block:
            return base
        return base + "\n" + block

    _ra_module.build_system_prompt = patched_build_system
    try:
        yield
    finally:
        _ra_module.build_system_prompt = orig_build_system


def _atomic_write_json(path: str, data: dict) -> None:
    """Write JSON atomically: write to .tmp + fsync + rename. Guarantees that
    even if the process is killed mid-write, the file on disk is either the
    previous version or the new version — never a half-written one. Critical
    for the mid-trial resume guarantee."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _atomic_write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _save_trial_trajectory(result: dict, trial: int, log_dir: str, model: str) -> str:
    path = os.path.join(
        log_dir,
        f"trajectory_seed{result['seed']}_trial{trial}_raw{model}.json",
    )
    data = {
        "seed": result["seed"],
        "trial": trial,
        "success": result["success"],
        "action": result["action"],
        "iterations": result["iterations"],
        "elapsed_time": result.get("elapsed_time"),
        "trajectory": [
            {"thought": t, "action": a, "observation": o}
            for t, a, o in result.get("trajectory", [])
        ],
    }
    _atomic_write_json(path, data)
    print(f"[Trial {trial}] Trajectory saved → {os.path.abspath(path)}")
    return path


def _trajectory_path(seed: int, trial: int, log_dir: str, model: str) -> str:
    return os.path.join(
        log_dir, f"trajectory_seed{seed}_trial{trial}_raw{model}.json"
    )


def _reflection_path(seed: int, trial: int, log_dir: str, model: str) -> str:
    return os.path.join(
        log_dir, f"seed{seed}_trial{trial}_raw{model}_reflection.txt"
    )


def _save_trial_reflection(reflection: str, seed: int, trial: int, log_dir: str, model: str) -> None:
    if not reflection:
        return
    path = _reflection_path(seed, trial, log_dir, model)
    _atomic_write_text(path, reflection)
    print(f"[Trial {trial}] Reflection saved → {os.path.abspath(path)}")


def _load_existing_trial(seed: int, trial: int, log_dir: str, model: str) -> dict | None:
    """If trajectory_seed{S}_trial{T}_raw{model}.json exists AND is valid JSON,
    load it into the same flat result-dict shape produced by ReactAgent.solve().
    Returns None for missing files OR corrupt files (graceful — the caller then
    re-runs the trial fresh)."""
    path = _trajectory_path(seed, trial, log_dir, model)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(
            f"[Trial {trial}] WARNING: existing trajectory at {path} is unreadable "
            f"({type(e).__name__}: {e}). Re-running this trial fresh."
        )
        return None
    trajectory_tuples = [
        (t["thought"], t["action"], t["observation"])
        for t in data.get("trajectory", [])
    ]
    return {
        "seed": data.get("seed", seed),
        "trial": trial,
        "success": data.get("success", False),
        "action": data.get("action"),
        "iterations": data.get("iterations", 0),
        "elapsed_time": data.get("elapsed_time", 0.0),
        "trajectory": trajectory_tuples,
        "final_observation": "",
    }


def _load_existing_reflection(seed: int, trial: int, log_dir: str, model: str) -> str:
    path = _reflection_path(seed, trial, log_dir, model)
    if not os.path.exists(path):
        return ""
    with open(path) as f:
        return f.read().strip()


def run_seed_with_reflexion(
    seed: int,
    model_fn,
    args,
    k_trials: int = 2,
    reflection_model: str = "claude-sonnet-4-6",
) -> dict:
    """Run K trials on a single seed with Reflexion memory between trials.

    Stops early on the first successful trial. Saves one trajectory file per
    trial plus a per-seed reflexion summary JSON.

    Returns a flat result dict (compatible with `run_react_claude.save_results`)
    augmented with a `reflexion_summary` field.
    """
    print(f"\n{'#'*60}")
    print(f"  Reflexion Seed: {seed}  (K={k_trials} trials)")
    print(f"{'#'*60}")

    memory = ReflexionMemory(max_size=3)
    trial_records = []
    final_success = False
    trial_of_success = None
    cumulative_iters = 0
    total_elapsed = 0.0
    last_result = None
    successful_action = None

    for trial in range(1, k_trials + 1):
        print(
            f"\n=== Trial {trial}/{k_trials} for seed {seed} "
            f"(memory size: {len(memory.reflections)}) ==="
        )

        # Resume: if this trial's trajectory file already exists on disk, skip
        # the actor call and reload the result. Re-attach any prior reflection
        # so memory state matches what the next trial would have seen.
        existing = _load_existing_trial(seed, trial, args.eval_dir, args.model) if args.eval_dir else None
        if existing is None and args.eval_dir:
            expected = _trajectory_path(seed, trial, args.eval_dir, args.model)
            print(f"[Trial {trial}] No existing trajectory at {expected} — running fresh.")
        if existing is not None:
            print(
                f"[Trial {trial}] Resumed from {_trajectory_path(seed, trial, args.eval_dir, args.model)} "
                f"(iterations={existing.get('iterations')}, success={existing.get('success')}). "
                f"Skipping actor call."
            )
            result = existing
            elapsed = result.get("elapsed_time") or 0.0
            cumulative_iters += result.get("iterations", 0) or 0
            total_elapsed += elapsed
            last_result = result

            reflection_added = None
            if result.get("success"):
                final_success = True
                trial_of_success = trial
                successful_action = result.get("action")
                trial_records.append({
                    "trial": trial,
                    "success": True,
                    "iterations": result.get("iterations"),
                    "elapsed_time": elapsed,
                    "reflection_added": None,
                })
                print(f"[Trial {trial}] SUCCESS (resumed) — stopping early.")
                break

            if trial < k_trials:
                cached_reflection = _load_existing_reflection(seed, trial, args.eval_dir, args.model)
                if cached_reflection:
                    memory.add(cached_reflection)
                    reflection_added = cached_reflection
                    print(f"[Trial {trial}] Reflection reloaded ({len(cached_reflection)} chars).")
                else:
                    reflection = reflect_on_trajectory(
                        trajectory=result.get("trajectory", []),
                        final_observation=result.get("final_observation", ""),
                        success=False,
                        past_reflections=list(memory.reflections),
                        model=reflection_model,
                    )
                    if reflection:
                        memory.add(reflection)
                        reflection_added = reflection
                        _save_trial_reflection(reflection, seed, trial, args.eval_dir, args.model)
                        print(f"[Trial {trial}] Reflection regenerated ({len(reflection)} chars).")
                    else:
                        print(f"[Trial {trial}] Reflection empty — proceeding without new lesson.")

            trial_records.append({
                "trial": trial,
                "success": False,
                "iterations": result.get("iterations"),
                "elapsed_time": elapsed,
                "reflection_added": reflection_added,
            })
            continue

        # Fresh env + agent each trial — environment state must NOT carry over.
        toolkit = InterphyreToolkit(level_name=args.level, seed=seed, is_oss=False)

        agent = ReactAgent(
            model_fn=model_fn,
            toolkit=toolkit,
            level_name=args.level,
            max_iterations=args.max_iterations,
            verbose=args.verbose,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
        )

        if args.eval_dir:
            os.makedirs(args.eval_dir, exist_ok=True)
            img_initial = toolkit.get_image()
            cv2.imwrite(
                os.path.join(
                    args.eval_dir,
                    f"seed{seed}_trial{trial}_raw{args.model}_initial.png",
                ),
                img_initial,
            )

        start_time = time.time()
        with _inject_reflections_into_system_prompt(memory):
            result = agent.solve()
        elapsed = time.time() - start_time

        if args.eval_dir:
            img_final = toolkit.get_image()
            cv2.imwrite(
                os.path.join(
                    args.eval_dir,
                    f"seed{seed}_trial{trial}_raw{args.model}_final.png",
                ),
                img_final,
            )

        result["seed"] = seed
        result["elapsed_time"] = elapsed
        result["trial"] = trial

        toolkit.close()

        cumulative_iters += result.get("iterations", 0) or 0
        total_elapsed += elapsed
        last_result = result

        if args.eval_dir:
            _save_trial_trajectory(result, trial, args.eval_dir, args.model)

        reflection_added = None

        if result.get("success"):
            final_success = True
            trial_of_success = trial
            successful_action = result.get("action")
            trial_records.append({
                "trial": trial,
                "success": True,
                "iterations": result.get("iterations"),
                "elapsed_time": elapsed,
                "reflection_added": None,
            })
            print(f"[Trial {trial}] SUCCESS — stopping early.")
            break

        # Failed trial. Reflect only if we have a next trial to use the lesson.
        if trial < k_trials:
            reflection = reflect_on_trajectory(
                trajectory=result.get("trajectory", []),
                final_observation=result.get("final_observation", ""),
                success=False,
                past_reflections=list(memory.reflections),
                model=reflection_model,
            )
            if reflection:
                memory.add(reflection)
                reflection_added = reflection
                if args.eval_dir:
                    _save_trial_reflection(reflection, seed, trial, args.eval_dir, args.model)
                print(f"[Trial {trial}] Reflection added ({len(reflection)} chars).")
            else:
                print(f"[Trial {trial}] Reflection empty — proceeding without new lesson.")

        trial_records.append({
            "trial": trial,
            "success": False,
            "iterations": result.get("iterations"),
            "elapsed_time": elapsed,
            "reflection_added": reflection_added,
        })

    # Per-seed summary
    summary = {
        "seed": seed,
        "k_trials": k_trials,
        "trials": trial_records,
        "final_success": final_success,
        "trial_of_success": trial_of_success,
        "successful_action": successful_action,
        "total_iterations": cumulative_iters,
        "total_elapsed_time": total_elapsed,
        "reflections": list(memory.reflections),
    }

    if args.eval_dir:
        path = os.path.join(args.eval_dir, f"seed{seed}_reflexion_summary.json")
        _atomic_write_json(path, summary)
        print(f"[Seed {seed}] Reflexion summary saved → {os.path.abspath(path)}")

    # Flat result dict (same shape as run_single_seed's return) for the outer
    # run loop. Use the successful trial's action where applicable.
    flat = dict(last_result) if last_result else {"trajectory": []}
    flat["seed"] = seed
    flat["success"] = final_success
    flat["action"] = successful_action if final_success else (last_result or {}).get("action")
    flat["iterations"] = cumulative_iters
    flat["elapsed_time"] = total_elapsed
    flat["reflexion_summary"] = summary
    return flat
