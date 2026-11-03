"""Pigey trial.json records as HeXA trajectory dicts.

HeXA's distillation code consumes::

    {"seed": ..., "success": bool, "iterations": int,
     "trajectory": [{"thought": ..., "action": ..., "observation": ...}, ...]}

A Pigey trial is one LLM tool loop, so each step is the orchestrator's reasoning,
the tool it selected with its input, and the harness result text. The seed is the
string ``t<task>_e<episode>`` (task id and LIBERO init state), the stable identity
of the problem instance; string seeds round-trip through ``Skill.source_seeds``
unharmed, as hexa_core notes. Trials that ended in a backend error are skipped:
they say nothing about the method.
"""
from __future__ import annotations

import json
from pathlib import Path

OBSERVATION_MAX_CHARS = 400
THOUGHT_MAX_CHARS = 500
# Pigey's --max-llm-steps in run_trial.sbatch; trial.json does not record it.
DEFAULT_MAX_LLM_STEPS = 35


def seed_of(task_id: int, episode: int) -> str:
    return f"t{int(task_id):02d}_e{int(episode):02d}"


def _clip(text, limit: int) -> str:
    text = str(text or "")
    return text if len(text) <= limit else text[:limit] + "... [truncated]"


def strip_detections(result: str) -> str:
    """Drop the per-frame ``objects: #0 world=... bbox=...`` list from a tool result.

    The list is most of every result's text and is specific to one frame; the
    teacher needs the outcome line (HELD, RELEASED, TASK SUCCESS, errors), not the
    coordinates.
    """
    head, sep, _ = str(result or "").partition("\nobjects:")
    return head + (" [detection list omitted]" if sep else "")


def transcript_steps(transcript: list[dict]) -> list[dict]:
    """Group transcript entries into (thought, action, observation) steps.

    Every ``assistant`` entry starts a step; the ``tool_call``/``tool_result``
    pairs that follow belong to it. An assistant turn with no tool call is kept,
    with the harness's "Please use a tool to proceed" as its observation, because
    a wasted step is evidence too.
    """
    steps = []
    current = None
    for entry in transcript:
        role = entry.get("role")
        if role == "assistant":
            current = {"thought": entry.get("content", ""), "calls": []}
            steps.append(current)
        elif role == "tool_call" and current is not None:
            current["calls"].append({"tool": entry.get("tool", "?"), "input": entry.get("content", ""), "result": ""})
        elif role == "tool_result" and current is not None and current["calls"]:
            current["calls"][-1]["result"] = strip_detections(entry.get("content", ""))
    out = []
    for step in steps:
        if not step["calls"]:
            out.append({"thought": step["thought"], "action": "(no tool call)",
                        "observation": "Please use a tool to proceed."})
            continue
        action = "; ".join(f"{c['tool']}({c['input']})" for c in step["calls"])
        observation = " | ".join(f"{c['tool']}: {c['result']}" for c in step["calls"])
        out.append({"thought": step["thought"], "action": action, "observation": observation})
    return out


def end_reason(trial: dict, max_llm_steps: int = DEFAULT_MAX_LLM_STEPS) -> str:
    if trial.get("success"):
        return "success"
    roles = {entry.get("role") for entry in trial.get("transcript", [])}
    if "error" in roles:
        return "error"
    if "episode_terminated" in roles:
        return "horizon"  # the simulator ended the episode at LIBERO_HORIZON env steps
    if (trial.get("llm_steps") or 0) >= max_llm_steps:
        return "llm_budget"
    return "other"


def trial_to_trajectory(trial: dict, path: str | None = None) -> dict:
    steps = transcript_steps(trial.get("transcript", []))
    return {
        "seed": seed_of(trial["task_id"], trial["episode"]),
        "task_id": trial["task_id"],
        "episode": trial["episode"],
        "task_name": trial.get("suite", ""),
        "task_description": trial.get("task_description", ""),
        "success": bool(trial.get("success")),
        "iterations": int(trial.get("llm_steps") or len(steps)),
        "env_steps": trial.get("env_steps"),
        "end_reason": end_reason(trial),
        "trajectory": steps,
        "model": trial.get("model"),
        "path": path,
    }


def trajectory_reward(trajectory: dict) -> float:
    """HeXA's shaped reward, with Pigey's LLM steps as the iteration count.

    Successes: 1.0 (<=3 steps), 0.75 (<=8), 0.5 (<=15), else 0.25. Failures:
    -0.5 when the episode spent its budget (the env horizon or the LLM step cap),
    -0.75 when it ended early (an LLM error, or the loop exited without using
    either budget). The [-1, 1] range is what ``_compute_initial_confidence`` maps
    onto confidence.
    """
    if trajectory.get("success"):
        steps = trajectory.get("iterations", 0)
        if steps <= 3:
            return 1.0
        if steps <= 8:
            return 0.75
        if steps <= 15:
            return 0.5
        return 0.25
    return -0.5 if trajectory.get("end_reason") in ("horizon", "llm_budget") else -0.75


def format_trajectory_for_teacher(trajectory: dict, max_steps: int = 12) -> str:
    """One trial for the teacher: header, then compact reasoning/tool/result lines."""
    import skillrl.distillation.teacher_prompts as _tp

    reward = _tp.compute_trajectory_reward(trajectory)
    verdict = "SUCCESS" if trajectory.get("success") else f"FAILURE ({trajectory.get('end_reason', '?')})"
    lines = [f"--- Trial {trajectory.get('seed', '?')} | Task {trajectory.get('task_id', '?')}: "
             f"\"{trajectory.get('task_description', '')}\" | {verdict} | "
             f"LLM steps: {trajectory.get('iterations', '?')} | env steps: {trajectory.get('env_steps', '?')} | "
             f"Reward: {reward:+.2f} ---"]
    steps = trajectory.get("trajectory", [])
    if len(steps) > max_steps:
        lines.append(f"  (showing last {max_steps} of {len(steps)} steps)")
        steps = steps[-max_steps:]
    for i, step in enumerate(steps, 1):
        lines.append(f"  Step {i}:")
        lines.append(f"    Reasoning: {_clip(step.get('thought'), THOUGHT_MAX_CHARS)}")
        lines.append(f"    Tool: {_clip(step.get('action'), OBSERVATION_MAX_CHARS)}")
        lines.append(f"    Result: {_clip(step.get('observation'), OBSERVATION_MAX_CHARS)}")
    return "\n".join(lines)


def iter_trials(traj_dir: str | Path):
    """Yield (path, trial dict) for every trial.json under ``traj_dir``.

    Only ``trial.json`` files are read, so a directory of symlinks staged by
    hexa_loop and a raw results tree both work.
    """
    for path in sorted(Path(traj_dir).rglob("trial.json")):
        try:
            yield path, json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[Warning] Skipping unreadable trial {path}: {exc}")


def load_trial_trajectories(traj_dir: str | Path, level_name: str | None = None,
                            *, suite: str | None = None) -> tuple[list[dict], list[dict]]:
    """``(successes, failures)`` from the trial.json files under ``traj_dir``.

    Signature-compatible with ``skillrl.distillation.distill.load_trajectories``,
    which ``hexa_bridge.patch(load_trajectories_fn=...)`` rebinds in both
    ``distill`` and ``evolving_distill``. ``level_name`` is HeXA's level (the suite
    here); trials from other suites and backend-error trials are skipped.
    """
    traj_dir = Path(traj_dir)
    successes: list[dict] = []
    failures: list[dict] = []
    if not traj_dir.exists():
        print(f"[Warning] Trajectory directory not found: {traj_dir}")
        return successes, failures
    wanted = suite or level_name
    skipped = 0
    for path, trial in iter_trials(traj_dir):
        if trial.get("status") == "backend_error" or trial.get("success") is None:
            skipped += 1
            continue
        if wanted and trial.get("suite") != wanted:
            continue
        trajectory = trial_to_trajectory(trial, str(path))
        (successes if trajectory["success"] else failures).append(trajectory)
    if skipped:
        print(f"  Skipped {skipped} backend-error trial(s) under {traj_dir}")
    return successes, failures
