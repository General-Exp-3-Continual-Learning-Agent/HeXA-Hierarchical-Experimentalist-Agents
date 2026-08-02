"""Trajectory → text, plus rollout accounting.

Episodic RAG stores rollouts *verbatim* (that is the point of the baseline), so
we truncate long observations head+tail rather than dropping steps — the same
compromise `react_agent/reflexion/reflect.py` makes.

`count_simulator_calls` is the rollout-budget accounting axis: an InterPhyre
rollout is a `simulate_action` / `simulate_partial` tool call, not a ReAct
iteration.  Every baseline run logs it so method comparisons can be made at
equal simulator budget as well as equal iteration cap.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

SIM_TOOLS = ("simulate_action", "simulate_partial")


def _step_fields(step: Any) -> tuple[str, str, str]:
    if isinstance(step, dict):
        return (
            str(step.get("thought") or ""),
            str(step.get("action") or ""),
            str(step.get("observation") or ""),
        )
    if isinstance(step, (list, tuple)):
        pad = list(step) + [""] * (3 - len(step))
        return str(pad[0] or ""), str(pad[1] or ""), str(pad[2] or "")
    return "", "", ""


def _truncate(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    head = limit // 2
    tail = limit - head
    return text[:head] + " ...[truncated]... " + text[-tail:]


def count_simulator_calls(result: dict) -> int:
    """Number of physics rollouts the agent spent in this episode."""
    n = 0
    for step in result.get("trajectory", []) or []:
        _, action, _ = _step_fields(step)
        if any(action.strip().lower().startswith(tool) for tool in SIM_TOOLS):
            n += 1
    return n


def trajectory_reward(result: dict) -> float:
    """HeXA's shaped trajectory reward in [-1, 1] (outcome + iteration cost)."""
    from skillrl.distillation.teacher_prompts import compute_trajectory_reward

    return float(compute_trajectory_reward(result))


def format_trajectory(
    result: dict,
    *,
    max_steps: Optional[int] = None,
    max_obs_chars: int = 600,
    include_thoughts: bool = True,
) -> str:
    """Render an episode as numbered Thought/Action/Observation steps."""
    steps: Sequence[Any] = result.get("trajectory", []) or []
    lines: list[str] = []
    if max_steps is not None and len(steps) > max_steps:
        lines.append(f"(showing last {max_steps} of {len(steps)} steps)")
        steps = steps[-max_steps:]

    for i, step in enumerate(steps, 1):
        thought, action, obs = _step_fields(step)
        lines.append(f"--- Step {i} ---")
        if thought and include_thoughts:
            lines.append(f"Thought: {thought.strip()}")
        if action:
            lines.append(f"Action: {action.strip()}")
        if obs:
            lines.append(f"Observation: {_truncate(obs.strip(), max_obs_chars)}")
    return "\n".join(lines)


def format_verifier_feedback(result: dict) -> str:
    """The environment's verdict on an episode, as shown alongside a rollout."""
    action = result.get("action")
    placement = (
        f"x={action[0]}, y={action[1]}, radius={action[2]}"
        if isinstance(action, (list, tuple)) and len(action) >= 3
        else "none"
    )
    final_obs = _truncate(str(result.get("final_observation") or "").strip(), 600)
    return "\n".join(
        [
            f"Verifier: {'SUCCESS' if result.get('success') else 'FAILURE'}",
            f"Iterations used: {result.get('iterations', '?')}",
            f"Simulator calls: {count_simulator_calls(result)}",
            f"Shaped reward: {trajectory_reward(result):+.2f}",
            f"Final placement: {placement}",
            f"Final observation: {final_obs}",
        ]
    )
