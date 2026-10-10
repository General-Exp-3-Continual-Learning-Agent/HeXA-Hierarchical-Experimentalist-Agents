"""Trajectory → text, plus rollout accounting.

Episodic RAG stores rollouts *verbatim* (that is the point of the baseline), so
we truncate long observations head+tail rather than dropping steps — the same
compromise `react_agent/reflexion/reflect.py` makes.

`count_simulator_calls` is the rollout-budget accounting axis: an InterPhyre
rollout is a `simulate_action` / `simulate_partial` tool call, not a ReAct
iteration.  Every baseline run logs it so method comparisons can be made at
equal simulator budget as well as equal iteration cap.

Three functions here are *dict-first*: if the episode dict already carries the
quantity (`simulator_calls`, `reward`, `verifier_feedback`), it is returned
as-is and nothing is recomputed.  That is what lets `baselines/orbit/` feed
ORBIT `InstanceRun`s through `rag.py` and `memrl.py` unmodified — an ORBIT
instance has no `simulate_action` calls to count, its reward is
`hexa_orbit.reward.instance_reward`, and its verdict is per-episode rather than
a single placement.  The InterPhyre path sets none of those keys and so is
byte-for-byte unaffected.
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


def instance_label(result: dict, level: str, seed: int) -> str:
    """How a stored memory names the episode it came from, in its own header.

    `MemoryRecord.seed` is an int, which is right for InterPhyre where a seed
    *is* an int.  An ORBIT instance is identified by a hex digest, and
    `baselines/orbit/intent.seed_key` truncates it to an int only to satisfy
    that field — printing that number in the injected block would put an
    identifier in the prompt that matches nothing in any run file, and would
    disagree with the real id the block header already shows.  So the dict's own
    `instance_id` wins where there is one.
    """
    stated = result.get("instance_id")
    if stated:
        return f"Task: {level}  |  Instance: {stated}"
    return f"Level: {level}  |  Seed: {seed}"


def count_simulator_calls(result: dict) -> int:
    """Number of physics rollouts the agent spent in this episode.

    A caller that already knows its own budget axis states it under
    `simulator_calls` and it is returned unchanged — ORBIT counts environment
    steps, which are not tool calls and would otherwise score zero here.
    """
    stated = result.get("simulator_calls")
    if isinstance(stated, int) and not isinstance(stated, bool):
        return stated

    n = 0
    for step in result.get("trajectory", []) or []:
        _, action, _ = _step_fields(step)
        if any(action.strip().lower().startswith(tool) for tool in SIM_TOOLS):
            n += 1
    return n


def trajectory_reward(result: dict) -> float:
    """HeXA's shaped trajectory reward in [-1, 1] (outcome + iteration cost).

    A `reward` already on the dict wins.  `hexa_orbit.trajectory` stamps one
    (`hexa_orbit.reward.instance_reward`), and recomputing would be wrong twice
    over: `compute_trajectory_reward` bins on InterPhyre's 25-iteration budget,
    against which every 3-27 step ORBIT instance lands in the same bucket, and
    importing it drags in `skillrl` for a run that needs none of it.
    """
    stated = result.get("reward")
    if isinstance(stated, (int, float)) and not isinstance(stated, bool):
        return float(stated)

    from skillrl.distillation.teacher_prompts import compute_trajectory_reward

    return float(compute_trajectory_reward(result))


def _render_steps(
    steps: Sequence[Any],
    *,
    max_steps: Optional[int],
    max_obs_chars: int,
    include_thoughts: bool,
    first_index: int = 1,
) -> list[str]:
    lines: list[str] = []
    if max_steps is not None and len(steps) > max_steps:
        lines.append(f"(showing last {max_steps} of {len(steps)} steps)")
        steps = steps[-max_steps:]

    for i, step in enumerate(steps, first_index):
        thought, action, obs = _step_fields(step)
        lines.append(f"--- Step {i} ---")
        if thought and include_thoughts:
            lines.append(f"Thought: {thought.strip()}")
        if action:
            lines.append(f"Action: {action.strip()}")
        if obs:
            lines.append(f"Observation: {_truncate(obs.strip(), max_obs_chars)}")
    return lines


def _render_turn(
    step: Any, *, max_obs_chars: int, include_thoughts: bool
) -> list[str]:
    """One multi-episode turn, in the order it actually happened.

    Observation **before** action, and the action's outcome after it.  The flat
    renderer prints Thought/Action/Observation because an InterPhyre step's
    `observation` is the result of its action.  A multi-episode step's is not:
    it is the prompt the agent was answering, and its outcome arrives in the
    *next* step's observation, which is why `step_entries` lifts it into a
    separate `result` field.  Printing them in the flat order reads as "the
    agent moved down and arrived where it already was", so a model reading a
    retrieved transcript would learn the wrong transition for every turn in it.
    """
    thought, action, observation = _step_fields(step)
    lines = [f"--- Turn {step.get('turn', '?')} ---"]
    if observation:
        lines.append(f"Saw: {_truncate(observation.strip(), max_obs_chars)}")
    if thought and include_thoughts:
        lines.append(f"Thought: {thought.strip()}")
    if action:
        lines.append(f"Chose: {action.strip()}")
    outcome = step.get("result")
    if outcome:
        position = step.get("result_position")
        lines.append(
            f"Result: {outcome}" + (f" -> {tuple(position)}" if position else "")
        )
    return lines


def _episode_of(step: Any) -> Optional[int]:
    """The 1-based episode a step belongs to, when the caller records one."""
    if not isinstance(step, dict):
        return None
    value = step.get("episode")
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None


def format_trajectory(
    result: dict,
    *,
    max_steps: Optional[int] = None,
    max_obs_chars: int = 600,
    include_thoughts: bool = True,
) -> str:
    """Render a rollout as numbered Thought/Action/Observation steps.

    When the steps carry an `episode` field the render is grouped under episode
    headings and `max_steps` applies **per episode** rather than to the rollout
    as a whole.  That is not a cosmetic choice.  An ORBIT instance is three
    episodes replayed on one hidden task, and the cross-episode contrast — what
    episode 1 discovered, what episode 2 did with it — is the entire content of
    the memory.  Truncating a 27-step instance to its last 12 steps would keep
    only episode 3 and silently discard exactly that.

    InterPhyre steps carry no `episode` field, so they render flat, as before.
    """
    steps: Sequence[Any] = result.get("trajectory", []) or []
    episodes = [_episode_of(s) for s in steps]
    if not steps or any(e is None for e in episodes):
        return "\n".join(
            _render_steps(
                steps,
                max_steps=max_steps,
                max_obs_chars=max_obs_chars,
                include_thoughts=include_thoughts,
            )
        )

    grouped: dict[int, list[Any]] = {}
    for episode, step in zip(episodes, steps):
        grouped.setdefault(int(episode), []).append(step)

    successes = list(result.get("episode_successes") or [])
    lines: list[str] = []
    for episode in sorted(grouped):
        turns = grouped[episode]
        verdict = ""
        if 0 < episode <= len(successes):
            verdict = " — SOLVED" if successes[episode - 1] else " — FAILED"
        lines.append(f"== Episode {episode}{verdict} ({len(turns)} turns) ==")
        shown = turns
        if max_steps is not None and len(turns) > max_steps:
            # First N, not last N. Truncating from the end would cut the opening
            # of an episode, and the opening moves are where the exploration
            # decision a reader is meant to learn from actually happens.
            shown = turns[:max_steps]
        for step in shown:
            lines.extend(
                _render_turn(
                    step,
                    max_obs_chars=max_obs_chars,
                    include_thoughts=include_thoughts,
                )
            )
        if len(turns) > len(shown):
            lines.append(f"... {len(turns) - len(shown)} further turns omitted")
    return "\n".join(lines)


def format_verifier_feedback(result: dict) -> str:
    """The environment's verdict on an episode, as shown alongside a rollout.

    A caller whose verdict has a different shape states it under
    `verifier_feedback` and it is returned verbatim — an ORBIT instance is
    graded per episode, and "Final placement: none" would be noise.
    """
    stated = result.get("verifier_feedback")
    if isinstance(stated, str) and stated.strip():
        return stated

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
