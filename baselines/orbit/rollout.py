"""Play one ORBIT instance with a retrieved-memory block injected.

The analogue of `baselines/runner/memory_runner.py`, and the same idea: only the
text of the injected block differs between methods, everything else -- the
system prompt, the observation stream, the turn cap, the actor model -- is the
no-bank baseline's.

    HeXA           the distilled skill bank
    Episodic RAG   the nearest past instance transcripts, verbatim
    MemRL          Q-ranked summarised experiences
    no bank        nothing

The injection point is `MazeActor(guidelines=...)`, which appends to the system
prompt and nowhere else.  That is the seam Design B already uses, so a memory
block costs tokens in exactly the place a skill bank does and the two are
comparable at equal added context.

`InstanceRun` is converted to the trajectory dict `memory/rag.py` and
`memory/memrl.py` consume by `hexa_orbit.trajectory.instance_trajectory` (or its
Mastermind twin), which already emits the right shape and already stamps the
graded instance reward.  What this module adds is the three fields those methods
would otherwise try to compute the InterPhyre way -- see the dict-first
functions in `baselines/memory/trajectory_fmt.py`.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Optional

from hexa_orbit.actor import ContextExhausted, GenerationFailed, InstanceRun, MazeActor

from baselines.memory.blocks import count_tokens


def errored_run(spec, message: str, *, leaked: Optional[bool] = None) -> InstanceRun:
    """An instance that never produced a transcript.

    Recorded with `error` set and excluded from every metric downstream, which
    is `play_batch`'s convention: a transport failure is evidence of nothing,
    and folding it in as a failure would depress the arm for a reason unrelated
    to the agent.
    """
    return InstanceRun(
        instance_id=spec.instance_id,
        dataset_seed=spec.dataset_seed,
        resolved_seed=spec.resolved_seed,
        shortest_path_len=spec.shortest_path_len,
        leaked=leaked,
        steps=[], messages=[], metrics={},
        episode_successes=[], episode_lengths=[],
        success_count=0, added_tokens=0, token_counter="",
        error=message,
    )


def play_instance(
    orbit_task,
    spec,
    memory_block: Optional[str],
    config,
    *,
    leaked: Optional[bool] = None,
    client_factory: Optional[Callable[[Any], Any]] = None,
) -> InstanceRun:
    """Play one instance to completion. Never raises for a model-side failure.

    `ContextExhausted` is already handled inside `MazeActor.run_instance`, which
    stops the instance and lets the episodes played so far stand.  It is caught
    here too because it can also surface from the very first call, before any
    episode exists -- with a large enough retrieved block that is a real
    possibility, and it is a property of the injected context rather than of the
    agent, so it must be recorded as an error rather than as three failures.
    """
    instance = orbit_task.instance(spec)
    # A factory rather than a shared client: an oracle or scripted stub reads the
    # live instance, and `MazeActor` takes its client at construction. Returning
    # `None` (the default, and what a real run does) leaves `MazeActor` to build
    # its own `VLLMClient` lazily.
    actor = MazeActor(
        config,
        guidelines=memory_block or None,
        client=client_factory(instance) if client_factory else None,
    )
    try:
        return actor.run_instance(instance, leaked=leaked)
    except (GenerationFailed, ContextExhausted) as exc:
        return errored_run(spec, f"{type(exc).__name__}: {exc}", leaked=leaked)


#: Fields that exist on a trajectory dict but must never reach the actor.
#:
#: This text goes into `MemoryRecord.content`, which is injected into the system
#: prompt, so anything here is something the agent gets to read.  Three of the
#: trajectory's fields are computed from state the agent cannot observe and are
#: therefore excluded:
#:
#: `shortest_path_len`
#:     Comes from the maze config's path-length filter, not from anything the
#:     agent saw.  It is `audit.STRUCTURAL_REGULARITIES` #2 -- "the route is
#:     always 7-8 moves, so re-plan past 8" -- and an arm that won because its
#:     memories quoted it would have won on a privileged channel.
#:     `teacher.render_episodes` withholds it from the *evolver* for exactly
#:     this reason; withholding it from the actor matters more.
#: `best_progress` on Maze
#:     `adapter._episode_progress` computes it with `oracle.optimal_steps` on
#:     the known map.  Same leak, one step removed.
#: `reward` on Maze
#:     `reward.instance_reward` penalises steps in excess of the known-map
#:     optimum, so the number encodes that optimum.
#:
#: On Mastermind `best_progress` is black pegs read out of the feedback text the
#: agent itself saw (`mastermind.episode_progress`), so it is observable and is
#: shown.  The distinction is the point: what is excluded is privileged state,
#: not "anything that looks like a score".
_PRIVILEGED_ON_MAZE = ("shortest_path_len", "best_progress", "reward")


def format_verifier_feedback(result: dict, task: str) -> str:
    """The environment's verdict on a three-episode instance.

    Replaces the InterPhyre "final placement" verdict, which has no analogue
    here.  What a reader of a retrieved memory needs is *when* the instance was
    solved, not merely whether -- an instance solved at episode 3 after two
    failures is a different lesson from one solved immediately.

    Observable facts only; see :data:`_PRIVILEGED_ON_MAZE`.
    """
    successes = list(result.get("episode_successes") or [])
    first = result.get("first_success_episode")
    verdict = f"solved first at episode {first}" if first else "never solved"
    lines = [
        f"Verifier: {verdict} "
        f"(R={result.get('success_count', 0)}/{len(successes) or 3})",
        "Per-episode outcome: "
        + ", ".join(
            f"ep{i}={'SOLVED' if s else 'FAILED'}"
            for i, s in enumerate(successes, 1)
        ),
        f"Turns used per episode: {list(result.get('episode_lengths') or [])}",
    ]
    if task == "mastermind":
        lines.append(
            f"Best guess placed {float(result.get('best_progress', 0.0)):.0%} "
            f"of the digits correctly"
        )
    if not result.get("adapted_across_episodes", True):
        lines.append(
            "NOTE: every episode repeated the same action sequence -- the agent "
            "learned nothing from its own attempts."
        )
    return "\n".join(lines)


def to_result(
    orbit_task,
    run: InstanceRun,
    *,
    memory_block: str = "",
    elapsed: Optional[float] = None,
    retrieval_diagnostics: Optional[dict] = None,
) -> dict:
    """`InstanceRun` -> the dict `rag.py` / `memrl.py` store and score.

    Three fields exist so those two modules need no ORBIT branch:

    `simulator_calls`
        The rollout-budget axis.  On InterPhyre it counts `simulate_action` tool
        calls; here it is environment steps, which is what an ORBIT instance
        actually spends and what `total_step_cap` bounds.  Without it
        `count_simulator_calls` would scan for tool names and report 0.
    `reward`
        Already stamped by `instance_trajectory` from
        `hexa_orbit.reward.instance_reward`.  Named here only to say it is load
        -bearing: `trajectory_reward` returns it instead of importing
        `skillrl`'s InterPhyre-tuned binning, which puts every ORBIT instance in
        the same bucket.
    `verifier_feedback`
        The per-episode verdict above, in place of a final placement.
    """
    result = dict(orbit_task.to_trajectory(run))
    result["simulator_calls"] = len(run.steps)
    result["verifier_feedback"] = format_verifier_feedback(result, orbit_task.name)
    result["final_observation"] = (
        run.messages[-1]["content"] if run.messages else ""
    )
    result["task"] = orbit_task.name
    result["instance_id"] = run.instance_id
    result["dataset_seed"] = run.dataset_seed
    result["leaked"] = run.leaked
    result["error"] = run.error
    result["context_exhausted"] = run.context_exhausted
    result["added_context_tokens"] = count_tokens(memory_block)
    result["added_context_chars"] = len(memory_block)
    # What the actor itself measured the block to cost, from ORBIT's tokenizer
    # rather than tiktoken. Kept alongside rather than instead of: the former is
    # what `MAZE_RESULTS.md` reports, the latter is the CORE paper's axis.
    result["actor_added_tokens"] = run.added_tokens
    result["elapsed_time"] = elapsed
    result["retrieval_diagnostics"] = retrieval_diagnostics or {}
    return result


def play_and_record(
    orbit_task,
    spec,
    memory_block: Optional[str],
    config,
    *,
    leaked: Optional[bool] = None,
    client_factory: Optional[Callable[[Any], Any]] = None,
    retrieval_diagnostics: Optional[dict] = None,
) -> tuple[InstanceRun, Optional[dict]]:
    """Play one instance and convert it. `None` result means the run errored."""
    started = time.perf_counter()
    run = play_instance(
        orbit_task, spec, memory_block, config,
        leaked=leaked, client_factory=client_factory,
    )
    elapsed = time.perf_counter() - started
    if run.error is not None:
        return run, None
    result = to_result(
        orbit_task,
        run,
        memory_block=memory_block or "",
        elapsed=elapsed,
        retrieval_diagnostics=retrieval_diagnostics,
    )
    return run, result
