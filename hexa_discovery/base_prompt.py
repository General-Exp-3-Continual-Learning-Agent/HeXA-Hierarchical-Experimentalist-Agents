"""Expose the agent's OWN system prompt to the teacher, so skills add to it.

The actor already receives ~9k characters of detailed methodology from the
benchmark itself -- how to run experiments, that `**params` is optional, that
"bounds are required and matter", that it should "hard-code anything you already
know", that positions carry noise so it should use "repeated measurements to
average over noise", and that it should "get the *functional form* right, not
the exact constants".

The teacher has never seen any of it. So it distils lessons the agent was
already given, and the injected block restates the task specification. Measured
on `fractional`: at least 3 of 10 synthesised skills reproduced base-prompt
guidance, and the agent's behaviour was identical to no-bank -- same number of
declared parameters, same experiment types. On `circle`, 4 of 5 "new" behaviours
appeared in 5/5 NO-BANK episodes too.

86% of the base prompt is shared across worlds, so one world's copy is a fair
representation of what every actor already knows.

This is the same class of fix as the judge critique and the refit penalty: the
teacher cannot reason about evidence it is never shown.
"""

from __future__ import annotations

from functools import lru_cache

#: Trimmed to keep the teacher prompt manageable; the methodology core is at the
#: front, the per-world instructions tail is the part that varies.
MAX_CHARS = 6000


@lru_cache(maxsize=None)
def base_prompt_for(world: str) -> str:
    """The system prompt the ACTOR receives for `world` (best effort, cached)."""
    try:
        from scienceagent.worlds import get_world
        import scienceagent.agent as _agent
        wd = get_world(world, engine="nbody", noise_std=0.075, noise_seed=0)
        return _agent._load_system_prompt(wd["system_prompt"], wd["instructions"])
    except Exception:
        return ""


def already_known_block(world: str, max_chars: int = MAX_CHARS) -> str:
    """Render the 'the agent already knows this' section for a teacher prompt."""
    base = base_prompt_for(world)
    if not base:
        return ""
    body = base[:max_chars]
    if len(base) > max_chars:
        body += "\n... [truncated]"
    return (
        "\n**WHAT THE AGENT IS ALREADY TOLD.** Every episode the agent receives "
        "the instructions below, verbatim, before it starts. A skill that "
        "restates any of this is worse than useless: it consumes the injected "
        "prompt and teaches nothing. Do NOT emit skills about running "
        "experiments in general, averaging repeated measurements, declaring "
        "fittable parameters, choosing bounds, or writing a clear explanation "
        "*unless* your episodes show a specific failure that these instructions "
        "did not prevent. Extract only what goes BEYOND the following:\n"
        "-------- BEGIN AGENT'S EXISTING INSTRUCTIONS --------\n"
        f"{body}\n"
        "-------- END AGENT'S EXISTING INSTRUCTIONS --------\n"
    )
