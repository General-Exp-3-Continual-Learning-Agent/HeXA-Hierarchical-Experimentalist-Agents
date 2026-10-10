#!/usr/bin/env python
"""Reflexion-on-ORBIT plumbing, without a server.

Covers the part the `--stub` oracle cannot: the oracle solves every maze, so a
smoke run never reaches the reflect-and-retry path at all. Here the actor
always walks into a wall, so every instance fails, every instance earns a
reflection, and trial 2 actually carries one.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hexa_core.llm.base import Generation  # noqa: E402
from hexa_orbit.actor import ActorConfig, MazeActor  # noqa: E402

from baselines.orbit import reflexion as rfx  # noqa: E402
from baselines.orbit.rollout import format_verifier_feedback  # noqa: E402
from baselines.orbit.tasks import build_task  # noqa: E402


class AlwaysUp:
    """Answers 'up' forever. Fails essentially every maze."""

    def generate_structured(self, **_):
        return Generation.from_parts("", r"\boxed{up}", finish_reason="stop")


class CannedReflector:
    def __init__(self):
        self.prompts = []

    def generate_structured(self, *, messages=None, **_):
        self.prompts.append(messages)
        return Generation.from_parts(
            "", "You only ever moved up and re-entered (1,1). Sweep right first.",
            finish_reason="stop")


def main() -> int:
    task = build_task("maze")
    specs = task.eval_specs(2)
    cfg = ActorConfig(max_response_tokens=64)

    # ── trial 1: nothing injected ─────────────────────────────────────────
    runs = []
    for spec in specs:
        actor = MazeActor(cfg, guidelines=None, client=AlwaysUp())
        runs.append(actor.run_instance(task.instance(spec), leaked=False))
    assert all(r.success_count == 0 for r in runs), "expected the wall-walker to fail"
    assert all(r.added_tokens == 0 for r in runs), "trial 1 must be byte-identical to no-bank"
    print(f"trial 1: {len(runs)} instances, all R=0/3, 0 added tokens  OK")

    # ── reflect ───────────────────────────────────────────────────────────
    reflector = CannedReflector()
    lessons = {}
    for run in runs:
        result = dict(task.to_trajectory(run))
        feedback = format_verifier_feedback(result, "maze")
        text = rfx.reflect(reflector, "maze", run, feedback, [])
        assert text, "reflection came back empty"
        lessons[run.instance_id] = [text]
    print(f"reflect: {len(lessons)} reflections  OK")

    # The reflector must never be handed privileged state.
    blob = "\n".join(m["content"] for msgs in reflector.prompts for m in msgs)
    assert "shortest_path" not in blob and "optimal" not in blob.lower(), \
        "privileged map knowledge reached the reflector"
    # `reward` on maze encodes the known-map optimum (rollout._PRIVILEGED_ON_MAZE),
    # so the trial rendering must carry actions and observations, not scores.
    assert "reward" not in blob.lower(), "a graded reward reached the reflector"
    print("privilege check: no map knowledge or reward in the reflection prompt  OK")

    # ── trial 2: the lesson is injected, and only there ───────────────────
    for spec in specs:
        block = rfx.reflexion_block(lessons[spec.instance_id])
        assert block and "Sweep right first" in block
        actor = MazeActor(cfg, guidelines=block, client=AlwaysUp())
        run = actor.run_instance(task.instance(spec), leaked=False)
        assert run.added_tokens > 0, "trial 2 injected nothing"
        assert block in actor.system_prompt_for(0), "block missing from the system prompt"
        assert block in actor.system_prompt_for(2), "block missing at episode 3"
    print("trial 2: lesson present in every episode's system prompt  OK")

    # ── an empty reflection degrades to no-bank, it does not crash ────────
    assert rfx.reflexion_block([]) is None
    assert rfx.reflexion_block(["", "  "]) is None

    class Dead:
        def generate_structured(self, **_):
            raise RuntimeError("server gone")

    assert rfx.reflect(Dead(), "maze", runs[0], "verdict", []) == ""

    class Nothing:
        def generate_structured(self, **_):
            return None

    assert rfx.reflect(Nothing(), "maze", runs[0], "verdict", []) == ""
    print("degradation: dead reflector -> empty -> no-bank condition  OK")

    print("\nall reflexion plumbing checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
