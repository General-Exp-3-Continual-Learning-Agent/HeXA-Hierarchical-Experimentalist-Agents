#!/usr/bin/env python
"""Episode-level Reflexion on ORBIT: the reset, the lessons, the ablation.

Proves the three things that distinguish `episode-reset` from the no-bank
baseline and from its own ablation, without a server:

1. episode 1 is byte-identical to no-bank (nothing injected, 0 added tokens);
2. with reset_context the chat really is wiped -- episode 2 opens on
   [system, observation] and contains no turn from episode 1;
3. without it the transcript survives, which is the `episode-keep` ablation.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hexa_core.llm.base import Generation  # noqa: E402
from hexa_orbit.actor import ActorConfig  # noqa: E402

from baselines.orbit.reflexion import EpisodeReflexionActor  # noqa: E402
from baselines.orbit.tasks import build_task  # noqa: E402


class AlwaysUp:
    """Answers 'up' forever, so every episode fails and every boundary reflects."""

    def generate_structured(self, **_):
        return Generation.from_parts("", r"\boxed{up}", finish_reason="stop")


class CountingReflector:
    def __init__(self):
        self.calls = 0
        self.prompts = []

    def generate_structured(self, *, messages=None, **_):
        self.calls += 1
        self.prompts.append(messages)
        return Generation.from_parts(
            "", f"Lesson {self.calls}: stop walking into the north wall.",
            finish_reason="stop")


def run(reset: bool):
    task = build_task("maze")
    spec = task.eval_specs(1)[0]
    reflector = CountingReflector()
    actor = EpisodeReflexionActor(
        ActorConfig(max_response_tokens=64),
        reflector=reflector, task_name="maze", reset_context=reset,
        client=AlwaysUp(),
    )
    run_ = actor.run_instance(task.instance(spec), leaked=False)
    return actor, reflector, run_


def main() -> int:
    base = build_task("maze")
    base_prompt = EpisodeReflexionActor(
        ActorConfig(max_response_tokens=64),
        reflector=CountingReflector(), task_name="maze", client=AlwaysUp(),
    ).base_system_prompt

    # ── episode-reset ────────────────────────────────────────────────────
    actor, reflector, run_ = run(reset=True)
    assert run_.success_count == 0, "the wall-walker should fail every episode"
    assert len(run_.episode_successes) == 3, run_.episode_successes

    # episode 1 must be untouched
    assert actor.system_prompt_for(0) == base_prompt, "episode 1 was not no-bank"
    assert 0 not in actor.guidelines, "a block leaked into episode 1"

    # a reflection per failed boundary (2 boundaries in a 3-episode instance)
    assert reflector.calls == 2, f"expected 2 reflections, got {reflector.calls}"
    assert len(actor.reflection_log) == 2
    assert all(not e["empty"] for e in actor.reflection_log)

    # lessons accumulate and reach later episodes
    assert "Lesson 1" in actor.system_prompt_for(1)
    assert "Lesson 1" in actor.system_prompt_for(2)
    assert "Lesson 2" in actor.system_prompt_for(2)
    assert "Lesson 2" not in actor.system_prompt_for(1), "lesson 2 leaked backwards"
    assert actor.added_tokens > 0
    print(f"episode-reset: 2 reflections, lessons accumulate, "
          f"added_tokens={actor.added_tokens}  OK")

    # the reset actually happened: the final transcript holds ONLY episode 3
    ep3_turns = sum(1 for s in run_.steps if s.episode_index == 2)
    user_turns = [m for m in run_.messages if m["role"] == "user"]
    assert len(user_turns) == ep3_turns + 1, (
        f"chat carries {len(user_turns)} observations but episode 3 had "
        f"{ep3_turns} turns -- the reset did not happen")
    assert run_.messages[0]["role"] == "system"
    assert "Lesson" in run_.messages[0]["content"]
    print(f"episode-reset: final chat holds only episode 3 "
          f"({len(user_turns)} observations)  OK")

    # every episode is still recorded even though the chat was wiped
    assert len({s.episode_index for s in run_.steps}) == 3, "steps lost an episode"
    print("episode-reset: all 3 episodes still in `steps`  OK")

    # ── episode-keep (the ablation) ──────────────────────────────────────
    actor_k, reflector_k, run_k = run(reset=False)
    assert reflector_k.calls == 2
    keep_user_turns = [m for m in run_k.messages if m["role"] == "user"]
    assert len(keep_user_turns) == len(run_k.steps) + 1, (
        "episode-keep should retain every observation")
    assert len(keep_user_turns) > ep3_turns + 1, (
        "episode-keep kept no more than episode-reset")
    assert "Lesson 1" in actor_k.system_prompt_for(1)
    print(f"episode-keep: transcript retained "
          f"({len(keep_user_turns)} observations vs {len(user_turns)})  OK")

    # ── an empty reflection must not inject an empty block ───────────────
    class DeadReflector:
        def generate_structured(self, **_):
            return None

    task = build_task("maze")
    spec = task.eval_specs(1)[0]
    actor_d = EpisodeReflexionActor(
        ActorConfig(max_response_tokens=64),
        reflector=DeadReflector(), task_name="maze", client=AlwaysUp(),
    )
    run_d = actor_d.run_instance(task.instance(spec), leaked=False)
    assert actor_d.guidelines == {}, "an empty reflection injected a block"
    assert actor_d.added_tokens == 0
    assert all(e["empty"] for e in actor_d.reflection_log)
    assert run_d.success_count == 0
    print("dead reflector: no block injected, degrades to no-bank  OK")

    print("\nall episode-level reflexion checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
