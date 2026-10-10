"""Offline checks of the ORBIT memory port: no server, no GPU, no embedder download.

Covers the four things the port had to get right and the two defects the first
end-to-end run actually exposed:

  * retrieval keys use only observable information, and say so loudly when
    that leaves them degenerate (Mastermind);
  * a multi-episode transcript renders in the order it happened -- the first
    version printed each turn's action against the observation it was
    *answering*, which reads as the wrong state transition for every turn;
  * nothing computed from the hidden map reaches the injected prompt -- the
    first version put `shortest_path_len` in the verdict, which is
    `audit.STRUCTURAL_REGULARITIES` #2 straight into the actor's context;
  * `rag.py` and `memrl.py` are unmodified, so the dict-first seams in
    `trajectory_fmt.py` have to carry the whole adaptation.

Needs a real ORBIT checkout (it constructs instances), but nothing else.

    python baselines/tests/test_orbit_memory.py
"""
import sys
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from hexa_core.llm.base import Generation
from hexa_orbit import oracle
from hexa_orbit.actor import ActorConfig

from baselines.memory.memrl import MemRL, MemRLConfig
from baselines.memory.rag import EpisodicRAG, RAGConfig
from baselines.memory.store import MemoryStore, cosine
from baselines.memory.summarize import mock_summarizer
from baselines.memory.trajectory_fmt import (
    count_simulator_calls,
    format_trajectory,
    format_verifier_feedback,
    instance_label,
    trajectory_reward,
)
from baselines.orbit import blocks, stream
from baselines.orbit.embedding import HashingEmbedder
from baselines.orbit.intent import (
    IntentCache,
    build_intent,
    seed_key,
    strip_boilerplate,
)
from baselines.orbit.rollout import play_and_record
from baselines.orbit.tasks import build_task

ok = 0
def check(name, cond, extra=""):
    global ok
    print(("  PASS  " if cond else "  FAIL  ") + name + ("  " + str(extra) if extra else ""))
    if cond: ok += 1
    else: check.failed = True
check.failed = False


def scripted_client(task_name):
    """Oracle on Maze, cycling legal guesses on Mastermind. Mirrors the runner's."""
    guesses = ["1 2 3", "4 5 6", "1 4 2", "3 6 5", "2 1 4", "5 3 6"]

    class Stub:
        def __init__(self, instance):
            self.instance, self.queue, self.i = instance, [], 0

        def generate_structured(self, *, messages, **_):
            if task_name == "mastermind":
                self.i += 1
                return Generation.from_parts(
                    "", oracle.boxed(guesses[self.i % len(guesses)]),
                    finish_reason="stop")
            if not self.queue:
                self.queue = oracle.instance_optimal_actions(self.instance)
            return Generation.from_parts(
                "", oracle.boxed(self.queue.pop(0)), finish_reason="stop")

    return Stub


# ── retrieval keys ────────────────────────────────────────────────────────
print("\n== intent ==")
maze = build_task("maze")
mm = build_task("mastermind")

maze_texts = [build_intent(maze, s).text for s in maze.eval_specs(5)]
check("maze intents differ across instances", len(set(maze_texts)) == 5,
      len(set(maze_texts)))
check("maze intent drops the episode header",
      not any(t.startswith("[Episode") for t in maze_texts))
check("maze salient keeps only the varying clause",
      build_intent(maze, maze.eval_specs(1)[0], "salient").text.startswith(
          "you are at position"))

mm_specs = mm.eval_specs(5)
mm_texts = [build_intent(mm, s).text for s in mm_specs]
# The whole reason Mastermind is reported as a degenerate case rather than
# silently run: if this ever becomes >1 the key has started keying on something,
# and the only per-instance thing in that prompt is the seed-derived example.
check("mastermind intents are all identical", len(set(mm_texts)) == 1,
      len(set(mm_texts)))
check("mastermind example guess is masked out",
      "\\boxed{...}" in mm_texts[0] and not any(
          str(d) in mm_texts[0].split("for example")[1][:20] for d in mm_specs[0].code))
check("mastermind salient collapses to empty",
      build_intent(mm, mm_specs[0], "salient").text == "")

check("seed_key is stable and int",
      seed_key("abcdef1234") == seed_key("abcdef1234") == int("abcdef12", 16))
check("strip_boilerplate is idempotent",
      strip_boilerplate(strip_boilerplate(maze_texts[0], "maze", "observation"),
                        "maze", "observation") == maze_texts[0])

print("\n== hashing embedder ==")
he = HashingEmbedder()
a, b = he.embed_one("position (4, 3) up wall"), he.embed_one("position (5, 2) up path")
check("identical text -> cosine 1", abs(cosine(a, he.embed_one("position (4, 3) up wall")) - 1) < 1e-9)
check("different text -> cosine < 1", cosine(a, b) < 0.999, round(cosine(a, b), 4))
check("empty text -> zero vector scores 0", cosine(he.embed_one(""), a) == 0.0)


# ── dict-first seams in trajectory_fmt ────────────────────────────────────
print("\n== trajectory_fmt dict-first seams ==")
orbit_like = {
    "success": True, "iterations": 4, "reward": 0.6, "simulator_calls": 4,
    "verifier_feedback": "Verifier: solved first at episode 2",
    "instance_id": "deadbeefcafe0001",
    "episode_successes": [False, True],
    "trajectory": [
        {"episode": 1, "turn": 1, "action": "down", "observation": "at (1,1)",
         "result": "moved", "result_position": (2, 1), "thought": ""},
        {"episode": 1, "turn": 2, "action": "up", "observation": "at (2,1)",
         "result": "wall", "thought": ""},
        {"episode": 2, "turn": 1, "action": "right", "observation": "at (1,1)",
         "result": "GOAL", "thought": ""},
    ],
}
check("stated reward wins over skillrl binning",
      trajectory_reward(orbit_like) == 0.6)
check("stated simulator_calls wins", count_simulator_calls(orbit_like) == 4)
check("stated verifier feedback wins",
      format_verifier_feedback(orbit_like).startswith("Verifier: solved first"))
check("instance_id beats the hashed seed in the header",
      instance_label(orbit_like, "maze", 12345)
      == "Task: maze  |  Instance: deadbeefcafe0001")
check("an InterPhyre dict still routes to skillrl-style fields",
      instance_label({}, "catapult", 7) == "Level: catapult  |  Seed: 7")

print("\n== multi-episode rendering ==")
rendered = format_trajectory(orbit_like)
check("groups by episode", rendered.count("== Episode") == 2, rendered.count("== Episode"))
check("labels each episode's outcome",
      "Episode 1 — FAILED" in rendered and "Episode 2 — SOLVED" in rendered)
# The defect the first end-to-end run exposed: an ORBIT step's `observation` is
# the prompt it answered, so Action-then-Observation reads as the wrong
# transition. Saw must precede Chose, and the outcome must come after.
first = rendered.split("--- Turn 1 ---")[1]
check("observation precedes action within a turn",
      first.index("Saw:") < first.index("Chose:"))
check("the action's own outcome is rendered",
      "Result: moved -> (2, 1)" in rendered)
check("per-episode truncation keeps the opening, not the tail",
      "Turn 1" in format_trajectory(orbit_like, max_steps=1)
      and "Turn 2" not in format_trajectory(orbit_like, max_steps=1))
flat = {"success": True, "iterations": 2, "trajectory": [
    {"action": "simulate_action({})", "observation": "o", "thought": "t"}]}
check("a dict with no episode field still renders flat",
      "--- Step 1 ---" in format_trajectory(flat) and "== Episode" not in format_trajectory(flat))


# ── the leak guard ────────────────────────────────────────────────────────
print("\n== no privileged state reaches the prompt ==")
spec = maze.eval_specs(1)[0]
cfg = ActorConfig(model="stub", reasoning_effort=None)
run, result = play_and_record(
    maze, spec, "", cfg, client_factory=scripted_client("maze")
)
check("a maze instance plays end to end", result is not None and run.error is None,
      run.error)
store = MemoryStore(None)
rag = EpisodicRAG(store, RAGConfig())
record = rag.add_episode(result=result, level="maze", seed=seed_key(spec.instance_id),
                         intent="k", embedding=[1.0], step_index=0)
record.meta["instance_id"] = spec.instance_id
injected = blocks.format_rag_block([record])
# `shortest_path_len` comes from the config's path-length filter, `best_progress`
# from `oracle.optimal_steps` on the known map, and the maze reward penalises
# excess over that optimum. None may appear in text the actor reads.
check("shortest path length is not in the injected block",
      "Shortest" not in injected and str(spec.shortest_path_len) + " moves" not in injected)
check("known-map progress is not in the injected block",
      "route closed" not in injected and "best_progress" not in injected)
check("the reward is not in the injected block", "reward" not in injected.lower())
check("the real instance id is what the block names",
      spec.instance_id in injected and str(seed_key(spec.instance_id)) not in injected)
check("the verdict still states when it was solved",
      "solved first at episode" in injected or "never solved" in injected)
check("mastermind keeps its OBSERVABLE progress",
      "placed" in __import__("baselines.orbit.rollout", fromlist=["x"])
      .format_verifier_feedback({"episode_successes": [False],
                                 "best_progress": 0.66}, "mastermind"))


# ── frozen-memory fingerprint ─────────────────────────────────────────────
print("\n== frozen guard ==")
fp_before = stream._memory_fingerprint(store)
check("fingerprint is stable when nothing changes",
      fp_before == stream._memory_fingerprint(store))
store.records[0].q = 0.5
check("fingerprint catches a Q write",
      fp_before != stream._memory_fingerprint(store))


# ── MemRL on a degenerate key ─────────────────────────────────────────────
print("\n== MemRL under a degenerate key ==")
# Every intent embeds identically, which is Mastermind. Phase A tests `sim >
# delta` strictly, so the paper's top-20% quantile (1.0) would admit nothing and
# MemRL would silently become the no-bank arm; the calibrator lowers delta below
# the floor instead.
dstore = MemoryStore(None)
memrl = MemRL(dstore, MemRLConfig(delta=0.99, sim_norm_mean=1.0, sim_norm_std=1.0,
                                  add_similarity_threshold=1.01, novelty_threshold=1.01),
              summarizer=mock_summarizer)
key = [1.0, 0.0]
for i in range(4):
    r = dict(orbit_like, instance_id=f"inst{i}", success=(i % 2 == 0), reward=0.4 * i)
    rec = memrl.add_episode(result=r, level="mastermind", seed=i, intent="same",
                            embedding=key, step_index=i)
    rec.q = 0.1 * i
check("identical keys do not collapse the bank", len(dstore.records) == 4,
      len(dstore.records))
retrieved = memrl.retrieve(key, level="mastermind")
check("Phase A admits under the lowered delta",
      retrieved.diagnostics["phase_a"] == "ok", retrieved.diagnostics["phase_a"])
zs = {c["similarity_z"] for c in retrieved.diagnostics["selected"]}
check("similarity z-score is constant, so selection is Q-only", zs == {0.0}, zs)
check("selection is ordered by Q",
      [c["q"] for c in retrieved.diagnostics["selected"]]
      == sorted((c["q"] for c in retrieved.diagnostics["selected"]), reverse=True))
# The failure mode this guards: at MemRL's configured 0.90-0.99 every write
# folds into one record per outcome class, and the "bank" never grows.
collapsing = MemRL(MemoryStore(None), MemRLConfig(add_similarity_threshold=0.90),
                   summarizer=mock_summarizer)
for i in range(4):
    collapsing.add_episode(result=dict(orbit_like, instance_id=f"c{i}"),
                           level="mastermind", seed=i, intent="same",
                           embedding=key, step_index=i)
check("the collapse this defends against is real",
      len(collapsing.store.records) == 1, len(collapsing.store.records))


# ── both tasks produce dicts the methods accept ───────────────────────────
print("\n== both tasks end to end ==")
for task in (maze, mm):
    spec = task.eval_specs(1)[0]
    run, result = play_and_record(
        task, spec, "", cfg, client_factory=scripted_client(task.name)
    )
    s = MemoryStore(None)
    m = MemRL(s, MemRLConfig(delta=-1.0), summarizer=mock_summarizer)
    rec = m.add_episode(result=result, level=task.name, seed=seed_key(spec.instance_id),
                        intent="k", embedding=[1.0], step_index=0)
    check(f"{task.name}: memrl stores a record", rec is not None)
    check(f"{task.name}: reward is the graded instance reward in [-1,1]",
          -1.0 <= rec.reward <= 1.0 and rec.reward == result["reward"], rec.reward)
    check(f"{task.name}: simulator_calls counts env steps",
          rec.simulator_calls == len(run.steps), rec.simulator_calls)
    check(f"{task.name}: the block is non-empty and token-counted",
          blocks.count_tokens(blocks.format_memrl_block([rec])) > 0)

# ── the summarizer defect the first real run shipped ──────────────────────
print("\n== summarizer must never store deliberation ==")
from baselines.orbit import summarize as _sum
from hexa_core.llm.base import Generation


class _ThinkingClient:
    """A reasoning model that spent its whole budget in the analysis channel.

    This is what Qwen3-8B did on 44 of 80 Mastermind memories at a 1024-token
    cap: `final_text` empty, `reasoning_text` full of mid-sentence thinking.
    `raw_text` is their concatenation, so a fallback to it stores the
    scratchpad as the memory -- which is exactly what shipped.
    """

    def generate_structured(self, **_):
        return Generation.from_parts(
            "Okay, let's try to figure out the strategy the agent used here. So",
            "",
            finish_reason="length",
        )


class _Cfg:
    model, base_url, top_p = "stub", None, 0.95


fn = _sum.make_vllm_model_fn(_Cfg(), client=_ThinkingClient())
out = fn([{"role": "user", "content": "x"}], 0.0, 128)
check("an empty answer channel yields an empty summary, not the reasoning",
      out == "", repr(out[:60]))

q = _sum.SummaryQuality()
for _ in range(12):
    q.observe("Okay, let's try to figure out the strategy...")
check("quality guard flags a degraded summarizer", q.bad_fraction == 1.0)
check("quality guard counts deliberation separately from empty",
      q.to_dict()["deliberation"] == 12 and q.to_dict()["empty"] == 0)
q2 = _sum.SummaryQuality()
for _ in range(12):
    q2.observe("1. Open with 1 2 3.\n2. Use black pegs to fix positions.")
check("a real numbered script is not flagged", q2.bad_fraction == 0.0)
check("deliberation test does not fire on task text",
      not _sum.looks_like_deliberation("Guess only codes consistent with feedback."))
check("the summarizer wrapper exposes its quality counter",
      hasattr(_sum.make_summarizer(fn, "mastermind"), "quality"))

print(f"\n{ok} checks passed; failures: {check.failed}")
sys.exit(1 if check.failed else 0)
