"""Offline checks of the memory logic: no InterPhyre, no LLM, no embeddings.

Covers retrieval for both baselines, MemRL's two-phase selection and
Monte-Carlo utility update, the novelty/merge path, reward variants, and block
formatting.  Runs in ~2s with no GPU, so use it instead of burning simulator
time to catch a logic bug.

    python baselines/tests/test_memory_logic.py
"""
import sys, types, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Stub skillrl.distillation.teacher_prompts so trajectory_fmt's reward import
# does not pull in the whole distillation stack.
mod = types.ModuleType("skillrl.distillation.teacher_prompts")
def compute_trajectory_reward(t):
    if t.get("success"):
        i = t.get("iterations", 25)
        return 1.0 if i <= 3 else 0.75 if i <= 7 else 0.5 if i <= 15 else 0.25
    return -0.75 if t.get("iterations", 25) < 10 else -0.5
mod.compute_trajectory_reward = compute_trajectory_reward
import skillrl, skillrl.distillation
sys.modules["skillrl.distillation.teacher_prompts"] = mod

from baselines.memory.store import MemoryStore, cosine, SUCCESS, FAILURE
from baselines.memory.rag import EpisodicRAG, RAGConfig
from baselines.memory.memrl import MemRL, MemRLConfig
from baselines.memory.blocks import format_rag_block, format_memrl_block, count_tokens
from baselines.memory.trajectory_fmt import count_simulator_calls, format_trajectory
from baselines.memory.summarize import mock_summarizer
from baselines.calibrate import quantile
from baselines.memory.features import (
    FeatureScaler, NumericEmbedder, parse_features, rank_agreement, spearman,
    top_k_overlap,
)

ok = 0
def check(name, cond, extra=""):
    global ok
    print(("  PASS  " if cond else "  FAIL  ") + name + ("  " + str(extra) if extra else ""))
    if cond: ok += 1
    else: check.failed = True
check.failed = False

def episode(seed, success, iters, sims=2):
    traj = [{"thought": "t", "action": "get_level_state({})", "observation": "obs " * 50}]
    traj += [{"thought": "t", "action": "simulate_action({'x':1})", "observation": "FAILURE " * 40}] * sims
    return {"seed": seed, "success": success, "iterations": iters, "trajectory": traj,
            "action": [1.0, 2.0, 0.5], "final_observation": "done"}

print("\n== trajectory_fmt ==")
e = episode(1, True, 5, sims=3)
check("counts simulator calls", count_simulator_calls(e) == 3, count_simulator_calls(e))
check("truncates observations", "[truncated]" in format_trajectory(e, max_obs_chars=50))
check("max_steps keeps the tail", format_trajectory(e, max_steps=2).count("--- Step") == 2)

print("\n== store ==")
check("cosine identity", abs(cosine([1, 0], [1, 0]) - 1.0) < 1e-9)
check("cosine orthogonal", abs(cosine([1, 0], [0, 1])) < 1e-9)
check("cosine empty", cosine([], [1]) == 0.0)

if True:
    # A real directory, not tempfile: cleanup of temp dirs on this cluster's
    # scratch NFS trips over lingering .nfs* handles.
    td = Path(__file__).resolve().parent / "_teststore"
    shutil.rmtree(td, ignore_errors=True)
    td.mkdir(parents=True, exist_ok=True)
    p = td / "store.json"
    store = MemoryStore(p)
    rag = EpisodicRAG(store, RAGConfig())

    print("\n== episodic rag ==")
    r = rag.retrieve([1.0, 0.0], level="lv")
    check("empty store -> nothing", r.records == [] and r.diagnostics["mode"] == "empty")

    rag.add_episode(result=episode(10, False, 25), level="lv", seed=10,
                    intent="i10", embedding=[1.0, 0.0], step_index=0)
    r = rag.retrieve([1.0, 0.0], level="lv")
    check("only-failures -> similarity_only fallback", r.diagnostics["mode"] == "similarity_only")

    rag.add_episode(result=episode(11, True, 5), level="lv", seed=11,
                    intent="i11", embedding=[0.0, 1.0], step_index=1)
    r = rag.retrieve([1.0, 0.0], level="lv")
    kinds = sorted(x.kind for x in r.records)
    check("both classes -> one success + one failure",
          r.diagnostics["mode"] == "success_and_failure" and kinds == [FAILURE, SUCCESS], kinds)

    r_self = rag.retrieve([1.0, 0.0], level="lv", exclude_seeds=[10])
    check("excludes own seed", all(x.seed != 10 for x in r_self.records))
    check("other level is invisible", rag.retrieve([1.0, 0.0], level="other").records == [])
    check("verifier feedback stored", "Verifier: SUCCESS" in store.get("rag_00002").content)
    check("q untouched by rag", all(x.q == 0.0 and x.q_visits == 0 for x in store.records))

    block = format_rag_block(r.records)
    check("rag block has header + rollouts", "Retrieved Past Rollouts" in block and block.count("### Past rollout") == 2)
    check("token count works", (count_tokens(block) or 0) > 50, count_tokens(block))
    check("empty block is empty", format_rag_block([]) == "")

    store.save()
    reloaded = MemoryStore(p)
    check("store round-trips", len(reloaded.records) == 2 and reloaded.next_id("x") == "x_00003")

print("\n== memrl ==")
cfg = MemRLConfig(k1=5, k2=2, delta=0.5, sim_norm_mean=0.5, sim_norm_std=0.1,
                  alpha=0.3, q_init=0.0, add_similarity_threshold=0.99,
                  novelty_threshold=0.98)
store = MemoryStore(None)
m = MemRL(store, cfg, summarizer=mock_summarizer)

r = m.retrieve([1.0, 0.0], level="lv")
check("empty store -> no injection", r.records == [] and r.diagnostics["phase_a"] == "empty_store")

# Three memories at descending similarity to the query [1,0].
import math as _m
def unit(deg):
    a = _m.radians(deg)
    return [_m.cos(a), _m.sin(a)]
for i, (deg, success) in enumerate([(0, True), (30, False), (80, True)]):
    m.add_episode(result=episode(20 + i, success, 5 if success else 25), level="lv",
                  seed=20 + i, intent=f"i{20+i}", embedding=unit(deg), step_index=i)
check("write-back grew memory", len(store.records) == 3, len(store.records))
check("success content is proceduralized", "SCRIPT:" in store.records[0].content)
check("failure content is a reflection", "TASK REFLECTION" in store.records[1].content)

r = m.retrieve(unit(0), level="lv")
sims = [c["similarity"] for c in r.diagnostics["selected"]]
check("phase A drops below-delta (cos80=0.17)", len(r.records) == 2 and min(sims) > 0.5, sims)
check("phase B respects k2", len(r.records) <= cfg.k2)

# All-Q-equal: ordering is by similarity.
check("ties broken by similarity", r.records[0].seed == 20, [x.seed for x in r.records])

# Utility update: r=-1 on failure for both injected memories.
info = m.update_utilities(r, episode(99, False, 25))
check("MC update magnitude", all(abs(u["new_q"] - (0.7 * 0 + 0.3 * -1)) < 1e-9 for u in info["updates"]),
      [u["new_q"] for u in info["updates"]])
check("visits incremented", all(x.q_visits == 1 for x in r.records))
check("reward is +-1 by default", info["reward"] == -1.0)

# Now a positive episode on the *second* memory only, to check Q reordering.
m.update_utilities(type(r)(records=[store.records[1]], diagnostics={}), episode(98, True, 5))
q0, q1 = store.records[0].q, store.records[1].q
check("Q differentiates memories", q1 > q0, (round(q0, 3), round(q1, 3)))

r2 = m.retrieve(unit(0), level="lv")
check("higher-Q memory can outrank higher-sim one",
      r2.records[0].seed == 21, [(x.seed, round(x.q, 3)) for x in r2.records])

# lambda_q = 0 -> pure similarity (the ablation)
cfg0 = MemRLConfig(**{**cfg.__dict__, "weight_q": 0.0, "weight_sim": 1.0})
m0 = MemRL(store, cfg0, summarizer=mock_summarizer)
r3 = m0.retrieve(unit(0), level="lv")
check("lambda_q=0 ignores Q", r3.records[0].seed == 20, [(x.seed, round(x.q, 3)) for x in r3.records])

# Phase A empty when everything is below delta
cfg_hi = MemRLConfig(**{**cfg.__dict__, "delta": 0.999})
r4 = MemRL(store, cfg_hi, summarizer=mock_summarizer).retrieve(unit(45), level="lv")
check("below-delta -> zero-shot", r4.records == [] and r4.diagnostics["phase_a"] == "below_delta")

# q_min_threshold filter
cfg_qmin = MemRLConfig(**{**cfg.__dict__, "q_min_threshold": 10.0})
r5 = MemRL(store, cfg_qmin, summarizer=mock_summarizer).retrieve(unit(0), level="lv")
check("q floor filters everything", r5.records == [] and r5.diagnostics["phase_a"] == "q_floor_filtered")

# Near-duplicate intent -> update in place, no growth
cfg_dup = MemRLConfig(**{**cfg.__dict__, "add_similarity_threshold": 0.5})
m_dup = MemRL(store, cfg_dup, summarizer=mock_summarizer)
n_before = len(store.records)
m_dup.add_episode(result=episode(30, True, 5), level="lv", seed=30,
                  intent="i30", embedding=unit(1), step_index=9)
check("near-duplicate folds in place", len(store.records) == n_before, len(store.records))
check("merged seed recorded", 30 in store.records[0].meta.get("merged_seeds", []))
# The merged record must describe the episode it now holds, not the one that
# created the slot — otherwise seed/iterations/reward contradict the content.
merged = store.records[0]
check("merge adopts new seed", merged.seed == 30, merged.seed)
check("merge adopts new outcome", merged.success is True and merged.kind == SUCCESS)
check("merge adopts new iterations", merged.iterations == 5, merged.iterations)
check("merge keeps its learned Q", merged.q_visits > 0, merged.q_visits)

# An empty summary must not silently produce a memory with a blank SCRIPT.
def _blank(kind, intent, traj): return "   "
m_blank = MemRL(MemoryStore(None), MemRLConfig(), summarizer=_blank)
rec = m_blank.add_episode(result=episode(70, True, 5), level="lv", seed=70,
                          intent="i70", embedding=[1.0, 0.0], step_index=0)
check("blank summary is detectable", "SCRIPT:\n\n\nTRAJECTORY:" in rec.content)

block = format_memrl_block(r2.records)
check("memrl block shows Q", "utility Q=" in block and "Archived Memories" in block)

print("\n== shaped reward variant ==")
m_shaped = MemRL(MemoryStore(None), MemRLConfig(use_shaped_reward=True), summarizer=mock_summarizer)
check("shaped success reward", m_shaped.episode_reward(episode(1, True, 5)) == 0.75)
check("shaped failure reward", m_shaped.episode_reward(episode(1, False, 25)) == -0.5)

print("\n== features / numeric key ==")
v = parse_features("green_ball -0.33 +0.39 r+0.25 dyn\ngreen->blue dx-1.31 d+4.62")
check("parses signed fixed-point floats", v == [-0.33, 0.39, 0.25, -1.31, 4.62], v)
check("ignores unsigned tokens", parse_features("a 1.5 +2.00") == [2.0])

sc = FeatureScaler.fit([[0.0, 10.0], [2.0, 10.0], [4.0, 10.0]])
check("scaler mean per dim", sc.mean == [2.0, 10.0], sc.mean)
check("zero-variance dim gets std 1", sc.std[1] == 1.0, sc.std)
check("transform centers", sc.transform([2.0, 10.0]) == [0.0, 0.0])
try:
    sc.transform([1.0]); check("wrong dim raises", False)
except ValueError:
    check("wrong dim raises", True)

sc_path = td / "features.json"
sc.save(sc_path)
check("scaler round-trips", FeatureScaler.load(sc_path).mean == sc.mean)
check("numeric embedder standardizes",
      NumericEmbedder(FeatureScaler([0.0, 0.0], [1.0, 1.0])).embed_one("a +1.00 +2.00") == [1.0, 2.0])

check("spearman perfect", abs(spearman([1, 2, 3, 4], [10, 20, 30, 40]) - 1.0) < 1e-9)
check("spearman inverted", abs(spearman([1, 2, 3, 4], [40, 30, 20, 10]) + 1.0) < 1e-9)
check("spearman handles ties", spearman([1, 1, 2, 2], [1, 1, 2, 2]) == 1.0)
rows = [[0, 3, 2, 1], [3, 0, 1, 2]]
check("top-k overlap identical", top_k_overlap(rows, rows, 1) == 1.0)
check("top-k overlap disjoint", top_k_overlap(rows, [[0, 1, 2, 3], [1, 0, 3, 2]], 1) == 0.0)
# Same metric on both sides: identical keys must score a perfect rho, otherwise
# the diagnostic understates agreement (cosine-vs-euclidean bug).
keys = [[0.0, 0.5], [1.0, 0.0], [0.0, 1.0], [2.0, 2.0], [-1.0, 0.3]]
ra = rank_agreement(keys, keys, ks=(1, 2))
check("identical keys -> rho 1", abs(ra["spearman"] - 1.0) < 1e-9, ra["spearman"])
check("identical keys -> full overlap", ra["top_k_overlap"]["top1"] == 1.0)

print("\n== calibrate quantile ==")
vals = [0.1, 0.2, 0.3, 0.4, 0.5]
check("p80 of 5 values", abs(quantile(vals, 0.8) - 0.42) < 1e-9, quantile(vals, 0.8))
check("p50 median", abs(quantile(vals, 0.5) - 0.3) < 1e-9)

print(f"\n{ok} checks passed; failures: {check.failed}")
sys.exit(1 if check.failed else 0)
