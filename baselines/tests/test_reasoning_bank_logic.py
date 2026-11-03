"""Offline checks of the ReasoningBank logic: no InterPhyre, no LLM, no embeddings.

Covers the Markdown item parser, top-k cosine retrieval, append-only write-back
(no merge/cap/utility), strategy-vs-lesson rendering, and block formatting. Runs
in ~1s with no GPU.

    python baselines/tests/test_reasoning_bank_logic.py
"""
import sys, types, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Stub skillrl.distillation.teacher_prompts so trajectory_fmt's reward import does
# not pull in the whole distillation stack (same shim as test_memory_logic.py).
mod = types.ModuleType("skillrl.distillation.teacher_prompts")
def compute_trajectory_reward(t):
    if t.get("success"):
        i = t.get("iterations", 25)
        return 1.0 if i <= 3 else 0.75 if i <= 7 else 0.5 if i <= 15 else 0.25
    return -0.75 if t.get("iterations", 25) < 10 else -0.5
mod.compute_trajectory_reward = compute_trajectory_reward
import skillrl, skillrl.distillation  # noqa: F401
sys.modules["skillrl.distillation.teacher_prompts"] = mod

from baselines.memory.store import MemoryStore, SUCCESS, FAILURE
from baselines.reasoning_bank.rb import ReasoningBank, ReasoningBankConfig
from baselines.reasoning_bank.extract import parse_memory_items, mock_extractor
from baselines.reasoning_bank.blocks import format_reasoning_bank_block
from baselines.memory.blocks import count_tokens

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

print("\n== parser ==")
md = """```
# Memory Item 1
## Title Drop on the long lever arm
## Description When a lever/pivot is present and the target is far
## Content Release the red ball above the far end of the lever to maximise the green ball's launch arc.

# Memory Item 2
## Title Mind the pivot side
## Description Use when the lever is asymmetric
## Content Placing on the short arm wastes energy; prefer the arm whose tip is lower.
```"""
items = parse_memory_items(md)
check("parses two items", len(items) == 2, len(items))
check("captures inline title", items[0]["title"] == "Drop on the long lever arm", items[0]["title"])
check("captures content", items[0]["content"].startswith("Release the red ball"), items[0]["content"][:20])
check("respects max_items", len(parse_memory_items(md, max_items=1)) == 1)

# value on the line after the header, mixed heading depths, no fences
md2 = "### Memory Item 1\n# Title\nCheck the gap width\n#### Content\nMeasure the gap before placing."
it2 = parse_memory_items(md2)
check("handles value-on-next-line + mixed depths", it2 and it2[0]["title"] == "Check the gap width", it2)
check("empty text -> no items", parse_memory_items("   ") == [])
off = parse_memory_items("just some prose with no headers at all")
check("off-format blob is kept, not swallowed", len(off) == 1 and off[0]["content"].startswith("just some"))

print("\n== retrieval ==")
td = Path(__file__).resolve().parent / "_teststore_rb"
shutil.rmtree(td, ignore_errors=True)
td.mkdir(parents=True, exist_ok=True)
store = MemoryStore(td / "store.json")
rb = ReasoningBank(store, ReasoningBankConfig(k=1), extractor=mock_extractor)

r = rb.retrieve([1.0, 0.0], level="lv")
check("empty store -> nothing", r.records == [] and r.diagnostics["mode"] == "empty")

print("\n== append-only write-back ==")
rec_s = rb.add_episode(result=episode(10, True, 5), level="lv", seed=10,
                       intent="i10", embedding=[1.0, 0.0], step_index=0)
rec_f = rb.add_episode(result=episode(11, False, 25), level="lv", seed=11,
                       intent="i11", embedding=[0.0, 1.0], step_index=1)
check("success record kind", rec_s.kind == SUCCESS)
check("failure record kind", rec_f.kind == FAILURE)
check("items kept in meta", rec_s.meta.get("num_items") == 1 and "items" in rec_s.meta)
check("success renders as Strategy", "[Strategy:" in rec_s.content, rec_s.content[:40])
check("failure renders as Lesson", "[Lesson:" in rec_f.content, rec_f.content[:40])

# Re-adding a near-identical intent must GROW the bank (append-only; no merge).
n_before = len(store.records)
rb.add_episode(result=episode(12, True, 5), level="lv", seed=12,
               intent="i10-again", embedding=[1.0, 0.0001], step_index=2)
check("append-only: duplicate intent still grows the bank",
      len(store.records) == n_before + 1, len(store.records))

print("\n== top-k retrieval ==")
r = rb.retrieve([1.0, 0.0], level="lv")
check("k=1 returns one record", len(r.records) == 1 and r.diagnostics["mode"] == "topk_similarity")
check("k=1 returns the most similar (seed 10 or 12, not 11)", r.records[0].seed in (10, 12), r.records[0].seed)
check("excludes own seed", all(x.seed != 10 for x in rb.retrieve([1.0, 0.0], level="lv", exclude_seeds=[10, 12]).records))
check("other level invisible", rb.retrieve([1.0, 0.0], level="other").records == [])

rb3 = ReasoningBank(store, ReasoningBankConfig(k=3), extractor=mock_extractor)
r3 = rb3.retrieve([1.0, 0.0], level="lv")
check("k=3 widens the injected budget", len(r3.records) == 3, len(r3.records))

print("\n== no-op utilities + no utility drift ==")
info = rb.update_utilities(r, episode(99, False, 25))
check("update_utilities is a no-op", info == {"updates": []})
check("q never touched", all(x.q == 0.0 and x.q_visits == 0 for x in store.records))

print("\n== empty extraction ==")
rb_empty = ReasoningBank(MemoryStore(None), ReasoningBankConfig(),
                         extractor=lambda s, i, t: [])
rec = rb_empty.add_episode(result=episode(70, True, 5), level="lv", seed=70,
                           intent="i70", embedding=[1.0, 0.0], step_index=0)
check("no items -> nothing added", rec is None and len(rb_empty.store.records) == 0)

print("\n== block formatting ==")
block = format_reasoning_bank_block(r3.records)
check("block has header", "Learned Reasoning Memory" in block)
check("block lists memories", block.count("### Memory ") == 3, block.count("### Memory "))
check("token count works", (count_tokens(block) or 0) > 20, count_tokens(block))
check("empty block is empty", format_reasoning_bank_block([]) == "")

print("\n== store round-trip ==")
store.save()
reloaded = MemoryStore(td / "store.json")
check("store round-trips", len(reloaded.records) == len(store.records))
check("meta survives round-trip", reloaded.records[0].meta.get("num_items") == 1)

print(f"\n{ok} checks passed; failures: {check.failed}")
sys.exit(1 if check.failed else 0)
