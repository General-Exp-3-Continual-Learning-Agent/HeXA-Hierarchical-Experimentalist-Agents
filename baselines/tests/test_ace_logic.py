"""Offline checks of the ACE baseline: no InterPhyre, no LLM, no GPU.

  1. the vendored playbook mechanics behave identically to the reference
     `ace/playbook_utils.py` on the same operation sequence;
  2. bullet-ID citation, bullet-tag and Curator-op parsing;
  3. a full stream with scripted Generator/Reflector/Curator: warmup, counter
     updates, ADD merge, frozen seeds, resume after a crash between the
     trajectory write and the state write (re-curated, never re-solved);
  4. reflect-and-re-solve rounds are logged as training cost, never scored.

    python baselines/tests/test_ace_logic.py
"""
import importlib.util
import json
import shutil
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Same stub as test_memory_logic.py: keep trajectory_fmt's reward import light.
mod = types.ModuleType("skillrl.distillation.teacher_prompts")
mod.compute_trajectory_reward = lambda t: 1.0 if t.get("success") else -0.5
import skillrl, skillrl.distillation  # noqa: E401,F401
sys.modules["skillrl.distillation.teacher_prompts"] = mod
# stream.py imports memory_runner (-> react_agent -> interphyre); not needed here.
mr = types.ModuleType("baselines.runner.memory_runner")
mr.run_memory_augmented = None
sys.modules["baselines.runner.memory_runner"] = mr

from baselines.ace import playbook as pb
from baselines.ace.loop import (
    ACE, ACEConfig, ACEState, cited_bullet_ids, extract_bullet_tags,
    seed_state_from, validate_operations,
)
from baselines.ace.prompts import EMPTY_PLAYBOOK

ok = 0


def check(cond, msg):
    global ok
    if not cond:
        raise AssertionError(msg)
    ok += 1


# ── 1. vendored == reference ────────────────────────────────────────────────
utils_stub = types.ModuleType("utils")
utils_stub.get_section_slug = pb.get_section_slug
sys.modules["utils"] = utils_stub
spec = importlib.util.spec_from_file_location("ace_ref_playbook_utils", ROOT / "ace" / "playbook_utils.py")
ref = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref)
del sys.modules["utils"]

ops = [
    {"type": "ADD", "section": "strategies_and_insights", "content": "Simulate before finishing."},
    {"type": "ADD", "section": "common_mistakes_to_avoid", "content": "Do not overlap the green ball."},
    {"type": "ADD", "section": "no_such_section", "content": "Goes to OTHERS."},
]
mine, nid_m = pb.apply_curator_operations(EMPTY_PLAYBOOK, ops, 1)
theirs, nid_t = ref.apply_curator_operations(EMPTY_PLAYBOOK, ops, 1)
check(mine == theirs and nid_m == nid_t == 4, "apply_curator_operations diverges from reference")
tags = [{"id": "sai-00001", "tag": "helpful"}, {"id": "err-00002", "tag": "harmful"}]
check(pb.update_bullet_counts(mine, tags) == ref.update_bullet_counts(theirs, tags), "update_bullet_counts diverges")
check(pb.get_playbook_stats(mine) == ref.get_playbook_stats(theirs), "get_playbook_stats diverges")
check("[sai-00001] helpful=0 harmful=0 :: Simulate before finishing." in mine, "bullet id/format")
check("[misc-00003]" in mine, "unknown section -> OTHERS")
for name in ("parse_playbook_line", "extract_json_from_text", "extract_playbook_bullets",
             "format_playbook_line", "get_next_global_id"):
    a, b = getattr(pb, name), getattr(ref, name)
    check(a.__code__.co_code == b.__code__.co_code, f"{name} bytecode differs from reference")

# ── 2. parsing ─────────────────────────────────────────────────────────────
res = {"trajectory": [
    {"thought": "Following [sai-00001] and [err-00002], inspect.", "action": "get_level_state({})", "observation": "..."},
    {"thought": "Again [sai-00001]; ignore [Not-1] and [ab-00001].", "action": "simulate_action({})", "observation": "..."},
]}
check(cited_bullet_ids(res) == ["sai-00001", "err-00002"], "cited ids, order + dedupe + regex")
check(extract_bullet_tags('noise ```json\n{"reasoning":"r","bullet_tags":[{"id":"a-1","tag":"helpful"}]}\n```')
      == [{"id": "a-1", "tag": "helpful"}], "tags from fenced json")
check(extract_bullet_tags('prefix "bullet_tags": [{"id": "x-00001", "tag": "harmful"}] trailing {')
      == [{"id": "x-00001", "tag": "harmful"}], "tags via reference bracket fallback")
check(extract_bullet_tags("no json at all") == [], "no tags")
check(len(validate_operations('{"reasoning": "r", "operations": [{"type": "ADD", "section": "s", "content": "c"}]}')) == 1, "valid ops")
for bad in ('{"operations": []}', '{"reasoning": "r", "operations": [{"type": "ADD", "section": "s"}]}', "garbage"):
    try:
        validate_operations(bad)
        check(False, f"should reject {bad}")
    except ValueError:
        check(True, "")

# ── 3. scripted stream ────────────────────────────────────────────────────
OUT = ROOT / "baselines" / "tests" / "_test_ace_out"
shutil.rmtree(OUT, ignore_errors=True)

SUCCESS_SEEDS = {10, 12}
solve_calls = []


def fake_solve(seed, block):
    solve_calls.append((seed, "Reflection on your previous attempt" in block))
    cites = "per [sai-00001]" if "[sai-00001]" in block else "no playbook yet"
    ok_ = seed in SUCCESS_SEEDS
    return {
        "seed": seed, "level": "down_to_earth", "success": ok_,
        "action": [1.0, 2.0, 0.5], "iterations": 4 if ok_ else 25,
        "trajectory": [{"thought": f"I place the ball {cites}.",
                        "action": 'simulate_action({"x":1,"y":2,"r":0.5})',
                        "observation": "SUCCESS" if ok_ else "FAILURE"}],
        "final_observation": "done", "simulator_calls": 1,
        "added_context_tokens": len(block) // 4, "elapsed_time": 0.1,
    }


llm_log = []


def fake_llm(prompt, max_tokens):
    if prompt.startswith("You are a master curator"):
        llm_log.append("curator")
        n = sum(1 for r in llm_log if r == "curator")
        if n == 3:
            return "this is not json"           # one invalid curator reply
        return json.dumps({"reasoning": "r", "operations": [
            {"type": "ADD", "section": "strategies_and_insights", "content": f"insight {n}"},
            {"type": "UPDATE", "section": "x", "content": "ignored"},
        ]})
    llm_log.append("reflector")
    tag = "helpful" if "SUCCESS" in prompt.split("**Environment Feedback:**")[1][:80] else "harmful"
    return json.dumps({"reasoning": "r", "error_identification": "e", "root_cause_analysis": "c",
                       "correct_approach": "a", "key_insight": "k",
                       "bullet_tags": [{"id": "sai-00001", "tag": tag}]})


def make(seeds, **kw):
    cfg = ACEConfig(level="down_to_earth", seeds=seeds, output_dir=OUT, **kw)
    return ACE(cfg, model_fn=None, text_llm=fake_llm, task_text=lambda: "Level: down_to_earth\nSolve it.",
               solve=fake_solve)


warm = [fake_solve(1, ""), fake_solve(2, "")]
solve_calls.clear()
summary = make([10, 11, 12, 13], freeze_after=3).run(offline_trajectories=warm)

st = ACEState.load(OUT / "ace_state.json")
check(st.offline_seeds == [1, 2], "warmup recorded")
# warmup 2 curations + seeds 10,11,12 curated (13 frozen); call 3 invalid -> 4 bullets
check(st.counters["curator_calls"] == 5 and st.counters["curator_invalid"] == 1, f"curator counts {st.counters}")
check(st.counters["bullets_added"] == 4 and st.counters["non_add_ops_ignored"] == 4, "ADD-only applied")
check(pb.get_playbook_stats(st.playbook)["total_bullets"] == 4, "4 bullets in playbook")
check(st.curated_through == 2, "frozen seed not curated")
first = pb.parse_playbook_line([l for l in st.playbook.splitlines() if l.startswith("[sai-00001]")][0])
# seeds 10, 12 helpful; 11 harmful; warmup seed 2 cited nothing but the scripted
# Reflector tags it harmful anyway -- applied, as in the reference, and counted.
check((first["helpful"], first["harmful"]) == (2, 2), f"counters on sai-00001: {first}")
check(st.counters["tags_on_uncited_bullets"] == 2, f"uncited tags counted (both warmup seeds) {st.counters}")
check(summary["successes"] == 2 and summary["total_seeds"] == 4, "scored first attempts")
check(summary["per_seed"][3]["memory_frozen"] is True, "frozen flag")
check(solve_calls == [(10, False), (11, False), (12, False), (13, False)], "one solve per seed, no retries")
check((OUT / "playbooks" / "step002_seed12.txt").exists() and (OUT / "manifest.json").exists() is False,
      "snapshots written (manifest is the CLI's job)")

# resume: nothing to do
solve_calls.clear()
n_llm = len(llm_log)
make([10, 11, 12, 13], freeze_after=3).run(offline_trajectories=warm)
check(solve_calls == [] and len(llm_log) == n_llm, "full resume re-runs nothing")

# crash between trajectory write and state write: roll state back one step
state = json.loads((OUT / "ace_state.json").read_text())
state["curated_through"] = 1
(OUT / "ace_state.json").write_text(json.dumps(state))
solve_calls.clear()
make([10, 11, 12, 13], freeze_after=3).run(offline_trajectories=warm)
check(solve_calls == [], "re-curation does not re-solve")
check(ACEState.load(OUT / "ace_state.json").curated_through == 2, "re-curated seed 12")

# ── 4. reflect-and-re-solve ───────────────────────────────────────────────
shutil.rmtree(OUT, ignore_errors=True)
solve_calls.clear()
SUCCESS_SEEDS.clear()
s = make([20], resolve_rounds=2).run()
st = ACEState.load(OUT / "ace_state.json")
check(solve_calls == [(20, False), (20, True), (20, True)], f"2 retries with reflection: {solve_calls}")
check(st.counters["training_episodes"] == 2 and st.counters["training_simulator_calls"] == 2, "retry cost logged")
check(s["successes"] == 0 and s["total_simulator_calls"] == 1, "retries not scored")

# ── 5. frozen held-out eval from a finished run ──────────────────────────
SRC = OUT
FRZ = ROOT / "baselines" / "tests" / "_test_ace_frozen"
shutil.rmtree(FRZ, ignore_errors=True)
FRZ.mkdir()
src_state = ACEState.load(SRC / "ace_state.json")
(SRC / "summary_ace.json").rename(SRC / "summary_ace.json.bak")
try:
    seed_state_from(SRC, FRZ)
    check(False, "unfinished source must be refused")
except SystemExit:
    check(True, "")
(SRC / "summary_ace.json.bak").rename(SRC / "summary_ace.json")
info = seed_state_from(SRC, FRZ)
check(info["init_playbook_bullets"] == pb.get_playbook_stats(src_state.playbook)["total_bullets"], "init info")
fresh = ACEState.load(FRZ / "ace_state.json")
check(fresh.playbook == src_state.playbook and fresh.curated_through == -1
      and fresh.counters["curator_calls"] == 0, "only the playbook carries over")
check(seed_state_from(SRC, FRZ) == info, "resume onto the same playbook is allowed")
other = ACEState.load(FRZ / "ace_state.json"); other.playbook += "\n[x] helpful=0 harmful=0 :: y"
other.save(FRZ / "ace_state.json")
try:
    seed_state_from(SRC, FRZ)
    check(False, "resume onto a different playbook must be refused")
except SystemExit:
    check(True, "")
fresh.save(FRZ / "ace_state.json")

n_llm = len(llm_log); solve_calls.clear()
cfg = ACEConfig(level="down_to_earth", seeds=[57, 58], output_dir=FRZ, freeze_after=0)
fs = ACE(cfg, model_fn=None, text_llm=fake_llm, task_text=lambda: "t", solve=fake_solve).run()
after = ACEState.load(FRZ / "ace_state.json")
check(len(llm_log) == n_llm, "frozen eval makes no Reflector/Curator calls")
check(after.playbook == src_state.playbook, "frozen playbook unchanged")
check(all(p["memory_frozen"] for p in fs["per_seed"]) and len(solve_calls) == 2, "both seeds frozen, solved once")
shutil.rmtree(FRZ, ignore_errors=True)

# ── 6. grow-and-refine dedup (stub embedder, scripted merge) ───────────────
import numpy as np
from baselines.ace import dedup as dd
from baselines.ace.llm import make_text_llm

class StubEmbedder:
    """Same vector for texts sharing their first word, orthogonal otherwise."""
    def encode(self, texts, convert_to_numpy=True, show_progress_bar=False):
        keys = sorted({t.split()[0] for t in texts})
        return np.array([[1.0 if t.split()[0] == k else 0.0 for k in keys] for t in texts], dtype=np.float32)

merge_calls = []
def merge_llm(prompt, max_tokens, temperature=None):
    merge_calls.append(temperature)
    bid = re.search(r"Keep the ID from the first entry: \[([^\]]+)\]", prompt).group(1)
    h = re.search(r"combined counts: helpful=(\d+) harmful=(\d+)", prompt)
    return f"[{bid}] helpful={h.group(1)} harmful={h.group(2)} :: Simulate first; merged."

import re
dd.DEDUP_AVAILABLE = True
an = dd.BulletpointAnalyzer(merge_llm, "m", 512)
an.embedding_model = StubEmbedder()
book = EMPTY_PLAYBOOK.replace("## STRATEGIES & INSIGHTS\n", "## STRATEGIES & INSIGHTS\n"
    "[sai-00001] helpful=2 harmful=1 :: Simulate before finishing.\n"
    "[sai-00002] helpful=0 harmful=3 :: Simulate twice before finishing.\n"
    "[sai-00003] helpful=1 harmful=0 :: Place red near the edge.\n")
out = an.analyze(book, threshold=0.9, merge=True)
ids = re.findall(r"^\[([^\]]+)\]", out, re.M)
check(ids == ["sai-00001", "sai-00003"], f"duplicates merged into first id: {ids}")
check("[sai-00001] helpful=2 harmful=4 :: Simulate first; merged." in out, "counts summed in merge")
check(merge_calls == [0.3], "merge runs at the reference temperature 0.3")
check(an.analyze(EMPTY_PLAYBOOK) == EMPTY_PLAYBOOK, "empty playbook untouched")

# temperature forwarding through make_text_llm, and the greedy path
seen = []
fn = lambda msgs, t, m: (seen.append(t), "x")[1]
call = make_text_llm("hf", fn)
call("p", 8); call("p", 8, temperature=0.3)
check(seen == [0.0, 0.3], f"role temp 0 by default, per-call override: {seen}")
src = open(ROOT / "react_agent" / "react_agent.py").read()
check('{"do_sample": False}' in src and "temp and temp > 0" in src, "load_qwen_model decodes greedily at temp 0")

# ── edit ops (--edit-ops): UPDATE / MERGE / DELETE ──────────────────────────
from baselines.ace.edit_ops import apply_edit_operations, validate_edit_operations, CURATOR_PROMPT_EDIT
book = EMPTY_PLAYBOOK.replace("## STRATEGIES & INSIGHTS\n", "## STRATEGIES & INSIGHTS\n"
    "[sai-00001] helpful=2 harmful=1 :: Simulate before finishing.\n"
    "[sai-00002] helpful=0 harmful=3 :: Simulate twice.\n"
    "[sai-00003] helpful=1 harmful=0 :: Place red near the edge.\n")
eops = validate_edit_operations([
    {"type": "UPDATE", "bullet_id": "sai-00003", "content": "Place red at the ramp edge."},
    {"type": "MERGE", "source_ids": ["sai-00001", "sai-00002"], "content": "Always simulate first."},
    {"type": "DELETE", "bullet_id": "[sai-00099]"},
    {"type": "ADD", "section": "common_mistakes_to_avoid", "content": "Do not guess radii."},
    {"type": "ADD", "section": "no_such_section", "content": "Misc note."},
    {"type": "CREATE_META", "section_name": "x"},
])
out, nid, st = apply_edit_operations(book, eops, 4)
check("[sai-00003] helpful=1 harmful=0 :: Place red at the ramp edge." in out, "UPDATE keeps id+counts")
check("[sai-00001] helpful=2 harmful=4 :: Always simulate first." in out and "sai-00002" not in out,
      "MERGE keeps first id, sums counts, drops the rest")
check("[err-00004] helpful=0 harmful=0 :: Do not guess radii." in out, "ADD into named section")
check("[misc-00005] helpful=0 harmful=0 :: Misc note." in out and nid == 6, "unknown section -> OTHERS")
check(out.index("err-00004") < out.index("## PROBLEM-SOLVING"), "ADD lands inside its section")
check(st == {"added": 2, "updated": 1, "merged_sources_removed": 1, "merges": 1, "deleted": 0,
             "skipped_unknown_id": 1, "skipped_unknown_type": 1}, f"edit stats {st}")
out2, _, st2 = apply_edit_operations(out, [{"type": "DELETE", "bullet_id": "sai-00003"}], nid)
check("sai-00003" not in out2 and st2["deleted"] == 1, "DELETE removes the bullet")
for bad in ([{"type": "UPDATE", "bullet_id": "a"}], [{"type": "MERGE", "source_ids": ["a"], "content": "c"}]):
    try:
        validate_edit_operations(bad); check(False, f"should reject {bad}")
    except ValueError:
        check(True, "")
CURATOR_PROMPT_EDIT.format(current_step=1, total_samples=2, token_budget=3, playbook_stats="{}",
                           recent_reflection="r", current_playbook="p", question_context="q")

# edit ops end to end through ACE.curate
EDT = OUT.parent / "_ace_edit_test"
shutil.rmtree(EDT, ignore_errors=True)
resp = json.dumps({"reasoning": "r", "operations": [
    {"type": "ADD", "section": "strategies_and_insights", "content": "A."},
    {"type": "ADD", "section": "strategies_and_insights", "content": "B."},
    {"type": "MERGE", "source_ids": ["sai-00001", "sai-00002"], "content": "A and B."},
    {"type": "UPDATE", "bullet_id": "sai-00001", "content": "A and B, refined."}]})
seen_prompts = []
ed = ACE(ACEConfig(level="down_to_earth", seeds=[1], output_dir=EDT, edit_ops=True), model_fn=None,
         text_llm=lambda p, m, **k: (seen_prompts.append(p), resp)[1], task_text=lambda: "t", solve=None)
ed.curate("refl", 1, 1, "t")
c = ed.state.counters
check("[sai-00001] helpful=0 harmful=0 :: A and B, refined." in ed.state.playbook
      and "sai-00002" not in ed.state.playbook, "edit ops applied through curate")
check((c["bullets_added"], c["bullets_merged_away"], c["bullets_updated"]) == (2, 1, 1), f"edit counters {c}")
check("DELETE" in seen_prompts[0] and "only supports ADD" not in seen_prompts[0], "edit curator prompt used")
shutil.rmtree(EDT, ignore_errors=True)

# pruning trigger (--token-cap): lowest utility first, oldest on ties
from baselines.ace.prune import prune_to_cap
from baselines.memory.blocks import count_tokens as _ct
book = EMPTY_PLAYBOOK.replace("## STRATEGIES & INSIGHTS\n", "## STRATEGIES & INSIGHTS\n"
    "[sai-00001] helpful=3 harmful=0 :: " + "keep this useful one " * 20 + "\n"
    "[sai-00002] helpful=0 harmful=0 :: " + "old untagged " * 20 + "\n"
    "[sai-00003] helpful=0 harmful=2 :: " + "harmful advice " * 20 + "\n"
    "[sai-00004] helpful=0 harmful=0 :: " + "new untagged " * 20 + "\n")
same, rm0 = prune_to_cap(book, 10**6)
check(same == book and rm0 == [], "under cap: untouched")
full = _ct(book)
out, rm = prune_to_cap(book, full - 1)
check([r["id"] for r in rm] == ["sai-00003"], f"net-harmful bullet pruned first: {rm}")
out, rm = prune_to_cap(book, full - 70)
check([r["id"] for r in rm] == ["sai-00003", "sai-00002"] and "sai-00004" in out and "sai-00001" in out,
      f"then oldest untagged: {rm}")
check(_ct(out) <= full - 70 and "## OTHERS" in out, "result under cap, headers kept")

shutil.rmtree(OUT, ignore_errors=True)
print(f"OK: {ok} checks passed")
