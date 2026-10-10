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
    validate_operations,
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

shutil.rmtree(OUT, ignore_errors=True)
print(f"OK: {ok} checks passed")
