"""Offline checks for the HeXA skill-bank port: no teacher calls, simulator, or servers."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # baselines/pigey_setup

from hexa import teacher as hexa_teacher  # noqa: E402
from hexa import trajectory as hexa_trajectory  # noqa: E402
from hexa.trajectory import load_trial_trajectories, transcript_steps, trajectory_reward, trial_to_trajectory  # noqa: E402
from runner import run_claude  # noqa: E402

SUITE = "libero_goal_task"
UPSTREAM_PERCEPTION = ['--perception', 'gemini', '--verifier', 'gemini', '--sam', 'inprocess']


def trial(task_id, episode, success, llm_steps=4, status="completed", suite=SUITE, terminated=False, cost=0.1):
    transcript = [
        {"role": "system", "content": "prompt"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "Grasp the bottle.", "tool_calls": ["Grasp"]},
        {"role": "tool_call", "content": json.dumps({"target_index": 3}), "tool": "Grasp"},
        {"role": "tool_result", "content": "HELD", "tool": "Grasp", "images": ["a.png"]},
        {"role": "assistant", "content": "Thinking only.", "tool_calls": []},
        {"role": "assistant", "content": "Place it.", "tool_calls": ["Place", "GoHome"]},
        {"role": "tool_call", "content": json.dumps({"target_index": 1, "surface_type": "flat"}), "tool": "Place"},
        {"role": "tool_result", "content": "RELEASED", "tool": "Place"},
        {"role": "tool_call", "content": "{}", "tool": "GoHome"},
        {"role": "tool_result", "content": "home", "tool": "GoHome"},
    ]
    if terminated:
        transcript.append({"role": "episode_terminated", "content": "exited"})
    return {"suite": suite, "task_id": task_id, "episode": episode, "task_description": f"task {task_id}",
            "transcript": transcript, "success": None if status == "backend_error" else success, "status": status,
            "llm_steps": llm_steps, "env_steps": 900, "model": "claude-sonnet-5-5",
            "claude_code_calls": [{"total_cost_usd": cost}]}


def write_trials(root: Path, trials):
    for i, data in enumerate(trials):
        path = root / f"job{i // 3}" / "sonnet" / f"{data['suite']}_t{data['task_id']:02d}_e{data['episode']:02d}" / "ts"
        path.mkdir(parents=True)
        (path / "trial.json").write_text(json.dumps(data))


class TrajectoryTests(unittest.TestCase):
    def test_transcript_groups_reasoning_tools_and_results(self):
        steps = transcript_steps(trial(0, 0, True)["transcript"])
        self.assertEqual(len(steps), 3)
        self.assertEqual(steps[0]["thought"], "Grasp the bottle.")
        self.assertEqual(steps[0]["action"], 'Grasp({"target_index": 3})')
        self.assertEqual(steps[0]["observation"], "Grasp: HELD")
        self.assertEqual(steps[1]["action"], "(no tool call)")
        self.assertIn("Place(", steps[2]["action"])
        self.assertIn("GoHome({})", steps[2]["action"])
        self.assertEqual(steps[2]["observation"], "Place: RELEASED | GoHome: home")

    def test_trajectory_fields_and_reward(self):
        fast = trial_to_trajectory(trial(5, 3, True, llm_steps=2))
        self.assertEqual(fast["seed"], "t05_e03")
        self.assertEqual(fast["task_name"], SUITE)
        self.assertEqual(fast["iterations"], 2)
        self.assertEqual(trajectory_reward(fast), 1.0)
        self.assertEqual(trajectory_reward(trial_to_trajectory(trial(5, 3, True, llm_steps=8))), 0.75)
        self.assertEqual(trajectory_reward(trial_to_trajectory(trial(5, 3, True, llm_steps=20))), 0.25)
        horizon = trial_to_trajectory(trial(0, 0, False, llm_steps=3, terminated=True))
        self.assertEqual(horizon["end_reason"], "horizon")
        self.assertEqual(trajectory_reward(horizon), -0.5)
        budget = trial_to_trajectory(trial(0, 0, False, llm_steps=35))
        self.assertEqual(budget["end_reason"], "llm_budget")
        self.assertEqual(trajectory_reward(budget), -0.5)
        early = trial_to_trajectory(trial(0, 0, False, llm_steps=3))
        self.assertEqual(early["end_reason"], "other")
        self.assertEqual(trajectory_reward(early), -0.75)

    def test_loader_reads_nested_trials_and_skips_errors_and_other_suites(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            write_trials(root, [trial(0, 0, True), trial(1, 0, False), trial(2, 0, True, status="backend_error"),
                                trial(3, 0, True, suite="libero_spatial")])
            successes, failures = load_trial_trajectories(root, SUITE)
        self.assertEqual([t["seed"] for t in successes], ["t00_e00"])
        self.assertEqual([t["seed"] for t in failures], ["t01_e00"])
        self.assertTrue(successes[0]["path"].endswith("trial.json"))

    def test_teacher_rendering_names_task_and_reward(self):
        hexa_teacher.install(actor_prompt="")
        text = hexa_trajectory.format_trajectory_for_teacher(trial_to_trajectory(trial(8, 1, False, terminated=True)))
        self.assertIn('Trial t08_e01 | Task 8: "task 8" | FAILURE (horizon)', text)
        self.assertIn("Reward: -0.50", text)
        self.assertIn("Tool: Grasp(", text)


class SelectionTests(unittest.TestCase):
    def test_same_task_pairs_lead_then_balanced_backfill(self):
        hexa_teacher.install(actor_prompt="")
        trajs = [trial_to_trajectory(trial(0, e, s, llm_steps=n)) for e, s, n in
                 [(0, True, 2), (1, False, 35), (2, True, 9)]]
        trajs += [trial_to_trajectory(trial(1, 0, False, llm_steps=35)), trial_to_trajectory(trial(2, 0, True, llm_steps=3))]
        chosen = hexa_teacher.select_task_paired(trajs, 4, trajectory_reward)
        self.assertEqual([t["seed"] for t in chosen[:2]], ["t00_e00", "t00_e01"])
        self.assertEqual(len(chosen), 4)
        self.assertEqual(len({t["seed"] for t in chosen}), 4)

    def test_backfill_spreads_tied_failures_across_tasks(self):
        hexa_teacher.install(actor_prompt="")
        trajs = [trial_to_trajectory(trial(task, e, False, llm_steps=35)) for task in (0, 1, 2) for e in range(4)]
        chosen = hexa_teacher.select_task_paired(trajs, 6, trajectory_reward)
        self.assertEqual(sorted(t["task_id"] for t in chosen), [0, 0, 1, 1, 2, 2])

    def test_detection_lists_are_stripped_from_results(self):
        text = "Grasp 'plate' -> HELD. step=184, task done=False\nobjects: #0 world=(1,2,3) bbox=[1,2,3,4]; #1 ..."
        self.assertEqual(hexa_trajectory.strip_detections(text), "Grasp 'plate' -> HELD. step=184, task done=False [detection list omitted]")
        self.assertEqual(hexa_trajectory.strip_detections("plain"), "plain")


class InstallTests(unittest.TestCase):
    def setUp(self):
        hexa_teacher.install(actor_prompt="SYSTEM PROMPT {ROLL} with {braces}")

    def test_prompts_format_with_hexa_placeholders_and_quote_actor_prompt(self):
        import skillrl.distillation.distill as distill
        import skillrl.distillation.evolving_distill as evolving
        prompts = hexa_teacher.build_prompts("SYSTEM PROMPT {ROLL}")
        self.assertIs(distill.CONTRASTIVE_DISTILLATION_PROMPT.__class__, str)
        text = prompts["contrastive"].format(level_name=SUITE, level_description="d", n_successes=1, n_failures=1,
                                              success_block="S", failure_block="F")
        self.assertIn("SYSTEM PROMPT {ROLL}", text)
        self.assertIn("t05_e03", text)
        prompts["mistakes"].format(level_name=SUITE, level_description="d", failure_block="F")
        evolved = prompts["evolution"].format(level_name=SUITE, level_description="d", existing_skills_block="B",
                                              n_successes=1, n_failures=2, new_trajectories_block="T", max_skills=10,
                                              max_mistakes=5)
        self.assertIn("at most 10 skills and 5 mistakes", evolved)
        self.assertIs(evolving.SKILL_BANK_EVOLUTION_PROMPT, distill.__dict__ and evolving.SKILL_BANK_EVOLUTION_PROMPT)

    def test_distillation_uses_pigey_loader_reward_and_string_seeds(self):
        import skillrl.distillation.distill as distill
        from skillrl.distillation.distill import run_distillation
        responses = iter([
            json.dumps([{"title": "Grasp by index", "principle": "p", "when_to_apply": "w", "source_seeds": ["t00_e00", "t01_e00"]}]),
            json.dumps({"mistakes": [{"description": "d", "why_it_happens": "y", "how_to_avoid": "h"}], "partial_skills": []}),
        ])
        prompts = []

        def fake_teacher(prompt, model="x"):
            prompts.append(prompt)
            return next(responses)

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            write_trials(root / "trials", [trial(0, 0, True, llm_steps=2), trial(1, 0, False, llm_steps=35)])
            with patch.object(distill, "call_teacher", fake_teacher):
                bank = run_distillation(traj_dirs={SUITE: root / "trials"}, levels=[SUITE],
                                        output_path=root / "bank.json", teacher_model="claude-x", skip_general=True)
            skills = bank.level_skills[SUITE]
            self.assertEqual(skills[0].source_seeds, ["t00_e00", "t01_e00"])
            # Mean reward of (+1.0, -0.5) mapped from [-1, 1] onto [0.1, 1.0].
            self.assertAlmostEqual(skills[0].confidence, 0.625)
            self.assertEqual(len(bank.level_mistakes[SUITE]), 1)
            self.assertIn('Trial t00_e00 | Task 0: "task 0" | SUCCESS', prompts[0])
            self.assertIn("SYSTEM PROMPT {ROLL} with {braces}", prompts[0])
            self.assertIn("Suite: libero_goal_task", prompts[0])
            self.assertIn("LIBERO-Goal", prompts[0])

    def test_evolution_reads_a_round_directory_of_trials(self):
        import skillrl.distillation.evolving_distill as evolving
        from skillrl.core.skill_bank import Skill, SkillBank
        from skillrl.distillation.evolving_distill import evolve_skill_bank
        prev = SkillBank()
        prev.add_skill(Skill("lib_sk_000", "Old skill", "p", "w", source_level=SUITE, confidence=0.9))
        seen = {}

        def fake_teacher(prompt, model="x"):
            seen["prompt"] = prompt
            return json.dumps({"skills": [
                {"title": "Old skill", "principle": "p", "when_to_apply": "w", "source_seeds": [], "confidence": 0.8, "is_new": False},
                {"title": "New skill", "principle": "n", "when_to_apply": "w", "source_seeds": ["t04_e10"], "confidence": 0.5, "is_new": True},
            ], "mistakes": [], "removed_skill_titles": [], "reasoning": "r"})

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            write_trials(root / "round_1", [trial(4, 10, True, llm_steps=3), trial(7, 10, False, llm_steps=35)])
            with patch.object(evolving, "call_teacher", fake_teacher):
                bank = evolve_skill_bank(SUITE, prev, root / "round_1", root / "bank2.json", teacher_model="claude-x")
        titles = {s.title: s for s in bank.level_skills[SUITE]}
        self.assertEqual(set(titles), {"Old skill", "New skill"})
        self.assertEqual(titles["Old skill"].skill_id, "lib_sk_000")
        self.assertAlmostEqual(titles["Old skill"].confidence, 0.8)
        self.assertAlmostEqual(titles["New skill"].confidence, 1.0)  # a 3-step success is reward +1.0
        self.assertIn("Trial t04_e10", seen["prompt"])
        self.assertIn("Trial t07_e10", seen["prompt"])
        self.assertIn("[90%] Old skill", seen["prompt"])

    def test_render_block_uses_manipulation_heading_within_budget(self):
        from skillrl.core.skill_bank import Mistake, Skill, SkillBank
        bank = SkillBank()
        for i in range(3):
            bank.add_skill(Skill(f"s{i}", f"Skill {i}", "principle " * 20, "when", source_level=SUITE, confidence=1 - i / 10))
        bank.add_mistake(Mistake("m0", "desc", "why", "how", source_level=SUITE))
        block = hexa_teacher.render_block(bank, SUITE, budget_tokens=2000)
        self.assertTrue(block.block_text.startswith(hexa_teacher.SKILLS_BLOCK_HEADING))
        self.assertIn("Task-Specific Strategies (libero_goal_task)", block.block_text)
        self.assertEqual([s.skill_id for s in block.skills], ["s0", "s1", "s2"])
        self.assertEqual(len(block.mistakes), 1)
        small = hexa_teacher.render_block(bank, SUITE, budget_tokens=120)
        self.assertLess(len(small.skills), 3)
        self.assertTrue(small.budget_bound)

    def test_default_budget_fits_a_full_bank(self):
        from skillrl.core.skill_bank import Mistake, Skill, SkillBank
        bank = SkillBank()
        long = "A specific sentence about grasping, placing and verifying the held object. " * 4
        for i in range(10):
            bank.add_skill(Skill(f"s{i}", f"Skill number {i} with a title", long, long[:120], example=long[:100],
                                 source_level=SUITE, confidence=1 - i / 20))
        for i in range(5):
            bank.add_mistake(Mistake(f"m{i}", long[:150], long[:120], long, source_level=SUITE))
        block = hexa_teacher.render_block(bank, SUITE)
        self.assertEqual(len(block.skills), 10)
        self.assertEqual(len(block.mistakes), 5)
        self.assertFalse(block.budget_bound)
        self.assertLess(block.tokens, hexa_teacher.DEFAULT_INJECTION_BUDGET)


class RunnerInjectionTests(unittest.TestCase):
    def test_runner_appends_bank_to_system_prompt_and_records_it(self):
        from skillrl.core.skill_bank import Skill, SkillBank
        observed = {}
        backend = SimpleNamespace(errors=[], calls=[])

        class TrialLog:
            def __init__(self):
                self.data = {}
            def finish(self, success, *unused):
                observed.update(self.data, success=success)

        module = SimpleNamespace(TrialLog=TrialLog, get_system_prompt=lambda task, roll: f"UPSTREAM {task} {roll}")

        def main():
            observed["prompt"] = module.get_system_prompt("open the drawer", 500)
            module.TrialLog().finish(True, 10, 1, 1)
            return 0
        module.main = main
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            bank = SkillBank()
            bank.add_skill(Skill("lib_sk_000", "Verify the held object", "Re-perceive after a grasp.", "After HELD",
                                 source_level=SUITE, confidence=0.9))
            bank_path = Path(tmp) / "bank.json"
            bank.save(bank_path)
            with patch.dict('os.environ', {'SLURM_JOB_ID': 'test', 'GEMINI_API_KEY': 'test'}), \
                    patch('runner.run_claude.ClaudeCodeBackend', return_value=backend), \
                    patch('runner.run_claude.load_harness', return_value=module):
                code = run_claude.main(['--suite', SUITE, '--skill-bank', str(bank_path), *UPSTREAM_PERCEPTION])
        self.assertEqual(code, 0)
        self.assertTrue(observed["prompt"].startswith("UPSTREAM open the drawer 500\n\n=== LEARNED ROBOT"))
        self.assertIn("Verify the held object", observed["prompt"])
        self.assertEqual(observed["skill_injection"]["skill_ids"], ["lib_sk_000"])
        self.assertEqual(observed["skill_injection"]["level"], SUITE)
        self.assertGreater(observed["skill_injection"]["block_tokens"], 0)
        self.assertEqual(observed["skill_injection"]["seed"], "t00_e00")
        self.assertEqual(observed["skill_injection"]["parent_overlap"], [])

    def test_runner_reports_injected_entries_distilled_from_this_trial(self):
        from skillrl.core.skill_bank import Skill, SkillBank
        # Scratch can report a just-emptied directory as non-empty (see claude_code.py).
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            bank = SkillBank()
            bank.add_skill(Skill("lib_sk_001", "From this very trial", "p", "w", source_level=SUITE,
                                 source_seeds=["t04_e40"], confidence=0.9))
            bank_path = Path(tmp) / "bank.json"
            bank.save(bank_path)
            injection = run_claude.skill_injection(str(bank_path), SUITE, 4000, seed="t04_e40")
            clean = run_claude.skill_injection(str(bank_path), SUITE, 4000, seed="t04_e41")
        self.assertEqual(injection["parent_overlap"], ["lib_sk_001"])
        self.assertEqual(clean["parent_overlap"], [])
        self.assertEqual(run_claude.trial_seed(["--suite", SUITE, "--task", "4", "--episode", "40"]), "t04_e40")

    def test_runner_without_bank_leaves_prompt_alone(self):
        module = SimpleNamespace(TrialLog=type("T", (), {"__init__": lambda self: setattr(self, "data", {}),
                                                        "finish": lambda self, *a: None}),
                                 get_system_prompt=lambda task, roll: "UPSTREAM")
        module.main = lambda: 0
        backend = SimpleNamespace(errors=[], calls=[])
        with patch.dict('os.environ', {'SLURM_JOB_ID': 'test', 'GEMINI_API_KEY': 'test'}, clear=False), \
                patch('runner.run_claude.ClaudeCodeBackend', return_value=backend), \
                patch('runner.run_claude.load_harness', return_value=module):
            import os
            os.environ.pop("PIGEY_SKILL_BANK", None)
            run_claude.main(UPSTREAM_PERCEPTION)
        self.assertEqual(module.get_system_prompt("t", 1), "UPSTREAM")


class TeacherLogTests(unittest.TestCase):
    def test_logs_never_overwrite_earlier_calls(self):
        """hexa_loop clears ``calls`` each round and reuses one teacher dir across runs."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            tmp = Path(tmp)
            fake = tmp / "fake_claude"
            fake.write_text("#!/bin/sh\ncat >/dev/null\n"
                            "echo '{\"result\": \"ok\", \"subtype\": \"success\", \"is_error\": false}'\n")
            fake.chmod(0o755)
            log_dir = tmp / "teacher"
            first = hexa_teacher.ClaudeTeacher(log_dir, executable=str(fake))
            first("round one prompt")
            first.calls.clear()  # what hexa_loop does at every round boundary
            first("round two prompt")
            second = hexa_teacher.ClaudeTeacher(log_dir, executable=str(fake))  # a resumed run
            second("final evolution prompt")
            prompts = {p.name: p.read_text() for p in log_dir.glob("teacher_*_prompt.txt")}
            self.assertEqual(prompts, {"teacher_01_prompt.txt": "round one prompt",
                                       "teacher_02_prompt.txt": "round two prompt",
                                       "teacher_03_prompt.txt": "final evolution prompt"})
            self.assertEqual(second.calls[0]["index"], 3)
            self.assertEqual((log_dir / "teacher_03_response.txt").read_text(), "ok")


if __name__ == '__main__':
    unittest.main()
