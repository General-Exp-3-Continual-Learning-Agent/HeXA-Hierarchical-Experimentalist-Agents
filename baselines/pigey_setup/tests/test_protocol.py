"""Offline checks for the protocol pieces: initial-bank seeding, resume listing, paired comparison."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # baselines/pigey_setup

from hexa import compare, loop  # noqa: E402
from runner import completed_trials  # noqa: E402


def write_trial(root, task, episode, success, *, status="completed", suite="libero_goal_task", cost=0.1,
                injection=None, stamp="2026-11-03T00-00-00"):
    directory = Path(root) / "sonnet" / f"{suite}_t{task:02d}_e{episode:02d}" / stamp
    directory.mkdir(parents=True, exist_ok=True)
    trial = {"suite": suite, "task_id": task, "episode": episode, "task_description": f"task {task}",
             "success": None if status == "backend_error" else success, "status": status, "llm_steps": 5,
             "claude_code_calls": [{"total_cost_usd": cost}]}
    if injection is not None:
        trial["skill_injection"] = injection
    (directory / "trial.json").write_text(json.dumps(trial))


class SeedBankTests(unittest.TestCase):
    def test_copies_once_and_refuses_a_different_bank(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            tmp = Path(tmp)
            source = tmp / "initial.json"
            source.write_text('{"a": 1}')
            out = tmp / "run"
            out.mkdir()
            target = loop.seed_bank(source, out)
            self.assertEqual(target, out / "bank_round_1.json")
            self.assertEqual(target.read_text(), '{"a": 1}')
            provenance = json.loads((out / "initial_bank_source.json").read_text())
            self.assertEqual(provenance["source"], str(source.resolve()))
            self.assertEqual(len(provenance["sha256"]), 64)
            self.assertEqual(loop.seed_bank(source, out), target)  # resuming is a no-op
            other = tmp / "other.json"
            other.write_text('{"a": 2}')
            with self.assertRaises(SystemExit):
                loop.seed_bank(other, out)
            self.assertEqual(target.read_text(), '{"a": 1}')


class CompletedTrialsTests(unittest.TestCase):
    def test_lists_finished_pairs_of_the_suite_only(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            write_trial(tmp, 3, 40, True)
            write_trial(tmp, 3, 41, False)
            write_trial(tmp, 4, 40, True, status="backend_error")
            write_trial(tmp, 5, 40, True, suite="libero_spatial")
            self.assertEqual(completed_trials.completed(tmp, "libero_goal_task"), [(3, 40), (3, 41)])
            self.assertEqual(completed_trials.completed(tmp), [(3, 40), (3, 41), (5, 40)])


class CompareTests(unittest.TestCase):
    def test_mcnemar_exact(self):
        self.assertAlmostEqual(compare.mcnemar_exact(11, 3), 0.0574, places=4)
        self.assertEqual(compare.mcnemar_exact(0, 0), 1.0)
        self.assertEqual(compare.mcnemar_exact(2, 2), 1.0)

    def test_paired_report_and_audit(self):
        injection = {"skill_ids": ["s1", "s2"], "mistake_ids": ["m1"], "dropped_skills": [], "dropped_mistakes": [],
                     "parent_overlap": [], "block_tokens": 900}
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            bank, none = Path(tmp) / "bank", Path(tmp) / "none"
            for episode, outcome in zip((25, 26, 27), (True, True, False)):
                write_trial(bank, 0, episode, outcome, injection=injection, cost=0.5)
            write_trial(bank, 1, 25, True, injection=dict(injection, parent_overlap=["s1"]))
            write_trial(bank, 1, 26, True, status="backend_error")                     # unfinished
            write_trial(bank, 1, 26, False, stamp="2026-11-03T01-00-00", injection=injection)  # rerun counts
            for episode, outcome in zip((25, 26, 27), (False, True, False)):
                write_trial(none, 0, episode, outcome, cost=1.0)
            write_trial(none, 1, 25, True)
            write_trial(none, 1, 26, True)
            write_trial(none, 1, 27, False)                                           # missing from the bank arm
            result = compare.compare(compare.load_arm(bank, "libero_goal_task"), compare.load_arm(none, "libero_goal_task"))
        self.assertEqual(result["paired_episodes"], 5)
        self.assertEqual((result["with_bank"]["successes"], result["no_bank"]["successes"]), (3, 3))
        self.assertEqual(result["only_with_bank"], ["t00_e25"])
        self.assertEqual(result["only_no_bank"], ["t01_e26"])
        self.assertEqual(result["mcnemar_p"], 1.0)
        self.assertEqual(result["per_task"][0], {"episodes": 3, "with_bank": 2, "no_bank": 1, "description": "task 0"})
        self.assertEqual(result["with_bank"]["cost_usd"], 1.7)
        audit = result["audit"]
        self.assertEqual((audit["injected"], audit["dropped"], audit["parent_overlap"]), (5, 0, 1))
        self.assertEqual(audit["no_bank_arm_injected"], 0)
        self.assertEqual(result["missing"], {"with_bank": ["t01_e27"], "no_bank": []})
        report = compare.format_report(result)
        self.assertIn("with bank : 3/5", report)
        self.assertIn("1 parent-overlap", report)
        self.assertIn("t01_e27", report)


if __name__ == "__main__":
    unittest.main()
