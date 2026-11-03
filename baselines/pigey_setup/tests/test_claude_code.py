"""Offline transport checks; no model requests, simulator, or checkpoint."""
import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # baselines/pigey_setup

from runner.claude_code import ClaudeCodeBackend, ClaudeCodeError, action_schema, conversation_content, parse_result  # noqa: E402
from runner import run_claude  # noqa: E402

TOOLS = [{"name": "Grasp", "input_schema": {"type": "object", "properties": {
    "target_index": {"type": "integer"}}, "required": ["target_index"], "additionalProperties": False}}]
# Upstream Gemini perception keeps these runner tests about orchestration only.
UPSTREAM_PERCEPTION = ['--perception', 'gemini', '--verifier', 'gemini', '--sam', 'inprocess']
ACTION = {"reasoning": "Select detection 2.", "tool_calls": [{"name": "Grasp", "input": {"target_index": 2}}]}


def stream(action=None, **extra):
    result = {"type": "result", "subtype": "success", "is_error": False,
              "structured_output": ACTION if action is None else action,
              "usage": {"input_tokens": 3, "cache_read_input_tokens": 5,
                        "cache_creation_input_tokens": 7, "output_tokens": 11}, **extra}
    return json.dumps({"type": "system", "subtype": "init"}) + '\n' + json.dumps(result) + '\n'


def assistant_tool_use(name):
    return json.dumps({"type": "assistant", "message": {"content": [
        {"type": "tool_use", "id": "x", "name": name, "input": {}}]}}) + '\n'


class ClaudeCodeTests(unittest.TestCase):
    def test_images_and_order_survive_tool_result_history(self):
        image = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "camera-bytes"}}
        messages = [{"role": "assistant", "content": [{"type": "tool_use", "id": "prior", "name": "Perceive", "input": {}}]},
                    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "prior", "content": [
                        {"type": "text", "text": "agentview next"}, image,
                        {"type": "text", "text": "wrist next"}, image]}]}]
        before = copy.deepcopy(messages)
        content = conversation_content(messages)
        positions = [i for i, block in enumerate(content) if block['type'] == 'image']
        self.assertEqual(len(positions), 2)
        self.assertEqual(content[positions[0] - 1]['text'], 'agentview next')
        self.assertEqual(content[positions[1] - 1]['text'], 'wrist next')
        self.assertIn('prior', json.dumps(content))
        self.assertEqual(messages, before)

    def test_schema_and_usage_convert_back_to_pigey(self):
        response, _ = parse_result(stream(), action_schema(TOOLS))
        self.assertEqual(response['usage'], {'input_tokens': 15, 'output_tokens': 11})
        call = response['content'][1]
        self.assertEqual((call['name'], call['input']), ('Grasp', {'target_index': 2}))
        self.assertTrue(call['id'].startswith('toolu_cli_'))
        self.assertEqual(response['stop_reason'], 'tool_use')

    def test_direct_pigey_tool_calls_are_recorded(self):
        output = assistant_tool_use("Perceive") + assistant_tool_use("StructuredOutput") + stream()
        _, result = parse_result(output, action_schema(TOOLS))
        self.assertEqual(result["native_tool_attempts"], ["Perceive"])
        _, result = parse_result(stream(), action_schema(TOOLS))
        self.assertEqual(result["native_tool_attempts"], [])

    def test_invalid_actions_and_cli_errors_fail_loudly(self):
        bad_actions = [
            {'reasoning': '', 'tool_calls': [{'name': 'Bash', 'input': {}}]},
            {'reasoning': '', 'tool_calls': [{'name': 'Grasp', 'input': {}}]},
            {'reasoning': '', 'tool_calls': [{'name': 'Grasp', 'input': {'target_index': '2'}}]},
        ]
        for action in bad_actions:
            with self.subTest(action=action), self.assertRaises(ClaudeCodeError):
                parse_result(stream(action), action_schema(TOOLS))
        for output in ('not json', '{}\n', stream(is_error=True), stream(subtype='error_max_turns')):
            with self.subTest(output=output), self.assertRaises(ClaudeCodeError):
                parse_result(output, action_schema(TOOLS))

    @patch('runner.claude_code.shutil.which', return_value='/mock/claude')
    def test_cli_request_uses_native_images_and_cli_auth(self, _which):
        seen = {}
        def run(command, **kwargs):
            seen.update(command=command, **kwargs)
            self.assertTrue(Path(command[command.index('--system-prompt-file') + 1]).is_file())
            return subprocess.CompletedProcess(command, 0, stream(), '')
        with tempfile.TemporaryDirectory() as root, patch('runner.claude_code.subprocess.run', side_effect=run), \
                patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'pigey-claude-code-unused', 'CLAUDECODE': 'nested'}):
            backend = ClaudeCodeBackend(Path(root))
            backend('', '', [{'role': 'user', 'content': 'Select a tool.'}], 'Robot instructions.', TOOLS)
        self.assertNotIn('ANTHROPIC_API_KEY', seen['env'])
        self.assertNotIn('CLAUDECODE', seen['env'])
        self.assertEqual(seen['command'][seen['command'].index('--tools') + 1], '')
        self.assertIn('--no-session-persistence', seen['command'])
        self.assertNotIn('--bare', seen['command'])  # Bare mode ignores the subscription login.
        self.assertEqual(json.loads(seen['input'])['message']['role'], 'user')
        self.assertEqual(len(backend.calls), 1)

    @patch('runner.claude_code.shutil.which', return_value='/mock/claude')
    def test_usage_failure_is_not_retried_or_faked_as_a_robot_action(self, _which):
        with tempfile.TemporaryDirectory() as root, patch('runner.claude_code.subprocess.run') as run:
            run.return_value = subprocess.CompletedProcess([], 1, 'usage limit reached', '')
            backend = ClaudeCodeBackend(Path(root))
            with self.assertRaises(ClaudeCodeError) as caught:
                backend('', '', [{'role': 'user', 'content': 'go'}], '', TOOLS)
            self.assertEqual(run.call_count, 1)
            self.assertEqual(backend.errors[0]['kind'], 'nonzero_exit')
            self.assertNotIn('usage limit', str(caught.exception).lower())

    def test_runner_marks_backend_failure_and_restores_auth_environment(self):
        observed = {}
        backend = SimpleNamespace(errors=[{'kind': 'nonzero_exit', 'detail': 'usage exhausted'}], calls=[])

        class TrialLog:
            def __init__(self):
                self.data = {}
            def finish(self, success, *unused):
                observed.update(self.data, success=success)

        module = SimpleNamespace(TrialLog=TrialLog)
        def main():
            module.TrialLog().finish(False, 0, 0, 0)
            self.assertIs(module.call_claude, backend)
            return 0  # Upstream returns 0 even when the LLM call failed.
        module.main = main
        with patch.dict('os.environ', {'SLURM_JOB_ID': 'test', 'GEMINI_API_KEY': 'test',
                                      'ANTHROPIC_API_KEY': 'existing-value'}), \
                patch('runner.run_claude.ClaudeCodeBackend', return_value=backend), \
                patch('runner.run_claude.load_harness', return_value=module):
            self.assertEqual(run_claude.main(['--suite', 'libero_goal_task', *UPSTREAM_PERCEPTION]), 1)
            import os
            self.assertEqual(os.environ['ANTHROPIC_API_KEY'], 'existing-value')
        self.assertIsNone(observed['success'])
        self.assertEqual(observed['status'], 'backend_error')

    def test_runner_preserves_normal_task_failure_as_boolean(self):
        observed = {}
        backend = SimpleNamespace(errors=[], calls=[{'model': 'claude-sonnet-4-6'}])

        class TrialLog:
            def __init__(self):
                self.data = {}
            def finish(self, success, *unused):
                observed.update(self.data, success=success)

        module = SimpleNamespace(TrialLog=TrialLog)
        module.main = lambda: module.TrialLog().finish(False, 10, 1, 1) or 0
        with patch.dict('os.environ', {'SLURM_JOB_ID': 'test', 'GEMINI_API_KEY': 'test'}), \
                patch('runner.run_claude.ClaudeCodeBackend', return_value=backend), \
                patch('runner.run_claude.load_harness', return_value=module):
            self.assertEqual(run_claude.main(UPSTREAM_PERCEPTION), 0)
        self.assertIs(observed['success'], False)
        self.assertEqual(observed['status'], 'completed')


if __name__ == '__main__':
    unittest.main()
