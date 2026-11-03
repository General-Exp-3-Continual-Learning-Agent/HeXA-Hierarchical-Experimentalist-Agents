"""Offline checks for the perception replacements; no servers or model requests."""
from http.server import HTTPServer
import json
import os
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # baselines/pigey_setup

from runner.claude_code import ClaudeCodeError  # noqa: E402
from runner.perception import (ClaudeER, LOCAL_ER_MODEL, VERIFY_TOOLS, LocalER, SamClient, claude_yesno, route_local_er,
                        to_gemini_detections)
from runner import run_claude  # noqa: E402
from runner import sam_server  # noqa: E402


class DetectionFormatTests(unittest.TestCase):
    def test_qwen_boxes_become_gemini_order(self):
        text = '```json\n[{"label": "red_can_a", "bbox_2d": [100, 200, 300, 400]}]\n```'
        self.assertEqual(json.loads(to_gemini_detections(text)),
                         [{"label": "red_can_a", "bbox": [200, 100, 400, 300]}])

    def test_boxes_are_clamped_and_ordered(self):
        text = '[{"label": "bowl", "bbox_2d": [900, 50, -10, 1200.4]}]'
        self.assertEqual(json.loads(to_gemini_detections(text)), [{"label": "bowl", "bbox": [50, 0, 1000, 900]}])

    def test_truncated_output_is_salvaged_and_bad_entries_dropped(self):
        text = ('[{"label": "plate", "bbox_2d": [1, 2, 3, 4]}, {"label": "", "bbox_2d": [1, 2, 3, 4]}, '
                '{"label": "cup", "bbox_2d": [1, 2]}, {"label": "box", "bbox_2d": [5, 6, 7')
        self.assertEqual(json.loads(to_gemini_detections(text)), [{"label": "plate", "bbox": [2, 1, 4, 3]}])
        self.assertEqual(to_gemini_detections("no objects"), "[]")

    def test_reasoning_and_answer_tags_are_removed(self):
        text = '<think>\nA plate {maybe} near the bowl.\n</think>\n\n<answer>\n[{"label": "plate", "bbox_2d": [1, 2, 3, 4]}]\n</answer>'
        self.assertEqual(json.loads(to_gemini_detections(text)), [{"label": "plate", "bbox": [2, 1, 4, 3]}])


class LocalERTests(unittest.TestCase):
    def setUp(self):
        self.er = LocalER("http://127.0.0.1:1/v1", "Qwen/Qwen3-VL-8B-Instruct")
        self.er.client = MagicMock()
        self.messages = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            {"type": "text", "text": "Gemini prompt"}]}]

    def test_only_local_er_requests_are_rerouted(self):
        self.er.client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='[{"label": "mug", "bbox_2d": [1, 2, 3, 4]}]'),
                                     finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=300, completion_tokens=20))
        original = MagicMock(return_value="provider")
        litellm = SimpleNamespace(completion=original)
        route_local_er(litellm, self.er)
        self.assertEqual(litellm.completion(model="gemini/gemini-3.1-flash-lite", messages=[]), "provider")
        response = litellm.completion(model=LOCAL_ER_MODEL, api_key="sentinel", max_tokens=8000, messages=self.messages)
        self.assertEqual(json.loads(response.choices[0].message.content), [{"label": "mug", "bbox": [2, 1, 4, 3]}])
        request = self.er.client.chat.completions.create.call_args.kwargs
        self.assertEqual(request["model"], "Qwen/Qwen3-VL-8B-Instruct")
        self.assertEqual(request["messages"][0]["content"][0], self.messages[0]["content"][0])
        self.assertNotIn("sentinel", json.dumps(request))
        self.assertEqual((self.er.calls[0]["detections"], self.er.calls[0]["input_tokens"]), (1, 300))
        self.assertEqual((request["temperature"], self.er.calls[0]["truncated"]), (0, False))
        self.assertIn("bbox_2d", self.er.last_raw)
        original.assert_called_once()

    def test_default_sampling_leaves_generation_config_to_the_server(self):
        self.er.client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='[{"label": "mug"'), finish_reason="length")],
            usage=SimpleNamespace(prompt_tokens=300, completion_tokens=4096))
        self.er.sampling = "default"
        self.er.complete(self.messages)
        request = self.er.client.chat.completions.create.call_args.kwargs
        self.assertFalse({"temperature", "top_p", "top_k"} & set(request))
        self.assertEqual((self.er.calls[0]["sampling"], self.er.calls[0]["truncated"]), ("default", True))

    def test_server_errors_are_recorded(self):
        self.er.client.chat.completions.create.side_effect = ConnectionError("refused")
        with self.assertRaises(ConnectionError):
            self.er.complete(self.messages)
        self.assertEqual(self.er.errors[0]["kind"], "local_er")


class ClaudeERTests(unittest.TestCase):
    def test_pixel_boxes_become_gemini_order(self):
        import base64, io
        from PIL import Image
        buffer = io.BytesIO()
        Image.new("RGB", (500, 250)).save(buffer, format="PNG")
        url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
        backend = MagicMock(model="claude-sonnet-5-5", calls=[{"total_cost_usd": 0.01}])
        backend.return_value = {"content": [{"type": "tool_use", "name": "Detections", "input": {
            "objects": [{"label": "plate", "bbox": [50, 25, 100, 50]}]}}]}
        er = ClaudeER(backend)
        response = er.complete([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}]}])
        self.assertEqual(json.loads(response.choices[0].message.content), [{"label": "plate", "bbox": [100, 100, 200, 200]}])
        system, tools = backend.call_args.args[3], backend.call_args.args[4]
        self.assertIn("500x250", system)
        self.assertEqual(tools[0]["input_schema"]["properties"]["objects"]["maxItems"], 30)
        self.assertEqual((er.calls[0]["detections"], er.calls[0]["cost_usd"]), (1, 0.01))


class VerifierTests(unittest.TestCase):
    def test_claude_verdict_and_native_image(self):
        backend = MagicMock(model="claude-haiku-4-5", return_value={"content": [
            {"type": "text", "text": "A red can."},
            {"type": "tool_use", "id": "x", "name": "Answer", "input": {"verdict": "NO"}}]})
        self.assertEqual(claude_yesno(backend)(None, b"png", "Is it a bowl?"), "NO")
        _, _, messages, _, tools = backend.call_args.args
        image, text = messages[0]["content"]
        self.assertEqual((image["type"], image["source"]["data"]), ("image", "cG5n"))
        self.assertIn("Is it a bowl?", text["text"])
        self.assertEqual(tools, VERIFY_TOOLS)

    def test_missing_answer_is_unsure_and_errors_propagate(self):
        backend = MagicMock(model="haiku", return_value={"content": []})
        self.assertEqual(claude_yesno(backend)(None, b"png", "?"), "UNSURE")
        backend.side_effect = ClaudeCodeError("deadline_exceeded", "timeout")
        with self.assertRaises(ClaudeCodeError):
            claude_yesno(backend)(None, b"png", "?")


class SamServiceTests(unittest.TestCase):
    def test_mask_round_trip_preserves_upstream_arguments(self):
        import numpy as np
        seen = {}

        def fake_sam(rgb, bbox, llm_view_rotated=True, img_h=256, img_w=256):
            seen.update(rgb=rgb, bbox=bbox, rotated=llm_view_rotated, size=(img_h, img_w))
            mask = np.zeros(rgb.shape[:2], dtype=bool)
            mask[2:5, 3:7] = True
            return mask

        server = HTTPServer(("127.0.0.1", 0), sam_server.make_handler(fake_sam))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            client = SamClient(f"http://127.0.0.1:{server.server_port}")
            rgb = np.arange(9 * 11 * 3, dtype=np.uint8).reshape(9, 11, 3)[::-1, ::-1]
            mask = client(rgb, (1.0, 2, 8, 6), llm_view_rotated=False, img_h=9, img_w=11)
        finally:
            server.shutdown()
            server.server_close()
        self.assertEqual(mask.shape, (9, 11))
        self.assertEqual(int(mask.sum()), 12)
        self.assertTrue(mask[2:5, 3:7].all())
        np.testing.assert_array_equal(seen["rgb"], rgb)
        self.assertEqual((seen["bbox"], seen["rotated"], seen["size"]), ([1, 2, 8, 6], False, (9, 11)))
        self.assertEqual((client.calls, client.errors), (1, []))

    def test_unreachable_server_falls_back_and_records_error(self):
        import numpy as np
        client = SamClient("http://127.0.0.1:1", timeout=2)
        self.assertIsNone(client(np.zeros((4, 4, 3), dtype=np.uint8), (0, 0, 2, 2)))
        self.assertEqual(client.errors[0]["kind"], "sam_server")


class RunnerTests(unittest.TestCase):
    def test_default_run_needs_no_gemini_key_and_reports_er_errors(self):
        observed = {}
        backend = SimpleNamespace(errors=[], calls=[], model="claude-sonnet-4-6")
        local_er = SimpleNamespace(errors=[], calls=[], model="Qwen/Qwen3-VL-8B-Instruct")

        class TrialLog:
            def __init__(self):
                self.data = {}

            def finish(self, success, *unused):
                observed.update(self.data, success=success)

        class LiberoRunner:
            pass

        module = SimpleNamespace(TrialLog=TrialLog, LiberoRunner=LiberoRunner)

        def main():
            observed["env"] = {key: os.environ.get(key) for key in ("GEMINI_ER_MODEL", "GEMINI_API_KEY", "CORTEX_NO_PHASE0", "HARNESS_PERCEPTION")}
            local_er.errors.append({"kind": "local_er", "detail": "refused"})
            module.TrialLog().finish(True, 0, 0, 0)
            return 0

        module.main = main
        env = {"SLURM_JOB_ID": "test", "PIGEY_ER_URL": "http://127.0.0.1:1/v1", "PIGEY_SAM_URL": "http://127.0.0.1:2"}
        with patch.dict("os.environ", env, clear=True), \
                patch.dict("sys.modules", {"litellm": SimpleNamespace()}), \
                patch("runner.run_claude.ClaudeCodeBackend", return_value=backend), \
                patch("runner.run_claude.LocalER", return_value=local_er), \
                patch("runner.run_claude.route_local_er") as route, \
                patch("runner.run_claude.route_sam") as route_sam, \
                patch("runner.run_claude.load_harness", return_value=module):
            self.assertEqual(run_claude.main([]), 1)
            self.assertNotIn("GEMINI_ER_MODEL", os.environ)
            self.assertNotIn("GEMINI_API_KEY", os.environ)
            self.assertNotIn("CORTEX_NO_PHASE0", os.environ)
        route.assert_called_once()
        route_sam.assert_called_once()
        self.assertEqual(observed["sam"], "service")
        self.assertTrue(callable(LiberoRunner._vlm_yesno))
        self.assertEqual(observed["env"]["GEMINI_ER_MODEL"], LOCAL_ER_MODEL)
        self.assertEqual((observed["env"]["CORTEX_NO_PHASE0"], observed["phase0"]), ("1", "skip"))
        self.assertEqual(observed["env"]["HARNESS_PERCEPTION"], "gemini_er")
        self.assertEqual((observed["status"], observed["success"]), ("backend_error", None))
        self.assertEqual((observed["perception"], observed["verifier"]), ("local", "claude"))
        self.assertEqual(observed["er_model"], "Qwen/Qwen3-VL-8B-Instruct")

    def test_local_servers_need_explicit_urls(self):
        for argv in ([], ["--perception", "gemini", "--verifier", "gemini"]):
            with patch.dict("os.environ", {"SLURM_JOB_ID": "test", "GEMINI_API_KEY": "test"}, clear=True), \
                    patch("runner.run_claude.ClaudeCodeBackend"), self.assertRaises(SystemExit):
                run_claude.main(argv)

    def test_gemini_choices_still_require_a_key(self):
        with patch.dict("os.environ", {"SLURM_JOB_ID": "test"}, clear=True), \
                patch("runner.run_claude.ClaudeCodeBackend"), self.assertRaises(SystemExit):
            run_claude.main(["--perception", "gemini"])


if __name__ == "__main__":
    unittest.main()
