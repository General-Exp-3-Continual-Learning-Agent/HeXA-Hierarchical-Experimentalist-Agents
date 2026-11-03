"""Replacements for Pigey's Gemini perception calls.

- ER detection: Gemini Robotics-ER is replaced by an open-weight VLM served
  locally with vLLM's OpenAI-compatible API (default Qwen3-VL-8B-Instruct).
  Requests are routed at the LiteLLM call so that Pigey's own parsing,
  bbox -> world unprojection, filtering, and caching stay unchanged.
- VerifyCandidate: Gemini Flash-Lite is replaced by a Claude Code CLI call.
"""
from __future__ import annotations

import base64
import json
import re
import time
from types import SimpleNamespace

# Pigey's ER model name is read from GEMINI_ER_MODEL; this value marks requests
# that must go to the local server instead of a LiteLLM provider.
LOCAL_ER_MODEL = "pigey-local-er"

# Same labelling rules as Pigey's Gemini ER prompt, but in Qwen3-VL's native
# grounding format: bbox_2d = [x1, y1, x2, y2], each normalized to 0-1000.
ER_PROMPT = (
    "This is a tabletop manipulation scene. Detect every distinct manipulable "
    "object resting on the table/floor.\n\n"
    "Reply as a single JSON array; each entry has:\n"
    "  - 'label': a short PURELY VISUAL descriptor - describe what you SEE, "
    "do NOT guess semantic identity. Use color + shape + size, in snake_case. "
    "Examples of good labels: 'red_can_short', 'red_bottle_tall', "
    "'orange_carton', 'white_box_flat', 'wicker_basket', 'brown_bottle_small'. "
    "If two objects look visually similar, distinguish them with an _a / _b suffix "
    "(e.g., 'red_can_a', 'red_can_b'). Do NOT name them by what they might be "
    "semantically ('tomato_sauce', 'ketchup', etc.) - that is the agent's job.\n"
    "  - 'bbox_2d': [x1, y1, x2, y2] for the object.\n\n"
    "Be conservative: skip the table, the robot arm, the background. Only include "
    "actual manipulable objects. Output ONLY the JSON array."
)

VERIFY_TOOLS = [{"name": "Answer", "description": "Report whether the image shows the object.",
                 "input_schema": {"type": "object", "properties": {
                     "verdict": {"type": "string", "enum": ["YES", "NO", "UNSURE"]}},
                     "required": ["verdict"], "additionalProperties": False}}]
VERIFY_SYSTEM = ("You verify object identity for a robot. Look at the image and answer the question "
                 "with exactly one Answer call whose verdict is YES, NO, or UNSURE.")


def _boxes(text: str) -> list:
    # Reasoning models (e.g. Cosmos-Reason) may think first or wrap the answer.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    answer = re.search(r"<answer>(.*?)(?:</answer>|$)", text, flags=re.S)
    text = answer.group(1) if answer else text
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        items = json.loads(text)
        if isinstance(items, dict):
            items = [items]
        if isinstance(items, list):
            return items
    except json.JSONDecodeError:
        pass
    # Salvage complete entries from truncated or slightly malformed output.
    items = []
    for match in re.finditer(r"\{[^{}]*\}", text):
        try:
            items.append(json.loads(match.group(0)))
        except json.JSONDecodeError:
            continue
    return items


def to_gemini_detections(text: str) -> str:
    """Convert Qwen [x1, y1, x2, y2] boxes to Gemini's [ymin, xmin, ymax, xmax]."""
    detections = []
    for item in _boxes(text):
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "")).strip()
        box = item.get("bbox_2d", item.get("bbox"))
        if not label or not isinstance(box, list) or len(box) != 4:
            continue
        try:
            x1, y1, x2, y2 = (max(0, min(1000, int(round(float(v))))) for v in box)
        except (TypeError, ValueError):
            continue
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        detections.append({"label": label, "bbox": [y1, x1, y2, x2]})
    return json.dumps(detections)


class LocalER:
    """Pigey ER detection through a local OpenAI-compatible VLM server."""

    def __init__(self, base_url: str, model: str, timeout: int = 120, sampling: str = "greedy"):
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key="EMPTY", timeout=timeout, max_retries=1)
        self.base_url, self.model = base_url, model
        # greedy: temperature 0. default: no sampling arguments, so the server
        # applies the model's shipped generation_config, as upstream leaves
        # Gemini at its default sampling.
        self.sampling = sampling
        self.last_raw = ""
        self.calls = []
        self.errors = []

    def complete(self, messages: list):
        started = time.monotonic()
        try:
            image = next(block for block in messages[0]["content"] if block.get("type") == "image_url")
            sampling = {"temperature": 0} if self.sampling == "greedy" else {}
            response = self.client.chat.completions.create(
                model=self.model, max_tokens=4096, **sampling,
                messages=[{"role": "user", "content": [image, {"type": "text", "text": ER_PROMPT}]}])
            raw = self.last_raw = response.choices[0].message.content or ""
        except Exception as exc:
            # Pigey treats a failed ER call as "no detections"; record it so the
            # trial is marked as a backend error instead of a task failure.
            self.errors.append({"kind": "local_er", "detail": f"{type(exc).__name__}: {exc}"[:3000]})
            raise
        text = to_gemini_detections(raw)
        usage = getattr(response, "usage", None)
        self.calls.append({"model": self.model, "sampling": self.sampling, "detections": len(json.loads(text)),
                           "truncated": response.choices[0].finish_reason == "length",
                           "input_tokens": getattr(usage, "prompt_tokens", None),
                           "output_tokens": getattr(usage, "completion_tokens", None),
                           "duration_s": round(time.monotonic() - started, 3)})
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


# Claude as the ER detector: same labelling rules, pixel boxes returned through
# the Claude Code structured-output schema (so no free-form JSON to loop on).
CLAUDE_ER_SYSTEM = (
    "You are an object detector for a robot. Look at the camera image and report every distinct "
    "manipulable object resting on the table/floor, each exactly once, as a single Detections entry in tool_calls.\n"
    "- label: a short PURELY VISUAL descriptor - describe what you SEE, do NOT guess semantic identity. "
    "Use color + shape + size, in snake_case (e.g. 'red_can_short', 'white_box_flat', 'brown_bottle_small'). "
    "If two objects look alike, give each a different visual descriptor. Do NOT name them by what they "
    "might be semantically ('tomato_sauce', 'ketchup', etc.).\n"
    "- bbox: [x1, y1, x2, y2] tight pixel box in this {width}x{height} image (x right, y down).\n"
    "Be conservative: skip the table, the robot arm, the background. Only include actual manipulable objects.")
CLAUDE_ER_TOOLS = [{"name": "Detections", "description": "Report the detected objects.",
                    "input_schema": {"type": "object", "properties": {"objects": {
                        "type": "array", "maxItems": 30, "items": {
                            "type": "object", "properties": {
                                "label": {"type": "string"},
                                "bbox": {"type": "array", "items": {"type": "integer"}, "minItems": 4, "maxItems": 4}},
                            "required": ["label", "bbox"], "additionalProperties": False}}},
                        "required": ["objects"], "additionalProperties": False}}]


class ClaudeER:
    """Pigey ER detection through the Claude Code CLI (drop-in for LocalER)."""

    def __init__(self, backend):
        self.backend, self.model = backend, backend.model
        self.sampling = "claude"
        self.last_raw = ""
        self.calls = []
        self.errors = []

    def complete(self, messages: list):
        import io
        from PIL import Image
        started = time.monotonic()
        url = next(block for block in messages[0]["content"] if block.get("type") == "image_url")["image_url"]["url"]
        data = url.split(",", 1)[1]
        width, height = Image.open(io.BytesIO(base64.b64decode(data))).size
        image = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}}
        try:
            response = self.backend("", self.model, [{"role": "user", "content": [
                image, {"type": "text", "text": "Detect the objects in this image."}]}],
                CLAUDE_ER_SYSTEM.format(width=width, height=height), CLAUDE_ER_TOOLS)
        except Exception as exc:
            self.errors.append({"kind": "claude_er", "detail": f"{type(exc).__name__}: {exc}"[:3000]})
            raise
        calls = [block for block in response["content"] if block["type"] == "tool_use"]
        objects = calls[0]["input"]["objects"] if calls else []
        self.last_raw = json.dumps(objects)
        # Pixel boxes -> Qwen-style 0-1000 bbox_2d, then the shared conversion.
        scaled = [{"label": o["label"], "bbox_2d": [o["bbox"][0] * 1000 / width, o["bbox"][1] * 1000 / height,
                                                     o["bbox"][2] * 1000 / width, o["bbox"][3] * 1000 / height]}
                  for o in objects]
        text = to_gemini_detections(json.dumps(scaled))
        cli = self.backend.calls[-1] if self.backend.calls else {}
        self.calls.append({"model": self.model, "sampling": self.sampling, "detections": len(json.loads(text)),
                           "truncated": False, "output_tokens": cli.get("usage", {}).get("output_tokens"),
                           "cost_usd": cli.get("total_cost_usd"), "duration_s": round(time.monotonic() - started, 3)})
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


def route_local_er(litellm, local_er: LocalER):
    """Send LiteLLM requests for LOCAL_ER_MODEL to the local server."""
    original = litellm.completion

    def completion(*args, **kwargs):
        if kwargs.get("model") == LOCAL_ER_MODEL:
            return local_er.complete(kwargs["messages"])
        return original(*args, **kwargs)

    litellm.completion = completion


class SamClient:
    """Drop-in for spatial_tools._sam_get_mask that calls sam_server.py."""

    def __init__(self, url: str, timeout: int = 60):
        self.url, self.timeout = url.rstrip("/"), timeout
        self.calls = 0
        self.errors = []

    def __call__(self, rgb_image, bbox, llm_view_rotated=True, img_h=256, img_w=256):
        import io
        import urllib.request
        import numpy as np
        from PIL import Image
        buffer = io.BytesIO()
        Image.fromarray(np.ascontiguousarray(rgb_image)).save(buffer, format="PNG")
        request = {"image": base64.b64encode(buffer.getvalue()).decode("ascii"),
                   "bbox": [int(v) for v in bbox], "llm_view_rotated": bool(llm_view_rotated),
                   "img_h": int(img_h), "img_w": int(img_w)}
        try:
            with urllib.request.urlopen(urllib.request.Request(
                    self.url + "/mask", data=json.dumps(request).encode(),
                    headers={"Content-Type": "application/json"}), timeout=self.timeout) as response:
                reply = json.loads(response.read())
        except Exception as exc:
            # Upstream falls back to box depth when SAM fails; record server
            # failures so the trial is marked as a backend error.
            self.errors.append({"kind": "sam_server", "detail": f"{type(exc).__name__}: {exc}"[:3000]})
            return None
        self.calls += 1
        if reply.get("mask") is None:
            return None
        height, width = reply["shape"]
        bits = np.frombuffer(base64.b64decode(reply["mask"]), dtype=np.uint8)
        return np.unpackbits(bits, count=height * width).reshape(height, width).astype(bool)


def route_sam(sam_client: SamClient):
    """Point Pigey's spatial tools (already on sys.path) at the SAM server."""
    import spatial_tools
    spatial_tools._sam_get_mask = sam_client


def claude_yesno(backend):
    """Build a drop-in for LiberoRunner._vlm_yesno backed by Claude Code."""

    def _vlm_yesno(self, image_png_bytes, prompt_text):
        image = {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                             "data": base64.b64encode(image_png_bytes).decode("ascii")}}
        response = backend("", backend.model, [{"role": "user", "content": [
            image, {"type": "text", "text": prompt_text + " Answer YES, NO, or UNSURE."}]}],
            VERIFY_SYSTEM, VERIFY_TOOLS)
        calls = [block for block in response["content"] if block["type"] == "tool_use"]
        return calls[0]["input"]["verdict"] if len(calls) == 1 else "UNSURE"

    return _vlm_yesno
