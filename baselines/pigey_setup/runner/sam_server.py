"""Serve Pigey's own SAM masking function from a CUDA-enabled environment.

The simulator venv's CPU Torch 1.11 cannot run Pigey's SAM path. This server
runs in the vLLM venv (CUDA Torch, current transformers), imports the pinned
upstream spatial_tools, and calls its unmodified _sam_get_mask.
"""
import argparse
import base64
from http.server import BaseHTTPRequestHandler, HTTPServer
import io
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image


def decode_image(data: str) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB"))


def encode_mask(mask) -> dict:
    if mask is None:
        return {"mask": None}
    mask = np.asarray(mask, dtype=bool)
    return {"mask": base64.b64encode(np.packbits(mask).tobytes()).decode("ascii"), "shape": list(mask.shape)}


def make_handler(sam_get_mask):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self._reply(200 if self.path == "/health" else 404, {"ok": self.path == "/health"})

        def do_POST(self):
            if self.path != "/mask":
                self._reply(404, {"error": "unknown path"})
                return
            try:
                request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                mask = sam_get_mask(decode_image(request["image"]), request["bbox"],
                                    llm_view_rotated=request["llm_view_rotated"],
                                    img_h=request["img_h"], img_w=request["img_w"])
            except Exception as exc:
                self._reply(400, {"error": f"{type(exc).__name__}: {exc}"})
                return
            self._reply(200, encode_mask(mask))

        def _reply(self, status, body):
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--pigey-repo", default=os.environ.get("PIGEY_REPO"))
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.pigey_repo) / "sim"))
    import spatial_tools
    # Load the model before reporting healthy; upstream returns None on failure.
    blank = np.zeros((64, 64, 3), dtype=np.uint8)
    if spatial_tools._sam_get_mask(blank, (8, 8, 40, 40), img_h=64, img_w=64) is None:
        raise SystemExit("SAM failed to load or run on CUDA")
    print("SAM ready", flush=True)
    # Single-threaded: one GPU model, requests handled in order.
    HTTPServer(("127.0.0.1", args.port), make_handler(spatial_tools._sam_get_mask)).serve_forever()


if __name__ == "__main__":
    main()
