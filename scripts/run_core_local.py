#!/usr/bin/env python
"""Run CORE's own ``scripts/run_core.py`` against our local vLLM server.

Why this file exists
--------------------
``core-reasoning`` is read-only for this project, and its LLM client hardcodes
NVIDIA's hosted endpoint::

    # src/model/nvidia_client.py:24
    self.base_url = "https://integrate.api.nvidia.com/v1"

There is no environment variable, no constructor argument, and no ``--base-url``
flag to move it. So the only way to run CORE on the same weights, the same
server and the same generation settings as HeXA -- which is what makes a
head-to-head comparison mean anything -- is to rebind that one attribute from
the outside. Everything else about ``NvidiaClient`` is already what we want:

* it is a plain ``openai.OpenAI`` client, so an OpenAI-compatible vLLM server is
  a drop-in;
* it already sets ``reasoning_effort="high"`` for any model whose name contains
  ``gpt-oss-120b`` (``nvidia_client.py:102``);
* it already builds ``raw_text = reasoning_text + final_text`` with no separator
  (``nvidia_client.py:142``), which is the concatenation our own client was
  written to reproduce.

So the patch is a single attribute, applied in ``__init__``, and nothing about
CORE's algorithm, prompts, accounting or verifiers is touched.

Safety
------
Two guards, because pointing this at the wrong host would either leak prompts to
a third party or silently produce numbers from a different provider than the one
we claim in the paper:

1. The base URL must resolve to this machine (localhost, 127.x, or a bare
   hostname with no dots -- SLURM compute nodes look like ``gpu013``). Anything
   that looks like a public endpoint is refused outright.
2. ``NVIDIA_API_KEY`` is *overwritten* with a placeholder rather than merely
   defaulted. If a real key happens to be in the environment we do not want it
   travelling anywhere, and vLLM ignores the value.

Output
------
``run_core.py`` does ``os.chdir(ROOT)`` at import and writes to
``runs/<task>/<model>/contrastive/<timestamp>/`` relative to that. ``runs/`` is
listed in CORE's own ``.gitignore``, so this leaves ``git status`` clean, but the
artifacts still land inside their tree. ``--copy-to`` copies the finished run
directory into ours so results live with the rest of the project.

Usage
-----
    python scripts/run_core_local.py \
        --base-url http://gpu013:8000/v1 \
        --copy-to results/core_matched_v1/matchstick/run0 \
        -- \
        --task matchstick --num-train-items 60 --max-rollouts 600 ...

Everything after ``--`` is passed through to ``run_core.py`` verbatim.
"""

from __future__ import annotations

import argparse
import os
import runpy
import shutil
import sys
from pathlib import Path
from urllib.parse import urlparse


def _assert_local(base_url: str) -> None:
    host = (urlparse(base_url).hostname or "").strip()
    if not host:
        raise SystemExit(f"--base-url has no host: {base_url!r}")
    local = (
        host in {"localhost", "0.0.0.0"}
        or host.startswith("127.")
        or "." not in host  # bare SLURM node name, e.g. gpu013
    )
    if not local:
        raise SystemExit(
            f"Refusing to run: --base-url host {host!r} is not local.\n"
            f"This project runs on a local vLLM server only; a hosted endpoint "
            f"would both change the provider and send prompts off-machine."
        )


def main() -> int:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--base-url", required=True,
                    help="OpenAI-compatible endpoint of our vLLM server, "
                         "e.g. http://gpu013:8000/v1")
    ap.add_argument("--copy-to", default=None,
                    help="Directory to copy the finished CORE run into.")
    ap.add_argument("--core-repo", default=None,
                    help="Path to core-reasoning (default: $CORE_REPO, then "
                         "../core-reasoning relative to this repo).")
    ap.add_argument("-h", "--help", action="store_true")
    args, passthrough = ap.parse_known_args()

    if args.help:
        print(__doc__)
        return 0
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    _assert_local(args.base_url)

    here = Path(__file__).resolve().parents[1]
    core_root = Path(
        args.core_repo
        or os.environ.get("CORE_REPO")
        or here.parent / "core-reasoning"
    ).resolve()
    run_core = core_root / "scripts" / "run_core.py"
    if not run_core.is_file():
        raise SystemExit(f"CORE runner not found at {run_core}")

    # The embedder needs a writable HF cache; the cluster default is read-only.
    os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    os.environ.setdefault("HF_HUB_CACHE", os.environ["HF_HOME"] + "/hub")
    os.environ.pop("TRANSFORMERS_CACHE", None)
    os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)

    # vLLM ignores the key, but do not let a real one sit in the environment.
    os.environ["NVIDIA_API_KEY"] = "local-vllm-no-key-required"

    sys.path.insert(0, str(core_root / "src"))
    from model.nvidia_client import NvidiaClient  # noqa: PLC0415

    base_url = args.base_url
    original_init = NvidiaClient.__init__

    def patched_init(self, model: str, api_key: str) -> None:
        original_init(self, model=model, api_key=api_key)
        self.base_url = base_url

    NvidiaClient.__init__ = patched_init  # type: ignore[method-assign]
    print(f"[run_core_local] NvidiaClient.base_url -> {base_url}", flush=True)

    runs_root = core_root / "runs"
    before = set(runs_root.rglob("*/contrastive/*")) if runs_root.exists() else set()

    sys.argv = [str(run_core), *passthrough]
    try:
        runpy.run_path(str(run_core), run_name="__main__")
    except SystemExit as exc:
        if exc.code not in (0, None):
            return int(exc.code)

    after = set(runs_root.rglob("*/contrastive/*")) if runs_root.exists() else set()
    created = sorted(after - before)
    if not created:
        print("[run_core_local] WARNING: no new run directory found under "
              f"{runs_root}; nothing to copy.", flush=True)
        return 0
    produced = created[-1]
    print(f"[run_core_local] CORE run directory: {produced}", flush=True)

    if args.copy_to:
        dest = Path(args.copy_to).resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(produced, dest)
        print(f"[run_core_local] copied to: {dest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
