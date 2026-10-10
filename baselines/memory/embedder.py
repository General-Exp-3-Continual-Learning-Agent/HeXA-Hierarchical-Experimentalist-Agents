"""Text embedding backends for the memory baselines.

Defaults to `jinaai/jina-embeddings-v2-base-en` through local SentenceTransformers
inference, which is what the CORE paper used for CORE and Episodic RAG.  MemRL's
own release uses OpenAI `text-embedding-3-large`; pass
`--embedding-model openai:text-embedding-3-large` (needs `OPENAI_API_KEY`) to
match that instead.

This mirrors `core-reasoning/src/model/embedder.py` but keeps the dependency
surface local to `baselines/` and adds an on-disk cache, since intents here are
deterministic functions of (level, seed) and get re-embedded across runs.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Iterable, Optional, Sequence

DEFAULT_MODEL = "jinaai/jina-embeddings-v2-base-en"
OPENAI_PREFIXES = ("text-embedding-",)


def _shim_transformers_onnx() -> None:
    """Stub `transformers.onnx.OnnxConfig` on transformers>=5, which dropped it.

    Jina's own remote code (`jinaai/jina-bert-implementation`'s
    `configuration_bert.py`) does `from transformers.onnx import OnnxConfig` to
    define an unused `JinaBertOnnxConfig` subclass -- dead code on the plain
    encode() path this module uses, but the import itself raises
    `ModuleNotFoundError` on transformers>=5, which removed the whole `onnx`
    submodule. `hexa_core/CORE_COMPARISON.md` records pinning `transformers<5`
    as the fix for the same failure in the CORE-rerun environment; pinning here
    would require depending on a shared conda env this project does not own, so
    this stubs the one symbol actually referenced instead of downgrading
    whatever environment calls this module. `JinaBertOnnxConfig.inputs` reads
    `self.task`, so a bare `object`-derived stub is fine as long as nothing
    ever instantiates it -- which nothing on the SentenceTransformer.encode()
    path does.
    """
    import sys

    if "transformers.onnx" in sys.modules:
        return
    try:
        import transformers.onnx  # noqa: F401

        return  # transformers<5: the real module exists, nothing to shim
    except ModuleNotFoundError:
        pass

    import types

    shim = types.ModuleType("transformers.onnx")
    shim.OnnxConfig = type("OnnxConfig", (object,), {})
    sys.modules["transformers.onnx"] = shim


def _resolve(name: str, provider: str) -> tuple[str, str]:
    """Return (provider, model_name), honouring an `openai:` name prefix."""
    if name.startswith("openai:"):
        return "openai", name.split(":", 1)[1]
    if provider != "auto":
        return provider, name
    if name.startswith(OPENAI_PREFIXES):
        return "openai", name
    return "sentence-transformers", name


class Embedder:
    """Embeds text, caching by (model, sha1(text)) in memory and on disk."""

    def __init__(
        self,
        name: str = DEFAULT_MODEL,
        *,
        provider: str = "auto",
        cache_dir: Optional[Path] = None,
        device: str = "auto",
    ):
        self.provider, self.name = _resolve(name, provider)
        self._cache: dict[str, list[float]] = {}
        self._lock = threading.Lock()
        self._backend = None
        self.device = device

        self.cache_path: Optional[Path] = None
        if cache_dir is not None:
            slug = self.name.replace("/", "__")
            self.cache_path = Path(cache_dir) / f"emb_cache_{slug}.jsonl"
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._load_disk_cache()

    # ── backends ────────────────────────────────────────────────────────
    def _ensure_backend(self):
        if self._backend is not None:
            return
        if self.provider == "openai":
            from openai import OpenAI

            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY is required for OpenAI embeddings; use the "
                    "default jina model for a fully local run."
                )
            kwargs = {"api_key": api_key}
            base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
            if base_url:
                kwargs["base_url"] = base_url
            self._backend = OpenAI(**kwargs)
        elif self.provider == "sentence-transformers":
            _shim_transformers_onnx()
            from sentence_transformers import SentenceTransformer

            self._backend = SentenceTransformer(
                self.name, trust_remote_code=True, device=self._pick_device()
            )
        else:
            raise ValueError(f"Unsupported embedding provider: {self.provider}")

    def _pick_device(self) -> str:
        """Resolve the embedding device, refusing GPUs this torch cannot drive.

        Slurm's `gpu` partition still contains Maxwell-era cards (Tesla M40,
        sm_52) that current torch builds ship no kernels for; landing on one
        raises "no kernel image is available for execution on the device" the
        first time a tensor is touched.  A 137M embedder is fast enough on CPU
        that falling back is always preferable to failing a long run.
        """
        if self.device != "auto":
            return self.device
        try:
            import torch

            if not torch.cuda.is_available():
                return "cpu"
            major, _ = torch.cuda.get_device_capability(0)
            supported = torch.cuda.get_arch_list()
            if major < 7 or not any(
                arch.startswith("sm_")
                and int(arch[3:].rstrip("a")) // 10 == major
                for arch in supported
            ):
                print(
                    f"  [embedder] {torch.cuda.get_device_name(0)} "
                    f"(sm_{major}x) is not supported by torch "
                    f"{torch.__version__} ({supported}); using CPU.",
                    flush=True,
                )
                return "cpu"
            return "cuda"
        except Exception as exc:
            print(f"  [embedder] device probe failed ({exc}); using CPU.", flush=True)
            return "cpu"

    # ── cache ───────────────────────────────────────────────────────────
    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    def _load_disk_cache(self) -> None:
        if self.cache_path is None or not self.cache_path.exists():
            return
        with self.cache_path.open() as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("key") and rec.get("vec"):
                    self._cache[rec["key"]] = rec["vec"]

    def _append_disk_cache(self, key: str, vec: Sequence[float]) -> None:
        if self.cache_path is None:
            return
        with self._lock:
            with self.cache_path.open("a") as fh:
                fh.write(json.dumps({"key": key, "vec": list(vec)}) + "\n")

    # ── public API ──────────────────────────────────────────────────────
    def embed(self, texts: Iterable[str]) -> list[list[float]]:
        texts = list(texts)
        missing = [t for t in texts if self._key(t) not in self._cache]
        if missing:
            self._ensure_backend()
            uniq = list(dict.fromkeys(missing))
            if self.provider == "openai":
                resp = self._backend.embeddings.create(model=self.name, input=uniq)
                vecs = [d.embedding for d in resp.data]
            else:
                vecs = [
                    list(map(float, v))
                    for v in self._backend.encode(
                        uniq, normalize_embeddings=False, show_progress_bar=False
                    )
                ]
            for text, vec in zip(uniq, vecs):
                key = self._key(text)
                self._cache[key] = list(vec)
                self._append_disk_cache(key, vec)
        return [self._cache[self._key(t)] for t in texts]

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]
