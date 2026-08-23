#!/usr/bin/env python3
"""Phase 2 driver: the no-learning reproduction gate against CORE's published numbers.

Runs the empty-guidelines baseline (one call per problem, no memory, no retrieval)
over each task's 100 held-out problems, ``--runs`` times, and prints measured
versus published accuracy with the gap in SEM units.

Because we serve ``gpt-oss-120b`` on local vLLM and the published numbers come
from NVIDIA NIM, **this is a calibration run, not a reproduction**: the paper's
values are reference points and are not directly citable from it. Verdicts are
printed as ``PASS*``/``MISS*`` and the ``*`` is explained in the footnote.

**Which interpreter.** This driver needs ``openai``, which the system python does
not have. Run it from ``~/.conda/envs/phyreagent`` (openai 2.16.0) or from the
serving env ``/work/pi_sniekum_umass_edu/$USER/.conda/envs/vllm-gptoss`` (openai
2.53.0 *and* tiktoken, so added-token figures there are exact rather than
heuristic). The probe fails with a one-line message rather than a traceback if
``openai`` is missing.

Bring a server up first (see ``scripts/serve_gpt_oss_vllm.sh``), then::

    # cheap wiring check: 1 task, 20 problems, 1 run -> 20 calls
    python scripts/run_repro_gate.py --smoke 20 --runs 1 --tasks tower-of-hanoi \
        --base-url http://<node>:8000/v1

    # the real gate: 3 tasks x 3 runs x 100 problems = 900 calls
    python scripts/run_repro_gate.py --runs 3 \
        --base-url http://<node>:8000/v1

Per-problem results are cached, so an interrupted gate resumes for free -- rerun
the identical command.

Exit codes
    0  every task PASS*
    1  at least one MISS*
    2  probe / setup / import failure (nothing was measured)
    3  at least one task ended ERROR (failed API calls; not quotable either way)
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import sys
import textwrap
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

EXIT_OK = 0
EXIT_MISS = 1
EXIT_SETUP = 2
EXIT_ERROR = 3

DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
#: CORE passes this string as ``--nvidia-model``; the substring ``gpt-oss-120b`` is
#: also what makes the client send ``reasoning_effort="high"``
#: (``src/model/nvidia_client.py:102``), so renaming the served model silently
#: changes the generation config.
DEFAULT_MODEL = "openai/gpt-oss-120b"
#: CORE's own ``--seed``. It reaches only Tower of Hanoi's state-space sampler; the
#: other two tasks have file-backed or hard-coded splits. Not a CLI flag: changing
#: it would change which 100 Hanoi states are the held-out set.
TASK_SEED = 42

#: Written by ``scripts/serve_gpt_oss_vllm.sh``. It is the only channel through which
#: the client learns the server-side facts a reviewer needs to reconstruct the run
#: (vllm version, TP, precision, max_model_len, weights snapshot, and above all
#: ``VLLM_SYSTEM_START_DATE`` -- gpt-oss's harmony system prompt embeds a date, so an
#: unpinned server is a different experiment every calendar day). The driver usually
#: runs on a different node from the server, so none of it is discoverable locally.
DEFAULT_ENDPOINT_JSON = "logs/vllm_endpoint_latest.json"

SERVE_HINT = (
    "Bring a server up with:\n"
    "    sbatch scripts/serve_gpt_oss_vllm.sh          # 4x A100, TP=4 (MXFP4/MARLIN)\n"
    "    H100=1 bash scripts/serve_gpt_oss_vllm.sh     # 2x H100, TP=2 (native MXFP4)\n"
    "then read the endpoint out of logs/vllm_endpoint_latest.json and pass it as\n"
    "--base-url http://<node>:<port>/v1"
)



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_repro_gate.py",
        description="Phase 2 no-learning reproduction gate (CALIBRATION: local vLLM provider).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=SERVE_HINT,
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        metavar="TASK",
        help="tasks to gate (default: all three).",
    )
    parser.add_argument("--runs", type=int, default=3, help="independent eval passes (default: 3).")
    parser.add_argument(
        "--smoke",
        type=int,
        default=None,
        metavar="N",
        help="use only the first N eval problems, for cheap wiring checks. Smoke "
        "results live in their own directory and are never reused by the full gate.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help=f"OpenAI-compatible endpoint ($HEXA_VLLM_BASE_URL, else {DEFAULT_BASE_URL}).",
    )
    parser.add_argument(
        "--model", default=None, help=f"served model name (default {DEFAULT_MODEL})."
    )
    parser.add_argument("--api-key", default=None, help="ignored by vLLM, sent if given.")
    parser.add_argument(
        "--output-dir",
        default="results/repro_gate",
        help="results root; NOT timestamped, because resumability is the point.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="concurrent generations (default 10, matching CORE's --num-eval-batch-calls).",
    )

    tuning = parser.add_argument_group(
        "generation config",
        "CORE's published eval settings. Changing any of these invalidates the gate.",
    )
    tuning.add_argument("--temperature", type=float, default=None)
    tuning.add_argument("--top-p", type=float, default=None)
    tuning.add_argument("--max-tokens", type=int, default=None)
    tuning.add_argument("--llm-seed", type=int, default=None)
    tuning.add_argument("--reasoning-effort", default=None)
    tuning.add_argument("--timeout", type=float, default=None, help="per-request timeout, seconds.")

    other = parser.add_argument_group("other")
    other.add_argument(
        "--train-items",
        type=int,
        default=5,
        help="regime used only to construct the task. Eval ids are byte-identical "
        "across the 5/10/100 regimes, so this cannot affect the gate; 5 is the "
        "cheapest to build.",
    )
    other.add_argument(
        "--no-resume",
        action="store_true",
        help="re-issue every call instead of reusing per-problem results on disk.",
    )
    other.add_argument(
        "--sem-tolerance",
        type=float,
        default=None,
        help="how many combined SEM count as consistent with the paper (default 2.0).",
    )
    other.add_argument(
        "--strict-probe",
        action="store_true",
        help="abort if the probe cannot fully confirm reasoning_effort is honoured, "
        "not just accepted (default: a partial confirmation warns and continues).",
    )
    other.add_argument(
        "--endpoint-json",
        default=None,
        help="server-side provenance written by serve_gpt_oss_vllm.sh (default "
        f"{DEFAULT_ENDPOINT_JSON} if it exists). Recorded in every manifest; if it "
        "is missing the manifest says so explicitly instead of omitting it.",
    )
    other.add_argument("--quiet", action="store_true", help="suppress per-problem lines.")
    return parser


def _fail_setup(message: str) -> int:
    print(f"\n[setup] {message}", file=sys.stderr)
    return EXIT_SETUP


def _call_filtered(func: Any, **kwargs: Any) -> Any:
    """Call ``func`` with the non-``None`` subset of ``kwargs`` it actually accepts.

    Keeps an unset CLI flag from overriding the client's own default with ``None``
    (``--timeout`` is the one that bites: ``timeout=None`` means *no* timeout).
    """
    supplied = {k: v for k, v in kwargs.items() if v is not None}
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return func(**supplied)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
        return func(**supplied)
    return func(**{k: v for k, v in supplied.items() if k in signature.parameters})


def _server_provenance(path_arg: str | None, *, base_url: str) -> dict[str, Any]:
    """Read the serve script's endpoint JSON, or record precisely why we could not.

    Never returns ``{}``: an absent field in a manifest is indistinguishable from a
    field nobody thought to record, and the thing most likely to be missing here
    (``vllm_system_start_date``) is the one that decides whether two runs are even
    the same experiment.
    """
    explicit = path_arg is not None
    path = Path(path_arg or DEFAULT_ENDPOINT_JSON)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent.parent / path

    if not path.exists():
        message = (
            f"server provenance UNKNOWN: {path} does not exist. vllm version, "
            "tensor-parallel size, precision, weights snapshot and "
            "VLLM_SYSTEM_START_DATE are therefore unrecorded for this run."
        )
        print(f"[warn] {message}", file=sys.stderr)
        if explicit:
            print("[warn] (--endpoint-json was given explicitly)", file=sys.stderr)
        return {"source": str(path), "available": False, "reason": message}

    try:
        info = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        message = f"server provenance UNREADABLE: {path}: {exc!r}"
        print(f"[warn] {message}", file=sys.stderr)
        return {"source": str(path), "available": False, "reason": message}

    record: dict[str, Any] = {"source": str(path), "available": True, **info}

    # A stale latest-endpoint file describing a different server is worse than none:
    # it would put another job's GPUs and harmony date in this run's manifest.
    served = str(info.get("base_url") or "")
    if served and served.rstrip("/") != base_url.rstrip("/"):
        record["base_url_mismatch"] = (
            f"endpoint JSON describes {served}, but the gate is pointed at "
            f"{base_url}. The server-side fields below may belong to a different job."
        )
        print(f"[warn] {record['base_url_mismatch']}", file=sys.stderr)

    if not info.get("vllm_system_start_date"):
        record["harmony_date_unpinned"] = (
            "VLLM_SYSTEM_START_DATE is not recorded, so the server was probably "
            "started without it. gpt-oss's harmony system prompt then embeds the "
            "server's current date and changes daily; runs from different days are "
            "not the same experimental condition."
        )
        print(f"[warn] {record['harmony_date_unpinned']}", file=sys.stderr)
    else:
        print(
            f"[server] {path.name}: vllm {info.get('vllm_version')} "
            f"TP={info.get('tensor_parallel_size')} "
            f"harmony_date={info['vllm_system_start_date']}"
        )
    return record


def _fmt(value: float, width: int = 5, places: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a".rjust(width)
    return f"{value:.{places}f}"


def print_header(
    *, base_url: str, model: str, cfg: dict[str, Any], runs: int, n: int | None
) -> None:
    from hexa_core import baseline_runner as br

    bar = "=" * 78
    print(f"\n{bar}")
    print("  PHASE 2 REPRODUCTION GATE -- CALIBRATION RUN")
    print(f"  provider : vLLM @ {base_url}   (paper: NVIDIA NIM)")
    print(
        f"  model    : {model}   effort={cfg['reasoning_effort']}  "
        f"temp={cfg['temperature']}  top_p={cfg['top_p']}  max_tokens={cfg['max_tokens']}"
    )
    print(f"  eval     : {runs} run(s) x {n if n is not None else br.EVAL_ITEMS} problems per task")
    print(
        textwrap.fill(
            br.PROVIDER_CAVEAT.replace("CALIBRATION RUN: ", "This is NOT a reproduction. "),
            width=len(bar),
            initial_indent="  ",
            subsequent_indent="  ",
        )
    )
    print(bar)


def print_gate_table(results: Sequence[Any], comparisons: Sequence[Any], tolerance: float) -> None:
    print("\n" + "-" * 78)
    print(
        f"{'task':<16}{'runs':>5}{'n':>5}  {'measured':<16}{'paper (NIM)':<16}"
        f"{'delta':>8}{'d/SEM':>8}  verdict"
    )
    for result, cmp_ in zip(results, comparisons):
        target = (
            f"{cmp_.target.accuracy:.3f} +- {cmp_.target.sem:.3f}"
            if cmp_.target is not None
            else "-"
        )
        delta = "n/a" if math.isnan(cmp_.delta) else f"{cmp_.delta:+.3f}"
        dsem = "n/a" if math.isnan(cmp_.delta_in_sems) else f"{cmp_.delta_in_sems:+.2f}"
        print(
            f"{result.task_name:<16}{result.num_runs:>5}{result.n_problems:>5}  "
            f"{_fmt(cmp_.measured)} +- {_fmt(cmp_.measured_sem):<7}"
            f"{target:<16}{delta:>8}{dsem:>8}  {cmp_.verdict}"
        )

    print("-" * 78)
    print(
        f"{'task':<16}{'trunc(len)':>12}{'empty-parse':>13}{'no-reasoning':>14}"
        f"{'errors':>8}{'reused':>8}{'wall':>9}"
    )
    for result in results:
        attempts = result.attempts
        print(
            f"{result.task_name:<16}"
            f"{f'{result.truncated}/{attempts}':>12}"
            f"{f'{result.empty_parses}/{attempts}':>13}"
            f"{f'{result.no_reasoning}/{attempts}':>14}"
            f"{result.errors:>8}{result.reused:>8}"
            f"{result.wall_clock_s / 60:>8.1f}m"
        )
    print("-" * 78)

    from hexa_core.baseline_runner import VERDICT_FOOTNOTE

    print(VERDICT_FOOTNOTE.format(tol=tolerance))
    bases = sorted({c.sem_basis for c in comparisons})
    print(f"  SEM basis: {', '.join(bases) if bases else 'n/a'}.")
    print("  d/SEM divides the gap by the quadrature-combined SEM of measurement and paper.")
    print("  Both SEM bases are recorded in each task's summary.json; with 3 runs the")
    print("  across-run SEM is a 2-dof estimate, so read a marginal verdict loosely.")


def print_diagnostics(results: Sequence[Any], comparisons: Sequence[Any], probe: Any) -> None:
    """CORE's diagnostic order, annotated with the signal we already measured."""
    trunc = sum(r.truncated for r in results)
    no_reason = sum(r.no_reasoning for r in results)
    empty = sum(r.empty_parses for r in results)
    errors = sum(r.errors for r in results)
    attempts = sum(r.attempts for r in results) or 1
    cfg = results[0].config if results else {}

    print("\n" + "=" * 78)
    print("  DIAGNOSTIC CHECKLIST (fixed order -- work down it, do not skip ahead)")
    print("=" * 78)
    print(
        "1. reasoning effort not taking effect.\n"
        f"   probe reported: {json.dumps(_probe_effort(probe))}\n"
        f"   we sent reasoning_effort={cfg.get('reasoning_effort')!r} on every call.\n"
        "   Symptom: reasoning far shorter than expected. Older vLLM needs it via\n"
        "   chat_template_kwargs={'reasoning_effort': 'high'} rather than the flat field."
    )
    print(
        "2. reasoning_content missing from raw_text.\n"
        f"   measured: {no_reason}/{attempts} responses had empty reasoning "
        f"({100.0 * no_reason / attempts:.1f}%).\n"
        "   The runner already asserts raw_text == reasoning_text + final_text, so a\n"
        "   non-zero count here means the SERVER omitted the CoT. Check that vllm serve\n"
        "   was started with --reasoning-parser openai_gptoss."
    )
    print(
        "3. prompt mismatch.\n"
        "   ruled out by construction: every problem is checked for\n"
        "   build_prompt(None) == recorded baseline prompt before any call. To confirm\n"
        "   by hand:\n"
        "       PYTHONDONTWRITEBYTECODE=1 python -c \"from hexa_core.tasks import make_task; \\\n"
        "         p=make_task('zebralogic',5).eval_problems()[0]; print(p.prompt[:400])\""
    )
    print(
        "4. truncation at max_tokens.\n"
        f"   measured: {trunc}/{attempts} responses had finish_reason == 'length'; "
        f"{empty}/{attempts} parsed empty.\n"
        f"   max_tokens={cfg.get('max_tokens')}. If non-zero this is very likely the cause;\n"
        "   worst on tower-of-hanoi n=6, whose optimal solutions run to 63 moves. The\n"
        "   server's --max-model-len must exceed max_tokens PLUS the prompt."
    )
    print(
        "5. generator parameters.\n"
        f"   sent: temperature={cfg.get('temperature')} top_p={cfg.get('top_p')} "
        f"max_tokens={cfg.get('max_tokens')} llm_seed={cfg.get('llm_seed')} "
        f"seed_offset={cfg.get('core_eval_seed_offset')}\n"
        "   compare against core-reasoning/shell_scripts/sample_efficiency/*/run_core.sh"
    )
    if errors:
        print(
            f"\n!! {errors} generation(s) failed outright. An errored task is neither a\n"
            "   PASS nor a MISS -- fix the server and rerun; cached successes are reused."
        )
    print("=" * 78)


def _probe_effort(probe: Any) -> Any:
    if isinstance(probe, dict):
        keys = ("reasoning_effort_mechanism", "reasoning_effort", "mechanism", "reasoning_ok")
        found = {k: probe[k] for k in keys if k in probe}
        return found or sorted(probe)
    return repr(probe)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # Imported here, not at module scope, so a missing sibling module becomes a
    # one-line message instead of a traceback before argparse even runs.
    try:
        from hexa_core import baseline_runner as br
        from hexa_core.budget import BudgetExhausted, RolloutBudget
        from hexa_core.llm import VLLMClient
        from hexa_core.tasks import TASK_NAMES, make_task
    except ImportError as exc:
        return _fail_setup(f"hexa_core is not importable: {exc}")

    base_url = args.base_url or os.environ.get("HEXA_VLLM_BASE_URL") or DEFAULT_BASE_URL
    model = args.model or os.environ.get("HEXA_VLLM_MODEL") or DEFAULT_MODEL
    tasks = list(args.tasks) if args.tasks else list(TASK_NAMES)
    unknown = [t for t in tasks if t not in TASK_NAMES]
    if unknown:
        return _fail_setup(f"unknown task(s) {unknown}; known: {list(TASK_NAMES)}")
    if args.runs < 1:
        return _fail_setup(f"--runs must be >= 1, got {args.runs}")

    cfg: dict[str, Any] = {
        "temperature": br.DEFAULT_TEMPERATURE if args.temperature is None else args.temperature,
        "top_p": br.DEFAULT_TOP_P if args.top_p is None else args.top_p,
        "max_tokens": br.DEFAULT_MAX_TOKENS if args.max_tokens is None else args.max_tokens,
        "llm_seed": br.DEFAULT_LLM_SEED if args.llm_seed is None else args.llm_seed,
        "reasoning_effort": (
            br.DEFAULT_REASONING_EFFORT if args.reasoning_effort is None else args.reasoning_effort
        ),
    }
    concurrency = br.DEFAULT_CONCURRENCY if args.concurrency is None else args.concurrency
    tolerance = br.DEFAULT_SEM_TOLERANCE if args.sem_tolerance is None else args.sem_tolerance

    # Always the local vLLM client: this driver's whole output is labelled
    # CALIBRATION and that label is only honest for a non-paper provider.
    # `hexa_core.llm.make_client("nim", ...)` is the seam for a provider-matched
    # reproduction, and that is a different script with a different footnote.
    try:
        client = _call_filtered(
            VLLMClient,
            base_url=base_url,
            model=model,
            api_key=args.api_key,
            timeout=args.timeout,
            temperature=cfg["temperature"],
            top_p=cfg["top_p"],
            max_tokens=cfg["max_tokens"],
            reasoning_effort=cfg["reasoning_effort"],
        )
    except Exception as exc:
        return _fail_setup(f"could not construct VLLMClient: {exc!r}")

    # The probe runs before anything else, always. A 900-call gate against a server
    # that silently drops the chain of thought is 900 wasted calls.
    print(f"[probe] hexa_core.llm.VLLMClient -> {base_url} (model {model!r}) ...")
    try:
        probe: Any = _call_filtered(client.probe, strict=args.strict_probe)
    except Exception as exc:
        print(f"[probe] FAILED: {exc}", file=sys.stderr)
        if type(exc).__name__ == "MissingDependency":
            return _fail_setup(
                "the openai SDK is not installed in this environment, so no request "
                "can be made. Install it in the same env that runs this script "
                "(and see scripts/serve_gpt_oss_vllm.sh for vllm itself)."
            )
        return _fail_setup(f"server probe failed, nothing was measured.\n{SERVE_HINT}")
    print("[probe] ok: " + json.dumps(probe, indent=2, default=str))

    print_header(base_url=base_url, model=model, cfg=cfg, runs=args.runs, n=args.smoke)
    n_per_task = args.smoke if args.smoke is not None else br.EVAL_ITEMS
    print(
        f"\nplan: {len(tasks)} task(s) x {args.runs} run(s) x {n_per_task} problems "
        f"= {len(tasks) * args.runs * n_per_task} generations "
        f"(minus whatever is already cached under {args.output_dir}/)"
    )

    provider_info = {
        "kind": "vllm-local",
        "base_url": base_url,
        "model": model,
        "client": "hexa_core.llm.VLLMClient",
        "probe": probe,
        "paper_provider": "NVIDIA NIM",
        "server": _server_provenance(args.endpoint_json, base_url=base_url),
    }

    results: list[Any] = []
    interrupted = False
    for name in tasks:
        try:
            task = make_task(name, args.train_items, TASK_SEED)
        except Exception as exc:
            print(f"[{name}] could not construct task: {exc!r}", file=sys.stderr)
            return EXIT_SETUP if not results else EXIT_ERROR
        try:
            results.append(
                br.run_baseline_eval(
                    task,
                    client,
                    output_dir=args.output_dir,
                    runs=args.runs,
                    limit=args.smoke,
                    concurrency=concurrency,
                    model=model,
                    task_seed=TASK_SEED,
                    budget=RolloutBudget(max_train=0, max_eval=args.runs * n_per_task),
                    resume=not args.no_resume,
                    provider_info=provider_info,
                    progress=not args.quiet,
                    tolerance_sems=tolerance,
                    **cfg,
                )
            )
        except (KeyboardInterrupt, BudgetExhausted) as exc:
            # Per-problem results are already on disk, so a rerun resumes; report
            # what finished rather than throwing the completed tasks away.
            print(f"\n[{name}] stopped early: {type(exc).__name__}: {exc}", file=sys.stderr)
            interrupted = True
            break

    if not results:
        return _fail_setup("no task produced a result")

    comparisons = br.summarize_gate(results, tolerance_sems=tolerance)
    print_gate_table(results, comparisons, tolerance)
    if any(c.verdict in ("MISS*", "ERROR", "NO-TARGET") for c in comparisons):
        print_diagnostics(results, comparisons, probe)

    print(f"\nresults, manifests and per-problem records: {Path(args.output_dir).resolve()}")
    if interrupted:
        return EXIT_ERROR
    if any(c.verdict == "ERROR" for c in comparisons):
        return EXIT_ERROR
    if any(c.verdict != "PASS*" for c in comparisons):
        return EXIT_MISS
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
