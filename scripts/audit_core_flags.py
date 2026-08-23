#!/usr/bin/env python
"""Diff our CORE invocation against CORE's own published shell scripts.

Why this exists
---------------
Job 63064815 ran for 17 hours and had to be thrown away because our launcher
passed 17 flags to ``run_core.py`` and silently inherited argparse defaults for
everything else -- and two of those defaults are not what CORE publishes:

* ``--knn-max-z``: every one of their twelve scripts sets it to
  ``num_train_items``. Because ``knn_k = min(knn_max_z, |registered|)``, that
  makes the neighbourhood the whole training set, so insight utility is a global
  average -- query-independent by construction. The argparse default of 10
  against our ``n_train=60`` is a *different retrieval rule*, and a CORE number
  produced under it is not CORE as published.
* ``--eval-period``: 10000 in their sample-efficiency scripts (final accuracy
  only), 250 in the rollout-efficiency ones (they need the Figure 2 curve). The
  default is 250, which bought us two unintended 100-problem eval passes.

A flag that is wrong in this direction does not crash and does not look wrong in
the logs. It just quietly produces a number for a different algorithm. The only
defence is to diff, mechanically, every time the configuration moves.

What it checks
--------------
For each flag CORE's published scripts set:

1. If our launcher passes it explicitly -> report as an intentional override and
   show both values, so a deliberate deviation stays visible rather than
   becoming invisible through familiarity.
2. If we do not pass it -> compare their published value against
   ``run_core.py``'s argparse default. A mismatch is an *accidental* deviation
   and exits non-zero.

Defaults are read with ``ast``, not regex: ``add_argument`` calls span multiple
lines and a line-oriented regex silently reports "no default" for most of them,
which reads as a clean diff when it is really a blind one. That false-negative
is exactly the failure mode this script exists to prevent.

Usage
-----
    python scripts/audit_core_flags.py                    # 60-train profile
    python scripts/audit_core_flags.py --n-train 100
    python scripts/audit_core_flags.py --reference 100_train/matchstick

Exit status is 0 only when every flag we leave defaulted matches CORE's
published value.
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import sys
from pathlib import Path

# Flags our launcher passes explicitly (scripts/serve_gpt_oss_vllm.sh, the
# core_matched branch). Kept here rather than parsed out of the shell so that
# adding a flag there without thinking about this audit shows up as a diff.
OURS: dict[str, str] = {
    "--task": "<per-run>",
    "--nvidia-model": "<served model name>",
    "--num-train-items": "60        (theirs: 5/10/100 -- budget matching)",
    "--num-eval-items": "100       (same as theirs)",
    "--max-rollouts": "600       (theirs: 2000 -- budget matching)",
    "--num-baseline-train-runs": "5         (default 10 -- matches HeXA n_base)",
    "--solve-tokens": "60000     (theirs: 32768 -- matches HeXA v3)",
    "--reflection-tokens": "60000     (theirs: 32768 -- matches HeXA v3)",
    "--eval-tokens": "60000     (theirs: 32768 -- matches HeXA v3)",
    "--use-batching-in-train": "true      (they ship false; 5 is their own width)",
    "--num-train-batch-calls": "5         (their default)",
    "--use-batching-in-baseline-train": "true      (baseline has no memory)",
    "--num-baseline-train-batch-calls": "30        (baseline has no memory)",
    "--use-batching-in-eval": "true      (eval is frozen)",
    "--num-eval-batch-calls": "25        (eval is frozen)",
    "--knn-max-z": "= n_train  (tracks their convention)",
    "--eval-period": "10000     (their sample-efficiency value)",
    "--seed": "42",
    "--llm-seed": "42 + run_index",
}


def core_root() -> Path:
    here = Path(__file__).resolve().parents[1]
    return Path(
        os.environ.get("CORE_REPO") or here.parent / "core-reasoning"
    ).resolve()


def argparse_defaults(run_core: Path) -> dict[str, object]:
    """Every ``--flag`` in run_core.py mapped to its default, via ``ast``."""
    tree = ast.parse(run_core.read_text())
    out: dict[str, object] = {}
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
        ):
            continue
        flag = next(
            (
                a.value
                for a in node.args
                if isinstance(a, ast.Constant)
                and isinstance(a.value, str)
                and a.value.startswith("--")
            ),
            None,
        )
        if flag is None:
            continue
        default: object = "<no default>"
        for kw in node.keywords:
            if kw.arg == "default":
                try:
                    default = ast.literal_eval(kw.value)
                except Exception:
                    default = ast.unparse(kw.value)
        out[flag] = default
    return out


def published_flags(script: Path) -> dict[str, str]:
    text = script.read_text()
    return {
        m.group(1): m.group(2).strip().strip('"')
        for m in re.finditer(r'(--[\w-]+)\s+("?[^\\\n"]+"?)', text)
    }


def same(a: object, b: object) -> bool:
    return str(a).strip().strip("'\"").lower() == str(b).strip().strip("'\"").lower()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--reference",
        default="sample_efficiency/100_train/matchstick",
        help="shell_scripts/<this>/run_core.sh to diff against. Use a "
             "sample_efficiency profile for head-to-head accuracy runs; "
             "rollout_efficiency only if you want their Figure 2 eval cadence.",
    )
    ap.add_argument(
        "--n-train",
        type=int,
        default=60,
        help="Our num-train-items, used to check the knn-max-z convention.",
    )
    args = ap.parse_args()

    root = core_root()
    run_core = root / "scripts" / "run_core.py"
    ref = root / "shell_scripts" / args.reference / "run_core.sh"
    for p in (run_core, ref):
        if not p.is_file():
            print(f"ERROR: not found: {p}", file=sys.stderr)
            return 2

    defaults = argparse_defaults(run_core)
    theirs = published_flags(ref)

    print(f"CORE repo : {root}")
    print(f"reference : {ref.relative_to(root)}")
    print(f"our n_train: {args.n_train}\n")

    accidental: list[tuple[str, object, object]] = []
    intentional: list[tuple[str, str, str]] = []
    agree: list[str] = []

    for flag, their_value in sorted(theirs.items()):
        if flag in OURS:
            intentional.append((flag, their_value, OURS[flag]))
        elif same(defaults.get(flag, "<UNKNOWN FLAG>"), their_value):
            agree.append(f"{flag} = {their_value}")
        else:
            accidental.append((flag, their_value, defaults.get(flag, "<UNKNOWN FLAG>")))

    if intentional:
        print("Deliberate overrides (we pass these explicitly):")
        for flag, tv, ov in intentional:
            print(f"  {flag:<34} theirs={tv:<10} ours={ov}")
        print()

    print(f"Inherited defaults that match CORE ({len(agree)}):")
    for line in agree:
        print(f"  {line}")
    print()

    # The knn-max-z convention is a relationship, not a constant, so a value that
    # matches the reference script can still be wrong at our n_train.
    knn = OURS.get("--knn-max-z")
    if knn and "n_train" in knn:
        print(f"knn-max-z convention: theirs = num_train_items in every published "
              f"script; ours tracks n_train, so at n_train={args.n_train} it "
              f"resolves to {args.n_train}. OK\n")

    if accidental:
        print("*** ACCIDENTAL DEVIATIONS -- flags CORE sets that we leave at a "
              "different default ***")
        for flag, tv, dv in accidental:
            print(f"  {flag:<34} theirs={tv!r:<12} our_default={dv!r}")
        print("\nEither pass these explicitly in the launcher or record why the "
              "deviation is intended.")
        return 1

    print("No accidental deviations.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
