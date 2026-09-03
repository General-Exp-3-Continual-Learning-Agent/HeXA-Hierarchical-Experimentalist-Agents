#!/usr/bin/env python3
"""
Background metrics monitor: tails a Slurm .out log file and writes
recognised metric lines to metrics.csv in the run directory.

Launch as a background process from the Slurm script:
  python parse_metrics.py --log_file logs/train_$JOBID.out \
                           --output_csv logs/runN/metrics.csv &
"""
import argparse
import ast
import csv
import json
import re
import signal
import sys
import time
from pathlib import Path

# Keys that indicate a line is a metrics row (not random log noise).
ANCHOR_KEYS = {"global_step", "step", "epoch"}

# Metric keys we especially care about — used to filter noisy lines.
METRIC_KEYS = re.compile(
    r"(global_step|epoch|reward|score|success|n_turns|kl|entropy|loss|lr|grad_norm)"
)


def try_parse(line: str):
    """Return a flat dict of numeric/string metrics, or None."""
    line = line.strip()
    if not line:
        return None

    # ── 1. JSON object ────────────────────────────────────────────────────────
    if line.startswith("{") and line.endswith("}"):
        for loader in (json.loads, ast.literal_eval):
            try:
                d = loader(line)
                if isinstance(d, dict) and ANCHOR_KEYS & d.keys():
                    return {k: v for k, v in d.items() if isinstance(v, (int, float, str, bool))}
            except Exception:
                pass

    # ── 2. key=value or key: value pairs ─────────────────────────────────────
    if METRIC_KEYS.search(line) and ANCHOR_KEYS & set(re.findall(r"[\w/]+", line)):
        pairs = re.findall(
            r"([\w/]+)\s*[:=]\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)",
            line,
        )
        if pairs:
            d = {}
            for k, v in pairs:
                try:
                    d[k] = int(v) if "." not in v and "e" not in v.lower() else float(v)
                except ValueError:
                    d[k] = v
            if ANCHOR_KEYS & d.keys():
                return d

    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_file", required=True, help="Path to Slurm .out log")
    ap.add_argument("--output_csv", required=True, help="Path to write metrics.csv")
    ap.add_argument("--poll_interval", type=float, default=15.0, help="Seconds between polls")
    args = ap.parse_args()

    log_path = Path(args.log_file)
    csv_path = Path(args.output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    all_fields: list[str] = []
    offset = 0
    running = True

    def _shutdown(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    print(f"[metrics] watching {log_path} → {csv_path}", flush=True)

    while running:
        if log_path.exists():
            try:
                with open(log_path, "r", errors="replace") as f:
                    f.seek(offset)
                    new_lines = f.readlines()
                    offset = f.tell()
            except OSError:
                new_lines = []

            changed = False
            for line in new_lines:
                d = try_parse(line)
                if d:
                    rows.append(d)
                    for k in d:
                        if k not in all_fields:
                            all_fields.append(k)
                    changed = True

            if changed and rows:
                try:
                    with open(csv_path, "w", newline="") as f:
                        w = csv.DictWriter(f, fieldnames=all_fields, extrasaction="ignore")
                        w.writeheader()
                        w.writerows(rows)
                except OSError as e:
                    print(f"[metrics] write error: {e}", flush=True)

        time.sleep(args.poll_interval)

    print(f"[metrics] exiting. {len(rows)} rows written to {csv_path}", flush=True)


if __name__ == "__main__":
    main()
