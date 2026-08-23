"""Stateless CLI shim over `react_agent.tools.InterphyreToolkit`.

Exists so an agent (or a Workflow subagent) can interact with an InterPhyre
level from the shell — one process per probe, no server. Every call is
appended to a JSONL log so the whole interaction history is recoverable as a
trajectory.

Each invocation builds a fresh toolkit, so tools that depend on prior state
within the same process (e.g. `get_contact_log` after `simulate_action`) must
be issued together via the `seq` subcommand.

Examples
--------
    # scene state
    python scripts/interphyre_probe.py --level catapult --seed 6 state

    # test one placement
    python scripts/interphyre_probe.py --level catapult --seed 6 \
        simulate --x 2.0 --y 3.0 --radius 1.2

    # where does the red ball first hit?
    python scripts/interphyre_probe.py --level catapult --seed 6 \
        tool predict_first_contact --args '{"x": 2.0, "y": 3.0, "radius": 1.2}'

    # simulate, then read the contact log from the same rollout
    python scripts/interphyre_probe.py --level catapult --seed 6 seq \
        '[{"tool": "simulate_action", "args": {"x": 2.0, "y": 3.0, "radius": 1.2}},
          {"tool": "get_contact_log", "args": {}}]'
"""

import argparse
import json
import os
import re
import sys
import time

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "interphyre"))

from react_agent.tools import InterphyreToolkit  # noqa: E402


_REWARD_RE = re.compile(r"Reward:\s*(-?\d+\.?\d*)")

# Tools that step the physics engine, i.e. consume one environment rollout.
# Matches the accounting used for HeXA's per-seed rollout counts so budgets are
# comparable across harnesses. Read-only inspectors are free.
ROLLOUT_TOOLS = {
    "simulate_action",
    "simulate_partial",
    "simulate_with_trace",
    "predict_first_contact",
    "trace_green_ball",
}


class BudgetExhausted(Exception):
    pass


def _budget_spend(budget_file: str, key: str, limit: int, n: int = 1) -> int:
    """Increment and persist the rollout counter for `key`; raise if over `limit`.

    Enforced here rather than in the prompt so an agent cannot talk its way past
    the cap. Not a sandbox: an agent that imports the toolkit directly bypasses
    this entirely, so runs must still be audited for that.
    """
    os.makedirs(os.path.dirname(os.path.abspath(budget_file)), exist_ok=True)
    state = {}
    if os.path.exists(budget_file):
        try:
            with open(budget_file) as f:
                state = json.load(f)
        except Exception:
            state = {}
    used = int(state.get(key, 0))
    if used + n > limit:
        raise BudgetExhausted(
            f"ROLLOUT BUDGET EXHAUSTED for '{key}': {used}/{limit} used. "
            f"No further rollouts will run. Commit your best placement now."
        )
    state[key] = used + n
    with open(budget_file, "w") as f:
        json.dump(state, f)
    return state[key]


def _log(log_path: str, record: dict) -> None:
    if not log_path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps(record) + "\n")


def _run(level: str, seed: int, calls: list, log_path: str,
         budget_file: str = None, budget_key: str = None, budget: int = None) -> int:
    toolkit = InterphyreToolkit(level_name=level, seed=seed, is_oss=False)
    try:
        for call in calls:
            tool, args = call["tool"], call.get("args") or {}

            spent = None
            if budget and tool.strip().lower() in ROLLOUT_TOOLS:
                try:
                    spent = _budget_spend(budget_file, budget_key, budget)
                except BudgetExhausted as e:
                    print(f"=== {tool} REFUSED ===\n{e}\n")
                    _log(log_path, {"level": level, "seed": seed, "tool": tool,
                                    "args": args, "observation": str(e),
                                    "success": False, "reward": None,
                                    "refused_over_budget": True,
                                    "budget_key": budget_key})
                    return 3

            start = time.time()
            try:
                obs = toolkit.dispatch_tool(tool, args)
                err = None
            except Exception as e:  # a bad placement should not kill the probe
                obs = f"ERROR: {type(e).__name__}: {e}"
                err = str(e)
            elapsed = time.time() - start

            m = _REWARD_RE.search(obs)
            record = {
                "level": level,
                "seed": seed,
                "tool": tool,
                "args": args,
                "observation": obs,
                "success": "SUCCESS!" in obs,
                "reward": float(m.group(1)) if m else None,
                "elapsed_s": round(elapsed, 2),
                "error": err,
                "budget_key": budget_key,
                "rollouts_used": spent,
            }
            _log(log_path, record)

            hdr = f"=== {tool}({json.dumps(args)}) [{elapsed:.1f}s]"
            if spent is not None:
                hdr += f" | rollouts {spent}/{budget}"
            print(hdr + " ===")
            print(obs)
            if record["success"]:
                print(">>> VERDICT: SUCCESS")
            elif tool in ("simulate_action", "simulate_partial"):
                print(">>> VERDICT: FAILURE")
            print()
    finally:
        toolkit.close()
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Probe an InterPhyre level from the shell")
    p.add_argument("--level", default="catapult")
    p.add_argument("--seed", type=int, default=6)
    p.add_argument(
        "--log", default=None,
        help="JSONL file to append every probe call to (interaction history).",
    )
    p.add_argument("--budget", type=int, default=None,
                   help="Hard cap on physics-stepping calls for --budget-key.")
    p.add_argument("--budget-file", default=None,
                   help="JSON file persisting per-key rollout counters.")
    p.add_argument("--budget-key", default="default",
                   help="Counter name, e.g. the agent id sharing this budget.")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("state", help="get_level_state")

    sp = sub.add_parser("simulate", help="simulate_action to completion")
    sp.add_argument("--x", type=float, required=True)
    sp.add_argument("--y", type=float, required=True)
    sp.add_argument("--radius", type=float, required=True)

    tp = sub.add_parser("tool", help="call any toolkit tool by name")
    tp.add_argument("name")
    tp.add_argument("--args", default="{}", help="JSON dict of tool arguments")

    qp = sub.add_parser("seq", help="run several tools in ONE process (shared state)")
    qp.add_argument("calls", help='JSON list of {"tool": ..., "args": {...}}')

    a = p.parse_args()

    if a.cmd == "state":
        calls = [{"tool": "get_level_state", "args": {}}]
    elif a.cmd == "simulate":
        calls = [{"tool": "simulate_action",
                  "args": {"x": a.x, "y": a.y, "radius": a.radius}}]
    elif a.cmd == "tool":
        calls = [{"tool": a.name, "args": json.loads(a.args)}]
    else:
        calls = json.loads(a.calls)
        if isinstance(calls, dict):
            calls = [calls]

    if a.budget and not a.budget_file:
        p.error("--budget requires --budget-file")

    return _run(a.level, a.seed, calls, a.log,
                budget_file=a.budget_file, budget_key=a.budget_key, budget=a.budget)


if __name__ == "__main__":
    raise SystemExit(main())
