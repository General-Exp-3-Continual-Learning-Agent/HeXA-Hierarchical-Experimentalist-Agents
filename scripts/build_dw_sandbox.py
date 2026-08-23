"""Build the sanitised sandbox the DW agents run in.

The comparison is only valid if the agents cannot read the answer. Two facts
drive the design:

1. `interphyre` is pip-installed, and `interphyre/validation/oracles/<level>.py`
   is a solution *generator* whose docstring names the working placement zones.
   An agent that finds it can skip the physics entirely.
2. Python resolves `interphyre` from site-packages, not from the repo checkout.
   Sanitising a repo copy therefore accomplishes nothing — this copies the
   INSTALLED package and redacts that.

What survives: the physics engine and the `validation/` infrastructure that
`environment.py` imports (`from interphyre.validation.placement import
is_valid_placement`). What does not: every per-level oracle body.

This is mitigation, not a jail. Agents keep Bash and the real filesystem stays
readable, so runs must still be audited — see `audit_dw_run.py`.

    python scripts/build_dw_sandbox.py --out /path/to/sandbox_env
    python scripts/build_dw_sandbox.py --out /path/to/sandbox_env --verify-seeds 13,41
"""

import argparse
import os
import re
import shutil
import subprocess
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.expanduser("~/.conda/envs/phyreagent/bin/python")
REDACTION = "# redacted for controlled experiment: solution strategy removed\n"


def installed_interphyre() -> str:
    """Where Python actually resolves `interphyre` — the copy that matters."""
    out = subprocess.run(
        [PY, "-c", "import interphyre, os; print(os.path.dirname(interphyre.__file__))"],
        capture_output=True, text=True, check=True)
    return out.stdout.strip()


def build(out_dir: str) -> None:
    src = installed_interphyre()
    print(f"installed interphyre: {src}")
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    shutil.copytree(src, os.path.join(out_dir, "interphyre"),
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    # Redact every per-level oracle body. `__init__.py` is registry
    # infrastructure (register/get/list + a default runner) and carries no
    # level-specific strategy, so it is kept.
    odir = os.path.join(out_dir, "interphyre", "validation", "oracles")
    n = 0
    if os.path.isdir(odir):
        for fn in sorted(os.listdir(odir)):
            if not fn.endswith(".py") or fn == "__init__.py":
                continue
            with open(os.path.join(odir, fn), "w") as f:
                f.write(REDACTION)
            n += 1
    print(f"redacted {n} oracle modules in {odir}")

    # Minimal agent-facing surface: the toolkit and the probe CLI. Notably
    # absent: level_prompts (carries the "r > 1, drop on the RIGHT end" hint),
    # run_direct_answer*, Direct Results/, Initial_trajectories/, skillrl/.
    ra = os.path.join(out_dir, "react_agent")
    os.makedirs(ra)
    for fn in ("__init__.py", "tools.py"):
        s = os.path.join(_REPO, "react_agent", fn)
        if os.path.exists(s):
            shutil.copy(s, os.path.join(ra, fn))
        else:
            open(os.path.join(ra, fn), "w").close()

    sc = os.path.join(out_dir, "scripts")
    os.makedirs(sc)
    shutil.copy(os.path.join(_REPO, "scripts", "interphyre_probe.py"),
                os.path.join(sc, "interphyre_probe.py"))

    os.makedirs(os.path.join(out_dir, "work"))
    print(f"built sandbox at {out_dir}")


def audit(out_dir: str) -> int:
    """Fail loudly if anything solution-shaped survived the build."""
    bad = []
    for root, _, files in os.walk(out_dir):
        for fn in files:
            fp = os.path.join(root, fn)
            low = fp.lower()
            if any(k in low for k in
                   ("level_prompts", "run_direct", "skill_bank", "trajectory_seed",
                    "initial_trajectories")):
                bad.append(f"forbidden file present: {fp}")
            if "oracles" in low and fn.endswith(".py") and fn != "__init__.py":
                with open(fp) as f:
                    if f.read().strip() != REDACTION.strip():
                        bad.append(f"oracle not redacted: {fp}")
    for b in bad:
        print("AUDIT FAIL:", b, file=sys.stderr)
    print(f"sandbox audit: {len(bad)} problems")
    return 1 if bad else 0


def verify_physics(out_dir: str, level: str, seeds: list) -> int:
    """Sandbox must be physically identical to the real env, or results don't transfer."""
    probe = "scripts/interphyre_probe.py"
    ok = True
    for seed in seeds:
        args = ["simulate", "--x", "0.9", "--y", "-0.5", "--radius", "1.4"]
        env = dict(os.environ, PYTHONPATH=out_dir)
        a = subprocess.run([PY, os.path.join(out_dir, probe), "--level", level,
                            "--seed", str(seed)] + args,
                           capture_output=True, text=True, cwd=out_dir, env=env)
        b = subprocess.run([PY, os.path.join(_REPO, probe), "--level", level,
                            "--seed", str(seed)] + args,
                           capture_output=True, text=True, cwd=_REPO)

        def key(o):
            return re.findall(r"(green_ball|blue_ball): pos=\([^)]*\)", o) + \
                   re.findall(r"Reward:\s*(-?\d+\.?\d*)", o)
        ka, kb = key(a.stdout), key(b.stdout)
        same = ka == kb and ka != []
        ok &= same
        print(f"  seed {seed}: {'IDENTICAL' if same else 'MISMATCH'}  {ka[:2]}")
    print(f"physics verification: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--level", default="catapult")
    p.add_argument("--verify-seeds", default="13,41",
                   help="Comma-separated seeds to check physics parity on.")
    a = p.parse_args()

    build(a.out)
    rc = audit(a.out)
    seeds = [int(s) for s in a.verify_seeds.split(",") if s.strip()]
    rc |= verify_physics(a.out, a.level, seeds)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
