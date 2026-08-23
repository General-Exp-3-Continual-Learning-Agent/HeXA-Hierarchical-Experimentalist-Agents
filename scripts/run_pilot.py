#!/usr/bin/env python3
"""Stage 3a pilot: HExA-passive on one CORE task (`PHASE3_DESIGN.md` §9 step 5).

Wires the pieces built in Phase 3 into one runnable experiment:

    hexa_bridge.patch      CORE task descriptions, binary reward (M2), JSONL loader
    train_baseline         b_q per training problem (M1), seed namespace 1000
    passive_loop           curriculum rounds -> distil -> admit -> frozen eval
    evolve_skill_bank      HeXA's own evolver, teacher = gpt-oss-120b on local vLLM

Default task is **Matchstick** at the 10-example regime.

The design originally specified Hanoi, because "Hanoi is where CORE is weakest, so the
incumbent to beat is MemRL rather than CORE". That rationale died with the CORE
comparison (see the SCOPE CHANGE box in PHASE3_DESIGN.md), and the Hanoi smoke runs
argue against it on their own terms: ~25% of attempts truncate at 32768 tokens (a
token-budget failure no skill can fix), and the remaining failures are 30-move
bookkeeping slips rather than anything a strategy statement addresses.

Matchstick is the better first demonstration: base rate ~0.68 rather than ~0.18, short
answers so truncation is a non-issue, and five declared problem types
(value-loose-chunk, value-tight-chunk, hybrid, operator, tautological) that give the
type hierarchy something real to key on later. Hanoi stays as the hard case for the
generalisation story.

**Inspect before scaling.** The design lists five things to read off a pilot before
committing to the grid, and this script prints all of them: bank-size trajectory,
admission accept rate, same-problem contrast availability, added tokens per eval item,
and the bank itself for memorised answers.

Usage
-----
    python scripts/run_pilot.py --smoke              # ~40 calls, wiring check
    python scripts/run_pilot.py                      # the real pilot
    python scripts/run_pilot.py --task tower-of-hanoi --n-train 5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Never leave .pyc files in the CORE checkout (RECON.md issue 8).
sys.dont_write_bytecode = True

from hexa_core import hexa_bridge  # noqa: E402
from hexa_core.llm.vllm_client import VLLMClient  # noqa: E402
from hexa_core.passive_loop import (  # noqa: E402
    DEFAULT_SAMPLING_UNIFORM_MIX,
    PROBLEM_SAMPLINGS,
    make_schedule,
    run_passive_loop,
)
from hexa_core.injection import DEFAULT_MAX_INJECTED_SKILLS  # noqa: E402
from hexa_core.tasks import TASK_SPECS, make_task  # noqa: E402
from hexa_core.memorisation import (  # noqa: E402
    find_memorised_answers,
    references_from_task,
    report as memorisation_report,
)
from hexa_core.tokens import counter_name, is_exact  # noqa: E402
from hexa_core.train_baseline import estimate_train_baselines  # noqa: E402
from hexa_core.trajectory import load_attempt_trajectories  # noqa: E402

DEFAULT_MODEL = "openai/gpt-oss-120b"
#: ZebraLogic, per WEAKNESSES.md 6. The two tasks already run are structurally poor
#: showcases and both returned non-positive results: Hanoi because the model already
#: knows the algorithm (its failures are 40-move bookkeeping slips, which a prompt-time
#: strategy cannot fix), Matchstick because a 0.76 base rate leaves 0.24 of headroom
#: *and* guarantees enough successes per round to starve the failure channel. ZebraLogic
#: sits at ~0.51: real headroom, failures that are constraint-propagation errors a
#: strategy statement could plausibly repair, and five grid sizes for the type tier.
DEFAULT_TASK = "zebralogic"
#: 60, against the 10/20 the Hanoi and Matchstick campaigns ran. WEAKNESSES.md 4: the
#: bank reached ``max_skills`` in round 0 and stayed flat for nine rounds, and 92% of
#: every problem's 30 attempts were generated and discarded. Trading attempts for
#: problems at a fixed budget triples both the problem diversity and the evidence the
#: evolver ever reads (50 -> 150 trajectories at 600 calls). The cost is `b_q`: it is
#: charged ``n_base x n_train``, which is why :data:`DEFAULT_N_BASE` drops to 5.
DEFAULT_N_TRAIN = 60

#: 30, so 600 calls give 30 rounds x 2 problems x k=10 -- see WEAKNESSES.md 4's table.
DEFAULT_ROUND_TARGET = 30

#: 5, against CORE's 10. Halves `b_q`'s cost (60 problems: 300 calls, not 600) at the
#: price of quantising `b_q` to steps of 0.2. Acceptable *because of what `b_q` is used
#: for*: it is a sort key over at most ten skills and a grading term on the reward the
#: evolver reads, not a reported quantity. Raise it back to 10 if `b_q` ever becomes a
#: number the paper states.
DEFAULT_N_BASE = 5


def endpoint_from_json(path: Path) -> tuple[str, dict]:
    """Read the endpoint the serve script published, rather than guessing a port.

    The serve script scans for a free port when 8000 is taken by a neighbour, so
    hardcoding 8000 is how a run ends up talking to somebody else's server -- which
    happened during Phase 2 (job 62632641).
    """
    data = json.loads(path.read_text())
    return data["base_url"], data


def assert_endpoint_serves(base_url: str, model: str, endpoint_meta: dict) -> None:
    """Verify the endpoint is serving OUR model, right now.

    The endpoint JSON is a record of what was true when the server started. It goes
    **stale the moment that job ends**, and these are shared nodes where port 8000 is
    everyone's default -- so a stale file plus a neighbour's server is a live hazard,
    not a theoretical one. Observed directly: job 62661980 was cancelled externally
    after 2h09m, and `http://gpu015:8000/v1/models` then answered with
    ``{"id": "Qwen3.5-4B"}``. Trusting the JSON at that moment would have run the whole
    pilot against a 4B model and reported the result as gpt-oss-120b.

    The serve script performs this same check at startup; it has to happen again here
    because the two events are hours apart.
    """
    import urllib.error
    import urllib.request

    url = base_url.rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            listing = json.loads(response.read())
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        raise SystemExit(
            f"ERROR: cannot reach {url} ({type(exc).__name__}: {exc}).\n"
            f"       The server recorded in the endpoint JSON is gone. Start one:\n"
            f"           sbatch scripts/serve_gpt_oss_vllm.sh"
        )

    served = [entry.get("id") for entry in listing.get("data", [])]
    if model not in served:
        job = endpoint_meta.get("job_id", "?")
        raise SystemExit(
            f"ERROR: {base_url} is NOT serving {model!r}.\n"
            f"       It advertises: {served}\n"
            f"       The endpoint JSON was written by job {job}; that job has since\n"
            f"       ended and another process now holds this port. Running anyway\n"
            f"       would score a different model and report it under our name.\n"
            f"       Start a fresh server:\n"
            f"           sbatch scripts/serve_gpt_oss_vllm.sh"
        )


def build_evolver(teacher_model: str, max_skills: int, max_mistakes: int):
    """Return an ``evolve(level, bank, round_dir, round_index)`` callable.

    `evolve_skill_bank` is HeXA's own evolver, unmodified. It takes a *directory* and
    calls `load_trajectories` on it -- which `hexa_bridge.patch` has rebound to read our
    ``*.jsonl`` attempt files. So the integration is the patch plus the file, with no
    fork of the evolution logic.
    """
    from skillrl.distillation.evolving_distill import evolve_skill_bank

    def evolve(level_name, prev_bank, round_dir, round_index):
        from skillrl.core.skill_bank import SkillBank

        bank = prev_bank if prev_bank is not None else SkillBank()
        try:
            return evolve_skill_bank(
                level_name=level_name,
                prev_bank=bank,
                new_trajs_dir=Path(round_dir),
                output_path=Path(round_dir) / "bank.json",
                max_skills=max_skills,
                max_mistakes=max_mistakes,
                teacher_model=teacher_model,
            )
        except Exception as exc:  # noqa: BLE001
            # A failed evolution must not discard a round's rollouts. Report and carry
            # the previous bank forward: an empty round is a data point, a lost round
            # is 200 wasted calls.
            print(f"  [evolve] round {round_index} FAILED: {type(exc).__name__}: {exc}")
            return bank

    return evolve


def final_bank_skills(out_dir: Path, n_rounds: int) -> list:
    """Every skill in the last round's saved bank, across both tiers.

    Read back from disk rather than threaded out of the loop: the bank file is what a
    reviewer would open, so checking exactly that artifact is the point.
    """
    from skillrl.core.skill_bank import Skill

    path = out_dir / f"round{max(0, n_rounds - 1)}" / "bank.json"
    if not path.is_file():
        return []
    raw = json.loads(path.read_text())
    skills = [Skill(**s) for s in (raw.get("general_skills") or [])]
    for group in (raw.get("level_skills") or {}).values():
        skills.extend(Skill(**s) for s in group)
    return skills


def paired_stats(out_dir: Path) -> dict | None:
    """McNemar's exact test on the paired eval arms.

    Paired, not two-proportion: the arms ran the same problems with the same seeds, so
    only the *discordant* pairs -- problems one arm solved and the other did not --
    carry information. Ignoring the pairing throws away most of the power, which at
    n=100 is the difference between detecting a real effect and reporting a shrug.
    """
    import json as _json
    from math import comb

    paths = {
        arm: out_dir / f"eval_{arm}.jsonl" for arm in ("with_memory", "no_memory")
    }
    if not all(path.is_file() for path in paths.values()):
        return None

    rows = {
        arm: {
            r["problem_id"]: bool(r["correct"])
            for r in (_json.loads(l) for l in path.read_text().splitlines() if l.strip())
        }
        for arm, path in paths.items()
    }
    shared = set(rows["with_memory"]) & set(rows["no_memory"])
    helped = sum(1 for p in shared if rows["with_memory"][p] and not rows["no_memory"][p])
    hurt = sum(1 for p in shared if rows["no_memory"][p] and not rows["with_memory"][p])

    n = helped + hurt
    if n == 0:
        return {"helped": 0, "hurt": 0, "p": 1.0,
                "verdict": "identical on every problem -- the bank changed nothing"}

    # Two-sided exact binomial on the discordant pairs, H0: p = 0.5.
    k = min(helped, hurt)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    p = min(1.0, 2 * tail)

    if p >= 0.05:
        verdict = f"NOT significant (p={p:.3f}) -- more runs needed"
    elif helped > hurt:
        verdict = "memory HELPED, significant at 0.05"
    else:
        verdict = "memory HURT, significant at 0.05"
    return {"helped": helped, "hurt": hurt, "p": p, "verdict": verdict}


def contrast_availability(round_dir: Path) -> dict:
    """How often a same-problem contrast pair was actually available.

    §1a predicts this differs systematically by regime, and it is the diagnostic that
    says whether the curriculum is producing usable contrast at all. A round with only
    successes or only failures yields nothing contrastive to distil.
    """
    successes, failures = load_attempt_trajectories(round_dir)
    by_problem: dict[str, set[bool]] = {}
    for traj in successes:
        by_problem.setdefault(str(traj.get("seed")), set()).add(True)
    for traj in failures:
        by_problem.setdefault(str(traj.get("seed")), set()).add(False)
    mixed = sum(1 for outcomes in by_problem.values() if len(outcomes) == 2)
    return {
        "n_successes": len(successes),
        "n_failures": len(failures),
        "problems": len(by_problem),
        "problems_with_same_problem_contrast": mixed,
        "has_any_contrast": bool(successes and failures),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--n-train", type=int, default=DEFAULT_N_TRAIN)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--teacher-model", default=None,
                        help="default: vllm:<--model>, i.e. the same server")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--endpoint-json", default="logs/vllm_endpoint_latest.json")
    parser.add_argument("--budget-calls", type=int, default=2000)
    parser.add_argument("--round-target", type=int, default=DEFAULT_ROUND_TARGET)
    parser.add_argument("--visits-per-problem", type=int, default=1,
                        help="How many rounds each training problem appears in. 1 is "
                             "the frozen streaming regime (each problem seen once, "
                             "never again). >1 re-presents it in later rounds so a "
                             "skill distilled from a failure can be validated after "
                             "the bank has changed. Paid for out of attempts per "
                             "problem, so total calls are unchanged.")
    parser.add_argument("--problem-sampling", default="round_robin",
                        choices=list(PROBLEM_SAMPLINGS),
                        help="How the curriculum allocates problem visits. "
                             "'difficulty' apportions the surplus by measured "
                             "hardness (1 - b_q), mirroring CORE's biased problem "
                             "sampling. Requires --m1 for the b_q pass.")
    parser.add_argument("--sampling-uniform-mix", type=float,
                        default=DEFAULT_SAMPLING_UNIFORM_MIX,
                        help="Uniform share of the difficulty-weighted allocation. "
                             "0.0 is pure hardness weighting; 1.0 is uniform. "
                             "CORE publishes 0.5.")
    parser.add_argument("--n-base", type=int, default=DEFAULT_N_BASE)
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="output budget spanning reasoning + answer. Default is "
                             "the client's 32768 (CORE's setting), which truncated 53 "
                             "of 100 Hanoi eval problems. Server context is 65536")
    parser.add_argument("--run-index", type=int, default=0,
                        help="distinguishes repeat runs: selects the CORE eval seed "
                             "namespace offset, so runs are independent samples")
    parser.add_argument("--concurrency", type=int, default=40,
                        help="in-flight requests. Deliberately above CORE's "
                             "--num-eval-batch-calls default of 10: vLLM reports "
                             "~87x max concurrency for this server, and a near-idle "
                             "GPU is both slow and what the cluster's "
                             "underutilisation reaper cancels")
    parser.add_argument("--injection-budget-tokens", type=int, default=1000)
    parser.add_argument("--truncation-aware-evolution", action="store_true",
                        help="add the truncation-aware addendum to the evolution "
                             "prompt: count truncated attempts and, when they dominate "
                             "the failures, bank a search-shortening skill instead of "
                             "another verification check (WEAKNESSES.md 11). OFF by "
                             "default -- every banked result was produced without it, "
                             "and off keeps the prompt byte-identical to those runs")
    parser.add_argument("--max-injected-skills", type=int,
                        default=DEFAULT_MAX_INJECTED_SKILLS,
                        help="how many task-tier skills reach the prompt "
                             "(WEAKNESSES.md 11 / 5). The default 16 exceeds any bank "
                             "we build, i.e. no cap -- which is what every completed "
                             "campaign effectively ran at. Lower it to trim the block "
                             "from its low-confidence tail: the lever for truncation on "
                             "tasks where the block adds per-candidate work")
    # Flat retrieval. `task:type` splits an already-small bank across type keys and
    # the specific tier stays empty until distillation runs per type, so it buys
    # nothing yet and costs a moving part.
    parser.add_argument("--hierarchy", default="task", choices=["task", "task:type"])

    # --- borrowed from CORE: OFF by default -----------------------------------
    # HeXA has no admission gate (PHASE3_DESIGN.md 5c: "the evolver's output goes
    # straight into the bank"). Running it on would measure HeXA-plus-CORE's-gate,
    # not HeXA. It also emptied the bank outright on Hanoi -- 0 of 4 candidates
    # admitted, because a single trial on a ~19%-base-rate task rejects good skills
    # ~4 times in 5.
    parser.add_argument("--admission", action="store_true",
                        help="enable CORE's admission gate (M5). Off by default: it "
                             "is not part of HeXA")
    parser.add_argument("--admission-trials", type=int, default=1)

    # M1's baseline-relative confidence is a genuine change to HeXA's method, so the
    # headline runs without it. The *pooling* half of M1 is not optional and is always
    # on -- it is a bug fix for k-attempts-per-problem, not a modification.
    parser.add_argument("--contrast-selection",
                        default=hexa_bridge.DEFAULT_TRAJECTORY_SELECTION,
                        choices=list(hexa_bridge.TRAJECTORY_SELECTIONS),
                        help="which 5 trajectories the evolver sees. paired (default) "
                             "= same-problem contrast pairs first; balanced = a global "
                             "success/failure quota; top5 = HeXA's sort-by-reward, "
                             "which sent ZERO failures in every round of every Hanoi "
                             "AND Matchstick run -- kept as the ablation arm")
    parser.add_argument("--max-trajs", type=int,
                        default=hexa_bridge.DEFAULT_MAX_TRAJS,
                        help="how many trajectories the evolver sees per round. "
                             "Overrides the 5 hardcoded in evolving_distill; 5 is the "
                             "ablation arm. Bounded by the teacher's input context")
    parser.add_argument("--m1", action="store_true",
                        help="use baseline-relative utility (correct - b_q) as THE "
                             "reward: ranks trajectories for the evolver, labels them "
                             "in its prompt, and sets skill confidence. Costs "
                             "n_base x |D_train| extra calls")
    parser.add_argument("--no-parent-exclusion", action="store_true")
    parser.add_argument("--max-skills", type=int, default=10)
    parser.add_argument("--max-mistakes", type=int, default=5)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--skip-baseline", action="store_true",
                        help=argparse.SUPPRESS)  # superseded by --m1
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--eval-n-train", type=int, default=None,
                        help="Additionally evaluate the trained bank on the eval split "
                             "belonging to THIS n_train, on top of the native one. Only "
                             "meaningful where the eval split moves with the regime "
                             "(mathgap): --n-train 60 --eval-n-train 100 trains on "
                             "data[:60] and also scores data[100:200], the window CORE's "
                             "published MathGAP numbers use. Refuses any value whose "
                             "window overlaps the training items.")
    parser.add_argument("--smoke", action="store_true",
                        help="tiny budget + n_base=2 + no eval: wiring check only")
    args = parser.parse_args()

    # Config guards run before the endpoint probe: a flag combination that cannot
    # work should not cost a GPU allocation to discover.
    if args.problem_sampling != "round_robin" and not args.m1:
        parser.error(
            "--problem-sampling difficulty weights by b_q, which only the --m1 "
            "baseline pass measures. Pass --m1 or use round_robin."
        )

    if args.smoke:
        args.budget_calls = min(args.budget_calls, 40)
        args.round_target = min(args.round_target, 2)
        args.n_base = min(args.n_base, 2)
        args.n_train = min(args.n_train, 4)
        args.no_eval = True

    # ---- endpoint ----------------------------------------------------------
    base_url = args.base_url
    endpoint_meta: dict = {}
    if base_url is None:
        path = Path(args.endpoint_json)
        if not path.is_file():
            print(f"ERROR: no endpoint JSON at {path}. Start the server first:")
            print("    sbatch scripts/serve_gpt_oss_vllm.sh")
            print("or pass --base-url explicitly.")
            return 2
        base_url, endpoint_meta = endpoint_from_json(path)

    # Before anything else: is this endpoint still ours? See the docstring.
    assert_endpoint_serves(base_url, args.model, endpoint_meta)

    teacher_model = args.teacher_model or f"vllm:{args.model}"
    # The teacher resolves its endpoint from the environment (llm/teacher.py).
    os.environ.setdefault("HEXA_TEACHER_BASE_URL", base_url)

    out_dir = Path(args.output_dir or f"results/pilot/{args.task}/n{args.n_train}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- patch skillrl for CORE tasks --------------------------------------
    # Binary reward (M2), M1 baseline-relative confidence, CORE task descriptions,
    # and the JSONL trajectory loader.
    #
    # M1 matters here more than it looks. Left on the stock confidence function,
    # `_compute_initial_confidence` reads the binary outcome and keeps one trajectory
    # per problem, so under k-attempts-per-problem every skill distilled from an
    # ever-solved problem scores exactly 1.0 -- and confidence is the retrieval
    # ranking key. `--no-m1` runs that arm deliberately, for comparison.
    # The pooled confidence function is installed ALWAYS. It is not M1 and not a
    # method change: the stock `_compute_initial_confidence` keeps one trajectory per
    # seed, which is correct for InterPhyre (one attempt per level) and simply wrong
    # here (k = 20..400 attempts per problem), discarding all but the first and
    # saturating every skill at 1.0.
    #
    # `--m1` controls the *other* half -- whether b_q is measured and stamped onto the
    # attempts. `attempt_utility` reads `baseline` when present and falls back to the
    # binary outcome when absent, so the two compose without a second switch:
    #
    #   --m1 off : confidence = pooled mean of (+1/-1) over every attempt
    #   --m1 on  : confidence = pooled mean of (correct - b_q) over every attempt
    #
    # An earlier version tied both to `--m1`, so the default run silently got the
    # stock first-occurrence function back and every skill scored 1.000 again.
    hexa_bridge.patch(
        load_trajectories_fn=load_attempt_trajectories,
        # With --m1, baseline-relative utility IS the reward -- one number doing all
        # three jobs the paper gives the reward: ranking which trajectories the evolver
        # reads, labelling them in its prompt, and setting skill confidence.
        #
        # This matters most for the first job. Under binary reward every success ties
        # at +1.0, so "top 5 by reward" picks 5 arbitrary successes. Under utility a
        # success on a hard problem (b_q=0.15 -> +0.85) outranks one on an easy problem
        # (b_q=0.85 -> +0.15), so the evolver reads the most informative evidence --
        # which is what InterPhyre's graded reward (1.0/0.75/0.5/0.25) does via speed,
        # and what a single-turn task has no turns to express.
        reward_fn=(
            hexa_bridge.attempt_utility if args.m1
            else hexa_bridge.binary_trajectory_reward
        ),
        confidence_fn=hexa_bridge.compute_initial_confidence_core,
        trajectory_selection=args.contrast_selection,
        max_trajs=args.max_trajs,
        truncation_aware_evolution=args.truncation_aware_evolution,
    )

    task = make_task(args.task, args.n_train)
    schedule = make_schedule(
        args.n_train,
        budget=args.budget_calls,
        round_target=args.round_target,
        admission_trials=args.admission_trials,
        admission_enabled=args.admission,
        visits_per_problem=args.visits_per_problem,
    )
    client = VLLMClient(
        model=args.model,
        base_url=base_url,
        **({"max_tokens": args.max_tokens} if args.max_tokens else {}),
    )

    print("=" * 78)
    print("  Stage 3a pilot -- HExA-passive on CORE")
    print(f"  task        : {args.task}   n_train={args.n_train}")
    print(f"  endpoint    : {base_url}   model={args.model}")
    print(f"  teacher     : {teacher_model}")
    print(f"  schedule    : {schedule.rounds} rounds x "
          f"{schedule.problems_per_round} problems x "
          f"{schedule.attempts_per_problem} attempts = {schedule.total_calls} calls"
          f" (slack {schedule.slack})")
    print(f"  sampling    : {args.problem_sampling} "
          f"(visits/problem {args.visits_per_problem}, "
          f"{schedule.problem_visits} visits total"
          + (f", uniform_mix {args.sampling_uniform_mix}"
             if args.problem_sampling != "round_robin" else "") + ")")
    if args.m1:
        print(f"  b_q         : n_base={args.n_base} -> "
              f"{args.n_base * args.n_train} calls (outside max_train)")
    print(f"  injection   : {args.injection_budget_tokens} tokens, "
          f"hierarchy={args.hierarchy}, counter={counter_name()} exact={is_exact()}")
    if args.admission:
        print(f"  admission   : {args.admission_trials} trial(s)  [CORE M5, opt-in]"
              f"   reserve={schedule.admission_reserve} "
              f"(<={schedule.admission_cap_per_round}/round)")
    else:
        print("  admission   : off -- HeXA has no accept/reject gate")
    print(f"  confidence  : pooled over all attempts, "
          f"{'baseline-relative (M1)' if args.m1 else 'binary'}")
    print(f"  contrast    : {args.contrast_selection}")
    print(f"  concurrency : {args.concurrency}")
    print(f"  output      : {out_dir}")
    if endpoint_meta:
        print(f"  server      : reasoning_field={endpoint_meta.get('reasoning_field')} "
              f"effort={endpoint_meta.get('reasoning_effort_mechanism')} "
              f"harmony_date={endpoint_meta.get('vllm_system_start_date')}")
    print("=" * 78)

    # ---- probe before spending anything ------------------------------------
    print("\n[probe] verifying the server produces comparable output ...")
    evidence = client.probe(strict=False, differential=True)
    print(f"  reasoning_parser={evidence.get('reasoning_parser')} "
          f"field={evidence.get('reasoning_field')} "
          f"effort_verified={evidence.get('effort_verified')}")
    if not evidence.get("reasoning_parser"):
        print("ERROR: no reasoning channel. raw_text would drop the chain of thought,")
        print("       so every score would be computed on the wrong string. Aborting.")
        return 3
    (out_dir / "probe.json").write_text(json.dumps(evidence, indent=2, default=str))

    started = time.time()

    # ---- b_q ---------------------------------------------------------------
    baselines = None
    if args.m1:
        print(f"\n[b_q] estimating no-memory baselines "
              f"({args.n_base} x {args.n_train} = {args.n_base * args.n_train} calls) ...")
        baselines = estimate_train_baselines(
            task,
            client.generate,
            n_train=args.n_train,
            n_base=args.n_base,
            output_path=out_dir / "b_q.json",
            concurrency=args.concurrency,
        )
        print(f"  mean b_q = {baselines.mean_b_q}  "
              f"usable {sum(1 for r in baselines.baselines.values() if r.usable)}"
              f"/{len(baselines.baselines)}")
        for pid, record in sorted(baselines.baselines.items()):
            print(f"    {pid}: b_q={record.b_q:.2f} ({record.successes}/{record.n_base})"
                  + (f" errors={record.errors}" if record.errors else ""))

    # ---- the loop ----------------------------------------------------------
    def generate_meta(prompt: str, seed: int | None = None) -> tuple[str, str]:
        """(raw_text, finish_reason), seeded.

        Two things depend on this signature.

        `finish_reason`: `client.generate` discards it, and truncation is not a
        footnote here -- it accounted for 53 of 100 eval problems in the first Hanoi
        run, identically in both arms. Without it a weak number cannot be told apart
        from a token-limit artefact.

        `seed`: **accepting this keyword is what makes the two eval arms paired.**
        `_generate_many` passes the CORE eval seed per item and falls back to an
        unseeded call on TypeError -- so an earlier version of this function, which
        took only `prompt`, silently produced two *independent* samples while the
        run summary recorded seeds as though they were paired. The McNemar test
        assumes pairing; without it the test is simply the wrong test.
        """
        generation = client.generate_structured(prompt=prompt, seed=seed)
        if generation is None:
            return "", "error"
        return (generation.raw_text or ""), (generation.finish_reason or "")

    print(f"\n[loop] running {schedule.rounds} curriculum rounds ...")
    result = run_passive_loop(
        task,
        client.generate,
        evolve=build_evolver(teacher_model, args.max_skills, args.max_mistakes),
        n_train=args.n_train,
        budget_calls=args.budget_calls,
        round_target=args.round_target,
        generate_meta=generate_meta,
        baselines=baselines,
        run_index=args.run_index,
        injection_budget_tokens=args.injection_budget_tokens,
        max_injected_skills=args.max_injected_skills,
        parent_exclusion=not args.no_parent_exclusion,
        admission_enabled=args.admission,
        admission_trials=args.admission_trials,
        hierarchy=args.hierarchy,
        output_dir=out_dir,
        evaluate=not args.no_eval,
        concurrency=args.concurrency,
        visits_per_problem=args.visits_per_problem,
        sampling=args.problem_sampling,
        sampling_uniform_mix=args.sampling_uniform_mix,
    )

    # ---- second eval window (mathgap only, in practice) --------------------
    # MathGAP's held-out set is `data[n_train : n_train + 100]`, so it *moves with the
    # training regime* -- training on `data[:60]` evaluates on `data[60:160]`, while
    # CORE's published MathGAP numbers are the `n_train=100` window, `data[100:200]`.
    # The first three replicates got that window from a separate job
    # (`scripts/rerun_frozen_eval.py` against the frozen bank), which costs a whole
    # second GPU allocation and a second queue wait for 200 calls of work.
    #
    # Nothing required that. `_frozen_eval` takes the task it is handed and touches it
    # exactly once, and eval is not charged against the training budget, so a second
    # task built purely for its eval split can be evaluated in the same job. The
    # training task is untouched: `--eval-n-train` only ever supplies
    # `eval_problems()`, never train items.
    #
    # This is additive. The native-window eval above still runs and is still what
    # `run_summary.json` reports, so a run using this flag stays directly comparable to
    # every run that predates it.
    if args.eval_n_train is not None and not args.no_eval:
        from hexa_core.budget import RolloutBudget  # noqa: PLC0415
        from hexa_core.passive_loop import _frozen_eval  # noqa: PLC0415
        from skillrl.core.skill_bank import SkillBank  # noqa: PLC0415

        # `PassiveRunResult` carries no bank -- the bank is persisted per round, so the
        # trained bank is the last round's file. Same object the native-window eval used,
        # and the same one `rerun_frozen_eval.py` would have loaded in a separate job.
        final_round = max(r.round_index for r in result.rounds)
        bank_path = out_dir / f"round{final_round}" / "bank.json"
        trained_bank = SkillBank.load(str(bank_path))

        eval_task = make_task(args.task, args.eval_n_train)
        eval_manifest = eval_task.manifest()
        eval_out = out_dir / f"eval_window_n{args.eval_n_train}"
        eval_out.mkdir(parents=True, exist_ok=True)

        # The training split came from `task`, the eval split from `eval_task`; nothing
        # in the type system stops those from overlapping, so check by id rather than
        # trusting the arithmetic. Cheap, and the failure it catches is silent.
        train_ids = {p.id for p in task.train_problems()}
        window_ids = {p.id for p in eval_task.eval_problems()}
        overlap = train_ids & window_ids
        if overlap:
            raise SystemExit(
                f"--eval-n-train {args.eval_n_train} produces an eval window that "
                f"overlaps the {len(train_ids)} training items in {len(overlap)} "
                f"places (e.g. {sorted(overlap)[:3]}). Refusing to score a bank on "
                f"problems it trained on."
            )

        eval_kwargs = dict(
            budget=RolloutBudget(max_train=0),
            injection_budget_tokens=args.injection_budget_tokens,
            max_injected_skills=args.max_injected_skills,
            hierarchy=args.hierarchy,
            parent_exclusion=not args.no_parent_exclusion,
            concurrency=args.concurrency,
            generate_meta=generate_meta,
            run_index=args.run_index,
        )
        print(f"\n[eval-window] n_train={args.eval_n_train} -> "
              f"{eval_manifest['num_eval_items']} items, "
              f"digest {eval_manifest['eval_ids_sha256'][:16]}")
        w_acc, w_n, w_tr = _frozen_eval(
            eval_task, trained_bank, client.generate, with_memory=True,
            record_path=eval_out / "eval_with_memory.jsonl", **eval_kwargs,
        )
        n_acc, _, n_tr = _frozen_eval(
            eval_task, None, client.generate, with_memory=False,
            record_path=eval_out / "eval_no_memory.jsonl", **eval_kwargs,
        )
        print(f"  with-memory {w_acc:.3f} (trunc {w_tr}/{w_n})   "
              f"no-memory {n_acc:.3f} (trunc {n_tr}/{w_n})   delta {w_acc - n_acc:+.3f}")
        (eval_out / "rerun_config.json").write_text(json.dumps({
            "task": args.task, "n_train": args.eval_n_train,
            "train_n_train": args.n_train, "in_training_job": True,
            "eval_ids_sha256": eval_manifest["eval_ids_sha256"],
            "max_tokens": args.max_tokens, "run_index": args.run_index,
            "with_memory_acc": w_acc, "no_memory_acc": n_acc,
            "delta": w_acc - n_acc,
            "with_memory_truncated": w_tr, "no_memory_truncated": n_tr,
        }, indent=2))
        provenance["eval_split_secondary"] = {
            "n_train": args.eval_n_train,
            "eval_ids_sha256": eval_manifest["eval_ids_sha256"],
            "num_eval_items": eval_manifest["num_eval_items"],
        }
        (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))

    # ---- the five things to inspect before scaling -------------------------
    elapsed = (time.time() - started) / 60
    print("\n" + "=" * 78)
    print("  PILOT DIAGNOSTICS -- read all seven before scaling to the grid")
    print("=" * 78)

    print("\n1. bank size vs round (is it growing, and does it plateau?)")
    for record in result.rounds:
        contrast = contrast_availability(out_dir / f"round{record.round_index}")
        print(f"   round {record.round_index}: "
              f"skills {record.n_skills_before} -> {record.n_skills_after}  "
              f"train_acc={record.summary.get('accuracy', 0):.3f}  "
              f"contrast(S/F)={contrast['n_successes']}/{contrast['n_failures']}"
              f"{'  NO-CONTRAST' if not contrast['has_any_contrast'] else ''}")

    print("\n2. admission accept rate (CORE's cheapest structural advantage)")
    for record in result.rounds:
        info = record.admission
        if info.get("skipped"):
            print(f"   round {record.round_index}: no evolution")
            continue
        print(f"   round {record.round_index}: judged={info.get('judged')} "
              f"admitted={info.get('admitted_of_judged')} "
              f"rate={info.get('accept_rate_judged')} "
              f"unjudged={info.get('unjudged')} {info.get('unjudged_reasons') or ''}")

    print("\n3. same-problem contrast availability (§1a: regime-dependent)")
    for record in result.rounds:
        contrast = contrast_availability(out_dir / f"round{record.round_index}")
        print(f"   round {record.round_index}: "
              f"{contrast['problems_with_same_problem_contrast']}"
              f"/{contrast['problems']} problems had both outcomes")

    print("\n4. added tokens per attempt (Pareto x-axis; CORE = 0.92k)")
    for record in result.rounds:
        added = [i["added_tokens"] for i in record.injections]
        mean = sum(added) / len(added) if added else 0
        print(f"   round {record.round_index}: mean={mean:.0f} "
              f"max={max(added) if added else 0}  "
              f"starved={record.starved_injections} "
              f"budget_bound={record.budget_bound_injections}")
    print(f"   token counter = {counter_name()}  exact = {is_exact()}"
          + ("" if is_exact() else "   <- NOT citable; install tiktoken for figures"))

    print("\n5. truncation (only ever depresses accuracy; gate saw Hanoi 9/20)")
    for record in result.rounds:
        reasons: dict[str, int] = {}
        for attempt in record.attempts:
            reasons[attempt.finish_reason or "?"] = (
                reasons.get(attempt.finish_reason or "?", 0) + 1
            )
        n_trunc = reasons.get("length", 0)
        total = len(record.attempts) or 1
        flag = "   <- truncation is material; raise max_tokens" if n_trunc / total > 0.1 else ""
        print(f"   round {record.round_index}: length={n_trunc}/{len(record.attempts)}"
              f"  reasons={reasons}{flag}")
    if result.eval_accuracy is not None:
        print(f"   eval: length={result.eval_truncated}/{result.eval_n}")

    print("\n6. contamination + curriculum integrity")
    print(f"   parent exclusions fired : {result.excluded_parent_total}"
          + ("   <- under pure streaming this should be 0 (curriculum leak?)"
             if result.streaming_leak else "   (clean)"))
    print(f"   starved injections      : {result.starved_total}"
          + ("   <- these attempts were secretly no-memory"
             if result.starved_total else "   (clean)"))
    print(f"   problems attempted      : {len(result.attempted_ids)}")
    print(f"   budget                  : {result.budget.report() if result.budget else '-'}")

    if result.eval_accuracy is not None:
        print("\n" + "=" * 78)
        print("  THE RESULT -- held-out, bank frozen, paired against no memory")
        print("=" * 78)
        print(f"   HeXA (with bank)  : {result.eval_accuracy:.3f}   (n={result.eval_n})")
        if result.eval_accuracy_no_memory is not None:
            delta = result.eval_accuracy - result.eval_accuracy_no_memory
            print(f"   no memory         : {result.eval_accuracy_no_memory:.3f}   "
                  f"(same problems, same seeds)")
            print(f"   delta             : {delta:+.3f}")
            stats = paired_stats(out_dir)
            if stats:
                print(f"   discordant pairs  : {stats['helped']} helped / "
                      f"{stats['hurt']} hurt   (McNemar exact p = {stats['p']:.3f})")
                print(f"   verdict           : {stats['verdict']}")
        else:
            print("   no-memory control : NOT RUN -- this number has nothing to beat")
        if baselines is not None and baselines.mean_b_q is not None:
            print(f"   train mean b_q    : {baselines.mean_b_q:.3f} "
                  f"(different split -- difficulty reference, not a comparison)")
        print("=" * 78)

    print("\n7. memorised training answers (PHASE3_DESIGN.md 5b)")
    # Automated because the manual read is easy to skip and easy to miss: the first
    # completed smoke run hid two verbatim moves from problem s201 inside a skill's
    # `example` field, and the prompt-side instruction did not prevent it.
    try:
        leaks = find_memorised_answers(
            final_bank_skills(out_dir, len(result.rounds)),
            task.train_problems(args.n_train),
            references_from_task(task),
        )
        print(memorisation_report(leaks))
    except Exception as exc:  # noqa: BLE001
        # Never let the guard's own failure read as an all-clear.
        print(f"   CHECK DID NOT RUN: {type(exc).__name__}: {exc}")

    # Provenance: which CORE checkout supplied the tasks, and was it modified.
    # "CORE unmodified" underwrites every number here, so it is recorded per run
    # rather than asserted in prose.
    from hexa_core.core_bridge import core_provenance

    provenance = core_provenance()
    # The configuration that produced these numbers. Three completed campaigns cannot
    # be attributed because this was never recorded: the run summary held the schedule
    # and the result, but nothing about which reward, confidence or selection was
    # bound -- so "did the evolver see r - b_q?" had no answer in the artifacts.
    provenance["config"] = {
        "task": args.task,
        "n_train": args.n_train,
        "budget_calls": args.budget_calls,
        "round_target": args.round_target,
        "visits_per_problem": args.visits_per_problem,
        "problem_sampling": args.problem_sampling,
        "sampling_uniform_mix": (
            args.sampling_uniform_mix
            if args.problem_sampling != "round_robin"
            else None
        ),
        "max_tokens": args.max_tokens,
        "run_index": args.run_index,
        "m1_baseline_relative_reward": bool(args.m1),
        "n_base": args.n_base if args.m1 else 0,
        "reward_fn": "attempt_utility" if args.m1 else "binary_trajectory_reward",
        "confidence_fn": "compute_initial_confidence_core",
        "contrast_selection": args.contrast_selection,
        # How wide the evolver's evidence window actually was. Not inferable from the
        # code, since this overrides a hardcoded 5 in shared `skillrl` code.
        "max_trajs": args.max_trajs,
        "success_slots": hexa_bridge.default_success_slots(args.max_trajs),
        "hierarchy": args.hierarchy,
        "injection_budget_tokens": args.injection_budget_tokens,
        "max_injected_skills": args.max_injected_skills,
        "truncation_aware_evolution": bool(args.truncation_aware_evolution),
        "max_skills": args.max_skills,
        "max_mistakes": args.max_mistakes,
        "admission": bool(args.admission),
        "parent_exclusion": not args.no_parent_exclusion,
        "model": args.model,
        "teacher_model": teacher_model,
        "baselines_stamped": result.baselines_stamped,
        "baselines_missing": result.baselines_missing,
    }
    # Which 100 items were actually evaluated. For three of the four tasks this is
    # implied by (task, regime) and the digest is merely a cheap cross-check; for
    # MathGAP it is the only thing that is, because its eval set is
    # `data[n_train : n_train + 100]` and so *moves with the regime*. Two MathGAP runs
    # may only be pooled if these agree. Recorded rather than derived, so an artifact
    # from a future CORE checkout can still be checked against one from today.
    task_manifest = task.manifest()
    provenance["eval_split"] = {
        "eval_ids_sha256": task_manifest["eval_ids_sha256"],
        "num_eval_items": task_manifest["num_eval_items"],
        "eval_ids_regime_invariant": TASK_SPECS[args.task].eval_split_regime_invariant,
        "core_task_name": task_manifest["core_task_name"],
    }
    # The task description is the only part of the prompt stack that is not pinned by
    # the CORE checkout SHA, so an ablation that swaps it leaves no trace anywhere else
    # in this file. Stamp the digest unconditionally -- on a normal run it is the
    # constant that proves the description was *not* touched.
    from hexa_core.prompts_core_tasks import (  # noqa: PLC0415
        CORE_TASK_DESCRIPTIONS,
        MINIMAL_DESCRIPTION_TASKS,
    )
    _desc = CORE_TASK_DESCRIPTIONS[args.task]
    provenance["task_description"] = {
        "sha256": hashlib.sha256(_desc.encode()).hexdigest(),
        "length_chars": len(_desc),
        "minimal_description_ablation": args.task in MINIMAL_DESCRIPTION_TASKS,
        "minimal_description_tasks": list(MINIMAL_DESCRIPTION_TASKS),
    }
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))
    if args.task in MINIMAL_DESCRIPTION_TASKS:
        print(f"\n   *** ABLATION: {args.task} is running with the MINIMAL task "
              f"description ({len(_desc)} ch). This run is NOT comparable to the "
              f"headline runs and must not be pooled with them. ***")
    if not TASK_SPECS[args.task].eval_split_regime_invariant:
        print(f"\n   NOTE: {args.task}'s eval split depends on n_train={args.n_train}; "
              f"pool only with runs whose eval_ids_sha256 is "
              f"{task_manifest['eval_ids_sha256'][:12]}")
    print(f"   reward to evolver: {provenance['config']['reward_fn']}"
          f"   contrast: {args.contrast_selection}x{args.max_trajs}"
          f"   baselines stamped: {result.baselines_stamped}")
    print(f"\n   CORE checkout: {str(provenance['core_git_revision'])[:12]} "
          f"dirty={provenance['core_git_dirty']}"
          + ("   <-- CORE WAS MODIFIED; results are not trustworthy"
             if provenance["core_git_dirty"] else ""))

    print(f"\n   wall clock: {elapsed:.1f} min")
    print(f"   artifacts : {out_dir}")
    print(f"   final bank: {out_dir}/round{max(0, len(result.rounds) - 1)}/bank.json")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
