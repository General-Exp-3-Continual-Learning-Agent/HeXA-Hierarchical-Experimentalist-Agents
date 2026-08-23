"""Two silent-failure guards, each pinned in both directions.

Both defects here shared a shape: the code kept running and produced a number that
looked fine, so nothing downstream could notice. That makes them unusually
important to pin, and pinning them requires negative controls — a test that only
checks "the guard fires" would still pass if the guard fired *always*, which for
the second one would mean a warning on every problem of every run.

1. :func:`hexa_core.core_bridge.core_repo_root` must treat an explicitly set
   ``$CORE_REPO`` as authoritative. It previously fell through to the sibling
   ``../core-reasoning`` whenever ``$CORE_REPO`` was a typo or a stale/partial
   clone, so the run measured a *different* CORE than it declared and every
   manifest recorded the sibling's ``core_git_revision``.

2. :func:`hexa_core.hexa_bridge.retrieve_within_token_budget` must say something
   when a populated bank is budgeted down to an empty block. An empty block makes
   ``Problem.build_prompt(block)`` byte-identical to ``build_prompt(None)`` (every
   CORE template gates the guidelines slot on truthiness), so a memory-augmented
   run silently becomes its own no-memory baseline and the experiment reports "no
   gain from HeXA" from a config typo.

The ``$CORE_REPO`` checks run in subprocesses: ``core_repo_root`` is
``lru_cache``-d, so mutating the environment in process would leak a poisoned
cache into every later suite sharing the interpreter.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import tempfile
from pathlib import Path

from . import check, check_eq, run_module

_REPO_ROOT = Path(__file__).resolve().parents[2]

_RESOLVE = (
    "from hexa_core import core_bridge\n"
    "try:\n"
    "    print('OK', core_bridge.core_repo_root())\n"
    "except core_bridge.CoreRepoNotFound as exc:\n"
    "    print('RAISED', exc)\n"
)


def _resolve_with(core_repo: str | None) -> str:
    """Resolve the CORE root in a fresh interpreter with ``$CORE_REPO`` set or unset."""
    env = {
        "PATH": "/usr/bin:/bin",
        "PYTHONPATH": str(_REPO_ROOT),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    if core_repo is not None:
        env["CORE_REPO"] = core_repo
    out = subprocess.run(
        [sys.executable, "-c", _RESOLVE],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        check=False,
    )
    if out.returncode != 0:
        raise AssertionError(f"probe crashed: {out.stderr[-800:]}")
    return out.stdout.strip()


def test_unset_core_repo_still_finds_the_sibling_checkout() -> None:
    """Negative control: the fallback must survive the fix, or nothing else runs."""
    result = _resolve_with(None)
    check("unset $CORE_REPO resolves to the sibling checkout", result.startswith("OK "), result)
    check(
        "resolved root is the real core-reasoning",
        result.endswith("core-reasoning"),
        result,
    )


def test_valid_core_repo_is_honoured() -> None:
    """Second negative control: an explicit, correct value must still be accepted."""
    sibling = _REPO_ROOT.parent / "core-reasoning"
    result = _resolve_with(str(sibling))
    check(f"explicit $CORE_REPO={sibling} accepted", result == f"OK {sibling}", result)


def test_empty_core_repo_is_treated_as_unset() -> None:
    result = _resolve_with("")
    check("$CORE_REPO='' falls back rather than raising", result.startswith("OK "), result)


def test_nonexistent_core_repo_is_fatal() -> None:
    """The typo case: must NOT silently resolve to the sibling."""
    result = _resolve_with("/nonexistent-core-checkout")
    check("nonexistent $CORE_REPO raises", result.startswith("RAISED"), result)
    check("message names the offending value", "/nonexistent-core-checkout" in result, result)
    check("message says why it did not fall back", "Refusing to fall back" in result, result)
    check("message states the specific defect", "no such directory" in result, result)


def test_partial_core_checkout_is_fatal() -> None:
    """The stale/partial-clone case: has src/tasks/single_turn, nothing else."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        (Path(tmp) / "src" / "tasks" / "single_turn").mkdir(parents=True)
        result = _resolve_with(tmp)
    check("partial $CORE_REPO raises", result.startswith("RAISED"), result)
    check(
        "message lists the missing markers",
        "scripts/run_core.py" in result
        and "src/reflection/single_turn_prompt_templates.py" in result,
        result,
    )


def test_core_repo_pointing_at_this_repo_is_fatal() -> None:
    """The plausible mix-up — $CORE_REPO set to the HeXA checkout — must not pass."""
    result = _resolve_with(str(_REPO_ROOT))
    check("$CORE_REPO=<HeXA root> raises", result.startswith("RAISED"), result)


# -- guard 2: a bank budgeted down to nothing -----------------------------


def _bank(n_skills: int = 3, n_mistakes: int = 3, level: str = "tower-of-hanoi"):
    from skillrl.core.skill_bank import Mistake, Skill, SkillBank

    bank = SkillBank()
    for i in range(n_skills):
        bank.add_skill(Skill(
            skill_id=f"s{i}",
            title=f"Strategy {i}",
            principle="Restate the peg contents after every move. " * 6,
            when_to_apply="Whenever the goal is a non-canonical position. " * 3,
            source_level=level,
            source_type="contrastive",
            confidence=0.9 - i * 0.1,
        ))
    for i in range(n_mistakes):
        bank.add_mistake(Mistake(
            mistake_id=f"m{i}",
            description="Lifted a disk from under a smaller one. " * 3,
            why_it_happens="Peg state was tracked loosely. " * 3,
            how_to_avoid="Re-state all three stacks each move. " * 3,
            source_level=level,
        ))
    return bank


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def _retrieve(**kwargs) -> tuple[object, list[str]]:
    """Call the retriever, returning its result and any warnings it logged."""
    from hexa_core import hexa_bridge

    capture = _Capture()
    logger = logging.getLogger(hexa_bridge.__name__)
    logger.addHandler(capture)
    previous = logger.propagate
    logger.propagate = False
    try:
        result = hexa_bridge.retrieve_within_token_budget(**kwargs)
    finally:
        logger.removeHandler(capture)
        logger.propagate = previous
    return result, capture.messages


def test_starved_budget_warns_and_reports_the_drop() -> None:
    result, warnings = _retrieve(bank=_bank(), task_name="tower-of-hanoi", budget_tokens=40)
    check("nothing was injected", result.block_text == "" and result.tokens == 0)
    check("the drop is visible in the return value", result.budget_bound)
    check_eq("every candidate is counted as dropped",
             (result.dropped_skills, result.dropped_mistakes), (3, 3))
    check("exactly one warning", len(warnings) == 1, repr(warnings))
    message = warnings[0]
    for fragment in ("budget 40 tokens", "NOTHING is injected", "no-memory baseline",
                     "skip_oversized", "tower-of-hanoi"):
        check(f"warning mentions {fragment!r}", fragment in message, message)


def test_overhead_starvation_warns() -> None:
    """The subtler config error: the budget is generous, the reservation eats it."""
    result, warnings = _retrieve(bank=_bank(), task_name="tower-of-hanoi",
                                 budget_tokens=1000, overhead_tokens=995)
    check("nothing injected", result.block_text == "")
    check("warned", len(warnings) == 1, repr(warnings))
    check("warning reports the overhead", "overhead 995" in warnings[0], warnings[0])


def test_healthy_retrieval_is_silent() -> None:
    """Negative control: a guard that always fires is not a guard."""
    result, warnings = _retrieve(bank=_bank(), task_name="tower-of-hanoi",
                                 budget_tokens=100_000)
    check("everything fit", len(result.skills) == 3 and len(result.mistakes) == 3)
    check("nothing was dropped", not result.budget_bound)
    check("no warning", warnings == [], repr(warnings))


def test_partial_drop_is_silent() -> None:
    """A truncated-but-non-empty block is normal operation, reported structurally."""
    result, warnings = _retrieve(bank=_bank(), task_name="tower-of-hanoi", budget_tokens=250)
    check("a non-empty block came back", bool(result.block_text))
    check("some candidates were dropped", result.budget_bound)
    check("no warning for a partial drop", warnings == [], repr(warnings))


def test_explicit_zero_budget_is_silent() -> None:
    """``budget_tokens<=0`` is how the no-memory ablation arm asks for nothing.

    Warning here would emit one line per problem per run — 300 lines for a single
    3-run gate — and train the reader to ignore the warning that matters.
    """
    for budget in (0, -1):
        result, warnings = _retrieve(bank=_bank(), task_name="tower-of-hanoi",
                                    budget_tokens=budget)
        check(f"budget_tokens={budget} injects nothing", result.block_text == "")
        check(f"budget_tokens={budget} still reports the drop", result.budget_bound)
        check(f"budget_tokens={budget} is silent", warnings == [], repr(warnings))


def test_empty_bank_and_none_bank_are_silent() -> None:
    """No candidates means no drop: there is nothing to warn about."""
    from skillrl.core.skill_bank import SkillBank

    result, warnings = _retrieve(bank=SkillBank(), task_name="tower-of-hanoi",
                                 budget_tokens=10)
    check("empty bank is silent", warnings == [], repr(warnings))
    check("empty bank reports no drop", not result.budget_bound)

    result, warnings = _retrieve(bank=None, task_name="tower-of-hanoi", budget_tokens=10)
    check("bank=None is silent", warnings == [], repr(warnings))
    check("bank=None returns an empty block", result.block_text == "")


def test_empty_block_is_indistinguishable_from_no_memory() -> None:
    """Why guard 2 matters at all, asserted against a real CORE prompt."""
    from . import cached_task

    problem = cached_task("tower-of-hanoi", 5).eval_problems()[0]
    starved, _ = _retrieve(bank=_bank(), task_name="tower-of-hanoi", budget_tokens=40)
    check_eq("build_prompt('') == build_prompt(None)",
             problem.build_prompt(starved.block_text), problem.build_prompt(None))
    healthy, _ = _retrieve(bank=_bank(), task_name="tower-of-hanoi", budget_tokens=100_000)
    check("a real block does change the prompt",
          problem.build_prompt(healthy.block_text) != problem.build_prompt(None))


def main() -> int:
    return run_module(sys.modules[__name__])


if __name__ == "__main__":
    raise SystemExit(main())
