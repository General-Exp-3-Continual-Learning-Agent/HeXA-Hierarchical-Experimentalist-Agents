"""Rollout accounting, mirroring CORE's so the two are comparable.

CORE's ``--max-rollouts 2000`` counts *training* generations only. Its counter is
initialised after baseline estimation and advances by
``solve_rollout_count + admission_rollout_count`` per episode
(``scripts/run_core.py:2487-2493``). The ``n_base = 10`` no-memory samples per
training problem are spent *outside* that budget, so a full run really costs
``2000 + 10 * |D_train|`` generations: **2050 / 2100 / 3000** for N = 5 / 10 / 100.
Evaluation calls are on top of that and appear in neither figure.

:meth:`RolloutBudget.report` emits both numbers under names that cannot be mistaken
for each other -- ``training_rollouts`` is the one to quote next to CORE's 2000,
``all_in_total`` is the honest cost including baseline and eval.

Overruns raise :class:`BudgetExhausted` rather than being clamped: a silently
truncated training run looks like a fair comparison and is not one.
"""

from __future__ import annotations

from dataclasses import dataclass


class BudgetExhausted(RuntimeError):
    """A charge would have pushed the gated counter past its cap."""


@dataclass
class RolloutBudget:
    """Generation-call accounting with four separate counters.

    Only ``train_solve + admission`` is gated by :attr:`max_train`, matching CORE.
    ``baseline`` is charged (so it shows up in the honest total) but never gated.
    ``eval`` is gated only if :attr:`max_eval` is set, which is a local guardrail
    and not part of CORE parity.
    """

    max_train: int
    max_eval: int | None = None
    train_solve: int = 0
    baseline: int = 0
    admission: int = 0
    eval: int = 0

    def __post_init__(self) -> None:
        if self.max_train < 0:
            raise ValueError(f"max_train must be >= 0, got {self.max_train}")
        if self.max_eval is not None and self.max_eval < 0:
            raise ValueError(f"max_eval must be >= 0, got {self.max_eval}")

    # -- gated (CORE-comparable) ---------------------------------------------

    @property
    def training_rollouts(self) -> int:
        """The CORE-comparable figure: the number that goes next to ``--max-rollouts``."""
        return self.train_solve + self.admission

    @property
    def train_remaining(self) -> int:
        return max(0, self.max_train - self.training_rollouts)

    def can_afford_train(self, n: int = 1) -> bool:
        """Pre-flight check, so a batch loop can stop at a clean round boundary."""
        if n < 0:
            # A negative n comes from a subtraction that went wrong, and would
            # green-light a charge that then raises ValueError anyway.
            raise ValueError(f"cannot check affordability of a negative charge: {n}")
        return self.training_rollouts + n <= self.max_train

    def charge_train_solve(self, n: int = 1) -> None:
        self._charge_train("train_solve", n)

    def charge_admission(self, n: int = 1) -> None:
        self._charge_train("admission", n)

    def _charge_train(self, counter: str, n: int) -> None:
        if n < 0:
            raise ValueError(f"cannot charge a negative number of rollouts: {n}")
        # Raise *before* incrementing, so the counters stay truthful and a caller
        # that catches this can still write a correct manifest.
        if self.training_rollouts + n > self.max_train:
            raise BudgetExhausted(
                f"charging {n} to {counter} would take training rollouts to "
                f"{self.training_rollouts + n}, past max_train={self.max_train} "
                f"(train_solve={self.train_solve}, admission={self.admission})"
            )
        setattr(self, counter, getattr(self, counter) + n)

    # -- ungated --------------------------------------------------------------

    def charge_baseline(self, n: int = 1) -> None:
        """Charge no-memory baseline samples. Never gated by ``max_train``, per CORE."""
        if n < 0:
            raise ValueError(f"cannot charge a negative number of rollouts: {n}")
        self.baseline += n

    def charge_eval(self, n: int = 1) -> None:
        if n < 0:
            raise ValueError(f"cannot charge a negative number of rollouts: {n}")
        if self.max_eval is not None and self.eval + n > self.max_eval:
            raise BudgetExhausted(
                f"charging {n} eval rollouts would take eval to {self.eval + n}, "
                f"past max_eval={self.max_eval}"
            )
        self.eval += n

    # -- reporting ------------------------------------------------------------

    @property
    def all_in_total(self) -> int:
        """Every generation call made, including baseline and eval."""
        return self.train_solve + self.admission + self.baseline + self.eval

    def report(self) -> dict[str, int | None]:
        return {
            "training_rollouts": self.training_rollouts,
            "max_train": self.max_train,
            "train_remaining": self.train_remaining,
            "train_solve": self.train_solve,
            "admission": self.admission,
            "baseline": self.baseline,
            "eval": self.eval,
            "max_eval": self.max_eval,
            "all_in_total": self.all_in_total,
        }

    def __str__(self) -> str:
        return (
            f"rollouts train={self.training_rollouts}/{self.max_train} "
            f"(solve={self.train_solve} admission={self.admission}) "
            f"baseline={self.baseline} eval={self.eval} all-in={self.all_in_total}"
        )
