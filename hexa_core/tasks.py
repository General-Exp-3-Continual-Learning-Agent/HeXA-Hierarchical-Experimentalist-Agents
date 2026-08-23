"""Registry of the three CORE tasks in scope, with the paper's exact constructor args.

Construction goes through CORE's own factory, ``tasks.single_turn.get_task`` --
the same entry point ``scripts/run_core.py`` uses (``init_task``, line 3658). We
therefore inherit CORE's mapping from argument names to constructor keywords, its
lazy module imports and its per-task defaults (matchstick's
``dataset="default"``), rather than re-deriving any of them here. What *is* ours
is the set of argument *values* below: they are transcribed from CORE's own shell
scripts and are the published configuration, so they are not tunable knobs and
live in one place only.

Note that CORE's ``get_task`` defaults are **not** the paper config -- it defaults
Hanoi to ``train_disks=[3,4,5,6], eval_disks=[7,8]`` -- which is exactly why the
values have to be passed explicitly and why they are pinned here.

``num_train_items`` is the regime (5 / 10 / 100); eval is always 100 items and its
id list is byte-identical across regimes for a given task.

Names are both our stable registry keys and CORE's ``get_task`` names. Note that
``"tower-of-hanoi"`` yields a CORE instance whose own ``.name`` is
``"tower-of-hanoi-state"`` (``variant="state"``); the two must not be conflated.
See :meth:`hexa_core.adapter.CoreTask.manifest`.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

from .adapter import EVAL_ITEMS, CoreTask
from .core_bridge import install_core_on_path

TASK_NAMES: tuple[str, ...] = ("matchstick", "mathgap", "zebralogic", "tower-of-hanoi")

#: Training-set sizes the paper grid sweeps. Other values are allowed by
#: :func:`make_task` (smoke tests want 2), they are just not the published grid.
REGIMES: tuple[int, ...] = (5, 10, 100)


@dataclass(frozen=True)
class TaskSpec:
    """The paper's arguments for one CORE task, minus the regime.

    ``core_kwargs`` is written in ``get_task``'s vocabulary (``hanoi_train_disks``,
    ``zebralogic_size_group``, ...), not the underlying constructors', so it can be
    splatted straight into CORE's factory with no translation step of ours in
    between. Keys omitted here fall through to CORE's own defaults on purpose --
    matchstick's ``matchstick_dataset`` is one such, exactly as in ``init_task``.
    """

    name: str
    core_kwargs: Mapping[str, Any] = field(default_factory=dict)
    #: Whether ``seed`` actually changes the split. ``get_task`` accepts it for
    #: every task and CORE's driver always passes it, but only Hanoi's state-space
    #: sampler consumes it -- matchstick, MathGAP and ZebraLogic have
    #: seed-independent splits (checked-in files / a hard-coded
    #: ``random.Random(0)`` shuffle).
    seed_affects_split: bool = False
    #: Whether the *eval* id list is the same at every regime. True for three of
    #: the four tasks, whose eval sets come from a fixed file, a fixed shuffle
    #: slice or a disjoint disk count, and so do not move when ``num_train_items``
    #: changes. **False for MathGAP**, which slices one shuffled pool as
    #: ``train = data[:n]`` / ``eval = data[n : n + 100]``
    #: (``mathgap_task.py:163-164``): raising the regime slides the eval window
    #: forward, so MathGAP numbers are comparable *within* a regime and not
    #: across. Splits stay disjoint either way, so this is a comparability
    #: constraint, not a contamination risk. Asserted both ways in
    #: ``tests/test_splits.py``.
    eval_split_regime_invariant: bool = True
    #: Size of the pool both splits are carved out of, where that pool is small
    #: enough to bind. MathGAP ships 280 rows, so ``num_train_items`` above 180
    #: cannot yield 100 eval items; ``CoreTask`` catches the shortfall, and this
    #: turns it into an error that names the cause. ``None`` means unconstrained
    #: for the regimes we run.
    pool_size: int | None = None
    #: Whether CORE leaves ``output_instruction`` empty because the answer-format
    #: line is already baked into each problem's own text upstream. True only for
    #: MathGAP, whose prompt template is the bare ``{problem}`` and which never
    #: overrides the base class's ``""``. Recorded here rather than in the tests so
    #: the several places that assert on ``output_instruction`` share one source of
    #: truth, and so a future CORE release that adds a real instruction fails
    #: loudly instead of silently changing every MathGAP prompt.
    output_instruction_empty: bool = False


TASK_SPECS: Mapping[str, TaskSpec] = MappingProxyType(
    {
        "matchstick": TaskSpec(name="matchstick"),
        # MathGAP takes no task-specific kwargs at all: CORE's `get_task` forwards
        # only the two split sizes (`__init__.py:38-43`), and the constructor's own
        # defaults (`num_eval_items=40`) are *not* the paper config, which is why
        # 100 is always passed explicitly by `make_task`.
        "mathgap": TaskSpec(
            name="mathgap",
            eval_split_regime_invariant=False,
            pool_size=280,
            output_instruction_empty=True,
        ),
        "zebralogic": TaskSpec(
            name="zebralogic",
            core_kwargs=MappingProxyType({"zebralogic_size_group": "xlarge"}),
        ),
        "tower-of-hanoi": TaskSpec(
            name="tower-of-hanoi",
            core_kwargs=MappingProxyType(
                {
                    "hanoi_variant": "state",
                    "hanoi_train_disks": [5],
                    "hanoi_eval_disks": [6],
                    "hanoi_enforce_optimal": False,
                    "hanoi_min_distance": 1,
                }
            ),
            seed_affects_split=True,
        ),
    }
)


def make_task(name: str, num_train_items: int, seed: int = 42) -> CoreTask:
    """Construct a CORE task at the given regime and wrap it as a :class:`CoreTask`.

    ``seed`` is always forwarded, mirroring ``init_task``, but only reaches a
    sampler for ``tower-of-hanoi``; see :attr:`TaskSpec.seed_affects_split`.
    """
    if name not in TASK_SPECS:
        raise ValueError(f"Unknown task {name!r}. Known tasks: {list(TASK_NAMES)}")
    if not isinstance(num_train_items, int) or isinstance(num_train_items, bool):
        raise ValueError(f"num_train_items must be an int, got {num_train_items!r}")
    if num_train_items <= 0:
        raise ValueError(f"num_train_items must be positive, got {num_train_items}")

    spec = TASK_SPECS[name]
    # CORE soft-degrades an over-large request: `_prepare_data` prints "Warning: Not
    # enough evaluation items found" and returns a short split. `CoreTask` asserts
    # the size and would catch it, but reports only the shortfall; checking the pool
    # here names the actual cause.
    if spec.pool_size is not None and num_train_items + EVAL_ITEMS > spec.pool_size:
        raise ValueError(
            f"{name}: num_train_items={num_train_items} leaves "
            f"{spec.pool_size - num_train_items} rows for eval, fewer than the "
            f"{EVAL_ITEMS} required -- its train and eval splits are consecutive "
            f"slices of one {spec.pool_size}-row pool, so num_train_items must be "
            f"<= {spec.pool_size - EVAL_ITEMS}"
        )

    install_core_on_path()
    import tasks.single_turn as core_single_turn

    # deepcopy so a CORE constructor that normalises a list in place (Hanoi's
    # `_normalize_disk_list` currently does not, but it is one edit away from it)
    # cannot poison the shared TASK_SPECS entry for later calls.
    kwargs: dict[str, Any] = {k: copy.deepcopy(v) for k, v in spec.core_kwargs.items()}
    core_task = core_single_turn.get_task(
        spec.name,
        num_train_items=num_train_items,
        num_eval_items=EVAL_ITEMS,
        seed=seed,
        **kwargs,
    )
    return CoreTask(
        name=name,
        core_task=core_task,
        expected_train_items=num_train_items,
        expected_eval_items=EVAL_ITEMS,
    )
