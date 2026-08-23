"""Runnable checks for `hexa_core`. No pytest: `python -m hexa_core.tests.<module>`.

pytest is not installed in this environment, so every suite here is a plain module
of ``def test_*()`` functions plus a ``__main__`` driver. Both entry points work::

    PYTHONDONTWRITEBYTECODE=1 python -m hexa_core.tests.test_splits
    pytest hexa_core/tests            # if pytest ever lands

The older suites (``test_foundation``, ``test_llm_clients``, ``test_logging_schema``,
``test_hexa_seam``, ``test_baseline_runner``) predate this harness and drive
themselves from ``main()``. The adapter / splits / verifiers suites use the
helpers below, which give per-check progress lines and keep going after a failure
so one broken invariant does not hide the rest.

Nothing in this package makes a network call or touches a GPU. Suites that need
CORE construct tasks in process (all nine task x regime combinations cost ~0.1 s)
and exercise pure functions only.
"""

from __future__ import annotations

import importlib
import time
import traceback
from functools import lru_cache
from types import ModuleType
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:  # avoid importing CORE (or even hexa_core.tasks) at module scope
    from ..adapter import CoreTask

__all__ = [
    "CheckFailed",
    "check",
    "check_eq",
    "check_raises",
    "run_module",
    "cached_task",
    "fresh_task",
    "core_module",
]


class CheckFailed(AssertionError):
    """A failed check. Subclasses AssertionError so pytest renders it normally."""


def check(label: str, condition: object, detail: str = "") -> None:
    """Assert ``condition``, printing one progress line either way."""
    if condition:
        print(f"    ok   {label}")
        return
    print(f"    FAIL {label}" + (f" -- {detail}" if detail else ""))
    raise CheckFailed(label if not detail else f"{label}: {detail}")


def check_eq(label: str, actual: Any, expected: Any, *, show: int = 240) -> None:
    """Assert equality; on mismatch report truncated reprs and, for strings, the
    first differing offset (a byte-identical prompt failure is unreadable otherwise)."""
    if actual == expected:
        print(f"    ok   {label}")
        return

    detail = f"actual={_clip(actual, show)} expected={_clip(expected, show)}"
    if isinstance(actual, str) and isinstance(expected, str):
        offset = next(
            (i for i, (a, b) in enumerate(zip(actual, expected)) if a != b),
            min(len(actual), len(expected)),
        )
        detail = (
            f"first difference at offset {offset} "
            f"(len {len(actual)} vs {len(expected)})\n"
            f"      actual  ...{actual[max(0, offset - 40):offset + 40]!r}\n"
            f"      expected...{expected[max(0, offset - 40):offset + 40]!r}"
        )
    print(f"    FAIL {label} -- {detail}")
    raise CheckFailed(f"{label}: {detail}")


def check_raises(
    label: str,
    exc: type[BaseException],
    fn: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> BaseException:
    """Assert ``fn(*args, **kwargs)`` raises ``exc``; return it so the caller can
    assert on the message."""
    try:
        result = fn(*args, **kwargs)
    except exc as caught:
        print(f"    ok   {label}")
        return caught
    except BaseException as wrong:  # noqa: BLE001 -- reported, then re-raised as a failure
        detail = f"raised {type(wrong).__name__}({wrong}) instead of {exc.__name__}"
        print(f"    FAIL {label} -- {detail}")
        raise CheckFailed(f"{label}: {detail}") from wrong
    detail = f"returned {_clip(result, 120)} instead of raising {exc.__name__}"
    print(f"    FAIL {label} -- {detail}")
    raise CheckFailed(f"{label}: {detail}")


def _clip(value: Any, limit: int) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[:limit] + f"... (+{len(text) - limit} chars)"


def run_module(module: ModuleType) -> int:
    """Run every ``test_*`` callable in ``module`` in source order.

    Returns 0 if all passed, 1 otherwise -- the exit code for ``python -m``.
    Tests are independent, so a failure is recorded and the run continues.
    """
    tests = [
        (name, fn)
        for name, fn in vars(module).items()
        # `__module__` filters out helpers imported from this package.
        if name.startswith("test_")
        and callable(fn)
        and getattr(fn, "__module__", None) == module.__name__
    ]
    tests.sort(key=lambda pair: getattr(getattr(pair[1], "__code__", None), "co_firstlineno", 0))

    started = time.perf_counter()
    failed: list[str] = []
    for name, fn in tests:
        print(f"-- {name}")
        try:
            fn()
        except CheckFailed:
            failed.append(name)
        except BaseException:  # noqa: BLE001 -- a crashing test is a failing test
            traceback.print_exc()
            failed.append(name)

    elapsed = time.perf_counter() - started
    print(f"\n{len(tests) - len(failed)} passed, {len(failed)} failed in {elapsed:.2f}s")
    if failed:
        print("failed: " + ", ".join(failed))
        return 1
    return 0


def fresh_task(name: str, num_train_items: int, seed: int = 42) -> "CoreTask":
    """A brand-new task instance. Required by any check that must not see a cache."""
    from ..tasks import make_task  # imported lazily: pulls CORE onto sys.path

    return make_task(name, num_train_items, seed=seed)


@lru_cache(maxsize=None)
def cached_task(name: str, num_train_items: int, seed: int = 42) -> "CoreTask":
    """Shared task instance, so nine constructions are not repeated per test function."""
    return fresh_task(name, num_train_items, seed=seed)


def core_module(dotted: str) -> ModuleType:
    """Import a CORE module (e.g. ``tasks.single_turn.zebralogic_task``) by name."""
    from ..core_bridge import install_core_on_path

    install_core_on_path()
    return importlib.import_module(dotted)
