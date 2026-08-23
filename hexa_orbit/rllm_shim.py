"""Minimal stand-ins for the two rLLM objects ORBIT's environments import.

ORBIT's ``envs/*.py`` are the single source of truth for maze generation,
transition rules, termination conditions and episode/success semantics, so they
are imported in place and never forked (RECON §7). Importing them, however,
drags in ``rllm`` -- and ORBIT pins ``rllm`` to its own fork on branch
``fix/token-in-token-out``, installed as ``-e .[verl]``, which pulls
verl/vLLM/torch (RECON §0a).

The entire rLLM surface those modules touch is two names::

    from rllm.agents.agent import Action                     # a value wrapper
    from rllm.environments.base.base_env import BaseEnv      # an ABC

That is DECISIONS.md D4: shim the two, skip the install. :func:`install`
registers synthetic ``rllm.*`` modules in :data:`sys.modules` so the real
imports resolve to the definitions below.

**What this shim is and is not.** It is an import-compatibility layer, not a
reimplementation of rLLM. Nothing here participates in maze rules or reward
logic -- ``Action`` is unwrapped by ``isinstance`` checks and ``BaseEnv`` is
never called through. If a future ORBIT revision starts *using* rLLM behaviour
(an engine, a tokenizer, a trajectory type), this shim stops being sufficient
and :func:`install` will fail loudly on the new import rather than silently
supplying a wrong object -- which is why it registers only these two modules and
does not create a catch-all package.

``install()`` is idempotent and refuses to shadow a real ``rllm`` if one is
importable, so a machine that later does install ORBIT's fork gets the real
thing without a code change here.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

#: Set by :func:`install` to record which implementation is in play, so a run
#: manifest can state "shimmed" vs "real rllm" instead of leaving it ambiguous.
INSTALLED_BY_SHIM = "shim"
INSTALLED_REAL = "real"


@dataclass
class Action:
    """Wrapper around one action payload.

    ORBIT only ever constructs ``Action(action=...)`` and reads ``.action``, via
    the ``raw_action = action.action if isinstance(action, Action) else action``
    idiom (``envs/maze_env_adapter.py:488``, ``envs/multi_episode_env.py:192``,
    ``envs/gem_env_adapter.py:80``). Our own actor passes bare strings, so the
    ``isinstance`` branch is not even taken -- the class exists so the import
    resolves and so a caller that *does* wrap stays supported.

    ``dataclass`` rather than ``NamedTuple``: rLLM's own ``Action`` carries
    mutable auxiliary fields, and a tuple subclass would make
    ``isinstance(action, Action)`` true for unrelated tuples.
    """

    action: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseEnv(ABC):
    """The environment contract ORBIT's adapters subclass.

    Only ``reset`` and ``step`` are abstract, because those are the only methods
    every ORBIT adapter defines. ``close`` and ``from_dict`` get defaults so a
    partially-implemented adapter still instantiates -- ORBIT's own code calls
    both defensively (``hasattr(self.inner_env, "close")``,
    ``envs/multi_episode_env.py:307``).

    ``__init__`` accepts and ignores arguments so that ``MultiEpisodeEnv``'s
    ``super().__init__()`` (``envs/multi_episode_env.py:58``) works, and so would
    a subclass that forwards constructor arguments upward.

    Signatures are deliberately loose (``Any``): pinning them to ORBIT's current
    shapes would make this shim a second, competing specification of an interface
    we do not own.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: D107
        pass

    @abstractmethod
    def reset(self, *args: Any, **kwargs: Any) -> Tuple[Any, dict]:
        """Return ``(observation, info)``."""

    @abstractmethod
    def step(self, action: Any) -> Tuple[Any, float, bool, dict]:
        """Return ``(observation, reward, done, info)``."""

    def close(self) -> None:
        return None

    @staticmethod
    def from_dict(info: dict) -> "BaseEnv":
        raise NotImplementedError(
            "from_dict is not provided by the rllm shim; the concrete ORBIT "
            "adapter is expected to override it."
        )

    @staticmethod
    def is_multithread_safe() -> bool:
        return False


def _real_rllm_available() -> bool:
    """True if a genuine ``rllm`` is importable and exposes what ORBIT needs."""
    if "rllm" in sys.modules and getattr(sys.modules["rllm"], "__shimmed__", False):
        return False
    try:
        if importlib.util.find_spec("rllm.environments.base.base_env") is None:
            return False
        return importlib.util.find_spec("rllm.agents.agent") is not None
    except (ImportError, ValueError, ModuleNotFoundError):
        return False


def _module(name: str, **attrs: Any) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__shimmed__ = True  # type: ignore[attr-defined]
    for key, value in attrs.items():
        setattr(mod, key, value)
    return mod


def install() -> str:
    """Make ``rllm.agents.agent`` and ``rllm.environments.base.base_env`` importable.

    Returns :data:`INSTALLED_REAL` when a real ``rllm`` is present and left
    alone, :data:`INSTALLED_BY_SHIM` when the stubs were registered. Idempotent.
    """
    if _real_rllm_available():
        return INSTALLED_REAL
    if getattr(sys.modules.get("rllm"), "__shimmed__", False):
        return INSTALLED_BY_SHIM

    # Parent packages must exist and must declare their children as attributes,
    # or `from rllm.agents.agent import Action` fails on attribute lookup even
    # with every module registered in sys.modules.
    agents = _module("rllm.agents")
    agent_mod = _module("rllm.agents.agent", Action=Action)
    agents.agent = agent_mod  # type: ignore[attr-defined]

    environments = _module("rllm.environments")
    base_pkg = _module("rllm.environments.base")
    base_env_mod = _module("rllm.environments.base.base_env", BaseEnv=BaseEnv)
    base_pkg.base_env = base_env_mod  # type: ignore[attr-defined]
    environments.base = base_pkg  # type: ignore[attr-defined]

    root = _module("rllm", agents=agents, environments=environments)

    sys.modules.update(
        {
            "rllm": root,
            "rllm.agents": agents,
            "rllm.agents.agent": agent_mod,
            "rllm.environments": environments,
            "rllm.environments.base": base_pkg,
            "rllm.environments.base.base_env": base_env_mod,
        }
    )
    return INSTALLED_BY_SHIM
