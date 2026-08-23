"""HExA Design B on ORBIT's Maze environment.

Glue only. ORBIT's maze generator, transition rules, termination conditions and
episode-success rule are imported in place from the ORBIT checkout and never
forked -- a reimplemented wall-collision rule or shortest-path filter would
silently invalidate every number in a HExA-vs-ORBIT comparison.

Scope is set by ``DECISIONS.md``: Maze only (D1 dropped Mastermind over its
120-code hidden-state space), exactly three episodes per instance (D2), ORBIT
imported behind a two-object rLLM shim rather than a verl install (D4), serial
execution (D5), and instance identity keyed to the resolved maze layout rather
than the dataset seed (D6).

Modules, in dependency order:

* :mod:`~hexa_orbit.rllm_shim` -- the two rLLM names ORBIT's envs import (D4).
* :mod:`~hexa_orbit.orbit_bridge` -- locate the checkout, install the shim, skip
  ORBIT's GEM-dependent ``envs/__init__``.
* :mod:`~hexa_orbit.pools` -- ORBIT's seed derivation, instance identity,
  train/eval disjointness and the leaked/clean partition.
* :mod:`~hexa_orbit.adapter` -- ``MazeTask`` / ``MazeInstance``: one fixed maze
  played for three episodes, with an observation stream byte-identical to
  ``MultiEpisodeEnv``'s.
* :mod:`~hexa_orbit.oracle` -- known-map BFS optimum, for regret.

Imports are cheap: nothing here touches ``sys.path`` or the ORBIT checkout at
import time. ``orbit_bridge.install_orbit_on_path`` is called lazily by the code
that needs it.
"""

from __future__ import annotations
