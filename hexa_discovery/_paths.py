"""Path bootstrap so the bridge can import both `skillrl` (HeXA core) and
`scienceagent` (DiscoverPhysics) without either being pip-installed.

Layout:
    HeXA/                     <- HEXA_ROOT  (contains skillrl/ and hexa_discovery/)
    HeXA/discovery-agents/    <- DISCOVERY_ROOT  (upstream DiscoverPhysics submodule;
                                 must be CWD while agents run, since scienceagent
                                 loads prompts by relative path -- the runners
                                 switch into it themselves)
    HeXA/discovery-agents/ScienceAgent/   <- scienceagent package lives here
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
HEXA_ROOT = _HERE.parents[1]               # HeXA
DISCOVERY_ROOT = HEXA_ROOT / "discovery-agents"
SCIENCEAGENT_DIR = DISCOVERY_ROOT / "ScienceAgent"
PHYSSCHOOL_DIR = DISCOVERY_ROOT / "PhysicsSchool"


def bootstrap() -> None:
    """Insert HeXA root and ScienceAgent on sys.path (idempotent)."""
    for p in (HEXA_ROOT, SCIENCEAGENT_DIR):
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)
