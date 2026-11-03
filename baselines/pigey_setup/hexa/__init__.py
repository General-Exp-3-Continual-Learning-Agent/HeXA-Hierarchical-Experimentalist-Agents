"""HeXA's skill bank over Pigey: trajectories, teacher, evolving loop, comparison.

Importing this package puts the repository root on ``sys.path`` so that
``skillrl`` and ``hexa_core`` resolve; ``baselines/pigey_setup`` itself must
already be there (every entry script adds it).
"""
import sys
from pathlib import Path

SETUP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SETUP_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))
