"""CLI entry point for interphyre.viewer module.

Usage:
    python -m interphyre.viewer catapult --seed 42 --action 0.5 3.0 0.6
"""

import sys
import os

# Ensure project root is in path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

# Import and run the main CLI from _viewer
from interphyre.viewer._viewer import main

if __name__ == "__main__":
    main()
