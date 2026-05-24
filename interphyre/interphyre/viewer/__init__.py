"""Interphyre Viewer - Visualization tools for physics puzzles.

This module provides tools for visualizing levels, solutions, and running demos.

Usage:
    # As CLI
    python -m interphyre.viewer catapult --seed 42 --action 0.5 3.0 0.6

    # As module
    from interphyre.viewer import view_action
    view_action("catapult", 42, (0.5, 3.0, 0.6))
"""

from interphyre.viewer._viewer import (
    run_random_demo,
    view_action,
    view_bundle_solution,
    view_solutions_from_file,
)

__all__ = [
    "view_action",
    "view_bundle_solution",
    "view_solutions_from_file",
    "run_random_demo",
]
