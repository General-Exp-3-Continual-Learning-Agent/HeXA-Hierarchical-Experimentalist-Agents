"""Central configuration for the SkillRL pipeline."""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent          # physics-reasoning-agents/
SKILLRL_DIR = Path(__file__).resolve().parent               # skillrl/
DATA_DIR = SKILLRL_DIR / "data"
SKILL_BANK_PATH = DATA_DIR / "skill_bank.json"
SEED_TRAJ_DIR = DATA_DIR / "seed_trajectories"
EVAL_RESULTS_DIR = DATA_DIR / "eval_results"
EVOLUTION_LOG_DIR = DATA_DIR / "evolution_logs"

# Existing trajectory locations (already collected by the user)
EXISTING_RESULTS = {
    "basket_case": BASE_DIR / "results" / "basket_case",
    "catapult": BASE_DIR / "results" / "catapult_claude",
    "falling_into_place": BASE_DIR / "results" / "falling_into_place_claude",
    "pass_the_parcel": BASE_DIR / "results" / "pass_the_parcel",
    "two_body_problem": BASE_DIR / "All Results" / "results_full" / "results_claude_two_body",
    "down_to_earth": BASE_DIR / "All Results" / "results_full" / "DTE_Full" / "results_down_to_earth_14B",
}

# ── Levels ─────────────────────────────────────────────────────────────
ALL_LEVELS = [
    "down_to_earth",
    "two_body_problem",
    "catapult",
    "falling_into_place",
    "basket_case",
    "pass_the_parcel",
    "cliffhanger"
]

# ── Level descriptions (provided to teacher for context during distillation) ──
LEVEL_DESCRIPTIONS = {
    "down_to_earth": (
        "A green ball falls onto a raised black platform, with gaps between "
        "the platform and the walls on either side. Without intervention it "
        "stays on the platform. The agent places one red ball. Success "
        "requires the green ball to touch the purple ground for 3+ seconds."
    ),
    "two_body_problem": (
        "A green ball and a blue ball start separated horizontally, both "
        "dynamic and falling under gravity from rest. Without intervention "
        "they fall straight down and never meet. The agent places one red "
        "ball. Success requires green-blue contact for 3+ seconds."
    ),
    "catapult": (
        "A gray bar rests on a gray pivot ball on the left platform, with a "
        "green ball on its left end. A blue ball sits in a basket on the "
        "right ledge. The agent places one red ball. Success requires "
        "green-blue contact for 3+ seconds."
    ),
    "falling_into_place": (
        "A green ball sits on one of two black platforms separated by a gap, "
        "and a blue jar (opening facing down) is positioned above and falls "
        "under gravity. The agent places one red ball. Success requires "
        "green-blue contact for 3+ seconds."
    ),
    "basket_case": (
        "A green ball is positioned directly above an upright basket near the "
        "purple ground; without intervention it falls into the basket. The "
        "agent places one red ball. Success requires the green ball to touch "
        "the purple ground for 3+ seconds."
    ),
    "pass_the_parcel": (
        "A green ball is trapped under an inverted basket on a black platform, "
        "with a ramp rising from the platform's left edge. A blue ball sits in "
        "an upright basket below the platform. The agent places one red ball. "
        "Success requires green-blue contact for 3+ seconds."
    ),
}

# ── Evolution parameters ───────────────────────────────────────────────
ACCURACY_THRESHOLD = 0.5       # Levels below this trigger skill evolution
MAX_EVOLUTION_ROUNDS = 3
MAX_SKILLS_PER_LEVEL = 10
MAX_GENERAL_SKILLS = 15

# ── Teacher model ──────────────────────────────────────────────────────
TEACHER_MODEL = "claude-sonnet-4-6"

# ── Agent defaults ─────────────────────────────────────────────────────
DEFAULT_MAX_ITERATIONS = 25
DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_NEW_TOKENS = 800
