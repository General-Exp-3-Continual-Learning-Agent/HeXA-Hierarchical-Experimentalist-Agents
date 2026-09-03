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
        "A green ball sits on a raised platform with gaps on either side. "
        "The agent must place a red ball so it collides with the green ball "
        "and pushes it off the platform edge, through a gap, down to the "
        "purple ground below. Success requires the green ball to touch the "
        "purple ground for 3+ seconds. Key physics: collision impulse "
        "direction, platform edge geometry, gap width analysis."
    ),
    "two_body_problem": (
        "A green ball and a blue ball start separated horizontally, both "
        "dynamic and falling under gravity. Without intervention they fall "
        "straight down and never meet. The agent must place a red ball so "
        "that it falls onto the green ball and pushes it horizontally into "
        "the blue ball. Success requires green-blue contact for 3+ seconds. "
        "Key physics: collision angle (along center-to-center line), "
        "horizontal offset controls push direction, radius controls mass."
    ),
    "catapult": (
        "A lever (bar) rests on a pivot with a green ball on one end. "
        "A blue ball sits in a basket at a distance. The agent must drop "
        "a red ball onto the other end of the lever to catapult the green "
        "ball through an arc into the basket where the blue ball is. "
        "Success requires green-blue contact for 3+ seconds. Key physics: "
        "lever arm ratio controls launch angle, drop position relative to "
        "pivot matters more than ball mass/radius, arc height vs distance "
        "tradeoff."
    ),
    "falling_into_place": (
        "A green ball sits on a platform and a blue jar is positioned above, "
        "ready to fall. The agent must place a red ball to push the green "
        "ball across a gap so it intercepts the falling blue jar. Success "
        "requires green-blue contact for 3+ seconds. Key physics: timing "
        "the horizontal push to match the jar's vertical fall, intercept "
        "geometry, gap crossing."
    ),
    "basket_case": (
        "A green ball is positioned above a basket. The agent must place a "
        "red ball to deflect the green ball so it MISSES the basket and "
        "lands on the ground instead. Success requires the green ball to "
        "touch the ground for 3+ seconds. Key physics: deflection angle, "
        "collision timing before the ball enters the basket, push direction "
        "away from basket opening."
    ),
    "pass_the_parcel": (
        "A green ball sits on/in an inverted basket on a platform, with a "
        "ramp nearby. A blue ball is in a bottom basket below. The agent "
        "must place a red ball to knock the green ball out of the top "
        "basket and guide it down (via ramp or direct fall) into the bottom "
        "basket where the blue ball is. Success requires green-blue contact "
        "for 3+ seconds. Key physics: dislodging from inverted basket, "
        "ramp trajectory, multi-stage chain of events."
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
