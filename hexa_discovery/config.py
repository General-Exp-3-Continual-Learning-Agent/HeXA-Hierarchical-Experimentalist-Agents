"""Configuration for running HeXA (skill-evolution) on DiscoverPhysics.

Central knobs:
  - the synthetic "level" name the cross-world meta-bank lives under
  - per-world ground-truth trajectory variances (for norm_MSE)
  - benchmark pass thresholds
  - the 4 world groups (clustering that makes cross-world transfer a real test)
  - ANSWER-FREE factual blocks per world, shown to the teacher as the
    `level_description`.  These describe the observable setup and the knobs the
    agent controls, and pose the goal as an open question.  They intentionally
    do NOT state the hidden law (no "inverse-square", "screening length",
    "time-dependent", etc.) so distilled skills stay at the method altitude and
    cannot memorise a world's answer.
"""

from __future__ import annotations

# The whole cross-world bank is stored under one synthetic level so HeXA's
# level-keyed distill/evolve machinery can be reused unchanged.
DISCOVERY_LEVEL = "discovery"

# mean_pos_error / _WORLD_VARS[world] must be < PASS_ERR_THRESHOLD, and the
# explanation-judge score must be >= PASS_SCORE_THRESHOLD, for a trial to pass.
# Values copied from discovery-agents/scripts/run_benchmark.py.
PASS_ERR_THRESHOLD = 0.1
PASS_SCORE_THRESHOLD = 0.75

WORLD_VARS: dict[str, float] = {
    "circle": 6.596,
    "coulomb_easy": 11.465,
    "dark_matter": 63.303,
    "ether": 21.259,
    "extra_dimensions": 4.248,
    "fractional": 5.721,
    "gravity": 4.283,
    "hubble": 41.189,
    "oscillator": 6.332,
    "three_species": 28.717,
    "yukawa": 5.677,
}

# Clusters by shared discovery task.  Used to build leave-one-group-out
# transfer splits and to reason about which skills should transfer.
WORLD_GROUPS: dict[str, list[str]] = {
    "central_force": [
        "gravity", "yukawa", "fractional",
        "coulomb_easy", "extra_dimensions", "circle",
    ],
    "hidden_sources": ["three_species", "dark_matter"],
    "global_term": ["ether", "hubble"],
    "time_dependent": ["oscillator"],
}

BENCHMARK_WORLDS: list[str] = [w for ws in WORLD_GROUPS.values() for w in ws]

# nbody is the default engine; a few worlds require it explicitly.
WORLD_ENGINE: dict[str, str] = {w: "nbody" for w in BENCHMARK_WORLDS}

# Worlds whose evaluator fits free parameters from collected trajectories
# (mirrors _FIT_WORLDS in run_discovery.py, restricted to our 11).
FIT_WORLDS = {
    "gravity", "yukawa", "fractional", "oscillator",
    "extra_dimensions", "circle", "ether", "hubble",
}

# ── Answer-free factual blocks (shown as level_description to the teacher) ──
_GOAL = (
    "Goal: discover the governing law of motion (the force law) that reproduces "
    "the observed particle trajectories, and be able to state it as executable "
    "Python plus a short physical explanation."
)

WORLD_FACTS: dict[str, str] = {
    "gravity": (
        "Several particles interact through a static field that each sources via "
        "a scalar property p. You choose every particle's initial position, "
        "velocity and p, and the measurement times, then observe noisy positions "
        "over time. " + _GOAL
    ),
    "yukawa": (
        "Particles source a static field via a scalar property p. You set "
        "positions, velocities, p and the measurement times and observe noisy "
        "positions over time. " + _GOAL
    ),
    "fractional": (
        "Particles source a static field via a scalar property p. You set "
        "positions, velocities, p and the measurement times and observe noisy "
        "positions over time. " + _GOAL
    ),
    "coulomb_easy": (
        "Exactly two particles; particle 1 is fixed at the origin. You control "
        "particle 2's initial position and velocity, both scalar properties p1 "
        "and p2, and the measurement times. You observe noisy positions over "
        "time. " + _GOAL
    ),
    "extra_dimensions": (
        "Particles source a static field via a scalar property p. You set "
        "positions, velocities, p and the measurement times across a wide range "
        "of separations and observe noisy positions over time. " + _GOAL
    ),
    "circle": (
        "A centre particle plus ten more arranged on a ring, all of which source "
        "the field. Your ONLY controls are the ring's radius, the ring particles' "
        "initial tangential velocity (shared by all of them), and the measurement "
        "times. You cannot place probes, address individual particles, set masses, "
        "or choose positions. " + _GOAL
    ),
    "three_species": (
        "Several particle classes are present together with neutral probe "
        "particles that do not source the field. You observe all trajectories "
        "and control probe placement and measurement times. " + _GOAL
    ),
    "dark_matter": (
        "A set of particles is observed together with neutral probes; some "
        "sources present in the scene may not be directly observable. You "
        "control probe placement and measurement times and observe noisy "
        "trajectories. " + _GOAL
    ),
    "ether": (
        "A central configuration (a heavy anchor, orbiting particles, and neutral "
        "probes). You control probe initial conditions and the measurement times "
        "and observe noisy trajectories. " + _GOAL
    ),
    "hubble": (
        "A central configuration (a heavy anchor, orbiting particles, and neutral "
        "probes). You control probe initial conditions and the measurement times "
        "and observe noisy trajectories. " + _GOAL
    ),
    "oscillator": (
        "A central two-/N-body configuration. You control initial positions, "
        "velocities, the scalar property p, the measurement times AND the total "
        "observation duration, and observe noisy positions over time. " + _GOAL
    ),
}


#: The keys the agent must actually emit in <run_experiment>, per world, taken
#: from each world's `experiment_format`. WORLD_FACTS is prose the teacher reads;
#: it does not name these, so a teacher (and pi_ev) could write skills about
#: controls that do not exist. Measured on `circle`: of 8 injected skills, 4 were
#: physically unexecutable ("probe individual ring members", "probe empty
#: regions", "check mass-independence", "angular sweep") and were acted on in
#: 0/5 episodes, while the one shared behaviour was something the no-bank agent
#: already did. Half the bank was inert, which is why "good skills" produced no
#: behavioural change.
WORLD_CONTROLS: dict[str, str] = {
    **{w: ("p1 (source coupling of the fixed particle 1), p2 (inertia of the "
           "mobile particle 2), pos2 (particle 2's initial position), velocity2 "
           "(its initial velocity), measurement_times")
       for w in ("gravity", "yukawa", "fractional", "coulomb_easy",
                 "extra_dimensions", "oscillator")},
    "circle": ("ring_radius (one scalar), initial_tangential_velocity (one "
               "scalar, shared by every ring particle), measurement_times. "
               "NOTHING ELSE — no probe placement, no per-particle control, no "
               "masses, no positions"),
    "three_species": ("probe_positions (5 neutral probes), probe_velocities, "
                      "measurement_times. The background particles cannot be moved"),
    "dark_matter": ("probe_positions (5 neutral probes), probe_velocities, "
                    "measurement_times. The sources cannot be moved"),
    "ether": ("probe_positions (5 probes), probe_velocities, probe_masses, "
              "measurement_times. The anchor and ring orbiters are fixed"),
    "hubble": ("probe_positions (5 probes), probe_velocities, probe_masses, "
               "measurement_times. The anchor and ring orbiters are fixed"),
}


def world_facts_with_controls(world: str, include_base_prompt: bool = False) -> str:
    """WORLD_FACTS plus the exact controls the agent can set.

    A skill the agent cannot execute is worse than no skill: it occupies the
    injected prompt and is silently ignored. Stating the action space is what
    keeps a distilled or synthesised skill actionable.
    """
    facts = WORLD_FACTS[world]
    ctrl = WORLD_CONTROLS.get(world)
    if not ctrl:
        return facts
    out = (f"{facts}\n\nControls you can set in each experiment (and no "
           f"others): {ctrl}.")
    if include_base_prompt:
        # The teacher otherwise distils lessons the agent was already given;
        # see hexa_discovery.base_prompt for the measurement.
        from hexa_discovery.base_prompt import already_known_block
        out += already_known_block(world)
    return out


def pooled_world_facts(worlds: list[str]) -> str:
    """Answer-free `level_description` for a POOLED, multi-world trajectory set.

    `pooled` mode distils from several worlds at once. Labelling that set with
    `WORLD_FACTS[train_worlds[0]]` — as the loop used to — tells the teacher
    every episode came from one world's setup, which is false for all but one of
    them: it reads "exactly two particles; particle 1 is fixed at the origin"
    while looking at eleven-particle ring trajectories. This states the shared
    structure instead. It does NOT name the worlds: several world names ARE the
    answer ("yukawa", "hubble", "ether", "dark_matter", ...), which is also why
    the trajectory header shows an anonymous label -- see
    `teacher_prompts_discovery.world_label`.
    """
    return (
        f"These episodes come from {len(worlds)} DIFFERENT simulated worlds, "
        "each with its own hidden governing law; the trajectory header labels "
        "which world each episode came from. In all of them you "
        "choose particle initial positions, velocities and scalar properties, "
        "and the measurement times, then observe noisy positions over time. "
        "The worlds differ in how many particles are present, whether some "
        "sources are observable, and what produces the motion. " + _GOAL
    )


def world_group(world: str) -> str | None:
    """Return the group name a world belongs to, or None."""
    for g, ws in WORLD_GROUPS.items():
        if world in ws:
            return g
    return None


def leave_one_out_split(held_out: str) -> tuple[list[str], list[str]]:
    """Return (train_worlds, [held_out]) for a leave-one-world-out transfer test."""
    train = [w for w in BENCHMARK_WORLDS if w != held_out]
    return train, [held_out]


def leave_one_group_out_split(held_out_group: str) -> tuple[list[str], list[str]]:
    """Return (train_worlds, held_out_worlds) leaving a whole group out."""
    held = WORLD_GROUPS[held_out_group]
    train = [w for w in BENCHMARK_WORLDS if w not in held]
    return train, held
