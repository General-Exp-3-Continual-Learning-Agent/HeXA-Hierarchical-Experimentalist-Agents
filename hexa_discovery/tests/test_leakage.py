"""Law-form redaction, bank auditing, and pooled-mode merging.

No LLM / scienceagent needed.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # HeXA (hexa_discovery, skillrl)

from hexa_discovery.bank_audit import attribute, audit_bank
from hexa_discovery.config import pooled_world_facts
from hexa_discovery.traj_adapter import (
    build_trajectory,
    redact_text,
    redact_law_forms,
    scan_for_law_forms,
)

# Verbatim from the 2-world smoke bank (dis_ev_1_001) — the leak that motivated
# all of this.
_LEAKED = (
    "If a straight line emerges, the functional form is (const/r^n)*exp(-r/L). "
    "Confirm the screening length L from the slope of that linear fit. A "
    "straight line confirms Yukawa-type decay."
)

# A genuinely transferable method: must survive redaction untouched.
_METHOD = (
    "Place the probe at rest at geometrically spaced distances along a single "
    "axis and measure two early-time positions. Plot log(a) vs log(r) to "
    "identify the radial power-law exponent, or detect curvature that signals a "
    "non-power-law form. Sweep one variable per experiment and fit parameters "
    "before submitting."
)


# ── redaction ──────────────────────────────────────────────────────────

def test_masks_eponymous_law_names():
    assert "yukawa" not in redact_law_forms(_LEAKED).lower()
    assert "coulomb" not in redact_law_forms("a Coulomb-like 1/r^2 law").lower()


def test_masks_closed_form_exponentials():
    assert "exp(-r/L)" not in redact_law_forms(_LEAKED)


def test_masks_screening_language():
    assert "screening length" not in redact_law_forms(_LEAKED).lower()


def test_masks_every_worlds_identifying_term():
    for text in ("a dark matter halo", "the luminiferous ether",
                 "Kaluza-Klein tower", "Hubble flow", "fractional Laplacian",
                 "an inverse-square law", "a time-dependent coupling",
                 "the coupling periodically reverses sign"):
        out = redact_law_forms(text)
        assert "<law-form>" in out, text


def test_preserves_transferable_method_language():
    """Over-redaction would gut exactly what the bank exists to carry."""
    assert redact_law_forms(_METHOD) == _METHOD
    # 'power law' is a model class you test for anywhere — must NOT be masked.
    assert "power-law" in redact_law_forms("identify the radial power-law exponent")


def test_full_redact_does_numbers_and_forms():
    out = redact_text("Yukawa decay with alpha=1.5 and L=2.0")
    assert "<law-form>" in out and "<n>" in out
    assert "1.5" not in out and "Yukawa" not in out


def test_scan_reports_what_is_present():
    assert "yukawa" in scan_for_law_forms(_LEAKED)
    assert scan_for_law_forms(_METHOD) == []


# ── adapter integration ────────────────────────────────────────────────

def _episode(reply, law):
    return dict(
        seed=0, world="yukawa",
        conversation_log=[{"action": "final_law", "llm_reply": reply,
                           "final_law": law}],
        explanation="it is a screened Yukawa force",
        law_source=law,
        metrics={"norm_mse": 0.004, "expl_score": 0.9, "mean_pos_error": 0.02,
                 "passed_benchmark": True},
        max_rounds=10,
    )


def test_adapter_redacts_law_forms_in_thoughts_and_law():
    traj = build_trajectory(**_episode(
        "The data confirms Yukawa-type exp(-r/L)/r decay with L=2.0.",
        "def discovered_law(): return k*exp(-r/2.0)/r",
    ))
    blob = json.dumps(traj["trajectory"])
    assert "Yukawa" not in blob and "exp(-" not in blob
    assert "2.0" not in blob


def test_adapter_redaction_can_be_disabled():
    traj = build_trajectory(redact=False, **_episode(
        "Yukawa-type decay.", "def discovered_law(): pass"))
    assert "Yukawa" in json.dumps(traj["trajectory"])


def test_fit_parameter_names_are_redacted():
    """A key named `screening_length` leaks even with its value masked."""
    from hexa_discovery.traj_adapter import _fit_summary
    out = _fit_summary({"fitted_params": {"screening_length": 2.0}}, True)
    assert "screening_length" not in out
    assert "2.0" not in out


# ── bank audit ─────────────────────────────────────────────────────────

class _Skill:
    def __init__(self, skill_id, title, principle):
        self.skill_id, self.title, self.principle = skill_id, title, principle
        self.when_to_apply = self.example = ""


class _Bank:
    def __init__(self, skills):
        self.level_skills = {"discovery": skills}
        self.level_mistakes = {}


def test_audit_flags_the_real_leaked_skill():
    bank = _Bank([_Skill("dis_ev_1_001", "Detect Screened Forms", _LEAKED)])
    audit = audit_bank(bank, "discovery", held_out_worlds=["yukawa"])
    assert audit["n_flagged"] == 1
    assert audit["n_critical"] == 1
    assert audit["findings"][0]["leaks_held_out"] == ["yukawa"]


def test_audit_is_quiet_when_the_named_world_is_in_train():
    bank = _Bank([_Skill("dis_ev_1_001", "Detect Screened Forms", _LEAKED)])
    audit = audit_bank(bank, "discovery", held_out_worlds=["oscillator"])
    assert audit["n_flagged"] == 1      # still reported...
    assert audit["n_critical"] == 0     # ...but it leaks nothing held out


def test_audit_clean_on_method_only_bank():
    bank = _Bank([_Skill("dis_sk_000", "Distance Sweep", _METHOD)])
    audit = audit_bank(bank, "discovery", held_out_worlds=["yukawa"])
    assert audit["n_flagged"] == 0


def test_terms_attribute_to_their_world():
    assert attribute("yukawa") == ["yukawa"]
    assert attribute("dark matter") == ["dark_matter"]
    assert attribute("kaluza-klein") == ["extra_dimensions"]


# ── pooled mode ────────────────────────────────────────────────────────

def test_pooled_facts_state_the_count_but_never_the_names():
    """World names are not safe to print: "yukawa" IS the answer."""
    facts = pooled_world_facts(["gravity", "yukawa", "oscillator"])
    assert "3 DIFFERENT" in facts
    for w in ("gravity", "yukawa", "oscillator"):
        assert w not in facts.lower()
    # Answer-free: no law form may appear in what the teacher is told.
    assert scan_for_law_forms(facts) == []


def test_pooled_merge_makes_seeds_unique(tmp_path):
    from hexa_discovery.cross_world_loop import build_pooled_dir

    dirs = []
    for world in ("gravity", "yukawa"):
        d = tmp_path / world / "discovery"
        d.mkdir(parents=True)
        for seed in (0, 1):
            # Same filenames AND same seeds in both worlds — the collision.
            (d / f"trajectory_seed{seed}_skillrl.json").write_text(
                json.dumps({"seed": seed, "world": world, "success": True})
            )
        dirs.append(d)

    merged = build_pooled_dir(dirs, tmp_path / "out")
    files = sorted(merged.glob("trajectory_seed*.json"))
    assert len(files) == 4, "a naive copy would have overwritten two of these"

    loaded = [json.loads(f.read_text()) for f in files]
    assert len({d["seed"] for d in loaded}) == 4, "seeds must be globally unique"
    assert {d["world"] for d in loaded} == {"gravity", "yukawa"}
    assert sorted(d["orig_seed"] for d in loaded) == [0, 0, 1, 1]


def test_pooled_merge_is_readable_by_skillrl(tmp_path):
    """The whole point of the merge is that load_trajectories can read it."""
    from skillrl.distillation.distill import load_trajectories
    from hexa_discovery.cross_world_loop import build_pooled_dir

    dirs = []
    for world, ok in (("gravity", True), ("yukawa", False)):
        d = tmp_path / world / "discovery"
        d.mkdir(parents=True)
        (d / "trajectory_seed0_skillrl.json").write_text(
            json.dumps({"seed": 0, "world": world, "success": ok})
        )
        dirs.append(d)

    merged = build_pooled_dir(dirs, tmp_path / "out")
    successes, failures = load_trajectories(merged, "discovery")
    assert len(successes) == 1 and len(failures) == 1


# ── world-name anonymisation ───────────────────────────────────────────

def test_world_name_never_reaches_the_teacher():
    """Several world names ARE the answer, so the header must not print them."""
    from hexa_discovery import reward as _reward
    from hexa_discovery import teacher_prompts_discovery as _prompts

    _reward.patch()
    _prompts.reset_world_labels()
    block = _prompts.format_trajectory_for_teacher_discovery(
        {"seed": 0, "world": "yukawa", "success": True, "norm_mse": 0.004,
         "expl_score": 0.9, "trajectory": []}
    )
    assert "yukawa" not in block.lower()
    assert "World: W1" in block


def test_world_labels_are_stable_and_distinguish_worlds():
    from hexa_discovery import teacher_prompts_discovery as _prompts

    _prompts.reset_world_labels()
    a1 = _prompts.world_label("gravity")
    b = _prompts.world_label("yukawa")
    a2 = _prompts.world_label("gravity")
    assert a1 == a2 != b, "same world -> same label, different worlds -> different"


def test_formatter_scrubs_stale_unredacted_trajectories():
    """Files written before law-form redaction are reused on resume."""
    from hexa_discovery import reward as _reward
    from hexa_discovery import teacher_prompts_discovery as _prompts

    _reward.patch()
    stale = {
        "seed": 0, "world": "yukawa", "success": True, "redacted": True,
        "norm_mse": 0.004, "expl_score": 0.9,
        "trajectory": [{"thought": "Confirms Yukawa-type exp(-r/L)/r decay",
                        "action": "submit_final_law: k*exp(-r/L)/r",
                        "observation": "passed"}],
    }
    block = _prompts.format_trajectory_for_teacher_discovery(stale)
    assert "Yukawa" not in block and "exp(-r/L)" not in block


def test_formatter_respects_explicit_no_redact():
    from hexa_discovery import reward as _reward
    from hexa_discovery import teacher_prompts_discovery as _prompts

    _reward.patch()
    raw = {
        "seed": 0, "world": "yukawa", "success": True, "redacted": False,
        "norm_mse": 0.004, "expl_score": 0.9,
        "trajectory": [{"thought": "Yukawa decay", "action": "x", "observation": "y"}],
    }
    assert "Yukawa" in _prompts.format_trajectory_for_teacher_discovery(raw)


def test_pooled_bootstrap_refuses_held_out_worlds(tmp_path):
    """A bootstrap pool containing a held-out world would poison the transfer test."""
    import pytest
    from hexa_discovery.cross_world_loop import build_pooled_dir, run_cross_world_loop

    dirs = []
    for world in ("gravity", "ether"):        # ether is held out below
        d = tmp_path / world / "discovery"
        d.mkdir(parents=True)
        (d / "trajectory_seed0_skillrl.json").write_text(
            json.dumps({"seed": 0, "world": world, "success": True, "trajectory": []}))
        dirs.append(d)
    merged = build_pooled_dir(dirs, tmp_path / "out")

    with pytest.raises(ValueError, match="HELD-OUT"):
        run_cross_world_loop(
            train_worlds=["gravity"], held_out_worlds=["ether", "hubble"],
            offline_traj_dir=merged, output_dir=tmp_path / "run",
            teacher_model="claude-sonnet-4-6",
        )


# ── action-space grounding ─────────────────────────────────────────────

def test_controls_are_appended_to_every_world():
    """A skill naming a control the world lacks is inert: measured on circle,
    4 of 8 injected skills were unexecutable and acted on in 0/5 episodes."""
    from hexa_discovery.config import BENCHMARK_WORLDS, world_facts_with_controls
    for w in BENCHMARK_WORLDS:
        assert "Controls you can set" in world_facts_with_controls(w), w


def test_circle_description_no_longer_promises_probes():
    """circle's real action space is {ring_radius, tangential_velocity, times}."""
    from hexa_discovery.config import WORLD_FACTS, world_facts_with_controls
    t = world_facts_with_controls("circle").lower()
    assert "additional probe particles" not in t
    assert "ring_radius" in t and "initial_tangential_velocity" in t
    assert "no probe placement" in t


def test_controls_match_the_real_experiment_format():
    """Guard against drift between WORLD_CONTROLS and the simulator."""
    from hexa_discovery.config import WORLD_CONTROLS
    expected = {
        "circle": ["ring_radius", "initial_tangential_velocity"],
        "hubble": ["probe_positions", "probe_velocities", "probe_masses"],
        "three_species": ["probe_positions", "probe_velocities"],
        "gravity": ["p1", "p2", "pos2", "velocity2"],
    }
    for w, keys in expected.items():
        for k in keys:
            assert k in WORLD_CONTROLS[w], f"{w} missing {k}"
    # circle must NOT claim probe controls
    assert "probe_positions" not in WORLD_CONTROLS["circle"]


# ── oracle feedback must not reach the teacher ─────────────────────────

def test_judge_critique_is_not_shown_to_the_teacher():
    """The judge reads the ground-truth explanation; its critique is an oracle.

    Measured: on oscillator it said the student "correctly identifies a cos(wt)
    with the right w~pi/2", which is the hidden law. The SCORE stays (legitimate
    evaluation signal); the critique does not.
    """
    from hexa_discovery import reward as _reward, teacher_prompts_discovery as _p
    _reward.patch()
    out = _p.format_trajectory_for_teacher_discovery(
        {"seed": 0, "world": "oscillator", "success": False, "redacted": True,
         "norm_mse": 0.01, "expl_score": 0.6, "trajectory": [],
         "expl_reasoning": "correctly identifies a cos(wt) with the right w"})
    assert "[explanation judge]" not in out
    assert "cos(" not in out
    assert "expl_score" in out, "the score itself is legitimate signal and must stay"


def test_trig_and_phase_notation_is_redacted():
    """Law-form masking caught 'Laplacian' but had no pattern for trig."""
    from hexa_discovery.traj_adapter import scan_for_law_forms
    assert scan_for_law_forms("a cos(wt) term with omega ~ pi/2")
    assert scan_for_law_forms("G(t) = G0 cos(wt + phi)")
    # genuine method language must survive
    assert scan_for_law_forms("sweep r over decades and fit the exponent") == []
