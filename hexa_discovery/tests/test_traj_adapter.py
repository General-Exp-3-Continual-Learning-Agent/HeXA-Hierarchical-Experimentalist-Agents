"""Unit tests for the trajectory adapter + numeric redaction. No LLM needed."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # HeXA (hexa_discovery, skillrl)

from hexa_discovery.traj_adapter import (
    build_trajectory,
    redact_numeric,
    _reasoning_only,
)


def _fake_log():
    return [
        {
            "round": 1,
            "action": "experiment",
            "llm_reply": "Let me probe short timescales.\n<run_experiment>[{\"pos2\":[3.0,0.0]}]</run_experiment>",
            "experiment_input": [{"pos2": [3.0, 0.0]}],
            "experiment_output": [{"trajectory": [[3.0, 0.0], [2.9, 0.1]]}],
        },
        {
            "round": 2,
            "action": "mse_fit",
            "llm_reply": "Fitting the exponent.\n<run_mse_fit>def discovered_law(): alpha=1.5</run_mse_fit>",
            "mse_fit_input": "def discovered_law(): alpha=1.5",
            "mse_fit_output": {"loss_after": 0.01, "fitted_params": {"alpha": 1.5}},
        },
        {
            "round": 3,
            "action": "final_law",
            "llm_reply": "Submitting.\n<final_law>def discovered_law(): alpha=1.5</final_law><explanation>power law</explanation>",
            "final_law": "def discovered_law(): alpha=1.5",
            "explanation": "power law",
        },
    ]


def test_redact_numeric_masks_constants():
    out = redact_numeric("alpha=1.5, G=6.67e-11, n=-3")
    assert "1.5" not in out and "6.67e-11" not in out and "-3" not in out
    assert "<n>" in out and "alpha" in out and "G" in out


def test_reasoning_only_strips_action_bodies():
    r = _reasoning_only(
        "Think hard.\n<run_experiment>[{\"x\":1}]</run_experiment>\nmore thoughts"
    )
    assert "Think hard." in r and "more thoughts" in r
    assert "run_experiment" not in r and '{"x":1}' not in r


def test_build_trajectory_schema_and_success():
    metrics = {
        "mean_pos_error": 0.05, "norm_mse": 0.008,
        "expl_score": 0.9, "passed_benchmark": True,
    }
    traj = build_trajectory(
        seed=7, world="oscillator", conversation_log=_fake_log(),
        explanation="power law", law_source="def discovered_law(): alpha=1.5",
        metrics=metrics, max_rounds=10, reward=0.9, redact=True,
    )
    # HeXA-required fields
    assert traj["seed"] == 7 and traj["world"] == "oscillator"
    assert traj["success"] is True
    assert traj["action"] is None            # no InterPhyre [x,y,r]
    assert isinstance(traj["trajectory"], list) and len(traj["trajectory"]) == 3
    for step in traj["trajectory"]:
        assert set(step) == {"thought", "action", "observation"}
    # graded-reward inputs present
    assert traj["norm_mse"] == 0.008 and traj["expl_score"] == 0.9
    assert traj["reward"] == 0.9
    assert traj["iterations"] == 1           # one experiment round


def test_build_trajectory_redacts_law_in_visible_steps():
    metrics = {"mean_pos_error": 9.0, "norm_mse": 9.0,
               "expl_score": 0.1, "passed_benchmark": False}
    traj = build_trajectory(
        seed=1, world="fractional", conversation_log=_fake_log(),
        explanation="x", law_source="def discovered_law(): alpha=1.5",
        metrics=metrics, max_rounds=10, redact=True,
    )
    blob = "\n".join(
        f"{s['action']} {s['observation']}" for s in traj["trajectory"]
    )
    # The fitted constant must not appear in what the teacher will read.
    assert "1.5" not in blob
    # ...but the raw law is preserved for our own analysis.
    assert traj["final_law_raw"] == "def discovered_law(): alpha=1.5"
    assert traj["success"] is False


def test_reasoning_prose_answer_is_redacted():
    """Regression for the gravity smoke leak: reasoning stated 'a ∝ 1/r ...
    k≈0.16' and an <explanation> block leaked the answer into the thought."""
    log = [{
        "round": 2, "action": "final_law",
        "llm_reply": (
            "Round 2 confirms acceleration scales as k*p1/(p2*r), k≈0.16, "
            "a proportional to 1/r not 1/r^2.\n"
            "<final_law>def discovered_law(): k=0.16</final_law>"
            "<explanation>Force falls off as 1/r like a 2D line charge.</explanation>"
        ),
        "final_law": "def discovered_law(): k=0.16",
        "explanation": "Force falls off as 1/r like a 2D line charge.",
    }]
    metrics = {"mean_pos_error": 1e-4, "norm_mse": 1e-5,
               "expl_score": 0.9, "passed_benchmark": True}
    traj = build_trajectory(
        seed=0, world="gravity", conversation_log=log,
        explanation="Force falls off as 1/r like a 2D line charge.",
        law_source="def discovered_law(): k=0.16",
        metrics=metrics, max_rounds=4, reward=0.95, redact=True,
    )
    visible = "\n".join(
        f"{s['thought']} {s['action']} {s['observation']}" for s in traj["trajectory"]
    )
    assert "0.16" not in visible            # constant masked
    assert "line charge" not in visible     # explanation body stripped
    assert "1/r" not in visible             # numeric form masked -> <n>/r
    # method-level words still survive for the teacher to learn from
    assert "acceleration scales" in visible.lower()
    # full answer still retained for our own analysis, off the teacher's path
    assert traj["explanation"] == "Force falls off as 1/r like a 2D line charge."


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
