"""Convert a DiscoverPhysics episode into HeXA's trajectory JSON schema.

HeXA's teacher reads files named `trajectory_seed{N}_skillrl.json` with fields:
  success (bool), iterations (int), seed (int),
  trajectory: [ {thought, action, observation}, ... ],  action (final).

DiscoveryAgent instead records `conversation_log` rounds with
llm_reply / experiment_input / experiment_output / mse_fit_output / final_law.
This module maps one to the other and stamps the graded-reward inputs
(`norm_mse`, `expl_score`, `reward`) that reward.py consumes.

Leakage guard: the final-law source and mid-round fits can contain the hidden
answer's fitted constants (alpha, omega, screening length, ...) AND the name of
its functional form. When `redact=True` (default) two passes run over everything
the teacher reads: numeric literals become <n>, and terms naming a specific
world's answer become <law-form> (see `_LAW_FORM_PATTERNS`). A distilled skill
keeps the *method* -- that a fit was run, how the data was read -- but neither
the fitted values nor which law they belong to. The law-form pass is a
blocklist and cannot prove absence; `scan_for_law_forms` audits the teacher's
output for what got through.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

# Matches int/float/scientific literals, e.g. -3, 0.075, 1.5e-3, .5
_NUM_RE = re.compile(r"(?<![A-Za-z_0-9])[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")

# Tag bodies stripped from reasoning prose. <explanation> is included because
# the agent's prose explanation states the discovered answer ("falls off as
# 1/r", "line charge") almost verbatim — it must not leak into a banked skill.
_ACTION_TAG_RE = {
    "run_experiment": re.compile(r"<run_experiment>.*?</run_experiment>", re.DOTALL),
    "run_mse_fit": re.compile(r"<run_mse_fit>.*?</run_mse_fit>", re.DOTALL),
    "final_law": re.compile(r"<final_law>.*?</final_law>", re.DOTALL),
    "explanation": re.compile(r"<explanation>.*?</explanation>", re.DOTALL),
}


def redact_numeric(text: Optional[str]) -> str:
    """Mask numeric literals with <n> to keep methods but hide fitted values."""
    if not text:
        return ""
    return _NUM_RE.sub("<n>", text)


# ── Law-form redaction ─────────────────────────────────────────────────
#
# Numeric redaction masks `alpha=1.5` but leaves `exp(-r/L)` and the word
# "Yukawa" untouched, so the model CLASS crosses worlds intact. That was not
# hypothetical: the first 2-world smoke bank produced a skill reading "a
# straight line confirms Yukawa-type exp(-r/L)/r decay and gives L from the
# slope", which is the yukawa answer in method's clothing.
#
# What belongs here is a term that IDENTIFIES one world's answer. What does not
# is a term naming a method or a model class you would legitimately test for on
# any world: "power law", "log-log slope", "central force", "fit the exponent"
# all survive on purpose — masking them would gut exactly the transferable
# content the bank exists to carry.
#
# THIS IS A BLOCKLIST AND BLOCKLISTS ARE INCOMPLETE. It raises the cost of a
# leak; it does not prove absence. A teacher can still describe a screened
# exponential in words this never matches. Treat it as one layer, keep reporting
# leave-one-GROUP-out as the headline (see docs/DISCOVERY_PHYSICS.md §5), and
# read `scan_for_law_forms` output on every run.
_LAW_FORM_PATTERNS = (
    # Eponymous laws — each names exactly one world's answer.
    r"yukawa", r"coulomb", r"newtonian", r"kaluza[\s_-]*klein",
    r"hubble", r"poisson", r"helmholtz", r"riesz", r"bessel", r"kepler",
    # Named force-law families (NOT "power law" — that is a class you test for).
    r"inverse[\s_-]*square", r"inverse[\s_-]*cube",
    r"screened", r"screening[\s_-]*length",
    r"fractional[\s_-]*laplacian", r"anomalous[\s_-]*diffusion",
    # Closed-form exponential/screening expressions.
    r"exp\s*\(\s*-[^)]*\)", r"e\^\{?\s*-[^}\s]*\}?",
    # Trigonometric / phase notation. A judge critique leaked "cos(wt) with the
    # right w~pi/2" straight through, because the law-form masking caught
    # "Laplacian" but had no pattern for trig. Greek omega and phi included --
    # they ARE the oscillator's hidden parameters.
    r"\b(?:cos|sin|tan)\s*\([^)]*\)", r"\bcos\b", r"\bsin\b",
    # `\bcos\b` does NOT match "cosine" -- the word boundary stops it -- and the
    # bare-word forms below were all absent. Measured on the Episodic RAG blocks
    # injected into oscillator's eval: 15/15 episodes carried "cosine" (45
    # occurrences) and "periodic" (45), i.e. every episode was shown a retrieved
    # transcript reading "modulated by a cosine in t" -- the hidden law in plain
    # text. oscillator's HeXA bank also carried "periodic" and "reverses
    # direction". A blocklist is only as good as its longest-tail entry; see
    # bank_audit's docstring on why the bank is scanned rather than trusted.
    r"cosine", r"\bsine\b", r"sine[\s_-]*wave", r"harmonic",
    r"\bperiodic\b", r"alternat\w*[\s_-]*sign",
    r"flips?[\s_-]*sign", r"reverses?[\s_-]*direction",
    r"[\u03c9\u03a9]", r"\bomega\b", r"[\u03c6\u03d5\u03a6]", r"\bphi\b",
    # World-identifying phenomena.
    r"dark[\s_-]*(?:matter|halo)", r"\bether\b", r"luminiferous",
    r"extra[\s_-]*dimension\w*", r"compactification",
    r"preferred[\s_-]*direction", r"hubble[\s_-]*flow",
    # The oscillator's answer, in the forms it actually gets written.
    # Oscillator's answer, in every phrasing it has actually been written in.
    # The original patterns required "sinusoidally VARYING" and "time-dependent
    # COUPLING"; a within-world bank wrote "velocity swings sinusoidally ... with
    # a period" and "time-dependence (periodicity, sign-flips)", which slipped
    # through and lifted expl 0.32 -> 0.90 with 3/3 agents naming the mechanism.
    # Match the CONCEPTS, not one phrasing.
    r"sinusoid\w*", r"periodicity", r"\bperiod\b", r"oscillat\w*",
    r"time[\s_-]*depend\w*", r"time[\s_-]*vary\w*", r"time[\s_-]*modulat\w*",
    r"sign[\s_-]*flip\w*", r"reverses? sign", r"phase\b",
    r"periodically[\s_-]*(?:revers\w+|flip\w*|chang\w+)",
)

LAW_FORM_RE = re.compile("|".join(_LAW_FORM_PATTERNS), re.IGNORECASE)

LAW_FORM_PLACEHOLDER = "<law-form>"


def redact_law_forms(text: Optional[str]) -> str:
    """Mask terms that name a specific world's answer, keeping method language."""
    if not text:
        return ""
    return LAW_FORM_RE.sub(LAW_FORM_PLACEHOLDER, text)


def scan_for_law_forms(text: Optional[str]) -> list[str]:
    """Return the distinct law-form terms present in `text`, lowercased.

    Used to audit the teacher's OUTPUT. Input redaction cannot guarantee the
    teacher never writes such a term (it may infer one), so the bank is scanned
    rather than scrubbed: scrubbing a skill would mangle prose we cannot
    regenerate, while a flag tells you which run to distrust.
    """
    if not text:
        return []
    return sorted({m.group(0).lower() for m in LAW_FORM_RE.finditer(text)})


def redact_text(text: Optional[str]) -> str:
    """Full redaction: numeric literals, then law-form names.

    NOT named `redact`: `_step_from_round` and `build_trajectory` already take a
    `redact: bool` flag, which would shadow the function inside their bodies.
    """
    return redact_law_forms(redact_numeric(text))


def _reasoning_only(reply: Optional[str]) -> str:
    """Strip action-tag bodies from an LLM reply, leaving the prose reasoning."""
    if not reply:
        return ""
    out = reply
    for rx in _ACTION_TAG_RE.values():
        out = rx.sub("", out)
    # Belt-and-suspenders: drop any dangling (unclosed) explanation tag too.
    out = re.sub(r"<explanation>.*", "", out, flags=re.DOTALL)
    return out.strip()


def _truncate(text: str, limit: int) -> str:
    if text is None:
        return ""
    text = str(text)
    return text if len(text) <= limit else text[:limit] + "... [truncated]"


def _fit_summary(fit_out: Optional[dict], redact: bool) -> str:
    """Summarise an mse_fit_output for the teacher.

    Keeps the qualitative signal (loss_before/after, error) but, under redaction,
    masks the fitted/declared parameter VALUES — those are the closed-form answer
    (e.g. alpha=1.5) and must not become a banked skill constant. Raw experiment
    trajectory positions are left intact elsewhere (they are legitimate
    observations, not a handed-over formula).
    """
    if not isinstance(fit_out, dict):
        return _truncate(json.dumps(fit_out), 400)
    summary = {
        k: fit_out.get(k)
        for k in ("loss_before", "loss_after", "n_training", "error")
        if k in fit_out
    }
    for pk in ("fitted_params", "declared_params"):
        if pk in fit_out and fit_out[pk]:
            params = fit_out[pk]
            if redact and isinstance(params, dict):
                summary[pk] = {redact_law_forms(k): "<n>" for k in params}
            else:
                summary[pk] = params
    return _truncate(json.dumps(summary), 400)


def _step_from_round(
    entry: dict, *, redact: bool, obs_limit: int, law_source: str, metrics: dict
) -> Optional[dict]:
    """Map one conversation_log round to a {thought, action, observation} step."""
    action_type = entry.get("action")
    thought = _reasoning_only(entry.get("llm_reply"))
    # The reasoning prose routinely states the discovered law + constants
    # ("a ∝ 1/r ... k≈0.16"). Under redaction, mask numeric literals so the
    # transferable METHOD survives ("measure a·r", "scan p1 separately") but the
    # answer's values/exponents do not.
    if redact:
        thought = redact_text(thought)

    if action_type == "experiment":
        exp_in = entry.get("experiment_input")
        exp_out = entry.get("experiment_output")
        err = entry.get("experiment_error")
        action = "run_experiment: " + _truncate(json.dumps(exp_in), 600)
        if err:
            observation = f"experiment error: {err}"
        else:
            observation = _truncate(json.dumps(exp_out, separators=(",", ":")), obs_limit)
        # A fit may co-occur with an experiment; fold its result in.
        fit_out = entry.get("mse_fit_output")
        if fit_out is not None:
            observation += "\n[mse_fit] " + _fit_summary(fit_out, redact)
        return {"thought": thought, "action": action, "observation": observation}

    if action_type == "mse_fit":
        fit_in = entry.get("mse_fit_input") or ""
        if redact:
            fit_in = redact_text(fit_in)
        return {
            "thought": thought,
            "action": "run_mse_fit: " + _truncate(fit_in, 600),
            "observation": _fit_summary(entry.get("mse_fit_output"), redact),
        }

    if action_type == "final_law":
        law = law_source or entry.get("final_law") or ""
        if redact:
            law = redact_text(law)
        # Do NOT fall back to the prose explanation — it states the answer.
        # (The full explanation is kept only in the top-level `explanation`
        # field, which the teacher never reads.)
        obs = (
            f"submitted final law (redacted={redact}); "
            f"norm_mse={metrics.get('norm_mse')}, expl_score={metrics.get('expl_score')}, "
            f"passed={metrics.get('passed_benchmark')}"
        )
        return {
            "thought": thought or "submitted final law",
            "action": "submit_final_law: " + _truncate(law, 800),
            "observation": obs,
        }

    # warning / no_tag / random rounds — keep the reasoning as a lightweight step.
    if thought:
        return {
            "thought": thought,
            "action": action_type or "note",
            "observation": _truncate(entry.get("system_message") or "", 300),
        }
    return None


def build_trajectory(
    *,
    seed: int,
    world: str,
    conversation_log: list[dict],
    explanation: Optional[str],
    law_source: Optional[str],
    metrics: dict,
    max_rounds: int,
    reward: Optional[float] = None,
    redact: bool = True,
    obs_limit: int = 1400,
) -> dict[str, Any]:
    """Assemble a HeXA-schema trajectory dict from a DiscoverPhysics episode.

    `metrics` must contain: mean_pos_error, norm_mse, expl_score,
    passed_benchmark. `reward` is stored precomputed if given (else reward.py
    recomputes from norm_mse/expl_score).
    """
    steps: list[dict] = []
    n_experiment_rounds = 0
    for entry in conversation_log:
        if entry.get("action") in ("experiment", "random_experiment"):
            n_experiment_rounds += 1
        step = _step_from_round(
            entry, redact=redact, obs_limit=obs_limit,
            law_source=law_source or "", metrics=metrics,
        )
        if step is not None:
            steps.append(step)

    return {
        "seed": seed,
        "world": world,
        # `iterations` = experiment rounds used (kept for logging; reward no
        # longer depends on it).
        "iterations": n_experiment_rounds or len(conversation_log),
        "success": bool(metrics.get("passed_benchmark")),
        "trajectory": steps,
        # Top-level `action` is None on purpose: HeXA's default formatter would
        # try action[0]/[1]/[2] (InterPhyre's [x,y,r]); our patched formatter
        # ignores it, and None keeps the stock formatter harmless too.
        "action": None,
        # Graded-reward inputs consumed by reward.compute_trajectory_reward:
        "norm_mse": metrics.get("norm_mse"),
        # Same law, the agent's own constants, no evaluator refit. The gap
        # between these two is the only signal saying "declaring this parameter
        # fittable cost you"; see discovery_runner.evaluate_episode.
        "norm_mse_own": metrics.get("norm_mse_own"),
        "expl_score": metrics.get("expl_score"),
        "mean_pos_error": metrics.get("mean_pos_error"),
        "reward": reward,
        # Kept for our own analysis; NOT shown to the teacher (redaction aside,
        # the patched formatter only reads `trajectory`).
        # Judge critique: the only signal saying WHICH rubric dimension was
        # missed. Redacted like everything else the teacher reads -- critiques
        # name the answer outright ("the fractional Laplacian operator").
        "expl_reasoning": (redact_text(metrics.get("expl_reasoning"))
                           if redact else (metrics.get("expl_reasoning") or "")),
        "final_law_raw": law_source,
        "explanation": explanation,
        "redacted": redact,
        "max_rounds": max_rounds,
    }
