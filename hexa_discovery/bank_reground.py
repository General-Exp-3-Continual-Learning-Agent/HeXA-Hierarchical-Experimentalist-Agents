"""Re-ground an evolved bank's confidence and lineage against the evidence.

`evolve_skill_bank` trusts the teacher for two fields on RETAINED skills:

- **confidence** is `raw.get("confidence", 0.5)` — whatever the teacher typed.
  Measured on the 2026-09-12 `logo_global_term` run: **6 of 12 final skills had
  an empty `source_seeds` yet carried confidence 0.95-0.98**. The rescaled map in
  `confidence.py` only reaches NEW skills with real source seeds, so confidence
  drifts straight back to the ceiling by echo and retrieval ranking — which is
  what confidence is FOR — stops discriminating.
- **generation** is set to `prev_gen`, so after eight rounds every skill reads
  generation 7-8 and lineage is erased: a surviving round-1 skill is
  indistinguishable from one minted last round.

This is the same failure mode ORBIT's citation verifier exists to catch, and the
discovery path has none installed (`_CITATION_VERIFIER is None`). This is the
cheap version: it cannot tell that a cited seed was invented, but it stops an
unevidenced claim from outranking an evidenced one.

Applied as a post-process on the bank `evolve_skill_bank` returns — no `skillrl`
edit — so it composes with the empty-bank guard (a round that returns the
previous bank unchanged is re-grounded to exactly itself).
"""

from __future__ import annotations

from typing import Optional

#: Per-round multiplier for a retained skill the round produced no evidence for.
#: A skill nothing has confirmed in five rounds sinks from 0.98 to ~0.76 rather
#: than sitting at the ceiling forever. Set to 1.0 to disable decay.
UNSUPPORTED_DECAY = 0.95

#: Never decay below this; a skill should become unattractive, not vanish.
DECAY_FLOOR = 0.15


def reground_bank(
    new_bank,
    prev_bank,
    round_trajs: list[dict],
    level: str,
    *,
    decay: float = UNSUPPORTED_DECAY,
    verbose: bool = True,
):
    """Restore evidence-grounded confidence and original generation, in place.

    - retained + cited seeds  -> recompute confidence from those episodes' rewards
    - retained + no citation  -> keep the PREVIOUS confidence, decayed
    - new                     -> leave alone (already computed on the way in)

    Returns `new_bank` for chaining.
    """
    from hexa_discovery.confidence import compute_initial_confidence

    prev = {s.skill_id: s for s in prev_bank.level_skills.get(level, [])}
    # Titles too: retained skills are re-matched by exact title string upstream,
    # so a reworded title reissues the skill under a fresh id and would otherwise
    # look "new" here and lose both its history and its grounding.
    prev_by_title = {s.title: s for s in prev_bank.level_skills.get(level, [])}

    regrounded = recomputed = decayed = 0
    for skill in new_bank.level_skills.get(level, []):
        old = prev.get(skill.skill_id) or prev_by_title.get(skill.title)
        if old is None:
            continue  # genuinely new: confidence already computed from evidence

        # Lineage: restore what the skill actually was, not prev_gen.
        skill.generation = old.generation
        regrounded += 1

        if skill.source_seeds:
            before = skill.confidence
            skill.confidence = compute_initial_confidence(
                skill.source_seeds, round_trajs, fallback_confidence=old.confidence
            )
            if abs(skill.confidence - before) > 1e-9:
                recomputed += 1
        else:
            skill.confidence = max(DECAY_FLOOR, old.confidence * decay)
            decayed += 1

    if verbose and regrounded:
        print(f"  Re-grounded {regrounded} retained skill(s): "
              f"{recomputed} confidence recomputed from cited episodes, "
              f"{decayed} decayed for lack of any citation")
    return new_bank
