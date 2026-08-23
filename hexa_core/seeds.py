"""CORE's per-call seed derivation and its phase namespaces.

Extracted from ``baseline_runner`` when that module was archived. These are the one
part of the CORE-parity machinery that stayed load-bearing after the project dropped
the exact-comparison claim, because they are what make the two eval arms **paired**:
the with-memory and no-memory passes draw the same seed for the same problem, so the
only difference between them is the injected guidelines. McNemar's test assumes that
pairing; without it, it is the wrong test.

The namespaces matter for a second reason. CORE gives each phase its own, so the same
``(run, item)`` pair draws a different generation seed in each. Reusing the train-solve
namespace for admission would make an admission trial a *replay* of an attempt already
made -- it would pass trivially and measure nothing. All values were read off the call
sites; they are unrelated numbers, not offsets from a common base.
"""

from __future__ import annotations

#: Eval / validation passes (``run_core.py:1421``, used at ``:1462``).
CORE_EVAL_SEED_OFFSET = 2000

#: ``run_baseline_training``'s no-memory passes -- the ``n_base`` samples that
#: estimate the per-problem baseline ``b_q`` (``run_core.py:1193``). **M1 must use
#: this one**, not :data:`CORE_EVAL_SEED_OFFSET`; ``b_q`` is a training-item quantity,
#: and drawing it from the eval namespace would make it a replay of an eval sample.
CORE_BASELINE_SEED_OFFSET = 1000

#: Training solve attempts (``run_core.py:1993``).
CORE_TRAIN_SOLVE_SEED_OFFSET = 3000

#: Reflection generation (``run_core.py:2603``).
CORE_REFLECTION_SEED_OFFSET = 4000

#: Lesson-admission trials, in ``_compute_admission_seed``: 5000 for a global trial,
#: 5001 when the trial is per-lesson.
CORE_ADMISSION_SEED_OFFSETS = (5000, 5001)

#: The biased training-item sampler (``_sample_biased_training_item``). Unlike every
#: constant above, this one is applied to ``args.seed`` rather than ``args.llm_seed``
#: -- it seeds problem *selection*, not generation. Unused here (M6 is not
#: implemented); kept so `test_core_fidelity` can still check it against CORE.
CORE_BIASED_SAMPLER_SEED_OFFSET = 7000


def core_generation_seed(base_seed: int, *components: int) -> int:
    """CORE's per-call seed derivation.

    Transcribed from ``core-reasoning/scripts/run_core.py:1063-1067`` rather than
    imported, because ``run_core.py`` calls ``os.chdir(ROOT)`` at import time (line 24)
    and importing it would silently move our process into the CORE checkout. Four lines
    of pure arithmetic -- no verifier, prompt or generator is duplicated here.
    """
    seed = int(base_seed)
    for component in components:
        seed = (seed * 6364136223846793005 + int(component) + 1442695040888963407) % (2**31 - 1)
    return seed
