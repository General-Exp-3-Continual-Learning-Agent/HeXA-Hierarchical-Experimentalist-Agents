# Cross-level meta-transfer

Stage 1 of the cross-level experiment: a teacher LLM reads expert (fully-evolved) skill banks from N **source** levels plus factual scene descriptions of a **target** level, and synthesises a target-level skill bank — *without ever seeing target-level trajectories*. Stage 2 evaluates that synthesised bank using the offline loop.

## Why this is interesting

The headline HeXA skill banks accumulate physics priors that look transferable on inspection (lever arms, drop position, collision angle, gap geometry). Cross-level transfer asks: can a teacher unbundle the level-specific cosmetics from the underlying physics primitives and re-ground them on a new scene it has never seen solved?

## Hard constraints baked into the prompt

The synthesis prompt in [skillrl/distillation/cross_level_synthesis.py](../skillrl/distillation/cross_level_synthesis.py) enforces:

1. **Every emitted skill cites source skills** — `{"source_level": ..., "skill_id": ...}` audit trail
2. **No invented coordinates** — the teacher has not seen the target solved, so concrete `(x, y, r)` placements are forbidden
3. **Target-side entities only** — principle and `when_to_apply` must reference the target scene's objects (skills phrased as "place ball on the ramp" cannot transfer to a level with no ramp)
4. **`transfer_rationale` field** — 1–2 sentences naming the bridging physics primitive (e.g. "torque about a pivot scales with moment arm × force")
5. **Confidence calibration** — only assign ≥0.7 when the primitive appears in ≥2 source banks AND the target scene clearly invokes it
6. **No platitudes** — banned: "use physics intuition", "consider gravity", "be careful"

Two prompt variants are auto-selected:

- **Claude teachers** (`claude-*`) get the long 8-rule prompt with the full audit envelope (`source_skills`, `transfer_rationale`)
- **Qwen / smaller teachers** get a slim 3-rule prompt with no audit fields (the longer prompt produced platitude-heavy hallucinations on Qwen-7B)

## Pipeline

```
source bank: down_to_earth   ─┐
source bank: two_body_problem├─▶ teacher synthesises ─▶ skill_bank_xl_<target>.json
source bank: pass_the_parcel ─┘                        + audit JSON

       │
       ▼
   stage 2 (offline_loop)
       │
       ▼
   target-level evaluation
```

`scripts/run_cross_level.py` is a composite wrapper that runs both stages in sequence:

```bash
python scripts/run_cross_level.py \
    --target catapult \
    --source-bank down_to_earth=path/to/skill_bank_evolving_<N>.json \
    --source-bank two_body_problem=path/to/skill_bank_evolving_<N>.json \
    --source-bank pass_the_parcel=path/to/skill_bank_evolving_<N>.json \
    --synthesis-output-dir results/cross_level/catapult \
    --offline-output-dir results/cross_level/catapult/offline_eval \
    --eval-seeds 6 7 8 9 10 11 12 13 14 15 \
    --teacher-model claude-sonnet-4-6
```

Source banks are the fully-evolved per-level banks produced by `scripts/run_hexa.py` (round-N output: `skill_bank_evolving_<N>.json`). This release ships only code; obtain the paper's headline banks by re-running HeXA per source level, or pull them from the companion data archive.

The synthesised bank is staged as `<offline-output-dir>/skill_bank_offline.json`; `offline_loop` reuses an existing bank by that name verbatim, so step 2 runs zero extra distillation calls.

## Existing results

The paper ran cross-level on three target levels: `catapult`, `falling_into_place` (Qwen-7B teacher), and `two_body_problem` (Qwen-7B teacher). Each run produces a synthesised bank (`skill_bank_xl_<target>.json`), an audit sidecar with `source_skills` citations, the raw teacher response (`_synthesis_raw_<target>.txt`), and the offline-evaluation outputs. The pre-computed outputs are not bundled in this code-only release.

## Optional `--structural-hint`

For small teachers that hallucinate mechanisms (Qwen-7B inventing levers on a level with no pivot), you can pass a one-paragraph hint relating the source and target scenes — see the docstring in `cross_level_synthesis.py` for guidance. The Claude variants generally don't need it.
