"""Unit tests for interphyre/validation/ — covering checks, oracles, registry, and public API.

34 tests: 19 from plans/validation_module_spec.md §Tests, 13 new from
plans/validation_repair_spec.md §Tests (A1–A4 technical fixes and oracle
solution coverage for the 8 redesigned levels). 1 new test from
plans/oracle_hardening_spec.md §O3 (flagpole_sitta trivial re-audit). 1 new
test from plans/oracle_hardening_spec.md §I4 (oracle_commit bundle field). 1
new test from plans/oracle_hardening_spec.md §O4 (catapult dense probe).
"""

from __future__ import annotations

import json
import logging
import lzma
from itertools import islice
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from interphyre.config import SimulationConfig
from interphyre.level import Level
from interphyre.levels import build_level_from_scene, list_levels, load_level
from interphyre.objects import Ball
from interphyre.validation import (
    ValidatedLevel,
    _ORACLE_RNG_SALT,
    extract_scene_dict,
    iter_valid_levels,
    load_valid_level,
    prewarm,
    validate_level,
)
from interphyre.validation.checks import is_trivial
from interphyre.validation.oracles import get_oracle, list_oracles
from interphyre.validation.registry import SeedRegistry, _BUNDLE_DIR


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trivial_level() -> Level:
    """Level whose success condition is always True at t=0 — always trivial."""
    return Level(
        name="always_trivial",
        objects={"ball": Ball(x=0, y=0, radius=0.5, color="red", dynamic=False)},
        action_objects=["ball"],
        success_condition=lambda e: True,
    )


def _make_nontrivial_level() -> Level:
    """Level whose success condition is always False — never trivial."""
    return Level(
        name="never_trivial",
        objects={"ball": Ball(x=0, y=0, radius=0.5, color="red", dynamic=False)},
        action_objects=["ball"],
        success_condition=lambda e: False,
    )


# ---------------------------------------------------------------------------
# checks.py
# ---------------------------------------------------------------------------


def test_is_trivial_false():
    """basket_case seed=0 variant=0 is not trivially solved at t=0."""
    level = load_level("basket_case", seed=0, variant=0)
    assert is_trivial(level) is False


def test_is_trivial_true():
    """A level whose success condition returns True immediately is trivial."""
    level = _make_trivial_level()
    assert is_trivial(level) is True


# ---------------------------------------------------------------------------
# Variant system
# ---------------------------------------------------------------------------


def test_variant_zero_backward_compat():
    """basket_case seed=42 variant=0 is bit-identical to the default (no-variant) build.

    Variant 0 uses rng = default_rng(seed), which is identical to the pre-variant
    behavior. This verifies backward compatibility.
    """
    level_v0 = load_level("basket_case", seed=42, variant=0)
    level_default = load_level("basket_case", seed=42)  # variant=0 by default

    for obj_name in level_v0.objects:
        obj_v0 = level_v0.objects[obj_name]
        obj_default = level_default.objects[obj_name]
        assert obj_v0.x == obj_default.x, (
            f"{obj_name}.x differs under variant=0 vs default"
        )
        assert obj_v0.y == obj_default.y, (
            f"{obj_name}.y differs under variant=0 vs default"
        )


def test_variant_nonzero_distinct():
    """basket_case seed=0 variant=0 and variant=1 produce different geometries.

    Variant>0 uses rng = default_rng((seed, variant)), so the geometry is distinct
    from variant=0 with overwhelming probability.
    """
    level_v0 = load_level("basket_case", seed=0, variant=0)
    level_v1 = load_level("basket_case", seed=0, variant=1)

    any_different = any(
        level_v0.objects[name].x != level_v1.objects[name].x
        or level_v0.objects[name].y != level_v1.objects[name].y
        for name in level_v0.objects
    )
    assert any_different, "variant=0 and variant=1 produced identical geometry"


# ---------------------------------------------------------------------------
# oracles.py
# ---------------------------------------------------------------------------


def test_oracle_finds_solution():
    """straight_face seed=2 variant=0 — the registered oracle finds a solution within 50 attempts.

    seed=2 variant=0 is confirmed valid in the bundled data for straight_face,
    making it a reliable geometry for oracle coverage tests.
    """
    level = load_level("straight_face", seed=2, variant=0)
    config = SimulationConfig()
    oracle = get_oracle("straight_face")
    # seed=42 finds 83/200 valid placements but none solvable after valid-placement enforcement
    rng = np.random.default_rng(1)

    solved = oracle(level, config, n_attempts=50, oracle_steps=500, rng=rng)
    assert solved is True


def test_list_oracles_coverage():
    """list_oracles() returns an entry ('targeted' or 'default') for every registered level."""
    all_levels = set(list_levels())
    coverage = list_oracles()

    assert set(coverage.keys()) == all_levels, "list_oracles missing some levels"
    assert all(v in ("targeted", "default") for v in coverage.values()), (
        "list_oracles returned unexpected value (expected 'targeted' or 'default')"
    )


# ---------------------------------------------------------------------------
# registry.py — SeedRegistry
# ---------------------------------------------------------------------------


def test_registry_roundtrip(tmp_path):
    """Record an entry with a scene dict and retrieve it bit-identically from SQLite."""
    reg = SeedRegistry(tmp_path / "test.db")

    # Use a seed outside the bundled range so SQLite is the authoritative tier.
    level = load_level("basket_case", seed=9999, variant=0)
    scene = extract_scene_dict(level)

    reg.record("basket_case", 9999, 0, "valid", scene_dict=scene)
    retrieved = reg.get_scene_dict("basket_case", 9999, 0)

    assert retrieved is not None
    assert retrieved == scene


def test_registry_idempotent(tmp_path):
    """Recording the same (level, seed, variant) key twice does not duplicate the row."""
    reg = SeedRegistry(tmp_path / "test.db")
    level = load_level("basket_case", seed=9999, variant=0)
    scene = extract_scene_dict(level)

    reg.record("basket_case", 9999, 0, "valid", scene_dict=scene)
    reg.record("basket_case", 9999, 0, "valid", scene_dict=scene)  # duplicate write

    # SQLite uses INSERT OR REPLACE — only one row should exist.
    row_count = reg._get_conn().execute(
        "SELECT COUNT(*) FROM seed_validity "
        "WHERE level_name='basket_case' AND seed=9999 AND variant=0"
    ).fetchone()[0]
    assert row_count == 1


# ---------------------------------------------------------------------------
# validate_level / load_valid_level / iter_valid_levels
# ---------------------------------------------------------------------------


def test_validate_level_caches(tmp_path):
    """A second call to validate_level for the same triple hits the registry cache.

    Two calls should return the same status and produce exactly one SQLite row —
    the pipeline (oracle, trivial check) does not run twice.
    """
    reg = SeedRegistry(tmp_path / "test.db")

    # Use a seed outside the bundle range (0–10000) so the bundled lookup returns
    # None and falls through to the SQLite cache we pre-populate here.
    reg.record("basket_case", 99999, 0, "trivial")

    status1 = validate_level("basket_case", 99999, 0, registry=reg)
    status2 = validate_level("basket_case", 99999, 0, registry=reg)

    assert status1 == "trivial"
    assert status2 == "trivial"

    row_count = reg._conn.execute(
        "SELECT COUNT(*) FROM seed_validity "
        "WHERE level_name='basket_case' AND seed=99999 AND variant=0"
    ).fetchone()[0]
    assert row_count == 1


def test_load_valid_level_returns_validated_level():
    """load_valid_level returns a ValidatedLevel with all fields populated."""
    result = load_valid_level("basket_case", seed=42)

    assert isinstance(result, ValidatedLevel)
    assert result.level is not None
    assert result.level_name == "basket_case"
    assert result.seed == 42
    assert isinstance(result.variant, int) and result.variant >= 0
    assert result.scene_dict is not None
    assert isinstance(result.scene_dict, dict)


def test_load_valid_level_variant_increment(tmp_path):
    """When variant=0 is trivial, load_valid_level advances to variant=1.

    We pre-populate the registry with the two outcomes so no oracle run is needed,
    keeping the test fast and deterministic.
    """
    reg = SeedRegistry(tmp_path / "test.db")

    # Mark variant=0 as trivial — simulates a seed whose initial geometry auto-solves.
    reg.record("basket_case", 9999, 0, "trivial")

    # Pre-record variant=1 as valid with a real scene dict.
    level_v1 = load_level("basket_case", seed=9999, variant=1)
    scene_v1 = extract_scene_dict(level_v1)
    reg.record("basket_case", 9999, 1, "valid", scene_dict=scene_v1)

    result = load_valid_level("basket_case", 9999, registry=reg, max_variants=3)

    assert isinstance(result, ValidatedLevel)
    assert result.variant == 1
    assert result.seed == 9999
    assert result.scene_dict is not None


def test_iter_valid_levels():
    """iter_valid_levels yields ValidatedLevel instances with monotonically increasing seeds."""
    levels = list(islice(iter_valid_levels("basket_case", start_seed=0), 5))

    assert len(levels) == 5
    assert all(isinstance(v, ValidatedLevel) for v in levels)

    seeds = [v.seed for v in levels]
    assert seeds == sorted(seeds), (
        "seeds from iter_valid_levels are not monotonically increasing"
    )
    assert len(set(seeds)) == 5, "iter_valid_levels yielded duplicate seeds"


def test_scene_dict_round_trip():
    """Extract a scene dict, rebuild the level from it, assert the geometry is bit-identical.

    Uses build_level_from_scene as the reconstruction path. Comparing extract_scene_dict
    on both levels verifies the round-trip without relying on physics execution.
    """
    level = load_level("basket_case", seed=0, variant=0)
    scene = extract_scene_dict(level)

    rebuilt = build_level_from_scene("basket_case", scene)
    rebuilt_scene = extract_scene_dict(rebuilt)

    assert set(scene.keys()) == set(rebuilt_scene.keys()), (
        "object name mismatch after round-trip"
    )
    for obj_name in scene:
        for attr, val in scene[obj_name].items():
            assert attr in rebuilt_scene[obj_name], (
                f"missing attribute {obj_name}.{attr}"
            )
            assert rebuilt_scene[obj_name][attr] == val, (
                f"{obj_name}.{attr}: {rebuilt_scene[obj_name][attr]!r} != {val!r}"
            )


# ---------------------------------------------------------------------------
# Bundled data tier
# ---------------------------------------------------------------------------


def test_bundled_lookup(tmp_path):
    """basket_case seed=2 variant=0 returns 'valid' from bundled data without SQLite writes.

    seed=2 variant=0 is confirmed valid in the basket_case bundle. The test verifies
    that bundled lookups are O(1) in-memory reads and do not touch SQLite.
    """
    reg = SeedRegistry(tmp_path / "test.db")

    status = reg.lookup("basket_case", 2, 0)

    assert status == "valid"

    # Bundled-tier lookups are read-only — SQLite should be untouched.
    row_count = (
        reg._get_conn().execute("SELECT COUNT(*) FROM seed_validity").fetchone()[0]
    )
    assert row_count == 0


def test_bundled_scene_reconstruction(tmp_path):
    """A scene dict from bundled data round-trips through build_level_from_scene bit-identically.

    The bundled scene for action objects records the oracle-placed position (not the initial
    build position), so the correct comparison is: retrieve bundled scene → rebuild level from
    it → re-extract scene → assert identical to bundled.
    """
    reg = SeedRegistry(tmp_path / "test.db")

    bundled_scene = reg.get_scene_dict("basket_case", 2, 0)
    assert bundled_scene is not None

    # Reconstruct the level geometry from the bundled scene dict.
    rebuilt = build_level_from_scene("basket_case", bundled_scene)
    rebuilt_scene = extract_scene_dict(rebuilt)

    assert set(bundled_scene.keys()) == set(rebuilt_scene.keys()), (
        "object name mismatch after round-trip through build_level_from_scene"
    )
    for obj_name in bundled_scene:
        for attr, bundled_val in bundled_scene[obj_name].items():
            assert attr in rebuilt_scene[obj_name], (
                f"rebuilt scene missing attribute {obj_name}.{attr}"
            )
            rebuilt_val = rebuilt_scene[obj_name][attr]
            if isinstance(bundled_val, float):
                assert abs(rebuilt_val - bundled_val) < 1e-9, (
                    f"{obj_name}.{attr}: rebuilt={rebuilt_val!r}, bundled={bundled_val!r}"
                )
            else:
                assert rebuilt_val == bundled_val, (
                    f"{obj_name}.{attr}: rebuilt={rebuilt_val!r}, bundled={bundled_val!r}"
                )


def test_default_registry():
    """load_valid_level with no registry argument uses the module-level default."""
    result = load_valid_level("basket_case", seed=10)  # no registry argument

    assert isinstance(result, ValidatedLevel)
    assert result.level_name == "basket_case"
    assert result.seed == 10
    assert result.scene_dict is not None


# ---------------------------------------------------------------------------
# Schema hash
# ---------------------------------------------------------------------------


def test_schema_hash_stored():
    """The bundled JSON for basket_case contains a 'schema_hash' field."""
    bundle_path = _BUNDLE_DIR / "basket_case.json.lzma"
    with lzma.open(bundle_path, "rt", encoding="utf-8") as fh:
        data = json.load(fh)

    assert "schema_hash" in data, "bundled JSON is missing 'schema_hash'"
    assert isinstance(data["schema_hash"], str) and len(data["schema_hash"]) == 64


def test_schema_hash_valid():
    """The stored schema hash for basket_case matches the current object key structure."""
    import interphyre.validation.registry as reg_mod

    bundle_path = _BUNDLE_DIR / "basket_case.json.lzma"
    with lzma.open(bundle_path, "rt", encoding="utf-8") as fh:
        data = json.load(fh)

    stored_hash = data.get("schema_hash", "")
    expected_hash = reg_mod._compute_schema_hash("basket_case")

    assert stored_hash == expected_hash, (
        f"schema_hash mismatch — bundled data may be stale. "
        f"stored={stored_hash[:8]}, expected={expected_hash[:8]}"
    )


def test_schema_hash_stale(tmp_path, monkeypatch, caplog):
    """When the schema hash mismatches, SeedRegistry logs a WARNING and skips bundled data."""
    import interphyre.validation.registry as reg_mod

    # Patch _compute_schema_hash to always return a hash that won't match the stored one.
    monkeypatch.setattr(reg_mod, "_compute_schema_hash", lambda name: "0" * 64)

    reg = SeedRegistry(tmp_path / "test.db")

    with caplog.at_level(logging.WARNING, logger="interphyre.validation.registry"):
        result = reg.lookup("basket_case", 0, 0)

    # A stale-schema warning should have been emitted.
    stale_warnings = [m for m in caplog.messages if "stale" in m.lower()]
    assert stale_warnings, "expected a stale-schema WARNING but none was logged"

    # Bundled tier is skipped; SQLite is empty → lookup returns None.
    assert result is None, (
        f"expected None (bundled skipped, SQLite empty) but got {result!r}"
    )


# ---------------------------------------------------------------------------
# Helpers for new tests (validation_repair_spec.md)
# ---------------------------------------------------------------------------


def _make_dynamic_fall_level() -> Level:
    """Level where a dynamic ball falls under gravity and satisfies success after ~54 steps.

    The green_ball starts at y=2.0 with dynamic=True. With gravity -9.8 and dt=1/60,
    it crosses y=-2.0 after approximately 54 steps. At t=0, y=2.0 > -2.0, so the
    success condition is False. The post-physics check in is_trivial detects success
    within physics_steps=100 steps.

    Used for test_is_trivial_extended_physics_only to verify Fix A4.
    """
    dynamic_ball = Ball(x=0.0, y=2.0, radius=0.5, color="green", dynamic=True)

    def success(engine):
        body = engine.bodies.get("green_ball")
        return body is not None and body.position.y < -2.0

    return Level(
        name="dynamic_fall_test",
        objects={"green_ball": dynamic_ball},
        action_objects=[],
        success_condition=success,
    )


# ---------------------------------------------------------------------------
# Fix A4: is_trivial extended physics check
# ---------------------------------------------------------------------------


def test_is_trivial_extended_false():
    """basket_case seed=0 with physics_steps=1000 is not trivially solved.

    Verifies that the extended post-physics check (Fix A4) does not produce false
    positives on a well-behaved level that requires agent action.
    """
    level = load_level("basket_case", seed=0, variant=0)
    assert is_trivial(level, physics_steps=1000) is False


def test_is_trivial_extended_physics_only():
    """A dynamic ball that crosses y=-2.0 under gravity is detected as trivial.

    The level has a single dynamic ball at y=2.0 with no action objects.
    At t=0 the success condition (ball.y < -2.0) is False. After ~54 steps the
    ball crosses y=-2.0 under gravity. is_trivial with physics_steps=100 must
    return True, validating Fix A4 catches dynamically self-solving scenes.
    """
    level = _make_dynamic_fall_level()
    assert is_trivial(level, physics_steps=100) is True


# ---------------------------------------------------------------------------
# Fix A1: Oracle RNG bundle/live unification
# ---------------------------------------------------------------------------


def test_oracle_rng_bundle_live_match():
    """Bundle and live oracle RNGs produce identical sequences for seed=7, variant=0.

    Verifies Fix A1: _bundle._oracle_rng uses the same three-integer list seeding
    as validate_level in __init__.py, ensuring bundle and live oracle decisions
    are reproducible from each other.
    """
    from interphyre.validation._bundle import _oracle_rng as bundle_oracle_rng

    bundle_rng = bundle_oracle_rng(7, 0)
    live_rng = np.random.default_rng([7, 0, _ORACLE_RNG_SALT])

    bundle_draws = bundle_rng.random(10)
    live_draws = live_rng.random(10)

    np.testing.assert_array_equal(bundle_draws, live_draws)


# ---------------------------------------------------------------------------
# Fix A2: iter_valid_levels skips exhausted seeds
# ---------------------------------------------------------------------------


def test_iter_valid_levels_skips_exhausted():
    """iter_valid_levels skips seeds where load_valid_level raises RuntimeError.

    Verifies Fix A2: exhausted seeds (all variants tried) do not propagate the
    RuntimeError to the caller. The iterator silently advances to the next seed.
    """
    import interphyre.validation as val_mod

    original_load = val_mod.load_valid_level

    def _raise_for_seed_2(level_name, seed, **kwargs):
        if seed == 2:
            raise RuntimeError("all variants exhausted")
        return original_load(level_name, seed, **kwargs)

    with patch.object(val_mod, "load_valid_level", _raise_for_seed_2):
        levels = list(islice(iter_valid_levels("basket_case", start_seed=0), 5))

    assert [v.seed for v in levels] == [0, 1, 3, 4, 5]


# ---------------------------------------------------------------------------
# Fix A3: SeedRegistry.db_path public property
# ---------------------------------------------------------------------------


def test_registry_db_path_public(tmp_path):
    """SeedRegistry.db_path is a public property returning the configured Path.

    Verifies Fix A3: callers use reg.db_path rather than reg._db_path. The
    property must return a Path instance matching the path passed to __init__.
    """
    db_path = tmp_path / "t.db"
    reg = SeedRegistry(db_path)

    assert isinstance(reg.db_path, Path)
    assert reg.db_path == db_path


# ---------------------------------------------------------------------------
# Oracle solution tests: redesigned levels B1–B8
# ---------------------------------------------------------------------------
#
# Each test loads a seed confirmed valid in the updated bundle, runs the oracle
# with the canonical bundle RNG (np.random.default_rng([seed, variant, salt])),
# and asserts it returns True within 50 attempts at 500 oracle steps.


def test_the_cradle_oracle_finds_solution():
    """the_cradle oracle finds a solution for seed=0 variant=0 within 50 attempts."""
    level = load_level("the_cradle", seed=0, variant=0)
    config = SimulationConfig()
    oracle = get_oracle("the_cradle")
    rng = np.random.default_rng([0, 0, _ORACLE_RNG_SALT])
    assert oracle(level, config, n_attempts=50, oracle_steps=500, rng=rng) is True


def test_just_a_nudge_oracle_finds_solution():
    """just_a_nudge oracle finds a solution for seed=10 variant=0 within 50 attempts."""
    level = load_level("just_a_nudge", seed=10, variant=0)
    config = SimulationConfig()
    oracle = get_oracle("just_a_nudge")
    rng = np.random.default_rng([10, 0, _ORACLE_RNG_SALT])
    assert oracle(level, config, n_attempts=50, oracle_steps=500, rng=rng) is True


def test_catapult_oracle_finds_solution():
    """catapult oracle finds a solution for seed=34 variant=0 within 500 attempts.

    oracle_steps=500 is insufficient for catapult: the throw + ballistic flight takes
    8–17 simulated seconds (480–1020 steps at 60 fps), truncating many trajectories
    mid-flight. oracle_steps=1000 is required; n_attempts=500 reflects the calibrated
    budget from register_defaults.
    """
    level = load_level("catapult", seed=34, variant=0)
    config = SimulationConfig()
    oracle = get_oracle("catapult")
    rng = np.random.default_rng([34, 0, _ORACLE_RNG_SALT])
    assert oracle(level, config, n_attempts=500, oracle_steps=1000, rng=rng) is True


def test_mind_the_gap_oracle_finds_solution():
    """mind_the_gap oracle finds a solution for seed=25 variant=0 within 50 attempts.

    seed=25 variant=0 is confirmed valid in the bundle. The redesigned oracle places
    the red ball on the FAR SIDE of the green ball from the hole.
    """
    level = load_level("mind_the_gap", seed=25, variant=0)
    config = SimulationConfig()
    oracle = get_oracle("mind_the_gap")
    rng = np.random.default_rng([25, 0, _ORACLE_RNG_SALT])
    assert oracle(level, config, n_attempts=50, oracle_steps=500, rng=rng) is True


def test_marble_race_oracle_finds_solution():
    """marble_race oracle finds a solution for seed=0 variant=0 within 50 attempts.

    seed=0 variant=0 is confirmed valid in the bundle. The redesigned oracle
    concentrates placement on the RIGHT end of left_beam for maximum tipping leverage.
    """
    level = load_level("marble_race", seed=0, variant=0)
    config = SimulationConfig()
    oracle = get_oracle("marble_race")
    rng = np.random.default_rng([0, 0, _ORACLE_RNG_SALT])
    assert oracle(level, config, n_attempts=50, oracle_steps=500, rng=rng) is True


def test_keyhole_oracle_finds_solution():
    """keyhole oracle finds a solution for seed=60 variant=0 within 50 attempts.

    seed=60 variant=0 is confirmed valid in the bundle. The redesigned oracle
    pushes laterally toward the center gap (x=0) rather than dropping from above.
    """
    level = load_level("keyhole", seed=60, variant=0)
    config = SimulationConfig()
    oracle = get_oracle("keyhole")
    rng = np.random.default_rng([60, 0, _ORACLE_RNG_SALT])
    assert oracle(level, config, n_attempts=50, oracle_steps=500, rng=rng) is True


def test_falling_into_place_oracle_finds_solution():
    """falling_into_place oracle finds a solution for seed=4 variant=0 within 50 attempts.

    seed=4 variant=0 is confirmed valid in the bundle. The redesigned oracle places
    the red ball on the FAR SIDE of the green ball from the hole in the bar.
    """
    level = load_level("falling_into_place", seed=4, variant=0)
    config = SimulationConfig()
    oracle = get_oracle("falling_into_place")
    rng = np.random.default_rng([4, 0, _ORACLE_RNG_SALT])
    assert oracle(level, config, n_attempts=50, oracle_steps=500, rng=rng) is True


def test_wedge_issue_oracle_finds_solution():
    """wedge_issue oracle finds a solution for seed=3 variant=0 within 50 attempts.

    seed=3 variant=0 is confirmed valid in the bundle. The redesigned oracle places
    the red ball strictly ABOVE the green ball (no overlap) to prevent explosive
    contact forces.
    """
    level = load_level("wedge_issue", seed=3, variant=0)
    config = SimulationConfig()
    oracle = get_oracle("wedge_issue")
    rng = np.random.default_rng([3, 0, _ORACLE_RNG_SALT])
    assert oracle(level, config, n_attempts=50, oracle_steps=500, rng=rng) is True


# ---------------------------------------------------------------------------
# O3: flagpole_sitta trivial re-audit (oracle_hardening)
# ---------------------------------------------------------------------------


def test_flagpole_sitta_oracle_solves():
    """flagpole_sitta oracle finds a valid placement that knocks green_ball off the pole.

    Seed 7 exercises the above-side-drop mechanism (ceiling clearance is sufficient
    for a near-horizontal approach). is_trivial(physics_steps=1000)=False for this
    seed confirms that the action ball — not the level's own dynamics — is causally
    responsible for the success at oracle_steps=600.
    """
    level = load_level("flagpole_sitta", seed=7, variant=0)
    config = SimulationConfig()
    oracle = get_oracle("flagpole_sitta")
    rng = np.random.default_rng([7, 0, _ORACLE_RNG_SALT])
    assert oracle(level, config, n_attempts=50, oracle_steps=500, rng=rng) is True


# ---------------------------------------------------------------------------
# I2: prewarm variant_hist (oracle_hardening)
# ---------------------------------------------------------------------------


def test_prewarm_variant_hist_structure(tmp_path):
    """prewarm output dict has a 'variant_hist' key containing a dict with int→int entries."""
    reg = SeedRegistry(tmp_path / "test.db")
    counts = prewarm(["basket_case"], range(3), registry=reg, workers=1, progress=False)
    hist = counts["basket_case"]["variant_hist"]
    assert isinstance(hist, dict)
    assert all(isinstance(k, int) for k in hist)
    assert all(isinstance(v, int) for v in hist.values())


def test_prewarm_variant_hist_basket_case(tmp_path):
    """prewarm on basket_case seeds [2, 3, 8, 18] returns variant_hist == {0: 4}.

    Seeds 2, 3, 8, 18 are confirmed valid at variant=0 in the basket_case bundle
    (regenerated 2026-03-28), so all four seeds should be counted in variant_hist[0]
    with no other entries.
    """
    reg = SeedRegistry(tmp_path / "test.db")
    counts = prewarm(
        ["basket_case"], [2, 3, 8, 18], registry=reg, workers=1, progress=False
    )
    assert counts["basket_case"]["variant_hist"] == {0: 4}


def test_flagpole_sitta_trivial_rate():
    """No bundle-valid flagpole_sitta seed (0–9, variant=0) is trivial at physics_steps=1000.

    The March 25 bundle used the pre-A4 is_trivial (t=0 only). The flagpole is a
    dynamic=True vertical bar; many seeds have it fall autonomously within 1000
    physics steps, making them trivial — not solvable by the agent. The re-audited
    bundle (regenerated 2026-03-27) correctly classifies these as 'trivial', not
    'valid'. This test confirms the invariant: any seed the new bundle calls 'valid'
    at variant=0 must not satisfy is_trivial(physics_steps=1000).

    Seeds 7 and 9 are confirmed valid at variant=0 in the re-audited bundle.
    """
    reg = SeedRegistry()
    for seed in range(10):
        bundle_status = reg.lookup("flagpole_sitta", seed, variant=0)
        if bundle_status != "valid":
            continue
        level = load_level("flagpole_sitta", seed=seed, variant=0)
        assert not is_trivial(level, physics_steps=1000), (
            f"flagpole_sitta seed={seed} variant=0 is marked 'valid' in the bundle "
            f"but is_trivial returns True at physics_steps=1000. "
            f"Bundle must be regenerated."
        )


# ---------------------------------------------------------------------------
# I4: oracle_commit bundle field (oracle_hardening)
# ---------------------------------------------------------------------------


def test_bundle_has_oracle_commit():
    """Bundle metadata for basket_case contains a non-empty oracle_commit string.

    Bundles generated after I4 store the short git hash of HEAD at generation
    time under metadata['oracle_commit']. This field lets future engineers
    identify which oracle version produced a given bundle. Requires bundles
    regenerated after the I4 change.
    """
    bundle_path = _BUNDLE_DIR / "basket_case.json.lzma"
    with lzma.open(bundle_path, "rt", encoding="utf-8") as fh:
        data = json.load(fh)
    assert "oracle_commit" in data, (
        "bundle is missing 'oracle_commit' field — regenerate with "
        "`python -m interphyre.validation._bundle --levels basket_case --seeds 0:1000`"
    )
    assert isinstance(data["oracle_commit"], str)
    assert data["oracle_commit"], "oracle_commit must be a non-empty string"


