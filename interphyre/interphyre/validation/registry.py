"""Registry for validated level entries.

Two-tier lookup:
  Tier 1 — bundled lzma-compressed JSON shipped with the package
            (interphyre/data/levels/{level_name}.json.lzma). Loaded lazily
            per level, cached in-memory as {(seed, variant): entry}. O(1)
            after first access per level.
  Tier 2 — user SQLite cache at ~/.cache/interphyre/seed_registry.db,
            following the XDG convention used by PHYRE (~/.cache/phyre/).
            Configurable via INTERPHYRE_CACHE_DIR env var. WAL mode,
            check_same_thread=False for safe multi-threaded prewarm.

Writes always go to SQLite. Reads check bundled data first, then SQLite.

Schema hash: on first access to bundled data for a level, SeedRegistry
recomputes SHA-256 of the attribute key structure from extract_scene_dict
(build_level(seed=0)) and compares it to the stored schema_hash field.
On mismatch, a WARNING is logged and the bundled tier is skipped for that
level — live validation continues via SQLite. This prevents stale bundled
geometry from being served silently after constructor changes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import lzma
import os
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from interphyre.level import Level

logger = logging.getLogger(__name__)

# Bundled level data lives at interphyre/data/levels/ relative to this package.
_BUNDLE_DIR = Path(__file__).parent.parent / "data" / "levels"


@dataclass
class ValidatedLevel:
    """A level that has passed trivial and solvability checks.

    Bundles the level with its provenance so experiments can cite it exactly.
    The (level_name, seed, variant) triple is the short-form reference;
    scene_dict is the long-form reproducibility artifact.
    """

    level: Level
    level_name: str
    seed: int
    variant: int
    scene_dict: dict  # full geometry, JSON-serializable


def _default_cache_path() -> Path:
    """Return the default SQLite cache path, respecting INTERPHYRE_CACHE_DIR."""
    cache_dir_env = os.environ.get("INTERPHYRE_CACHE_DIR")
    cache_dir = (
        Path(cache_dir_env) if cache_dir_env else Path.home() / ".cache" / "interphyre"
    )
    return cache_dir / "seed_registry.db"


def _compute_schema_hash(level_name: str) -> str:
    """Return SHA-256 of the attribute key structure for level_name at seed=0.

    Hashes {object_name: sorted_attr_names} from extract_scene_dict applied
    to build_level(seed=0, variant=0). Detects constructor changes (new
    attributes, renames) that would make stored scene dicts produce wrong
    geometry on round-trip.

    Deferred imports break the import cycle: registry <- levels <- validation.
    """
    from interphyre.levels import load_level
    from interphyre.validation.checks import extract_scene_dict

    level = load_level(level_name, seed=0, variant=0)
    scene = extract_scene_dict(level)
    schema_repr = {name: sorted(attrs.keys()) for name, attrs in scene.items()}
    schema_str = json.dumps(schema_repr, sort_keys=True)
    return hashlib.sha256(schema_str.encode()).hexdigest()


class SeedRegistry:
    """Two-tier registry for validated (level_name, seed, variant) entries."""

    def __init__(self, cache_path: str | Path | None = None):
        """
        Args:
            cache_path: Override for the SQLite cache path. When None, uses
                ~/.cache/interphyre/seed_registry.db or INTERPHYRE_CACHE_DIR.
        """
        resolved = Path(cache_path) if cache_path else _default_cache_path()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = resolved

        # Per-level in-memory bundled cache: missing key = not yet loaded;
        # empty dict = no bundled data (file absent or schema stale).
        # Indexed by seed (one entry per seed — the valid variant or a single
        # impossible marker).
        self._bundled: dict[str, dict[int, dict]] = {}
        self._schema_checked: set[str] = set()

        # Per-level schema hash cache: computed once per session per level.
        # Used to validate SQLite entries so stale cached results (written by
        # an older oracle or constructor) are treated as cache misses.
        self._current_schema_hashes: dict[str, str] = {}

        # Connection is opened lazily on first SQLite access so that callers
        # working entirely within the in-memory bundle tier (seeds 0–10000)
        # never touch the database file. This eliminates SQLite locking errors
        # when multiple processes hold the registry open simultaneously but only
        # read from the in-memory bundle (e.g. test suites, concurrent experiments).
        self._conn: sqlite3.Connection | None = None

        # When False, record() skips the per-write commit. Use batched() to set
        # this temporarily so a caller can flush at a coarser granularity.
        self._auto_flush: bool = True

    @property
    def db_path(self) -> Path:
        return self._db_path

    @contextmanager
    def batched(self):
        """Defer SQLite commits until the context exits.

        Use this to batch multiple record() calls into a single fsync rather
        than committing after every write. All writes are committed atomically
        when the block exits normally; on exception the transaction is rolled
        back and _auto_flush is restored.

        Example — one commit per seed instead of one per variant:
            with registry.batched():
                for variant in range(max_variants):
                    validate_level(..., registry=registry, ...)
        """
        self._auto_flush = False
        try:
            yield
            self._get_conn().commit()
        except Exception:
            self._get_conn().rollback()
            raise
        finally:
            self._auto_flush = True

    def _get_conn(self) -> sqlite3.Connection:
        """Return the SQLite connection, opening it on first call.

        Deferred so that callers using only the in-memory bundle tier never
        open the database file — eliminating locking contention when multiple
        processes hold SeedRegistry instances simultaneously.
        """
        if self._conn is None:
            conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS seed_validity (
                    level_name    TEXT    NOT NULL,
                    seed          INTEGER NOT NULL,
                    variant       INTEGER NOT NULL DEFAULT 0,
                    status        TEXT    NOT NULL,
                    scene_json    TEXT,
                    checked_at    TEXT    NOT NULL,
                    solution_json TEXT,
                    schema_hash   TEXT,
                    PRIMARY KEY (level_name, seed, variant)
                )
            """)
            # Add columns introduced after initial schema; suppressed if already present.
            for col_def in (
                "ALTER TABLE seed_validity ADD COLUMN solution_json TEXT DEFAULT NULL",
                "ALTER TABLE seed_validity ADD COLUMN schema_hash TEXT DEFAULT NULL",
            ):
                try:
                    conn.execute(col_def)
                except sqlite3.OperationalError:
                    pass
            conn.commit()
            self._conn = conn
        return self._conn

    def _get_current_schema_hash(self, level_name: str) -> str:
        """Return the current schema hash for level_name, computing it once per session."""
        if level_name not in self._current_schema_hashes:
            self._current_schema_hashes[level_name] = _compute_schema_hash(level_name)
        return self._current_schema_hashes[level_name]

    def _load_bundled(self, level_name: str) -> None:
        """Load and validate bundled lzma data for level_name into memory.

        If the bundle file is absent, _bundled[level_name] is set to an empty
        dict (no bundled data available; falls through to SQLite).

        If the schema hash mismatches, a WARNING is logged, _bundled[level_name]
        is set to an empty dict, and the bundled tier is skipped for this level
        for the remainder of the process. Experiment continues via SQLite.
        """
        self._schema_checked.add(level_name)
        bundle_path = _BUNDLE_DIR / f"{level_name}.json.lzma"

        if not bundle_path.exists():
            self._bundled[level_name] = {}
            return

        with lzma.open(bundle_path, "rt", encoding="utf-8") as fh:
            data = json.load(fh)

        stored_hash = data.get("schema_hash", "")
        current_hash = self._get_current_schema_hash(level_name)

        if stored_hash != current_hash:
            logger.warning(
                "SeedRegistry: bundled data for '%s' has a stale schema hash "
                "(stored=%.8s, current=%.8s). Bundled tier will be skipped "
                "for this level. Run `python -m interphyre.validation._bundle` "
                "to regenerate.",
                level_name,
                stored_hash,
                current_hash,
            )
            self._bundled[level_name] = {}
            return

        entries = data.get("entries", [])
        # Index by seed: one entry per seed.  Bundles generated before the
        # compact-format change may contain multiple entries per seed (one per
        # variant tried); in that case, prefer the valid entry over any
        # impossible entries so the correct scene and solution are served.
        seed_map: dict[int, dict] = {}
        for entry in entries:
            s = entry["seed"]
            if s not in seed_map or entry["status"] == "valid":
                seed_map[s] = entry
        self._bundled[level_name] = seed_map
        logger.debug(
            "SeedRegistry: loaded bundle for '%s' (%d entries, oracle_commit=%s)",
            level_name,
            len(entries),
            data.get("oracle_commit", "unknown"),
        )

    def _ensure_bundled(self, level_name: str) -> None:
        """Load bundled data for level_name on first access."""
        if level_name not in self._schema_checked:
            self._load_bundled(level_name)

    def get_valid_entry(self, level_name: str, seed: int) -> dict | None:
        """Return the stored bundle entry for seed, or None if not in the bundle.

        The entry dict has {seed, variant, status, scene, solution}.  Callers
        can inspect entry["status"] to distinguish "valid" from "impossible"
        without paying for a variant scan or oracle call.

        Only covers the bundled tier; live-validated seeds go through lookup().
        """
        self._ensure_bundled(level_name)
        return self._bundled.get(level_name, {}).get(seed)

    def lookup(self, level_name: str, seed: int, variant: int = 0) -> str | None:
        """Return the status string for (level_name, seed, variant), or None.

        Checks bundled data first (O(1) in-memory after first access per level),
        then falls back to user SQLite cache.

        For the bundled tier, a seed's entry stores the first valid variant
        (or variant=0 for impossible seeds).  A query for a different variant
        of a valid seed returns "impossible" so the caller's scan loop keeps
        advancing until it reaches the stored valid variant.
        """
        self._ensure_bundled(level_name)

        entry = self._bundled.get(level_name, {}).get(seed)
        if entry is not None:
            if entry["status"] == "valid":
                # Only the stored variant is confirmed valid.
                return "valid" if entry["variant"] == variant else "impossible"
            return entry["status"]  # "impossible"

        row = (
            self._get_conn()
            .execute(
                "SELECT status, schema_hash FROM seed_validity WHERE level_name=? AND seed=? AND variant=?",
                (level_name, seed, variant),
            )
            .fetchone()
        )
        if row is None:
            return None
        stored_hash = row[1]
        # Treat NULL or mismatched schema_hash as a cache miss so stale entries
        # written by an older oracle or constructor are re-validated automatically.
        if stored_hash != self._get_current_schema_hash(level_name):
            return None
        return row[0]

    def record(
        self,
        level_name: str,
        seed: int,
        variant: int,
        status: str,
        scene_dict: dict | None = None,
        solution: list | None = None,
    ) -> None:
        """Write or overwrite an entry in the user SQLite cache.

        solution, when provided, is a list of [x, y, radius] lists — one per
        action object.  Stored as JSON in solution_json and retrievable via
        get_solution().
        """
        scene_json = json.dumps(scene_dict) if scene_dict is not None else None
        solution_json = json.dumps(solution) if solution is not None else None
        schema_hash = self._get_current_schema_hash(level_name)
        checked_at = datetime.now(tz=timezone.utc).isoformat()
        self._get_conn().execute(
            """
            INSERT OR REPLACE INTO seed_validity
                (level_name, seed, variant, status, scene_json, checked_at, solution_json, schema_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                level_name,
                seed,
                variant,
                status,
                scene_json,
                checked_at,
                solution_json,
                schema_hash,
            ),
        )
        if self._auto_flush:
            self._get_conn().commit()

    def get_scene_dict(self, level_name: str, seed: int, variant: int) -> dict | None:
        """Return the stored scene dict from bundled data or SQLite, or None."""
        self._ensure_bundled(level_name)

        entry = self._bundled.get(level_name, {}).get(seed)
        if entry is not None and entry.get("variant") == variant:
            return entry.get("scene")

        row = (
            self._get_conn()
            .execute(
                "SELECT scene_json FROM seed_validity WHERE level_name=? AND seed=? AND variant=?",
                (level_name, seed, variant),
            )
            .fetchone()
        )
        if row and row[0]:
            return json.loads(row[0])
        return None

    def get_solution(self, level_name: str, seed: int, variant: int) -> list | None:
        """Return the stored solution from bundled data or SQLite, or None.

        The solution is a list of [x, y, radius] lists — one per action object —
        representing the winning placement found by the solver.  Returns None when
        no solution was recorded (levels without a registered solver, impossible
        seeds, or bundles generated before the solver-registry refactor).
        """
        self._ensure_bundled(level_name)

        entry = self._bundled.get(level_name, {}).get(seed)
        if entry is not None and entry.get("variant") == variant:
            return entry.get("solution")

        row = (
            self._get_conn()
            .execute(
                "SELECT solution_json FROM seed_validity WHERE level_name=? AND seed=? AND variant=?",
                (level_name, seed, variant),
            )
            .fetchone()
        )
        if row and row[0]:
            return json.loads(row[0])
        return None

    def count(self, level_name: str, status: str) -> int:
        """Count seeds with the given status across both tiers.

        Deduplicates by seed: SQLite entries whose seed is already covered by
        the bundled tier are not double-counted.
        """
        self._ensure_bundled(level_name)

        bundled = self._bundled.get(level_name, {})
        bundled_count = sum(1 for e in bundled.values() if e["status"] == status)
        bundled_seeds = set(bundled.keys())

        sql_rows = (
            self._get_conn()
            .execute(
                "SELECT seed, variant FROM seed_validity WHERE level_name=? AND status=?",
                (level_name, status),
            )
            .fetchall()
        )
        sql_count = sum(1 for row in sql_rows if row[0] not in bundled_seeds)

        return bundled_count + sql_count

    def valid_entries(self, level_name: str) -> list[tuple[int, int]]:
        """Return all (seed, variant) pairs with status 'valid', sorted by seed.

        Merges bundled and SQLite tiers; deduplicates by seed.
        """
        self._ensure_bundled(level_name)

        bundled = self._bundled.get(level_name, {})
        valid: dict[int, int] = {
            seed: entry["variant"]
            for seed, entry in bundled.items()
            if entry["status"] == "valid"
        }

        sql_rows = (
            self._get_conn()
            .execute(
                "SELECT seed, variant FROM seed_validity WHERE level_name=? AND status='valid'",
                (level_name,),
            )
            .fetchall()
        )
        for row in sql_rows:
            if row[0] not in valid:
                valid[row[0]] = row[1]

        return sorted(valid.items(), key=lambda pair: pair[0])

    def bundle_valid_rate(self, level_name: str) -> float | None:
        """Return the fraction of bundled seeds with status 'valid', or None if no bundle.

        Returns None when the level has no bundle file or the bundle is empty —
        callers should treat None as "unknown" rather than "0% valid".
        Used to warn before entering the oracle search loop on known-impossible
        or near-impossible levels.
        """
        self._ensure_bundled(level_name)
        bundled = self._bundled.get(level_name, {})
        if not bundled:
            return None
        total = len(bundled)
        valid = sum(1 for entry in bundled.values() if entry["status"] == "valid")
        return valid / total

    def close(self) -> None:
        """Close the underlying SQLite connection if it was opened."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> SeedRegistry:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
