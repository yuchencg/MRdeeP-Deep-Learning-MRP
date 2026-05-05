"""Discovery, validation, and caching layer for MRP estimate CSVs.

The loader scans `data/state/` (and, when populated, `data/county/`) for files
matching the project's naming convention and builds a registry mapping
`outcome_id -> {model_id: filepath}`. New CSVs are picked up automatically
on the next process start — no code change is required to add a model or
outcome. Reads are validated against the expected schema and cached in
memory so repeated callbacks do not re-parse the same file.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import pandas as pd

from constants import (
    COUNTY_DIR,
    COUNTY_SUFFIX,
    FILENAME_TOKEN_PATTERN,
    MODELS_FILE,
    OUTCOMES_FILE,
    REQUIRED_COUNTY_COLUMNS,
    REQUIRED_STATE_COLUMNS,
    STATE_DIR,
    STATE_SUFFIX,
    STATE_FIPS_WIDTH,
    COUNTY_FIPS_WIDTH,
)

logger = logging.getLogger(__name__)


# --- Registry types -------------------------------------------------------


@dataclass(frozen=True)
class Registry:
    """Index of available estimate files by geography level.

    Maps `outcome_id -> model_id -> Path`. A `(state, county)` registry pair
    is built at app startup; downstream code never touches the filesystem
    directly.
    """

    state: dict[str, dict[str, Path]] = field(default_factory=dict)
    county: dict[str, dict[str, Path]] = field(default_factory=dict)

    def models_for(self, outcome_id: str, level: str = "state") -> list[str]:
        """Return model_ids that have a CSV for the given outcome and level.

        Returns an empty list (not an error) if the outcome is unknown for
        that level — callers can render a friendly empty state.
        """
        target = self.state if level == "state" else self.county
        return sorted(target.get(outcome_id, {}).keys())

    def outcomes(self, level: str = "state") -> list[str]:
        """Return outcome_ids that have at least one model file at this level."""
        target = self.state if level == "state" else self.county
        return sorted(target.keys())


# --- Discovery ------------------------------------------------------------


def _parse_filename(
    name: str, suffix: str, known_models: set[str]
) -> tuple[str, str] | None:
    """Split `<model_id>_<outcome_id><suffix>` using the known-model catalog.

    Both ids are snake_case and either may contain underscores, so the
    only unambiguous way to split is to try each known model_id as a
    prefix and pick the longest match. Returns (model_id, outcome_id) or
    None if no known model matches.
    """
    if not name.endswith(suffix):
        return None
    stem = name[: -len(suffix)]
    candidates = [m for m in known_models if stem == m or stem.startswith(f"{m}_")]
    if not candidates:
        return None
    model = max(candidates, key=len)
    outcome = stem[len(model) + 1 :] if stem != model else ""
    if not outcome or not FILENAME_TOKEN_PATTERN.match(outcome):
        return None
    return model, outcome


def _scan_dir(
    directory: Path, suffix: str, known_models: set[str]
) -> dict[str, dict[str, Path]]:
    """Walk a directory and group matching CSVs by outcome then model.

    Files whose model prefix isn't in `known_models` are skipped with a
    warning — contributors can stage WIP files in the folder without
    breaking the app. Returns an empty dict if the directory is missing.
    """
    registry: dict[str, dict[str, Path]] = {}
    if not directory.exists():
        return registry
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix != ".csv":
            continue
        parsed = _parse_filename(path.name, suffix, known_models)
        if parsed is None:
            logger.warning(
                "Skipping %s — no known model_id prefix in filename", path.name
            )
            continue
        model_id, outcome_id = parsed
        registry.setdefault(outcome_id, {})[model_id] = path
    return registry


def build_registry(
    state_dir: Path = STATE_DIR,
    county_dir: Path = COUNTY_DIR,
    known_models: set[str] | None = None,
) -> Registry:
    """Build the file registry for both geography levels.

    `known_models` defaults to the model_id column in `models.csv`. Pass
    a custom set in tests to point at a fixture directory without needing
    a separate models.csv on disk.
    """
    if known_models is None:
        known_models = set(load_models()["model_id"].astype(str))
    return Registry(
        state=_scan_dir(state_dir, STATE_SUFFIX, known_models),
        county=_scan_dir(county_dir, COUNTY_SUFFIX, known_models),
    )


# --- Schema validation ----------------------------------------------------


class SchemaError(ValueError):
    """Raised when a CSV does not conform to the expected estimate schema."""


def _validate_estimates(
    df: pd.DataFrame, required_columns: Iterable[str], fips_col: str, fips_width: int
) -> pd.DataFrame:
    """Coerce and validate an estimates DataFrame in place-ish.

    Verifies required columns exist, casts the FIPS column to a zero-padded
    string, casts `n_respondents` to int, casts `estimate` to float, and
    checks that estimates lie in [0, 1]. Raises SchemaError with a clear
    message on the first violation.
    """
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise SchemaError(f"Missing required columns: {missing}")

    df = df[list(required_columns)].copy()

    # FIPS as zero-padded string, regardless of source dtype.
    df[fips_col] = (
        df[fips_col]
        .astype(str)
        .str.split(".", n=1)
        .str[0]
        .str.zfill(fips_width)
    )
    bad_fips = df[fips_col].str.len() != fips_width
    if bad_fips.any():
        raise SchemaError(
            f"{fips_col} values must be {fips_width}-digit strings; "
            f"got {df.loc[bad_fips, fips_col].tolist()[:3]} ..."
        )

    df["estimate"] = df["estimate"].astype(float)
    out_of_range = ~df["estimate"].between(0.0, 1.0)
    if out_of_range.any():
        raise SchemaError(
            f"`estimate` values must lie in [0, 1]; "
            f"found {df.loc[out_of_range, 'estimate'].tolist()[:3]} ..."
        )

    df["n_respondents"] = df["n_respondents"].astype(int)
    return df


# --- Cached loaders -------------------------------------------------------
# lru_cache keyed on the absolute path string — Path objects hash by value
# but the explicit string is easier to invalidate in tests via cache_clear.


@lru_cache(maxsize=128)
def _load_state_csv(path_str: str) -> pd.DataFrame:
    df = pd.read_csv(path_str)
    return _validate_estimates(
        df, REQUIRED_STATE_COLUMNS, "state_fips", STATE_FIPS_WIDTH
    )


@lru_cache(maxsize=128)
def _load_county_csv(path_str: str) -> pd.DataFrame:
    df = pd.read_csv(path_str)
    return _validate_estimates(
        df, REQUIRED_COUNTY_COLUMNS, "county_fips", COUNTY_FIPS_WIDTH
    )


def load_estimates(path: Path, level: str = "state") -> pd.DataFrame:
    """Load and validate one estimates CSV. Returns a cached copy."""
    if level == "state":
        return _load_state_csv(str(path))
    if level == "county":
        return _load_county_csv(str(path))
    raise ValueError(f"Unknown level {level!r}; expected 'state' or 'county'.")


# --- Metadata loaders -----------------------------------------------------


def load_outcomes(path: Path = OUTCOMES_FILE) -> pd.DataFrame:
    """Load the human-readable outcome catalog (outcome_id, label, description)."""
    df = pd.read_csv(path)
    required = {"outcome_id", "label", "description"}
    missing = required - set(df.columns)
    if missing:
        raise SchemaError(f"outcomes.csv missing columns: {sorted(missing)}")
    return df


def load_models(path: Path = MODELS_FILE) -> pd.DataFrame:
    """Load the human-readable model catalog (model_id, label, description)."""
    df = pd.read_csv(path)
    required = {"model_id", "label", "description"}
    missing = required - set(df.columns)
    if missing:
        raise SchemaError(f"models.csv missing columns: {sorted(missing)}")
    return df


def clear_caches() -> None:
    """Invalidate cached CSV reads. Used by tests; not called at runtime."""
    _load_state_csv.cache_clear()
    _load_county_csv.cache_clear()
