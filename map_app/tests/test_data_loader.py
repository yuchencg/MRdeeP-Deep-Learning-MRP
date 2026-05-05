"""Tests for the file-discovery and schema-validation layer."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# Make the app package importable when pytest runs from anywhere.
APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR))

from constants import (  # noqa: E402
    REQUIRED_STATE_COLUMNS,
    STATE_SUFFIX,
    STATE_FIPS_WIDTH,
)
from data_loader import (  # noqa: E402
    SchemaError,
    _parse_filename,
    _validate_estimates,
    build_registry,
    clear_caches,
    load_estimates,
)


# --- Helpers --------------------------------------------------------------


def _write_state_csv(
    path: Path,
    *,
    fips=("01", "06", "36"),
    names=("Alabama", "California", "New York"),
    estimates=(0.4, 0.55, 0.62),
    n=(10, 25, 18),
) -> None:
    pd.DataFrame(
        {
            "state_fips": list(fips),
            "state_name": list(names),
            "estimate": list(estimates),
            "n_respondents": list(n),
        }
    ).to_csv(path, index=False)


# --- Discovery ------------------------------------------------------------

KNOWN_MODELS = {"baseline", "glm_mrp", "glmer_mrp", "stan_mrp", "srp", "mrdeep"}


def test_parse_filename_simple_outcome():
    parsed = _parse_filename(
        "baseline_happening_bin_state.csv", STATE_SUFFIX, KNOWN_MODELS
    )
    assert parsed == ("baseline", "happening_bin")


def test_parse_filename_multitoken_model_and_outcome():
    """`glm_mrp` (model) + `policy_renewable_bin` (outcome) — the case that
    naive regex splitting can't disambiguate."""
    parsed = _parse_filename(
        "glm_mrp_policy_renewable_bin_state.csv", STATE_SUFFIX, KNOWN_MODELS
    )
    assert parsed == ("glm_mrp", "policy_renewable_bin")


def test_parse_filename_picks_longest_matching_model_prefix():
    """If both 'glm' and 'glm_mrp' were known models, the longer wins."""
    parsed = _parse_filename(
        "glm_mrp_happening_bin_state.csv",
        STATE_SUFFIX,
        {"glm", "glm_mrp"},
    )
    assert parsed == ("glm_mrp", "happening_bin")


def test_parse_filename_rejects_unknown_model():
    assert (
        _parse_filename(
            "unknownmodel_happening_bin_state.csv", STATE_SUFFIX, KNOWN_MODELS
        )
        is None
    )


def test_parse_filename_rejects_bad_names():
    assert _parse_filename("not-a-valid-name.csv", STATE_SUFFIX, KNOWN_MODELS) is None
    assert _parse_filename("baseline_state.csv", STATE_SUFFIX, KNOWN_MODELS) is None
    assert (
        _parse_filename("UPPERCASE_outcome_state.csv", STATE_SUFFIX, KNOWN_MODELS)
        is None
    )


def test_build_registry_groups_by_outcome_and_model(tmp_path: Path):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    _write_state_csv(state_dir / "baseline_happening_bin_state.csv")
    _write_state_csv(state_dir / "glm_mrp_happening_bin_state.csv")
    _write_state_csv(state_dir / "baseline_worried_bin_state.csv")

    reg = build_registry(
        state_dir=state_dir,
        county_dir=tmp_path / "county",
        known_models=KNOWN_MODELS,
    )

    assert set(reg.outcomes("state")) == {"happening_bin", "worried_bin"}
    assert reg.models_for("happening_bin", "state") == ["baseline", "glm_mrp"]
    assert reg.models_for("worried_bin", "state") == ["baseline"]


def test_build_registry_skips_nonmatching_files(tmp_path: Path, caplog):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    _write_state_csv(state_dir / "baseline_happening_bin_state.csv")
    (state_dir / "README.md").write_text("not a csv")
    (state_dir / "weird_name.csv").write_text("a,b\n1,2")

    reg = build_registry(
        state_dir=state_dir,
        county_dir=tmp_path / "county",
        known_models=KNOWN_MODELS,
    )
    assert reg.outcomes("state") == ["happening_bin"]


def test_build_registry_handles_missing_directory(tmp_path: Path):
    reg = build_registry(
        state_dir=tmp_path / "does_not_exist",
        county_dir=tmp_path / "also_missing",
        known_models=KNOWN_MODELS,
    )
    assert reg.outcomes("state") == []
    assert reg.outcomes("county") == []


def test_models_for_unknown_outcome_returns_empty():
    from data_loader import Registry

    reg = Registry(state={"happening_bin": {"baseline": Path("/x")}})
    assert reg.models_for("nonexistent_bin", "state") == []


# --- Schema validation ---------------------------------------------------


def test_validate_estimates_pads_integer_fips():
    df = pd.DataFrame(
        {
            "state_fips": [1, 6, 36],
            "state_name": ["Alabama", "California", "New York"],
            "estimate": [0.4, 0.55, 0.62],
            "n_respondents": [10, 25, 18],
        }
    )
    out = _validate_estimates(
        df, REQUIRED_STATE_COLUMNS, "state_fips", STATE_FIPS_WIDTH
    )
    assert out["state_fips"].tolist() == ["01", "06", "36"]


def test_validate_estimates_rejects_out_of_range():
    df = pd.DataFrame(
        {
            "state_fips": ["01"],
            "state_name": ["Alabama"],
            "estimate": [1.5],
            "n_respondents": [10],
        }
    )
    with pytest.raises(SchemaError, match=r"\[0, 1\]"):
        _validate_estimates(
            df, REQUIRED_STATE_COLUMNS, "state_fips", STATE_FIPS_WIDTH
        )


def test_validate_estimates_rejects_missing_columns():
    df = pd.DataFrame({"state_fips": ["01"], "estimate": [0.5]})
    with pytest.raises(SchemaError, match="Missing required columns"):
        _validate_estimates(
            df, REQUIRED_STATE_COLUMNS, "state_fips", STATE_FIPS_WIDTH
        )


def test_validate_estimates_handles_columns_in_any_order():
    """The real glm_mrp CSV has columns in a different order than baseline."""
    df = pd.DataFrame(
        {
            "estimate": [0.5],
            "state_fips": [1],
            "n_respondents": [10],
            "state_name": ["Alabama"],
        }
    )
    out = _validate_estimates(
        df, REQUIRED_STATE_COLUMNS, "state_fips", STATE_FIPS_WIDTH
    )
    assert list(out.columns) == list(REQUIRED_STATE_COLUMNS)


# --- Loading + caching ----------------------------------------------------


def test_load_estimates_returns_validated_dataframe(tmp_path: Path):
    clear_caches()
    path = tmp_path / "baseline_happening_bin_state.csv"
    _write_state_csv(path, fips=(1, 6, 36))
    df = load_estimates(path, level="state")
    assert df["state_fips"].tolist() == ["01", "06", "36"]
    assert df["estimate"].between(0, 1).all()


def test_load_estimates_unknown_level_raises(tmp_path: Path):
    with pytest.raises(ValueError, match="Unknown level"):
        load_estimates(tmp_path / "x.csv", level="planet")


def test_load_estimates_missing_file_raises(tmp_path: Path):
    clear_caches()
    with pytest.raises(FileNotFoundError):
        load_estimates(tmp_path / "does_not_exist.csv", level="state")


# --- Real data sanity check ----------------------------------------------


def test_real_state_data_loads():
    """The three shipped CSVs in data/state/ all load and validate."""
    from constants import STATE_DIR

    reg = build_registry(state_dir=STATE_DIR)
    assert "happening_bin" in reg.outcomes("state")
    models = reg.models_for("happening_bin", "state")
    assert {"baseline", "glm_mrp", "glmer_mrp"}.issubset(models)
    for model in models:
        df = load_estimates(reg.state["happening_bin"][model], level="state")
        assert len(df) == 51
        assert df["state_fips"].str.len().eq(2).all()
