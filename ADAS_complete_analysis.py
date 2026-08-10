#!/usr/bin/env python3
"""
Complete ADAS / SOFI final-project analysis
===========================================

This script reruns the complete empirical pipeline after the analytical design
changes agreed for the final project.

PRIMARY COHORT ANALYSIS
-----------------------
- Vehicle age: 1-20 years
- Production cohorts: 2000-2004 through 2020-2024
- Reference cohort: 2000-2004
- Main model: Poisson count regression with log(total_km) offset
- Main controls: vehicle-age bin and fuel type
- Safety-feature indicators are NOT included in the main cohort model. They are
  introduced only in the dedicated feature and attenuation analyses.

FEATURE ANALYSIS
----------------
- Vehicle age: 1-20 years
- Production cohorts: 2010-2014 through 2020-2024
- Reference cohort: 2010-2014
- Individual-feature models, feature-count models, severity-specific feature
  models, and cohort attenuation analysis are rerun.

ROBUSTNESS CHECKS
-----------------
- Quasi-Poisson-style inference
- True NB2 maximum-likelihood model (with convergence fallback)
- No-COVID sample (excludes 2020-2021 observation years)
- Low-exposure restriction (default total_km >= 10,000)
- Low-imputation restriction (default imputed_frac <= 0.50)
- Cohort + calendar-year fixed effects, Poisson and NB2
- Additional sample-definition checks:
  * ages 1-25, cohorts 2000-2020
  * ages 0-20, cohorts 2000-2020
  * ages 1-20, cohorts 1995-2020, still referenced to 2000

SEVERITY ANALYSIS
-----------------
- Descriptive light, serious, fatal, and severe rates and shares by cohort
- Severity-specific cohort Poisson/quasi-Poisson models
- NB2 attempts for severity outcomes, recorded transparently if unstable
- Feature-level severity models in the 2010+ sample

USAGE
-----
Recommended repository structure:

    ADAS-Road-Safety-Thesis/
    ├── analysis/
    │   └── ADAS_complete_analysis.py
    ├── data/
    │   └── sofi_stratum_year.csv
    └── outputs/

From the repository root, run:

    python analysis/ADAS_complete_analysis.py

The script will automatically look for the required CSV in the repository's
data/ directory and will save outputs under outputs/ADAS_complete_analysis_outputs.

You may also provide explicit repository-relative paths:

    python analysis/ADAS_complete_analysis.py \
        --data "data/sofi_stratum_year.csv" \
        --output "outputs/ADAS_complete_analysis_outputs"

The script saves all tables, model summaries, plot data, and figures, including
paper-ready reproductions of Figures 2-12.  A final sensitivity section adds
engine-volume category (NEFAH_KV) as a categorical control to the main cohort
and feature analyses so the published specifications can be compared directly
with an engine-size-adjusted version.

Required packages:
    numpy, pandas, scipy, matplotlib, statsmodels, patsy

Interpretation reminder
-----------------------
These are observational associations in aggregated stratum-year data. Cohort
coefficients are not clean causal estimates of technology effects.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
import traceback
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import norm
from statsmodels.discrete.discrete_model import NegativeBinomial

# Match the visual style used in the submitted paper figures.
plt.style.use("ggplot")


# =============================================================================
# 1. CONSTANTS AND ANALYTICAL DEFINITIONS
# =============================================================================

AGE_MIN = 1
AGE_MAX = 20
AGE_BIN_EDGES = [1, 3, 6, 11, 16, 21]
AGE_BIN_LABELS = ["1_2", "3_5", "6_10", "11_15", "16_20"]
AGE_BIN_DISPLAY = {
    "1_2": "1–2",
    "3_5": "3–5",
    "6_10": "6–10",
    "11_15": "11–15",
    "16_20": "16–20",
}
AGE_REFERENCE = "1_2"

MAIN_COHORT_MIN = 2000
MAIN_COHORT_MAX = 2020
MAIN_COHORT_REFERENCE = "2000"
MAIN_COHORTS = [2000, 2005, 2010, 2015, 2020]
COHORT_DISPLAY = {
    1995: "1995–1999",
    2000: "2000–2004",
    2005: "2005–2009",
    2010: "2010–2014",
    2015: "2015–2019",
    2020: "2020–2023",
}

FEATURE_COHORT_MIN = 2010
FEATURE_COHORT_MAX = 2020
FEATURE_COHORT_REFERENCE = "2010"
FEATURE_COHORTS = [2010, 2015, 2020]

ENGINE_VOLUME_REFERENCE = "2-1,000"
ENGINE_VOLUME_ORDER = [
    "2-1,000",
    "1,001-1,300",
    "1,301-1,500",
    "1,501-1,600",
    "1,601-1,800",
    "1,801-2,000",
    "2,001-3,000",
    "3,001-4,000",
    "4,000+",
    "electric",
    "unknown",
]

COVID_YEARS = [2020, 2021]
DEFAULT_LOW_EXPOSURE_KM = 10_000
DEFAULT_MAX_IMPUTED_FRAC = 0.50
RATE_SCALE = 100_000_000
RAW_ANNUAL_ROWS_DOCUMENTED = 50_483

FEATURE_LABELS = {
    "zihuy_matzav_hitkarvut": "Forward distance / collision warning",
    "matzlemat_reverse_ind": "Reverse camera",
    "shlita_beorot": "Automatic light control",
    "bakarat_shyut": "Cruise control",
    "hayshaney_hagorot": "Seat-belt sensors/reminders",
    "zihuy_beshetah_nistar": "Blind-spot detection",
    "zihuy_tamrurey_tnua": "Traffic-sign recognition",
}

ACTIVE_FEATURES = [
    "shlita_beorot",
    "bakarat_shyut",
]

PASSIVE_FEATURES = [
    "zihuy_matzav_hitkarvut",
    "matzlemat_reverse_ind",
    "hayshaney_hagorot",
    "zihuy_beshetah_nistar",
    "zihuy_tamrurey_tnua",
]

SEVERITY_OUTCOMES = [
    "light_events",
    "serious_events",
    "severe_events",
    "fatal_events",
]

REQUIRED_COLUMNS = [
    "age",
    "n_vehicles",
    "total_km",
    "any_events",
    "fatal_events",
    "serious_events",
    "light_events",
    "SHNAT_YITZUR",
    "teur_sug_delek_mchv",
    "NEFAH_KV",
]


# =============================================================================
# 2. COMMAND-LINE ARGUMENTS AND PATHS
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the revised unified SOFI analysis pipeline."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Path to sofi_stratum_year.csv. If omitted, the repository data/ directory is searched first.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory. Default: outputs/ADAS_complete_analysis_outputs in the repository root.",
    )
    parser.add_argument(
        "--low-exposure-km",
        type=float,
        default=DEFAULT_LOW_EXPOSURE_KM,
        help=f"Minimum total_km for the low-exposure check (default {DEFAULT_LOW_EXPOSURE_KM:,}).",
    )
    parser.add_argument(
        "--max-imputed-frac",
        type=float,
        default=DEFAULT_MAX_IMPUTED_FRAC,
        help=f"Maximum imputed_frac for the low-imputation check (default {DEFAULT_MAX_IMPUTED_FRAC:.2f}).",
    )
    parser.add_argument(
        "--skip-nb2",
        action="store_true",
        help="Skip all true NB2 fits. Useful only for fast debugging.",
    )
    parser.add_argument(
        "--skip-feature-severity",
        action="store_true",
        help="Skip the full feature-by-severity model grid.",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Display figures interactively in addition to saving them.",
    )
    return parser.parse_args()


def resolve_data_path(explicit: Optional[Path]) -> Path:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent if script_dir.name.lower() == "analysis" else script_dir

    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")
        return path

    candidates = [
        project_root / "data" / "sofi_stratum_year.csv",
        project_root / "data" / "sofi_stratum_year(1).csv",
        project_root / "data" / "sofi_stratum_year(2).csv",
        Path.cwd() / "data" / "sofi_stratum_year.csv",
        Path.cwd() / "data" / "sofi_stratum_year(1).csv",
        Path.cwd() / "data" / "sofi_stratum_year(2).csv",
        script_dir / "sofi_stratum_year.csv",
        script_dir / "sofi_stratum_year(1).csv",
        script_dir / "sofi_stratum_year(2).csv",
        Path.cwd() / "sofi_stratum_year.csv",
        Path.cwd() / "sofi_stratum_year(1).csv",
        Path.cwd() / "sofi_stratum_year(2).csv",
    ]
    for path in candidates:
        if path.exists():
            return path.resolve()

    raise FileNotFoundError(
        "Could not find the SOFI CSV. Place it in the repository's data/ "
        "directory or provide its location with --data."
    )


def make_output_dirs(base: Path) -> dict[str, Path]:
    dirs = {
        "root": base,
        "sample": base / "01_sample_and_descriptives",
        "cohort": base / "02_main_cohort_models",
        "robust": base / "03_robustness_checks",
        "severity": base / "04_cohort_severity",
        "features": base / "05_feature_analysis",
        "feature_single": base / "05_feature_analysis" / "single_features",
        "feature_counts": base / "05_feature_analysis" / "feature_counts",
        "feature_attenuation": base / "05_feature_analysis" / "attenuation",
        "feature_severity": base / "05_feature_analysis" / "feature_severity",
        "engine": base / "06_engine_volume_sensitivity",
        "engine_single": base / "06_engine_volume_sensitivity" / "single_features",
        "engine_counts": base / "06_engine_volume_sensitivity" / "feature_counts",
        "engine_attenuation": base / "06_engine_volume_sensitivity" / "attenuation",
        "paper": base / "07_paper_figures",
        "logs": base / "99_logs",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


# =============================================================================
# 3. GENERAL UTILITIES
# =============================================================================


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def save_figure(fig: plt.Figure, path: Path, show: bool = False) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def safe_float(value: Any) -> float:
    try:
        value = float(value)
        return value if np.isfinite(value) else np.nan
    except Exception:
        return np.nan


def format_difference(irr: float) -> str:
    if not np.isfinite(irr):
        return "not available"
    pct = abs(1.0 - irr) * 100.0
    if irr < 1:
        return f"{pct:.1f}% lower"
    if irr > 1:
        return f"{pct:.1f}% higher"
    return "no difference"


def cohort_term(reference: str) -> str:
    return f"C(cohort_5yr, Treatment(reference='{reference}'))"


def age_term(reference: str = AGE_REFERENCE) -> str:
    return f"C(age_bin, Treatment(reference='{reference}'))"


def engine_term(reference: str = ENGINE_VOLUME_REFERENCE) -> str:
    return f"C(engine_volume_cat, Treatment(reference='{reference}'))"


def model_term_for_cohort(index: Iterable[str], cohort: int | str) -> Optional[str]:
    cohort = str(cohort)
    for term in index:
        if "cohort_5yr" in str(term) and f"[T.{cohort}]" in str(term):
            return str(term)
    return None


def make_irr_table(result: Any, drop_alpha: bool = True) -> pd.DataFrame:
    params = result.params.copy()
    conf = result.conf_int().copy()

    if drop_alpha:
        params = params.drop(labels=["alpha"], errors="ignore")
        conf = conf.drop(index="alpha", errors="ignore")

    pvalues = getattr(result, "pvalues", pd.Series(index=params.index, dtype=float))
    pvalues = pvalues.reindex(params.index)

    return pd.DataFrame(
        {
            "term": params.index,
            "coef": params.to_numpy(dtype=float),
            "IRR": np.exp(params.to_numpy(dtype=float)),
            "CI_low": np.exp(conf.loc[params.index, 0].to_numpy(dtype=float)),
            "CI_high": np.exp(conf.loc[params.index, 1].to_numpy(dtype=float)),
            "p_value": pvalues.to_numpy(dtype=float),
        }
    )


def make_quasi_table(classical_poisson_result: Any) -> tuple[pd.DataFrame, float]:
    dispersion = float(
        classical_poisson_result.pearson_chi2 / classical_poisson_result.df_resid
    )
    params = classical_poisson_result.params.copy()
    scaled_se = classical_poisson_result.bse * math.sqrt(dispersion)
    z_values = params / scaled_se
    p_values = 2 * (1 - norm.cdf(np.abs(z_values)))
    ci_low = params - 1.96 * scaled_se
    ci_high = params + 1.96 * scaled_se

    table = pd.DataFrame(
        {
            "term": params.index,
            "coef": params.to_numpy(dtype=float),
            "IRR": np.exp(params.to_numpy(dtype=float)),
            "CI_low": np.exp(ci_low.to_numpy(dtype=float)),
            "CI_high": np.exp(ci_high.to_numpy(dtype=float)),
            "p_value": np.asarray(p_values, dtype=float),
            "se_quasi": scaled_se.to_numpy(dtype=float),
        }
    )
    return table, dispersion


def cohort_rows_from_table(
    table: pd.DataFrame,
    cohorts: list[int],
    reference: int,
    model_name: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cohort in cohorts:
        if cohort == reference:
            rows.append(
                {
                    "model": model_name,
                    "cohort": cohort,
                    "coef": 0.0,
                    "IRR": 1.0,
                    "CI_low": 1.0,
                    "CI_high": 1.0,
                    "p_value": np.nan,
                    "reference": reference,
                }
            )
            continue

        term = model_term_for_cohort(table["term"].astype(str), cohort)
        if term is None:
            continue
        row = table.loc[table["term"] == term].iloc[0]
        rows.append(
            {
                "model": model_name,
                "cohort": cohort,
                "coef": row["coef"],
                "IRR": row["IRR"],
                "CI_low": row["CI_low"],
                "CI_high": row["CI_high"],
                "p_value": row["p_value"],
                "reference": reference,
            }
        )
    return pd.DataFrame(rows)


def comparison_from_table(
    table: pd.DataFrame,
    high_cohort: int,
    reference: int,
) -> dict[str, float]:
    if high_cohort == reference:
        return {
            "IRR": 1.0,
            "CI_low": 1.0,
            "CI_high": 1.0,
            "p_value": np.nan,
            "coef": 0.0,
        }

    term = model_term_for_cohort(table["term"].astype(str), high_cohort)
    if term is None:
        return {
            "IRR": np.nan,
            "CI_low": np.nan,
            "CI_high": np.nan,
            "p_value": np.nan,
            "coef": np.nan,
        }
    row = table.loc[table["term"] == term].iloc[0]
    return {
        "IRR": safe_float(row["IRR"]),
        "CI_low": safe_float(row["CI_low"]),
        "CI_high": safe_float(row["CI_high"]),
        "p_value": safe_float(row["p_value"]),
        "coef": safe_float(row["coef"]),
    }


def save_model_summary(result: Any, path: Path) -> None:
    try:
        text = result.summary().as_text()
    except Exception:
        text = repr(result)
    write_text(path, text)


@dataclass
class FittedModel:
    name: str
    formula: str
    data_rows: int
    table: pd.DataFrame
    result: Any
    model_family: str
    status: str = "ok"
    notes: str = ""
    dispersion: float = np.nan
    alpha: float = np.nan


# =============================================================================
# 4. DATA PREPARATION
# =============================================================================


def validate_and_numericize(df: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    numeric_columns = [
        "age",
        "n_vehicles",
        "total_km",
        "any_events",
        "fatal_events",
        "serious_events",
        "light_events",
        "SHNAT_YITZUR",
        "calendar_year",
        "imputed_frac",
        "n_imputed",
    ] + list(FEATURE_LABELS)

    out = df.copy()
    for col in numeric_columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def add_common_variables(
    df: pd.DataFrame,
    age_min: int,
    age_max: int,
    age_edges: list[int],
    age_labels: list[str],
) -> pd.DataFrame:
    out = df.copy()
    out = out[
        out["age"].between(age_min, age_max)
        & out["n_vehicles"].notna()
        & out["n_vehicles"].gt(0)
        & out["total_km"].notna()
        & out["total_km"].gt(0)
        & out["SHNAT_YITZUR"].notna()
        & out["teur_sug_delek_mchv"].notna()
        & out["any_events"].notna()
    ].copy()

    out["age"] = out["age"].astype(int)
    out["SHNAT_YITZUR"] = out["SHNAT_YITZUR"].astype(int)
    out["cohort_5yr_int"] = ((out["SHNAT_YITZUR"] // 5) * 5).astype(int)
    out["cohort_5yr"] = out["cohort_5yr_int"].astype(str)
    out["age_bin"] = pd.cut(
        out["age"],
        bins=age_edges,
        labels=age_labels,
        right=False,
        include_lowest=True,
    )
    out = out[out["age_bin"].notna()].copy()

    out["fuel"] = (
        out["teur_sug_delek_mchv"].astype(str).str.lower().str.strip()
    )
    out.loc[out["fuel"].isin(["nan", "none", ""]), "fuel"] = "unknown"
    out["engine_volume_cat"] = out["NEFAH_KV"].astype(str).str.strip()
    out.loc[
        out["engine_volume_cat"].isin(["nan", "none", ""]), "engine_volume_cat"
    ] = "unknown"
    unexpected_engine_categories = sorted(
        set(out["engine_volume_cat"].unique()) - set(ENGINE_VOLUME_ORDER)
    )
    if unexpected_engine_categories:
        warnings.warn(
            "Unexpected engine-volume categories found: "
            + ", ".join(unexpected_engine_categories)
        )
    out["severe_events"] = out["serious_events"] + out["fatal_events"]

    out["age_bin"] = pd.Categorical(out["age_bin"], categories=age_labels, ordered=True)
    return out


def restrict_cohorts(
    df: pd.DataFrame,
    cohort_min: int,
    cohort_max: int,
    cohort_order: list[int],
) -> pd.DataFrame:
    out = df[df["cohort_5yr_int"].between(cohort_min, cohort_max)].copy()
    categories = [str(c) for c in cohort_order if c in set(out["cohort_5yr_int"])]
    out["cohort_5yr"] = pd.Categorical(
        out["cohort_5yr"], categories=categories, ordered=True
    )
    out["fuel"] = pd.Categorical(out["fuel"]).remove_unused_categories()
    observed_engine_categories = [
        c for c in ENGINE_VOLUME_ORDER if c in set(out["engine_volume_cat"])
    ]
    observed_engine_categories += [
        c
        for c in sorted(set(out["engine_volume_cat"]))
        if c not in observed_engine_categories
    ]
    out["engine_volume_cat"] = pd.Categorical(
        out["engine_volume_cat"], categories=observed_engine_categories, ordered=False
    ).remove_unused_categories()
    out["age_bin"] = pd.Categorical(
        out["age_bin"], categories=list(out["age_bin"].cat.categories), ordered=True
    ).remove_unused_categories()
    return out


def prepare_feature_sample(df_main_age: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    feature_df = restrict_cohorts(
        df_main_age,
        FEATURE_COHORT_MIN,
        FEATURE_COHORT_MAX,
        FEATURE_COHORTS,
    )

    active = [col for col in ACTIVE_FEATURES if col in feature_df.columns]
    passive = [col for col in PASSIVE_FEATURES if col in feature_df.columns]
    all_features = active + passive

    invalid_rows: list[dict[str, Any]] = []
    for col in all_features:
        feature_df[col] = pd.to_numeric(feature_df[col], errors="coerce")
        invalid = feature_df[col].notna() & ~feature_df[col].isin([0, 1])
        invalid_rows.append(
            {
                "feature": col,
                "invalid_nonbinary_rows": int(invalid.sum()),
                "missing_rows": int(feature_df[col].isna().sum()),
            }
        )
        feature_df.loc[invalid, col] = np.nan

    feature_df["active_count"] = (
        feature_df[active].fillna(0).sum(axis=1) if active else 0
    )
    feature_df["passive_count"] = (
        feature_df[passive].fillna(0).sum(axis=1) if passive else 0
    )
    feature_df["total_feature_count"] = (
        feature_df["active_count"] + feature_df["passive_count"]
    )
    feature_df.attrs["feature_cleaning"] = pd.DataFrame(invalid_rows)
    return feature_df, active, passive


# =============================================================================
# 5. SAMPLE SUMMARIES AND DESCRIPTIVE FIGURES
# =============================================================================


def sample_metrics(df: pd.DataFrame) -> dict[str, float]:
    return {
        "rows": int(len(df)),
        "represented_vehicles": float(df["n_vehicles"].sum()),
        "total_km": float(df["total_km"].sum()),
        "accident_events": float(df["any_events"].sum()),
        "light_events": float(df["light_events"].sum()),
        "serious_events": float(df["serious_events"].sum()),
        "fatal_events": float(df["fatal_events"].sum()),
    }


def save_sample_tables(
    df_all: pd.DataFrame,
    age_df: pd.DataFrame,
    main_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    outdir: Path,
) -> None:
    original = sample_metrics(df_all)
    age = sample_metrics(age_df)
    main = sample_metrics(main_df)
    feature = sample_metrics(feature_df)

    rows = []
    for name, restriction, metrics in [
        ("Uploaded aggregated dataset", "No analytical filters", original),
        ("Age-valid analytical sample", "Age 1-20 and valid exposure/key fields", age),
        (
            "Primary cohort regression sample",
            "Age 1-20; production cohorts 2000-2020",
            main,
        ),
        (
            "Feature-analysis sample",
            "Age 1-20; production cohorts 2010-2020",
            feature,
        ),
    ]:
        row = {"sample": name, "restriction": restriction, **metrics}
        row["rows_pct_of_aggregated"] = metrics["rows"] / len(df_all) * 100
        row["vehicles_pct_of_aggregated"] = (
            metrics["represented_vehicles"] / original["represented_vehicles"] * 100
            if original["represented_vehicles"] > 0
            else np.nan
        )
        row["km_pct_of_aggregated"] = (
            metrics["total_km"] / original["total_km"] * 100
            if original["total_km"] > 0
            else np.nan
        )
        row["events_pct_of_aggregated"] = (
            metrics["accident_events"] / original["accident_events"] * 100
            if original["accident_events"] > 0
            else np.nan
        )
        rows.append(row)

    nested = pd.DataFrame(rows)
    nested.to_csv(outdir / "table_nested_analysis_samples.csv", index=False)

    main_avg_km = main["total_km"] / main["represented_vehicles"]
    final_table = pd.DataFrame(
        [
            {
                "measure": "Original raw annual SOFI rows",
                "value": RAW_ANNUAL_ROWS_DOCUMENTED,
                "interpretation_comment": (
                    "Previously documented number of annual source-file rows before aggregation; "
                    "not independently reconstructable from the uploaded aggregated CSV."
                ),
            },
            {
                "measure": "Aggregated stratum-year rows",
                "value": len(df_all),
                "interpretation_comment": "Rows in the uploaded aggregated SOFI CSV.",
            },
            {
                "measure": "Rows after age-valid filtering",
                "value": age["rows"],
                "interpretation_comment": (
                    f"{age['rows'] / len(df_all) * 100:.1f}% of aggregated rows and "
                    f"{age['rows'] / RAW_ANNUAL_ROWS_DOCUMENTED * 100:.1f}% of the documented raw annual rows."
                ),
            },
            {
                "measure": "Rows in primary cohort model",
                "value": main["rows"],
                "interpretation_comment": "Age 1-20 and production cohorts 2000-2020.",
            },
            {
                "measure": "Total represented vehicles",
                "value": main["represented_vehicles"],
                "interpretation_comment": (
                    "Summed represented-vehicle count across retained stratum-year rows; "
                    "not unique physical vehicles."
                ),
            },
            {
                "measure": "Total kilometers represented",
                "value": main["total_km"],
                "interpretation_comment": (
                    f"Mileage exposure in the main cohort sample; approximately {main_avg_km:,.0f} "
                    "km per represented vehicle."
                ),
            },
            {
                "measure": "Total accident events",
                "value": main["accident_events"],
                "interpretation_comment": "Light, serious, and fatal events in the main cohort sample.",
            },
            {
                "measure": "Light accidents",
                "value": main["light_events"],
                "interpretation_comment": (
                    f"{main['light_events'] / main['accident_events'] * 100:.1f}% of main-sample accident events."
                ),
            },
            {
                "measure": "Serious accidents",
                "value": main["serious_events"],
                "interpretation_comment": (
                    f"{main['serious_events'] / main['accident_events'] * 100:.1f}% of main-sample accident events."
                ),
            },
            {
                "measure": "Fatal accidents",
                "value": main["fatal_events"],
                "interpretation_comment": (
                    f"{main['fatal_events'] / main['accident_events'] * 100:.1f}% of main-sample accident events."
                ),
            },
        ]
    )
    final_table.to_csv(outdir / "table_main_analysis_sample.csv", index=False)


def descriptive_age_tables_and_figures(
    df_all_numeric: pd.DataFrame,
    age_df: pd.DataFrame,
    outdir: Path,
    show: bool,
) -> None:
    # Full observed age range diagnostic.
    full = df_all_numeric[
        df_all_numeric["age"].notna()
        & df_all_numeric["n_vehicles"].gt(0)
        & df_all_numeric["total_km"].gt(0)
        & df_all_numeric["any_events"].notna()
    ].copy()
    full["age"] = full["age"].astype(int)
    full_age = (
        full.groupby("age", as_index=False)
        .agg(
            represented_vehicles=("n_vehicles", "sum"),
            total_km=("total_km", "sum"),
            accident_events=("any_events", "sum"),
        )
        .sort_values("age")
    )
    full_age["accidents_per_100m_km"] = (
        full_age["accident_events"] / full_age["total_km"] * RATE_SCALE
    )
    full_age.to_csv(outdir / "table_full_age_rate_and_exposure.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(full_age["age"], full_age["accidents_per_100m_km"], marker="o", markersize=3)
    ax.set_xlabel("Vehicle age (years)")
    ax.set_ylabel("Accident events per 100 million km")
    ax.set_title("Descriptive accident rate across the full observed vehicle-age range")
    ax.grid(axis="y", alpha=0.3)
    save_figure(fig, outdir / "fig_full_age_accident_rate.png", show)

    # Main age 1-20 exact-age exposure table.
    exact = (
        age_df.groupby("age", as_index=False)
        .agg(
            represented_vehicles=("n_vehicles", "sum"),
            total_km=("total_km", "sum"),
            accident_events=("any_events", "sum"),
        )
        .sort_values("age")
    )
    exact["accidents_per_100m_km"] = (
        exact["accident_events"] / exact["total_km"] * RATE_SCALE
    )
    exact["represented_vehicles_millions"] = exact["represented_vehicles"] / 1e6
    exact["total_km_billions"] = exact["total_km"] / 1e9
    exact.to_csv(outdir / "table_age1_20_exact_rate_and_exposure.csv", index=False)

    # Binned rate and average annual km per represented vehicle.
    binned = (
        age_df.groupby("age_bin", observed=False)
        .agg(
            represented_vehicles=("n_vehicles", "sum"),
            total_km=("total_km", "sum"),
            accident_events=("any_events", "sum"),
        )
        .reset_index()
    )
    binned["accidents_per_100m_km"] = (
        binned["accident_events"] / binned["total_km"] * RATE_SCALE
    )
    binned["avg_annual_km_per_represented_vehicle"] = (
        binned["total_km"] / binned["represented_vehicles"]
    )
    binned["age_bin_display"] = binned["age_bin"].astype(str).map(AGE_BIN_DISPLAY)
    binned.to_csv(outdir / "table_age1_20_binned_rate_and_mileage.csv", index=False)

    fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    axes[0].plot(
        binned["age_bin_display"],
        binned["accidents_per_100m_km"],
        marker="o",
    )
    axes[0].set_ylabel("Accident events per 100 million km")
    axes[0].set_title("Panel A. Exposure-adjusted accident rate")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].plot(
        binned["age_bin_display"],
        binned["avg_annual_km_per_represented_vehicle"],
        marker="o",
    )
    axes[1].set_xlabel("Vehicle age bin (years)")
    axes[1].set_ylabel("Average annual km per represented vehicle")
    axes[1].set_title("Panel B. Average annual mileage")
    axes[1].grid(axis="y", alpha=0.3)
    fig.suptitle("Accident rate and annual mileage by vehicle age, ages 1–20", y=1.01)
    save_figure(fig, outdir / "fig_age1_20_rate_and_mileage_two_panel.png", show)

    # Exact-age represented vehicles and total km.
    fig, ax_left = plt.subplots(figsize=(10, 5.5))
    ax_right = ax_left.twinx()
    line_vehicles = ax_left.plot(
        exact["age"],
        exact["represented_vehicles_millions"],
        marker="o",
        label="Represented vehicles",
    )[0]
    line_km = ax_right.plot(
        exact["age"],
        exact["total_km_billions"],
        marker="s",
        linestyle="--",
        label="Total kilometers driven",
    )[0]
    ax_left.set_xlabel("Vehicle age (years)")
    ax_left.set_ylabel("Represented vehicles (millions)")
    ax_right.set_ylabel("Total kilometers driven (billions)")
    ax_left.set_title("Represented vehicles and mileage exposure by vehicle age, ages 1–20")
    ax_left.grid(axis="y", alpha=0.3)
    ax_left.legend([line_vehicles, line_km], [line_vehicles.get_label(), line_km.get_label()])
    save_figure(fig, outdir / "fig_age1_20_vehicles_and_km.png", show)


def paper_age_figures(
    df_all_numeric: pd.DataFrame,
    outdir: Path,
    show: bool,
) -> None:
    """Reproduce paper Figures 2 and 3 from the full descriptive age data."""
    full = df_all_numeric[
        df_all_numeric["age"].notna()
        & df_all_numeric["n_vehicles"].gt(0)
        & df_all_numeric["total_km"].gt(0)
        & df_all_numeric["any_events"].notna()
    ].copy()
    full["age"] = full["age"].astype(int)
    by_age = (
        full.groupby("age", as_index=False)
        .agg(
            represented_vehicles=("n_vehicles", "sum"),
            total_km=("total_km", "sum"),
            accident_events=("any_events", "sum"),
        )
        .sort_values("age")
    )
    by_age["rate_per_100m_km"] = (
        by_age["accident_events"] / by_age["total_km"] * RATE_SCALE
    )
    by_age["represented_vehicles_millions"] = by_age["represented_vehicles"] / 1e6
    by_age["total_km_billions"] = by_age["total_km"] / 1e9
    by_age.to_csv(outdir / "figure_02_data.csv", index=False)

    fig, axes = plt.subplots(2, 1, figsize=(11, 10), sharex=True)
    axes[0].plot(
        by_age["age"], by_age["rate_per_100m_km"], marker="o", markersize=3
    )
    axes[0].set_ylabel("Accident events per 100 million km")
    axes[0].text(-0.06, 1.03, "A", transform=axes[0].transAxes, fontweight="bold", fontsize=16)

    ax2 = axes[1].twinx()
    l1 = axes[1].plot(
        by_age["age"],
        by_age["represented_vehicles_millions"],
        marker="o",
        markersize=3,
        label="Represented vehicles",
    )[0]
    l2 = ax2.plot(
        by_age["age"],
        by_age["total_km_billions"],
        marker="s",
        markersize=3,
        linestyle="--",
        label="Total kilometers driven",
    )[0]
    axes[1].set_xlabel("Vehicle age (years)")
    axes[1].set_ylabel("Represented vehicles (millions)")
    ax2.set_ylabel("Total kilometers driven (billions)")
    axes[1].legend([l1, l2], [l1.get_label(), l2.get_label()], loc="upper right")
    axes[1].text(-0.06, 1.03, "B", transform=axes[1].transAxes, fontweight="bold", fontsize=16)
    fig.tight_layout()
    save_figure(fig, outdir / "figure_02_raw_accident_rate_and_exposure_by_vehicle_age.png", show)

    # Figure 3 deliberately retains ages 21-25 because the observed reversal in
    # that last bin is the descriptive justification for the final age<=20 model.
    age25 = full[full["age"].between(1, 25)].copy()
    labels25 = ["1–2", "3–5", "6–10", "11–15", "16–20", "21–25"]
    age25["age_bin_1_25"] = pd.cut(
        age25["age"],
        bins=[1, 3, 6, 11, 16, 21, 26],
        labels=labels25,
        right=False,
        include_lowest=True,
    )
    binned = (
        age25.groupby("age_bin_1_25", observed=False)
        .agg(
            represented_vehicles=("n_vehicles", "sum"),
            total_km=("total_km", "sum"),
            accident_events=("any_events", "sum"),
        )
        .reset_index()
    )
    binned["rate_per_100m_km"] = (
        binned["accident_events"] / binned["total_km"] * RATE_SCALE
    )
    binned["avg_annual_km_per_represented_vehicle"] = (
        binned["total_km"] / binned["represented_vehicles"]
    )
    binned.to_csv(outdir / "figure_03_data.csv", index=False)

    fig, axes = plt.subplots(2, 1, figsize=(11, 9.6), sharex=True)
    axes[0].plot(binned["age_bin_1_25"].astype(str), binned["rate_per_100m_km"], marker="o")
    axes[0].set_ylabel("Accident events per 100 million km")
    axes[0].text(-0.06, 1.03, "A", transform=axes[0].transAxes, fontweight="bold", fontsize=16)
    axes[1].plot(
        binned["age_bin_1_25"].astype(str),
        binned["avg_annual_km_per_represented_vehicle"],
        marker="o",
    )
    axes[1].set_xlabel("Vehicle age bin (years)")
    axes[1].set_ylabel("Average annual km per represented vehicle")
    axes[1].text(-0.06, 1.03, "B", transform=axes[1].transAxes, fontweight="bold", fontsize=16)
    fig.tight_layout()
    save_figure(fig, outdir / "figure_03_accident_rate_and_average_annual_mileage_by_age_bin.png", show)


def cohort_age_heatmap(
    data: pd.DataFrame,
    cohorts: list[int],
    outdir: Path,
    filename_stem: str,
    title: str,
    show: bool,
) -> None:
    heat = (
        data[data["cohort_5yr_int"].isin(cohorts)]
        .groupby(["cohort_5yr_int", "age_bin"], observed=False)
        .agg(
            accident_events=("any_events", "sum"),
            total_km=("total_km", "sum"),
            represented_vehicles=("n_vehicles", "sum"),
            rows=("any_events", "size"),
        )
        .reset_index()
    )
    heat["accidents_per_100m_km"] = np.where(
        heat["total_km"] > 0,
        heat["accident_events"] / heat["total_km"] * RATE_SCALE,
        np.nan,
    )
    heat.to_csv(outdir / f"{filename_stem}_data.csv", index=False)

    matrix = (
        heat.pivot(
            index="cohort_5yr_int",
            columns="age_bin",
            values="accidents_per_100m_km",
        )
        .reindex(index=cohorts)
        .reindex(columns=AGE_BIN_LABELS)
    )
    values = matrix.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(values)

    fig, ax = plt.subplots(figsize=(10, 6))
    image = ax.imshow(masked, aspect="auto")
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_xticklabels([AGE_BIN_DISPLAY[str(col)] for col in matrix.columns])
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels(
        [COHORT_DISPLAY.get(int(value), str(value)) for value in matrix.index]
    )
    ax.set_xlabel("Vehicle age bin (years)")
    ax.set_ylabel("Production cohort")

    finite_values = values[np.isfinite(values)]
    midpoint = np.nanmedian(finite_values) if finite_values.size else 0
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            if np.isfinite(value):
                text_color = "white" if value < midpoint else "black"
                ax.text(j, i, f"{value:.1f}", ha="center", va="center", color=text_color)

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Accident events per 100 million km")
    save_figure(fig, outdir / f"{filename_stem}.png", show)


# =============================================================================
# 6. MODEL FITTING HELPERS
# =============================================================================


def fit_poisson_models(
    name: str,
    formula: str,
    data: pd.DataFrame,
    outdir: Path,
    *,
    fit_robust: bool = True,
    save_summary: bool = False,
) -> tuple[FittedModel, FittedModel]:
    print(f"[Poisson] starting {name}: rows={len(data):,}", flush=True)
    model = smf.glm(
        formula=formula,
        data=data,
        family=sm.families.Poisson(),
        offset=np.log(data["total_km"]),
    )
    classical = model.fit()
    robust = model.fit(cov_type="HC1") if fit_robust else classical

    robust_table = make_irr_table(robust)
    quasi_table, dispersion = make_quasi_table(classical)

    poisson_suffix = "poisson_hc1" if fit_robust else "poisson_classical"
    robust_table.to_csv(outdir / f"{name}_{poisson_suffix}_irr_table.csv", index=False)
    quasi_table.to_csv(outdir / f"{name}_quasi_poisson_irr_table.csv", index=False)
    if save_summary:
        save_model_summary(robust, outdir / f"{name}_{poisson_suffix}_summary.txt")
        save_model_summary(classical, outdir / f"{name}_poisson_classical_summary.txt")

    robust_fit = FittedModel(
        name=f"{name}_{poisson_suffix}",
        formula=formula,
        data_rows=len(data),
        table=robust_table,
        result=robust,
        model_family="Poisson HC1" if fit_robust else "Poisson classical",
        dispersion=dispersion,
    )
    quasi_fit = FittedModel(
        name=f"{name}_quasi",
        formula=formula,
        data_rows=len(data),
        table=quasi_table,
        result=classical,
        model_family="Quasi-Poisson-style inference",
        dispersion=dispersion,
        notes="Poisson point estimates with Pearson-dispersion-scaled standard errors.",
    )
    print(f"[Poisson] finished {name}", flush=True)
    return robust_fit, quasi_fit


def estimate_nb2_alpha_moment(poisson_result: Any) -> float:
    mu = np.asarray(poisson_result.mu, dtype=float)
    y = np.asarray(poisson_result.model.endog, dtype=float)
    numerator = np.sum((y - mu) ** 2 - y)
    denominator = np.sum(mu**2)
    if denominator <= 0:
        return np.nan
    alpha = numerator / denominator
    if not np.isfinite(alpha):
        return np.nan
    return max(float(alpha), 1e-10)


def fit_true_nb2(
    name: str,
    formula: str,
    data: pd.DataFrame,
    outdir: Path,
    *,
    save_summary: bool = False,
) -> FittedModel:
    # Poisson starting values substantially improve NB2 convergence.
    poisson_start = smf.glm(
        formula=formula,
        data=data,
        family=sm.families.Poisson(),
        offset=np.log(data["total_km"]),
    ).fit()
    alpha_moment = estimate_nb2_alpha_moment(poisson_start)

    attempts: list[dict[str, Any]] = []
    alpha_starts = [alpha_moment, 0.001, 0.01, 0.1, 1.0]
    alpha_starts = [
        float(a) for a in dict.fromkeys(alpha_starts) if np.isfinite(a) and a > 0
    ]
    methods = ["bfgs", "lbfgs", "newton", "nm"]

    accepted_result = None
    accepted_method = None
    accepted_alpha_start = None

    for alpha_start in alpha_starts:
        for method in methods:
            try:
                model = NegativeBinomial.from_formula(
                    formula,
                    data=data,
                    exposure=data["total_km"],
                    loglike_method="nb2",
                )
                start_params = np.r_[
                    poisson_start.params.to_numpy(dtype=float), alpha_start
                ]
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    result = model.fit(
                        start_params=start_params,
                        method=method,
                        maxiter=500,
                        disp=False,
                    )

                converged = bool(
                    getattr(result, "mle_retvals", {}).get("converged", False)
                )
                finite_params = np.all(
                    np.isfinite(result.params.to_numpy(dtype=float))
                )
                try:
                    finite_conf = np.all(
                        np.isfinite(result.conf_int().to_numpy(dtype=float))
                    )
                except Exception:
                    finite_conf = False

                warning_text = "; ".join(str(w.message) for w in caught[:5])
                attempts.append(
                    {
                        "alpha_start": alpha_start,
                        "method": method,
                        "converged": converged,
                        "finite_params": finite_params,
                        "finite_confidence_intervals": finite_conf,
                        "aic": safe_float(getattr(result, "aic", np.nan)),
                        "warnings": warning_text,
                        "error": "",
                    }
                )
                if converged and finite_params and finite_conf:
                    accepted_result = result
                    accepted_method = method
                    accepted_alpha_start = alpha_start
                    break
            except Exception as exc:
                attempts.append(
                    {
                        "alpha_start": alpha_start,
                        "method": method,
                        "converged": False,
                        "finite_params": False,
                        "finite_confidence_intervals": False,
                        "aic": np.nan,
                        "warnings": "",
                        "error": str(exc)[:500],
                    }
                )
        if accepted_result is not None:
            break

    pd.DataFrame(attempts).to_csv(
        outdir / f"{name}_true_nb2_fit_attempts.csv", index=False
    )

    if accepted_result is not None:
        table = make_irr_table(accepted_result, drop_alpha=True)
        alpha = safe_float(accepted_result.params.get("alpha", np.nan))
        table.to_csv(outdir / f"{name}_true_nb2_irr_table.csv", index=False)
        if save_summary:
            save_model_summary(
                accepted_result, outdir / f"{name}_true_nb2_summary.txt"
            )
        return FittedModel(
            name=f"{name}_true_nb2",
            formula=formula,
            data_rows=len(data),
            table=table,
            result=accepted_result,
            model_family="True NB2 ML",
            alpha=alpha,
            notes=(
                f"Converged using {accepted_method} with alpha_start={accepted_alpha_start:.6g}."
            ),
        )

    # Transparent fallback: GLM NB2 variance with moment-estimated fixed alpha.
    fallback_model = smf.glm(
        formula=formula,
        data=data,
        family=sm.families.NegativeBinomial(alpha=alpha_moment),
        offset=np.log(data["total_km"]),
    )
    fallback_result = fallback_model.fit()
    fallback_table = make_irr_table(fallback_result)
    fallback_table.to_csv(
        outdir / f"{name}_glm_nb_fixed_alpha_fallback_irr_table.csv", index=False
    )
    if save_summary:
        save_model_summary(
            fallback_result, outdir / f"{name}_glm_nb_fixed_alpha_fallback_summary.txt"
        )
    return FittedModel(
        name=f"{name}_glm_nb_fixed_alpha_fallback",
        formula=formula,
        data_rows=len(data),
        table=fallback_table,
        result=fallback_result,
        model_family="GLM NB fixed alpha fallback",
        status="fallback",
        alpha=alpha_moment,
        notes=(
            "True NB2 did not converge with finite confidence intervals; used a GLM "
            "Negative Binomial model with moment-estimated fixed alpha."
        ),
    )


# =============================================================================
# 7. MAIN COHORT MODELS AND ROBUSTNESS CHECKS
# =============================================================================


def run_main_cohort_models(
    main_df: pd.DataFrame,
    dirs: dict[str, Path],
    low_exposure_km: float,
    max_imputed_frac: float,
    skip_nb2: bool,
    show: bool,
) -> tuple[dict[str, FittedModel], pd.DataFrame]:
    main_formula = (
        f"any_events ~ {age_term()} + {cohort_term(MAIN_COHORT_REFERENCE)} + C(fuel)"
    )
    write_text(dirs["cohort"] / "main_cohort_formula.txt", main_formula)

    models: dict[str, FittedModel] = {}

    main_poisson, main_quasi = fit_poisson_models(
        "main", main_formula, main_df, dirs["cohort"]
    )
    models["Main Poisson"] = main_poisson
    models["Quasi-Poisson-style inference"] = main_quasi

    if not skip_nb2:
        models["NB2 full main specification"] = fit_true_nb2(
            "main", main_formula, main_df, dirs["cohort"]
        )

    if "calendar_year" in main_df.columns:
        no_covid_df = main_df[~main_df["calendar_year"].isin(COVID_YEARS)].copy()
        no_covid_poisson, _ = fit_poisson_models(
            "no_covid", main_formula, no_covid_df, dirs["robust"]
        )
        models["No-COVID sample"] = no_covid_poisson

    low_exposure_df = main_df[main_df["total_km"] >= low_exposure_km].copy()
    low_exp_poisson, _ = fit_poisson_models(
        "low_exposure", main_formula, low_exposure_df, dirs["robust"]
    )
    models["Low-exposure restriction"] = low_exp_poisson

    if "imputed_frac" in main_df.columns:
        low_imp_df = main_df[
            main_df["imputed_frac"].notna()
            & main_df["imputed_frac"].le(max_imputed_frac)
        ].copy()
        low_imp_poisson, _ = fit_poisson_models(
            "low_imputation", main_formula, low_imp_df, dirs["robust"]
        )
        models["Low-imputation restriction"] = low_imp_poisson

    if "calendar_year" in main_df.columns:
        year_formula = (
            f"any_events ~ {cohort_term(MAIN_COHORT_REFERENCE)} + "
            "C(calendar_year) + C(fuel)"
        )
        year_poisson, _ = fit_poisson_models(
            "cohort_year_fe", year_formula, main_df, dirs["robust"]
        )
        models["Cohort + calendar-year FE, Poisson"] = year_poisson
        if not skip_nb2:
            models["Cohort + calendar-year FE, NB2"] = fit_true_nb2(
                "cohort_year_fe", year_formula, main_df, dirs["robust"]
            )

    comparison_rows: list[dict[str, Any]] = []
    cohort_frames: list[pd.DataFrame] = []
    for label, fitted in models.items():
        comparison = comparison_from_table(
            fitted.table, 2020, int(MAIN_COHORT_REFERENCE)
        )
        comparison_rows.append(
            {
                "model_check": label,
                "IRR_2020_vs_2000": comparison["IRR"],
                "CI_low": comparison["CI_low"],
                "CI_high": comparison["CI_high"],
                "p_value": comparison["p_value"],
                "approximate_difference": format_difference(comparison["IRR"]),
                "n_rows": fitted.data_rows,
                "model_family": fitted.model_family,
                "dispersion": fitted.dispersion,
                "alpha": fitted.alpha,
                "status": fitted.status,
                "notes": fitted.notes,
                "formula": fitted.formula,
            }
        )
        cohort_frames.append(
            cohort_rows_from_table(
                fitted.table,
                MAIN_COHORTS,
                int(MAIN_COHORT_REFERENCE),
                label,
            )
        )

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(
        dirs["cohort"] / "table_main_cohort_comparison_across_model_checks.csv",
        index=False,
    )

    all_cohort_irrs = pd.concat(cohort_frames, ignore_index=True)
    all_cohort_irrs.to_csv(
        dirs["cohort"] / "table_cohort_irrs_all_model_checks.csv", index=False
    )

    # Main robustness plot: keep year-FE checks separate because they answer a
    # stricter period-adjusted question and may compress the other lines.
    main_plot_labels = [
        "Main Poisson",
        "Quasi-Poisson-style inference",
        "NB2 full main specification",
        "No-COVID sample",
        "Low-exposure restriction",
        "Low-imputation restriction",
    ]
    main_plot = all_cohort_irrs[all_cohort_irrs["model"].isin(main_plot_labels)].copy()
    if not main_plot.empty:
        display_model = {
            "Main Poisson": "Primary Poisson",
            "Quasi-Poisson-style inference": "Quasi-Poisson-style",
            "NB2 full main specification": "NB2",
            "No-COVID sample": "No-COVID sample",
            "Low-exposure restriction": "Low-exposure restriction",
            "Low-imputation restriction": "Low-imputation restriction",
        }
        fig, ax = plt.subplots(figsize=(10, 6))
        for label, group in main_plot.groupby("model", sort=False):
            group = group.sort_values("cohort")
            ax.plot(
                group["cohort"],
                group["IRR"],
                marker="o",
                label=display_model.get(label, label),
            )
        ax.axhline(1, linestyle="--")
        ax.set_xticks(MAIN_COHORTS)
        ax.set_xticklabels([COHORT_DISPLAY[c] for c in MAIN_COHORTS])
        ax.set_xlabel("Production cohort")
        ax.set_ylabel("IRR relative to the 2000–2004 cohort")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        save_figure(fig, dirs["cohort"] / "fig_cohort_main_robustness_checks.png", show)

    year_plot_labels = [
        "Main Poisson",
        "Cohort + calendar-year FE, Poisson",
    ]
    year_plot = all_cohort_irrs[all_cohort_irrs["model"].isin(year_plot_labels)].copy()
    if not year_plot.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        for label, group in year_plot.groupby("model", sort=False):
            group = group.sort_values("cohort")
            display = (
                "Primary Poisson"
                if label == "Main Poisson"
                else "Cohort + calendar-year fixed effects"
            )
            ax.plot(group["cohort"], group["IRR"], marker="o", label=display)
        ax.axhline(1, linestyle="--")
        ax.set_xticks(MAIN_COHORTS)
        ax.set_xticklabels([COHORT_DISPLAY[c] for c in MAIN_COHORTS])
        ax.set_xlabel("Production cohort")
        ax.set_ylabel("IRR relative to the 2000–2004 cohort")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        save_figure(fig, dirs["cohort"] / "fig_cohort_year_fe_comparison.png", show)

    # Main Poisson cohort IRRs with confidence intervals.
    main_irrs = cohort_rows_from_table(
        main_poisson.table,
        MAIN_COHORTS,
        int(MAIN_COHORT_REFERENCE),
        "Main Poisson",
    )
    fig, ax = plt.subplots(figsize=(9, 5.5))
    yerr = np.vstack(
        [
            main_irrs["IRR"] - main_irrs["CI_low"],
            main_irrs["CI_high"] - main_irrs["IRR"],
        ]
    )
    # The reference cohort has an artificial zero-width CI by definition.
    ax.errorbar(
        main_irrs["cohort"],
        main_irrs["IRR"],
        yerr=yerr,
        marker="o",
        capsize=4,
    )
    ax.axhline(1, linestyle="--")
    ax.set_xticks(MAIN_COHORTS)
    ax.set_xlabel("Production cohort")
    ax.set_ylabel("Incidence rate ratio")
    ax.set_title("Main Poisson cohort estimates relative to the 2000 cohort")
    ax.grid(axis="y", alpha=0.3)
    save_figure(fig, dirs["cohort"] / "fig_main_poisson_cohort_irrs_ci.png", show)

    return models, comparison_df


def predicted_rates_by_age_and_cohort(
    main_model: FittedModel,
    main_df: pd.DataFrame,
    outdir: Path,
    show: bool,
) -> None:
    reference_fuel = str(main_df["fuel"].mode().iloc[0])
    rows = []
    for cohort in MAIN_COHORTS:
        for age_bin_value in AGE_BIN_LABELS:
            rows.append(
                {
                    "age_bin": age_bin_value,
                    "cohort_5yr": str(cohort),
                    "fuel": reference_fuel,
                    "total_km": RATE_SCALE,
                }
            )
    prediction_df = pd.DataFrame(rows)
    prediction_df["age_bin"] = pd.Categorical(
        prediction_df["age_bin"], categories=AGE_BIN_LABELS, ordered=True
    )
    prediction_df["cohort_5yr"] = pd.Categorical(
        prediction_df["cohort_5yr"],
        categories=[str(c) for c in MAIN_COHORTS],
        ordered=True,
    )
    prediction_df["fuel"] = pd.Categorical(
        prediction_df["fuel"], categories=main_df["fuel"].cat.categories
    )
    prediction_df["predicted_events_per_100m_km"] = main_model.result.predict(
        prediction_df, offset=np.log(prediction_df["total_km"])
    )
    prediction_df["age_bin_display"] = (
        prediction_df["age_bin"].astype(str).map(AGE_BIN_DISPLAY)
    )
    prediction_df.to_csv(outdir / "table_predicted_rates_by_age_and_cohort.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    for cohort in MAIN_COHORTS:
        group = prediction_df[
            prediction_df["cohort_5yr"].astype(str) == str(cohort)
        ]
        ax.plot(
            group["age_bin_display"],
            group["predicted_events_per_100m_km"],
            marker="o",
            label=f"Cohort {cohort}",
        )
    ax.set_xlabel("Vehicle age bin (years)")
    ax.set_ylabel("Predicted accident events per 100 million km")
    ax.set_title("Predicted accident rate by vehicle age and production cohort")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    save_figure(fig, outdir / "fig_predicted_rates_by_age_and_cohort.png", show)


# =============================================================================
# 8. ADDITIONAL SAMPLE-DEFINITION SENSITIVITY CHECKS
# =============================================================================


def fit_sample_variant(
    raw_numeric: pd.DataFrame,
    name: str,
    age_minimum: int,
    age_maximum: int,
    cohort_minimum: int,
    cohort_maximum: int,
    age_edges: list[int],
    age_labels: list[str],
    age_reference: str,
    outdir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    prepared = add_common_variables(
        raw_numeric,
        age_minimum,
        age_maximum,
        age_edges,
        age_labels,
    )
    cohort_order = list(range(cohort_minimum, cohort_maximum + 1, 5))
    prepared = restrict_cohorts(
        prepared,
        cohort_minimum,
        cohort_maximum,
        cohort_order,
    )
    if MAIN_COHORT_REFERENCE not in prepared["cohort_5yr"].cat.categories:
        raise ValueError(f"Variant {name} does not contain the 2000 reference cohort.")

    formula = (
        f"any_events ~ C(age_bin, Treatment(reference='{age_reference}')) + "
        f"{cohort_term(MAIN_COHORT_REFERENCE)} + C(fuel)"
    )
    robust, _ = fit_poisson_models(name, formula, prepared, outdir, fit_robust=False)
    comparison = comparison_from_table(robust.table, 2020, 2000)
    cohort_table = cohort_rows_from_table(
        robust.table,
        [c for c in cohort_order if c >= 2000],
        2000,
        name,
    )
    summary = {
        "sample_check": name,
        "restriction": (
            f"age {age_minimum}-{age_maximum}; cohorts {cohort_minimum}-{cohort_maximum}"
        ),
        "n_rows": len(prepared),
        "IRR_2020_vs_2000": comparison["IRR"],
        "CI_low": comparison["CI_low"],
        "CI_high": comparison["CI_high"],
        "p_value": comparison["p_value"],
        "approximate_difference": format_difference(comparison["IRR"]),
    }
    return cohort_table, summary


def run_sample_definition_checks(
    raw_numeric: pd.DataFrame,
    outdir: Path,
    show: bool,
) -> None:
    checks = [
        {
            "name": "primary_age1_20_cohort2000",
            "age_minimum": 1,
            "age_maximum": 20,
            "cohort_minimum": 2000,
            "cohort_maximum": 2020,
            "age_edges": [1, 3, 6, 11, 16, 21],
            "age_labels": ["1_2", "3_5", "6_10", "11_15", "16_20"],
            "age_reference": "1_2",
        },
        {
            "name": "age1_25_cohort2000",
            "age_minimum": 1,
            "age_maximum": 25,
            "cohort_minimum": 2000,
            "cohort_maximum": 2020,
            "age_edges": [1, 3, 6, 11, 16, 21, 26],
            "age_labels": ["1_2", "3_5", "6_10", "11_15", "16_20", "21_25"],
            "age_reference": "1_2",
        },
        {
            "name": "age0_20_cohort2000",
            "age_minimum": 0,
            "age_maximum": 20,
            "cohort_minimum": 2000,
            "cohort_maximum": 2020,
            "age_edges": [0, 3, 6, 11, 16, 21],
            "age_labels": ["0_2", "3_5", "6_10", "11_15", "16_20"],
            "age_reference": "0_2",
        },
        {
            "name": "age1_20_include1995",
            "age_minimum": 1,
            "age_maximum": 20,
            "cohort_minimum": 1995,
            "cohort_maximum": 2020,
            "age_edges": [1, 3, 6, 11, 16, 21],
            "age_labels": ["1_2", "3_5", "6_10", "11_15", "16_20"],
            "age_reference": "1_2",
        },
    ]

    frames = []
    summaries = []
    for spec in checks:
        frame, summary = fit_sample_variant(
            raw_numeric=raw_numeric,
            outdir=outdir,
            **spec,
        )
        frames.append(frame)
        summaries.append(summary)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(outdir / "table_sample_definition_sensitivity.csv", index=False)
    plot_df = pd.concat(frames, ignore_index=True)
    plot_df.to_csv(outdir / "table_sample_definition_cohort_irrs.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    for label, group in plot_df.groupby("model", sort=False):
        group = group.sort_values("cohort")
        ax.plot(group["cohort"], group["IRR"], marker="o", label=label)
    ax.axhline(1, linestyle="--")
    ax.set_xticks(MAIN_COHORTS)
    ax.set_xlabel("Production cohort")
    ax.set_ylabel("IRR relative to the 2000 cohort")
    ax.set_title("Sensitivity to age-window and cohort-window definitions")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    save_figure(fig, outdir / "fig_sample_definition_sensitivity.png", show)


# =============================================================================
# 9. COHORT SEVERITY ANALYSIS
# =============================================================================


def run_cohort_severity_analysis(
    main_df: pd.DataFrame,
    outdir: Path,
    skip_nb2: bool,
    show: bool,
) -> None:
    descriptive = (
        main_df.groupby("cohort_5yr_int")
        .agg(
            fatal_events=("fatal_events", "sum"),
            serious_events=("serious_events", "sum"),
            light_events=("light_events", "sum"),
            severe_events=("severe_events", "sum"),
            any_events=("any_events", "sum"),
            total_km=("total_km", "sum"),
            represented_vehicles=("n_vehicles", "sum"),
        )
        .reindex(MAIN_COHORTS)
        .reset_index()
    )
    for outcome in ["fatal_events", "serious_events", "light_events", "severe_events", "any_events"]:
        descriptive[f"{outcome}_rate_per_100m_km"] = (
            descriptive[outcome] / descriptive["total_km"] * RATE_SCALE
        )
    descriptive["fatal_share"] = descriptive["fatal_events"] / descriptive["any_events"]
    descriptive["serious_share"] = descriptive["serious_events"] / descriptive["any_events"]
    descriptive["severe_share"] = descriptive["severe_events"] / descriptive["any_events"]
    descriptive.to_csv(outdir / "table_severity_rates_and_shares_by_cohort.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        descriptive["cohort_5yr_int"],
        descriptive["light_events_rate_per_100m_km"],
        marker="o",
        label="Light",
    )
    ax.plot(
        descriptive["cohort_5yr_int"],
        descriptive["serious_events_rate_per_100m_km"],
        marker="o",
        label="Serious",
    )
    ax.plot(
        descriptive["cohort_5yr_int"],
        descriptive["fatal_events_rate_per_100m_km"],
        marker="o",
        label="Fatal",
    )
    ax.set_xticks(MAIN_COHORTS)
    ax.set_xticklabels([COHORT_DISPLAY[c] for c in MAIN_COHORTS])
    ax.set_xlabel("Production cohort")
    ax.set_ylabel("Accident events per 100 million km")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    save_figure(fig, outdir / "fig_severity_rates_by_cohort.png", show)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(descriptive["cohort_5yr_int"], descriptive["fatal_share"], marker="o", label="Fatal share")
    ax.plot(descriptive["cohort_5yr_int"], descriptive["serious_share"], marker="o", label="Serious share")
    ax.plot(descriptive["cohort_5yr_int"], descriptive["severe_share"], marker="o", label="Fatal + serious share")
    ax.set_xticks(MAIN_COHORTS)
    ax.set_xlabel("Production cohort")
    ax.set_ylabel("Share of total accident events")
    ax.set_title("Share of severe accident events by production cohort")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    save_figure(fig, outdir / "fig_severity_shares_by_cohort.png", show)

    summary_rows = []
    cohort_frames = []
    for outcome in SEVERITY_OUTCOMES:
        formula = (
            f"{outcome} ~ {age_term()} + {cohort_term(MAIN_COHORT_REFERENCE)} + C(fuel)"
        )
        poisson, quasi = fit_poisson_models(
            f"cohort_severity_{outcome}", formula, main_df, outdir, fit_robust=False
        )
        for label, fitted in [
            (f"{outcome}: Poisson point estimate", poisson),
            (f"{outcome}: Quasi-Poisson-style", quasi),
        ]:
            comparison = comparison_from_table(fitted.table, 2020, 2000)
            summary_rows.append(
                {
                    "outcome": outcome,
                    "model": label,
                    "IRR_2020_vs_2000": comparison["IRR"],
                    "CI_low": comparison["CI_low"],
                    "CI_high": comparison["CI_high"],
                    "p_value": comparison["p_value"],
                    "approximate_difference": format_difference(comparison["IRR"]),
                    "n_rows": fitted.data_rows,
                    "dispersion": fitted.dispersion,
                    "status": fitted.status,
                    "notes": fitted.notes,
                }
            )
        cohort_frames.append(
            cohort_rows_from_table(poisson.table, MAIN_COHORTS, 2000, outcome)
        )

        if not skip_nb2:
            nb2 = fit_true_nb2(f"cohort_severity_{outcome}", formula, main_df, outdir)
            comparison = comparison_from_table(nb2.table, 2020, 2000)
            summary_rows.append(
                {
                    "outcome": outcome,
                    "model": f"{outcome}: {nb2.model_family}",
                    "IRR_2020_vs_2000": comparison["IRR"],
                    "CI_low": comparison["CI_low"],
                    "CI_high": comparison["CI_high"],
                    "p_value": comparison["p_value"],
                    "approximate_difference": format_difference(comparison["IRR"]),
                    "n_rows": nb2.data_rows,
                    "dispersion": nb2.dispersion,
                    "status": nb2.status,
                    "notes": nb2.notes,
                }
            )

    pd.DataFrame(summary_rows).to_csv(
        outdir / "table_cohort_severity_model_summary.csv", index=False
    )

    plot_df = pd.concat(cohort_frames, ignore_index=True)
    plot_df.to_csv(outdir / "table_cohort_severity_irrs.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    severity_display = {
        "light_events": "Light",
        "serious_events": "Serious",
        "severe_events": "Serious + fatal",
        "fatal_events": "Fatal",
    }
    for outcome, group in plot_df.groupby("model", sort=False):
        group = group.sort_values("cohort")
        ax.plot(
            group["cohort"],
            group["IRR"],
            marker="o",
            label=severity_display.get(outcome, outcome),
        )
    ax.axhline(1, linestyle="--")
    ax.set_xticks(MAIN_COHORTS)
    ax.set_xticklabels([COHORT_DISPLAY[c] for c in MAIN_COHORTS])
    ax.set_xlabel("Production cohort")
    ax.set_ylabel("IRR relative to the 2000–2004 cohort")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    save_figure(fig, outdir / "fig_cohort_severity_irrs.png", show)


# =============================================================================
# 10. FEATURE DESCRIPTIVES AND MODELS
# =============================================================================


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    valid = values.notna() & weights.notna() & weights.gt(0)
    if not valid.any():
        return np.nan
    return float(np.average(values[valid], weights=weights[valid]))


def feature_descriptives(
    feature_df: pd.DataFrame,
    active: list[str],
    passive: list[str],
    outdir: Path,
    show: bool,
    diffusion_df: Optional[pd.DataFrame] = None,
) -> None:
    all_features = active + passive
    cleaning = feature_df.attrs.get("feature_cleaning")
    if isinstance(cleaning, pd.DataFrame):
        cleaning.to_csv(outdir / "table_feature_cleaning_diagnostics.csv", index=False)

    diag_rows = []
    for feature in all_features:
        diag_rows.append(
            {
                "feature": feature,
                "feature_label": FEATURE_LABELS.get(feature, feature),
                "group": "active" if feature in active else "passive/support",
                "unweighted_prevalence": feature_df[feature].mean(skipna=True),
                "vehicle_weighted_prevalence": weighted_mean(
                    feature_df[feature], feature_df["n_vehicles"]
                ),
                "missing_rate": feature_df[feature].isna().mean(),
            }
        )
    pd.DataFrame(diag_rows).to_csv(
        outdir / "table_feature_diagnostics.csv", index=False
    )

    display_df = feature_df.copy() if diffusion_df is None else diffusion_df.copy()
    display_df = display_df[display_df["cohort_5yr_int"].between(1995, 2020)].copy()
    for feature in all_features:
        display_df[feature] = pd.to_numeric(display_df[feature], errors="coerce")
        display_df.loc[~display_df[feature].isin([0, 1]), feature] = np.nan
    display_df["active_count"] = display_df[active].fillna(0).sum(axis=1)
    display_df["passive_count"] = display_df[passive].fillna(0).sum(axis=1)
    display_df["total_feature_count"] = (
        display_df["active_count"] + display_df["passive_count"]
    )

    cohort_rows = []
    for cohort, group in display_df.groupby("cohort_5yr_int"):
        row = {
            "cohort": int(cohort),
            "rows": len(group),
            "represented_vehicles": group["n_vehicles"].sum(),
            "total_km": group["total_km"].sum(),
            "mean_active_count_weighted": weighted_mean(
                group["active_count"], group["n_vehicles"]
            ),
            "mean_passive_count_weighted": weighted_mean(
                group["passive_count"], group["n_vehicles"]
            ),
            "mean_total_feature_count_weighted": weighted_mean(
                group["total_feature_count"], group["n_vehicles"]
            ),
        }
        for feature in all_features:
            row[f"{feature}_vehicle_share_pct"] = (
                weighted_mean(group[feature], group["n_vehicles"]) * 100
            )
        cohort_rows.append(row)
    cohort_summary = pd.DataFrame(cohort_rows).sort_values("cohort")
    cohort_summary.to_csv(outdir / "table_feature_diffusion_by_cohort.csv", index=False)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [0.85, 1.35]},
    )
    axes[0].plot(
        cohort_summary["cohort"],
        cohort_summary["mean_total_feature_count_weighted"],
        marker="o",
    )
    axes[0].set_ylabel("Mean measured feature count")
    axes[0].set_ylim(bottom=0)
    axes[0].grid(axis="y", alpha=0.3)

    for feature in all_features:
        axes[1].plot(
            cohort_summary["cohort"],
            cohort_summary[f"{feature}_vehicle_share_pct"],
            marker="o",
            label=FEATURE_LABELS.get(feature, feature),
        )
    axes[1].set_xlabel("Production cohort")
    axes[1].set_ylabel("Represented vehicles with feature (%)")
    axes[1].set_ylim(bottom=0)
    diffusion_cohorts = sorted(cohort_summary["cohort"].astype(int).unique().tolist())
    axes[1].set_xticks(diffusion_cohorts)
    axes[1].set_xticklabels(
        [COHORT_DISPLAY.get(c, str(c)) for c in diffusion_cohorts], rotation=0
    )
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
    fig.tight_layout(rect=(0, 0, 0.82, 1))
    fig.savefig(
        outdir / "fig_feature_diffusion_combined_two_panel.png",
        dpi=300,
        bbox_inches="tight",
    )
    if show:
        plt.show()
    plt.close(fig)

    # Feature-count distributions with represented-vehicle weights.
    distribution_rows = []
    for count_col in ["active_count", "passive_count", "total_feature_count"]:
        grouped = (
            feature_df.groupby(count_col)
            .agg(
                rows=("any_events", "size"),
                represented_vehicles=("n_vehicles", "sum"),
                total_km=("total_km", "sum"),
                accident_events=("any_events", "sum"),
            )
            .reset_index()
        )
        grouped["count_variable"] = count_col
        grouped["accidents_per_100m_km"] = (
            grouped["accident_events"] / grouped["total_km"] * RATE_SCALE
        )
        distribution_rows.append(grouped)
    pd.concat(distribution_rows, ignore_index=True).to_csv(
        outdir / "table_feature_count_distributions.csv", index=False
    )


def feature_baseline_formula(extra_controls: str = "") -> str:
    formula = (
        f"any_events ~ {age_term()} + {cohort_term(FEATURE_COHORT_REFERENCE)} + C(fuel)"
    )
    if extra_controls.strip():
        formula += " + " + extra_controls.strip().lstrip("+").strip()
    return formula


def run_single_feature_models(
    feature_df: pd.DataFrame,
    active: list[str],
    passive: list[str],
    outdir: Path,
    show: bool,
    extra_controls: str = "",
) -> pd.DataFrame:
    all_features = active + passive
    baseline_formula = feature_baseline_formula(extra_controls)
    baseline_poisson, baseline_quasi = fit_poisson_models(
        "feature_baseline", baseline_formula, feature_df, outdir, fit_robust=False
    )

    summary_rows = []
    for feature in all_features:
        subset = feature_df[feature].notna()
        data = feature_df.loc[subset].copy()
        values = sorted(data[feature].dropna().unique().tolist())
        if values != [0.0, 1.0] and values != [0, 1]:
            summary_rows.append(
                {
                    "feature": feature,
                    "feature_label": FEATURE_LABELS.get(feature, feature),
                    "group": "active" if feature in active else "passive/support",
                    "n_rows": len(data),
                    "prevalence": data[feature].mean(skipna=True),
                    "IRR": np.nan,
                    "CI_low": np.nan,
                    "CI_high": np.nan,
                    "p_value": np.nan,
                    "dispersion": np.nan,
                    "status": "skipped_no_binary_variation",
                }
            )
            continue

        formula = baseline_formula + f" + {feature}"
        _, quasi = fit_poisson_models(
            f"single_feature_{feature}", formula, data, outdir, fit_robust=False
        )
        row = quasi.table.loc[quasi.table["term"] == feature]
        if row.empty:
            result_row = {
                "IRR": np.nan,
                "CI_low": np.nan,
                "CI_high": np.nan,
                "p_value": np.nan,
            }
        else:
            result_row = row.iloc[0].to_dict()
        summary_rows.append(
            {
                "feature": feature,
                "feature_label": FEATURE_LABELS.get(feature, feature),
                "group": "active" if feature in active else "passive/support",
                "n_rows": len(data),
                "prevalence": data[feature].mean(skipna=True),
                "IRR": result_row.get("IRR", np.nan),
                "CI_low": result_row.get("CI_low", np.nan),
                "CI_high": result_row.get("CI_high", np.nan),
                "p_value": result_row.get("p_value", np.nan),
                "dispersion": quasi.dispersion,
                "status": "ok",
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(outdir / "table_single_feature_model_summary.csv", index=False)

    plot = summary[summary["status"] == "ok"].copy().sort_values("IRR")
    if not plot.empty:
        y = np.arange(len(plot))
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = {"active": "tab:orange", "passive/support": "tab:blue"}
        for yi, (_, row) in enumerate(plot.iterrows()):
            ax.errorbar(
                row["IRR"],
                yi,
                xerr=np.array(
                    [[row["IRR"] - row["CI_low"]], [row["CI_high"] - row["IRR"]]]
                ),
                fmt="o",
                capsize=4,
                color=colors[row["group"]],
            )
        ax.axvline(1, linestyle="--", color="gray")
        ax.set_yticks(y)
        ax.set_yticklabels(plot["feature_label"])
        ax.invert_yaxis()
        ax.set_xlabel("Incidence rate ratio")
        ax.scatter([], [], color="tab:orange", label="Active-designated features")
        ax.scatter([], [], color="tab:blue", label="Warning/safety-support features")
        ax.legend(loc="lower left")
        ax.grid(axis="x", alpha=0.3)
        save_figure(fig, outdir / "fig_single_feature_forest_plot.png", show)
    return summary


def feature_count_model_specs(
    outcome: str = "any_events", extra_controls: str = ""
) -> list[tuple[str, str, list[str]]]:
    base_rhs = (
        f"{age_term()} + {cohort_term(FEATURE_COHORT_REFERENCE)} + C(fuel)"
    )
    if extra_controls.strip():
        base_rhs += " + " + extra_controls.strip().lstrip("+").strip()
    return [
        ("baseline", f"{outcome} ~ {base_rhs}", []),
        ("active_only", f"{outcome} ~ {base_rhs} + active_count", ["active_count"]),
        ("passive_only", f"{outcome} ~ {base_rhs} + passive_count", ["passive_count"]),
        (
            "active_and_passive",
            f"{outcome} ~ {base_rhs} + active_count + passive_count",
            ["active_count", "passive_count"],
        ),
        (
            "total_feature_count",
            f"{outcome} ~ {base_rhs} + total_feature_count",
            ["total_feature_count"],
        ),
    ]


def run_feature_count_and_attenuation_models(
    feature_df: pd.DataFrame,
    count_outdir: Path,
    attenuation_outdir: Path,
    show: bool,
    extra_controls: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    focus_rows = []
    model_tables: dict[str, pd.DataFrame] = {}
    model_results: dict[str, FittedModel] = {}

    for model_name, formula, focus_terms in feature_count_model_specs(
        extra_controls=extra_controls
    ):
        _, quasi = fit_poisson_models(
            f"feature_count_{model_name}", formula, feature_df, count_outdir, fit_robust=False
        )
        model_tables[model_name] = quasi.table
        model_results[model_name] = quasi
        if not focus_terms:
            focus_rows.append(
                {
                    "model": model_name,
                    "term": "none",
                    "IRR": np.nan,
                    "CI_low": np.nan,
                    "CI_high": np.nan,
                    "p_value": np.nan,
                    "dispersion": quasi.dispersion,
                }
            )
        for term in focus_terms:
            row = quasi.table.loc[quasi.table["term"] == term]
            if row.empty:
                continue
            row = row.iloc[0]
            focus_rows.append(
                {
                    "model": model_name,
                    "term": term,
                    "IRR": row["IRR"],
                    "CI_low": row["CI_low"],
                    "CI_high": row["CI_high"],
                    "p_value": row["p_value"],
                    "dispersion": quasi.dispersion,
                }
            )

    focus_summary = pd.DataFrame(focus_rows)
    focus_summary.to_csv(count_outdir / "table_feature_count_model_summary.csv", index=False)

    plot = focus_summary[focus_summary["term"] != "none"].copy()
    if not plot.empty:
        plot["label"] = plot["model"] + ": " + plot["term"]
        y = np.arange(len(plot))
        fig, ax = plt.subplots(figsize=(10, 6))
        xerr = np.vstack(
            [plot["IRR"] - plot["CI_low"], plot["CI_high"] - plot["IRR"]]
        )
        ax.errorbar(plot["IRR"], y, xerr=xerr, fmt="o", capsize=4)
        ax.axvline(1, linestyle="--")
        ax.set_yticks(y)
        ax.set_yticklabels(plot["label"])
        ax.set_xlabel("Incidence rate ratio per additional measured feature")
        ax.set_title("Feature-count associations with total accident rate")
        ax.grid(axis="x", alpha=0.3)
        save_figure(fig, count_outdir / "fig_feature_count_forest_plot.png", show)

    # Attenuation of the 2020 vs 2010 cohort coefficient.
    baseline_comp = comparison_from_table(
        model_tables["baseline"], 2020, int(FEATURE_COHORT_REFERENCE)
    )
    attenuation_rows = []
    cohort_frames = []
    for model_name, table in model_tables.items():
        comparison = comparison_from_table(
            table, 2020, int(FEATURE_COHORT_REFERENCE)
        )
        attenuation = np.nan
        if (
            model_name != "baseline"
            and np.isfinite(baseline_comp["coef"])
            and baseline_comp["coef"] != 0
            and np.isfinite(comparison["coef"])
        ):
            attenuation = 100 * (
                1 - comparison["coef"] / baseline_comp["coef"]
            )
        attenuation_rows.append(
            {
                "model": model_name,
                "IRR_2020_vs_2010": comparison["IRR"],
                "CI_low": comparison["CI_low"],
                "CI_high": comparison["CI_high"],
                "p_value": comparison["p_value"],
                "log_coef_2020_vs_2010": comparison["coef"],
                "baseline_log_coef": baseline_comp["coef"],
                "attenuation_pct": attenuation,
                "n_rows": len(feature_df),
            }
        )
        cohort_frames.append(
            cohort_rows_from_table(
                table,
                FEATURE_COHORTS,
                int(FEATURE_COHORT_REFERENCE),
                model_name,
            )
        )

    attenuation_df = pd.DataFrame(attenuation_rows)
    attenuation_df.to_csv(
        attenuation_outdir / "table_feature_cohort_attenuation.csv", index=False
    )

    cohort_plot = pd.concat(cohort_frames, ignore_index=True)
    cohort_plot.to_csv(
        attenuation_outdir / "table_cohort_irrs_across_feature_models.csv", index=False
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    feature_model_display = {
        "baseline": "Baseline",
        "active_only": "Active count",
        "passive_only": "Warning/support count",
        "active_and_passive": "Active + warning/support counts",
        "total_feature_count": "Total feature count",
    }
    for model_name, group in cohort_plot.groupby("model", sort=False):
        group = group.sort_values("cohort")
        ax.plot(
            group["cohort"],
            group["IRR"],
            marker="o",
            label=feature_model_display.get(model_name, model_name),
        )
    ax.axhline(1, linestyle="--")
    ax.set_xticks(FEATURE_COHORTS)
    ax.set_xticklabels([COHORT_DISPLAY[c] for c in FEATURE_COHORTS])
    ax.set_xlabel("Production cohort")
    ax.set_ylabel("IRR relative to the 2010–2014 cohort")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    save_figure(
        fig,
        attenuation_outdir / "fig_cohort_irrs_across_feature_models.png",
        show,
    )
    return focus_summary, attenuation_df


def run_feature_severity_models(
    feature_df: pd.DataFrame,
    outdir: Path,
    show: bool,
) -> pd.DataFrame:
    summary_rows = []
    for outcome in SEVERITY_OUTCOMES:
        outcome_dir = outdir / outcome
        outcome_dir.mkdir(parents=True, exist_ok=True)
        for model_name, formula, focus_terms in feature_count_model_specs(outcome):
            _, quasi = fit_poisson_models(
                f"{outcome}_{model_name}", formula, feature_df, outcome_dir, fit_robust=False
            )
            if not focus_terms:
                summary_rows.append(
                    {
                        "outcome": outcome,
                        "model": model_name,
                        "term": "none",
                        "IRR": np.nan,
                        "CI_low": np.nan,
                        "CI_high": np.nan,
                        "p_value": np.nan,
                        "dispersion": quasi.dispersion,
                    }
                )
            for term in focus_terms:
                row = quasi.table.loc[quasi.table["term"] == term]
                if row.empty:
                    continue
                row = row.iloc[0]
                summary_rows.append(
                    {
                        "outcome": outcome,
                        "model": model_name,
                        "term": term,
                        "IRR": row["IRR"],
                        "CI_low": row["CI_low"],
                        "CI_high": row["CI_high"],
                        "p_value": row["p_value"],
                        "dispersion": quasi.dispersion,
                    }
                )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(outdir / "table_feature_severity_model_summary.csv", index=False)

    plot = summary[summary["term"] != "none"].copy()
    if not plot.empty:
        # Paper Figure 11: separately estimated active, warning/support, and
        # total feature-count models, compared across severity outcomes.
        paper_specs = [
            ("active_only", "active_count", "Active count"),
            ("passive_only", "passive_count", "Warning/support count"),
            ("total_feature_count", "total_feature_count", "Total feature count"),
        ]
        outcome_order = ["light_events", "serious_events", "severe_events", "fatal_events"]
        outcome_display = ["Light", "Serious", "Serious + fatal", "Fatal"]
        x = np.arange(len(outcome_order), dtype=float)
        width = 0.23
        fig, ax = plt.subplots(figsize=(10, 6))
        for j, (model_name, term, label) in enumerate(paper_specs):
            vals = []
            for outcome in outcome_order:
                row = plot[
                    (plot["outcome"] == outcome)
                    & (plot["model"] == model_name)
                    & (plot["term"] == term)
                ]
                vals.append(float(row.iloc[0]["IRR"]) if not row.empty else np.nan)
            ax.bar(x + (j - 1) * width, vals, width=width, label=label)
        ax.axhline(1, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(outcome_display)
        ax.set_ylabel("Incidence rate ratio")
        finite = plot["IRR"].replace([np.inf, -np.inf], np.nan).dropna()
        if not finite.empty:
            ax.set_ylim(min(0.88, float(finite.min()) - 0.02), 1.02)
        ax.legend(loc="upper right")
        save_figure(fig, outdir / "fig_feature_bundle_associations_by_severity.png", show)

        # One line per outcome for the total-feature-count model provides the
        # cleanest comparable severity view.
        total_plot = plot[
            (plot["model"] == "total_feature_count")
            & (plot["term"] == "total_feature_count")
        ].copy()
        if not total_plot.empty:
            y = np.arange(len(total_plot))
            fig, ax = plt.subplots(figsize=(9, 5.5))
            xerr = np.vstack(
                [
                    total_plot["IRR"] - total_plot["CI_low"],
                    total_plot["CI_high"] - total_plot["IRR"],
                ]
            )
            ax.errorbar(total_plot["IRR"], y, xerr=xerr, fmt="o", capsize=4)
            ax.axvline(1, linestyle="--")
            ax.set_yticks(y)
            ax.set_yticklabels(total_plot["outcome"])
            ax.set_xlabel("IRR per additional measured feature")
            ax.set_title("Total feature-count association by accident severity")
            ax.grid(axis="x", alpha=0.3)
            save_figure(fig, outdir / "fig_total_feature_count_by_severity.png", show)
    return summary


# =============================================================================
# 11. ENGINE-VOLUME SENSITIVITY ANALYSIS
# =============================================================================


def run_engine_volume_sensitivity(
    main_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    active: list[str],
    passive: list[str],
    original_main_models: dict[str, FittedModel],
    original_single_summary: pd.DataFrame,
    original_count_summary: pd.DataFrame,
    original_attenuation: pd.DataFrame,
    dirs: dict[str, Path],
    skip_nb2: bool,
    show: bool,
) -> pd.DataFrame:
    """Add engine-volume category to the key published model families."""
    control = engine_term()
    main_formula = (
        f"any_events ~ {age_term()} + {cohort_term(MAIN_COHORT_REFERENCE)} "
        f"+ C(fuel) + {control}"
    )
    write_text(dirs["engine"] / "main_plus_engine_volume_formula.txt", main_formula)

    engine_poisson, engine_quasi = fit_poisson_models(
        "main_plus_engine_volume", main_formula, main_df, dirs["engine"]
    )
    engine_nb2: Optional[FittedModel] = None
    if not skip_nb2:
        engine_nb2 = fit_true_nb2(
            "main_plus_engine_volume", main_formula, main_df, dirs["engine"]
        )

    main_rows = []
    main_models_for_compare: list[tuple[str, FittedModel]] = [
        ("Published primary Poisson", original_main_models["Main Poisson"]),
        ("Poisson + engine-volume category", engine_poisson),
        ("Quasi-Poisson-style + engine-volume category", engine_quasi),
    ]
    if engine_nb2 is not None:
        main_models_for_compare.append(("NB2 + engine-volume category", engine_nb2))

    for label, fitted in main_models_for_compare:
        comp = comparison_from_table(fitted.table, 2020, 2000)
        main_rows.append(
            {
                "model": label,
                "IRR_2020_vs_2000": comp["IRR"],
                "CI_low": comp["CI_low"],
                "CI_high": comp["CI_high"],
                "p_value": comp["p_value"],
                "log_coef_2020_vs_2000": comp["coef"],
                "percent_lower_rate": (1 - comp["IRR"]) * 100,
                "n_rows": fitted.data_rows,
                "model_family": fitted.model_family,
            }
        )
    main_comparison = pd.DataFrame(main_rows)
    main_comparison.to_csv(
        dirs["engine"] / "table_main_cohort_original_vs_engine_control.csv",
        index=False,
    )

    # Visual comparison of the published primary cohort coefficients with the
    # engine-volume-adjusted Poisson coefficients.
    original_cohorts = cohort_rows_from_table(
        original_main_models["Main Poisson"].table,
        MAIN_COHORTS,
        2000,
        "Published primary Poisson",
    )
    engine_cohorts = cohort_rows_from_table(
        engine_poisson.table,
        MAIN_COHORTS,
        2000,
        "Poisson + engine-volume category",
    )
    cohort_compare = pd.concat([original_cohorts, engine_cohorts], ignore_index=True)
    cohort_compare.to_csv(
        dirs["engine"] / "table_cohort_irrs_original_vs_engine_control.csv",
        index=False,
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, group in cohort_compare.groupby("model", sort=False):
        group = group.sort_values("cohort")
        ax.plot(group["cohort"], group["IRR"], marker="o", label=label)
    ax.axhline(1, linestyle="--")
    ax.set_xticks(MAIN_COHORTS)
    ax.set_xticklabels([COHORT_DISPLAY[c] for c in MAIN_COHORTS])
    ax.set_xlabel("Production cohort")
    ax.set_ylabel("IRR relative to the 2000–2004 cohort")
    ax.legend()
    save_figure(
        fig,
        dirs["engine"] / "fig_main_cohort_original_vs_engine_control.png",
        show,
    )

    engine_single = run_single_feature_models(
        feature_df,
        active,
        passive,
        dirs["engine_single"],
        show,
        extra_controls=control,
    )
    single_compare = original_single_summary.merge(
        engine_single,
        on=["feature", "feature_label", "group"],
        how="outer",
        suffixes=("_original", "_engine_control"),
    )
    single_compare.to_csv(
        dirs["engine"] / "table_single_features_original_vs_engine_control.csv",
        index=False,
    )

    engine_count, engine_attenuation = run_feature_count_and_attenuation_models(
        feature_df,
        dirs["engine_counts"],
        dirs["engine_attenuation"],
        show,
        extra_controls=control,
    )
    count_compare = original_count_summary.merge(
        engine_count,
        on=["model", "term"],
        how="outer",
        suffixes=("_original", "_engine_control"),
    )
    count_compare.to_csv(
        dirs["engine"] / "table_feature_counts_original_vs_engine_control.csv",
        index=False,
    )

    attenuation_compare = original_attenuation.merge(
        engine_attenuation,
        on="model",
        how="outer",
        suffixes=("_original", "_engine_control"),
    )
    attenuation_compare.to_csv(
        dirs["engine"] / "table_attenuation_original_vs_engine_control.csv",
        index=False,
    )

    def attenuation_value(frame: pd.DataFrame, model: str) -> float:
        row = frame.loc[frame["model"] == model, "attenuation_pct"]
        return float(row.iloc[0]) if not row.empty else np.nan

    def count_irr(frame: pd.DataFrame, model: str, term: str) -> float:
        row = frame.loc[
            (frame["model"] == model) & (frame["term"] == term), "IRR"
        ]
        return float(row.iloc[0]) if not row.empty else np.nan

    original_comp = comparison_from_table(
        original_main_models["Main Poisson"].table, 2020, 2000
    )
    engine_comp = comparison_from_table(engine_poisson.table, 2020, 2000)
    key_summary = pd.DataFrame(
        [
            {
                "result": "Main cohort IRR: 2020 vs 2000",
                "original": original_comp["IRR"],
                "engine_adjusted": engine_comp["IRR"],
                "change_engine_minus_original": engine_comp["IRR"] - original_comp["IRR"],
            },
            {
                "result": "Attenuation: active + warning/support counts (%)",
                "original": attenuation_value(original_attenuation, "active_and_passive"),
                "engine_adjusted": attenuation_value(engine_attenuation, "active_and_passive"),
                "change_engine_minus_original": (
                    attenuation_value(engine_attenuation, "active_and_passive")
                    - attenuation_value(original_attenuation, "active_and_passive")
                ),
            },
            {
                "result": "Attenuation: total feature count (%)",
                "original": attenuation_value(original_attenuation, "total_feature_count"),
                "engine_adjusted": attenuation_value(engine_attenuation, "total_feature_count"),
                "change_engine_minus_original": (
                    attenuation_value(engine_attenuation, "total_feature_count")
                    - attenuation_value(original_attenuation, "total_feature_count")
                ),
            },
            {
                "result": "Feature-count IRR: active count (separate model)",
                "original": count_irr(original_count_summary, "active_only", "active_count"),
                "engine_adjusted": count_irr(engine_count, "active_only", "active_count"),
                "change_engine_minus_original": (
                    count_irr(engine_count, "active_only", "active_count")
                    - count_irr(original_count_summary, "active_only", "active_count")
                ),
            },
            {
                "result": "Feature-count IRR: warning/support count (separate model)",
                "original": count_irr(original_count_summary, "passive_only", "passive_count"),
                "engine_adjusted": count_irr(engine_count, "passive_only", "passive_count"),
                "change_engine_minus_original": (
                    count_irr(engine_count, "passive_only", "passive_count")
                    - count_irr(original_count_summary, "passive_only", "passive_count")
                ),
            },
            {
                "result": "Feature-count IRR: active count (joint model)",
                "original": count_irr(original_count_summary, "active_and_passive", "active_count"),
                "engine_adjusted": count_irr(engine_count, "active_and_passive", "active_count"),
                "change_engine_minus_original": (
                    count_irr(engine_count, "active_and_passive", "active_count")
                    - count_irr(original_count_summary, "active_and_passive", "active_count")
                ),
            },
            {
                "result": "Feature-count IRR: warning/support count (joint model)",
                "original": count_irr(original_count_summary, "active_and_passive", "passive_count"),
                "engine_adjusted": count_irr(engine_count, "active_and_passive", "passive_count"),
                "change_engine_minus_original": (
                    count_irr(engine_count, "active_and_passive", "passive_count")
                    - count_irr(original_count_summary, "active_and_passive", "passive_count")
                ),
            },
            {
                "result": "Feature-count IRR: total measured feature count",
                "original": count_irr(original_count_summary, "total_feature_count", "total_feature_count"),
                "engine_adjusted": count_irr(engine_count, "total_feature_count", "total_feature_count"),
                "change_engine_minus_original": (
                    count_irr(engine_count, "total_feature_count", "total_feature_count")
                    - count_irr(original_count_summary, "total_feature_count", "total_feature_count")
                ),
            },
        ]
    )
    key_summary.to_csv(
        dirs["engine"] / "table_engine_volume_sensitivity_key_results.csv",
        index=False,
    )
    return key_summary


def collect_paper_figures(dirs: dict[str, Path]) -> None:
    """Place one clearly numbered copy of every data-driven paper figure together."""
    mappings = [
        (
            dirs["sample"] / "fig_cohort_age_accident_rate_heatmap_main_2000_2020.png",
            dirs["paper"] / "figure_04_accident_rates_by_cohort_and_age_bin.png",
        ),
        (
            dirs["cohort"] / "fig_cohort_main_robustness_checks.png",
            dirs["paper"] / "figure_05_cohort_effects_main_robustness_checks.png",
        ),
        (
            dirs["cohort"] / "fig_cohort_year_fe_comparison.png",
            dirs["paper"] / "figure_06_cohort_effects_calendar_year_adjustment.png",
        ),
        (
            dirs["severity"] / "fig_severity_rates_by_cohort.png",
            dirs["paper"] / "figure_07_exposure_adjusted_rates_by_severity_and_cohort.png",
        ),
        (
            dirs["severity"] / "fig_cohort_severity_irrs.png",
            dirs["paper"] / "figure_08_severity_specific_cohort_irrs.png",
        ),
        (
            dirs["features"] / "fig_feature_diffusion_combined_two_panel.png",
            dirs["paper"] / "figure_09_feature_diffusion_across_cohorts.png",
        ),
        (
            dirs["feature_single"] / "fig_single_feature_forest_plot.png",
            dirs["paper"] / "figure_10_single_feature_associations.png",
        ),
        (
            dirs["feature_severity"] / "fig_feature_bundle_associations_by_severity.png",
            dirs["paper"] / "figure_11_feature_bundle_associations_by_severity.png",
        ),
        (
            dirs["feature_attenuation"] / "fig_cohort_irrs_across_feature_models.png",
            dirs["paper"] / "figure_12_cohort_irrs_with_and_without_feature_counts.png",
        ),
    ]
    missing = []
    for source, destination in mappings:
        if source.exists():
            shutil.copy2(source, destination)
        else:
            missing.append(str(source))
    if missing:
        warnings.warn(
            "Some paper figures were not collected (usually because a skip flag was used):\n"
            + "\n".join(missing)
        )


# =============================================================================
# 12. RUN LOG AND MANIFEST
# =============================================================================


def save_manifest(
    data_path: Path,
    dirs: dict[str, Path],
    args: argparse.Namespace,
    df_all: pd.DataFrame,
    age_df: pd.DataFrame,
    main_df: pd.DataFrame,
    feature_df: pd.DataFrame,
) -> None:
    manifest = {
        "data_file": data_path.name,
        "output_directory": dirs["root"].name,
        "uploaded_aggregated_rows": len(df_all),
        "age_valid_rows": len(age_df),
        "main_cohort_rows": len(main_df),
        "feature_rows": len(feature_df),
        "main_age_range": [AGE_MIN, AGE_MAX],
        "main_cohorts": [MAIN_COHORT_MIN, MAIN_COHORT_MAX],
        "main_reference_cohort": MAIN_COHORT_REFERENCE,
        "feature_cohorts": [FEATURE_COHORT_MIN, FEATURE_COHORT_MAX],
        "feature_reference_cohort": FEATURE_COHORT_REFERENCE,
        "engine_volume_variable": "NEFAH_KV",
        "engine_volume_reference": ENGINE_VOLUME_REFERENCE,
        "engine_volume_modeled_as": "categorical sensitivity control",
        "low_exposure_km": args.low_exposure_km,
        "max_imputed_frac": args.max_imputed_frac,
        "covid_years_excluded_in_sensitivity": COVID_YEARS,
        "skip_nb2": args.skip_nb2,
        "skip_feature_severity": args.skip_feature_severity,
        "python_version": sys.version,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "statsmodels_version": sm.__version__,
    }
    write_text(
        dirs["logs"] / "analysis_manifest.json",
        json.dumps(manifest, indent=2, ensure_ascii=False),
    )


# =============================================================================
# 13. MAIN EXECUTION
# =============================================================================


def main() -> int:
    args = parse_args()
    data_path = resolve_data_path(args.data)
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent if script_dir.name.lower() == "analysis" else script_dir
    output_root = (
        args.output.expanduser().resolve()
        if args.output is not None
        else project_root / "outputs" / "ADAS_complete_analysis_outputs"
    )
    dirs = make_output_dirs(output_root)

    print("=" * 80)
    print("COMPLETE ADAS / SOFI FINAL-PROJECT ANALYSIS")
    print("=" * 80)
    print("Data file:", data_path)
    print("Output directory:", output_root)
    print("Primary sample: vehicle ages 1-20, cohorts 2000-2020, ref=2000")
    print("Feature sample: vehicle ages 1-20, cohorts 2010-2020, ref=2010")

    raw = pd.read_csv(data_path)
    numeric = validate_and_numericize(raw)

    age_df = add_common_variables(
        numeric,
        AGE_MIN,
        AGE_MAX,
        AGE_BIN_EDGES,
        AGE_BIN_LABELS,
    )
    main_df = restrict_cohorts(
        age_df,
        MAIN_COHORT_MIN,
        MAIN_COHORT_MAX,
        MAIN_COHORTS,
    )
    feature_df, active_features, passive_features = prepare_feature_sample(age_df)

    print("Uploaded aggregated rows:", f"{len(raw):,}")
    print("Age-valid rows (1-20):", f"{len(age_df):,}")
    print("Primary cohort rows (2000-2020):", f"{len(main_df):,}")
    print("Feature-analysis rows (2010-2020):", f"{len(feature_df):,}")
    print("Active features:", active_features)
    print("Passive/support features:", passive_features)

    save_sample_tables(raw, age_df, main_df, feature_df, dirs["sample"])
    descriptive_age_tables_and_figures(
        numeric, age_df, dirs["sample"], args.show_plots
    )
    paper_age_figures(numeric, dirs["paper"], args.show_plots)

    # Analytic heat map for the final main cohort sample.
    cohort_age_heatmap(
        main_df,
        MAIN_COHORTS,
        dirs["sample"],
        "fig_cohort_age_accident_rate_heatmap_main_2000_2020",
        "Exposure-adjusted accident rates by production cohort and vehicle age",
        args.show_plots,
    )
    # Optional descriptive bridge retaining 1995, matching the earlier discussion.
    descriptive_1995 = age_df[
        age_df["cohort_5yr_int"].between(1995, 2020)
    ].copy()
    cohort_age_heatmap(
        descriptive_1995,
        [1995, 2000, 2005, 2010, 2015, 2020],
        dirs["sample"],
        "fig_cohort_age_accident_rate_heatmap_descriptive_1995_2020",
        "Descriptive accident rates by production cohort and vehicle age",
        args.show_plots,
    )

    models, comparison_df = run_main_cohort_models(
        main_df,
        dirs,
        args.low_exposure_km,
        args.max_imputed_frac,
        args.skip_nb2,
        args.show_plots,
    )
    predicted_rates_by_age_and_cohort(
        models["Main Poisson"], main_df, dirs["cohort"], args.show_plots
    )
    run_sample_definition_checks(numeric, dirs["robust"], args.show_plots)
    run_cohort_severity_analysis(
        main_df, dirs["severity"], args.skip_nb2, args.show_plots
    )

    diffusion_df = age_df[age_df["cohort_5yr_int"].between(1995, 2020)].copy()
    feature_descriptives(
        feature_df,
        active_features,
        passive_features,
        dirs["features"],
        args.show_plots,
        diffusion_df=diffusion_df,
    )
    original_single_summary = run_single_feature_models(
        feature_df,
        active_features,
        passive_features,
        dirs["feature_single"],
        args.show_plots,
    )
    original_count_summary, original_attenuation = run_feature_count_and_attenuation_models(
        feature_df,
        dirs["feature_counts"],
        dirs["feature_attenuation"],
        args.show_plots,
    )
    if not args.skip_feature_severity:
        run_feature_severity_models(
            feature_df, dirs["feature_severity"], args.show_plots
        )

    engine_summary = run_engine_volume_sensitivity(
        main_df,
        feature_df,
        active_features,
        passive_features,
        models,
        original_single_summary,
        original_count_summary,
        original_attenuation,
        dirs,
        args.skip_nb2,
        args.show_plots,
    )

    collect_paper_figures(dirs)

    save_manifest(data_path, dirs, args, raw, age_df, main_df, feature_df)

    print("\n" + "=" * 80)
    print("MAIN COHORT COMPARISON TABLE")
    print("=" * 80)
    print(
        comparison_df[
            [
                "model_check",
                "IRR_2020_vs_2000",
                "CI_low",
                "CI_high",
                "approximate_difference",
                "n_rows",
                "status",
            ]
        ].to_string(index=False)
    )
    print("\n" + "=" * 80)
    print("ENGINE-VOLUME SENSITIVITY: KEY RESULTS")
    print("=" * 80)
    print(engine_summary.to_string(index=False))
    print("\nPaper Figures 2-12:", dirs["paper"])
    print("\nAll outputs saved in:", output_root)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("\nFATAL ERROR:", exc, file=sys.stderr)
        traceback.print_exc()
        raise
