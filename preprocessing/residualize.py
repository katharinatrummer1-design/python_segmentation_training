"""Temperature residualization of tabular features.

Scientific motivation
----------------------
Temperature systematically affects cricket calling-song structure (pulse rate
and spectral properties scale with temperature). If recording temperature
co-varies with individual identity, a classifier may exploit temperature rather
than a temperature-invariant individual signature.

Residualization removes the *linear* temperature component from each feature:
for every feature ``x`` we fit ``x ~ a + b * temperature`` and keep the residual
``x - (a + b * temperature)``. Re-running identification on residualized features
tests whether identity survives temperature correction.

Data dependency
---------------
This requires populated temperature values (``temperature_c``). In datasets
without temperature logs the step is a no-op: it writes a clear *blocker* report
and leaves the pipeline runnable, rather than fabricating values.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from cricket_id.io.manifest import _write_parquet_compatible
from cricket_id.models.pca_baseline import EXCLUDED_FEATURE_COLUMNS
from cricket_id.utils.paths import resolve_data_dir, resolve_reports_dir

TEMP_COLUMN = "temperature_c"


def _feature_columns(df: pd.DataFrame) -> list[str]:
    return [
        column
        for column in df.columns
        if column not in EXCLUDED_FEATURE_COLUMNS
        and pd.api.types.is_numeric_dtype(df[column])
    ]


def _temperature_available(df: pd.DataFrame) -> bool:
    if TEMP_COLUMN not in df.columns:
        return False
    temperature = pd.to_numeric(df[TEMP_COLUMN], errors="coerce")
    return bool(temperature.notna().sum() >= 2 and temperature.nunique(dropna=True) >= 2)


def _write_blocker_report(report_path: Path, reason: str) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "# Temperature Residualization — BLOCKED\n\n"
        f"Status: **blocked** — {reason}\n\n"
        "## What this step would do\n\n"
        "Remove the linear temperature component from every tabular feature "
        "(`x - (a + b * temperature_c)`) and re-export the feature table, so the "
        "PCA/CNN identification can be re-run on temperature-corrected features.\n\n"
        "## Why it is blocked\n\n"
        "No usable `temperature_c` values are present in the feature table. The "
        "`temperature.csv` manifest is empty for this dataset, so there is nothing "
        "to regress against.\n\n"
        "## How to unblock\n\n"
        "1. Provide a populated `data/raw/temperature.csv` with columns "
        "`timestamp_utc, temperature_c, recording_id`.\n"
        "2. Re-run `validate-manifest` (merges temperature into `recordings_qc.parquet`) "
        "and `extract-features` (propagates `temperature_c` onto segments).\n"
        "3. Re-run `residualize-temp` — it will then produce "
        "`data/processed/features_tabular_residualized.parquet` and a coverage report.\n",
        encoding="utf-8",
    )


def run_temperature_residualization(
    features_df: pd.DataFrame,
    config: dict,
    *,
    project_root: str | Path | None = None,
) -> dict:
    """Residualize features on temperature when available; otherwise emit a blocker.

    Returns a status dict with ``status`` in {``ok``, ``blocked``}.
    """
    reports_dir = resolve_reports_dir(config, project_root=project_root)
    data_dir = resolve_data_dir(config, project_root=project_root)
    report_path = reports_dir / "temperature_residualization.md"

    if not _temperature_available(features_df):
        reason = (
            "no usable temperature_c values in the feature table "
            "(temperature.csv is empty for this dataset)"
        )
        _write_blocker_report(report_path, reason)
        return {
            "status": "blocked",
            "reason": reason,
            "report": str(report_path),
        }

    working = features_df.copy()
    temperature = pd.to_numeric(working[TEMP_COLUMN], errors="coerce").to_numpy(dtype=np.float64)
    valid = np.isfinite(temperature)
    feature_columns = _feature_columns(working)

    coefficients: dict[str, dict[str, float]] = {}
    temp_centered = temperature - np.nanmean(temperature[valid])
    denom = float(np.sum(temp_centered[valid] ** 2)) or 1.0

    for column in feature_columns:
        values = working[column].to_numpy(dtype=np.float64)
        rows = valid & np.isfinite(values)
        if rows.sum() < 2:
            coefficients[column] = {"slope": 0.0, "intercept": 0.0, "n": int(rows.sum())}
            continue
        y = values[rows]
        x = temp_centered[rows]
        slope = float(np.sum(x * (y - y.mean())) / (np.sum(x ** 2) or 1.0))
        intercept = float(y.mean() - slope * x.mean())
        prediction = intercept + slope * temp_centered
        residual = values - prediction
        # Leave rows without temperature unchanged (residual == original mean-centered).
        working.loc[rows, column] = residual[rows]
        coefficients[column] = {
            "slope": slope,
            "intercept": intercept,
            "n": int(rows.sum()),
        }

    output_path = data_dir / "processed" / "features_tabular_residualized.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_parquet_compatible(working, output_path)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    n_features = len(feature_columns)
    report_path.write_text(
        "# Temperature Residualization\n\n"
        f"Residualized {n_features} features against `temperature_c` "
        f"(temperature range "
        f"{float(np.nanmin(temperature[valid])):.2f}–{float(np.nanmax(temperature[valid])):.2f} °C, "
        f"{int(valid.sum())} segments with temperature).\n\n"
        "Output: `data/processed/features_tabular_residualized.parquet`. "
        "Point the PCA/CNN at this table to test temperature-invariant identity.\n",
        encoding="utf-8",
    )

    metrics_summary = {
        "status": "ok",
        "n_features": n_features,
        "n_segments_with_temperature": int(valid.sum()),
        "temperature_min_c": float(np.nanmin(temperature[valid])),
        "temperature_max_c": float(np.nanmax(temperature[valid])),
        "output": str(output_path),
        "report": str(report_path),
        "coefficients": coefficients,
    }
    return metrics_summary
