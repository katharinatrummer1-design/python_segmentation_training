from __future__ import annotations

from pathlib import Path

import pandas as pd

from cricket_id.io.manifest import _write_parquet_compatible


def merge_temperature(recordings_df: pd.DataFrame, temp_df: pd.DataFrame) -> pd.DataFrame:
    aggregated_temp_df = (
        temp_df.groupby("recording_id")["temperature_c"]
        .agg(["mean", "std", "min", "max"])
        .rename(
            columns={
                "mean": "temp_mean_c",
                "std": "temp_std_c",
                "min": "temp_min_c",
                "max": "temp_max_c",
            }
        )
        .reset_index()
    )
    return recordings_df.merge(aggregated_temp_df, on="recording_id", how="left")


def apply_temperature_qc(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    result_df = df.copy()
    temperature_qc = config.get("temperature_qc", {})

    if temperature_qc.get("enabled", False):
        target_temp = float(temperature_qc["target_temp_c"])
        tolerance = float(temperature_qc["temp_tolerance_c"])
        result_df["temp_qc_pass"] = (
            result_df["temp_mean_c"]
            .between(target_temp - tolerance, target_temp + tolerance, inclusive="both")
            .fillna(False)
        )
    else:
        result_df["temp_qc_pass"] = True

    return result_df


def save_recordings_qc(df: pd.DataFrame, output_path: str) -> None:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_parquet_compatible(df, destination)
