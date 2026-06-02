from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_RECORDING_COLUMNS = [
    "recording_id",
    "individual_id",
    "session_id",
    "song_type",
    "context",
    "audio_path",
    "temperature_log_path",
    "recording_start_utc",
    "duration_s",
    "sr",
    "bit_depth",
    "notes",
]


def _read_parquet_compatible(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except (ImportError, ModuleNotFoundError, OSError, ValueError):
        return pd.read_pickle(path)


def _write_parquet_compatible(df: pd.DataFrame, path: Path) -> None:
    try:
        df.to_parquet(path, index=False)
    except (ImportError, ModuleNotFoundError, OSError, ValueError):
        df.to_pickle(path)


def _load_tabular_manifest(path: str) -> pd.DataFrame:
    manifest_path = Path(path)
    suffix = manifest_path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(manifest_path)
    if suffix == ".parquet":
        return _read_parquet_compatible(manifest_path)

    raise ValueError(f"Unsupported manifest format: {manifest_path}")


def _resolve_audio_path(audio_path: object, raw_dir: Path) -> Path | None:
    if pd.isna(audio_path):
        return None

    candidate = Path(str(audio_path))
    if not str(candidate).strip():
        return None

    return candidate if candidate.is_absolute() else raw_dir / candidate


def load_recordings_manifest(path: str) -> pd.DataFrame:
    return _load_tabular_manifest(path)


def load_temperature_manifest(path: str) -> pd.DataFrame:
    return _load_tabular_manifest(path)


def validate_manifest(
    recordings_df: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, list[str]]:
    issues: list[str] = []
    missing_columns = [
        column for column in REQUIRED_RECORDING_COLUMNS if column not in recordings_df.columns
    ]

    if missing_columns:
        issues.append(
            "Missing required columns: " + ", ".join(sorted(missing_columns))
        )
        return pd.DataFrame(), issues

    scope = config.get("mvp_scope", {})

    song_type_scope = scope.get("song_type", "calling")
    if isinstance(song_type_scope, (list, tuple, set)):
        song_type_mask = recordings_df["song_type"].isin(list(song_type_scope))
    else:
        song_type_mask = recordings_df["song_type"] == song_type_scope

    context_scope = scope.get("context", "lab")
    if isinstance(context_scope, (list, tuple, set)):
        context_mask = recordings_df["context"].isin(list(context_scope))
    else:
        context_mask = recordings_df["context"] == context_scope

    scoped_df = recordings_df.loc[song_type_mask & context_mask].copy()

    individual_missing = scoped_df["individual_id"].isna() | (
        scoped_df["individual_id"].astype(str).str.strip() == ""
    )
    if individual_missing.any():
        problem_ids = scoped_df.loc[individual_missing, "recording_id"].astype(str).tolist()
        issues.append(
            "Rows with missing individual_id: " + ", ".join(problem_ids)
        )

    session_missing = scoped_df["session_id"].isna() | (
        scoped_df["session_id"].astype(str).str.strip() == ""
    )
    if session_missing.any():
        problem_ids = scoped_df.loc[session_missing, "recording_id"].astype(str).tolist()
        issues.append("Rows with missing session_id: " + ", ".join(problem_ids))

    raw_dir = Path(config["paths"]["data_dir"]) / "raw"
    missing_audio_ids: list[str] = []
    for row in scoped_df.itertuples(index=False):
        resolved_path = _resolve_audio_path(getattr(row, "audio_path"), raw_dir)
        if resolved_path is None or not resolved_path.exists():
            missing_audio_ids.append(str(getattr(row, "recording_id")))

    if missing_audio_ids:
        issues.append(
            "Audio files not found for recording_id values: "
            + ", ".join(missing_audio_ids)
        )

    return scoped_df, issues


def check_session_sufficiency(
    df: pd.DataFrame,
    min_sessions: int,
) -> tuple[bool, pd.DataFrame]:
    summary_df = (
        df.groupby("individual_id", dropna=False)["session_id"]
        .nunique()
        .reset_index(name="session_count")
        .sort_values("individual_id")
        .reset_index(drop=True)
    )
    summary_df.attrs["min_sessions"] = min_sessions

    all_have_enough = bool(
        not summary_df.empty and summary_df["session_count"].ge(min_sessions).all()
    )
    return all_have_enough, summary_df


def generate_data_gap_report(summary_df: pd.DataFrame, output_path: str) -> None:
    report_path = Path(output_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    min_sessions = int(summary_df.attrs.get("min_sessions", 0))
    if min_sessions > 0:
        insufficient_df = summary_df.loc[summary_df["session_count"] < min_sessions]
    else:
        insufficient_df = summary_df.copy()

    lines = [
        "# Data Gap Report",
        "",
        f"Minimum sessions required per individual: {min_sessions}",
        "",
    ]

    if insufficient_df.empty:
        lines.append("All individuals meet the minimum session requirement.")
    else:
        lines.extend(
            [
                "| individual_id | session_count |",
                "| --- | ---: |",
            ]
        )
        for row in insufficient_df.itertuples(index=False):
            lines.append(f"| {row.individual_id} | {row.session_count} |")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
