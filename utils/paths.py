from __future__ import annotations

from pathlib import Path

import pandas as pd


def resolve_project_root(
    config: dict,
    *,
    project_root: str | Path | None = None,
    path_hint: str | Path | None = None,
) -> Path:
    if project_root is not None:
        return Path(project_root).resolve()

    data_dir_value = config.get("paths", {}).get("data_dir", "data")
    data_dir = Path(str(data_dir_value))
    if data_dir.is_absolute():
        return data_dir.parent.resolve()

    if path_hint is not None:
        hint_path = Path(path_hint)
        if hint_path.is_absolute() and len(hint_path.parents) >= 3:
            return hint_path.parent.parent.parent.resolve()

    return Path.cwd().resolve()


def resolve_data_dir(
    config: dict,
    *,
    project_root: str | Path | None = None,
    path_hint: str | Path | None = None,
) -> Path:
    root = resolve_project_root(config, project_root=project_root, path_hint=path_hint)
    data_dir = Path(str(config.get("paths", {}).get("data_dir", "data")))
    return data_dir.resolve() if data_dir.is_absolute() else (root / data_dir).resolve()


def resolve_reports_dir(
    config: dict,
    *,
    project_root: str | Path | None = None,
    path_hint: str | Path | None = None,
) -> Path:
    root = resolve_project_root(config, project_root=project_root, path_hint=path_hint)
    reports_dir = Path(str(config.get("paths", {}).get("reports_dir", "reports")))
    return (
        reports_dir.resolve()
        if reports_dir.is_absolute()
        else (root / reports_dir).resolve()
    )


def resolve_audio_path(
    audio_path: object,
    config: dict,
    *,
    project_root: str | Path | None = None,
    path_hint: str | Path | None = None,
) -> Path:
    candidate = Path(str(audio_path))
    if candidate.is_absolute():
        return candidate.resolve()

    data_dir = resolve_data_dir(config, project_root=project_root, path_hint=path_hint)
    return (data_dir / "raw" / candidate).resolve()


def first_existing_path_hint(df: pd.DataFrame, column_name: str) -> Path | None:
    if column_name not in df.columns:
        return None

    for value in df[column_name]:
        if pd.isna(value):
            continue
        candidate = Path(str(value))
        if candidate.is_absolute():
            return candidate

    return None
