from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from cricket_id.utils.paths import resolve_project_root, resolve_reports_dir


EXCLUDED_FEATURE_COLUMNS = {
    "segment_id",
    "individual_id",
    "session_id",
    "recording_id",
    "temperature_c",
}


def _resolve_artifacts_dir(
    config: dict,
    *,
    project_root: str | Path | None = None,
) -> Path:
    root = resolve_project_root(config, project_root=project_root)
    artifacts_dir = Path(str(config.get("paths", {}).get("artifacts_dir", "artifacts")))
    return artifacts_dir if artifacts_dir.is_absolute() else (root / artifacts_dir).resolve()


def _feature_columns(features_df: pd.DataFrame) -> list[str]:
    return [
        column
        for column in features_df.columns
        if column not in EXCLUDED_FEATURE_COLUMNS
        and pd.api.types.is_numeric_dtype(features_df[column])
    ]


def _feature_family(feature_name: str) -> str:
    if feature_name.startswith("delta_mfcc_"):
        return "delta_mfcc"
    if feature_name.startswith("mfcc_"):
        return "mfcc"
    if feature_name.startswith("spectral_") or feature_name in {
        "peak_freq_hz",
        "bandwidth_hz_proxy",
    }:
        return "spectral"
    if feature_name.startswith("rms_"):
        return "energy"
    if feature_name.startswith("zcr_") or feature_name == "duration_ms":
        return "temporal"
    return "other"


def _short_feature_name(name: str) -> str:
    return name.replace("spectral_", "spec_").replace("delta_mfcc_", "d_mfcc_")


def _context_from_session(session_id: object) -> str:
    value = str(session_id).lower()
    if "__night__" in value or "night" in value:
        return "night_calling"
    if "__day__" in value or "day" in value:
        return "day_calling"
    return "unknown"


def _recording_level_frame(
    features_df: pd.DataFrame,
    feature_columns: list[str],
    recordings_df: pd.DataFrame | None,
) -> pd.DataFrame:
    metadata_columns = ["recording_id", "individual_id", "session_id"]
    missing = set(metadata_columns).difference(features_df.columns)
    if missing:
        raise ValueError(
            "features_df is missing required metadata columns: "
            + ", ".join(sorted(missing))
        )

    working_df = features_df.loc[:, metadata_columns + feature_columns].copy()
    for column in metadata_columns:
        working_df[column] = working_df[column].astype(str)

    recording_df = (
        working_df.groupby(metadata_columns, sort=True)[feature_columns]
        .median()
        .reset_index()
    )
    segment_counts = (
        working_df.groupby(metadata_columns, sort=True)
        .size()
        .reset_index(name="segment_count")
    )
    recording_df = recording_df.merge(segment_counts, on=metadata_columns, how="left")

    if isinstance(recordings_df, pd.DataFrame) and not recordings_df.empty:
        merge_columns = [
            column for column in ["recording_id", "context", "song_type"] if column in recordings_df.columns
        ]
        if "recording_id" in merge_columns:
            metadata_df = recordings_df.loc[:, merge_columns].drop_duplicates("recording_id").copy()
            metadata_df["recording_id"] = metadata_df["recording_id"].astype(str)
            recording_df = recording_df.merge(metadata_df, on="recording_id", how="left")

    if "context" not in recording_df.columns:
        recording_df["context"] = recording_df["session_id"].map(_context_from_session)
    else:
        recording_df["context"] = recording_df["context"].fillna(
            recording_df["session_id"].map(_context_from_session)
        )

    return recording_df


def _eta_squared(values: np.ndarray, labels: np.ndarray) -> float:
    finite_mask = np.isfinite(values)
    values = values[finite_mask]
    labels = labels[finite_mask]
    if values.size < 3 or len(set(labels.tolist())) < 2:
        return 0.0
    grand_mean = float(np.mean(values))
    ss_total = float(np.sum((values - grand_mean) ** 2))
    if ss_total <= 0.0:
        return 0.0
    ss_between = 0.0
    for label in sorted(set(labels.tolist())):
        group = values[labels == label]
        ss_between += float(len(group) * (float(np.mean(group)) - grand_mean) ** 2)
    return float(max(0.0, min(1.0, ss_between / ss_total)))


def _stratified_bootstrap_eta(
    values: np.ndarray,
    labels: np.ndarray,
    *,
    rng: np.random.Generator,
    iterations: int,
) -> tuple[float, float]:
    if iterations <= 0:
        eta = _eta_squared(values, labels)
        return eta, eta

    unique_labels = sorted(set(labels.tolist()))
    index_by_label = {
        label: np.flatnonzero(labels == label)
        for label in unique_labels
    }
    boot_values: list[float] = []
    for _ in range(iterations):
        sampled_indices = [
            rng.choice(indices, size=len(indices), replace=True)
            for indices in index_by_label.values()
            if len(indices) > 0
        ]
        if not sampled_indices:
            continue
        sample_index = np.concatenate(sampled_indices)
        boot_values.append(_eta_squared(values[sample_index], labels[sample_index]))
    if not boot_values:
        eta = _eta_squared(values, labels)
        return eta, eta
    return (
        float(np.percentile(boot_values, 2.5)),
        float(np.percentile(boot_values, 97.5)),
    )


def _permutation_p_value(
    values: np.ndarray,
    labels: np.ndarray,
    observed_eta: float,
    *,
    rng: np.random.Generator,
    iterations: int,
) -> float:
    if iterations <= 0:
        return float("nan")
    hits = 0
    labels_copy = labels.copy()
    for _ in range(iterations):
        rng.shuffle(labels_copy)
        permuted_eta = _eta_squared(values, labels_copy)
        hits += int(permuted_eta >= observed_eta)
    return float((hits + 1) / (iterations + 1))


def _benjamini_hochberg(p_values: list[float]) -> list[float]:
    p_array = np.asarray([1.0 if not np.isfinite(value) else value for value in p_values])
    n = len(p_array)
    if n == 0:
        return []
    order = np.argsort(p_array)
    adjusted = np.empty(n, dtype=np.float64)
    running = 1.0
    for rank_index in range(n - 1, -1, -1):
        original_index = int(order[rank_index])
        rank = rank_index + 1
        running = min(running, float(p_array[original_index] * n / rank))
        adjusted[original_index] = running
    return [float(min(1.0, value)) for value in adjusted]


def _rank_features_with_uncertainty(
    recording_df: pd.DataFrame,
    feature_columns: list[str],
    *,
    rng: np.random.Generator,
    bootstrap_iterations: int,
    permutation_iterations: int,
) -> pd.DataFrame:
    labels = recording_df["individual_id"].astype(str).to_numpy()
    rows: list[dict[str, object]] = []
    for feature_name in feature_columns:
        values = recording_df[feature_name].to_numpy(dtype=np.float64, copy=True)
        observed_eta = _eta_squared(values, labels)
        ci_low, ci_high = _stratified_bootstrap_eta(
            values,
            labels,
            rng=rng,
            iterations=bootstrap_iterations,
        )
        permutation_p = _permutation_p_value(
            values,
            labels,
            observed_eta,
            rng=rng,
            iterations=permutation_iterations,
        )
        rows.append(
            {
                "feature": feature_name,
                "family": _feature_family(feature_name),
                "eta_squared": observed_eta,
                "eta_ci_low": ci_low,
                "eta_ci_high": ci_high,
                "permutation_p": permutation_p,
            }
        )

    q_values = _benjamini_hochberg([float(row["permutation_p"]) for row in rows])
    for row, q_value in zip(rows, q_values, strict=True):
        row["fdr_q"] = q_value

    ranking_df = pd.DataFrame(rows).sort_values(
        ["eta_squared", "feature"], ascending=[False, True]
    )
    ranking_df.insert(0, "rank", np.arange(1, len(ranking_df) + 1, dtype=np.int64))
    return ranking_df


def _cliffs_delta(group_a: np.ndarray, group_b: np.ndarray) -> float:
    group_a = group_a[np.isfinite(group_a)]
    group_b = group_b[np.isfinite(group_b)]
    if group_a.size == 0 or group_b.size == 0:
        return 0.0
    comparisons = np.sign(group_a[:, None] - group_b[None, :])
    return float(np.mean(comparisons))


def _pairwise_feature_matrix(
    recording_df: pd.DataFrame,
    selected_feature: str,
) -> tuple[list[str], np.ndarray, list[dict[str, object]]]:
    individuals = sorted(recording_df["individual_id"].astype(str).unique().tolist())
    matrix = np.zeros((len(individuals), len(individuals)), dtype=np.float64)
    rows: list[dict[str, object]] = []
    for left, right in combinations(individuals, 2):
        left_values = recording_df.loc[
            recording_df["individual_id"] == left,
            selected_feature,
        ].to_numpy(dtype=np.float64)
        right_values = recording_df.loc[
            recording_df["individual_id"] == right,
            selected_feature,
        ].to_numpy(dtype=np.float64)
        delta = _cliffs_delta(left_values, right_values)
        p_value = 1.0
        try:
            if left_values.size > 0 and right_values.size > 0:
                _, p_value_raw = stats.mannwhitneyu(left_values, right_values, alternative="two-sided")
                p_value = float(p_value_raw)
        except ValueError:
            pass
        i = individuals.index(left)
        j = individuals.index(right)
        matrix[i, j] = delta
        matrix[j, i] = -delta
        rows.append(
            {
                "individual_a": left,
                "individual_b": right,
                "feature": selected_feature,
                "cliffs_delta": delta,
                "abs_cliffs_delta": abs(delta),
                "mann_whitney_p": p_value,
                "median_a": float(np.nanmedian(left_values)) if left_values.size else None,
                "median_b": float(np.nanmedian(right_values)) if right_values.size else None,
                "n_a": int(left_values.size),
                "n_b": int(right_values.size),
            }
        )
    return individuals, matrix, sorted(rows, key=lambda row: -float(row["abs_cliffs_delta"]))


def _profile_distance_matrix(
    recording_df: pd.DataFrame,
    top_features: list[str],
) -> tuple[list[str], np.ndarray, list[dict[str, object]]]:
    individuals = sorted(recording_df["individual_id"].astype(str).unique().tolist())
    profile_df = recording_df.groupby("individual_id", sort=True)[top_features].median()
    scaled = profile_df.copy()
    for feature_name in top_features:
        values = scaled[feature_name].to_numpy(dtype=np.float64)
        center = np.nanmedian(values)
        spread = np.nanstd(values)
        scaled[feature_name] = 0.0 if spread <= 0 else (values - center) / spread
    matrix = np.zeros((len(individuals), len(individuals)), dtype=np.float64)
    rows: list[dict[str, object]] = []
    for left, right in combinations(individuals, 2):
        left_vector = scaled.loc[left].to_numpy(dtype=np.float64)
        right_vector = scaled.loc[right].to_numpy(dtype=np.float64)
        distance = float(np.sqrt(np.nanmean((left_vector - right_vector) ** 2)))
        i = individuals.index(left)
        j = individuals.index(right)
        matrix[i, j] = distance
        matrix[j, i] = distance
        rows.append(
            {
                "individual_a": left,
                "individual_b": right,
                "profile_distance": distance,
            }
        )
    return individuals, matrix, sorted(rows, key=lambda row: -float(row["profile_distance"]))


def _stability_rows(recording_df: pd.DataFrame, feature_columns: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for feature_name in feature_columns:
        medians = (
            recording_df.groupby("individual_id", sort=True)[feature_name]
            .median()
            .to_numpy(dtype=np.float64)
        )
        within_stds = (
            recording_df.groupby("individual_id", sort=True)[feature_name]
            .std()
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .to_numpy(dtype=np.float64)
        )
        between = float(np.nanstd(medians)) if medians.size else 0.0
        within = float(np.nanmedian(within_stds)) if within_stds.size else 0.0
        ratio = float(between / within) if within > 0 else float("inf")
        rows.append(
            {
                "feature": feature_name,
                "family": _feature_family(feature_name),
                "between_individual_std": between,
                "within_individual_recording_std_median": within,
                "stability_ratio": ratio,
            }
        )
    return sorted(rows, key=lambda row: -float(row["stability_ratio"]))


def _context_rows(recording_df: pd.DataFrame, top_features: list[str]) -> list[dict[str, object]]:
    contexts = set(recording_df["context"].astype(str).tolist())
    if not {"night_calling", "day_calling"}.issubset(contexts):
        return []

    rows: list[dict[str, object]] = []
    for feature_name in top_features:
        night = recording_df.loc[
            recording_df["context"].astype(str) == "night_calling",
            feature_name,
        ].to_numpy(dtype=np.float64)
        day = recording_df.loc[
            recording_df["context"].astype(str) == "day_calling",
            feature_name,
        ].to_numpy(dtype=np.float64)
        night = night[np.isfinite(night)]
        day = day[np.isfinite(day)]
        if night.size == 0 or day.size == 0:
            continue
        pooled = np.concatenate([night, day])
        pooled_std = float(np.nanstd(pooled))
        standardized_shift = 0.0 if pooled_std <= 0 else float((np.nanmedian(day) - np.nanmedian(night)) / pooled_std)
        p_value = 1.0
        try:
            _, p_raw = stats.mannwhitneyu(night, day, alternative="two-sided")
            p_value = float(p_raw)
        except ValueError:
            pass
        rows.append(
            {
                "feature": feature_name,
                "family": _feature_family(feature_name),
                "night_median": float(np.nanmedian(night)),
                "day_median": float(np.nanmedian(day)),
                "standardized_day_minus_night": standardized_shift,
                "mann_whitney_p": p_value,
                "night_recordings": int(night.size),
                "day_recordings": int(day.size),
            }
        )
    return sorted(rows, key=lambda row: -abs(float(row["standardized_day_minus_night"])))


def _plot_effect_ci(ranking_df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    top_df = ranking_df.head(20).iloc[::-1]
    y = np.arange(len(top_df))
    eta = top_df["eta_squared"].to_numpy(dtype=np.float64)
    lower = eta - top_df["eta_ci_low"].to_numpy(dtype=np.float64)
    upper = top_df["eta_ci_high"].to_numpy(dtype=np.float64) - eta
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    ax.errorbar(eta, y, xerr=np.vstack([lower, upper]), fmt="o", color="#2a6fbb", ecolor="#8bb8e8")
    ax.set_yticks(y)
    ax.set_yticklabels([_short_feature_name(f) for f in top_df["feature"]])
    ax.set_xlabel("Eta squared with bootstrap 95% CI")
    ax.set_title("Feature effect strength with uncertainty")
    ax.set_xlim(0, min(1.0, max(eta.max() + 0.1, 0.2)))
    ax.grid(axis="x", alpha=0.25)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_pairwise_matrix(
    labels: list[str],
    matrix: np.ndarray,
    selected_feature: str,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    image = ax.imshow(matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title(f"Pairwise Cliff's delta: {_short_feature_name(selected_feature)}")
    fig.colorbar(image, ax=ax, label="Cliff's delta")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_distance_matrix(labels: list[str], matrix: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    image = ax.imshow(matrix, cmap="viridis")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title("Pairwise profile distance across top features")
    fig.colorbar(image, ax=ax, label="RMS robust-z distance")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_stability(stability: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    top = stability[:20][::-1]
    values = [min(float(row["stability_ratio"]), 10.0) for row in top]
    labels = [_short_feature_name(str(row["feature"])) for row in top]
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    ax.barh(labels, values, color="#4c956c")
    ax.set_xlabel("Between-individual std / within-individual recording std")
    ax.set_title("Feature stability across recordings")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_context_shift(context: list[dict[str, object]], output_path: Path) -> None:
    if not context:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    top = context[:20][::-1]
    labels = [_short_feature_name(str(row["feature"])) for row in top]
    values = [float(row["standardized_day_minus_night"]) for row in top]
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    ax.barh(labels, values, color=["#b56576" if v < 0 else "#457b9d" for v in values])
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Standardized median shift: day - night")
    ax.set_title("Day/night context shifts in top features")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_selected_boxplot(
    recording_df: pd.DataFrame,
    selected_feature: str,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    individuals = sorted(recording_df["individual_id"].astype(str).unique().tolist())
    values = [
        recording_df.loc[recording_df["individual_id"] == individual, selected_feature]
        .dropna()
        .to_numpy(dtype=np.float64)
        for individual in individuals
    ]
    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    ax.boxplot(values, labels=individuals, showfliers=False)
    ax.set_title(f"Recording-level distribution: {_short_feature_name(selected_feature)}")
    ax.tick_params(axis="x", labelrotation=75, labelsize=8)
    ax.set_ylabel(selected_feature)
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _markdown_table(headers: list[str], rows: list[list[object]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return lines


def _relative_path(from_dir: Path, to_path: Path) -> str:
    return os.path.relpath(to_path, start=from_dir).replace("\\", "/")


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def _write_report(payload: dict, report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    top_feature_rows = [
        [
            row["rank"],
            row["feature"],
            row["family"],
            f"{float(row['eta_squared']):.3f}",
            f"{float(row['eta_ci_low']):.3f}-{float(row['eta_ci_high']):.3f}",
            f"{float(row['fdr_q']):.3f}",
        ]
        for row in payload["feature_rankings"][:15]
    ]
    pairwise_rows = [
        [
            row["individual_a"],
            row["individual_b"],
            f"{float(row['abs_cliffs_delta']):.3f}",
            f"{float(row['median_a']):.3f}" if row["median_a"] is not None else "n/a",
            f"{float(row['median_b']):.3f}" if row["median_b"] is not None else "n/a",
        ]
        for row in payload["pairwise_selected_feature"][:12]
    ]
    context_rows = [
        [
            row["feature"],
            row["family"],
            f"{float(row['standardized_day_minus_night']):.3f}",
            row["day_recordings"],
            row["night_recordings"],
        ]
        for row in payload["context_shift"][:12]
    ]

    lines = [
        "# Statistical Feature Comparison",
        "",
        f"Generated: {payload['generated_at']}",
        "",
        "This dashboard is designed for cautious comparisons, not absolute biological claims by itself.",
        "It uses recording-level medians and reports uncertainty so individuals with many chirps do not dominate the evidence.",
        "",
        "## Scope",
        "",
    ]
    lines.extend(
        _markdown_table(
            ["Metric", "Value"],
            [
                ["Analysis level", payload["analysis_level"]],
                ["Segments", payload["input_counts"]["segments"]],
                ["Recordings", payload["input_counts"]["recordings"]],
                ["Individuals", payload["input_counts"]["individuals"]],
                ["Features", payload["input_counts"]["feature_columns"]],
                ["Bootstrap iterations", payload["settings"]["bootstrap_iterations"]],
                ["Permutation iterations", payload["settings"]["permutation_iterations"]],
                ["Selected pairwise feature", payload["selected_feature"]],
            ],
        )
    )
    lines.extend(["", "## Feature Effects With Uncertainty", ""])
    lines.extend(
        _markdown_table(
            ["Rank", "Feature", "Family", "Eta^2", "95% CI", "FDR q"],
            top_feature_rows,
        )
    )
    lines.extend(["", "## Strongest Pairwise Separations", ""])
    lines.extend(
        _markdown_table(
            ["Individual A", "Individual B", "|Cliff's delta|", "Median A", "Median B"],
            pairwise_rows,
        )
    )
    if context_rows:
        lines.extend(["", "## Day/Night Context Check", ""])
        lines.extend(
            _markdown_table(
                ["Feature", "Family", "Std day-night shift", "Day recordings", "Night recordings"],
                context_rows,
            )
        )
    lines.extend(["", "## Figures", ""])
    for label, path_value in payload["artifact_paths"]["figures"].items():
        path = Path(path_value)
        if path.exists():
            lines.extend(["", f"![{label}]({_relative_path(report_path.parent, path)})", ""])
    lines.extend(
        [
            "",
            "## What Would Support Stronger Claims",
            "",
            "- Effects remain large after bootstrap uncertainty and permutation testing.",
            "- The same feature is stable within individuals across recordings/sessions.",
            "- Pairwise separations are not driven only by one context such as day or night recordings.",
            "- The pattern holds on future recordings not used to choose the feature.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def run_statistical_feature_comparison(
    features_df: pd.DataFrame,
    config: dict,
    *,
    recordings_df: pd.DataFrame | None = None,
    project_root: str | Path | None = None,
) -> dict:
    project_root_path = resolve_project_root(config, project_root=project_root)
    artifacts_dir = _resolve_artifacts_dir(config, project_root=project_root_path)
    reports_dir = resolve_reports_dir(config, project_root=project_root_path)
    metrics_dir = artifacts_dir / "metrics"
    figures_dir = artifacts_dir / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    stats_config = config.get("feature_comparison", {})
    seed = int(config.get("seed", 0))
    rng = np.random.default_rng(seed)
    bootstrap_iterations = int(stats_config.get("bootstrap_iterations", 200))
    permutation_iterations = int(stats_config.get("permutation_iterations", 200))
    top_feature_count = int(stats_config.get("top_feature_count", 12))

    feature_columns = _feature_columns(features_df)
    if not feature_columns:
        raise ValueError("No numeric feature columns found for statistical comparison.")

    recording_df = _recording_level_frame(features_df, feature_columns, recordings_df)
    ranking_df = _rank_features_with_uncertainty(
        recording_df,
        feature_columns,
        rng=rng,
        bootstrap_iterations=bootstrap_iterations,
        permutation_iterations=permutation_iterations,
    )
    selected_feature = str(stats_config.get("selected_feature") or ranking_df.iloc[0]["feature"])
    top_features = ranking_df["feature"].head(top_feature_count).astype(str).tolist()

    pairwise_labels, pairwise_matrix, pairwise_rows = _pairwise_feature_matrix(
        recording_df,
        selected_feature,
    )
    distance_labels, distance_matrix, distance_rows = _profile_distance_matrix(
        recording_df,
        top_features,
    )
    stability = _stability_rows(recording_df, feature_columns)
    context_shift = _context_rows(recording_df, top_features)

    figure_paths = {
        "Feature effects with CI": figures_dir / "comparison_feature_effect_ci.png",
        "Selected feature pairwise matrix": figures_dir / "comparison_pairwise_feature_matrix.png",
        "Top-feature profile distance": figures_dir / "comparison_profile_distance_matrix.png",
        "Feature stability": figures_dir / "comparison_feature_stability.png",
        "Day-night context shift": figures_dir / "comparison_context_shift.png",
        "Selected feature boxplot": figures_dir / "comparison_selected_feature_boxplot.png",
    }
    _plot_effect_ci(ranking_df, figure_paths["Feature effects with CI"])
    _plot_pairwise_matrix(
        pairwise_labels,
        pairwise_matrix,
        selected_feature,
        figure_paths["Selected feature pairwise matrix"],
    )
    _plot_distance_matrix(
        distance_labels,
        distance_matrix,
        figure_paths["Top-feature profile distance"],
    )
    _plot_stability(stability, figure_paths["Feature stability"])
    _plot_context_shift(context_shift, figure_paths["Day-night context shift"])
    _plot_selected_boxplot(
        recording_df,
        selected_feature,
        figure_paths["Selected feature boxplot"],
    )

    payload = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "analysis_level": "recording_median",
        "settings": {
            "seed": seed,
            "bootstrap_iterations": bootstrap_iterations,
            "permutation_iterations": permutation_iterations,
            "top_feature_count": top_feature_count,
        },
        "input_counts": {
            "segments": int(len(features_df)),
            "recordings": int(recording_df["recording_id"].nunique()),
            "individuals": int(recording_df["individual_id"].nunique()),
            "feature_columns": int(len(feature_columns)),
        },
        "selected_feature": selected_feature,
        "feature_rankings": ranking_df.head(30).replace({np.nan: None}).to_dict(orient="records"),
        "pairwise_selected_feature": pairwise_rows,
        "profile_distances": distance_rows,
        "stability": stability[:30],
        "context_shift": context_shift,
        "artifact_paths": {
            "metrics": str(metrics_dir / "feature_comparison.json"),
            "report": str(reports_dir / "feature_comparison.md"),
            "figures": {
                label: str(path)
                for label, path in figure_paths.items()
                if path.exists()
            },
        },
    }
    payload = _json_safe(payload)
    metrics_path = metrics_dir / "feature_comparison.json"
    metrics_path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_report(payload, reports_dir / "feature_comparison.md")
    return payload
