from __future__ import annotations

import json
import os
import pickle
from datetime import datetime, timezone
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


def _recording_level_frame(features_df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
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
    return recording_df.merge(segment_counts, on=metadata_columns, how="left")


def _eta_squared(values: np.ndarray, labels: np.ndarray) -> tuple[float, float | None, float | None]:
    finite_mask = np.isfinite(values)
    values = values[finite_mask]
    labels = labels[finite_mask]
    if values.size < 3 or len(set(labels.tolist())) < 2:
        return 0.0, None, None

    grand_mean = float(np.mean(values))
    ss_total = float(np.sum((values - grand_mean) ** 2))
    if ss_total <= 0.0:
        return 0.0, None, None

    groups = [values[labels == label] for label in sorted(set(labels.tolist()))]
    ss_between = float(
        sum(len(group) * (float(np.mean(group)) - grand_mean) ** 2 for group in groups)
    )
    eta = ss_between / ss_total

    usable_groups = [group for group in groups if len(group) >= 2 and np.std(group) > 0]
    f_statistic: float | None = None
    p_value: float | None = None
    if len(usable_groups) >= 2:
        try:
            f_raw, p_raw = stats.f_oneway(*usable_groups)
            if np.isfinite(f_raw):
                f_statistic = float(f_raw)
            if np.isfinite(p_raw):
                p_value = float(p_raw)
        except ValueError:
            pass

    return float(eta), f_statistic, p_value


def _rank_features(recording_df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    labels = recording_df["individual_id"].astype(str).to_numpy()
    rows: list[dict[str, object]] = []
    for feature_name in feature_columns:
        values = recording_df[feature_name].to_numpy(dtype=np.float64, copy=True)
        eta, f_statistic, p_value = _eta_squared(values, labels)
        per_individual = [
            group_df[feature_name].to_numpy(dtype=np.float64, copy=True)
            for _, group_df in recording_df.groupby("individual_id", sort=True)
        ]
        individual_medians = [
            float(np.nanmedian(group_values))
            for group_values in per_individual
            if group_values.size > 0
        ]
        within_stds = [
            float(np.nanstd(group_values))
            for group_values in per_individual
            if group_values.size > 1
        ]
        between_std = float(np.nanstd(individual_medians)) if individual_medians else 0.0
        within_std = float(np.nanmedian(within_stds)) if within_stds else 0.0
        rows.append(
            {
                "feature": feature_name,
                "family": _feature_family(feature_name),
                "eta_squared": eta,
                "f_statistic": f_statistic,
                "p_value": p_value,
                "between_individual_median_std": between_std,
                "within_individual_recording_std_median": within_std,
                "recording_count": int(recording_df[feature_name].notna().sum()),
            }
        )

    ranking_df = pd.DataFrame(rows).sort_values(
        ["eta_squared", "feature"], ascending=[False, True]
    )
    ranking_df.insert(0, "rank", np.arange(1, len(ranking_df) + 1, dtype=np.int64))
    return ranking_df


def _family_summary(ranking_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for family, group_df in ranking_df.groupby("family", sort=True):
        top_row = group_df.sort_values("eta_squared", ascending=False).iloc[0]
        rows.append(
            {
                "family": str(family),
                "feature_count": int(len(group_df)),
                "median_eta_squared": float(group_df["eta_squared"].median()),
                "max_eta_squared": float(group_df["eta_squared"].max()),
                "top_feature": str(top_row["feature"]),
            }
        )
    return pd.DataFrame(rows).sort_values("max_eta_squared", ascending=False)


def _short_feature_name(name: str) -> str:
    return name.replace("spectral_", "spec_").replace("delta_mfcc_", "d_mfcc_")


def _plot_effect_size_ranking(ranking_df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    top_df = ranking_df.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    colors = [plt.get_cmap("tab10")(hash(family) % 10) for family in top_df["family"]]
    ax.barh([_short_feature_name(f) for f in top_df["feature"]], top_df["eta_squared"], color=colors)
    ax.set_xlabel("Eta squared on recording-level medians")
    ax.set_title("Top individual-separating features")
    ax.set_xlim(0.0, max(1.0, float(top_df["eta_squared"].max()) * 1.05))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_family_summary(summary_df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_df = summary_df.sort_values("max_eta_squared", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    ax.barh(plot_df["family"], plot_df["max_eta_squared"], alpha=0.85, label="max")
    ax.scatter(plot_df["median_eta_squared"], plot_df["family"], color="black", label="median", zorder=3)
    ax.set_xlabel("Eta squared")
    ax.set_title("Feature family signal strength")
    ax.legend(frameon=False)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_top_feature_boxplots(
    recording_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    top_features = ranking_df["feature"].head(6).tolist()
    individuals = sorted(recording_df["individual_id"].astype(str).unique().tolist())
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), constrained_layout=True)
    for ax, feature_name in zip(axes.ravel(), top_features, strict=False):
        values = [
            recording_df.loc[recording_df["individual_id"] == individual, feature_name]
            .dropna()
            .to_numpy(dtype=np.float64)
            for individual in individuals
        ]
        ax.boxplot(values, labels=individuals, showfliers=False)
        ax.set_title(_short_feature_name(feature_name))
        ax.tick_params(axis="x", labelrotation=80, labelsize=7)
        ax.grid(axis="y", alpha=0.2)
    for ax in axes.ravel()[len(top_features) :]:
        ax.set_axis_off()
    fig.suptitle("Top feature distributions by individual (recording medians)", fontsize=14)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_individual_profile_heatmap(
    recording_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    top_features = ranking_df["feature"].head(20).tolist()
    profile_df = (
        recording_df.groupby("individual_id", sort=True)[top_features]
        .median()
        .replace([np.inf, -np.inf], np.nan)
    )
    scaled = profile_df.copy()
    for feature_name in top_features:
        values = scaled[feature_name].to_numpy(dtype=np.float64)
        center = np.nanmedian(values)
        spread = np.nanstd(values)
        scaled[feature_name] = 0.0 if spread <= 0 else (values - center) / spread

    fig, ax = plt.subplots(figsize=(12, 6.5), constrained_layout=True)
    image = ax.imshow(scaled.to_numpy(dtype=np.float64), aspect="auto", cmap="coolwarm", vmin=-2.5, vmax=2.5)
    ax.set_yticks(np.arange(len(scaled.index)))
    ax.set_yticklabels(scaled.index, fontsize=8)
    ax.set_xticks(np.arange(len(top_features)))
    ax.set_xticklabels([_short_feature_name(f) for f in top_features], rotation=70, ha="right", fontsize=8)
    ax.set_title("Individual profiles across top features")
    fig.colorbar(image, ax=ax, label="Robust z-score across individuals")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _load_pickle(path: Path) -> object | None:
    if not path.exists():
        return None
    try:
        with path.open("rb") as handle:
            return pickle.load(handle)
    except (pickle.UnpicklingError, EOFError, ValueError):
        return None


def _pca_loading_rows(
    feature_columns: list[str],
    pca_metrics: dict | None,
    pca_model: object | None,
) -> list[dict[str, object]]:
    if pca_model is None or not hasattr(pca_model, "components_"):
        return []
    metric_features = (
        pca_metrics.get("feature_columns", [])
        if isinstance(pca_metrics, dict)
        else []
    )
    ordered_features = metric_features if metric_features else feature_columns
    components = np.asarray(getattr(pca_model, "components_"), dtype=np.float64)
    if components.ndim != 2 or components.shape[1] != len(ordered_features):
        return []

    rows: list[dict[str, object]] = []
    for pc_index in range(min(3, components.shape[0])):
        component = components[pc_index]
        top_indices = np.argsort(np.abs(component))[::-1][:12]
        for feature_index in top_indices:
            feature_name = str(ordered_features[feature_index])
            rows.append(
                {
                    "pc": f"PC{pc_index + 1}",
                    "feature": feature_name,
                    "family": _feature_family(feature_name),
                    "loading": float(component[feature_index]),
                    "abs_loading": float(abs(component[feature_index])),
                }
            )
    return rows


def _plot_pca_loadings(loadings: list[dict[str, object]], output_path: Path) -> None:
    if not loadings:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    loadings_df = pd.DataFrame(loadings)
    pcs = ["PC1", "PC2", "PC3"]
    feature_order = (
        loadings_df.groupby("feature")["abs_loading"]
        .max()
        .sort_values(ascending=False)
        .head(24)
        .index
        .tolist()
    )
    matrix = np.zeros((len(feature_order), len(pcs)), dtype=np.float64)
    for row_index, feature_name in enumerate(feature_order):
        for col_index, pc_name in enumerate(pcs):
            match_df = loadings_df.loc[
                (loadings_df["feature"] == feature_name) & (loadings_df["pc"] == pc_name)
            ]
            if not match_df.empty:
                matrix[row_index, col_index] = float(match_df["loading"].iloc[0])

    fig, ax = plt.subplots(figsize=(7, 9), constrained_layout=True)
    vmax = max(0.05, float(np.max(np.abs(matrix))))
    image = ax.imshow(matrix, cmap="coolwarm", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(pcs)))
    ax.set_xticklabels(pcs)
    ax.set_yticks(np.arange(len(feature_order)))
    ax.set_yticklabels([_short_feature_name(f) for f in feature_order], fontsize=8)
    ax.set_title("Largest PCA feature loadings")
    fig.colorbar(image, ax=ax, label="Loading")
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


def _write_report(
    payload: dict,
    ranking_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    figure_paths: dict[str, Path],
    report_path: Path,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    top_rows = [
        [
            int(row.rank),
            str(row.feature),
            str(row.family),
            f"{float(row.eta_squared):.3f}",
            "n/a" if pd.isna(row.p_value) else f"{float(row.p_value):.2e}",
        ]
        for row in ranking_df.head(15).itertuples(index=False)
    ]
    family_rows = [
        [
            str(row.family),
            int(row.feature_count),
            f"{float(row.max_eta_squared):.3f}",
            f"{float(row.median_eta_squared):.3f}",
            str(row.top_feature),
        ]
        for row in summary_df.itertuples(index=False)
    ]

    lines = [
        "# Feature Analysis",
        "",
        f"Generated: {payload['generated_at']}",
        "",
        "This analysis aggregates chirp-level features to recording-level medians before comparing individuals.",
        "That keeps recordings, rather than individual chirps, as the main evidence unit and reduces domination by individuals with many segments.",
        "",
        "## Dataset",
        "",
    ]
    lines.extend(
        _markdown_table(
            ["Metric", "Value"],
            [
                ["Segments", payload["input_counts"]["segments"]],
                ["Recordings", payload["input_counts"]["recordings"]],
                ["Individuals", payload["input_counts"]["individuals"]],
                ["Feature columns", payload["input_counts"]["feature_columns"]],
                ["Analysis level", payload["analysis_level"]],
            ],
        )
    )
    lines.extend(["", "## Top Features", ""])
    lines.extend(_markdown_table(["Rank", "Feature", "Family", "Eta squared", "p-value"], top_rows))
    lines.extend(["", "## Feature Families", ""])
    lines.extend(
        _markdown_table(
            ["Family", "Features", "Max eta squared", "Median eta squared", "Top feature"],
            family_rows,
        )
    )
    lines.extend(["", "## Figures", ""])
    for label, path in figure_paths.items():
        if path.exists():
            lines.extend(["", f"![{label}]({_relative_path(report_path.parent, path)})", ""])
    lines.extend(
        [
            "",
            "## How To Read This",
            "",
            "- Eta squared close to 1 means the feature varies strongly between individuals relative to total recording-level variation.",
            "- Boxplots show recording-level medians, not every chirp.",
            "- PCA loadings indicate which features drive the already-trained PCA axes; they are not classifier importances by themselves.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def run_feature_analysis(
    features_df: pd.DataFrame,
    config: dict,
    project_root: str | Path | None = None,
) -> dict:
    project_root_path = resolve_project_root(config, project_root=project_root)
    artifacts_dir = _resolve_artifacts_dir(config, project_root=project_root_path)
    reports_dir = resolve_reports_dir(config, project_root=project_root_path)
    metrics_dir = artifacts_dir / "metrics"
    figures_dir = artifacts_dir / "figures"

    feature_columns = _feature_columns(features_df)
    if not feature_columns:
        raise ValueError("No numeric feature columns found for feature analysis.")

    recording_df = _recording_level_frame(features_df, feature_columns)
    ranking_df = _rank_features(recording_df, feature_columns)
    summary_df = _family_summary(ranking_df)

    pca_metrics_path = metrics_dir / "pca_metrics.json"
    pca_metrics = (
        json.loads(pca_metrics_path.read_text(encoding="utf-8"))
        if pca_metrics_path.exists()
        else None
    )
    pca_model = _load_pickle(artifacts_dir / "models" / "pca.pkl")
    pca_loadings = _pca_loading_rows(feature_columns, pca_metrics, pca_model)

    figure_paths = {
        "Feature effect-size ranking": figures_dir / "feature_effect_size_ranking.png",
        "Feature family summary": figures_dir / "feature_family_summary.png",
        "Top feature boxplots": figures_dir / "feature_top_boxplots.png",
        "Individual profile heatmap": figures_dir / "feature_individual_profile_heatmap.png",
        "PCA loadings": figures_dir / "feature_pca_loadings.png",
    }
    _plot_effect_size_ranking(ranking_df, figure_paths["Feature effect-size ranking"])
    _plot_family_summary(summary_df, figure_paths["Feature family summary"])
    _plot_top_feature_boxplots(recording_df, ranking_df, figure_paths["Top feature boxplots"])
    _plot_individual_profile_heatmap(recording_df, ranking_df, figure_paths["Individual profile heatmap"])
    _plot_pca_loadings(pca_loadings, figure_paths["PCA loadings"])

    payload = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "analysis_level": "recording_median",
        "input_counts": {
            "segments": int(len(features_df)),
            "recordings": int(recording_df["recording_id"].nunique()),
            "individuals": int(recording_df["individual_id"].nunique()),
            "feature_columns": int(len(feature_columns)),
        },
        "top_features": ranking_df.head(30).replace({np.nan: None}).to_dict(orient="records"),
        "feature_family_summary": summary_df.replace({np.nan: None}).to_dict(orient="records"),
        "pca_top_loadings": pca_loadings,
        "artifact_paths": {
            "metrics": str(metrics_dir / "feature_analysis.json"),
            "report": str(reports_dir / "feature_analysis.md"),
            "figures": {label: str(path) for label, path in figure_paths.items() if path.exists()},
        },
    }

    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / "feature_analysis.json"
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _write_report(
        payload,
        ranking_df,
        summary_df,
        figure_paths,
        reports_dir / "feature_analysis.md",
    )
    return payload
