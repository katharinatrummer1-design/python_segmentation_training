from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from cricket_id.io.manifest import _read_parquet_compatible
from cricket_id.utils.paths import (
    resolve_data_dir,
    resolve_project_root,
    resolve_reports_dir,
)


def _resolve_artifacts_dir(
    config: dict,
    *,
    project_root: str | Path | None = None,
) -> Path:
    root = resolve_project_root(config, project_root=project_root)
    artifacts_dir = Path(str(config.get("paths", {}).get("artifacts_dir", "artifacts")))
    return artifacts_dir if artifacts_dir.is_absolute() else (root / artifacts_dir).resolve()


def _runtime_cache(config: dict) -> dict[str, object]:
    runtime_cache = config.get("__runtime_cache__", {})
    return runtime_cache if isinstance(runtime_cache, dict) else {}


def _load_frame(config: dict, cache_key: str, path: Path) -> pd.DataFrame:
    runtime_value = _runtime_cache(config).get(cache_key)
    if isinstance(runtime_value, pd.DataFrame):
        return runtime_value.copy()
    if path.exists():
        return _read_parquet_compatible(path)
    return pd.DataFrame()


def _format_metric(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


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


def _metric_table_rows(model_metrics: dict) -> list[list[str]]:
    rows: list[list[str]] = []
    for split_name in ("val", "test"):
        split_metrics = model_metrics.get("metrics", {}).get(split_name, {})
        rows.append(
            [
                split_name,
                _format_metric(split_metrics.get("accuracy")),
                _format_metric(split_metrics.get("balanced_accuracy")),
                _format_metric(split_metrics.get("macro_f1")),
                _format_metric(split_metrics.get("session_level_accuracy")),
                str(split_metrics.get("sample_count", "n/a")),
                str(split_metrics.get("session_count", "n/a")),
            ]
        )
    return rows


def _comparison_rows(evaluation_results: dict) -> list[list[str]]:
    rows: list[list[str]] = []
    comparison = evaluation_results.get("comparison", {})
    for split_name in ("val", "test"):
        split_comparison = comparison.get(split_name, {})
        for metric_name in (
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "session_level_accuracy",
        ):
            metric_values = split_comparison.get(metric_name, {})
            rows.append(
                [
                    split_name,
                    metric_name,
                    _format_metric(metric_values.get("pca")),
                    _format_metric(metric_values.get("cnn")),
                    _format_metric(metric_values.get("delta_cnn_minus_pca")),
                    str(metric_values.get("winner", "n/a")),
                ]
            )
    return rows


def _feature_columns(features_df: pd.DataFrame) -> list[str]:
    excluded_columns = {
        "segment_id",
        "individual_id",
        "session_id",
        "recording_id",
        "temperature_c",
    }
    return [column for column in features_df.columns if column not in excluded_columns]


def _figure_block(report_dir: Path, figure_paths: list[tuple[str, Path]]) -> list[str]:
    lines: list[str] = []
    for label, path in figure_paths:
        if path.exists():
            lines.extend(["", f"![{label}]({_relative_path(report_dir, path)})", ""])
    return lines


def _error_analysis_lines(model_name: str, model_analysis: dict) -> list[str]:
    lines = [f"### {model_name.upper()}", ""]
    for split_name in ("val", "test"):
        split_analysis = model_analysis.get(split_name, {})
        lines.extend([f"#### {split_name}", ""])

        top_confusions = split_analysis.get("top_confusions", [])
        if top_confusions:
            lines.append("Most common confusions:")
            for confusion in top_confusions:
                lines.append(
                    "- "
                    f"{confusion['true_individual_id']} -> {confusion['predicted_individual_id']}: "
                    f"{confusion['count']} segments"
                )
        else:
            lines.append("Most common confusions: none observed.")
        lines.append("")

        session_errors = split_analysis.get("session_specific_errors", [])
        if session_errors:
            lines.append("Worst sessions:")
            for session_error in session_errors:
                lines.append(
                    "- "
                    f"{session_error['session_id']} ({session_error['individual_id']}): "
                    f"segment_accuracy={session_error['segment_accuracy']:.3f}, "
                    f"segments={session_error['segment_count']}, "
                    f"mean_snr={_format_metric(session_error['mean_snr_proxy'])}, "
                    f"mean_temp={_format_metric(session_error['mean_temperature_c'])}"
                )
        else:
            lines.append("Worst sessions: none observed.")
        lines.append("")

        for band_name, label in (
            ("temperature_effect", "Temperature bins"),
            ("snr_effect", "SNR bins"),
        ):
            band_summary = split_analysis.get(band_name, {})
            if band_summary.get("status") != "ok":
                lines.append(f"{label}: {band_summary.get('status', 'n/a')}.")
                continue
            lines.append(f"{label}:")
            for band in band_summary.get("bins", []):
                lines.append(
                    "- "
                    f"{band['value_min']:.3f} to {band['value_max']:.3f}: "
                    f"accuracy={band['accuracy']:.3f} over {band['sample_count']} samples"
                )
        lines.append("")
    return lines


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None


def _advanced_analyses_lines(metrics_dir: Path) -> tuple[list[str], bool]:
    """Render cross-context, embedding-verification, and metric-learning results.

    Returns (lines, cross_context_present) so the Limitations section can adapt.
    """
    lines: list[str] = []
    cross = _load_json(metrics_dir / "cross_context.json")
    ce_emb = _load_json(metrics_dir / "cnn_embeddings.json")
    triplet_emb = _load_json(metrics_dir / "cnn_embeddings_triplet.json")
    triplet = _load_json(metrics_dir / "cnn_triplet_metrics.json")
    cv = _load_json(metrics_dir / "pca_cross_validation.json")

    if not any([cross, ce_emb, triplet_emb, triplet, cv]):
        return lines, False

    lines.extend(["## Advanced Analyses", ""])

    if cv and cv.get("pooled"):
        lines.extend(["### Honest estimate: session-grouped cross-validation (PCA)", ""])
        lines.append(
            "Leakage-free `StratifiedGroupKFold` by session (no session shared between train and "
            "test), class-balanced PCA refit per fold. This is the \"recognise on a new day\" number; "
            "compare it to the optimistic recording-holdout test metrics above."
        )
        lines.append("")
        boot = cv.get("pooled_bootstrap", {})
        lines.append("| Metric | Pooled held-out | 95% CI (session bootstrap) |")
        lines.append("| --- | --- | --- |")
        for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
            ci = boot.get(metric, {}) if isinstance(boot, dict) else {}
            ci_str = (
                f"[{_format_metric(ci.get('ci_low'))}, {_format_metric(ci.get('ci_high'))}]"
                if isinstance(ci, dict) and "ci_low" in ci else "n/a"
            )
            lines.append(f"| {metric} | {_format_metric(cv['pooled'].get(metric))} | {ci_str} |")
        lines.append("")
        lines.append(
            f"Evaluated on {cv.get('n_individuals_evaluated')}/{cv.get('n_individuals_total')} "
            "individuals (single-session animals cannot be tested without leakage)."
        )
        lines.append("")

    if cross:
        lines.extend(["### Identity vs. Recording Context (cross-context validation)", ""])
        lines.append(
            "Train on one diel context and test on the other. A large within-vs-cross gap "
            "means apparent identity is partly confounded with the recording context "
            f"(`{cross.get('context_column')}`)."
        )
        lines.append("")
        rows = []
        for direction in cross.get("directions", []):
            rows.append(
                [
                    direction.get("label", "n/a"),
                    _format_metric(direction.get("within_context", {}).get("macro_f1")),
                    _format_metric(direction.get("cross_context", {}).get("macro_f1")),
                    _format_metric(direction.get("penalty", {}).get("macro_f1")),
                ]
            )
        lines.extend(
            _markdown_table(
                ["Direction", "Within-context Macro-F1", "Cross-context Macro-F1", "Penalty"],
                rows,
            )
        )
        lines.append("")

    if ce_emb or triplet_emb:
        lines.extend(["### Learned embedding & open-set verification", ""])
        lines.append(
            "Cosine verification (same vs. different individual) on the CNN's 128-d embedding — "
            "an open-set capability the closed-set PCA classifier does not provide."
        )
        lines.append("")
        rows = []
        if ce_emb:
            ver = ce_emb.get("verification", {})
            rows.append(["cross-entropy", _format_metric(ver.get("auc")), _format_metric(ver.get("eer"))])
        if triplet_emb:
            ver = triplet_emb.get("verification", {})
            rows.append(["metric learning (triplet)", _format_metric(ver.get("auc")), _format_metric(ver.get("eer"))])
        lines.extend(_markdown_table(["Embedding", "Verification AUC", "EER"], rows))
        lines.append("")

    if triplet:
        test_block = triplet.get("final", {}).get("test", {})
        ctx = test_block.get("context_similarity", {})
        within = ctx.get("within_context_mean_sim")
        cross_sim = ctx.get("cross_context_mean_sim")
        if within is not None and cross_sim is not None:
            lines.extend(["### Context invariance of the metric-learning embedding", ""])
            lines.append(
                "Same-individual cosine similarity for within-context vs. cross-context pairs "
                "(test split). A small gap means the embedding is largely context-invariant — "
                "the opposite of the cross-entropy classifier, which collapses across context."
            )
            lines.append("")
            lines.extend(
                _markdown_table(
                    ["Within-context same-cosine", "Cross-context same-cosine", "Gap"],
                    [[_format_metric(within), _format_metric(cross_sim), _format_metric(within - cross_sim)]],
                )
            )
            lines.append("")

    return lines, cross is not None


def _significance_lines(evaluation_results: dict) -> list[str]:
    significance = evaluation_results.get("significance", {})
    if not significance:
        return []

    lines = ["## Statistical Significance", ""]
    lines.append(
        "Session-level cluster bootstrap 95% CIs (resampling whole sessions, so the interval "
        "reflects between-session variance) and a one-sided label-permutation p-value vs. a "
        "no-information null. Reported on the test split."
    )
    lines.append("")
    rows: list[list[str]] = []
    for model_name in ("pca", "cnn"):
        model_block = significance.get(model_name, {})
        for split_name, split_block in model_block.items():
            bootstrap = split_block.get("bootstrap", {})
            permutation = split_block.get("permutation", {})
            if bootstrap.get("status") != "ok":
                continue
            for metric_name in ("accuracy", "balanced_accuracy", "macro_f1", "session_level_accuracy"):
                metric_ci = bootstrap.get(metric_name, {})
                perm = permutation.get(metric_name, {}) if isinstance(permutation, dict) else {}
                p_value = perm.get("p_value") if isinstance(perm, dict) else None
                rows.append(
                    [
                        model_name.upper(),
                        split_name,
                        metric_name,
                        _format_metric(metric_ci.get("observed")),
                        f"[{_format_metric(metric_ci.get('ci_low'))}, {_format_metric(metric_ci.get('ci_high'))}]",
                        "n/a" if p_value is None else (f"{p_value:.4f}" if p_value >= 1e-4 else "<0.0001"),
                    ]
                )
    if not rows:
        return []
    lines.extend(
        _markdown_table(
            ["Model", "Split", "Metric", "Observed", "95% CI", "Permutation p"],
            rows,
        )
    )
    lines.append("")
    return lines


def generate_mvp_report(
    evaluation_results: dict,
    pca_metrics: dict,
    cnn_metrics: dict,
    segments_df: pd.DataFrame,
    split_dict: dict,
    config: dict,
    project_root: str | Path | None = None,
    *,
    recordings_qc_df: pd.DataFrame | None = None,
    features_df: pd.DataFrame | None = None,
    spectrogram_index_df: pd.DataFrame | None = None,
) -> None:
    project_root_path = resolve_project_root(config, project_root=project_root)
    data_dir = resolve_data_dir(config, project_root=project_root_path)
    reports_dir = resolve_reports_dir(config, project_root=project_root_path)
    artifacts_dir = _resolve_artifacts_dir(config, project_root=project_root_path)
    reports_dir.mkdir(parents=True, exist_ok=True)

    resolved_recordings_qc_df = (
        recordings_qc_df.copy()
        if isinstance(recordings_qc_df, pd.DataFrame)
        else _load_frame(config, "recordings_qc_df", data_dir / "interim" / "recordings_qc.parquet")
    )
    resolved_features_df = (
        features_df.copy()
        if isinstance(features_df, pd.DataFrame)
        else _load_frame(config, "features_df", data_dir / "processed" / "features_tabular.parquet")
    )
    resolved_spectrogram_index_df = (
        spectrogram_index_df.copy()
        if isinstance(spectrogram_index_df, pd.DataFrame)
        else _load_frame(
            config,
            "spectrogram_index_df",
            data_dir / "processed" / "spectrogram_index.parquet",
        )
    )

    segments_working_df = segments_df.copy()
    for column in ("segment_id", "recording_id", "session_id", "individual_id"):
        if column in segments_working_df.columns:
            segments_working_df[column] = segments_working_df[column].astype(str)

    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    split_counts = split_dict.get("metadata", {}).get("counts", {})
    feature_columns = _feature_columns(resolved_features_df)
    spectrogram_shape = "n/a"
    if not resolved_spectrogram_index_df.empty:
        spectrogram_shape = (
            f"{int(resolved_spectrogram_index_df['shape_0'].iloc[0])} x "
            f"{int(resolved_spectrogram_index_df['shape_1'].iloc[0])}"
        )

    report_path = reports_dir / "mvp_report.md"
    figure_paths = {
        "pca_explained_variance": artifacts_dir / "figures" / "pca_explained_variance.png",
        "pca_scatter_individual": artifacts_dir / "figures" / "pca_scatter_individual.png",
        "pca_scatter_session": artifacts_dir / "figures" / "pca_scatter_session.png",
        "pca_confusion_matrix": artifacts_dir / "figures" / "pca_confusion_matrix.png",
        "cnn_confusion_matrix": artifacts_dir / "figures" / "cnn_confusion_matrix.png",
        "cnn_training_history": artifacts_dir / "figures" / "cnn_training_history.png",
    }

    qc_flag_counts = (
        segments_working_df["qc_flag"].value_counts().sort_index()
        if "qc_flag" in segments_working_df.columns
        else pd.Series(dtype="int64")
    )
    temperature_pass_count = (
        int(resolved_recordings_qc_df["temp_qc_pass"].sum())
        if "temp_qc_pass" in resolved_recordings_qc_df.columns
        else 0
    )
    temperature_fail_count = (
        int((~resolved_recordings_qc_df["temp_qc_pass"]).sum())
        if "temp_qc_pass" in resolved_recordings_qc_df.columns
        else 0
    )

    lines: list[str] = [
        "# Cricket ID MVP Report",
        "",
        f"Generated: {now_utc}",
        "",
        "## Dataset Overview",
        "",
    ]
    lines.extend(
        _markdown_table(
            ["Metric", "Value"],
            [
                ["Recordings", int(resolved_recordings_qc_df["recording_id"].nunique()) if "recording_id" in resolved_recordings_qc_df.columns else "n/a"],
                ["Individuals", int(segments_working_df["individual_id"].nunique())],
                ["Sessions", int(segments_working_df["session_id"].nunique())],
                ["Segments", int(len(segments_working_df))],
                ["Train sessions", split_counts.get("sessions", {}).get("train", "n/a")],
                ["Val sessions", split_counts.get("sessions", {}).get("val", "n/a")],
                ["Test sessions", split_counts.get("sessions", {}).get("test", "n/a")],
                ["Train segments", split_counts.get("segments", {}).get("train", "n/a")],
                ["Val segments", split_counts.get("segments", {}).get("val", "n/a")],
                ["Test segments", split_counts.get("segments", {}).get("test", "n/a")],
            ],
        )
    )

    lines.extend(["", "## Segmentation Statistics", ""])
    lines.extend(
        _markdown_table(
            ["Metric", "Value"],
            [
                ["Segments per recording (mean)", f"{segments_working_df.groupby('recording_id').size().mean():.2f}" if not segments_working_df.empty else "n/a"],
                ["Segment duration fixed (s)", f"{segments_working_df['duration_s_fixed'].iloc[0]:.3f}" if "duration_s_fixed" in segments_working_df.columns and not segments_working_df.empty else "n/a"],
                ["Raw duration mean (s)", f"{segments_working_df['duration_s_raw'].mean():.3f}" if "duration_s_raw" in segments_working_df.columns and not segments_working_df.empty else "n/a"],
                ["Raw duration min (s)", f"{segments_working_df['duration_s_raw'].min():.3f}" if "duration_s_raw" in segments_working_df.columns and not segments_working_df.empty else "n/a"],
                ["Raw duration max (s)", f"{segments_working_df['duration_s_raw'].max():.3f}" if "duration_s_raw" in segments_working_df.columns and not segments_working_df.empty else "n/a"],
                ["SNR proxy mean", f"{segments_working_df['snr_proxy'].mean():.3f}" if "snr_proxy" in segments_working_df.columns and not segments_working_df.empty else "n/a"],
                ["SNR proxy min", f"{segments_working_df['snr_proxy'].min():.3f}" if "snr_proxy" in segments_working_df.columns and not segments_working_df.empty else "n/a"],
                ["SNR proxy max", f"{segments_working_df['snr_proxy'].max():.3f}" if "snr_proxy" in segments_working_df.columns and not segments_working_df.empty else "n/a"],
            ],
        )
    )

    lines.extend(["", "## QC Statistics", ""])
    qc_rows: list[list[object]] = [
        ["Temperature QC pass", temperature_pass_count],
        ["Temperature QC fail", temperature_fail_count],
    ]
    for flag, count in qc_flag_counts.items():
        qc_rows.append([f"Segment QC: {flag}", int(count)])
    lines.extend(_markdown_table(["Check", "Count"], qc_rows))

    lines.extend(["", "## Feature Overview", ""])
    lines.extend(
        _markdown_table(
            ["Metric", "Value"],
            [
                ["Tabular feature columns", len(feature_columns)],
                ["Example features", ", ".join(feature_columns[:8]) if feature_columns else "n/a"],
                ["Spectrogram count", int(len(resolved_spectrogram_index_df))],
                ["Spectrogram shape", spectrogram_shape],
                ["Temperature used as model feature", "no"],
            ],
        )
    )

    lines.extend(["", "## PCA Results", ""])
    lines.append(
        f"Classifier: {pca_metrics.get('classifier', 'n/a')}; "
        f"effective_n_components: {pca_metrics.get('effective_n_components', 'n/a')}."
    )
    lines.append("")
    lines.extend(
        _markdown_table(
            [
                "Split",
                "Accuracy",
                "Balanced Accuracy",
                "Macro F1",
                "Session Accuracy",
                "Segments",
                "Sessions",
            ],
            _metric_table_rows(pca_metrics),
        )
    )
    lines.extend(
        _figure_block(
            reports_dir,
            [
                ("PCA explained variance", figure_paths["pca_explained_variance"]),
                ("PCA scatter by individual", figure_paths["pca_scatter_individual"]),
                ("PCA scatter by session", figure_paths["pca_scatter_session"]),
                ("PCA confusion matrix", figure_paths["pca_confusion_matrix"]),
            ],
        )
    )

    lines.extend(["## CNN Results", ""])
    lines.append(
        f"Architecture: {cnn_metrics.get('architecture', 'n/a')}; "
        f"best_epoch: {cnn_metrics.get('best_epoch', 'n/a')}; "
        f"epochs_ran: {cnn_metrics.get('epochs_ran', 'n/a')}."
    )
    lines.append("")
    lines.extend(
        _markdown_table(
            [
                "Split",
                "Accuracy",
                "Balanced Accuracy",
                "Macro F1",
                "Session Accuracy",
                "Segments",
                "Sessions",
            ],
            _metric_table_rows(cnn_metrics),
        )
    )
    lines.extend(
        _figure_block(
            reports_dir,
            [
                ("CNN confusion matrix", figure_paths["cnn_confusion_matrix"]),
                ("CNN training history", figure_paths["cnn_training_history"]),
            ],
        )
    )

    lines.extend(["## PCA vs CNN Comparison", ""])
    lines.append(
        "Uniform random chance baseline: "
        f"{_format_metric(evaluation_results.get('chance_baseline', {}).get('uniform_random', {}).get('accuracy'))}."
    )
    lines.append("")
    lines.extend(
        _markdown_table(
            ["Split", "Metric", "PCA", "CNN", "Delta (CNN-PCA)", "Winner"],
            _comparison_rows(evaluation_results),
        )
    )

    lines.extend(["", *_significance_lines(evaluation_results)])

    lines.extend(["", "## Error Analysis", ""])
    lines.extend(_error_analysis_lines("pca", evaluation_results.get("error_analysis", {}).get("pca", {})))
    lines.extend(_error_analysis_lines("cnn", evaluation_results.get("error_analysis", {}).get("cnn", {})))

    advanced_lines, cross_context_present = _advanced_analyses_lines(artifacts_dir / "metrics")
    lines.extend(advanced_lines)

    lines.extend(["## Limitations", ""])
    limitations = [
        "Current scope is restricted to laboratory calling songs and session-separated evaluation only.",
        "Temperature is used for QC filtering and reporting, not as a classifier feature; temperature residualization is blocked until temperature logs are populated.",
    ]
    if not cross_context_present:
        limitations.append(
            "The report does not yet include confidence intervals, permutation baselines, or cross-context validation."
        )
    else:
        limitations.append(
            "Confidence intervals and permutation baselines are not yet computed at the model level."
        )
    if len(segments_working_df) < 100:
        limitations.append(
            f"Evaluation is based on a small dataset ({len(segments_working_df)} segments), so variance may be high."
        )
    if evaluation_results.get("best_model") == "tie":
        limitations.append("The current comparison does not identify a single clear winner on the held-out split.")
    lines.extend(f"- {item}" for item in limitations)

    lines.extend(["", "## Next Steps", ""])
    next_steps = [
        "Collect more sessions per individual and broaden temperature/SNR coverage.",
        "Add bootstrap confidence intervals and permutation baselines to the evaluation module.",
        "Extend validation beyond lab/calling data to new contexts or song types.",
        "Inspect the worst sessions in the error analysis and refine segmentation or augmentation accordingly.",
    ]
    lines.extend(f"- {item}" for item in next_steps)
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
