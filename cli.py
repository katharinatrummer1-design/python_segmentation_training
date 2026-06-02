from __future__ import annotations

import json
import sys
from pathlib import Path

import typer

from cricket_id.evaluation.cross_context import run_cross_context_eval
from cricket_id.evaluation.cross_validate import run_pca_cross_validation
from cricket_id.evaluation.embeddings import run_embedding_analysis
from cricket_id.evaluation.evaluate import run_evaluation
from cricket_id.evaluation.feature_analysis import run_feature_analysis
from cricket_id.evaluation.report import generate_mvp_report
from cricket_id.evaluation.summary_report import generate_overall_summary
from cricket_id.evaluation.statistical_comparison import (
    run_statistical_feature_comparison,
)
from cricket_id.features.spectrograms import run_spectrogram_extraction
from cricket_id.features.tabular import run_feature_extraction
from cricket_id.io.audio import build_audio_index
from cricket_id.io.ingest import build_recordings_manifest
from cricket_id.io.manifest import (
    check_session_sufficiency,
    generate_data_gap_report,
    load_recordings_manifest,
    load_temperature_manifest,
    validate_manifest as validate_recordings_manifest,
    _read_parquet_compatible,
    _write_parquet_compatible,
)
from cricket_id.models.cnn_trainer import (
    train_cnn_baseline as train_cnn_baseline_model,
)
from cricket_id.models.triplet_trainer import train_triplet_embedding
from cricket_id.models.pca_baseline import (
    train_pca_baseline as train_pca_baseline_model,
)
from cricket_id.preprocessing.residualize import run_temperature_residualization
from cricket_id.preprocessing.temperature import (
    apply_temperature_qc,
    merge_temperature,
    save_recordings_qc,
)
from cricket_id.segmentation.review import generate_segment_review
from cricket_id.segmentation.segment import run_segmentation
from cricket_id.splits.build_splits import (
    build_session_splits,
    save_splits,
    validate_split_integrity,
)
from cricket_id.utils.config import load_config
from cricket_id.utils.paths import resolve_data_dir, resolve_reports_dir


DEFAULT_CONFIG = "configs/mvp_lab_calling.yaml"

app = typer.Typer()


def _not_implemented(_: str) -> None:
    typer.echo("Not implemented yet")


def _cli_project_root(config_path: str) -> Path:
    resolved_config_path = Path(config_path).resolve()
    if resolved_config_path.parent.name == "configs":
        return resolved_config_path.parent.parent
    return Path.cwd().resolve()


def _resolve_artifacts_dir(config_dict: dict, project_root: Path) -> Path:
    artifacts_dir = Path(str(config_dict.get("paths", {}).get("artifacts_dir", "artifacts")))
    return artifacts_dir if artifacts_dir.is_absolute() else (project_root / artifacts_dir)


def _format_metric(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


def _resolve_project_path(path_value: str, project_root: Path) -> Path:
    candidate = Path(path_value)
    return candidate if candidate.is_absolute() else (project_root / candidate)


def _runtime_config(config_dict: dict, project_root: Path) -> dict:
    runtime_config = json.loads(json.dumps(config_dict))
    runtime_config.setdefault("paths", {})
    runtime_config["paths"]["data_dir"] = str(
        resolve_data_dir(config_dict, project_root=project_root)
    )
    runtime_config["paths"]["reports_dir"] = str(
        resolve_reports_dir(config_dict, project_root=project_root)
    )
    runtime_config["paths"]["artifacts_dir"] = str(
        _resolve_artifacts_dir(config_dict, project_root)
    )
    return runtime_config


def _manifest_input_paths(config_dict: dict, project_root: Path) -> tuple[Path, Path]:
    manifest_config = config_dict["manifest"]
    recordings_path = _resolve_project_path(manifest_config["recordings"], project_root)
    temperature_path = _resolve_project_path(manifest_config["temperature"], project_root)
    return recordings_path, temperature_path


def _load_required_json(path: Path, step_name: str) -> dict:
    if not path.exists():
        typer.echo(f"Missing required input: {path}. Run {step_name} first.")
        sys.exit(1)
    return json.loads(path.read_text(encoding="utf-8"))


def _maybe_ingest(config_dict: dict, project_root: Path) -> None:
    """Build recordings.csv from a raw dataset folder when configured.

    Driven by the optional ``ingest`` config block::

        ingest:
          source_root: "katharina grillen"
          auto: true            # regenerate during run-mvp
    """
    ingest_config = config_dict.get("ingest")
    if not ingest_config:
        return

    source_root_value = ingest_config.get("source_root")
    if not source_root_value:
        return

    source_root = _resolve_project_path(str(source_root_value), project_root)
    recordings_path, _ = _manifest_input_paths(config_dict, project_root)
    output_dir = recordings_path.parent

    if recordings_path.exists() and not bool(ingest_config.get("auto", False)):
        return

    interim_manifest_dir = (
        resolve_data_dir(config_dict, project_root=project_root)
        / "interim"
        / "manifest"
    )
    manifest_df, summary = build_recordings_manifest(
        source_root,
        output_dir,
        qc_thresholds=ingest_config.get("qc_thresholds", {}),
        interim_dir=interim_manifest_dir,
    )
    typer.echo(
        "Ingestion complete: "
        f"scanned={summary.total_wav_files}, kept={summary.kept}, "
        f"exact_duplicates={summary.exact_duplicates}, "
        f"name_duplicates={summary.name_duplicates}, "
        f"unparsed={summary.unparsed}, qc_warn={summary.qc_warn}, "
        f"qc_fail={summary.qc_fail}"
    )
    typer.echo(f"Manifest written: {recordings_path} ({len(manifest_df)} rows)")


@app.command("build-manifest")
def build_manifest_command(
    config: str = typer.Option(DEFAULT_CONFIG, "--config"),
) -> None:
    """Scan the raw dataset folder and (re)generate recordings.csv + temperature.csv."""
    project_root = _cli_project_root(config)
    config_dict = load_config(config)

    ingest_config = config_dict.get("ingest") or {}
    source_root_value = ingest_config.get("source_root")
    if not source_root_value:
        typer.echo(
            "No ingest.source_root configured. Add an 'ingest' block to the config "
            "to enable manifest generation from a raw dataset folder."
        )
        sys.exit(1)

    source_root = _resolve_project_path(str(source_root_value), project_root)
    recordings_path, _ = _manifest_input_paths(config_dict, project_root)
    output_dir = recordings_path.parent
    interim_manifest_dir = (
        resolve_data_dir(config_dict, project_root=project_root)
        / "interim"
        / "manifest"
    )

    manifest_df, summary = build_recordings_manifest(
        source_root,
        output_dir,
        qc_thresholds=ingest_config.get("qc_thresholds", {}),
        interim_dir=interim_manifest_dir,
    )

    typer.echo(
        "Ingestion complete: "
        f"scanned={summary.total_wav_files}, kept={summary.kept}, "
        f"exact_duplicates={summary.exact_duplicates}, "
        f"name_duplicates={summary.name_duplicates}, "
        f"unparsed={summary.unparsed}, qc_warn={summary.qc_warn}, "
        f"qc_fail={summary.qc_fail}"
    )
    if summary.normalizations:
        typer.echo(f"Normalizations applied: {len(summary.normalizations)}")
    typer.echo(f"Manifest written to {recordings_path} ({len(manifest_df)} rows)")
    by_context = manifest_df.groupby(["context", "song_type"]).size().to_dict()
    typer.echo(f"Rows by (context, song_type): {by_context}")


@app.command("validate-manifest")
def validate_manifest_command(
    config: str = typer.Option(DEFAULT_CONFIG, "--config"),
) -> None:
    project_root = _cli_project_root(config)
    config_dict = load_config(config)
    runtime_config = _runtime_config(config_dict, project_root)
    recordings_path, temperature_path = _manifest_input_paths(config_dict, project_root)
    data_dir = resolve_data_dir(runtime_config, project_root=project_root)
    interim_dir = data_dir / "interim"
    interim_dir.mkdir(parents=True, exist_ok=True)

    recordings_df = load_recordings_manifest(str(recordings_path))
    temperature_df = load_temperature_manifest(str(temperature_path))
    filtered_df, issues = validate_recordings_manifest(recordings_df, runtime_config)

    for issue in issues:
        typer.echo(f"ISSUE: {issue}")

    if issues:
        sys.exit(1)

    min_sessions = int(
        runtime_config.get("mvp_scope", {}).get("min_sessions_per_individual", 3)
    )
    all_have_enough, summary_df = check_session_sufficiency(filtered_df, min_sessions)
    enforce_gate = bool(
        runtime_config.get("mvp_scope", {}).get("enforce_min_sessions", True)
    )

    if not all_have_enough:
        report_dir = Path(runtime_config["paths"].get("reports_dir", "reports"))
        report_path = report_dir / "data_gap_report.md"
        generate_data_gap_report(summary_df, str(report_path))
        if enforce_gate:
            typer.echo(
                f"Insufficient session coverage. Data gap report written to {report_path}."
            )
            sys.exit(1)
        typer.echo(
            "WARNING: some individuals have fewer than "
            f"{min_sessions} sessions. Data gap report written to {report_path}. "
            "Continuing because mvp_scope.enforce_min_sessions is false."
        )

    validated_output_path = interim_dir / "recordings_validated.parquet"
    _write_parquet_compatible(filtered_df, validated_output_path)
    typer.echo(f"Validated recordings saved to {validated_output_path}")

    recordings_with_temp_df = merge_temperature(filtered_df, temperature_df)
    recordings_qc_df = apply_temperature_qc(recordings_with_temp_df, runtime_config)
    recordings_qc_output_path = interim_dir / "recordings_qc.parquet"
    save_recordings_qc(recordings_qc_df, str(recordings_qc_output_path))
    typer.echo(f"Temperature QC output saved to {recordings_qc_output_path}")

    audio_index_df = build_audio_index(filtered_df, runtime_config)
    typer.echo(f"Audio index built with {len(audio_index_df)} rows")


@app.command("segment")
def segment(
    config: str = typer.Option(DEFAULT_CONFIG, "--config"),
) -> None:
    config_dict = load_config(config)
    data_dir = resolve_data_dir(config_dict)
    recordings_qc_path = data_dir / "interim" / "recordings_qc.parquet"
    if not recordings_qc_path.exists():
        typer.echo(
            f"Missing required input: {recordings_qc_path}. Run validate-manifest first."
        )
        sys.exit(1)

    recordings_qc_df = _read_parquet_compatible(recordings_qc_path)
    segments_df = run_segmentation(recordings_qc_df, config_dict)
    generate_segment_review(segments_df, config_dict)

    typer.echo(
        "Segmentation complete: "
        f"{len(recordings_qc_df)} recordings -> {len(segments_df)} segments"
    )
    if not segments_df.empty:
        typer.echo(
            "QC flags: "
            + ", ".join(
                f"{flag}={count}"
                for flag, count in segments_df["qc_flag"].value_counts().items()
            )
        )
        typer.echo(
            "SNR proxy summary: "
            f"mean={segments_df['snr_proxy'].mean():.2f}, "
            f"min={segments_df['snr_proxy'].min():.2f}, "
            f"max={segments_df['snr_proxy'].max():.2f}"
        )


@app.command("extract-features")
def extract_features(
    config: str = typer.Option(DEFAULT_CONFIG, "--config"),
) -> None:
    config_dict = load_config(config)
    data_dir = resolve_data_dir(config_dict)
    segments_path = data_dir / "interim" / "segments.parquet"
    if not segments_path.exists():
        typer.echo(f"Missing required input: {segments_path}. Run segment first.")
        sys.exit(1)

    segments_df = _read_parquet_compatible(segments_path)
    features_df = run_feature_extraction(segments_df, config_dict)
    spectrogram_index_df = run_spectrogram_extraction(segments_df, config_dict)

    typer.echo(
        "Feature extraction complete: "
        f"{len(features_df)} tabular rows, {len(spectrogram_index_df)} spectrograms"
    )
    if not spectrogram_index_df.empty:
        typer.echo(
            "Spectrogram shape: "
            f"{int(spectrogram_index_df['shape_0'].iloc[0])} x "
            f"{int(spectrogram_index_df['shape_1'].iloc[0])}"
        )


@app.command("analyze-features")
def analyze_features(
    config: str = typer.Option(DEFAULT_CONFIG, "--config"),
) -> None:
    """Rank feature signal and generate feature-analysis plots/report."""
    config_dict = load_config(config)
    project_root = _cli_project_root(config)
    data_dir = resolve_data_dir(config_dict, project_root=project_root)
    features_path = data_dir / "processed" / "features_tabular.parquet"
    if not features_path.exists():
        typer.echo(
            f"Missing required input: {features_path}. Run extract-features first."
        )
        sys.exit(1)

    features_df = _read_parquet_compatible(features_path)
    results = run_feature_analysis(
        features_df,
        config_dict,
        project_root=project_root,
    )
    top_features = results.get("top_features", [])
    top_feature = top_features[0] if top_features else {}
    typer.echo(
        "Feature analysis complete: "
        f"recordings={results['input_counts']['recordings']}, "
        f"individuals={results['input_counts']['individuals']}, "
        f"features={results['input_counts']['feature_columns']}, "
        f"top_feature={top_feature.get('feature', 'n/a')}, "
        f"eta_squared={float(top_feature.get('eta_squared', 0.0)):.3f}"
    )
    typer.echo("Wrote artifacts/metrics/feature_analysis.json")
    typer.echo("Wrote reports/feature_analysis.md")


@app.command("compare-features")
def compare_features(
    config: str = typer.Option(DEFAULT_CONFIG, "--config"),
) -> None:
    """Run statistical feature comparisons with uncertainty and pairwise plots."""
    config_dict = load_config(config)
    project_root = _cli_project_root(config)
    data_dir = resolve_data_dir(config_dict, project_root=project_root)
    features_path = data_dir / "processed" / "features_tabular.parquet"
    recordings_path = data_dir / "interim" / "recordings_validated.parquet"
    if not features_path.exists():
        typer.echo(
            f"Missing required input: {features_path}. Run extract-features first."
        )
        sys.exit(1)

    features_df = _read_parquet_compatible(features_path)
    recordings_df = (
        _read_parquet_compatible(recordings_path)
        if recordings_path.exists()
        else None
    )
    results = run_statistical_feature_comparison(
        features_df,
        config_dict,
        recordings_df=recordings_df,
        project_root=project_root,
    )
    top_feature = results["feature_rankings"][0]
    typer.echo(
        "Feature comparison complete: "
        f"recordings={results['input_counts']['recordings']}, "
        f"individuals={results['input_counts']['individuals']}, "
        f"selected_feature={results['selected_feature']}, "
        f"top_feature={top_feature['feature']}, "
        f"eta_squared={float(top_feature['eta_squared']):.3f}, "
        f"ci=[{float(top_feature['eta_ci_low']):.3f}, {float(top_feature['eta_ci_high']):.3f}], "
        f"fdr_q={float(top_feature['fdr_q']):.3f}"
    )
    typer.echo("Wrote artifacts/metrics/feature_comparison.json")
    typer.echo("Wrote reports/feature_comparison.md")


@app.command("cross-context-eval")
def cross_context_eval(
    config: str = typer.Option(DEFAULT_CONFIG, "--config"),
) -> None:
    """Train on one diel context (e.g. night) and test on the other (e.g. day).

    Isolates the identity-vs-recording-context confound: reports within-context
    (ceiling) vs. cross-context (transfer) accuracy and the per-metric penalty.
    """
    config_dict = load_config(config)
    project_root = _cli_project_root(config)
    data_dir = resolve_data_dir(config_dict, project_root=project_root)
    features_path = data_dir / "processed" / "features_tabular.parquet"
    recordings_path = data_dir / "interim" / "recordings_validated.parquet"
    if not features_path.exists():
        typer.echo(
            f"Missing required input: {features_path}. Run extract-features first."
        )
        sys.exit(1)
    if not recordings_path.exists():
        typer.echo(
            f"Missing required input: {recordings_path}. Run validate-manifest first."
        )
        sys.exit(1)

    features_df = _read_parquet_compatible(features_path)
    recordings_df = _read_parquet_compatible(recordings_path)
    try:
        results = run_cross_context_eval(
            features_df,
            recordings_df,
            config_dict,
            project_root=project_root,
        )
    except ValueError as error:
        typer.echo(f"Cross-context evaluation could not run: {error}")
        sys.exit(1)

    typer.echo(
        "Cross-context evaluation complete: "
        f"context={results['context_column']}, "
        f"shared_individuals={results['n_shared_individuals']}, "
        f"contexts={results['contexts']}"
    )
    for direction in results["directions"]:
        within_f1 = direction["within_context"].get("macro_f1")
        cross_f1 = direction["cross_context"].get("macro_f1")
        penalty_f1 = direction["penalty"].get("macro_f1")
        typer.echo(
            f"{direction['label']}: "
            f"within_macro_f1={_format_metric(within_f1)}, "
            f"cross_macro_f1={_format_metric(cross_f1)}, "
            f"penalty={_format_metric(penalty_f1)}"
        )
    typer.echo("Wrote artifacts/metrics/cross_context.json")
    typer.echo("Wrote reports/cross_context.md")


@app.command("extract-embeddings")
def extract_embeddings_command(
    config: str = typer.Option(DEFAULT_CONFIG, "--config"),
    checkpoint: str = typer.Option("", "--checkpoint"),
) -> None:
    """Extract the CNN's learned 128-d embedding per chirp and analyse it.

    Produces a 2D embedding map (UMAP/t-SNE/PCA) and open-set verification
    (same-vs-different individual) with ROC/AUC/EER — capabilities the closed-set
    PCA baseline does not provide. Pass --checkpoint cnn_triplet.pt to analyse the
    metric-learning model (outputs are suffixed by the model's loss type).
    """
    config_dict = load_config(config)
    if str(checkpoint).strip():
        config_dict.setdefault("embeddings", {})
        config_dict["embeddings"]["checkpoint"] = str(checkpoint).strip()
    project_root = _cli_project_root(config)
    data_dir = resolve_data_dir(config_dict, project_root=project_root)
    spectrogram_index_path = data_dir / "processed" / "spectrogram_index.parquet"
    segments_path = data_dir / "interim" / "segments.parquet"
    if not spectrogram_index_path.exists():
        typer.echo(
            f"Missing required input: {spectrogram_index_path}. Run extract-features first."
        )
        sys.exit(1)
    if not segments_path.exists():
        typer.echo(f"Missing required input: {segments_path}. Run segment first.")
        sys.exit(1)

    spectrogram_index_df = _read_parquet_compatible(spectrogram_index_path)
    segments_df = _read_parquet_compatible(segments_path)
    try:
        results = run_embedding_analysis(
            spectrogram_index_df,
            segments_df,
            config_dict,
            project_root=project_root,
        )
    except ValueError as error:
        typer.echo(f"Embedding analysis could not run: {error}")
        sys.exit(1)

    verification = results["verification"]
    typer.echo(
        "Embedding analysis complete: "
        f"segments={results['n_segments']}, dim={results['embedding_dim']}, "
        f"projection={results['projection_method']}"
    )
    if verification.get("status") == "ok":
        typer.echo(
            "Verification: "
            f"auc={_format_metric(verification['auc'])}, "
            f"eer={_format_metric(verification['eer'])}, "
            f"mean_sim_same={_format_metric(verification['mean_sim_same'])}, "
            f"mean_sim_diff={_format_metric(verification['mean_sim_diff'])}"
        )
    else:
        typer.echo(f"Verification status: {verification.get('status')}")
    typer.echo(f"Model: {results.get('checkpoint')} (loss={results.get('loss')})")
    typer.echo(f"Wrote {results['artifact_paths']['metrics']}")
    typer.echo(f"Wrote {results['artifact_paths']['embeddings']}")


@app.command("cross-validate")
def cross_validate_command(
    config: str = typer.Option(DEFAULT_CONFIG, "--config"),
) -> None:
    """Honest session-grouped K-fold CV for the class-balanced PCA pipeline."""
    config_dict = load_config(config)
    project_root = _cli_project_root(config)
    data_dir = resolve_data_dir(config_dict, project_root=project_root)
    features_path = data_dir / "processed" / "features_tabular.parquet"
    if not features_path.exists():
        typer.echo(
            f"Missing required input: {features_path}. Run extract-features first."
        )
        sys.exit(1)

    features_df = _read_parquet_compatible(features_path)
    try:
        results = run_pca_cross_validation(features_df, config_dict, project_root=project_root)
    except ValueError as error:
        typer.echo(f"Cross-validation could not run: {error}")
        sys.exit(1)

    pooled = results["pooled"]
    boot = results.get("pooled_bootstrap", {})
    macro_ci = boot.get("macro_f1", {}) if isinstance(boot, dict) else {}
    typer.echo(
        "PCA session-grouped CV complete: "
        f"folds={results['n_splits']}, "
        f"individuals_evaluated={results['n_individuals_evaluated']}/{results['n_individuals_total']}"
    )
    typer.echo(
        "pooled: "
        f"accuracy={_format_metric(pooled['accuracy'])}, "
        f"balanced_accuracy={_format_metric(pooled['balanced_accuracy'])}, "
        f"macro_f1={_format_metric(pooled['macro_f1'])}"
    )
    if isinstance(macro_ci, dict) and "ci_low" in macro_ci:
        typer.echo(
            f"macro_f1 95% CI: [{_format_metric(macro_ci['ci_low'])}, {_format_metric(macro_ci['ci_high'])}]"
        )
    typer.echo("Wrote artifacts/metrics/pca_cross_validation.json")
    typer.echo("Wrote reports/pca_cross_validation.md")


@app.command("residualize-temp")
def residualize_temp(
    config: str = typer.Option(DEFAULT_CONFIG, "--config"),
) -> None:
    """Remove the linear temperature component from features (when temperature exists).

    No-op blocker (with a clear report) when no temperature values are available.
    """
    config_dict = load_config(config)
    project_root = _cli_project_root(config)
    data_dir = resolve_data_dir(config_dict, project_root=project_root)
    features_path = data_dir / "processed" / "features_tabular.parquet"
    if not features_path.exists():
        typer.echo(
            f"Missing required input: {features_path}. Run extract-features first."
        )
        sys.exit(1)

    features_df = _read_parquet_compatible(features_path)
    result = run_temperature_residualization(
        features_df,
        config_dict,
        project_root=project_root,
    )
    if result["status"] == "blocked":
        typer.echo(f"Temperature residualization BLOCKED: {result['reason']}")
        typer.echo(f"Blocker report written to {result['report']}")
        return
    typer.echo(
        "Temperature residualization complete: "
        f"features={result['n_features']}, "
        f"segments_with_temp={result['n_segments_with_temperature']}, "
        f"temp_range={result['temperature_min_c']:.2f}-{result['temperature_max_c']:.2f} C"
    )
    typer.echo(f"Wrote {result['output']}")


@app.command("build-splits")
def build_splits(
    config: str = typer.Option(DEFAULT_CONFIG, "--config"),
) -> None:
    config_dict = load_config(config)
    project_root = _cli_project_root(config)
    data_dir = resolve_data_dir(config_dict, project_root=project_root)
    segments_path = data_dir / "interim" / "segments.parquet"
    if not segments_path.exists():
        typer.echo(f"Missing required input: {segments_path}. Run segment first.")
        sys.exit(1)

    segments_df = _read_parquet_compatible(segments_path)
    split_dict = build_session_splits(segments_df, config_dict)
    issues = validate_split_integrity(split_dict, segments_df)
    for issue in issues:
        typer.echo(f"ISSUE: {issue}")
    if issues:
        sys.exit(1)

    artifacts_dir = _resolve_artifacts_dir(config_dict, project_root)
    split_path = artifacts_dir / "splits" / "split_v1.json"
    save_splits(split_dict, str(split_path))

    typer.echo(f"Saved session-level splits to {split_path}")
    counts = split_dict.get("metadata", {}).get("counts", {})
    typer.echo(
        "Split counts: "
        f"sessions={counts.get('sessions', {})}, "
        f"segments={counts.get('segments', {})}"
    )


@app.command("train-pca-baseline")
def train_pca_baseline(
    config: str = typer.Option(DEFAULT_CONFIG, "--config"),
) -> None:
    config_dict = load_config(config)
    project_root = _cli_project_root(config)
    data_dir = resolve_data_dir(config_dict, project_root=project_root)
    artifacts_dir = _resolve_artifacts_dir(config_dict, project_root)

    features_path = data_dir / "processed" / "features_tabular.parquet"
    split_path = artifacts_dir / "splits" / "split_v1.json"
    if not features_path.exists():
        typer.echo(
            f"Missing required input: {features_path}. Run extract-features first."
        )
        sys.exit(1)
    if not split_path.exists():
        typer.echo(f"Missing required input: {split_path}. Run build-splits first.")
        sys.exit(1)

    features_df = _read_parquet_compatible(features_path)
    split_dict = json.loads(split_path.read_text(encoding="utf-8"))
    results = train_pca_baseline_model(
        features_df,
        split_dict,
        config_dict,
        project_root=project_root,
    )

    typer.echo(
        "PCA baseline complete: "
        f"classifier={results['classifier']}, "
        f"components={results['effective_n_components']}"
    )
    for split_name in ("val", "test"):
        split_metrics = results["metrics"][split_name]
        typer.echo(
            f"{split_name}: "
            f"n={split_metrics['sample_count']}, "
            f"session_n={split_metrics['session_count']}, "
            f"accuracy={_format_metric(split_metrics['accuracy'])}, "
            f"balanced_accuracy={_format_metric(split_metrics['balanced_accuracy'])}, "
            f"macro_f1={_format_metric(split_metrics['macro_f1'])}, "
            f"session_accuracy={_format_metric(split_metrics['session_level_accuracy'])}"
        )


@app.command("train-cnn")
def train_cnn(
    config: str = typer.Option(DEFAULT_CONFIG, "--config"),
    loss: str = typer.Option("ce", "--loss"),
) -> None:
    """Train the CNN. --loss ce = classifier baseline; --loss triplet = metric-learning embedding."""
    config_dict = load_config(config)
    project_root = _cli_project_root(config)
    data_dir = resolve_data_dir(config_dict, project_root=project_root)
    artifacts_dir = _resolve_artifacts_dir(config_dict, project_root)

    spectrogram_index_path = data_dir / "processed" / "spectrogram_index.parquet"
    features_path = data_dir / "processed" / "features_tabular.parquet"
    split_path = artifacts_dir / "splits" / "split_v1.json"
    if not spectrogram_index_path.exists():
        typer.echo(
            "Missing required input: "
            f"{spectrogram_index_path}. Run extract-features first."
        )
        sys.exit(1)
    if not features_path.exists():
        typer.echo(
            f"Missing required input: {features_path}. Run extract-features first."
        )
        sys.exit(1)
    if not split_path.exists():
        typer.echo(f"Missing required input: {split_path}. Run build-splits first.")
        sys.exit(1)

    spectrogram_index_df = _read_parquet_compatible(spectrogram_index_path)
    features_df = _read_parquet_compatible(features_path)
    split_dict = json.loads(split_path.read_text(encoding="utf-8"))

    loss_mode = str(loss).strip().lower()
    if loss_mode == "triplet":
        segments_path = data_dir / "interim" / "segments.parquet"
        if not segments_path.exists():
            typer.echo(f"Missing required input: {segments_path}. Run segment first.")
            sys.exit(1)
        segments_df = _read_parquet_compatible(segments_path)
        try:
            triplet_results = train_triplet_embedding(
                spectrogram_index_df,
                segments_df,
                split_dict,
                config_dict,
                project_root=project_root,
            )
        except ValueError as error:
            typer.echo(f"Triplet training could not run: {error}")
            sys.exit(1)
        typer.echo(
            "Triplet embedding complete: "
            f"individuals={triplet_results['num_individuals']}, "
            f"best_epoch={triplet_results['best_epoch']}"
        )
        for split_name in ("val", "test"):
            block = triplet_results["final"].get(split_name, {})
            ver = block.get("verification", {})
            ctx = block.get("context_similarity", {})
            typer.echo(
                f"{split_name}: "
                f"auc={_format_metric(ver.get('auc'))}, "
                f"eer={_format_metric(ver.get('eer'))}, "
                f"same_within={_format_metric(ctx.get('within_context_mean_sim'))}, "
                f"same_cross={_format_metric(ctx.get('cross_context_mean_sim'))}"
            )
        typer.echo("Wrote artifacts/models/cnn_triplet.pt")
        typer.echo("Wrote artifacts/metrics/cnn_triplet_metrics.json")
        typer.echo("Wrote reports/cnn_triplet.md")
        return

    if loss_mode != "ce":
        typer.echo(f"Unsupported --loss '{loss}'. Expected ce or triplet.")
        sys.exit(1)

    results = train_cnn_baseline_model(
        spectrogram_index_df,
        features_df,
        split_dict,
        config_dict,
        project_root=project_root,
    )

    typer.echo(
        "CNN baseline complete: "
        f"model={results['architecture']}, "
        f"num_classes={results['num_classes']}, "
        f"best_epoch={results['best_epoch']}"
    )
    for split_name in ("val", "test"):
        split_metrics = results["metrics"][split_name]
        typer.echo(
            f"{split_name}: "
            f"n={split_metrics['sample_count']}, "
            f"session_n={split_metrics['session_count']}, "
            f"accuracy={_format_metric(split_metrics['accuracy'])}, "
            f"balanced_accuracy={_format_metric(split_metrics['balanced_accuracy'])}, "
            f"macro_f1={_format_metric(split_metrics['macro_f1'])}, "
            f"session_accuracy={_format_metric(split_metrics['session_level_accuracy'])}"
        )


@app.command("evaluate")
def evaluate(
    config: str = typer.Option(DEFAULT_CONFIG, "--config"),
) -> None:
    project_root = _cli_project_root(config)
    config_dict = load_config(config)
    runtime_config = _runtime_config(config_dict, project_root)
    data_dir = resolve_data_dir(runtime_config, project_root=project_root)
    artifacts_dir = _resolve_artifacts_dir(runtime_config, project_root)

    pca_metrics = _load_required_json(
        artifacts_dir / "metrics" / "pca_metrics.json",
        "train-pca-baseline",
    )
    cnn_metrics = _load_required_json(
        artifacts_dir / "metrics" / "cnn_metrics.json",
        "train-cnn",
    )

    segments_path = data_dir / "interim" / "segments.parquet"
    split_path = artifacts_dir / "splits" / "split_v1.json"
    if not segments_path.exists():
        typer.echo(f"Missing required input: {segments_path}. Run segment first.")
        sys.exit(1)
    if not split_path.exists():
        typer.echo(f"Missing required input: {split_path}. Run build-splits first.")
        sys.exit(1)

    segments_df = _read_parquet_compatible(segments_path)
    split_dict = json.loads(split_path.read_text(encoding="utf-8"))
    evaluation_results = run_evaluation(
        pca_metrics,
        cnn_metrics,
        segments_df,
        split_dict,
        runtime_config,
        project_root=project_root,
    )
    generate_mvp_report(
        evaluation_results,
        pca_metrics,
        cnn_metrics,
        segments_df,
        split_dict,
        runtime_config,
        project_root=project_root,
    )

    summary_path = generate_overall_summary(config_dict, project_root=project_root)
    typer.echo(f"Wrote {summary_path}")

    typer.echo(
        "Evaluation complete: "
        f"best_model={evaluation_results['best_model']}, "
        f"chance={_format_metric(evaluation_results['chance_baseline']['uniform_random']['accuracy'])}"
    )
    for split_name in ("val", "test"):
        comparison = evaluation_results["comparison"][split_name]["macro_f1"]
        typer.echo(
            f"{split_name}: "
            f"pca_macro_f1={_format_metric(comparison['pca'])}, "
            f"cnn_macro_f1={_format_metric(comparison['cnn'])}, "
            f"winner={comparison['winner']}"
        )


@app.command("summary")
def summary_command(
    config: str = typer.Option(DEFAULT_CONFIG, "--config"),
) -> None:
    """Generate the overall executive summary (reports/overall_summary.md)."""
    config_dict = load_config(config)
    project_root = _cli_project_root(config)
    output_path = generate_overall_summary(config_dict, project_root=project_root)
    typer.echo(f"Wrote {output_path}")


@app.command("run-mvp")
def run_mvp(
    config: str = typer.Option(DEFAULT_CONFIG, "--config"),
) -> None:
    project_root = _cli_project_root(config)
    config_dict = load_config(config)
    runtime_config = _runtime_config(config_dict, project_root)
    data_dir = resolve_data_dir(runtime_config, project_root=project_root)
    artifacts_dir = _resolve_artifacts_dir(runtime_config, project_root)
    recordings_path, temperature_path = _manifest_input_paths(config_dict, project_root)

    _maybe_ingest(config_dict, project_root)

    recordings_df = load_recordings_manifest(str(recordings_path))
    temperature_df = load_temperature_manifest(str(temperature_path))
    filtered_df, issues = validate_recordings_manifest(recordings_df, runtime_config)
    for issue in issues:
        typer.echo(f"ISSUE: {issue}")
    if issues:
        sys.exit(1)

    min_sessions = int(
        runtime_config.get("mvp_scope", {}).get("min_sessions_per_individual", 3)
    )
    all_have_enough, summary_df = check_session_sufficiency(filtered_df, min_sessions)
    enforce_gate = bool(
        runtime_config.get("mvp_scope", {}).get("enforce_min_sessions", True)
    )
    if not all_have_enough:
        report_path = resolve_reports_dir(runtime_config, project_root=project_root) / "data_gap_report.md"
        generate_data_gap_report(summary_df, str(report_path))
        if enforce_gate:
            typer.echo(
                f"Insufficient session coverage. Data gap report written to {report_path}."
            )
            sys.exit(1)
        typer.echo(
            "WARNING: some individuals have fewer than "
            f"{min_sessions} sessions. Data gap report written to {report_path}. "
            "Continuing because mvp_scope.enforce_min_sessions is false."
        )

    interim_dir = data_dir / "interim"
    interim_dir.mkdir(parents=True, exist_ok=True)
    validated_output_path = interim_dir / "recordings_validated.parquet"
    _write_parquet_compatible(filtered_df, validated_output_path)

    recordings_with_temp_df = merge_temperature(filtered_df, temperature_df)
    recordings_qc_df = apply_temperature_qc(recordings_with_temp_df, runtime_config)
    recordings_qc_output_path = interim_dir / "recordings_qc.parquet"
    save_recordings_qc(recordings_qc_df, str(recordings_qc_output_path))
    audio_index_df = build_audio_index(filtered_df, runtime_config)

    segments_df = run_segmentation(
        recordings_qc_df,
        runtime_config,
        project_root=project_root,
    )
    generate_segment_review(segments_df, runtime_config)

    features_df = run_feature_extraction(
        segments_df,
        runtime_config,
        project_root=project_root,
    )
    spectrogram_index_df = run_spectrogram_extraction(
        segments_df,
        runtime_config,
        project_root=project_root,
    )

    split_dict = build_session_splits(segments_df, runtime_config)
    split_issues = validate_split_integrity(split_dict, segments_df)
    for issue in split_issues:
        typer.echo(f"ISSUE: {issue}")
    if split_issues:
        sys.exit(1)
    split_path = artifacts_dir / "splits" / "split_v1.json"
    save_splits(split_dict, str(split_path))

    pca_results = train_pca_baseline_model(
        features_df,
        split_dict,
        runtime_config,
        project_root=project_root,
    )
    cnn_results = train_cnn_baseline_model(
        spectrogram_index_df,
        features_df,
        split_dict,
        runtime_config,
        project_root=project_root,
    )

    runtime_config["__runtime_cache__"] = {
        "recordings_qc_df": recordings_qc_df,
        "features_df": features_df,
        "spectrogram_index_df": spectrogram_index_df,
    }
    evaluation_results = run_evaluation(
        pca_results,
        cnn_results,
        segments_df,
        split_dict,
        runtime_config,
        project_root=project_root,
    )
    generate_mvp_report(
        evaluation_results,
        pca_results,
        cnn_results,
        segments_df,
        split_dict,
        runtime_config,
        project_root=project_root,
        recordings_qc_df=recordings_qc_df,
        features_df=features_df,
        spectrogram_index_df=spectrogram_index_df,
    )

    # Optional scientific extensions. Non-fatal: a dataset without two contexts
    # or without temperature must not break the core one-command run.
    if bool(runtime_config.get("cross_context", {}).get("run_in_mvp", True)):
        try:
            cross_context_results = run_cross_context_eval(
                features_df,
                filtered_df,
                runtime_config,
                project_root=project_root,
            )
            for direction in cross_context_results["directions"]:
                typer.echo(
                    f"cross-context {direction['label']}: "
                    f"within_macro_f1={_format_metric(direction['within_context'].get('macro_f1'))}, "
                    f"cross_macro_f1={_format_metric(direction['cross_context'].get('macro_f1'))}"
                )
        except ValueError as error:
            typer.echo(f"Cross-context evaluation skipped: {error}")

    try:
        residualization = run_temperature_residualization(
            features_df,
            runtime_config,
            project_root=project_root,
        )
        if residualization["status"] == "blocked":
            typer.echo(
                "Temperature residualization skipped (blocked): "
                f"{residualization['reason']}"
            )
    except Exception as error:  # pragma: no cover - defensive, never break run-mvp
        typer.echo(f"Temperature residualization skipped: {error}")

    if bool(runtime_config.get("cross_validation", {}).get("run_in_mvp", True)):
        try:
            cv_results = run_pca_cross_validation(
                features_df,
                runtime_config,
                project_root=project_root,
            )
            typer.echo(
                "PCA session-grouped CV (honest): "
                f"macro_f1={_format_metric(cv_results['pooled'].get('macro_f1'))} "
                f"on {cv_results['n_individuals_evaluated']}/{cv_results['n_individuals_total']} individuals"
            )
        except ValueError as error:
            typer.echo(f"Cross-validation skipped: {error}")

    if bool(runtime_config.get("embeddings", {}).get("run_in_mvp", True)):
        try:
            embedding_results = run_embedding_analysis(
                spectrogram_index_df,
                segments_df,
                runtime_config,
                project_root=project_root,
            )
            verification = embedding_results["verification"]
            if verification.get("status") == "ok":
                typer.echo(
                    "CNN embeddings: "
                    f"auc={_format_metric(verification['auc'])}, "
                    f"eer={_format_metric(verification['eer'])}"
                )
        except Exception as error:  # pragma: no cover - defensive, never break run-mvp
            typer.echo(f"Embedding analysis skipped: {error}")

    generate_overall_summary(runtime_config, project_root=project_root)

    typer.echo(
        "MVP pipeline complete: "
        f"recordings={len(filtered_df)}, "
        f"segments={len(segments_df)}, "
        f"features={len(features_df)}, "
        f"spectrograms={len(spectrogram_index_df)}, "
        f"audio_index={len(audio_index_df)}, "
        f"best_model={evaluation_results['best_model']}"
    )
