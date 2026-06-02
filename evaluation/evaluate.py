from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

from cricket_id.evaluation.statistics import compute_significance
from cricket_id.io.manifest import _read_parquet_compatible
from cricket_id.models.cnn_dataset import create_dataloaders
from cricket_id.models.cnn_model import CricketCNN, create_resnet18_model, stabilize_torch_backend
from cricket_id.models.pca_baseline import EXCLUDED_FEATURE_COLUMNS
from cricket_id.utils.paths import resolve_data_dir, resolve_project_root


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
    return _read_parquet_compatible(path)


def _load_pickle(path: Path) -> object:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _ordered_split_frame(frame_df: pd.DataFrame, split_dict: dict, split_name: str) -> pd.DataFrame:
    segment_ids = [str(segment_id) for segment_id in split_dict.get(split_name, [])]
    working_df = frame_df.copy()
    working_df["segment_id"] = working_df["segment_id"].astype(str)
    for column in ("individual_id", "session_id", "recording_id"):
        if column in working_df.columns:
            working_df[column] = working_df[column].astype(str)

    if not segment_ids:
        return working_df.iloc[0:0].copy()

    indexed_df = (
        working_df.drop_duplicates(subset=["segment_id"], keep="last")
        .set_index("segment_id", drop=False)
        .sort_index()
    )
    missing_segment_ids = [
        segment_id for segment_id in segment_ids if segment_id not in indexed_df.index
    ]
    if missing_segment_ids:
        raise ValueError(
            f"{split_name} references rows missing from the available frame: "
            + ", ".join(sorted(missing_segment_ids))
        )
    return indexed_df.loc[segment_ids].reset_index(drop=True).copy()


def _segments_lookup(segments_df: pd.DataFrame) -> pd.DataFrame:
    lookup_df = segments_df.copy()
    for column in ("segment_id", "recording_id", "session_id", "individual_id"):
        if column in lookup_df.columns:
            lookup_df[column] = lookup_df[column].astype(str)
    return lookup_df


def _majority_vote(values: list[object]) -> str:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _session_accuracy(prediction_df: pd.DataFrame) -> float | None:
    if prediction_df.empty:
        return None

    correct_sessions = 0
    total_sessions = 0
    for _, session_df in prediction_df.groupby("session_id", sort=True):
        predicted = _majority_vote(session_df["predicted_individual_id"].tolist())
        truth = _majority_vote(session_df["true_individual_id"].tolist())
        correct_sessions += int(predicted == truth)
        total_sessions += 1
    if total_sessions == 0:
        return None
    return float(correct_sessions / total_sessions)


def _score_predictions(prediction_df: pd.DataFrame) -> dict[str, float | None]:
    if prediction_df.empty:
        return {
            "accuracy": None,
            "balanced_accuracy": None,
            "macro_f1": None,
            "session_level_accuracy": None,
        }

    y_true = prediction_df["true_individual_id"].astype(str).to_numpy()
    y_pred = prediction_df["predicted_individual_id"].astype(str).to_numpy()
    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "session_level_accuracy": _session_accuracy(prediction_df),
    }


def _build_prediction_frame(
    base_df: pd.DataFrame,
    predicted_labels: list[str],
    split_name: str,
    model_name: str,
    segments_lookup_df: pd.DataFrame,
) -> pd.DataFrame:
    if base_df.empty:
        return pd.DataFrame(
            columns=[
                "segment_id",
                "recording_id",
                "session_id",
                "true_individual_id",
                "predicted_individual_id",
                "temperature_c",
                "snr_proxy",
                "split",
                "model",
                "correct",
            ]
        )

    prediction_df = base_df.loc[:, ["segment_id", "individual_id"]].copy()
    prediction_df = prediction_df.rename(columns={"individual_id": "true_individual_id"})
    prediction_df["predicted_individual_id"] = [str(label) for label in predicted_labels]
    prediction_df["split"] = split_name
    prediction_df["model"] = model_name

    merged_df = prediction_df.merge(
        segments_lookup_df.loc[
            :,
            ["segment_id", "recording_id", "session_id", "temperature_c", "snr_proxy"],
        ],
        on="segment_id",
        how="left",
        validate="one_to_one",
    )
    merged_df["correct"] = (
        merged_df["true_individual_id"].astype(str)
        == merged_df["predicted_individual_id"].astype(str)
    )
    merged_df["recording_id"] = merged_df["recording_id"].astype(str)
    merged_df["session_id"] = merged_df["session_id"].astype(str)
    return merged_df


def _pca_feature_columns(features_df: pd.DataFrame, pca_metrics: dict) -> list[str]:
    metric_columns = pca_metrics.get("feature_columns")
    if isinstance(metric_columns, list) and metric_columns:
        return [str(column) for column in metric_columns]

    return [
        column
        for column in features_df.columns
        if column not in EXCLUDED_FEATURE_COLUMNS
        and pd.api.types.is_numeric_dtype(features_df[column])
    ]


def _predict_with_pca(
    pca_metrics: dict,
    segments_df: pd.DataFrame,
    split_dict: dict,
    config: dict,
    *,
    project_root: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    data_dir = resolve_data_dir(config, project_root=project_root)
    artifacts_dir = _resolve_artifacts_dir(config, project_root=project_root)
    features_df = _load_frame(
        config,
        "features_df",
        data_dir / "processed" / "features_tabular.parquet",
    )
    features_df = features_df.copy()
    for column in ("segment_id", "individual_id", "session_id", "recording_id"):
        if column in features_df.columns:
            features_df[column] = features_df[column].astype(str)

    feature_columns = _pca_feature_columns(features_df, pca_metrics)
    scaler = _load_pickle(artifacts_dir / "models" / "scaler.pkl")
    pca = _load_pickle(artifacts_dir / "models" / "pca.pkl")
    classifier = _load_pickle(artifacts_dir / "models" / "pca_classifier.pkl")
    label_encoder = _load_pickle(artifacts_dir / "models" / "label_encoder.pkl")
    segments_lookup_df = _segments_lookup(segments_df)

    predictions: dict[str, pd.DataFrame] = {}
    for split_name in ("val", "test"):
        split_df = _ordered_split_frame(features_df, split_dict, split_name)
        if split_df.empty:
            predictions[split_name] = _build_prediction_frame(
                split_df,
                [],
                split_name,
                "pca",
                segments_lookup_df,
            )
            continue

        features = split_df.loc[:, feature_columns].to_numpy(dtype=np.float64, copy=True)
        transformed = pca.transform(scaler.transform(features))
        predicted_indices = classifier.predict(transformed)
        predicted_labels = label_encoder.inverse_transform(predicted_indices).tolist()
        predictions[split_name] = _build_prediction_frame(
            split_df,
            predicted_labels,
            split_name,
            "pca",
            segments_lookup_df,
        )

    return predictions


def _load_cnn_model(checkpoint_path: Path, dropout: float, num_classes: int) -> tuple[torch.nn.Module, list[str]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_name = str(checkpoint.get("model_name", "custom")).strip().lower()
    resolved_dropout = float(checkpoint.get("dropout", dropout))
    resolved_num_classes = int(checkpoint.get("num_classes", num_classes))
    if model_name == "custom":
        model = CricketCNN(num_classes=resolved_num_classes, dropout=resolved_dropout)
    elif model_name == "resnet18":
        model = create_resnet18_model(
            num_classes=resolved_num_classes,
            dropout=resolved_dropout,
        )
    else:
        raise ValueError(f"Unsupported CNN model architecture: {model_name}")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    label_classes = [str(label) for label in checkpoint.get("label_classes", [])]
    return model, label_classes


def _predict_with_cnn(
    cnn_metrics: dict,
    segments_df: pd.DataFrame,
    split_dict: dict,
    config: dict,
    *,
    project_root: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    data_dir = resolve_data_dir(config, project_root=project_root)
    artifacts_dir = _resolve_artifacts_dir(config, project_root=project_root)
    spectrogram_index_df = _load_frame(
        config,
        "spectrogram_index_df",
        data_dir / "processed" / "spectrogram_index.parquet",
    )
    spectrogram_index_df = spectrogram_index_df.copy()
    for column in ("segment_id", "individual_id", "session_id"):
        if column in spectrogram_index_df.columns:
            spectrogram_index_df[column] = spectrogram_index_df[column].astype(str)

    checkpoint_path = artifacts_dir / "models" / "cnn_baseline.pt"
    model, label_classes = _load_cnn_model(
        checkpoint_path,
        dropout=float(config.get("cnn", {}).get("dropout", 0.3)),
        num_classes=int(cnn_metrics.get("num_classes", 0) or 0),
    )
    label_encoder = LabelEncoder()
    label_encoder.fit(label_classes)
    dataloaders = create_dataloaders(
        spectrogram_index_df,
        split_dict,
        label_encoder,
        data_dir,
        config,
    )
    segments_lookup_df = _segments_lookup(segments_df)

    predictions: dict[str, pd.DataFrame] = {}
    for split_name in ("val", "test"):
        dataloader = dataloaders[split_name]
        predicted_indices: list[int] = []
        true_indices: list[int] = []
        with torch.no_grad():
            for inputs, labels in dataloader:
                logits = model(inputs)
                predicted_indices.extend(
                    torch.argmax(logits, dim=1).cpu().numpy().astype(int).tolist()
                )
                true_indices.extend(labels.cpu().numpy().astype(int).tolist())

        split_df = _ordered_split_frame(spectrogram_index_df, split_dict, split_name)
        if split_df.empty:
            predictions[split_name] = _build_prediction_frame(
                split_df,
                [],
                split_name,
                "cnn",
                segments_lookup_df,
            )
            continue

        predicted_labels = label_encoder.inverse_transform(
            np.asarray(predicted_indices, dtype=np.int64)
        ).tolist()
        true_labels = label_encoder.inverse_transform(
            np.asarray(true_indices, dtype=np.int64)
        ).tolist()
        if true_labels != split_df["individual_id"].astype(str).tolist():
            raise AssertionError("CNN dataloader label order does not match split metadata.")

        predictions[split_name] = _build_prediction_frame(
            split_df,
            predicted_labels,
            split_name,
            "cnn",
            segments_lookup_df,
        )

    return predictions


def _off_diagonal_confusions(prediction_df: pd.DataFrame, limit: int = 5) -> list[dict[str, object]]:
    if prediction_df.empty:
        return []

    confusion_df = (
        prediction_df.loc[
            ~prediction_df["correct"],
            ["true_individual_id", "predicted_individual_id"],
        ]
        .value_counts()
        .reset_index(name="count")
        .sort_values(
            ["count", "true_individual_id", "predicted_individual_id"],
            ascending=[False, True, True],
        )
        .head(limit)
    )
    return [
        {
            "true_individual_id": str(row.true_individual_id),
            "predicted_individual_id": str(row.predicted_individual_id),
            "count": int(row.count),
        }
        for row in confusion_df.itertuples(index=False)
    ]


def _session_error_summary(prediction_df: pd.DataFrame, limit: int = 5) -> list[dict[str, object]]:
    if prediction_df.empty:
        return []

    session_df = (
        prediction_df.groupby(["session_id", "true_individual_id"], dropna=False)
        .agg(
            segment_count=("segment_id", "size"),
            segment_accuracy=("correct", "mean"),
            mean_temperature_c=("temperature_c", "mean"),
            mean_snr_proxy=("snr_proxy", "mean"),
        )
        .reset_index()
        .sort_values(
            ["segment_accuracy", "segment_count", "session_id"],
            ascending=[True, False, True],
        )
    )
    session_df = session_df.loc[session_df["segment_accuracy"] < 1.0]

    return [
        {
            "session_id": str(row.session_id),
            "individual_id": str(row.true_individual_id),
            "segment_count": int(row.segment_count),
            "segment_accuracy": float(row.segment_accuracy),
            "mean_temperature_c": None
            if pd.isna(row.mean_temperature_c)
            else float(row.mean_temperature_c),
            "mean_snr_proxy": None if pd.isna(row.mean_snr_proxy) else float(row.mean_snr_proxy),
        }
        for row in session_df.head(limit).itertuples(index=False)
    ]


def _binned_accuracy(prediction_df: pd.DataFrame, column_name: str) -> dict[str, object]:
    if prediction_df.empty:
        return {"status": "no_samples", "bins": []}

    working_df = prediction_df.loc[:, [column_name, "correct"]].dropna().copy()
    if working_df.empty:
        return {"status": "missing_values", "bins": []}

    unique_values = int(working_df[column_name].nunique())
    if unique_values < 2:
        return {"status": "insufficient_variation", "bins": []}

    quantiles = min(3, unique_values)
    try:
        working_df["band"] = pd.qcut(
            working_df[column_name],
            q=quantiles,
            duplicates="drop",
        )
    except ValueError:
        return {"status": "insufficient_variation", "bins": []}

    grouped_df = (
        working_df.groupby("band", observed=False)
        .agg(
            sample_count=("correct", "size"),
            accuracy=("correct", "mean"),
            value_min=(column_name, "min"),
            value_max=(column_name, "max"),
        )
        .reset_index(drop=True)
    )
    return {
        "status": "ok",
        "bins": [
            {
                "sample_count": int(row.sample_count),
                "accuracy": float(row.accuracy),
                "value_min": float(row.value_min),
                "value_max": float(row.value_max),
            }
            for row in grouped_df.itertuples(index=False)
        ],
    }


def _error_analysis(predictions_by_split: dict[str, pd.DataFrame]) -> dict[str, object]:
    analysis: dict[str, object] = {}
    for split_name, prediction_df in predictions_by_split.items():
        analysis[split_name] = {
            "top_confusions": _off_diagonal_confusions(prediction_df),
            "session_specific_errors": _session_error_summary(prediction_df),
            "temperature_effect": _binned_accuracy(prediction_df, "temperature_c"),
            "snr_effect": _binned_accuracy(prediction_df, "snr_proxy"),
        }
    return analysis


def _metric_delta(cnn_value: object, pca_value: object) -> float | None:
    if cnn_value is None or pca_value is None:
        return None
    return float(cnn_value) - float(pca_value)


def _comparison_block(pca_metrics: dict, cnn_metrics: dict) -> dict[str, object]:
    comparison: dict[str, object] = {}
    for split_name in ("val", "test"):
        pca_split = pca_metrics.get("metrics", {}).get(split_name, {})
        cnn_split = cnn_metrics.get("metrics", {}).get(split_name, {})
        split_comparison: dict[str, object] = {}
        for metric_name in (
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "session_level_accuracy",
        ):
            pca_value = pca_split.get(metric_name)
            cnn_value = cnn_split.get(metric_name)
            if pca_value is None and cnn_value is None:
                winner = "tie"
            elif pca_value is None:
                winner = "cnn"
            elif cnn_value is None:
                winner = "pca"
            elif float(cnn_value) > float(pca_value):
                winner = "cnn"
            elif float(cnn_value) < float(pca_value):
                winner = "pca"
            else:
                winner = "tie"
            split_comparison[metric_name] = {
                "pca": pca_value,
                "cnn": cnn_value,
                "delta_cnn_minus_pca": _metric_delta(cnn_value, pca_value),
                "winner": winner,
            }
        comparison[split_name] = split_comparison
    return comparison


def _majority_class_baseline(segments_df: pd.DataFrame, split_dict: dict) -> dict[str, object]:
    working_df = _segments_lookup(segments_df)
    train_df = _ordered_split_frame(working_df, split_dict, "train")
    if train_df.empty:
        return {"majority_individual_id": None, "metrics": {}}

    majority_individual = str(
        train_df["individual_id"].astype(str).value_counts().sort_index().idxmax()
    )
    baseline_metrics: dict[str, object] = {}
    for split_name in ("val", "test"):
        split_df = _ordered_split_frame(working_df, split_dict, split_name)
        if split_df.empty:
            baseline_metrics[split_name] = {
                "accuracy": None,
                "balanced_accuracy": None,
                "macro_f1": None,
                "session_level_accuracy": None,
            }
            continue

        prediction_df = split_df.loc[:, ["segment_id", "session_id", "recording_id"]].copy()
        prediction_df["true_individual_id"] = split_df["individual_id"].astype(str).tolist()
        prediction_df["predicted_individual_id"] = majority_individual
        prediction_df["temperature_c"] = split_df.get("temperature_c")
        prediction_df["snr_proxy"] = split_df.get("snr_proxy")
        prediction_df["correct"] = (
            prediction_df["true_individual_id"] == prediction_df["predicted_individual_id"]
        )
        baseline_metrics[split_name] = _score_predictions(prediction_df)

    return {
        "majority_individual_id": majority_individual,
        "metrics": baseline_metrics,
    }


def _best_model(comparison: dict[str, object]) -> str:
    test_metrics = comparison.get("test", {})
    session_comparison = test_metrics.get("session_level_accuracy", {})
    winner = session_comparison.get("winner")
    if winner and winner != "tie":
        return str(winner)
    macro_f1_comparison = test_metrics.get("macro_f1", {})
    winner = macro_f1_comparison.get("winner")
    return "tie" if winner is None else str(winner)


def run_evaluation(
    pca_metrics: dict,
    cnn_metrics: dict,
    segments_df: pd.DataFrame,
    split_dict: dict,
    config: dict,
    project_root: str | Path | None = None,
) -> dict:
    project_root_path = resolve_project_root(config, project_root=project_root)
    artifacts_dir = _resolve_artifacts_dir(config, project_root=project_root_path)
    metrics_dir = artifacts_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Avoid the intermittent Windows MaxPool2d crash during CNN inference.
    stabilize_torch_backend()
    if config.get("cnn", {}).get("torch_num_threads") in (None, ""):
        torch.set_num_threads(1)

    working_segments_df = _segments_lookup(segments_df)
    pca_predictions = _predict_with_pca(
        pca_metrics,
        working_segments_df,
        split_dict,
        config,
        project_root=project_root_path,
    )
    cnn_predictions = _predict_with_cnn(
        cnn_metrics,
        working_segments_df,
        split_dict,
        config,
        project_root=project_root_path,
    )

    num_individuals = int(working_segments_df["individual_id"].astype(str).nunique())
    uniform_chance = None if num_individuals <= 0 else float(1.0 / num_individuals)
    comparison = _comparison_block(pca_metrics, cnn_metrics)

    stats_config = config.get("statistics", {}) or {}
    significance: dict[str, object] = {}
    if bool(stats_config.get("enabled", True)):
        bootstrap_iterations = int(stats_config.get("bootstrap_iterations", 2000))
        permutation_iterations = int(stats_config.get("permutation_iterations", 1000))
        alpha = float(stats_config.get("alpha", 0.05))
        seed = int(config.get("seed", 0))
        splits = stats_config.get("splits", ["test"])
        for model_name, predictions in (("pca", pca_predictions), ("cnn", cnn_predictions)):
            model_block: dict[str, object] = {}
            for split_name in splits:
                prediction_df = predictions.get(split_name)
                if prediction_df is None or prediction_df.empty:
                    continue
                model_block[split_name] = compute_significance(
                    prediction_df,
                    bootstrap_iterations=bootstrap_iterations,
                    permutation_iterations=permutation_iterations,
                    alpha=alpha,
                    seed=seed,
                )
            if model_block:
                significance[model_name] = model_block

    evaluation_payload = {
        "num_individuals": num_individuals,
        "chance_baseline": {
            "uniform_random": {
                "accuracy": uniform_chance,
                "balanced_accuracy": uniform_chance,
                "macro_f1": uniform_chance,
                "session_level_accuracy": uniform_chance,
            },
            "majority_class": _majority_class_baseline(working_segments_df, split_dict),
        },
        "comparison": comparison,
        "best_model": _best_model(comparison),
        "significance": significance,
        "error_analysis": {
            "pca": _error_analysis(pca_predictions),
            "cnn": _error_analysis(cnn_predictions),
        },
        "artifacts": {
            "evaluation_metrics": str(metrics_dir / "evaluation_comparison.json"),
        },
    }

    output_path = metrics_dir / "evaluation_comparison.json"
    output_path.write_text(
        json.dumps(evaluation_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    return evaluation_payload
