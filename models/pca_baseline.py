from __future__ import annotations

import json
import os
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix as sklearn_confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import LabelEncoder, StandardScaler

from cricket_id.utils.paths import resolve_project_root


REQUIRED_COLUMNS = {"segment_id", "individual_id", "session_id", "recording_id"}
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


def _majority_vote(values: list[object]) -> str:
    counts = Counter(str(value) for value in values)
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _compute_session_level_accuracy(
    predictions,
    segment_ids,
    segments_df: pd.DataFrame,
) -> float:
    prediction_df = pd.DataFrame(
        {
            "segment_id": [str(segment_id) for segment_id in segment_ids],
            "predicted_individual_id": [str(prediction) for prediction in predictions],
        }
    )
    if prediction_df.empty:
        return 0.0

    lookup_df = segments_df.loc[:, ["segment_id", "session_id", "individual_id"]].copy()
    lookup_df["segment_id"] = lookup_df["segment_id"].astype(str)
    lookup_df["session_id"] = lookup_df["session_id"].astype(str)
    lookup_df["individual_id"] = lookup_df["individual_id"].astype(str)

    merged_df = prediction_df.merge(
        lookup_df,
        on="segment_id",
        how="left",
        validate="one_to_one",
    )
    if merged_df[["session_id", "individual_id"]].isna().any().any():
        missing_segments = merged_df.loc[
            merged_df["session_id"].isna(), "segment_id"
        ].tolist()
        raise ValueError(
            "Missing session metadata for segment_ids: " + ", ".join(sorted(missing_segments))
        )

    correct_sessions = 0
    total_sessions = 0
    for _, session_df in merged_df.groupby("session_id", sort=True):
        predicted_individual = _majority_vote(
            session_df["predicted_individual_id"].tolist()
        )
        true_individual = _majority_vote(session_df["individual_id"].tolist())
        correct_sessions += int(predicted_individual == true_individual)
        total_sessions += 1

    return float(correct_sessions / total_sessions) if total_sessions else 0.0


def _feature_columns(features_df: pd.DataFrame) -> list[str]:
    return [
        column
        for column in features_df.columns
        if column not in EXCLUDED_FEATURE_COLUMNS
        and pd.api.types.is_numeric_dtype(features_df[column])
    ]


def _split_frames(
    features_df: pd.DataFrame,
    split_dict: dict,
) -> dict[str, pd.DataFrame]:
    working_df = features_df.copy()
    working_df["segment_id"] = working_df["segment_id"].astype(str)
    frames: dict[str, pd.DataFrame] = {}
    for split_name in ("train", "val", "test"):
        split_segment_ids = set(map(str, split_dict.get(split_name, [])))
        frames[split_name] = working_df.loc[
            working_df["segment_id"].isin(split_segment_ids)
        ].copy()
    return frames


def _transform_split(
    frame_df: pd.DataFrame,
    feature_columns: list[str],
    scaler: StandardScaler,
    pca: PCA,
) -> np.ndarray:
    if frame_df.empty:
        return np.empty((0, pca.n_components_), dtype=np.float64)

    features = frame_df.loc[:, feature_columns].to_numpy(dtype=np.float64, copy=True)
    scaled = scaler.transform(features)
    return pca.transform(scaled)


def _evaluate_split(
    split_name: str,
    split_df: pd.DataFrame,
    transformed: np.ndarray,
    classifier,
    label_encoder: LabelEncoder,
) -> dict[str, object]:
    labels = np.arange(len(label_encoder.classes_))
    zero_confusion = np.zeros((len(labels), len(labels)), dtype=int)
    zero_recall = {str(label): 0.0 for label in label_encoder.classes_}

    if split_df.empty:
        return {
            "split": split_name,
            "sample_count": 0,
            "session_count": 0,
            "accuracy": None,
            "balanced_accuracy": None,
            "macro_f1": None,
            "per_class_recall": zero_recall,
            "confusion_matrix": zero_confusion.tolist(),
            "session_level_accuracy": None,
            "predicted_labels": [],
            "true_labels": [],
        }

    true_labels_str = split_df["individual_id"].astype(str).to_numpy()
    true_labels = label_encoder.transform(true_labels_str)
    predicted_labels = classifier.predict(transformed)
    predicted_labels_str = label_encoder.inverse_transform(predicted_labels).tolist()

    confusion = sklearn_confusion_matrix(true_labels, predicted_labels, labels=labels)
    recall_values = recall_score(
        true_labels,
        predicted_labels,
        labels=labels,
        average=None,
        zero_division=0,
    )

    session_level_accuracy = _compute_session_level_accuracy(
        predicted_labels_str,
        split_df["segment_id"].astype(str).tolist(),
        split_df,
    )

    return {
        "split": split_name,
        "sample_count": int(len(split_df)),
        "session_count": int(split_df["session_id"].nunique()),
        "accuracy": float(accuracy_score(true_labels, predicted_labels)),
        "balanced_accuracy": float(
            balanced_accuracy_score(true_labels, predicted_labels)
        ),
        "macro_f1": float(
            f1_score(true_labels, predicted_labels, average="macro", zero_division=0)
        ),
        "per_class_recall": {
            str(label): float(recall)
            for label, recall in zip(label_encoder.classes_, recall_values, strict=True)
        },
        "confusion_matrix": confusion.astype(int).tolist(),
        "session_level_accuracy": float(session_level_accuracy),
        "predicted_labels": predicted_labels_str,
        "true_labels": true_labels_str.tolist(),
    }


def _projection_frame(
    split_frames: dict[str, pd.DataFrame],
    transformed_splits: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for split_name, frame_df in split_frames.items():
        transformed = transformed_splits[split_name]
        if frame_df.empty:
            continue

        pc1 = transformed[:, 0] if transformed.shape[1] >= 1 else np.zeros(len(frame_df))
        pc2 = transformed[:, 1] if transformed.shape[1] >= 2 else np.zeros(len(frame_df))
        for row_index, row in enumerate(frame_df.itertuples(index=False)):
            rows.append(
                {
                    "split": split_name,
                    "pc1": float(pc1[row_index]),
                    "pc2": float(pc2[row_index]),
                    "individual_id": str(row.individual_id),
                    "session_id": str(row.session_id),
                }
            )
    return pd.DataFrame(rows)


def _plot_explained_variance(pca: PCA, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    component_indices = np.arange(1, len(explained) + 1)
    ax.bar(component_indices, explained, alpha=0.7, label="Explained variance")
    ax.plot(
        component_indices,
        cumulative,
        color="black",
        marker="o",
        linewidth=1.5,
        label="Cumulative variance",
    )
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Variance ratio")
    ax.set_title("PCA explained variance")
    ax.set_ylim(0.0, 1.05)
    ax.legend()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_scatter(
    projection_df: pd.DataFrame,
    color_column: str,
    output_path: Path,
    title: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    if projection_df.empty:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
        ax.set_axis_off()
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        return

    unique_values = sorted(projection_df[color_column].astype(str).unique().tolist())
    color_map = plt.get_cmap("tab20", max(len(unique_values), 1))
    color_lookup = {
        value: color_map(index % max(len(unique_values), 1))
        for index, value in enumerate(unique_values)
    }

    for value, group_df in projection_df.groupby(color_column, sort=True):
        ax.scatter(
            group_df["pc1"],
            group_df["pc2"],
            s=26,
            alpha=0.85,
            label=str(value),
            color=color_lookup[str(value)],
            edgecolors="none",
        )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    if len(unique_values) <= 20:
        ax.legend(loc="best", fontsize=8, frameon=False)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_confusion_matrix(
    metrics_by_split: dict[str, dict[str, object]],
    label_encoder: LabelEncoder,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    selected_split = "test"
    if metrics_by_split["test"]["sample_count"] == 0:
        selected_split = "val" if metrics_by_split["val"]["sample_count"] > 0 else "train"

    confusion = np.asarray(metrics_by_split[selected_split]["confusion_matrix"], dtype=int)
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    display = ConfusionMatrixDisplay(
        confusion_matrix=confusion,
        display_labels=label_encoder.classes_,
    )
    display.plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation=45)
    ax.set_title(f"PCA confusion matrix ({selected_split})")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _save_pickle(obj: object, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(obj, handle)


def train_pca_baseline(
    features_df: pd.DataFrame,
    split_dict: dict,
    config: dict,
    project_root: str | Path | None = None,
) -> dict:
    missing_columns = REQUIRED_COLUMNS.difference(features_df.columns)
    if missing_columns:
        raise ValueError(
            "features_df is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    working_df = features_df.copy()
    working_df["segment_id"] = working_df["segment_id"].astype(str)
    working_df["individual_id"] = working_df["individual_id"].astype(str)
    working_df["session_id"] = working_df["session_id"].astype(str)
    working_df["recording_id"] = working_df["recording_id"].astype(str)

    feature_columns = _feature_columns(working_df)
    if not feature_columns:
        raise ValueError("No numeric feature columns were found for PCA training.")

    split_frames = _split_frames(working_df, split_dict)
    train_df = split_frames["train"]
    if train_df.empty:
        raise ValueError("Train split is empty; cannot train PCA baseline.")

    train_labels = train_df["individual_id"].astype(str).to_numpy()
    if len(set(train_labels.tolist())) < 2:
        raise ValueError("Train split must contain at least two individuals.")

    label_encoder = LabelEncoder()
    label_encoder.fit(train_labels)
    known_labels = set(label_encoder.classes_.tolist())
    for split_name in ("val", "test"):
        split_labels = set(split_frames[split_name]["individual_id"].astype(str).tolist())
        unseen_labels = split_labels.difference(known_labels)
        if unseen_labels:
            raise ValueError(
                f"{split_name} contains individuals not present in train: "
                + ", ".join(sorted(unseen_labels))
            )

    seed = int(config.get("seed", 0))
    requested_components = int(config.get("pca", {}).get("n_components", 30))
    classifier_name = str(
        config.get("pca", {}).get("classifier", "logistic_regression")
    ).strip()
    class_weight = config.get("pca", {}).get("class_weight")
    if isinstance(class_weight, str):
        class_weight = class_weight.strip() or None

    train_features = train_df.loc[:, feature_columns].to_numpy(dtype=np.float64, copy=True)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_features)

    effective_components = min(
        requested_components,
        train_scaled.shape[0],
        train_scaled.shape[1],
    )
    if effective_components < 1:
        raise ValueError("Effective PCA component count must be at least 1.")

    pca = PCA(n_components=effective_components)
    train_transformed = pca.fit_transform(train_scaled)

    if classifier_name == "logistic_regression":
        classifier = LogisticRegression(
            max_iter=2000,
            solver="lbfgs",
            random_state=seed,
            class_weight=class_weight,
        )
    elif classifier_name == "nearest_centroid":
        classifier = NearestCentroid()
    else:
        raise ValueError(
            "Unsupported PCA baseline classifier: "
            f"{classifier_name}. Expected logistic_regression or nearest_centroid."
        )

    classifier.fit(train_transformed, label_encoder.transform(train_labels))

    transformed_splits = {"train": train_transformed}
    for split_name in ("val", "test"):
        transformed_splits[split_name] = _transform_split(
            split_frames[split_name],
            feature_columns,
            scaler,
            pca,
        )

    metrics_by_split: dict[str, dict[str, object]] = {}
    for split_name in ("train", "val", "test"):
        metrics_by_split[split_name] = _evaluate_split(
            split_name,
            split_frames[split_name],
            transformed_splits[split_name],
            classifier,
            label_encoder,
        )

    projection_df = _projection_frame(split_frames, transformed_splits)

    artifacts_dir = _resolve_artifacts_dir(config, project_root=project_root)
    model_dir = artifacts_dir / "models"
    metrics_dir = artifacts_dir / "metrics"
    figures_dir = artifacts_dir / "figures"

    scaler_path = model_dir / "scaler.pkl"
    pca_path = model_dir / "pca.pkl"
    classifier_path = model_dir / "pca_classifier.pkl"
    label_encoder_path = model_dir / "label_encoder.pkl"
    metrics_path = metrics_dir / "pca_metrics.json"
    explained_variance_path = figures_dir / "pca_explained_variance.png"
    scatter_individual_path = figures_dir / "pca_scatter_individual.png"
    scatter_session_path = figures_dir / "pca_scatter_session.png"
    confusion_matrix_path = figures_dir / "pca_confusion_matrix.png"

    _save_pickle(scaler, scaler_path)
    _save_pickle(pca, pca_path)
    _save_pickle(classifier, classifier_path)
    _save_pickle(label_encoder, label_encoder_path)

    _plot_explained_variance(pca, explained_variance_path)
    _plot_scatter(
        projection_df,
        "individual_id",
        scatter_individual_path,
        "PCA projection by individual",
    )
    _plot_scatter(
        projection_df,
        "session_id",
        scatter_session_path,
        "PCA projection by session",
    )
    _plot_confusion_matrix(metrics_by_split, label_encoder, confusion_matrix_path)

    metrics_payload = {
        "model": "pca_baseline",
        "classifier": classifier_name,
        "class_weight": class_weight,
        "seed": seed,
        "feature_columns": feature_columns,
        "requested_n_components": requested_components,
        "effective_n_components": int(effective_components),
        "explained_variance_ratio": [float(value) for value in pca.explained_variance_ratio_],
        "metrics": {
            split_name: {
                key: value
                for key, value in split_metrics.items()
                if key not in {"predicted_labels", "true_labels"}
            }
            for split_name, split_metrics in metrics_by_split.items()
            if split_name in {"val", "test"}
        },
    }

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics_payload, indent=2) + "\n", encoding="utf-8")

    return {
        "metrics": metrics_payload["metrics"],
        "artifact_paths": {
            "scaler": str(scaler_path),
            "pca": str(pca_path),
            "classifier": str(classifier_path),
            "label_encoder": str(label_encoder_path),
            "metrics": str(metrics_path),
            "explained_variance_figure": str(explained_variance_path),
            "individual_scatter_figure": str(scatter_individual_path),
            "session_scatter_figure": str(scatter_session_path),
            "confusion_matrix_figure": str(confusion_matrix_path),
        },
        "effective_n_components": int(effective_components),
        "feature_columns": feature_columns,
        "classifier": classifier_name,
    }
