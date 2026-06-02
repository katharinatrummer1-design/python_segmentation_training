from __future__ import annotations

import json
import os
import random
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix as sklearn_confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.preprocessing import LabelEncoder
from torch import nn

from cricket_id.models.cnn_dataset import create_dataloaders
from cricket_id.models.cnn_model import CricketCNN, create_resnet18_model, stabilize_torch_backend
from cricket_id.utils.paths import resolve_data_dir, resolve_project_root


REQUIRED_SPECTROGRAM_COLUMNS = {"segment_id", "individual_id", "session_id"}
REQUIRED_METADATA_COLUMNS = {"segment_id", "individual_id", "session_id"}


def _resolve_artifacts_dir(
    config: dict,
    *,
    project_root: str | Path | None = None,
) -> Path:
    root = resolve_project_root(config, project_root=project_root)
    artifacts_dir = Path(str(config.get("paths", {}).get("artifacts_dir", "artifacts")))
    return artifacts_dir if artifacts_dir.is_absolute() else (root / artifacts_dir).resolve()


def _configure_torch_runtime(cnn_config: dict) -> None:
    thread_count = cnn_config.get("torch_num_threads")
    if thread_count not in (None, ""):
        torch.set_num_threads(max(1, int(thread_count)))

    interop_thread_count = cnn_config.get("torch_num_interop_threads")
    if interop_thread_count not in (None, ""):
        try:
            torch.set_num_interop_threads(max(1, int(interop_thread_count)))
        except RuntimeError:
            pass


def _select_device(cnn_config: dict) -> str:
    requested_device = str(cnn_config.get("device", "auto")).strip().lower()
    if requested_device == "cpu":
        return "cpu"
    if requested_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CNN config requested CUDA, but torch.cuda.is_available() is false.")
    if requested_device.startswith("cuda") or requested_device == "mps":
        return requested_device
    raise ValueError(
        f"Unsupported CNN device: {requested_device}. Expected auto, cpu, cuda, or cuda:<index>."
    )


def _majority_vote(values: list[object]) -> str:
    counts = Counter(str(value) for value in values)
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _split_frame(
    frame_df: pd.DataFrame,
    split_dict: dict,
    split_name: str,
    *,
    required_columns: set[str],
    frame_name: str,
) -> pd.DataFrame:
    missing_columns = required_columns.difference(frame_df.columns)
    if missing_columns:
        raise ValueError(
            f"{frame_name} is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    segment_ids = [str(segment_id) for segment_id in split_dict.get(split_name, [])]
    working_df = frame_df.copy()
    working_df["segment_id"] = working_df["segment_id"].astype(str)
    if "individual_id" in working_df.columns:
        working_df["individual_id"] = working_df["individual_id"].astype(str)
    if "session_id" in working_df.columns:
        working_df["session_id"] = working_df["session_id"].astype(str)

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
            f"{split_name} references rows missing from {frame_name}: "
            + ", ".join(sorted(missing_segment_ids))
        )

    return indexed_df.loc[segment_ids].reset_index(drop=True).copy()


def _segment_metadata_frame(
    segment_ids: list[str],
    segments_df: pd.DataFrame,
) -> pd.DataFrame:
    missing_columns = REQUIRED_METADATA_COLUMNS.difference(segments_df.columns)
    if missing_columns:
        raise ValueError(
            "segments_df is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    if not segment_ids:
        return pd.DataFrame(columns=["segment_id", "session_id", "individual_id"])

    lookup_df = segments_df.loc[:, ["segment_id", "session_id", "individual_id"]].copy()
    lookup_df["segment_id"] = lookup_df["segment_id"].astype(str)
    lookup_df["session_id"] = lookup_df["session_id"].astype(str)
    lookup_df["individual_id"] = lookup_df["individual_id"].astype(str)
    indexed_df = (
        lookup_df.drop_duplicates(subset=["segment_id"], keep="last")
        .set_index("segment_id", drop=False)
        .sort_index()
    )

    missing_segment_ids = [
        segment_id for segment_id in segment_ids if segment_id not in indexed_df.index
    ]
    if missing_segment_ids:
        raise ValueError(
            "Missing session metadata for segment_ids: "
            + ", ".join(sorted(missing_segment_ids))
        )

    return indexed_df.loc[segment_ids].reset_index(drop=True).copy()


def _metric_value(value: object, fallback: float) -> float:
    return fallback if value is None else float(value)


def _balanced_accuracy(true_array: np.ndarray, predicted_array: np.ndarray) -> float:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="y_pred contains classes not in y_true",
            category=UserWarning,
        )
        return float(balanced_accuracy_score(true_array, predicted_array))


def _compute_epoch_loss(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: str,
) -> float | None:
    dataset_size = len(dataloader.dataset)
    if dataset_size == 0:
        return None

    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = model(inputs)
            loss = criterion(logits, labels)
            total_loss += float(loss.item()) * int(labels.size(0))

    return float(total_loss / dataset_size)


def _plot_training_history(history: dict[str, list[float | None]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)

    axes[0].plot(epochs, history["train_loss"], label="train_loss", linewidth=1.8)
    axes[0].plot(
        epochs,
        [np.nan if value is None else value for value in history["val_loss"]],
        label="val_loss",
        linewidth=1.8,
    )
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("CNN loss")
    axes[0].legend(frameon=False)

    axes[1].plot(
        epochs,
        [np.nan if value is None else value for value in history["train_macro_f1"]],
        label="train_macro_f1",
        linewidth=1.8,
    )
    axes[1].plot(
        epochs,
        [np.nan if value is None else value for value in history["val_macro_f1"]],
        label="val_macro_f1",
        linewidth=1.8,
    )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro F1")
    axes[1].set_title("CNN macro F1")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].legend(frameon=False)

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
        selected_split = "val"

    confusion = np.asarray(metrics_by_split[selected_split]["confusion_matrix"], dtype=int)
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    display = ConfusionMatrixDisplay(
        confusion_matrix=confusion,
        display_labels=label_encoder.classes_,
    )
    display.plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation=45)
    ax.set_title(f"CNN confusion matrix ({selected_split})")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _metrics_payload(metrics: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in metrics.items()
        if key not in {"predicted_labels", "true_labels", "segment_ids"}
    }


def _evaluate_model(
    model: nn.Module,
    dataloader,
    label_encoder: LabelEncoder,
    segments_df: pd.DataFrame,
    device: str,
) -> dict:
    label_indices = np.arange(len(label_encoder.classes_))
    zero_confusion = np.zeros((len(label_indices), len(label_indices)), dtype=int)
    zero_recall = {str(label): 0.0 for label in label_encoder.classes_}

    dataset = dataloader.dataset
    segment_ids = [str(segment_id) for segment_id in getattr(dataset, "segment_ids", [])]
    if len(dataset) == 0:
        return {
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
            "segment_ids": [],
        }

    model.eval()
    predicted_indices: list[int] = []
    true_indices: list[int] = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = model(inputs)
            predicted_indices.extend(
                torch.argmax(logits, dim=1).cpu().numpy().astype(int).tolist()
            )
            true_indices.extend(labels.cpu().numpy().astype(int).tolist())

    predicted_array = np.asarray(predicted_indices, dtype=np.int64)
    true_array = np.asarray(true_indices, dtype=np.int64)
    predicted_labels = label_encoder.inverse_transform(predicted_array).tolist()
    true_labels = label_encoder.inverse_transform(true_array).tolist()
    confusion = sklearn_confusion_matrix(
        true_array,
        predicted_array,
        labels=label_indices,
    )
    recall_values = recall_score(
        true_array,
        predicted_array,
        labels=label_indices,
        average=None,
        zero_division=0,
    )

    metadata_df = _segment_metadata_frame(segment_ids, segments_df)
    session_level_accuracy = _compute_session_level_accuracy(
        predicted_labels,
        segment_ids,
        segments_df,
    )

    return {
        "sample_count": int(true_array.size),
        "session_count": int(metadata_df["session_id"].nunique()),
        "accuracy": float(accuracy_score(true_array, predicted_array)),
        "balanced_accuracy": _balanced_accuracy(true_array, predicted_array),
        "macro_f1": float(
            f1_score(true_array, predicted_array, average="macro", zero_division=0)
        ),
        "per_class_recall": {
            str(label): float(recall)
            for label, recall in zip(label_encoder.classes_, recall_values, strict=True)
        },
        "confusion_matrix": confusion.astype(int).tolist(),
        "session_level_accuracy": float(session_level_accuracy),
        "predicted_labels": predicted_labels,
        "true_labels": true_labels,
        "segment_ids": segment_ids,
    }


def _compute_session_level_accuracy(
    predictions,
    segment_ids,
    segments_df,
) -> float:
    prediction_df = pd.DataFrame(
        {
            "segment_id": [str(segment_id) for segment_id in segment_ids],
            "predicted_individual_id": [str(prediction) for prediction in predictions],
        }
    )
    if prediction_df.empty:
        return 0.0

    lookup_df = _segment_metadata_frame(prediction_df["segment_id"].tolist(), segments_df)
    merged_df = prediction_df.merge(
        lookup_df.loc[:, ["segment_id", "session_id", "individual_id"]],
        on="segment_id",
        how="left",
        validate="one_to_one",
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


def train_cnn_baseline(
    spectrogram_index_df: pd.DataFrame,
    features_df: pd.DataFrame,
    split_dict: dict,
    config: dict,
    project_root: str | Path | None = None,
) -> dict:
    missing_spectrogram_columns = REQUIRED_SPECTROGRAM_COLUMNS.difference(
        spectrogram_index_df.columns
    )
    if missing_spectrogram_columns:
        raise ValueError(
            "spectrogram_index_df is missing required columns: "
            + ", ".join(sorted(missing_spectrogram_columns))
        )

    missing_feature_columns = REQUIRED_METADATA_COLUMNS.difference(features_df.columns)
    if missing_feature_columns:
        raise ValueError(
            "features_df is missing required columns: "
            + ", ".join(sorted(missing_feature_columns))
    )

    seed = int(config.get("seed", 0))
    cnn_config = config.get("cnn", {})
    _configure_torch_runtime(cnn_config)
    stabilize_torch_backend()
    # Default to single-threaded: multithreaded MaxPool2d forwards intermittently
    # trigger a native access violation on some Windows torch builds.
    if cnn_config.get("torch_num_threads") in (None, ""):
        torch.set_num_threads(1)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    root = resolve_project_root(config, project_root=project_root)
    data_dir = resolve_data_dir(config, project_root=root)
    artifacts_dir = _resolve_artifacts_dir(config, project_root=root)

    train_df = _split_frame(
        spectrogram_index_df,
        split_dict,
        "train",
        required_columns=REQUIRED_SPECTROGRAM_COLUMNS,
        frame_name="spectrogram_index_df",
    )
    val_df = _split_frame(
        spectrogram_index_df,
        split_dict,
        "val",
        required_columns=REQUIRED_SPECTROGRAM_COLUMNS,
        frame_name="spectrogram_index_df",
    )
    test_df = _split_frame(
        spectrogram_index_df,
        split_dict,
        "test",
        required_columns=REQUIRED_SPECTROGRAM_COLUMNS,
        frame_name="spectrogram_index_df",
    )

    if train_df.empty:
        raise ValueError("Train split is empty; cannot train CNN baseline.")

    train_labels = train_df["individual_id"].astype(str).to_numpy()
    unique_train_labels = sorted(set(train_labels.tolist()))
    if len(unique_train_labels) < 2:
        raise ValueError("Train split must contain at least two individuals.")

    label_encoder = LabelEncoder()
    label_encoder.fit(train_labels)
    known_labels = set(label_encoder.classes_.tolist())
    for split_name, split_df in (("val", val_df), ("test", test_df)):
        split_labels = set(split_df["individual_id"].astype(str).tolist())
        unseen_labels = split_labels.difference(known_labels)
        if unseen_labels:
            raise ValueError(
                f"{split_name} contains individuals not present in train: "
                + ", ".join(sorted(unseen_labels))
            )

    dataloaders = create_dataloaders(
        spectrogram_index_df,
        split_dict,
        label_encoder,
        data_dir,
        config,
    )

    num_classes = int(len(label_encoder.classes_))
    train_targets = np.asarray(dataloaders["train"].dataset.labels, dtype=np.int64)
    if train_targets.size == 0:
        raise ValueError("Train dataloader is empty after CNN sampling limits.")

    class_counts = np.bincount(train_targets, minlength=num_classes).astype(np.float32)
    inverse_frequency = np.where(class_counts > 0, 1.0 / class_counts, 0.0)
    class_weights = inverse_frequency / max(float(np.sum(inverse_frequency)), 1.0)
    class_weights = class_weights * float(num_classes)

    model_name = str(cnn_config.get("model", "custom")).strip().lower()
    dropout = float(cnn_config.get("dropout", 0.3))
    if model_name == "custom":
        model = CricketCNN(num_classes=num_classes, dropout=dropout)
    elif model_name == "resnet18":
        model = create_resnet18_model(num_classes=num_classes, dropout=dropout)
    else:
        raise ValueError(
            f"Unsupported CNN model: {model_name}. Expected custom or resnet18."
        )

    device = _select_device(cnn_config)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cnn_config.get("lr", 1e-3)),
    )

    epochs = int(cnn_config.get("epochs", 50))
    patience = int(cnn_config.get("patience", 10))
    batch_sleep_s = float(cnn_config.get("batch_sleep_s", 0.0) or 0.0)

    model_dir = artifacts_dir / "models"
    metrics_dir = artifacts_dir / "metrics"
    figures_dir = artifacts_dir / "figures"
    checkpoint_path = model_dir / "cnn_baseline.pt"
    metrics_path = metrics_dir / "cnn_metrics.json"
    confusion_matrix_path = figures_dir / "cnn_confusion_matrix.png"
    history_path = figures_dir / "cnn_training_history.png"

    history: dict[str, list[float | None]] = {
        "train_loss": [],
        "val_loss": [],
        "train_macro_f1": [],
        "val_macro_f1": [],
    }

    best_val_macro_f1 = float("-inf")
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_total = 0.0
        train_predicted: list[int] = []
        train_true: list[int] = []
        for inputs, labels in dataloaders["train"]:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss_total += float(loss.item()) * int(labels.size(0))
            train_predicted.extend(
                torch.argmax(logits, dim=1).detach().cpu().numpy().astype(int).tolist()
            )
            train_true.extend(labels.detach().cpu().numpy().astype(int).tolist())
            if batch_sleep_s > 0.0:
                time.sleep(batch_sleep_s)

        train_loss = float(train_loss_total / len(dataloaders["train"].dataset))
        train_macro_f1 = float(
            f1_score(train_true, train_predicted, average="macro", zero_division=0)
        )
        val_loss = _compute_epoch_loss(model, dataloaders["val"], criterion, device)
        val_metrics = _evaluate_model(
            model,
            dataloaders["val"],
            label_encoder,
            features_df,
            device,
        )
        val_macro_f1 = (
            None if val_metrics["macro_f1"] is None else float(val_metrics["macro_f1"])
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_macro_f1"].append(train_macro_f1)
        history["val_macro_f1"].append(val_macro_f1)

        current_val_score = _metric_value(val_macro_f1, -1.0)
        if current_val_score > best_val_macro_f1:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": model_name,
                    "dropout": dropout,
                    "num_classes": num_classes,
                    "label_classes": label_encoder.classes_.tolist(),
                    "best_epoch": epoch,
                },
                checkpoint_path,
            )
            best_val_macro_f1 = current_val_score
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    metrics_by_split = {
        "val": _evaluate_model(
            model,
            dataloaders["val"],
            label_encoder,
            features_df,
            device,
        ),
        "test": _evaluate_model(
            model,
            dataloaders["test"],
            label_encoder,
            features_df,
            device,
        ),
    }

    _plot_confusion_matrix(metrics_by_split, label_encoder, confusion_matrix_path)
    _plot_training_history(history, history_path)

    metrics_payload = {
        "model": "cnn_baseline",
        "architecture": model_name,
        "seed": seed,
        "device": device,
        "num_classes": num_classes,
        "best_epoch": best_epoch,
        "epochs_ran": len(history["train_loss"]),
        "batch_sleep_s": batch_sleep_s,
        "sample_limits": {
            "max_train_samples_per_class": cnn_config.get("max_train_samples_per_class"),
            "max_val_samples_per_class": cnn_config.get("max_val_samples_per_class"),
            "max_test_samples_per_class": cnn_config.get("max_test_samples_per_class"),
        },
        "class_weights": [float(value) for value in class_weights.tolist()],
        "metrics": {
            split_name: _metrics_payload(split_metrics)
            for split_name, split_metrics in metrics_by_split.items()
        },
        "history": history,
    }

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics_payload, indent=2) + "\n", encoding="utf-8")

    return {
        **metrics_payload,
        "artifact_paths": {
            "model": str(checkpoint_path),
            "metrics": str(metrics_path),
            "confusion_matrix_figure": str(confusion_matrix_path),
            "training_history_figure": str(history_path),
        },
    }
