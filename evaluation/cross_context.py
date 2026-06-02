"""Cross-context (diel) generalization evaluation.

Scientific question
--------------------
The main MVP shows that individual identity is recoverable from calling song.
A central confound, however, is *recording context*: if a classifier is only
ever trained and tested within the same diel context (e.g. night recordings),
it may be learning session/context conditions rather than the individual's
stable acoustic signature.

This module isolates that confound with a transfer experiment:

* **cross-context** -- fit ``StandardScaler -> PCA -> classifier`` on *all*
  segments of context A and evaluate on *all* segments of context B. This is the
  honest "does identity survive a context switch" test.
* **within-context** -- a leakage-free, recording-grouped held-out split inside
  context A (the ceiling under the project's ``splits.group_by`` convention).
* **penalty** -- ``within - cross`` per metric, i.e. how much accuracy is lost
  purely to the context shift.

Only individuals present in *both* contexts (with at least
``cross_context.min_segments_per_context`` segments in each) are used, so the
two directions share an identical label set. Everything is fit on train data
only; nothing from the test context touches the scaler/PCA/classifier.
"""

from __future__ import annotations

import json
import os
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
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix as sklearn_confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import LabelEncoder, StandardScaler

from cricket_id.models.pca_baseline import (
    EXCLUDED_FEATURE_COLUMNS,
    _compute_session_level_accuracy,
)
from cricket_id.utils.paths import resolve_project_root

CONTEXT_COLUMN_DEFAULT = "daytime"
METRIC_KEYS = ("accuracy", "balanced_accuracy", "macro_f1", "session_level_accuracy")


def _resolve_artifacts_dir(config: dict, *, project_root: str | Path | None = None) -> Path:
    root = resolve_project_root(config, project_root=project_root)
    artifacts_dir = Path(str(config.get("paths", {}).get("artifacts_dir", "artifacts")))
    return artifacts_dir if artifacts_dir.is_absolute() else (root / artifacts_dir).resolve()


def _resolve_reports_dir(config: dict, *, project_root: str | Path | None = None) -> Path:
    root = resolve_project_root(config, project_root=project_root)
    reports_dir = Path(str(config.get("paths", {}).get("reports_dir", "reports")))
    return reports_dir if reports_dir.is_absolute() else (root / reports_dir).resolve()


def _feature_columns(df: pd.DataFrame, context_column: str) -> list[str]:
    excluded = set(EXCLUDED_FEATURE_COLUMNS) | {context_column, "__context__"}
    return [
        column
        for column in df.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(df[column])
    ]


def _attach_context(
    features_df: pd.DataFrame,
    recordings_df: pd.DataFrame,
    context_column: str,
) -> pd.DataFrame:
    if context_column not in recordings_df.columns:
        raise ValueError(
            f"Context column '{context_column}' not found in recordings manifest. "
            f"Available columns: {', '.join(recordings_df.columns)}"
        )
    lookup = recordings_df.loc[:, ["recording_id", context_column]].copy()
    lookup["recording_id"] = lookup["recording_id"].astype(str)
    lookup = lookup.rename(columns={context_column: "__context__"})
    lookup["__context__"] = lookup["__context__"].astype(str)

    working = features_df.copy()
    working["recording_id"] = working["recording_id"].astype(str)
    working["segment_id"] = working["segment_id"].astype(str)
    working["individual_id"] = working["individual_id"].astype(str)
    working["session_id"] = working["session_id"].astype(str)
    merged = working.merge(lookup.drop_duplicates("recording_id"), on="recording_id", how="left")
    merged = merged.loc[merged["__context__"].notna() & (merged["__context__"] != "nan")].copy()
    return merged


def _shared_individuals(df: pd.DataFrame, contexts: list[str], min_segments: int) -> list[str]:
    counts = (
        df.groupby(["individual_id", "__context__"]).size().unstack(fill_value=0)
    )
    for context in contexts:
        if context not in counts.columns:
            counts[context] = 0
    mask = np.logical_and.reduce([counts[context] >= min_segments for context in contexts])
    return sorted(counts.index[mask].astype(str).tolist())


def _recording_grouped_split(
    df: pd.DataFrame,
    train_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministic per-individual split on ``recording_id`` (leakage-free).

    All segments of one recording stay together. Individuals with a single
    recording are assigned entirely to train (and therefore contribute no
    within-context test segments).
    """
    rng = np.random.default_rng(seed)
    train_ids: set[str] = set()
    test_ids: set[str] = set()
    for _, individual_df in df.groupby("individual_id", sort=True):
        recordings = sorted(individual_df["recording_id"].astype(str).unique().tolist())
        if len(recordings) == 1:
            train_ids.update(recordings)
            continue
        order = rng.permutation(len(recordings))
        n_train = max(1, int(round(len(recordings) * train_ratio)))
        n_train = min(n_train, len(recordings) - 1)  # keep at least one test recording
        shuffled = [recordings[index] for index in order]
        train_ids.update(shuffled[:n_train])
        test_ids.update(shuffled[n_train:])
    train_df = df.loc[df["recording_id"].astype(str).isin(train_ids)].copy()
    test_df = df.loc[df["recording_id"].astype(str).isin(test_ids)].copy()
    return train_df, test_df


def _build_classifier(name: str, seed: int):
    if name == "logistic_regression":
        return LogisticRegression(max_iter=2000, solver="lbfgs", random_state=seed)
    if name == "nearest_centroid":
        return NearestCentroid()
    raise ValueError(
        f"Unsupported classifier '{name}'. Expected logistic_regression or nearest_centroid."
    )


def _fit_and_eval(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    label_encoder: LabelEncoder,
    config: dict,
) -> dict:
    seed = int(config.get("seed", 0))
    pca_config = config.get("pca", {})
    requested_components = int(pca_config.get("n_components", 30))
    classifier_name = str(pca_config.get("classifier", "logistic_regression")).strip()

    labels = np.arange(len(label_encoder.classes_))
    empty = {
        "sample_count": int(len(test_df)),
        "session_count": int(test_df["session_id"].nunique()) if not test_df.empty else 0,
        "train_sample_count": int(len(train_df)),
        "n_individuals": int(len(label_encoder.classes_)),
        "accuracy": None,
        "balanced_accuracy": None,
        "macro_f1": None,
        "session_level_accuracy": None,
        "per_class_recall": {str(c): 0.0 for c in label_encoder.classes_},
        "confusion_matrix": np.zeros((len(labels), len(labels)), dtype=int).tolist(),
    }
    if train_df.empty or test_df.empty:
        return empty

    train_features = train_df.loc[:, feature_columns].to_numpy(dtype=np.float64)
    test_features = test_df.loc[:, feature_columns].to_numpy(dtype=np.float64)

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_features)

    effective_components = min(requested_components, train_scaled.shape[0], train_scaled.shape[1])
    if effective_components < 1:
        return empty
    pca = PCA(n_components=effective_components)
    train_transformed = pca.fit_transform(train_scaled)
    test_transformed = pca.transform(scaler.transform(test_features))

    classifier = _build_classifier(classifier_name, seed)
    classifier.fit(train_transformed, label_encoder.transform(train_df["individual_id"].astype(str)))

    true_labels = label_encoder.transform(test_df["individual_id"].astype(str))
    predicted = classifier.predict(test_transformed)
    predicted_str = label_encoder.inverse_transform(predicted).tolist()

    confusion = sklearn_confusion_matrix(true_labels, predicted, labels=labels)
    recall_values = recall_score(true_labels, predicted, labels=labels, average=None, zero_division=0)
    session_accuracy = _compute_session_level_accuracy(
        predicted_str,
        test_df["segment_id"].astype(str).tolist(),
        test_df,
    )

    return {
        "sample_count": int(len(test_df)),
        "session_count": int(test_df["session_id"].nunique()),
        "train_sample_count": int(len(train_df)),
        "n_individuals": int(len(label_encoder.classes_)),
        "accuracy": float(accuracy_score(true_labels, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(true_labels, predicted)),
        "macro_f1": float(f1_score(true_labels, predicted, average="macro", zero_division=0)),
        "session_level_accuracy": float(session_accuracy),
        "per_class_recall": {
            str(label): float(recall)
            for label, recall in zip(label_encoder.classes_, recall_values, strict=True)
        },
        "confusion_matrix": confusion.astype(int).tolist(),
        "effective_n_components": int(effective_components),
    }


def _penalty(within: dict, cross: dict) -> dict:
    out: dict[str, float | None] = {}
    for key in METRIC_KEYS:
        a, b = within.get(key), cross.get(key)
        out[key] = (float(a) - float(b)) if (a is not None and b is not None) else None
    return out


def _plot_summary(directions: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    if not directions:
        ax.text(0.5, 0.5, "No cross-context data", ha="center", va="center")
        ax.set_axis_off()
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        return

    labels = [d["label"] for d in directions]
    within_vals = [d["within_context"].get("macro_f1") or 0.0 for d in directions]
    cross_vals = [d["cross_context"].get("macro_f1") or 0.0 for d in directions]
    x = np.arange(len(labels))
    width = 0.36
    ax.bar(x - width / 2, within_vals, width, label="within-context (ceiling)", color="#7aa7ff")
    ax.bar(x + width / 2, cross_vals, width, label="cross-context (transfer)", color="#4ec9b0")
    for xi, (w, c) in enumerate(zip(within_vals, cross_vals)):
        ax.annotate(f"-{(w - c):.2f}", (xi, max(w, c) + 0.02), ha="center", fontsize=9, color="#f48771")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Macro-F1")
    ax.set_title("Cross-context generalization (diel) — identity vs. context confound")
    ax.legend(loc="upper right", frameon=False)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _write_report(results: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Cross-Context (Diel) Generalization\n")
    lines.append(f"Generated: {results['generated_at']}\n")
    lines.append(
        "This experiment tests whether individual identity survives a recording-context "
        "switch. A large within-vs-cross gap means part of the apparent identity signal is "
        "confounded with the recording context (day vs. night), not the individual.\n"
    )
    lines.append("## Scope\n")
    lines.append("| Field | Value |")
    lines.append("| --- | --- |")
    lines.append(f"| Context column | `{results['context_column']}` |")
    lines.append(f"| Contexts | {', '.join(results['contexts'])} |")
    lines.append(f"| Shared individuals | {results['n_shared_individuals']} |")
    lines.append(f"| Min segments / context | {results['min_segments_per_context']} |")
    lines.append(f"| Classifier | {results['classifier']} |")
    if results["excluded_individuals"]:
        lines.append(
            f"| Excluded (single-context / too sparse) | {', '.join(results['excluded_individuals'])} |"
        )
    lines.append("")
    lines.append("## Results\n")
    lines.append("| Direction | Metric | Within-context | Cross-context | Penalty (within−cross) |")
    lines.append("| --- | --- | --- | --- | --- |")
    for direction in results["directions"]:
        for key in METRIC_KEYS:
            w = direction["within_context"].get(key)
            c = direction["cross_context"].get(key)
            p = direction["penalty"].get(key)
            lines.append(
                f"| {direction['label']} | {key} | "
                f"{'n/a' if w is None else f'{w:.3f}'} | "
                f"{'n/a' if c is None else f'{c:.3f}'} | "
                f"{'n/a' if p is None else f'{p:+.3f}'} |"
            )
    lines.append("")
    lines.append("![Cross-context summary](../artifacts/figures/cross_context_summary.png)\n")
    lines.append("## Interpretation\n")
    lines.append(
        "- A small penalty (cross-context macro-F1 close to within-context) supports a genuine, "
        "context-stable individual signature.\n"
        "- A large penalty indicates the model leans on context-specific cues; identity claims "
        "should then be qualified, and temperature/context residualization becomes important.\n"
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_cross_context_eval(
    features_df: pd.DataFrame,
    recordings_df: pd.DataFrame,
    config: dict,
    *,
    project_root: str | Path | None = None,
    generated_at: str = "",
) -> dict:
    """Run the cross-context transfer experiment and persist artifacts."""
    cc_config = config.get("cross_context", {}) or {}
    context_column = str(cc_config.get("context_column", CONTEXT_COLUMN_DEFAULT))
    min_segments = int(cc_config.get("min_segments_per_context", 20))
    within_train_ratio = float(cc_config.get("within_train_ratio", 0.7))
    seed = int(config.get("seed", 0))

    enriched = _attach_context(features_df, recordings_df, context_column)
    contexts = sorted(enriched["__context__"].unique().tolist())
    if len(contexts) < 2:
        raise ValueError(
            f"Need at least two values in context column '{context_column}'; "
            f"found: {contexts}"
        )
    # Limit to the two most populous contexts for a clean two-way transfer.
    if len(contexts) > 2:
        top = (
            enriched["__context__"].value_counts().index.tolist()[:2]
        )
        contexts = sorted(top)
        enriched = enriched.loc[enriched["__context__"].isin(contexts)].copy()

    all_individuals = sorted(enriched["individual_id"].unique().tolist())
    shared = _shared_individuals(enriched, contexts, min_segments)
    if len(shared) < 2:
        raise ValueError(
            "Fewer than two individuals are present in both contexts with at least "
            f"{min_segments} segments each; cross-context evaluation is not possible."
        )
    excluded = sorted(set(all_individuals) - set(shared))
    shared_df = enriched.loc[enriched["individual_id"].isin(shared)].copy()

    feature_columns = _feature_columns(shared_df, context_column)
    if not feature_columns:
        raise ValueError("No numeric feature columns available for cross-context evaluation.")

    label_encoder = LabelEncoder()
    label_encoder.fit(np.array(shared, dtype=object))

    directions: list[dict] = []
    for train_context in contexts:
        test_context = [c for c in contexts if c != train_context][0]
        train_ctx_df = shared_df.loc[shared_df["__context__"] == train_context].copy()
        test_ctx_df = shared_df.loc[shared_df["__context__"] == test_context].copy()

        cross_metrics = _fit_and_eval(
            train_ctx_df, test_ctx_df, feature_columns, label_encoder, config
        )

        within_train_df, within_test_df = _recording_grouped_split(
            train_ctx_df, within_train_ratio, seed
        )
        within_metrics = _fit_and_eval(
            within_train_df, within_test_df, feature_columns, label_encoder, config
        )

        directions.append(
            {
                "label": f"{train_context}->{test_context}",
                "train_context": train_context,
                "test_context": test_context,
                "within_context": within_metrics,
                "cross_context": cross_metrics,
                "penalty": _penalty(within_metrics, cross_metrics),
            }
        )

    results = {
        "generated_at": generated_at,
        "context_column": context_column,
        "contexts": contexts,
        "min_segments_per_context": min_segments,
        "within_train_ratio": within_train_ratio,
        "classifier": str(config.get("pca", {}).get("classifier", "logistic_regression")),
        "shared_individuals": shared,
        "n_shared_individuals": len(shared),
        "excluded_individuals": excluded,
        "segment_counts_by_context": {
            context: int((shared_df["__context__"] == context).sum()) for context in contexts
        },
        "directions": directions,
    }

    artifacts_dir = _resolve_artifacts_dir(config, project_root=project_root)
    metrics_path = artifacts_dir / "metrics" / "cross_context.json"
    figure_path = artifacts_dir / "figures" / "cross_context_summary.png"
    reports_dir = _resolve_reports_dir(config, project_root=project_root)
    report_path = reports_dir / "cross_context.md"

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    _plot_summary(directions, figure_path)
    _write_report(results, report_path)

    results["artifact_paths"] = {
        "metrics": str(metrics_path),
        "figure": str(figure_path),
        "report": str(report_path),
    }
    return results
