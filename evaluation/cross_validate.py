"""Session-grouped cross-validation for an honest PCA estimate.

A single train/val/test holdout is fragile here: most individuals have only
2-3 recording sessions, so a leakage-free *session-level* holdout leaves only a
handful of individuals in the test split. The brief's sanctioned fallback is
grouped cross-validation.

This module runs ``StratifiedGroupKFold`` with ``session_id`` as the group, so:
* no session is ever in both train and test of a fold (no recording-condition leakage),
* every eligible session is held out exactly once, so every individual with at
  least two sessions is evaluated on unseen sessions,
* class distribution is balanced across folds as far as the grouping allows.

The class-balanced ``StandardScaler -> PCA -> LogisticRegression`` pipeline is
refit from scratch on each fold's training data only. We report per-fold metrics,
their mean/std, and pooled metrics (one held-out prediction per eligible segment)
with a session-level bootstrap CI. Individuals with a single session cannot be
trained-and-tested without leakage and are reported as uncovered.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from cricket_id.evaluation.statistics import session_cluster_bootstrap
from cricket_id.io.manifest import _read_parquet_compatible
from cricket_id.models.pca_baseline import _feature_columns
from cricket_id.utils.paths import (
    resolve_data_dir,
    resolve_project_root,
    resolve_reports_dir,
)


def _resolve_artifacts_dir(config: dict, *, project_root: str | Path | None = None) -> Path:
    root = resolve_project_root(config, project_root=project_root)
    artifacts_dir = Path(str(config.get("paths", {}).get("artifacts_dir", "artifacts")))
    return artifacts_dir if artifacts_dir.is_absolute() else (root / artifacts_dir).resolve()


def _chirp_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def run_pca_cross_validation(
    features_df: pd.DataFrame,
    config: dict,
    *,
    project_root: str | Path | None = None,
) -> dict:
    cv_config = config.get("cross_validation", {}) or {}
    n_splits = int(cv_config.get("n_splits", 5))
    seed = int(config.get("seed", 0))
    pca_config = config.get("pca", {})
    requested_components = int(pca_config.get("n_components", 30))
    class_weight = pca_config.get("class_weight")
    if isinstance(class_weight, str):
        class_weight = class_weight.strip() or None

    working = features_df.copy()
    for column in ("segment_id", "individual_id", "session_id", "recording_id"):
        if column in working.columns:
            working[column] = working[column].astype(str)
    feature_columns = _feature_columns(working)
    if not feature_columns:
        raise ValueError("No numeric feature columns available for cross-validation.")

    individuals = working["individual_id"].to_numpy()
    sessions = working["session_id"].to_numpy()
    session_per_individual = working.groupby("individual_id")["session_id"].nunique()
    eligible = set(session_per_individual[session_per_individual >= 2].index.tolist())
    uncovered = sorted(set(individuals.tolist()) - eligible)

    n_sessions = working["session_id"].nunique()
    effective_splits = max(2, min(n_splits, n_sessions))

    X = working.loc[:, feature_columns].to_numpy(dtype=np.float64)
    splitter = StratifiedGroupKFold(n_splits=effective_splits, shuffle=True, random_state=seed)

    fold_metrics: list[dict] = []
    pooled = {"segment_id": [], "session_id": [], "true": [], "pred": []}
    for fold_index, (train_idx, test_idx) in enumerate(
        splitter.split(X, individuals, groups=sessions)
    ):
        train_labels = individuals[train_idx]
        seen = set(train_labels.tolist())
        # keep only test rows whose individual was trainable this fold (closed-set)
        keep = np.array([individuals[i] in seen for i in test_idx])
        kept_test = test_idx[keep]
        if kept_test.size == 0 or len(set(train_labels.tolist())) < 2:
            continue

        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(X[train_idx])
        n_components = min(requested_components, train_scaled.shape[0], train_scaled.shape[1])
        pca = PCA(n_components=n_components)
        train_transformed = pca.fit_transform(train_scaled)
        clf = LogisticRegression(
            max_iter=2000, solver="lbfgs", random_state=seed, class_weight=class_weight
        )
        clf.fit(train_transformed, train_labels)

        test_transformed = pca.transform(scaler.transform(X[kept_test]))
        pred = clf.predict(test_transformed)
        true = individuals[kept_test]

        m = _chirp_metrics(true, pred)
        m["fold"] = fold_index
        m["test_segments"] = int(kept_test.size)
        m["test_individuals"] = int(len(set(true.tolist())))
        fold_metrics.append(m)

        pooled["segment_id"].extend(working["segment_id"].to_numpy()[kept_test].tolist())
        pooled["session_id"].extend(sessions[kept_test].tolist())
        pooled["true"].extend(true.tolist())
        pooled["pred"].extend(pred.tolist())

    if not fold_metrics:
        raise ValueError("Cross-validation produced no usable folds.")

    def _agg(metric: str) -> dict[str, float]:
        values = [f[metric] for f in fold_metrics]
        return {"mean": float(np.mean(values)), "std": float(np.std(values))}

    pooled_df = pd.DataFrame(
        {
            "segment_id": pooled["segment_id"],
            "session_id": pooled["session_id"],
            "true_individual_id": pooled["true"],
            "predicted_individual_id": pooled["pred"],
        }
    )
    pooled_true = pooled_df["true_individual_id"].to_numpy()
    pooled_pred = pooled_df["predicted_individual_id"].to_numpy()
    pooled_metrics = _chirp_metrics(pooled_true, pooled_pred)
    bootstrap = session_cluster_bootstrap(
        pooled_df,
        n_iterations=int(cv_config.get("bootstrap_iterations", 2000)),
        alpha=float(cv_config.get("alpha", 0.05)),
        seed=seed,
    )

    results = {
        "model": "pca",
        "protocol": "StratifiedGroupKFold by session_id (leakage-free)",
        "class_weight": class_weight,
        "n_splits": effective_splits,
        "n_individuals_total": int(len(set(individuals.tolist()))),
        "n_individuals_evaluated": int(len(eligible)),
        "uncovered_individuals_single_session": uncovered,
        "pooled_test_segments": int(len(pooled_df)),
        "per_fold": fold_metrics,
        "aggregate": {m: _agg(m) for m in ("accuracy", "balanced_accuracy", "macro_f1")},
        "pooled": pooled_metrics,
        "pooled_bootstrap": bootstrap,
    }

    artifacts_dir = _resolve_artifacts_dir(config, project_root=project_root)
    metrics_path = artifacts_dir / "metrics" / "pca_cross_validation.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    reports_dir = resolve_reports_dir(config, project_root=project_root)
    _write_report(results, reports_dir / "pca_cross_validation.md")

    results["artifact_paths"] = {"metrics": str(metrics_path)}
    return results


def _write_report(results: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def fmt(v: object) -> str:
        try:
            return f"{float(v):.3f}"
        except (TypeError, ValueError):
            return "n/a"

    lines = ["# PCA Session-Grouped Cross-Validation\n"]
    lines.append(
        "Honest, leakage-free estimate: `StratifiedGroupKFold` by `session_id` "
        f"({results['n_splits']} folds), class-balanced PCA pipeline refit per fold. "
        "Every session is held out exactly once.\n"
    )
    lines.append("## Coverage\n")
    lines.append("| Field | Value |")
    lines.append("| --- | --- |")
    lines.append(f"| Individuals total | {results['n_individuals_total']} |")
    lines.append(f"| Individuals evaluated (>=2 sessions) | {results['n_individuals_evaluated']} |")
    lines.append(f"| Pooled held-out segments | {results['pooled_test_segments']} |")
    if results["uncovered_individuals_single_session"]:
        lines.append(
            f"| Uncovered (single session) | {', '.join(results['uncovered_individuals_single_session'])} |"
        )
    lines.append("")
    lines.append("## Metrics (pooled held-out predictions)\n")
    boot = results.get("pooled_bootstrap", {})
    lines.append("| Metric | Pooled | Fold mean +/- std | 95% CI (session bootstrap) |")
    lines.append("| --- | --- | --- | --- |")
    for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
        agg = results["aggregate"][metric]
        ci = boot.get(metric, {}) if isinstance(boot, dict) else {}
        ci_str = (
            f"[{fmt(ci.get('ci_low'))}, {fmt(ci.get('ci_high'))}]"
            if isinstance(ci, dict) and "ci_low" in ci
            else "n/a"
        )
        lines.append(
            f"| {metric} | {fmt(results['pooled'][metric])} | "
            f"{fmt(agg['mean'])} +/- {fmt(agg['std'])} | {ci_str} |"
        )
    lines.append("")
    lines.append("## Interpretation\n")
    lines.append(
        "- This is the number to trust: no session is shared between train and test, so it "
        "measures recognising an individual on a *new* recording day.\n"
        "- Compare it to the single-holdout test macro-F1 in `mvp_report.md`. A large drop there "
        "would have indicated the holdout was leaking recording-condition cues.\n"
        "- Single-session individuals cannot be evaluated without leakage; collecting more sessions "
        "per individual is the main lever to cover them and tighten the intervals.\n"
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
