"""Uncertainty quantification for classification metrics.

Two complementary tools, both designed to respect the session structure of the
data (segments within a session are not independent — pseudoreplication):

* **Session-level cluster bootstrap** -- resample whole *sessions* (not
  individual chirps) with replacement and recompute the metric each time. The
  resulting percentile interval reflects between-session variability, which is
  the honest source of uncertainty for this dataset. Reporting a naive
  segment-level bootstrap would understate the interval.
* **Label-permutation test** -- shuffle the true labels relative to the
  predictions many times to build a null distribution, then report the fraction
  of permutations that match or beat the observed metric (a one-sided p-value).
  This quantifies how unlikely the observed performance is under "predictions
  carry no individual information".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score

METRIC_NAMES = ("accuracy", "balanced_accuracy", "macro_f1", "session_level_accuracy")


def _chirp_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    if y_true.size == 0:
        return {"accuracy": float("nan"), "balanced_accuracy": float("nan"), "macro_f1": float("nan")}
    return {
        "accuracy": float(np.mean(y_true == y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def _majority(values: np.ndarray) -> str:
    return str(pd.Series(values).value_counts().sort_index().idxmax())


def _session_structure(prediction_df: pd.DataFrame) -> tuple[list[np.ndarray], np.ndarray]:
    """Per-session row positions and a per-session correctness flag (majority vote)."""
    true = prediction_df["true_individual_id"].astype(str).to_numpy()
    pred = prediction_df["predicted_individual_id"].astype(str).to_numpy()
    session_ids = prediction_df["session_id"].astype(str).to_numpy()
    positions = np.arange(len(prediction_df))

    session_indices: list[np.ndarray] = []
    session_correct: list[float] = []
    for sid in sorted(set(session_ids.tolist())):
        idx = positions[session_ids == sid]
        session_indices.append(idx)
        session_correct.append(float(_majority(pred[idx]) == _majority(true[idx])))
    return session_indices, np.asarray(session_correct, dtype=np.float64)


def _percentile_ci(samples: list[float], alpha: float) -> tuple[float, float]:
    arr = np.asarray([s for s in samples if np.isfinite(s)], dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    low = float(np.percentile(arr, 100 * (alpha / 2)))
    high = float(np.percentile(arr, 100 * (1 - alpha / 2)))
    return low, high


def session_cluster_bootstrap(
    prediction_df: pd.DataFrame,
    *,
    n_iterations: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, dict[str, float]] | dict[str, str]:
    """Percentile CIs for each metric via resampling whole sessions."""
    if prediction_df.empty:
        return {"status": "no_samples"}

    y_true = prediction_df["true_individual_id"].astype(str).to_numpy()
    y_pred = prediction_df["predicted_individual_id"].astype(str).to_numpy()
    session_indices, session_correct = _session_structure(prediction_df)
    n_sessions = len(session_indices)
    if n_sessions < 2:
        return {"status": "insufficient_sessions", "n_sessions": n_sessions}

    observed = _chirp_metrics(y_true, y_pred)
    observed["session_level_accuracy"] = float(np.mean(session_correct))

    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {name: [] for name in METRIC_NAMES}
    for _ in range(n_iterations):
        chosen = rng.integers(0, n_sessions, size=n_sessions)
        idx = np.concatenate([session_indices[c] for c in chosen])
        chirp = _chirp_metrics(y_true[idx], y_pred[idx])
        samples["accuracy"].append(chirp["accuracy"])
        samples["balanced_accuracy"].append(chirp["balanced_accuracy"])
        samples["macro_f1"].append(chirp["macro_f1"])
        samples["session_level_accuracy"].append(float(np.mean(session_correct[chosen])))

    result: dict[str, dict[str, float]] = {"status": "ok", "n_sessions": n_sessions, "n_iterations": n_iterations}  # type: ignore[dict-item]
    for name in METRIC_NAMES:
        low, high = _percentile_ci(samples[name], alpha)
        result[name] = {
            "observed": float(observed[name]),
            "ci_low": low,
            "ci_high": high,
            "ci_level": float(1 - alpha),
        }
    return result


def label_permutation_test(
    prediction_df: pd.DataFrame,
    *,
    n_iterations: int = 1000,
    seed: int = 0,
) -> dict[str, dict[str, float]] | dict[str, str]:
    """One-sided p-values vs. a shuffled-label null for chirp-level metrics."""
    if prediction_df.empty:
        return {"status": "no_samples"}

    y_true = prediction_df["true_individual_id"].astype(str).to_numpy()
    y_pred = prediction_df["predicted_individual_id"].astype(str).to_numpy()
    if y_true.size < 2:
        return {"status": "insufficient_samples"}

    observed = _chirp_metrics(y_true, y_pred)
    rng = np.random.default_rng(seed)
    ge_counts = {name: 0 for name in ("accuracy", "balanced_accuracy", "macro_f1")}
    null_means = {name: 0.0 for name in ge_counts}
    for _ in range(n_iterations):
        permuted = rng.permutation(y_true)
        null = _chirp_metrics(permuted, y_pred)
        for name in ge_counts:
            if null[name] >= observed[name] - 1e-12:
                ge_counts[name] += 1
            null_means[name] += null[name]

    result: dict[str, dict[str, float]] = {"status": "ok", "n_iterations": n_iterations}  # type: ignore[dict-item]
    for name in ge_counts:
        # add-one smoothing: minimum achievable p-value is 1/(n+1)
        p_value = (ge_counts[name] + 1) / (n_iterations + 1)
        result[name] = {
            "observed": float(observed[name]),
            "null_mean": float(null_means[name] / n_iterations),
            "p_value": float(p_value),
        }
    return result


def compute_significance(
    prediction_df: pd.DataFrame,
    *,
    bootstrap_iterations: int = 2000,
    permutation_iterations: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, object]:
    """Bundle bootstrap CIs and permutation p-values for one prediction frame."""
    return {
        "bootstrap": session_cluster_bootstrap(
            prediction_df, n_iterations=bootstrap_iterations, alpha=alpha, seed=seed
        ),
        "permutation": label_permutation_test(
            prediction_df, n_iterations=permutation_iterations, seed=seed
        ),
    }
