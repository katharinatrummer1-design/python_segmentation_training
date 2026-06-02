"""CNN embedding extraction, projection, and open-set verification.

Why this exists
---------------
The PCA path classifies hand-crafted features into a fixed set of individuals
(closed-set). The CNN additionally learns a representation: the 128-d vector
produced just before the classification head (``CricketCNN.features`` ->
``AdaptiveAvgPool2d`` -> flatten) is a *learned acoustic fingerprint* per chirp.

This module uses that fingerprint for things the tabular path cannot do well:

* **Embedding map** -- a 2D projection (UMAP if available, else t-SNE, else PCA)
  of the learned space, coloured by individual and by session.
* **Open-set verification** -- "are these two chirps the same individual?"
  scored by cosine similarity, summarised as an ROC curve, AUC and equal-error
  rate (EER). Positive pairs are drawn from *different recordings* of the same
  individual, so the score reflects identity rather than recording similarity.
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
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from torch import nn

from cricket_id.models.cnn_dataset import CricketSpectrogramDataset
from cricket_id.models.cnn_model import CricketCNN, create_resnet18_model
from cricket_id.utils.paths import resolve_data_dir, resolve_reports_dir, resolve_project_root


def _resolve_artifacts_dir(config: dict, *, project_root: str | Path | None = None) -> Path:
    root = resolve_project_root(config, project_root=project_root)
    artifacts_dir = Path(str(config.get("paths", {}).get("artifacts_dir", "artifacts")))
    return artifacts_dir if artifacts_dir.is_absolute() else (root / artifacts_dir).resolve()


def _load_embedding_model(checkpoint: dict, device: torch.device) -> nn.Module:
    """Reconstruct the trained model and strip its classification head."""
    model_name = str(checkpoint.get("model_name", "custom"))
    dropout = float(checkpoint.get("dropout", 0.3))
    num_classes = int(checkpoint["num_classes"])
    if model_name == "resnet18":
        model = create_resnet18_model(num_classes=num_classes, dropout=dropout)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.fc = nn.Identity()
    else:
        model = CricketCNN(num_classes=num_classes, dropout=dropout)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.classifier = nn.Identity()
    model.eval()
    return model.to(device)


def _compute_embeddings(
    model: nn.Module,
    segment_ids: list[str],
    data_dir: Path,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    dataset = CricketSpectrogramDataset(
        segment_ids,
        np.zeros(len(segment_ids), dtype=np.int64),
        data_dir,
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for tensors, _ in loader:
            tensors = tensors.to(device)
            out = model(tensors)
            chunks.append(out.detach().cpu().numpy().astype(np.float32))
    if not chunks:
        return np.empty((0, 0), dtype=np.float32)
    return np.concatenate(chunks, axis=0)


def _project_2d(embeddings: np.ndarray, seed: int, max_points: int) -> tuple[np.ndarray, np.ndarray, str]:
    """Return (2D coords, selected row indices, method name)."""
    n = embeddings.shape[0]
    rng = np.random.default_rng(seed)
    if n > max_points:
        selected = np.sort(rng.choice(n, size=max_points, replace=False))
    else:
        selected = np.arange(n)
    subset = embeddings[selected]

    try:  # UMAP gives the nicest manifold, but is optional.
        import umap  # type: ignore

        reducer = umap.UMAP(n_components=2, random_state=seed)
        coords = reducer.fit_transform(subset)
        return np.asarray(coords, dtype=np.float32), selected, "umap"
    except Exception:
        pass

    from sklearn.decomposition import PCA

    pre = subset
    max_pre = min(50, subset.shape[0] - 1, subset.shape[1])
    if subset.shape[1] > max_pre and max_pre >= 2:
        pre = PCA(n_components=max_pre, random_state=seed).fit_transform(subset)
    try:
        from sklearn.manifold import TSNE

        if subset.shape[0] < 10:
            raise ValueError("too few points for t-SNE")
        coords = TSNE(
            n_components=2,
            random_state=seed,
            init="pca",
            perplexity=min(30, max(5, subset.shape[0] // 4)),
        ).fit_transform(pre)
        return np.asarray(coords, dtype=np.float32), selected, "tsne"
    except Exception:
        coords = PCA(n_components=2, random_state=seed).fit_transform(subset)
        return np.asarray(coords, dtype=np.float32), selected, "pca"


def _sample_verification_pairs(
    embeddings: np.ndarray,
    individual_ids: np.ndarray,
    recording_ids: np.ndarray,
    n_pairs: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Cosine similarities and binary same-individual labels for sampled pairs.

    Positive pairs come from *different recordings* of the same individual so the
    score reflects identity, not within-recording similarity.
    """
    rng = np.random.default_rng(seed)
    normed = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)

    individuals = np.unique(individual_ids)
    by_individual = {ind: np.where(individual_ids == ind)[0] for ind in individuals}
    # individuals usable for positives need >=2 distinct recordings
    positive_pool = [
        ind for ind in individuals if len(np.unique(recording_ids[by_individual[ind]])) >= 2
    ]

    sims: list[float] = []
    labels: list[int] = []

    # positives
    attempts = 0
    while len([l for l in labels if l == 1]) < n_pairs and positive_pool and attempts < n_pairs * 20:
        attempts += 1
        ind = positive_pool[rng.integers(len(positive_pool))]
        idxs = by_individual[ind]
        a, b = idxs[rng.integers(len(idxs))], idxs[rng.integers(len(idxs))]
        if recording_ids[a] == recording_ids[b]:
            continue
        sims.append(float(np.dot(normed[a], normed[b])))
        labels.append(1)

    # negatives
    attempts = 0
    n_neg_target = len([l for l in labels if l == 1]) or n_pairs
    while len([l for l in labels if l == 0]) < n_neg_target and len(individuals) >= 2 and attempts < n_pairs * 20:
        attempts += 1
        ind_a, ind_b = individuals[rng.integers(len(individuals))], individuals[rng.integers(len(individuals))]
        if ind_a == ind_b:
            continue
        a = by_individual[ind_a][rng.integers(len(by_individual[ind_a]))]
        b = by_individual[ind_b][rng.integers(len(by_individual[ind_b]))]
        sims.append(float(np.dot(normed[a], normed[b])))
        labels.append(0)

    return np.asarray(sims, dtype=np.float64), np.asarray(labels, dtype=np.int64)


def _verification_metrics(sims: np.ndarray, labels: np.ndarray) -> dict:
    if len(np.unique(labels)) < 2:
        return {"status": "insufficient_pairs", "n_pairs": int(len(labels))}
    fpr, tpr, thresholds = roc_curve(labels, sims)
    auc = float(roc_auc_score(labels, sims))
    fnr = 1.0 - tpr
    eer_index = int(np.nanargmin(np.abs(fnr - fpr)))
    eer = float((fpr[eer_index] + fnr[eer_index]) / 2.0)
    eer_threshold = float(thresholds[eer_index])

    # subsample ROC to ~80 points for transport
    if len(fpr) > 80:
        keep = np.linspace(0, len(fpr) - 1, 80).astype(int)
        fpr_s, tpr_s = fpr[keep], tpr[keep]
    else:
        fpr_s, tpr_s = fpr, tpr

    return {
        "status": "ok",
        "n_pairs": int(len(labels)),
        "n_positive": int(labels.sum()),
        "n_negative": int((labels == 0).sum()),
        "auc": auc,
        "eer": eer,
        "eer_threshold": eer_threshold,
        "mean_sim_same": float(sims[labels == 1].mean()),
        "mean_sim_diff": float(sims[labels == 0].mean()),
        "roc": {"fpr": [float(v) for v in fpr_s], "tpr": [float(v) for v in tpr_s]},
    }


def _plot_scatter(coords: np.ndarray, color_values: np.ndarray, title: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    uniques = sorted(set(color_values.tolist()))
    cmap = plt.get_cmap("tab20", max(len(uniques), 1))
    for i, value in enumerate(uniques):
        mask = color_values == value
        ax.scatter(coords[mask, 0], coords[mask, 1], s=12, alpha=0.7,
                   color=cmap(i % max(len(uniques), 1)), label=str(value), edgecolors="none")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    if len(uniques) <= 20:
        ax.legend(loc="best", fontsize=7, frameon=False, ncol=2)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_roc(metrics: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5.5), constrained_layout=True)
    if metrics.get("status") == "ok":
        ax.plot(metrics["roc"]["fpr"], metrics["roc"]["tpr"], color="#4ec9b0", linewidth=2,
                label=f"AUC={metrics['auc']:.3f}, EER={metrics['eer']:.3f}")
        ax.plot([0, 1], [0, 1], "--", color="#8b95a5", linewidth=1)
        ax.legend(loc="lower right", frameon=False)
    else:
        ax.text(0.5, 0.5, "insufficient pairs", ha="center", va="center")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("CNN embedding verification ROC")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _write_report(results: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ver = results["verification"]
    lines = ["# CNN Embedding Analysis\n", f"Generated: {results['generated_at']}\n"]
    lines.append(
        "The CNN's penultimate 128-d activations are a learned acoustic fingerprint. "
        "This report projects them to 2D and tests open-set verification "
        "(same-individual vs. different-individual) by cosine similarity.\n"
    )
    lines.append("## Scope\n")
    lines.append("| Field | Value |")
    lines.append("| --- | --- |")
    lines.append(f"| Embedding dim | {results['embedding_dim']} |")
    lines.append(f"| Segments embedded | {results['n_segments']} |")
    lines.append(f"| Individuals | {results['n_individuals']} |")
    lines.append(f"| Projection method | {results['projection_method']} |")
    lines.append("")
    lines.append("## Verification\n")
    if ver.get("status") == "ok":
        lines.append("| Metric | Value |")
        lines.append("| --- | --- |")
        lines.append(f"| AUC | {ver['auc']:.3f} |")
        lines.append(f"| Equal-error rate (EER) | {ver['eer']:.3f} |")
        lines.append(f"| Mean cosine (same) | {ver['mean_sim_same']:.3f} |")
        lines.append(f"| Mean cosine (different) | {ver['mean_sim_diff']:.3f} |")
        lines.append(f"| Pairs (pos/neg) | {ver['n_positive']} / {ver['n_negative']} |")
    else:
        lines.append(f"Verification status: {ver.get('status')}\n")
    lines.append("")
    lines.append("![Embedding map by individual](../artifacts/figures/cnn_embedding_umap_individual.png)\n")
    lines.append("![Verification ROC](../artifacts/figures/cnn_embedding_verification_roc.png)\n")
    lines.append("## Interpretation\n")
    lines.append(
        "- AUC well above 0.5 and clear separation between same/different cosine means that "
        "the learned embedding captures individual identity even for unseen pairings.\n"
        "- A low EER means a single similarity threshold can accept/reject identity reliably — "
        "the basis for open-set re-identification that closed-set PCA cannot provide.\n"
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_embedding_analysis(
    spectrogram_index_df: pd.DataFrame,
    segments_df: pd.DataFrame,
    config: dict,
    *,
    project_root: str | Path | None = None,
    generated_at: str = "",
) -> dict:
    """Extract CNN embeddings, project to 2D, and run open-set verification."""
    emb_config = config.get("embeddings", {}) or {}
    max_points = int(emb_config.get("max_scatter_points", 2500))
    n_pairs = int(emb_config.get("verification_pairs", 8000))
    batch_size = int(config.get("cnn", {}).get("batch_size", 64))
    seed = int(config.get("seed", 0))

    artifacts_dir = _resolve_artifacts_dir(config, project_root=project_root)
    data_dir = resolve_data_dir(config, project_root=project_root)
    checkpoint_name = str(emb_config.get("checkpoint", "cnn_baseline.pt"))
    checkpoint_path = artifacts_dir / "models" / checkpoint_name
    if not checkpoint_path.exists():
        raise ValueError(
            f"CNN checkpoint not found at {checkpoint_path}. Run train-cnn first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = _load_embedding_model(checkpoint, device)

    # Output names depend on the model type so the CE and triplet analyses can
    # coexist (and be compared side-by-side in the UI).
    loss_tag = str(checkpoint.get("loss", "ce"))
    suffix = "" if loss_tag == "ce" else f"_{loss_tag}"

    index_df = spectrogram_index_df.copy()
    index_df["segment_id"] = index_df["segment_id"].astype(str)
    index_df["individual_id"] = index_df["individual_id"].astype(str)
    index_df["session_id"] = index_df["session_id"].astype(str)
    # attach recording_id from segments
    rec_lookup = segments_df.loc[:, ["segment_id", "recording_id"]].copy()
    rec_lookup["segment_id"] = rec_lookup["segment_id"].astype(str)
    rec_lookup["recording_id"] = rec_lookup["recording_id"].astype(str)
    index_df = index_df.merge(rec_lookup.drop_duplicates("segment_id"), on="segment_id", how="left")
    index_df["recording_id"] = index_df["recording_id"].fillna(index_df["segment_id"])

    segment_ids = index_df["segment_id"].tolist()
    embeddings = _compute_embeddings(model, segment_ids, data_dir, batch_size, device)
    if embeddings.size == 0:
        raise ValueError("No embeddings were produced (empty spectrogram index).")

    individual_ids = index_df["individual_id"].to_numpy()
    session_ids = index_df["session_id"].to_numpy()
    recording_ids = index_df["recording_id"].to_numpy()

    # 2D projection (subsampled)
    coords, selected, method = _project_2d(embeddings, seed, max_points)

    # verification on full set
    sims, labels = _sample_verification_pairs(
        embeddings, individual_ids, recording_ids, n_pairs, seed
    )
    verification = _verification_metrics(sims, labels)

    # persist embeddings parquet
    from cricket_id.io.manifest import _write_parquet_compatible

    embedding_columns = [f"e{i}" for i in range(embeddings.shape[1])]
    embeddings_df = pd.DataFrame(embeddings, columns=embedding_columns)
    embeddings_df.insert(0, "segment_id", segment_ids)
    embeddings_df.insert(1, "individual_id", individual_ids)
    embeddings_df.insert(2, "session_id", session_ids)
    embeddings_df.insert(3, "recording_id", recording_ids)
    embeddings_path = data_dir / "processed" / f"cnn_embeddings{suffix}.parquet"
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    _write_parquet_compatible(embeddings_df, embeddings_path)

    # figures
    figures_dir = artifacts_dir / "figures"
    _plot_scatter(coords, individual_ids[selected], f"CNN embedding map (by individual) [{loss_tag}]",
                  figures_dir / f"cnn_embedding{suffix}_umap_individual.png")
    _plot_scatter(coords, session_ids[selected], f"CNN embedding map (by session) [{loss_tag}]",
                  figures_dir / f"cnn_embedding{suffix}_umap_session.png")
    _plot_roc(verification, figures_dir / f"cnn_embedding{suffix}_verification_roc.png")

    # scatter points for UI (subsample already applied)
    scatter_points = [
        {
            "x": float(coords[i, 0]),
            "y": float(coords[i, 1]),
            "individual_id": str(individual_ids[selected[i]]),
            "session_id": str(session_ids[selected[i]]),
        }
        for i in range(len(selected))
    ]

    results = {
        "generated_at": generated_at,
        "embedding_dim": int(embeddings.shape[1]),
        "n_segments": int(embeddings.shape[0]),
        "n_individuals": int(len(np.unique(individual_ids))),
        "projection_method": method,
        "checkpoint": checkpoint_name,
        "loss": str(checkpoint.get("loss", "ce")),
        "scatter": scatter_points,
        "verification": verification,
        "best_epoch": int(checkpoint.get("best_epoch", -1)),
    }

    metrics_path = artifacts_dir / "metrics" / f"cnn_embeddings{suffix}.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    reports_dir = resolve_reports_dir(config, project_root=project_root)
    _write_report(results, reports_dir / f"cnn_embeddings{suffix}.md")

    results["artifact_paths"] = {
        "embeddings": str(embeddings_path),
        "metrics": str(metrics_path),
    }
    return results
