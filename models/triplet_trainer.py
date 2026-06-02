"""Metric-learning (triplet) trainer for context-invariant individual embeddings.

Motivation
----------
The cross-context experiment showed identity accuracy collapses across the diel
context (train night -> test day). A classifier trained with cross-entropy has no
incentive to be context-invariant. Metric learning does: a batch-hard triplet
loss pulls *same-individual* chirps together and pushes *different-individual*
chirps apart in embedding space. By sampling each individual's batch members
across both contexts (context-aware PK sampling), the hardest positive is often a
cross-context pair, so the optimiser is explicitly pressured to ignore the
recording context and keep the individual signature.

The trained network is a plain :class:`CricketCNN`; only its convolutional
``features`` (the 128-d pre-head embedding) are optimised. The checkpoint is
saved in the same format as the cross-entropy baseline, so ``extract-embeddings``
can consume it directly (point ``embeddings.checkpoint`` at ``cnn_triplet.pt``).
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler

from cricket_id.evaluation.embeddings import _sample_verification_pairs, _verification_metrics
from cricket_id.models.cnn_model import CricketCNN, stabilize_torch_backend
from cricket_id.models.cnn_trainer import (
    _configure_torch_runtime,
    _resolve_artifacts_dir,
    _select_device,
)
from cricket_id.utils.paths import resolve_data_dir, resolve_project_root, resolve_reports_dir


class _TripletDataset(Dataset):
    def __init__(self, segment_ids: list[str], labels: np.ndarray, data_dir: Path) -> None:
        self.segment_ids = [str(s) for s in segment_ids]
        self.labels = np.asarray(labels, dtype=np.int64)
        self.data_dir = Path(data_dir)

    def __len__(self) -> int:
        return len(self.segment_ids)

    def __getitem__(self, index: int):
        segment_id = self.segment_ids[index]
        path = self.data_dir / "processed" / "spectrograms" / f"{segment_id}.npy"
        spectrogram = np.load(path)
        tensor = torch.from_numpy(np.asarray(spectrogram, dtype=np.float32)).unsqueeze(0)
        return tensor, int(self.labels[index]), index


class _PKContextSampler(Sampler[list[int]]):
    """Yield batches of P individuals x K samples, spanning both contexts when possible."""

    def __init__(
        self,
        labels: np.ndarray,
        contexts: np.ndarray,
        p: int,
        k: int,
        steps: int,
        seed: int,
    ) -> None:
        self.labels = labels
        self.contexts = contexts
        self.p = p
        self.k = k
        self.steps = steps
        self.rng = np.random.default_rng(seed)
        self.by_label: dict[int, np.ndarray] = {
            int(label): np.where(labels == label)[0] for label in np.unique(labels)
        }
        self.eligible = [label for label, idx in self.by_label.items() if len(idx) >= 2]

    def __len__(self) -> int:
        return self.steps

    def _pick_for_label(self, label: int) -> list[int]:
        idx = self.by_label[label]
        ctx = self.contexts[idx]
        unique_ctx = np.unique(ctx)
        chosen: list[int] = []
        if len(unique_ctx) >= 2:
            half = self.k // 2
            for ci, want in zip(unique_ctx, (half, self.k - half)):
                pool = idx[ctx == ci]
                replace = len(pool) < want
                chosen.extend(self.rng.choice(pool, size=want, replace=replace).tolist())
        else:
            replace = len(idx) < self.k
            chosen.extend(self.rng.choice(idx, size=self.k, replace=replace).tolist())
        return chosen

    def __iter__(self):
        labels_pool = self.eligible if self.eligible else list(self.by_label.keys())
        for _ in range(self.steps):
            p = min(self.p, len(labels_pool))
            selected = self.rng.choice(labels_pool, size=p, replace=False)
            batch: list[int] = []
            for label in selected:
                batch.extend(self._pick_for_label(int(label)))
            yield batch


def _embed(model: CricketCNN, x: torch.Tensor) -> torch.Tensor:
    """128-d L2-normalised embedding (conv features before the classification head)."""
    feats = model.features(x)
    feats = torch.flatten(feats, 1)
    return F.normalize(feats, p=2, dim=1)


def _batch_all_triplet_loss(embeddings: torch.Tensor, labels: torch.Tensor, margin: float) -> torch.Tensor:
    """Batch-all triplet loss: mean over all positive-violating (a, p, n) triplets.

    Batch-all provides dense gradients from every valid triplet, which avoids the
    degenerate all-embeddings-equal collapse that batch-hard mining falls into on
    small/noisy datasets.
    """
    dist = torch.cdist(embeddings, embeddings, p=2)
    labels = labels.view(-1, 1)
    same = labels == labels.t()
    diff = ~same
    eye = torch.eye(dist.size(0), dtype=torch.bool, device=dist.device)
    pos_mask = same & ~eye  # (a, p)

    # triplet[a, p, n] = d(a, p) - d(a, n) + margin
    anchor_pos = dist.unsqueeze(2)
    anchor_neg = dist.unsqueeze(1)
    triplet = anchor_pos - anchor_neg + margin

    valid = pos_mask.unsqueeze(2) & diff.unsqueeze(1)  # (a, p, n)
    triplet = F.relu(triplet) * valid.float()
    num_positive = (triplet > 1e-16).sum()
    if num_positive == 0:
        return embeddings.sum() * 0.0
    return triplet.sum() / (num_positive + 1e-16)


def _attach_context(meta_df: pd.DataFrame, recordings_df: pd.DataFrame | None) -> np.ndarray:
    """Return a context code array aligned with meta_df rows (0 if unknown)."""
    if recordings_df is None or "daytime" not in recordings_df.columns:
        # fall back to session-derived context if recordings unavailable
        return np.zeros(len(meta_df), dtype=np.int64)
    lookup = recordings_df.loc[:, ["recording_id", "daytime"]].copy()
    lookup["recording_id"] = lookup["recording_id"].astype(str)
    merged = meta_df.merge(lookup.drop_duplicates("recording_id"), on="recording_id", how="left")
    codes, _ = pd.factorize(merged["daytime"].fillna("unknown"))
    return np.asarray(codes, dtype=np.int64)


def _compute_embeddings(model: CricketCNN, loader: DataLoader, device: str) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    embs: list[np.ndarray] = []
    order: list[int] = []
    with torch.no_grad():
        for tensors, _labels, indices in loader:
            tensors = tensors.to(device)
            embs.append(_embed(model, tensors).cpu().numpy().astype(np.float32))
            order.extend([int(i) for i in indices])
    if not embs:
        return np.empty((0, 0), dtype=np.float32), np.asarray(order, dtype=np.int64)
    return np.concatenate(embs, axis=0), np.asarray(order, dtype=np.int64)


def _context_split_similarity(
    embeddings: np.ndarray,
    individual_ids: np.ndarray,
    recording_ids: np.ndarray,
    contexts: np.ndarray,
    n_pairs: int,
    seed: int,
) -> dict:
    """Mean same-individual cosine similarity for within-context vs cross-context pairs."""
    rng = np.random.default_rng(seed)
    normed = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    individuals = np.unique(individual_ids)
    by_ind = {ind: np.where(individual_ids == ind)[0] for ind in individuals}
    within: list[float] = []
    cross: list[float] = []
    attempts = 0
    while (len(within) < n_pairs or len(cross) < n_pairs) and attempts < n_pairs * 40:
        attempts += 1
        ind = individuals[rng.integers(len(individuals))]
        idx = by_ind[ind]
        if len(idx) < 2:
            continue
        a, b = idx[rng.integers(len(idx))], idx[rng.integers(len(idx))]
        if recording_ids[a] == recording_ids[b]:
            continue
        sim = float(np.dot(normed[a], normed[b]))
        if contexts[a] == contexts[b]:
            if len(within) < n_pairs:
                within.append(sim)
        else:
            if len(cross) < n_pairs:
                cross.append(sim)
    return {
        "within_context_mean_sim": float(np.mean(within)) if within else None,
        "cross_context_mean_sim": float(np.mean(cross)) if cross else None,
        "within_pairs": len(within),
        "cross_pairs": len(cross),
    }


def _plot_history(history: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    axes[0].plot(epochs, history["train_loss"], color="#7aa7ff", linewidth=1.8)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Triplet loss"); axes[0].set_title("Triplet training loss")
    axes[1].plot(epochs, [np.nan if v is None else v for v in history["val_auc"]],
                 color="#4ec9b0", linewidth=1.8, label="val verification AUC")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("AUC"); axes[1].set_ylim(0.0, 1.05)
    axes[1].set_title("Validation verification AUC"); axes[1].legend(frameon=False)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _build_meta(spectrogram_index_df: pd.DataFrame, split_dict: dict, split_name: str,
                segments_df: pd.DataFrame) -> pd.DataFrame:
    segment_ids = [str(s) for s in split_dict.get(split_name, [])]
    idx = spectrogram_index_df.copy()
    idx["segment_id"] = idx["segment_id"].astype(str)
    idx = idx[idx["segment_id"].isin(segment_ids)]
    rec = segments_df.loc[:, ["segment_id", "recording_id"]].copy()
    rec["segment_id"] = rec["segment_id"].astype(str)
    rec["recording_id"] = rec["recording_id"].astype(str)
    meta = idx.merge(rec.drop_duplicates("segment_id"), on="segment_id", how="left")
    meta["individual_id"] = meta["individual_id"].astype(str)
    meta["session_id"] = meta["session_id"].astype(str)
    meta["recording_id"] = meta["recording_id"].fillna(meta["segment_id"])
    return meta.reset_index(drop=True)


def _cap_per_class(meta: pd.DataFrame, cap: int | None, seed: int) -> pd.DataFrame:
    if not cap or cap <= 0:
        return meta
    frames = []
    for _, g in meta.groupby("individual_id", sort=True):
        frames.append(g if len(g) <= cap else g.sample(n=cap, random_state=seed))
    return pd.concat(frames, ignore_index=True)


def train_triplet_embedding(
    spectrogram_index_df: pd.DataFrame,
    segments_df: pd.DataFrame,
    split_dict: dict,
    config: dict,
    *,
    project_root: str | Path | None = None,
    recordings_df: pd.DataFrame | None = None,
) -> dict:
    seed = int(config.get("seed", 0))
    cnn_config = config.get("cnn", {})
    _configure_torch_runtime(cnn_config)
    stabilize_torch_backend()
    # Default to single-threaded: repeated train/eval maxpool forwards trigger a
    # native access violation in some Windows torch builds under multithreading.
    if cnn_config.get("torch_num_threads") in (None, ""):
        torch.set_num_threads(1)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    tri = config.get("triplet", {}) or {}
    p = int(tri.get("p_individuals", 8))
    k = int(tri.get("k_per_individual", 4))
    steps = int(tri.get("steps_per_epoch", 80))
    epochs = int(tri.get("epochs", 15))
    patience = int(tri.get("patience", 5))
    margin = float(tri.get("margin", 0.3))
    lr = float(tri.get("lr", 1e-3))
    cap = int(tri.get("max_train_samples_per_class", 300))
    eval_pairs = int(tri.get("verification_pairs", 4000))
    dropout = float(cnn_config.get("dropout", 0.3))

    root = resolve_project_root(config, project_root=project_root)
    data_dir = resolve_data_dir(config, project_root=root)
    artifacts_dir = _resolve_artifacts_dir(config, project_root=root)
    device = _select_device(cnn_config)

    # recordings for context (load if not provided)
    if recordings_df is None:
        from cricket_id.io.manifest import _read_parquet_compatible

        rec_path = data_dir / "interim" / "recordings_validated.parquet"
        if rec_path.exists():
            recordings_df = _read_parquet_compatible(rec_path)

    train_meta = _build_meta(spectrogram_index_df, split_dict, "train", segments_df)
    val_meta = _build_meta(spectrogram_index_df, split_dict, "val", segments_df)
    test_meta = _build_meta(spectrogram_index_df, split_dict, "test", segments_df)
    if train_meta.empty:
        raise ValueError("Train split is empty; cannot train triplet embedding.")

    train_meta = _cap_per_class(train_meta, cap, seed).reset_index(drop=True)

    individuals = sorted(train_meta["individual_id"].unique().tolist())
    if len(individuals) < 2:
        raise ValueError("Need at least two individuals to train a triplet embedding.")
    label_to_idx = {ind: i for i, ind in enumerate(individuals)}

    train_labels = train_meta["individual_id"].map(label_to_idx).to_numpy(dtype=np.int64)
    train_contexts = _attach_context(train_meta, recordings_df)

    train_ds = _TripletDataset(train_meta["segment_id"].tolist(), train_labels, data_dir)
    sampler = _PKContextSampler(train_labels, train_contexts, p, k, steps, seed)
    train_loader = DataLoader(train_ds, batch_sampler=sampler)

    def _eval_loader(meta: pd.DataFrame) -> DataLoader:
        labels = meta["individual_id"].map(lambda v: label_to_idx.get(v, -1)).to_numpy(dtype=np.int64)
        ds = _TripletDataset(meta["segment_id"].tolist(), labels, data_dir)
        return DataLoader(ds, batch_size=int(cnn_config.get("batch_size", 64)), shuffle=False)

    val_loader = _eval_loader(val_meta)

    model = CricketCNN(num_classes=len(individuals), dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history: dict[str, list] = {"train_loss": [], "val_auc": []}
    best_auc = float("-inf")
    best_epoch = 0
    no_improve = 0
    checkpoint_path = artifacts_dir / "models" / "cnn_triplet.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    def _val_auc() -> float | None:
        if val_meta.empty:
            return None
        emb, order = _compute_embeddings(model, val_loader, device)
        if emb.size == 0:
            return None
        ind = val_meta["individual_id"].to_numpy()[order]
        rec = val_meta["recording_id"].to_numpy()[order]
        sims, labels = _sample_verification_pairs(emb, ind, rec, eval_pairs, seed)
        metrics = _verification_metrics(sims, labels)
        return metrics["auc"] if metrics.get("status") == "ok" else None

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for tensors, labels, _indices in train_loader:
            tensors = tensors.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            emb = _embed(model, tensors)
            loss = _batch_all_triplet_loss(emb, labels, margin)
            if loss.requires_grad:
                loss.backward()
                optimizer.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        history["train_loss"].append(float(epoch_loss / max(n_batches, 1)))
        val_auc = _val_auc()
        history["val_auc"].append(val_auc)

        score = -1.0 if val_auc is None else val_auc
        if score > best_auc:
            best_auc = score
            best_epoch = epoch
            no_improve = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": "custom",
                    "dropout": dropout,
                    "num_classes": len(individuals),
                    "label_classes": individuals,
                    "best_epoch": epoch,
                    "loss": "triplet",
                },
                checkpoint_path,
            )
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    # reload best
    if checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

    # final verification on val + test, plus within/cross-context similarity
    final: dict[str, dict] = {}
    for split_name, meta in (("val", val_meta), ("test", test_meta)):
        if meta.empty:
            final[split_name] = {"status": "empty"}
            continue
        emb, order = _compute_embeddings(model, _eval_loader(meta), device)
        ind = meta["individual_id"].to_numpy()[order]
        rec = meta["recording_id"].to_numpy()[order]
        ctx = _attach_context(meta, recordings_df)[order]
        sims, labels = _sample_verification_pairs(emb, ind, rec, eval_pairs, seed)
        verification = _verification_metrics(sims, labels)
        ctx_split = _context_split_similarity(emb, ind, rec, ctx, eval_pairs // 2, seed)
        final[split_name] = {"verification": verification, "context_similarity": ctx_split}

    figures_dir = artifacts_dir / "figures"
    _plot_history(history, figures_dir / "cnn_triplet_history.png")

    results = {
        "model": "cnn_triplet",
        "loss": "triplet",
        "seed": seed,
        "device": device,
        "num_individuals": len(individuals),
        "settings": {
            "p_individuals": p, "k_per_individual": k, "steps_per_epoch": steps,
            "epochs": epochs, "margin": margin, "lr": lr,
            "max_train_samples_per_class": cap,
        },
        "best_epoch": best_epoch,
        "epochs_ran": len(history["train_loss"]),
        "history": history,
        "final": final,
    }

    metrics_path = artifacts_dir / "metrics" / "cnn_triplet_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    # report
    reports_dir = resolve_reports_dir(config, project_root=root)
    _write_report(results, reports_dir / "cnn_triplet.md")

    results["artifact_paths"] = {
        "checkpoint": str(checkpoint_path),
        "metrics": str(metrics_path),
    }
    return results


def _write_report(results: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Triplet Metric-Learning Embedding\n"]
    lines.append(
        "Batch-hard triplet loss with context-aware PK sampling. Goal: a "
        "context-invariant individual embedding that survives the day/night switch "
        "that the cross-entropy CNN failed.\n"
    )
    s = results["settings"]
    lines.append("## Settings\n")
    lines.append("| Field | Value |")
    lines.append("| --- | --- |")
    lines.append(f"| Individuals | {results['num_individuals']} |")
    lines.append(f"| P x K | {s['p_individuals']} x {s['k_per_individual']} |")
    lines.append(f"| Steps/epoch | {s['steps_per_epoch']} |")
    lines.append(f"| Margin | {s['margin']} |")
    lines.append(f"| Best epoch | {results['best_epoch']} of {results['epochs_ran']} |")
    lines.append("")
    lines.append("## Verification & context invariance\n")
    lines.append("| Split | AUC | EER | same-cosine (within ctx) | same-cosine (cross ctx) |")
    lines.append("| --- | --- | --- | --- | --- |")
    for split_name in ("val", "test"):
        block = results["final"].get(split_name, {})
        ver = block.get("verification", {})
        ctx = block.get("context_similarity", {})
        auc = ver.get("auc")
        eer = ver.get("eer")
        wc = ctx.get("within_context_mean_sim")
        cc = ctx.get("cross_context_mean_sim")
        lines.append(
            f"| {split_name} | {'n/a' if auc is None else f'{auc:.3f}'} | "
            f"{'n/a' if eer is None else f'{eer:.3f}'} | "
            f"{'n/a' if wc is None else f'{wc:.3f}'} | "
            f"{'n/a' if cc is None else f'{cc:.3f}'} |"
        )
    lines.append("")
    lines.append("![Triplet training](../artifacts/figures/cnn_triplet_history.png)\n")
    lines.append("## Interpretation\n")
    lines.append(
        "- A small gap between within-context and cross-context same-individual cosine means "
        "the embedding generalises across the diel context — the goal of context-aware metric learning.\n"
        "- Compare verification AUC here against `cnn_embeddings.json` (cross-entropy embedding); "
        "an increase shows metric learning produced a stronger identity space.\n"
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
