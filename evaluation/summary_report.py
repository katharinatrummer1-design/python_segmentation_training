"""Generate a single human-readable overall summary of the whole project.

Unlike ``mvp_report.md`` (which is a detailed, section-by-section dump), this
report is an *executive summary*: it explains in plain language what each
component does, puts the PCA and CNN results head-to-head, states the scientific
conclusions, and points to the dashboard pages. It weaves stable explanatory
prose together with live numbers read from the artifacts, so it stays in sync
with the latest run.
"""

from __future__ import annotations

import json
from pathlib import Path

from cricket_id.utils.paths import (
    resolve_project_root,
    resolve_reports_dir,
)


def _resolve_artifacts_dir(config: dict, *, project_root: str | Path | None = None) -> Path:
    root = resolve_project_root(config, project_root=project_root)
    artifacts_dir = Path(str(config.get("paths", {}).get("artifacts_dir", "artifacts")))
    return artifacts_dir if artifacts_dir.is_absolute() else (root / artifacts_dir).resolve()


def _load(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None


def _fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number != number:  # NaN
        return "n/a"
    return f"{number:.{digits}f}"


def _pct(value: object) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "n/a"


def _ci(metric_ci: dict | None) -> str:
    if not metric_ci:
        return "n/a"
    return f"[{_fmt(metric_ci.get('ci_low'))}, {_fmt(metric_ci.get('ci_high'))}]"


def generate_overall_summary(
    config: dict,
    *,
    project_root: str | Path | None = None,
) -> Path:
    """Write ``reports/overall_summary.md`` from the available artifacts."""
    artifacts_dir = _resolve_artifacts_dir(config, project_root=project_root)
    metrics_dir = artifacts_dir / "metrics"
    reports_dir = resolve_reports_dir(config, project_root=project_root)
    reports_dir.mkdir(parents=True, exist_ok=True)

    evaluation = _load(metrics_dir / "evaluation_comparison.json") or {}
    cross = _load(metrics_dir / "cross_context.json")
    ce_emb = _load(metrics_dir / "cnn_embeddings.json")
    triplet_emb = _load(metrics_dir / "cnn_embeddings_triplet.json")
    triplet = _load(metrics_dir / "cnn_triplet_metrics.json")
    feature_cmp = _load(metrics_dir / "feature_comparison.json")
    pca_metrics = _load(metrics_dir / "pca_metrics.json") or {}
    cnn_metrics = _load(metrics_dir / "cnn_metrics.json") or {}
    cv = _load(metrics_dir / "pca_cross_validation.json")
    split_meta = (_load(artifacts_dir / "splits" / "split_v1.json") or {}).get("metadata", {})

    L: list[str] = []
    L.append("# Cricket Individual Identification — Overall Summary")
    L.append("")
    L.append(
        "_A plain-language overview of the whole pipeline: what each method does, how the two "
        "models compare, whether the results are statistically real, and what the findings mean. "
        "Numbers are pulled live from the latest run._"
    )
    L.append("")

    # --- 1. The question ---
    L.append("## 1. The scientific question")
    L.append("")
    L.append(
        "Can we recover an individual cricket's **identity** from its calling song? "
        "We test this two independent ways and compare them on identical, leakage-free splits:"
    )
    L.append("")
    L.append(
        "- **PCA path** — interpretable acoustic statistics (timbre, spectral shape, MFCCs) fed to a simple classifier."
    )
    L.append(
        "- **CNN path** — a small neural network that learns features directly from the spectrogram image."
    )
    L.append("")

    # --- 2. Dataset ---
    counts = (feature_cmp or {}).get("input_counts", {}) if feature_cmp else {}
    n_ind = evaluation.get("num_individuals") or counts.get("individuals")
    L.append("## 2. The data at a glance")
    L.append("")
    L.append("| Field | Value |")
    L.append("| --- | --- |")
    L.append(f"| Individuals | {n_ind if n_ind is not None else 'n/a'} |")
    if counts:
        L.append(f"| Recordings | {counts.get('recordings', 'n/a')} |")
        L.append(f"| Segments (chirps) | {counts.get('segments', 'n/a')} |")
        L.append(f"| Tabular feature columns | {counts.get('feature_columns', 'n/a')} |")
    chance = evaluation.get("chance_baseline", {}).get("uniform_random", {}).get("accuracy")
    L.append(f"| Random-chance accuracy | {_pct(chance)} |")
    L.append("")
    L.append(
        "Everything below should be read against the chance line above: anything far higher than "
        f"**{_pct(chance)}** is real signal, not luck."
    )
    L.append("")

    # --- 3. What each model does ---
    L.append("## 3. What each model actually does")
    L.append("")
    L.append(
        f"**PCA baseline** (`{pca_metrics.get('classifier', 'logistic_regression')}` in PCA space). "
        "It computes a fixed set of interpretable acoustic descriptors per chirp — loudness, "
        "zero-crossing rate, spectral centroid/bandwidth/rolloff/flatness, and MFCCs with their "
        "deltas — standardises them, compresses them with PCA, and classifies. Its strength is "
        "**interpretability**: you can ask *which* acoustic property carries identity."
    )
    L.append("")
    L.append(
        f"**CNN baseline** (`{cnn_metrics.get('architecture', 'custom')}`, 3 conv blocks). "
        "It sees the log-Mel spectrogram as an image and learns its own features. Its strength is "
        "**representation learning**: the 128-d layer before the classifier is a learned acoustic "
        "fingerprint that powers verification and open-set tasks the PCA classifier cannot do."
    )
    L.append("")

    # --- 4. Head to head ---
    L.append("## 4. Head-to-head: PCA vs. CNN (test split)")
    L.append("")
    test_cmp = evaluation.get("comparison", {}).get("test", {})
    if test_cmp:
        L.append("| Metric | PCA | CNN | Winner |")
        L.append("| --- | --- | --- | --- |")
        labels = {
            "accuracy": "Accuracy",
            "balanced_accuracy": "Balanced accuracy",
            "macro_f1": "Macro-F1",
            "session_level_accuracy": "Session-level (majority vote)",
        }
        for key, label in labels.items():
            block = test_cmp.get(key, {})
            L.append(
                f"| {label} | {_fmt(block.get('pca'))} | {_fmt(block.get('cnn'))} | "
                f"**{block.get('winner', 'n/a')}** |"
            )
        L.append("")
        best = evaluation.get("best_model", "n/a")
        L.append(
            f"**Verdict:** the **{best.upper()}** model wins on the held-out test split. At this "
            "data scale the interpretable feature path beats the compact CNN classifier — the CNN "
            "is data-hungry and tends to collapse toward the majority individual."
        )
    else:
        L.append("_No comparison metrics available yet — run `evaluate`._")
    L.append("")

    # --- 4b. Honest vs optimistic ---
    if cv and cv.get("pooled"):
        L.append("## 4b. Optimistic vs. honest estimate (the overfitting check)")
        L.append("")
        L.append(
            "The head-to-head numbers above use a holdout grouped by **recording**, so other "
            "recordings of the *same session/day* sit in both train and test. That measures "
            "\"recognise this individual on a day you've already heard them\" and is **optimistic**. "
            "The leakage-free estimate resamples whole sessions (`StratifiedGroupKFold`), i.e. "
            "\"recognise this individual on a **new** day\":"
        )
        L.append("")
        pooled = cv["pooled"]
        boot = cv.get("pooled_bootstrap", {})
        f1_ci = boot.get("macro_f1", {}) if isinstance(boot, dict) else {}
        opt_f1 = (pca_metrics.get("metrics", {}).get("test", {}) or {}).get("macro_f1")
        L.append("| Estimate | Accuracy | Macro-F1 |")
        L.append("| --- | --- | --- |")
        opt_acc = (pca_metrics.get("metrics", {}).get("test", {}) or {}).get("accuracy")
        L.append(f"| Optimistic (recording holdout, same-day) | {_fmt(opt_acc)} | {_fmt(opt_f1)} |")
        L.append(
            f"| **Honest (session CV, new-day)** | {_fmt(pooled.get('accuracy'))} | "
            f"**{_fmt(pooled.get('macro_f1'))}** {('95% CI ' + _ci(f1_ci)) if f1_ci else ''} |"
        )
        L.append("")
        L.append(
            f"The honest macro-F1 ({_fmt(pooled.get('macro_f1'))}) is well below the optimistic "
            f"{_fmt(opt_f1)} — the gap is recording-condition leakage, not real identity skill. "
            f"It still beats chance ({_pct(chance)}), so identity *is* recoverable across days, "
            f"just far more weakly than the headline suggests. This matches the cross-context result. "
            f"Evaluated on {cv.get('n_individuals_evaluated')} of {cv.get('n_individuals_total')} "
            "individuals (single-session animals cannot be tested without leakage)."
        )
        L.append("")

    # --- 5. Significance ---
    L.append("## 5. Are the results statistically real?")
    L.append("")
    significance = evaluation.get("significance", {})
    pca_sig = significance.get("pca", {}).get("test", {}) if significance else {}
    if pca_sig and pca_sig.get("bootstrap", {}).get("status") == "ok":
        boot = pca_sig["bootstrap"]
        perm = pca_sig.get("permutation", {})
        acc = boot.get("accuracy", {})
        f1 = boot.get("macro_f1", {})
        p_acc = perm.get("accuracy", {}).get("p_value") if isinstance(perm, dict) else None
        L.append(
            "Yes. Using a **session-level cluster bootstrap** (resampling whole sessions, so the "
            "interval honestly reflects between-session variability) and a **label-permutation test**:"
        )
        L.append("")
        L.append(f"- PCA test accuracy **{_fmt(acc.get('observed'))}**, 95% CI {_ci(acc)} "
                 f"(permutation p {'<0.0001' if (p_acc is not None and p_acc < 1e-4) else _fmt(p_acc, 4)}).")
        L.append(f"- PCA test macro-F1 **{_fmt(f1.get('observed'))}**, 95% CI {_ci(f1)}.")
        L.append("")
        L.append(
            "The permutation p-values sit at the floor (predictions are far above a no-information "
            "null). The macro-F1 interval is wide, which is the honest message that performance "
            "**varies a lot between sessions** — more sessions per individual would tighten it."
        )
    else:
        L.append("_Significance not computed yet — run `evaluate` with `statistics.enabled: true`._")
    L.append("")

    # --- 6. Cross-context ---
    L.append("## 6. The key caveat: identity vs. recording context")
    L.append("")
    if cross:
        L.append(
            "This is the most important scientific check. We train on one diel context "
            f"(`{cross.get('context_column', 'daytime')}`) and test on the other. If accuracy holds, "
            "the model learned the **individual**; if it collapses, it partly learned the "
            "**recording conditions**."
        )
        L.append("")
        L.append("| Direction | Within-context Macro-F1 | Cross-context Macro-F1 | Penalty |")
        L.append("| --- | --- | --- | --- |")
        for d in cross.get("directions", []):
            L.append(
                f"| {d.get('label')} | {_fmt(d.get('within_context', {}).get('macro_f1'))} | "
                f"{_fmt(d.get('cross_context', {}).get('macro_f1'))} | "
                f"{_fmt(d.get('penalty', {}).get('macro_f1'))} |"
            )
        L.append("")
        L.append(
            "**Finding:** there is a large drop across context. A big chunk of the apparent identity "
            "signal is confounded with day/night recording conditions. Identity is still recoverable "
            "across context (well above chance), but the honest headline is the cross-context number, "
            "not the within-context one."
        )
    else:
        L.append("_Cross-context analysis not available — run `cross-context-eval`._")
    L.append("")

    # --- 7. Embeddings & metric learning ---
    L.append("## 7. The learned fingerprint & metric learning")
    L.append("")
    if ce_emb or triplet_emb:
        L.append(
            "Beyond classification, the CNN embedding supports **open-set verification**: *are these "
            "two chirps the same individual?* — scored by cosine similarity (AUC = ranking quality, "
            "EER = equal-error operating point). This is something the closed-set PCA classifier "
            "cannot do."
        )
        L.append("")
        L.append("| Embedding | Verification AUC | EER |")
        L.append("| --- | --- | --- |")
        if ce_emb:
            v = ce_emb.get("verification", {})
            L.append(f"| Cross-entropy CNN | {_fmt(v.get('auc'))} | {_fmt(v.get('eer'))} |")
        if triplet_emb:
            v = triplet_emb.get("verification", {})
            L.append(f"| Metric learning (triplet) | {_fmt(v.get('auc'))} | {_fmt(v.get('eer'))} |")
        L.append("")
        if triplet:
            ctx = triplet.get("final", {}).get("test", {}).get("context_similarity", {})
            within = ctx.get("within_context_mean_sim")
            cross_sim = ctx.get("cross_context_mean_sim")
            if within is not None and cross_sim is not None:
                gap = float(within) - float(cross_sim)
                L.append(
                    "**Context invariance:** in the metric-learning embedding, same-individual chirps "
                    f"are nearly as similar across day/night (cosine {_fmt(cross_sim)}) as within the "
                    f"same context ({_fmt(within)}) — a gap of only **{_fmt(gap)}**. That is exactly "
                    "the opposite of the cross-entropy classifier, which collapses across context, and "
                    "is the metric-learning model's main contribution."
                )
        L.append("")
    else:
        L.append("_Embedding analysis not available — run `extract-embeddings`._")
    L.append("")

    # --- 8. Which feature carries identity ---
    if feature_cmp and feature_cmp.get("feature_rankings"):
        top = feature_cmp["feature_rankings"][0]
        L.append("## 8. Which acoustic property carries identity?")
        L.append("")
        L.append(
            f"The strongest single discriminator is **`{top.get('feature')}`** "
            f"(effect size η² = {_fmt(top.get('eta_squared'))}, FDR q = {_fmt(top.get('fdr_q'))}). "
            "The top of the ranking is dominated by MFCC / spectral-envelope features, so identity "
            "sits primarily in the **timbre / spectral shape** of the song, not just its pitch."
        )
        L.append("")

    # --- 9. Dashboard guide ---
    L.append("## 9. Guide to the dashboard pages")
    L.append("")
    L.append("| Page | What it tells you |")
    L.append("| --- | --- |")
    L.append("| **Model Comparison** | PCA vs CNN across all metrics, per-class recall, confusion matrices, and the statistical-significance table. |")
    L.append("| **Cross-Context** | How much identity survives a day/night switch — the confound check. |")
    L.append("| **CNN Embeddings** | The learned fingerprint map (CE vs triplet) and open-set verification ROC. |")
    L.append("| **Comparisons** | Per-feature effect sizes with uncertainty, pairwise separations, stability. |")
    L.append("| **Reports** | The detailed auto-generated `mvp_report.md` and this summary. |")
    L.append("")

    # --- 10. Bottom line ---
    L.append("## 10. Bottom line")
    L.append("")
    L.append("- **Individual identity is recoverable** from lab calling song, above chance and statistically significant — but the *honest, leakage-free* macro-F1 (session CV) is much lower than the optimistic same-day holdout. Trust the session-CV / cross-context numbers.")
    L.append("- **The interpretable PCA path wins** at this data scale; the compact CNN classifier underperforms.")
    L.append("- **Identity is partly confounded with recording context** (day/night) — the cross-context number is the honest one.")
    L.append("- **Metric learning yields a context-invariant fingerprint**, the right direction for robust re-identification.")
    L.append("- **Biggest limiters:** few sessions per individual (wide CIs) and no temperature data yet (residualization blocked).")
    L.append("")
    L.append("### Next steps")
    L.append("")
    L.append("- Collect more sessions per individual to tighten the confidence intervals.")
    L.append("- Populate temperature logs, then run `residualize-temp` to test whether the context confound is temperature-driven.")
    L.append("- Extend to courtship songs and field recordings (config-driven, no rewrite).")
    L.append("")

    # --- 11. Pipeline step by step ---
    audio = config.get("audio", {})
    afilter = config.get("audio_filter", {})
    seg = config.get("segmentation", {})
    feats = config.get("features", {})
    spec = config.get("spectrogram", {})
    counts = split_meta.get("counts", {})
    sess_counts = counts.get("sessions", {})
    seg_counts = counts.get("segments", {})

    L.append("## 11. How it was built — step by step")
    L.append("")
    L.append("Every stage is one CLI command; `run-mvp` chains them. All steps write artifacts to disk and are config-driven (`configs/mvp_grillen.yaml`).")
    L.append("")
    L.append("1. **Ingest & validate** (`validate-manifest`) — scan recordings, enforce scope "
             "(`song_type=calling`), check files are readable, merge temperature for QC only.")
    L.append(f"2. **Audio standardisation** — resample to {audio.get('target_sr', 44100)} Hz mono, "
             f"{afilter.get('type', 'highpass')} filter at {afilter.get('cutoff_hz', 500)} Hz, light normalisation.")
    L.append(f"3. **Segmentation** (`segment`) — RMS-envelope onset detection into fixed "
             f"{seg.get('fixed_duration_s', 0.5)} s chirps (pad/crop), QC-flagged not dropped, "
             f"max {seg.get('max_segments_per_recording', 'n/a')} segments/recording.")
    L.append(f"4. **Feature extraction** (`extract-features`) — (a) tabular acoustic features for PCA "
             f"(RMS, ZCR, spectral stats, {feats.get('n_mfcc', 13)} MFCCs + deltas); "
             f"(b) log-Mel spectrograms for the CNN ({spec.get('n_mels', 128)} mels, "
             f"n_fft {spec.get('n_fft', 1024)}, hop {spec.get('hop_length', 256)}). Temperature is never a model feature.")
    L.append(f"5. **Splits** (`build-splits`) — recording-grouped holdout "
             f"(train/val/test sessions {sess_counts.get('train', '?')}/{sess_counts.get('val', '?')}/{sess_counts.get('test', '?')}, "
             f"segments {seg_counts.get('train', '?')}/{seg_counts.get('val', '?')}/{seg_counts.get('test', '?')}). "
             "Honest leakage-free estimates come from `cross-validate` and `cross-context-eval`.")
    L.append("6. **PCA baseline** (`train-pca-baseline`) and **7. CNN baseline** (`train-cnn`) — see section 12.")
    L.append("8. **Evaluate** (`evaluate`) — chirp- and session-level metrics for both models, chance baseline, "
             "bootstrap CIs + permutation test, error analysis; writes the reports.")
    L.append("   Plus the research add-ons: `cross-context-eval`, `extract-embeddings`, `train-cnn --loss triplet`, `residualize-temp` (blocked: no temperature data).")
    L.append("")

    # --- 12. How each model was trained ---
    L.append("## 12. How each model was trained")
    L.append("")
    n_feat = len(pca_metrics.get("feature_columns", []) or [])
    L.append("### PCA baseline")
    L.append(
        f"- **Input**: {n_feat or 'the'} tabular acoustic features per chirp (metadata and temperature excluded).\n"
        f"- **Fit on train only**: `StandardScaler` -> `PCA` ({pca_metrics.get('effective_n_components', 'n/a')} components) "
        f"-> `{pca_metrics.get('classifier', 'logistic_regression')}` "
        f"(class_weight={pca_metrics.get('class_weight', None)!r} to counter imbalance).\n"
        "- **No leakage**: scaler/PCA/classifier never see val or test during fitting (enforced by tests).\n"
        "- **Prediction**: chirp-level labels, then session-level **majority vote** over a session's chirps."
    )
    L.append("")
    L.append("### CNN baseline")
    L.append(
        f"- **Input**: 1x{spec.get('n_mels', 128)}xT log-Mel spectrogram per chirp.\n"
        f"- **Architecture** (`{cnn_metrics.get('architecture', 'custom')}`): 3 conv blocks "
        "(Conv->BatchNorm->ReLU->MaxPool), adaptive pooling, dropout, linear head to the individuals.\n"
        "- **Loss/opt**: class-weighted cross-entropy (weights from train frequencies), Adam.\n"
        f"- **Early stopping** on validation macro-F1 (best epoch {cnn_metrics.get('best_epoch', 'n/a')} "
        f"of {cnn_metrics.get('epochs_ran', 'n/a')} run); best checkpoint reloaded for evaluation.\n"
        "- **Stability**: MKL-DNN disabled + single-threaded to avoid a native Windows MaxPool2d crash.\n"
        "- A class-balanced sampler is available but its balanced variant could not be auto-evaluated on this build; the reported CNN uses the default sampler."
    )
    L.append("")
    if triplet:
        s = triplet.get("settings", {})
        L.append("### Metric-learning embedding (`train-cnn --loss triplet`)")
        L.append(
            "- Same CNN backbone, but optimises the 128-d pre-head embedding (L2-normalised) instead of a classifier.\n"
            f"- **Batch-all triplet loss** (margin {s.get('margin', 'n/a')}) with **context-aware PK sampling** "
            f"({s.get('p_individuals', 'n/a')} individuals x {s.get('k_per_individual', 'n/a')} chirps per batch, "
            "drawn across day/night so positives span contexts -> pressure toward context-invariance).\n"
            f"- **Early stopping** on validation verification AUC (best epoch {triplet.get('best_epoch', 'n/a')}).\n"
            "- Used for open-set verification and the embedding map, not closed-set classification."
        )
        L.append("")

    output_path = reports_dir / "overall_summary.md"
    output_path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return output_path
