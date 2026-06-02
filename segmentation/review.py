from __future__ import annotations

import os
from pathlib import Path

_MPL_CONFIG_DIR = Path.cwd() / ".matplotlib"
_MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG_DIR))

import matplotlib
import numpy as np
import pandas as pd
import soundfile as sf

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cricket_id.features.spectrograms import compute_log_mel_spectrogram
from cricket_id.utils.paths import resolve_data_dir, resolve_reports_dir


def _segment_audio_path(segment_id: str, data_dir: Path) -> Path:
    return data_dir / "processed" / "segments_wav" / f"{segment_id}.wav"


def generate_segment_review(
    segments_df: pd.DataFrame,
    config: dict,
    n_samples: int = 50,
) -> None:
    data_dir = resolve_data_dir(config)
    reports_dir = resolve_reports_dir(config)
    plots_dir = reports_dir / "segment_review_plots"
    reports_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    report_path = reports_dir / "segment_review.md"
    if segments_df.empty:
        report_path.write_text(
            "# Segment Review\n\nNo segments available for review.\n",
            encoding="utf-8",
        )
        return

    sample_size = min(int(n_samples), len(segments_df))
    sampled_df = segments_df.sample(
        n=sample_size,
        random_state=int(config.get("seed", 42)),
    ).reset_index(drop=True)

    lines = [
        "# Segment Review",
        "",
        f"Reviewed {sample_size} randomly sampled segments out of {len(segments_df)} total.",
        "",
    ]

    for row in sampled_df.itertuples(index=False):
        segment_path = _segment_audio_path(str(row.segment_id), data_dir)
        audio, sr = sf.read(segment_path, always_2d=False)
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        spectrogram = compute_log_mel_spectrogram(audio, sr, config)

        time_axis = np.arange(audio.shape[0]) / sr
        figure, axes = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)
        axes[0].plot(time_axis, audio, linewidth=0.8)
        axes[0].set_title(f"Waveform: {row.segment_id}")
        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel("Amplitude")
        axes[1].imshow(spectrogram, aspect="auto", origin="lower", cmap="magma")
        axes[1].set_title("Normalized Log-Mel Spectrogram")
        axes[1].set_xlabel("Frame")
        axes[1].set_ylabel("Mel Bin")

        plot_path = plots_dir / f"{row.segment_id}.png"
        figure.savefig(plot_path, dpi=150)
        plt.close(figure)

        lines.extend(
            [
                f"## {row.segment_id}",
                "",
                "| Field | Value |",
                "| --- | --- |",
                f"| recording_id | {row.recording_id} |",
                f"| individual_id | {row.individual_id} |",
                f"| session_id | {row.session_id} |",
                f"| start_s | {float(row.start_s):.4f} |",
                f"| end_s | {float(row.end_s):.4f} |",
                f"| duration_s_raw | {float(row.duration_s_raw):.4f} |",
                f"| duration_s_fixed | {float(row.duration_s_fixed):.4f} |",
                f"| temperature_c | {float(row.temperature_c):.3f} |",
                f"| snr_proxy | {float(row.snr_proxy):.3f} |",
                f"| qc_flag | {row.qc_flag} |",
                "",
                f"![{row.segment_id}](segment_review_plots/{row.segment_id}.png)",
                "",
            ]
        )

    report_path.write_text("\n".join(lines), encoding="utf-8")
