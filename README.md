# cricket_id — Individual-Identifikation von Grillen aus Gesang

Reproduzierbarer MVP, der prüft, ob sich **einzelne Grillen-Individuen an ihrem
Gesang erkennen** lassen. Aus Audioaufnahmen werden einzelne *Chirps* (Zirp-Laute)
segmentiert, in zwei parallele Repräsentationen überführt und mit zwei Modellen
verglichen:

1. **PCA-Baseline** — interpretierbare tabellarische Akustik-Features → `StandardScaler`
   → PCA → einfacher Klassifikator (Logistic Regression).
2. **CNN-Baseline** — Log-Mel-Spektrogramme → kompaktes Custom-CNN (optional ResNet18).

Beide Modelle werden auf **identischen, leckage-freien Splits** ausgewertet (Trennung
auf Session-/Recording-Ebene, nie auf Chirp-Ebene), inklusive Chirp-Level- und
Session-Level-Metriken und automatischem Markdown-Report.

> Ein einziger Befehl führt die komplette Pipeline aus:
> ```bash
> python -m cricket_id run-mvp --config configs/mvp_lab_calling.yaml
> ```

> **Hinweis zum Repository-Layout:** Dieses Git-Repository enthält das
> Python-Paket `cricket_id` (Verzeichnis `src/cricket_id/`). Das **vollständige
> lauffähige Projekt** — `configs/`, `tests/`, `pyproject.toml`, `data/`,
> `RESEARCH_CONTEXT.md`, `task.md`, `TASKS.md` — liegt im **übergeordneten
> Projektverzeichnis** (`kathi/`), in dem dieses Paket unter `src/cricket_id/`
> eingehängt ist. Alle Befehle in diesem README werden aus diesem Projekt-Root
> ausgeführt. `environment.yml` liegt diesem README bei.

---

## Inhaltsverzeichnis

- [Schnellstart](#schnellstart)
- [Installation](#installation)
- [Abhängigkeiten (und wofür sie genutzt werden)](#abhängigkeiten-und-wofür-sie-genutzt-werden)
- [Projektstruktur](#projektstruktur)
- [Datenvertrag (Manifest-Schema)](#datenvertrag-manifest-schema)
- [Konfiguration (YAML)](#konfiguration-yaml)
- [Die Pipeline Schritt für Schritt](#die-pipeline-schritt-für-schritt)
- [CLI-Kommandos](#cli-kommandos)
- [Erzeugte Artefakte](#erzeugte-artefakte)
- [Glossar: DSP-Begriffe, Syntax & Einheiten](#glossar-dsp-begriffe-syntax--einheiten)
- [Tests](#tests)
- [Plattform-Hinweise (Windows, offline, Parquet)](#plattform-hinweise-windows-offline-parquet)
- [Optionale Web-UI](#optionale-web-ui)

---

## Schnellstart

```bash
# 1. Umgebung erstellen (Conda, empfohlen)
#    environment.yml liegt im Paketverzeichnis src/cricket_id/
conda env create -f src/cricket_id/environment.yml
conda activate cricket-id

# 2. Paket installierbar einbinden (src/-Layout)
pip install -e .

# 3. CLI prüfen
python -m cricket_id --help

# 4. Tests
pytest -q

# 5. Komplette Pipeline
python -m cricket_id run-mvp --config configs/mvp_lab_calling.yaml
```

Alle Befehle werden aus dem **Projekt-Root** (`kathi/`, das Verzeichnis mit
`pyproject.toml` und `configs/`) ausgeführt.

---

## Installation

### Variante A — Conda (empfohlen)

Die Datei [`environment.yml`](environment.yml) (im Paketverzeichnis
`src/cricket_id/`) installiert Python 3.11 und alle Abhängigkeiten inkl. PyTorch
(CPU-Build):

```bash
conda env create -f src/cricket_id/environment.yml
conda activate cricket-id
pip install -e .          # macht das Paket `cricket_id` importierbar
```

### Variante B — pip / venv

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/mac: source .venv/bin/activate
pip install -e .          # liest die Abhängigkeiten aus pyproject.toml
pip install pytest        # nur für die Tests
```

> **GPU/CUDA:** `environment.yml` und `pyproject.toml` installieren standardmäßig
> den **CPU-Build** von `torch`. Für CUDA stattdessen das passende Wheel von
> <https://pytorch.org/get-started/locally/> installieren. Der CNN-Pfad läuft auf
> CPU; GPU wird automatisch genutzt, falls verfügbar (siehe `cnn.device`).

### Variante C — ohne Installation (Fallback)

Das Repo enthält bewusste **dependency-freie Fallbacks**, damit es auch ohne
Installation lauffähig bleibt:

- `sitecustomize.py` legt beim Python-Start `src/` auf den Importpfad → `cricket_id`
  ist importierbar, ohne `pip install`.
- `typer.py` (Projekt-Root) ist ein minimaler **Typer-Stub** (nur `Option`, `echo`,
  `Typer` mit Subcommands). Er greift nur, wenn das echte `typer` nicht installiert
  ist, und reicht aus, um `python -m cricket_id <command> --config ...` auszuführen.

Für den vollen Funktionsumfang (echte Audio-/ML-Bibliotheken) ist eine echte
Installation (Variante A/B) nötig.

---

## Abhängigkeiten (und wofür sie genutzt werden)

| Bibliothek | Zweck im Projekt |
| --- | --- |
| **numpy** | Numerisches Rückgrat: Framing, STFT-Nachverarbeitung, Mel-Filterbank, alle Array-Mathematik. |
| **pandas** | Manifeste, Segment-/Feature-Tabellen, Parquet-I/O, Gruppierungen für Splits & Session-Aggregation. |
| **scipy** | Signalverarbeitung: `scipy.signal.butter`/`sosfilt`/`sosfiltfilt` (Filter), `stft` (Spektrogramme), `resample_poly` (Resampling), `scipy.fft.dct` (MFCC). |
| **librosa** | Robustes Audio-Laden, Mono-Konvertierung (`to_mono`) und Resampling (`resample`) in `io/audio.py`. |
| **soundfile** | Lesen/Schreiben von WAV-Dateien (Quell-Audio und standardisierte Segment-WAVs). |
| **scikit-learn** | `StandardScaler`, `PCA`, `LogisticRegression`/`NearestCentroid`, `LabelEncoder`, Metriken (Accuracy, Balanced Accuracy, Macro-F1, Recall, Confusion Matrix), gruppenbasierte CV. |
| **torch** | CNN-Definition, Training, DataLoader (`models/cnn_*`). |
| **torchvision** | Optionales `resnet18`-Backbone (`--model resnet18`). |
| **matplotlib** | Alle PNG-Figuren (Explained Variance, PCA-Scatter, Confusion Matrices, Training-History). Backend `Agg` (headless, keine GUI). |
| **pyyaml** | Laden der YAML-Konfiguration (`utils/config.py`). |
| **typer** | CLI-Framework (`cli.py`). Fallback-Stub vorhanden (siehe oben). |
| **pytest** | Testsuite (`tests/`). Dev-Dependency. |

**Transitive Abhängigkeiten / „dependencies von anderen libs":**

- `librosa` zieht u. a. `numba`, `soxr`/`samplerate` (Resampling-Kernels),
  `audioread`, `pooch`, `scikit-learn`, `joblib` und `soundfile` nach.
- `soundfile` bindet die native **libsndfile** (über das `cffi`-Wheel mitgeliefert).
- `scikit-learn` benötigt `numpy`, `scipy`, `joblib`, `threadpoolctl`.
- `torch` bringt seine eigenen nativen BLAS/MKL-DNN-Kerne mit (siehe Windows-Hinweis).
- **Parquet-I/O** verlangt `pyarrow` *oder* `fastparquet`. Fehlt beides, fällt der
  Code automatisch auf Pickle zurück (siehe [Plattform-Hinweise](#plattform-hinweise-windows-offline-parquet)).

---

## Projektstruktur

```text
kathi/                              # Projekt-Root (Befehle von hier ausführen)
├── pyproject.toml                  # Paket-Metadaten + Dependencies + CLI-Entrypoint
├── sitecustomize.py                # legt src/ auf den Importpfad (Fallback ohne Install)
├── typer.py                        # minimaler Typer-Stub (Fallback ohne typer)
├── configs/
│   ├── mvp_lab_calling.yaml         # Akzeptanz-MVP: nur context=lab, song_type=calling
│   └── mvp_grillen*.yaml            # reale Gryllus-campestris-Daten (High-Pass, Diel-Kontext)
├── data/
│   ├── raw/                        # Eingaben: recordings.csv, temperature.csv, WAVs
│   ├── interim/                    # validierte Manifeste, Audio-Index, Segmente
│   └── processed/                  # Feature-Tabelle, Spektrogramme, Segment-WAVs
├── artifacts/
│   ├── splits/   models/   metrics/   figures/
├── reports/                        # generierte Markdown-Reports + QC-Plots
├── tests/                          # pytest (Schema, Leakage, NaNs, Shapes, Smoke …)
└── src/cricket_id/                 # das eigentliche Paket (dieses Git-Repo)
    ├── README.md                   # diese Dokumentation
    ├── environment.yml             # Conda-Umgebung
    ├── cli.py                      # Typer-App: registriert alle Kommandos
    ├── __main__.py                 # ermöglicht `python -m cricket_id`
    ├── io/                         # ingest.py, manifest.py, audio.py
    ├── preprocessing/              # temperature.py (QC-Merge), residualize.py
    ├── segmentation/               # segment.py (RMS-Envelope), review.py (QC-Stichprobe)
    ├── features/                   # tabular.py (PCA-Features), spectrograms.py (CNN-Input)
    ├── splits/                     # build_splits.py (session-/recording-Gruppierung)
    ├── models/                     # pca_baseline.py, cnn_model.py, cnn_trainer.py, cnn_dataset.py, triplet_trainer.py
    ├── evaluation/                 # evaluate.py, report.py, summary_report.py, cross_context.py, cross_validate.py, embeddings.py, statistics.py …
    └── utils/                      # config.py (YAML), paths.py (Pfad-Auflösung ohne Hardcoding)
```

---

## Datenvertrag (Manifest-Schema)

Die Pipeline **rät niemals** das Datenschema. Es werden zwei CSV-/Parquet-Manifeste
erwartet (Pfade in der Config unter `manifest:`).

### `recordings.csv` — Pflichtspalten

| Spalte | Bedeutung |
| --- | --- |
| `recording_id` | eindeutige Aufnahme-ID (Join-Schlüssel) |
| `individual_id` | Identität der Grille = **Klassenlabel** |
| `session_id` | Aufnahme-Session (Basis der Split-Trennung gegen Leakage) |
| `song_type` | z. B. `calling` / `courtship` (MVP filtert auf `calling`) |
| `context` | z. B. `lab` / `field` (MVP filtert auf `lab`) |
| `audio_path` | Pfad zur WAV-Datei (relativ zu `data/raw/` oder absolut) |
| `temperature_log_path` | Pfad zum Temperatur-Log (optional) |
| `recording_start_utc` | Startzeit (UTC) |
| `duration_s` | Dauer in **Sekunden** |
| `sr` | Original-Samplingrate in **Hz** |
| `bit_depth` | Bittiefe |
| `notes` | Freitext |

### `temperature.csv` — Pflichtspalten

`timestamp_utc`, `temperature_c` (Grad **Celsius**), `recording_id`.

### Harte Regeln

- **Scope-Filter:** nur `song_type == calling` **und** `context == lab` (konfigurierbar).
- **Mindest-Sessions:** Hat ein Individuum **< 3 unabhängige Sessions**, wird *nicht*
  trainiert, sondern `reports/data_gap_report.md` erzeugt
  (`mvp_scope.enforce_min_sessions: false` macht daraus eine Warnung).
- **Temperatur ist QC-only** — sie ist *nie* ein Klassifikationsfeature (siehe Glossar).

> Für den realen Datensatz erzeugt `python -m cricket_id build-manifest` das
> `recordings.csv` automatisch aus einem Roh-Ordnerbaum (`io/ingest.py`, gesteuert
> über den optionalen `ingest:`-Block in der Config).

---

## Konfiguration (YAML)

Alles Einstellbare lebt in YAML — **keine hartcodierten Pfade, ein fester Seed**.
Auszug aus [`configs/mvp_lab_calling.yaml`](configs/mvp_lab_calling.yaml) mit
**Einheiten** je Parameter:

```yaml
seed: 42                      # globaler Zufalls-Seed (Reproduzierbarkeit)

audio:
  target_sr: 44100            # interne Ziel-Samplingrate [Hz]

bandpass:                     # (Legacy) Bandpass-Grenzen für die Lab-Config
  fmin_hz: 2000               # untere Grenzfrequenz [Hz]
  fmax_hz: 8000               # obere Grenzfrequenz [Hz]

# Alternative: High-Pass statt Bandpass (mvp_grillen.yaml)
# audio_filter:
#   type: highpass            # highpass | bandpass | none
#   cutoff_hz: 500            # Grenzfrequenz [Hz]
#   order: 4                  # Filterordnung (Butterworth)
#   zero_phase: true          # true -> sosfiltfilt (keine Phasenverzerrung)

segmentation:
  min_duration_s: 0.05        # kürzeste akzeptierte Chirp-Dauer [s]
  max_duration_s: 2.0         # längste akzeptierte Chirp-Dauer [s]
  fixed_duration_s: 0.5       # standardisierte Segmentlänge [s]
  merge_gap_s: 0.02           # Lücken < diesem Wert werden verschmolzen [s]
  rms_frame_length: 1024      # RMS-Fensterlänge [Samples]
  rms_hop_length: 512         # RMS-Schrittweite [Samples]
  threshold_db_below_peak: -20 # Aktivitätsschwelle relativ zum Peak [dB]

temperature_qc:
  enabled: true               # QC-Filter an/aus
  target_temp_c: 24.0         # Soll-Temperatur [°C]
  temp_tolerance_c: 1.0       # Toleranz [°C] (Segment ok in target ± tolerance)

features:                     # für die tabellarischen PCA-Features
  n_fft: 1024                 # FFT-Länge [Samples]
  hop_length: 256             # Frame-Schrittweite [Samples]
  win_length: 1024            # Analysefensterlänge [Samples]
  n_mfcc: 13                  # Anzahl MFCC-Koeffizienten
  n_mels: 128                 # Anzahl Mel-Bänder
  fmin_hz: 2000               # untere Mel-Grenze [Hz]
  fmax_hz: 8000               # obere Mel-Grenze [Hz]

spectrogram:                  # für den CNN-Input
  segment_length_s: 0.5       # Segmentlänge [s]
  sr: 44100                   # Samplingrate [Hz]
  n_fft: 1024                 # FFT-Länge [Samples]
  hop_length: 256             # Frame-Schrittweite [Samples]
  n_mels: 128                 # Mel-Bänder -> Höhe des Spektrogramms
  power: 2.0                  # 1.0 = Magnitude |X|, 2.0 = Power |X|^2

splits:
  train_ratio: 0.7            # Anteil Train (über Sessions, nicht Chirps)
  val_ratio: 0.15
  test_ratio: 0.15
  # group_by: recording_id    # optional: gruppiere auf Recording- statt Session-Ebene

pca:
  n_components: 30            # Ziel-Dimensionen der PCA
  classifier: logistic_regression  # oder nearest_centroid
  # class_weight: balanced    # gegen Klassenungleichgewicht

cnn:
  model: custom               # custom | resnet18
  epochs: 50
  batch_size: 32
  lr: 0.001                   # Lernrate (Adam)
  patience: 10                # Early-Stopping-Geduld (Epochen ohne val-F1-Gewinn)
  dropout: 0.3
  num_classes: 15

mvp_scope:
  song_type: calling
  context: lab
  min_sessions_per_individual: 3
  # enforce_min_sessions: true  # false -> Warnung statt Abbruch
```

Weitere optionale Blöcke (in `mvp_grillen.yaml`): `ingest`, `cross_context`,
`cross_validation`, `statistics`, `embeddings`, `triplet`.

---

## Die Pipeline Schritt für Schritt

| # | Schritt | Modul | Output |
| --- | --- | --- | --- |
| 1 | **Manifest validieren** | `io/manifest.py` | `recordings_validated.parquet`, ggf. `data_gap_report.md` |
| 2 | **Audio laden + QC** | `io/audio.py` | `audio_index.parquet` (Original-SR, Peak, Clipping-Flag …) |
| 3 | **Temperatur mergen + QC** | `preprocessing/temperature.py` | `recordings_qc.parquet` (`temp_mean_c` …, `temp_qc_pass`) |
| 4 | **Filtern + Segmentieren** | `segmentation/segment.py` | `segments.parquet` + Segment-WAVs |
| 5 | **QC-Sichtprüfung** | `segmentation/review.py` | `reports/segment_review.md` (50 Stichproben, Wellenform + Spektrogramm) |
| 6 | **Tabellarische Features** | `features/tabular.py` | `features_tabular.parquet` (RMS, ZCR, Spektral-Features, MFCC + Delta) |
| 7 | **Spektrogramme** | `features/spectrograms.py` | `spectrograms/*.npy` + `spectrogram_index.parquet` |
| 8 | **Splits bauen** | `splits/build_splits.py` | `artifacts/splits/split_v1.json` (leckage-frei) |
| 9 | **PCA-Baseline** | `models/pca_baseline.py` | `scaler.pkl`, `pca.pkl`, `pca_classifier.pkl`, `pca_metrics.json`, Figuren |
| 10 | **CNN-Baseline** | `models/cnn_trainer.py` | `cnn_baseline.pt`, `cnn_metrics.json`, Confusion Matrix, Training-History |
| 11 | **Evaluation + Report** | `evaluation/` | `mvp_report.md`, `overall_summary.md`, Vergleich PCA vs. CNN |

### Wichtige Design-Punkte

- **Leakage-Schutz (kritisch):** Splits trennen **immer auf `session_id`**
  (bzw. `recording_id`), nie auf Chirp-Ebene. `validate_split_integrity()` prüft
  hart auf Session-/Recording-Overlap und dass jedes Individuum im Train vorkommt.
  PCA und CNN nutzen **dieselbe** `split_v1.json` → fairer Vergleich.
- **Train-only Fits:** `StandardScaler` und `PCA` werden ausschließlich auf den
  Train-Daten gefittet, dann auf Val/Test angewendet (kein Information-Leak).
- **Determinismus:** ein Seed (`config.seed`) steuert NumPy, Python-`random`,
  Torch und die Split-Zuteilung; `cudnn.deterministic=True`.
- **Session-Level-Metrik:** zusätzlich zur Chirp-Accuracy wird pro Session per
  **Mehrheitsentscheid** über alle Segmente ein Individuum vorhergesagt
  (reduziert Pseudoreplikation).

---

## CLI-Kommandos

```bash
python -m cricket_id --help
```

**Kern-Pipeline** (alle akzeptieren `--config <pfad.yaml>`):

| Kommando | Beschreibung |
| --- | --- |
| `build-manifest` | erzeugt `recordings.csv`/`temperature.csv` aus einem Roh-Ordnerbaum (nur mit `ingest:`-Block) |
| `validate-manifest` | prüft Schema, Dateien, Scope; schreibt validiertes Manifest, Audio-Index, Temp-QC |
| `segment` | filtert, segmentiert in Chirps, schreibt Segment-Tabelle + WAVs |
| `extract-features` | tabellarische Features **und** Spektrogramme |
| `build-splits` | leckage-freie Train/Val/Test-Splits |
| `train-pca-baseline` | Scaler + PCA + Klassifikator |
| `train-cnn` | Custom-CNN (oder `--model resnet18`); `--loss triplet` für Metric-Learning |
| `evaluate` | PCA vs. CNN, erzeugt `mvp_report.md` + `overall_summary.md` |
| `run-mvp` | **End-to-end**: führt 1–11 in einem Befehl aus |

**Forschungs-Erweiterungen** (entworfen für spätere Courtship-/Feld-Szenarien):

| Kommando | Beschreibung |
| --- | --- |
| `analyze-features` / `compare-features` | Feature-Signalstärke (η², FDR, Bootstrap-CIs) |
| `cross-context-eval` | Training auf einem Diel-Kontext (Tag/Nacht), Test auf dem anderen |
| `cross-validate` | session-gruppierte K-Fold-CV (ehrliche Generalisierung) |
| `extract-embeddings` | 128-d CNN-Embedding, Open-Set-Verifikation (ROC/AUC/EER), 2D-Map |
| `residualize-temp` | entfernt linearen Temperatureinfluss aus den Features |
| `summary` | regeneriert nur die Management-Zusammenfassung |

---

## Erzeugte Artefakte

Nach `run-mvp` liegen u. a. vor:

```text
data/interim/recordings_validated.parquet
data/interim/audio_index.parquet
data/interim/recordings_qc.parquet
data/interim/segments.parquet
data/processed/features_tabular.parquet
data/processed/spectrogram_index.parquet
data/processed/spectrograms/*.npy
data/processed/segments_wav/*.wav
artifacts/splits/split_v1.json
artifacts/models/{scaler,pca,pca_classifier,label_encoder}.pkl
artifacts/models/cnn_baseline.pt
artifacts/metrics/{pca_metrics,cnn_metrics,evaluation_comparison}.json
artifacts/figures/*.png
reports/mvp_report.md
reports/overall_summary.md
reports/segment_review.md
reports/data_gap_report.md           # nur bei zu wenigen Sessions
```

---

## Glossar: DSP-Begriffe, Syntax & Einheiten

Damit kein Parameter mehrdeutig bleibt, hier die zentralen Signalverarbeitungs-
Begriffe mit **Einheit** und Definition. Diese Werte sind im Code als Inline-`#`-
Kommentare an den Rechenstellen dokumentiert.

| Begriff | Einheit | Definition |
| --- | --- | --- |
| **Samplingrate `sr` / `target_sr`** | Hz (Samples/s) | Abtastrate. Intern überall auf `44100 Hz` resampled, mono. |
| **`n_fft`** | **Samples** | Länge des FFT-Fensters. `1024 @ 44.1 kHz ≈ 23,2 ms`. Frequenzauflösung ≈ `sr / n_fft ≈ 43 Hz/Bin`. |
| **`hop_length`** | **Samples** | Schrittweite zwischen aufeinanderfolgenden STFT-Frames. `256 @ 44.1 kHz ≈ 5,8 ms`. |
| **`win_length`** | **Samples** | Länge des Analysefensters (`nperseg`); `noverlap = win_length − hop_length`. |
| **STFT** | komplex | Short-Time Fourier Transform: Folge von FFTs über überlappende Frames → Matrix `(Frequenz-Bins × Frames)`. |
| **FFT-Bin / `freqs`** | Hz | Mittenfrequenz eines Frequenzkanals; reale FFT liefert `n_fft//2 + 1` Bins von `0 Hz` bis zur **Nyquist-Frequenz `sr/2`**. |
| **Nyquist-Frequenz** | Hz | `sr/2` = höchste darstellbare Frequenz. |
| **`power`** | — | `1.0` = Magnitude `|X|`, `2.0` = Power-Spektrum `|X|²`. |
| **Mel-Skala** | mel | Perzeptive Tonhöhenskala (HTK: `mel = 2595·log10(1 + Hz/700)`); unter ~1 kHz nahezu linear. |
| **Mel-Filterbank** | — | `n_mels` dreieckige Filter, gleichverteilt in mel → fasst FFT-Energie in `n_mels` Bänder zusammen. |
| **Log-Mel-Spektrogramm** | dB | Mel-Power → Dezibel (`10·log10`), pro Spektrogramm min-max auf `[0,1]` normiert (= CNN-Input, Form `n_mels × Frames`). |
| **dB (Dezibel)** | dB | Logarithmisches Verhältnis. **Amplitude/RMS:** `dB = 20·log10(ratio)`. **Power:** `dB = 10·log10(ratio)`. → `threshold_db_below_peak = −20 dB` entspricht `0,1·Peak` in Amplitude. |
| **RMS** | linear (Amplitude) | Root-Mean-Square je Frame: Lautstärke-/Energie-Proxy; Basis der Segmentierungs-Hüllkurve. |
| **ZCR (Zero-Crossing-Rate)** | dimensionslos `[0,1]` | Anteil der Vorzeichenwechsel im Frame; hoch ≈ viel Hochfrequenz/Rauschen. |
| **Spectral Centroid** | Hz | Magnituden-gewichteter Frequenz-Schwerpunkt („Helligkeit"). |
| **Spectral Bandwidth** | Hz | Gewichtete Standardabweichung der Frequenz um den Centroid. |
| **Spectral Roll-off (85 %)** | Hz | Frequenz, unter der 85 % der Spektralenergie liegen. |
| **Spectral Flatness** | dimensionslos `[0,1]` | Geometrisches/arithmetisches Mittel des Spektrums; `~1` rauschartig, `~0` tonal. |
| **`peak_freq_hz`** | Hz | Dominante Frequenz (zeitgemittelt) ≈ Trägerfrequenz des Chirps. |
| **MFCC** | — | Mel-Frequency Cepstral Coefficients: Typ-2-**DCT** des Log-Mel-Spektrums, erste `n_mfcc=13` Koeffizienten; dekorreliert die Mel-Bänder. |
| **Delta-MFCC** | — | Erste zeitliche Ableitung der MFCC-Trajektorie (zentrale Differenz). |
| **SNR-Proxy** | dB | `20·log10(Segment-RMS / Rausch-Floor)`; *Proxy* (Floor = 10. Perzentil der RMS-Hüllkurve), nur für QC. |
| **`fixed_duration_s`** | s | Standardlänge `0,5 s` je Segment. Kürzer → symmetrisches Zero-Padding; länger → symmetrischer Center-Crop. |
| **Butterworth-Filter `order`** | — | Filterordnung; höher = steilere Flanke. Designt als SOS (numerisch stabil). `zero_phase=true` → `sosfiltfilt` (vorwärts+rückwärts, keine Phasenverzerrung). |

---

## Tests

```bash
pytest -q
```

Verpflichtende Tests (in `tests/`):

| Test | prüft |
| --- | --- |
| `test_manifest_schema.py` | Pflichtspalten / Schema-Validierung |
| `test_no_split_leakage.py` | kein Session-/Recording-Overlap zwischen Splits |
| `test_feature_table_has_no_nans.py` | keine NaNs in Trainingsfeatures |
| `test_spectrogram_shape_consistency.py` | alle Spektrogramme exakt gleiche Form |
| `test_scaler_fit_train_only.py` | Scaler nur auf Train gefittet (schlägt bei Leakage fehl) |
| `test_pca_fit_train_only.py` | PCA nur auf Train gefittet |
| `test_tiny_batch_overfit_cnn.py` | CNN überfittet einen Mini-Batch (Lern-Sanity-Check) |
| `test_run_mvp_smoke.py` | kompletter End-to-end-Lauf auf Mini-Fixture-Datensatz |

Der Smoke-Test erzeugt seinen eigenen winzigen Fixture-Datensatz, es werden keine
echten Audiodaten benötigt.

---

## Plattform-Hinweise (Windows, offline, Parquet)

- **Windows + PyTorch (`MaxPool2d`-Crash):** Manche Windows-Torch-Builds stürzen im
  nativen MKL-DNN-MaxPool-Kernel ab. `models/cnn_model.py:stabilize_torch_backend()`
  deaktiviert daher MKL-DNN, und der Trainer läuft standardmäßig **single-threaded**
  (`torch.set_num_threads(1)`). Langsamer, aber stabil. Steuerbar über
  `cnn.torch_num_threads` / `cnn.device`.
- **Parquet ohne Engine:** `io/manifest.py` kapselt alle Parquet-Reads/Writes. Ist
  weder `pyarrow` noch `fastparquet` installiert, fällt der Code transparent auf
  **Pickle** (`.parquet`-Datei mit Pickle-Inhalt) zurück, damit die Pipeline nicht
  abbricht. Für echtes Parquet `pip install pyarrow`.
- **Keine harten Pfade:** `utils/paths.py` löst Projekt-Root, `data/`, `reports/`,
  `artifacts/` und Audio-Pfade relativ zur Config bzw. zum CWD auf — das Repo ist
  ortsunabhängig.
- **Headless-Plots:** matplotlib nutzt das `Agg`-Backend und legt seinen Cache unter
  `.matplotlib/` ab; es wird keine GUI/kein Display benötigt.

---

## Optionale Web-UI

Unter `ui/` liegt eine eigenständige **Vue 3 + Express**-Oberfläche
(`cricket-id-ui`) zum Browsen der erzeugten Reports/Artefakte. Sie ist **nicht** Teil
der Kern-Pipeline und unabhängig:

```bash
cd ui
npm install
npm run dev        # startet Express-Server + Vite-Client
```

---

*Wissenschaftlicher Kontext und Akzeptanzkriterien: siehe `RESEARCH_CONTEXT.md`,
`task.md` und `TASKS.md` im übergeordneten Projektverzeichnis (`kathi/`).*
