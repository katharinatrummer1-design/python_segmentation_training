# cricket_id

Python-Paket für die **Individual-Identifikation von Grillen aus deren Gesang**
(PCA- und CNN-Baseline auf segmentierten Chirps, leckage-freie Session-Splits,
automatischer Report).

Dies ist nur der Paket-Quellcode (`src/`-Layout). Die **vollständige Dokumentation**
— Installation, Abhängigkeiten, Konfiguration, Pipeline, CLI, DSP-Glossar mit
Einheiten und Tests — steht im Projekt-Root:

➡️ siehe **`../../README.md`** (Projekt-Root `kathi/`).

## Schnellreferenz

```bash
# aus dem Projekt-Root ausführen:
conda env create -f environment.yml && conda activate cricket-id
pip install -e .
python -m cricket_id --help
python -m cricket_id run-mvp --config configs/mvp_lab_calling.yaml
```

### Paket-Module

| Modul | Aufgabe |
| --- | --- |
| `cli.py` | Typer-CLI, registriert alle Kommandos |
| `io/` | Manifest-Validierung (`manifest.py`), Audio-Laden/QC (`audio.py`), Roh-Ingest (`ingest.py`) |
| `preprocessing/` | Temperatur-Merge/QC (`temperature.py`), Residualisierung (`residualize.py`) |
| `segmentation/` | RMS-Envelope-Segmentierung (`segment.py`), QC-Stichprobe (`review.py`) |
| `features/` | tabellarische Akustik-Features (`tabular.py`), Log-Mel-Spektrogramme (`spectrograms.py`) |
| `splits/` | leckage-freie Session-/Recording-Splits (`build_splits.py`) |
| `models/` | PCA-Baseline, Custom-CNN/ResNet18, Trainer, Dataset, Triplet-Embedding |
| `evaluation/` | Metriken, Reports, Cross-Context, Cross-Validation, Embeddings, Statistik |
| `utils/` | YAML-Config (`config.py`), Pfad-Auflösung ohne Hardcoding (`paths.py`) |
