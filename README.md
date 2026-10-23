# NEUROFORGE

> **Build. Mutate. Ship.** — a cinematic Streamlit studio for evolving image-classification models.

NEUROFORGE searches over dynamically generated CNN architectures, evaluates candidates with a proxy training budget, tracks parent/mutation lineage, retrains the selected architecture, and supports single-image live inference.

## Current status

- Streamlit workspace with four product stages: Dataset, Search lab, Final model, and Live test.
- Dynamic CNN architecture generator with mutation operations across layers, filters, kernels, activations, dropout, pooling, and batch normalization.
- Deterministic class-aware splits for ImageFolder datasets.
- Roboflow `_classes.csv` export support.
- ZIP upload hardening for macOS `__MACOSX`, `._*`, `.DS_Store`, and non-image artifacts.
- Train-only image augmentation and inverse-frequency class-balanced loss for the ImageFolder path.
- Persisted candidate history, baseline comparison, architecture lineage, final checkpoint metadata, and held-out test metrics.
- Optional ResNet-18 transfer-learning baseline.
- Cinematic opening sequence and presentation-oriented dashboard styling.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Then open `http://localhost:8501`.

For a local image dataset, use either:

1. **Enter Local Path** — recommended for large datasets.
2. **Upload ZIP file** — supported for ImageFolder-style archives.

The supported ImageFolder shape is:

```text
dataset-root/
├── class-a/
│   ├── image-001.jpg
│   └── image-002.jpg
└── class-b/
    ├── image-003.jpg
    └── image-004.jpg
```

Roboflow exports may use `train/`, `valid/`, and `test/` folders with `_classes.csv` files.

## Demo flow

```text
Load data
  → Generate candidates
  → Mutate architectures
  → Select the strongest candidate
  → Retrain the selected model
  → Evaluate on held-out data
  → Test a new image
```

The default Search lab budget is currently:

```text
5 generations × 8 candidates = 40 candidate evaluations
3 proxy epochs per candidate
```

The settings can be changed in the Search lab UI.

## Repository map

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit UI, cinematic intro, dataset loading, search controls, final training, downloads, and live inference. |
| `architect.py` | Mutation-based neural architecture search and persisted candidate lineage. |
| `neuroforge_core.py` | DynamicCNN, dataset adapters, transforms, training, metrics, checkpoint evaluation, and prediction. |
| `.streamlit/config.toml` | Dark violet/cyan Streamlit theme. |
| `best_model_config.json` | Current selected architecture snapshot. |
| `best_model_metadata.json` | Current dataset, training, and held-out metric snapshot. |
| `best_model_weights.pth` | Current generated-model checkpoint. |
| `search_history.json` | Auditable candidate records, mutations, parents, and metrics. |
| `manual_baseline.json` | Fixed manual-baseline comparison record. |
| `docs/presentation-handoff.md` | Storyline, demo sequence, metrics, limitations, and presentation guidance for collaborators. |

## Dataset and generated-file policy

Local datasets, extracted uploads, CIFAR-10 downloads, Python environments, secrets, and transient candidate files are intentionally ignored by Git. Teammates should bring their own dataset or receive one through an approved private storage location.

The current checkpoint and JSON experiment artifacts are included because they make the project easier to analyze and present. They are snapshots, not a replacement for rerunning the pipeline.

## Important reproducibility note

The checked-in crop-pest artifact snapshot was produced before the latest augmentation and class-balanced-loss changes. For a fresh benchmark, rerun Search lab and then retrain the selected architecture. The UI records the resulting run metadata and held-out test metrics in the JSON artifacts.

## Verification commands

```bash
python3 -m py_compile app.py architect.py neuroforge_core.py
python3 -m streamlit run app.py --server.headless true --server.port 8501
```

The project also uses Streamlit's in-process `AppTest` smoke checks during development.
