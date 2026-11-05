# NEUROFORGE: Master Audit & Project Checklist

> **Role & Authority**: Head of Engineering & Autonomous Execution Lead  
> **Reconciled With**: `docs/neuroforge-ultimate-plan/`, `docs/learning-guide/`, `docs/superpowers/`, `server.py`, `neuroforge_core.py`, `architect.py`, `audio_pipeline.py`, and the live runtime at `http://127.0.0.1:8000`.

---

## 01. Executive Status Matrix

| System Pillar | Status | Operational Evidence |
| :--- | :---: | :--- |
| **Exhibition Studio (`/`)** | **DONE** | React/Vite/Three.js editorial surface with warm paper aesthetic, 3D topology specimen, trajectory spline, crucible inference, and export drawer. |
| **Mutation Workbench (`/?view=loop`)** | **DONE** | Live 4-chamber practitioner interface: Mount data, Set search budget, Live candidate stream, Forge the winner, and Baseline comparison table. |
| **Dynamic CNN Search (`architect.py`)** | **DONE** | Genetic search loop: parent selection, mutation, proxy evaluation, lineage recording, cache hit deduplication. |
| **Image Pipeline (`neuroforge_core.py`)** | **DONE** | ImageFolder + Roboflow CSV single-label loaders, deterministic split indexers, train-only augmentation, class-weighted cross-entropy. |
| **Audio Core (`audio_pipeline.py`)** | **PARTIAL (55%)** | Standard-library WAV reader, 16kHz resampling, STFT, 64 log-mel filterbanks, and fold split handlers are built. |
| **Audio Web Product Flow** | **PENDING** | Audio upload dropzone, live WAV playback, waveform/spectrogram rendering, and `/api/audio/predict` endpoint in FastAPI. |
| **Held-Out Test Final Evaluation** | **PENDING (ACTIVE RUN)** | Current 9-class Fruits dataset has completed proxy search (Gen 3 winner: 74.2% acc / 67.5% F1); 20-epoch full-training run needs execution to populate evaluation metrics on the live UI ribbon. |
| **Multi-Model Baseline Suite** | **DONE** | DynamicCNN vs. 2-layer manual baseline vs. ResNet-18 transfer learning side-by-side with accuracy, precision, recall, F1, MSE, and RMSE. |

---

## 02. Completed Checklist (What is Done)

- [x] **Standalone Server Architecture**: FastAPI backend (`server.py`) serving production React static bundle (`frontend/dist`) on `http://127.0.0.1:8000` with WebSocket mutation streaming (`/ws/search`).
- [x] **Awwwards-Caliber Materials UI**: High-contrast Swiss editorial typography (*"Models are forged, not found"*), warm paper palette (`#ede9df`), dark instrument bays, and light/dark theme persistence.
- [x] **Dual Interaction Modes**:
  - `/` — Exhibition / Story Mode (Presentation-ready, 3D specimen, trajectory telemetry, live crucible).
  - `/?view=loop` — Mutation Loop Mode (Workbench for mounting, budget configuration, candidate streaming, and baseline benchmarking).
- [x] **Interactive 3D Specimen**: Three.js WebGL canvas with studio directional lighting, pearl-titanium PBR nodes, internal light transmission, animated traveling photons, and mouse drag inertia.
- [x] **Roboflow & ImageFolder Compatibility**: Fixed macOS `__MACOSX` ghost-folder bug; normalized dataset roots; built single-label CSV parser with quality reporting on unlabelled/multi-label rows.
- [x] **Hardware Acceleration**: Automatic Apple Silicon MPS detection and utilization for sub-3ms forward passes.
- [x] **Lineage & Artifact Export**: Downloadable `best_model_weights.pth`, `best_model_config.json`, and one-click copyable production PyTorch inference snippets.

---

## 03. Pending Work & Action Checklist (Next Slices)

### Phase 1: Close the Loop on the Active 9-Class Model
- [ ] **Run Full Training for Current Champion**:
  - The live UI metric ribbon currently displays `—%` because the 9-class Fruits run only completed proxy search (3 gens × 4 candidates).
  - Execute 20-epoch final training to populate evaluation accuracy, macro F1, and per-sample latency on the live home ribbon.
- [ ] **Fix Minor Legend Text Collision**:
  - Resolve the slight text overlap between `Champion Classifier Head` and the latency readout on narrower browser frames.

### Phase 2: Audio Modality Web Flow Integration
- [ ] **FastAPI Audio Routing**:
  - Add `/api/audio/predict` supporting `.wav` uploads using `audio_pipeline.py`.
  - Add `/api/audio/samples` providing sample audio clips.
- [ ] **React Audio Component**:
  - Build audio dropzone, waveform/spectrogram canvas visualizer, and live playback in the crucible.
- [ ] **ESC-50 / ESC-10 Smoke Benchmark**:
  - Run fold-respecting audio classification benchmark to earn real acoustic accuracy numbers.

### Phase 3: Research Rigor & Model Registry
- [ ] **Immutable Run Storage**:
  - Move from overwriting single default files (`best_model_weights.pth`) to timestamped run directories (`runs/<run_id>/`).
- [ ] **Per-Class Error Analysis**:
  - Expose confusion matrix and class-level breakdown in a slide-out drawer so weak classes can't hide behind majority accuracy.

---

## 04. Execution Order (Autonomous Plan)
1. **Immediately execute full-training** on the current 9-class Fruits champion to light up all live UI metric cards and resolve the empty `—%` state.
2. **Apply minor layout polish** to the 3D specimen legend and drawer backdrops.
3. **Connect the audio pipeline** into FastAPI and the React crucible to deliver the promised visual + acoustic dual-modality studio.
4. **Sync progress to GitHub**: Commit working tree state and push to remote repository to maintain active contribution tracking.

