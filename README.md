# NEUROFORGE

A local studio that evolves CNN architectures for image and audio classification, shows the full family tree of every design it tried, and reports scores you can defend.

```text
mount data → evolve designs (ranked on validation) → train the winner → test once → ship a bundle
```

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
./run_studio.sh            # builds the React app if needed, serves http://127.0.0.1:8000
```

`run_studio.sh` refuses to start if the port is taken (it no longer kills whatever is there). Use `NEUROFORGE_PORT=8010 ./run_studio.sh` to pick another port. For frontend development, run the server, then `cd frontend && npm run dev` (Vite proxies `/api` and `/ws`).

## Evaluation protocol

Every number in the studio names its split. The rules come from `docs/neuroforge-ultimate-plan`, section 08:

| Split | Used for |
| --- | --- |
| train (70%) | weight updates |
| valid (15%) | ranking search candidates, choosing the training checkpoint, early stopping |
| test (15%) | measured **once**, on the locked checkpoint |

ImageFolder data is split by class (stratified) with a fixed seed. Roboflow exports keep their own train/valid/test folders. ESC-50 uses folds 1–3 / 4 / 5 and UrbanSound8K uses 1–8 / 9 / 10. Checkpoints trained before this protocol (schema v1) are flagged in the UI as legacy, because their test split was also used to pick them.

## Results: crop pest dataset

5 classes, 295 images (207 / 44 / 44), seed 42. Search: 5 generations × 8 designs × 3 proxy epochs. Every model was trained with the same protocol. Reproduce with:

```bash
python scripts/run_benchmark.py "datasets/crop pest dataset" --epochs 40 --baseline-epochs 25 \
  --baselines manual_cnn efficientnet_b0 resnet18 mobilenet_v3 shufflenet_v2
```

| Model | Valid F1 | Test accuracy | Test macro F1 |
| --- | ---: | ---: | ---: |
| NEUROFORGE winner (3-layer residual CNN, from scratch) | 36.0 | 34.1 | 28.7 |
| Manual 2-layer CNN (from scratch) | 28.7 | 34.1 | 28.9 |
| ResNet-18 (ImageNet-pretrained, frozen, 224px) | 92.9 | **93.2** | **93.8** |
| EfficientNet-B0 (pretrained) | 91.2 | 88.6 | 91.3 |
| MobileNet-V3 Small (pretrained) | 91.0 | 86.4 | 83.0 |
| ShuffleNet-V2 0.5× (pretrained) | 80.2 | 75.0 | 79.1 |

What this shows:

- **The search finds better designs on validation** (41.6% vs 9.3% proxy F1 for the manual CNN), but with 207 training images that advantage doesn't carry over to the test split. Networks trained from scratch can't learn general visual features from this little data.
- **Pretrained backbones win by a wide margin.** Earlier runs fed them 64px images with the wrong normalisation, which is why they once scored around 60%.
- **The test split has 44 images**, so one image moves accuracy by 2.3 points. Small gaps between models are noise.
- The previous crop-pest figure (37.7%) came from a checkpoint chosen on its own test split, so it was optimistic. The earlier fruits result (about 85%) was not affected: that Roboflow export ships separate valid and test folders.

Next steps for the search itself are in the roadmap below: a larger dataset (Oxford-IIIT Pet, ESC-50), repeated seeds, and a matched random-search baseline.

## Results: Oxford-IIIT Pet (does the search help with enough data?)

37 cat and dog breeds, 7,390 images (5,172 / 1,109 / 1,109), seed 42, the same search budget (5 × 8 × 3) and protocol as above. Chance is 2.7%. Prepare and reproduce with:

```bash
python scripts/prepare_oxford_pets.py images.tar.gz datasets/oxford-pets   # from robots.ox.ac.uk/~vgg/data/pets
python scripts/run_benchmark.py datasets/oxford-pets --epochs 40 --baseline-epochs 25 \
  --baselines manual_cnn efficientnet_b0 resnet18 mobilenet_v3 shufflenet_v2
```

| Model | Valid F1 | Test accuracy | Test macro F1 |
| --- | ---: | ---: | ---: |
| NEUROFORGE winner (4-layer GELU CNN, filters 64/64/64/128, from scratch) | 50.9 | **49.9** | **49.9** |
| Manual 2-layer CNN (from scratch) | 8.6 | 10.1 | 8.3 |
| ResNet-18 (ImageNet-pretrained, frozen, 224px) | 90.9 | 90.8 | 90.8 |
| EfficientNet-B0 (pretrained) | 89.6 | 88.6 | 88.6 |
| MobileNet-V3 Small (pretrained) | 83.9 | 85.5 | 85.5 |
| ShuffleNet-V2 0.5× (pretrained) | 72.8 | 74.9 | 74.3 |

What this shows:

- **With enough data the search clearly beats the hand-designed CNN: 49.9% vs 10.1% test accuracy (about 5×), on 1,109 test images**, where one image moves accuracy by only 0.09 points. On crop pest (207 training images) the two were indistinguishable; the difference is data, not the search.
- **The winner was still improving at the 40-epoch cap** (best epoch 40), so a longer final training run is the cheapest next gain.
- **Pretrained backbones remain far ahead (91%).** They bring features learned from 1.2M ImageNet photos; NEUROFORGE builds a network from nothing in about 45 minutes on a laptop. That gap is the honest framing for the viva: the search finds good from-scratch designs, but it does not replace transfer learning on small datasets.
- Single seed. Repeated seeds and a matched random-search baseline are still needed before claiming mutation beats random search.

## What the search explores

Each design is a JSON config: 2–5 conv blocks; filters 16/32/64/128 per layer; kernels 3/5; block type plain, residual or depthwise-separable; ReLU, GELU or SiLU; BatchNorm, pooling and dropout 0–0.4; image input size 64/96/128. Generation 1 mutates a seed design. Each later generation keeps its best design (the elite) and breeds mutated children from the top half. Fitness on validation is `0.45·acc + 0.25·precision + 0.30·F1 − params/1e6 − 100·latency`. Children inherit a blueprint, never weights.

## Artifacts

```text
runs/<run_id>/
  search_history.json      every candidate: parent, mutations, config, validation metrics, score
  search_summary.json      budget, objective, winner, manual-baseline comparison
  split_manifest.json      exact train/valid/test membership
  model-<kind>-<time>/     immutable training bundle
    config.json  weights.pth  labels.json  preprocessing.json
    split_manifest.json  metrics.json  lineage.json  README.md
```

A search never touches the active model. A training run writes its bundle first and only then promotes it to `best_model_weights.pth` / `best_model_metadata.json` / `best_model_config.json`, which the API serves. The studio's Ship section downloads the bundle as a zip.

## Repository map

| Path | Purpose |
| --- | --- |
| `server.py` | FastAPI: dataset intake, search WebSocket (with stop), training jobs, comparison, inference + Grad-CAM, downloads. Binds to localhost. |
| `architect.py` | Mutation search, run bundles, lineage, promotion. |
| `neuroforge_core.py` | DynamicCNN (plain / residual / depthwise), dataset adapters, splits, training, metrics, transfer baselines. |
| `audio_pipeline.py` | WAV decoding, log-mel features, ESC-50 / UrbanSound8K fold handling. |
| `frontend/` | React studio: editorial home (3D specimen, trajectory, lineage tree + inspector, learning curve, confusion heatmap, Coliseum, live crucible with Grad-CAM, DNA sequencer) and the mutation loop workbench (`/?view=loop`). New evidence panels live in `frontend/src/evidence/`. |
| `scripts/run_benchmark.py` | Reproducible search → train → baselines benchmark. |
| `tests/` | `python -m pytest tests -q` (`pip install -r requirements-dev.txt`). |
| `app.py` | Legacy Streamlit UI, no longer maintained. |
| `docs/` | Playbooks, system plan, presentation handoff, design archive. |

## Roadmap

1. ~~Larger image benchmark (Oxford-IIIT Pet)~~ done above; next ESC-50 for audio, and longer final training for the Pets winner.
2. Repeated seeds with mean ± spread, and a matched random-search baseline to test whether parent-guided mutation helps.
3. Grouped splits / near-duplicate audit for datasets with multiple crops of one source image.
4. Optional "search on top of a pretrained backbone" track, reported separately from from-scratch search.
5. ONNX / Core ML export with measured on-device latency.
