# NEUROFORGE — Ultimate System Plan

**Build. Mutate. Ship.** · Working-tree snapshot: 04 November 2026

This plan separates implemented behavior, recorded evidence, proposed work and design direction.

## 01. Executive direction

What NEUROFORGE becomes

NEUROFORGE becomes a local-first model-evolution studio for labelled visual and acoustic data. A user mounts a dataset, inspects its vocabulary, evolves task-specific CNN blueprints, trains the selected design, evaluates it honestly, and exports a reproducible inference bundle.

### North-star sentence

Build the model family in public. Mutate the blueprint with evidence. Ship the weights with their context.

| Outcome | Meaning |
| --- | --- |
| Image classification | Identify one known class from an image, with per-class metrics and live inference. |
| Audio classification | Identify one known sound class from a WAV clip represented as a log-mel spectrogram. |
| Architecture search | Compare finite CNN designs through parent selection, mutation, proxy training and lineage. |
| Evidence layer | Keep labels, splits, configuration, weights, metrics, timing and limitations together. |

## 02. Progress snapshot

What exists today and how far the MVP has moved

| Area | Current state | Evidence / gap |
| --- | --- | --- |
| Image MVP | Implemented and demoable | React studio, ImageFolder/Roboflow intake, DynamicCNN search, final training path, metrics, exports and live image inference exist. |
| Audio core | Foundation implemented | audio_pipeline.py handles WAV discovery, ESC-50/UrbanSound8K layouts, folds, log-mel tensors and audio prediction support. |
| Audio product flow | Next implementation slice | FastAPI prediction routing, React audio intake/playback/spectrogram and a real ESC-50 benchmark still need to be connected and verified. |
| Evolution engine | Implemented | architect.py records generations, parents, mutations, fitness and cached candidate evaluations. |
| Production UI | Strong local MVP | React/Vite/Three.js studio is served by FastAPI; motion, topology, mutation loop and export drawers are present. |
| Research rigor | Partially implemented | Roboflow has explicit splits; plain ImageFolder still needs a true independent test manifest and grouped leakage audit. |
| Deployment | Future | No registry, authentication, cloud deployment or multi-user experiment store yet. |

### Planning estimates

Image classification is roughly 85% of the intended local MVP; audio is roughly 55% of its core path and 25% of its complete product path. These are planning estimates, not accuracy scores.

## 03. The complete system architecture

One platform, two modality adapters, one evolution idea

```text
React/Vite/Three.js studio
        ↓ REST + WebSocket
FastAPI workspace service
   ├── dataset intake + status + exports
   ├── image adapter: RGB tensors
   └── audio adapter: WAV → log-mel tensors
        ↓
architect.py / candidate population + mutation + fitness
        ↓
neuroforge_core.py / DynamicCNN + training + metrics
        ↓
run bundle: config + weights + labels + preprocessing + splits + metrics
```

| Boundary | Contract |
| --- | --- |
| Frontend → API | Dataset metadata, search controls, streamed candidate events, training status and ranked predictions. |
| Audio adapter → core | One-channel feature tensor, ordered class names, fold-aware samples and preprocessing metadata. |
| Architect → core | A JSON architecture configuration; the child receives fresh weights during evaluation. |
| Checkpoint → inference | The model type, config, weights, class order and preprocessing must agree. |

### Important boundary

Image and audio use the same search machinery but remain separate classifiers. A future multimodal model would be a new experiment, not an accidental side effect of adding audio.

## 04. Image classification plan

From pixels to a defensible prediction

1. Mount ImageFolder or Roboflow data and produce an explicit class vocabulary.
2. Audit labels, duplicates, class counts, corrupt files and source relationships.
3. Create train/validation/test manifests; group related originals before splitting.
4. Resize and normalize RGB images; apply train-only augmentation.
5. Run a fixed manual CNN and optional ResNet-18 transfer baseline.
6. Search DynamicCNN configurations with a declared proxy budget.
7. Retrain the selected architecture from fresh weights using validation for checkpoint choice.
8. Evaluate the locked model once on an untouched test split and export the bundle.

| Input | Recommended first production profile |
| --- | --- |
| Tensor | 3 × 64 × 64 for the current MVP; compare 128 and 224 only through validation. |
| Model | DynamicCNN search space: 2–5 convolution blocks, filters 16/32/64/128, kernels 3/5, activation, pooling, BatchNorm and dropout. |
| Loss | Class-weighted cross-entropy for imbalance; document weights and class supports. |
| Metrics | Accuracy plus macro precision, recall, F1, confusion matrix, parameter count and latency. |

## 05. Audio classification plan

Give every sound a measurable shape

```text
WAV clip
  ↓ decode + mono mix
16 kHz resample
  ↓ center crop / pad to 4 s
STFT: 512 FFT, 256 hop, Hann window
  ↓ power spectrum
64 mel filters → log energy → normalize
  ↓
1 × 64 × ~251 feature tensor
  ↓
DynamicCNN → class probabilities
```

| Stage | Implementation plan |
| --- | --- |
| Input contract | Start with 16-bit PCM WAV; add safe decoder support for common formats only after tests. |
| Dataset layouts | Class folders first; ESC-50 and UrbanSound8K metadata/fold layouts as benchmark adapters. |
| Splits | ESC-50 folds 1–3 train, 4 validation, 5 test for the first smoke benchmark; later aggregate official cross-validation. |
| Augmentation | Feature gain, small noise and time masking initially; validate whether each improves macro F1. |
| Inference | Upload WAV, play it back, show waveform/spectrogram, predicted class, ranked probabilities and latency. |

### Audio quality rule

A synthetic WAV test proves plumbing. Only a real, fold-respecting ESC-50 or UrbanSound8K run can support a sound-recognition quality claim.

## 06. The Architect and the two learning loops

Keep the concept simple enough to defend

```text
OUTER LOOP / EVOLUTION
seed → evaluate → rank → select parents → mutate → next generation
                         ↓
INNER LOOP / TRAINING
batch → forward pass → loss → backprop → optimizer update
```

| Question | NEUROFORGE answer |
| --- | --- |
| What changes outside? | Layer count, filters, kernels, activation, BatchNorm, pooling and dropout. |
| What changes inside? | The trainable convolution and classifier weights. |
| Does a child inherit weights? | No. It inherits a configuration lineage; candidate weights are trained fresh. |
| Why keep an elite? | To preserve the best known design while exploration continues. |
| Is it globally optimal? | No. It is the best evaluated candidate under a finite space, split, score and budget. |

## 07. Model architecture book

The blueprint the evaluator should be able to draw

| Layer / component | Image path | Audio path |
| --- | --- | --- |
| Input | RGB tensor, 3 channels | Normalized log-mel map, 1 channel |
| Feature blocks | Conv2D → optional BatchNorm → activation → optional MaxPool/Dropout2d | Same DynamicCNN block pattern over frequency-time features |
| Compression | Adaptive average pooling to 1 × 1 | Adaptive average pooling to 1 × 1 |
| Classifier | Linear layer with N image classes | Linear layer with N sound classes |
| Output | Logits → softmax probabilities | Logits → softmax probabilities |

### Current selected image blueprint

The current JSON snapshot records 3 layers with filters [32, 64, 128], kernels [3, 5, 3], BatchNorm on, GELU activation, pooling on and dropout 0.0. The output head expands to the dataset's class count.

### Future architecture families

Add depthwise-separable CNNs for edge deployment, residual blocks for deeper image search, temporal convolution or lightweight CRNN options for audio, and an explicit architecture registry so every run can be reloaded without mutable default filenames.

## 08. Training, scoring and evaluation

Numbers must answer a precise question

| Phase | Purpose | Allowed data |
| --- | --- | --- |
| Proxy search | Cheaply rank candidate architectures | Training data for updates; validation for metrics and ranking. |
| Final training | Teach the selected architecture for the declared budget | Training data; validation for checkpoint selection and early stopping. |
| Final test | Measure the locked decision once | Untouched test data only. |

The current fitness combines accuracy, macro precision and macro F1 with small parameter and latency penalties. This is a ranking heuristic, not a probability and not a guarantee of deployment quality.

### Required report line

Every metric must name the dataset, model, split, class count, training budget, seed and averaging rule.

| Reliability upgrade | Why it matters |
| --- | --- |
| Grouped split | Prevents near-duplicate frames, crops or recordings leaking across partitions. |
| Multiple seeds | Shows whether a result is stable or lucky. |
| Per-class report | Prevents a majority class from hiding weak classes. |
| Calibration/OOD set | Tests whether confidence and in-scope detection can be trusted. |

## 09. Artifact and reproducibility contract

Ship context with the model

```text
run_id/
  ├── config.json
  ├── weights.pth
  ├── labels.json
  ├── preprocessing.json
  ├── split_manifest.json
  ├── metrics.json
  ├── lineage.json
  └── README.md
```

| Artifact | Must record |
| --- | --- |
| Configuration | Modality, input shape, layers, filters, kernels, activation, pooling and dropout. |
| Preprocessing | Image size/normalization or sample rate, clip length, FFT, hop, mel count and normalization. |
| Dataset identity | Source, version/hash, classes, counts, skipped files and split membership. |
| Training | Seed, device, epochs, optimizer, learning rate, scheduler, class weights and checkpoint rule. |
| Lineage | Candidate IDs, parents, mutations, scores, cache reuse and selection reason. |

## 10. UI theme and type system

An instrument, not a generic dashboard

### Design direction

NEUROFORGE is an exhibition-grade machine room: quiet paper surfaces carry the story, graphite instrument bays hold live computation, and cobalt/coral/acid-lime mark state changes.

| Role | Choice |
| --- | --- |
| Primary paper | #EDE9DF — warm mineral paper for narrative surfaces. |
| Light paper | #F7F4ED — readable panels and drawers. |
| Ink | #111315 — near-black graphite, never a flat pure black default. |
| Muted text | #656761 — evidence labels and supporting copy. |
| Cobalt | #2448DA — active structure, links and search state. |
| Coral | #FF5D42 — mutation, warning and physical offset shadow. |
| Acid lime | #D4ED42 — success, live energy and confirmation. |
| Night bay | #101215 — topology, spectrogram and model instruments. |

| Type role | Direction |
| --- | --- |
| Display | Space Grotesk-like geometric sans; tight tracking and large editorial scale for decisive headings. |
| Narrative accent | DM Serif Display-like italic serif for a single human, reflective counterpoint. |
| Telemetry | IBM Plex Mono-like monospace for run IDs, labels, metrics, code and state. |
| Rules | Sentence case for user-facing copy; reserved uppercase for telemetry and system labels. |

## 11. Motion and interaction system

Physics should explain the machine

| Motion | Purpose |
| --- | --- |
| Spring cursor field | Buttons and surfaces follow the pointer with damped return, making the material feel responsive. |
| Topology gravity | Nearby 3D nodes repel the pointer and flex their synapses, turning architecture into a living specimen. |
| Drag inertia | The 3D model keeps momentum after a gesture, then settles rather than stopping abruptly. |
| Impact ring | Pointer-down creates a local shockwave so contact has a visible physical consequence. |
| Reduced motion | Respect prefers-reduced-motion and keep the information legible without animation. |

### Restraint

One memorable motion per moment is stronger than ten unrelated effects. Motion should answer loading, selection, mutation, inference or contact.

## 12. Prompt library

Reusable prompts for the next build cycles

| Use | Prompt |
| --- | --- |
| Architecture review | Act as a senior ML architect. Inspect the current NEUROFORGE config, data shape and training loop. Return the smallest change that improves reliability, its acceptance criteria, and the exact checks to run. |
| Audio adapter | Extend NEUROFORGE's existing audio path without changing the image contract. Support WAV intake, log-mel preprocessing, ordered labels, fold-aware metadata and a user-visible prediction response. Identify every caller and add a smoke check. |
| NAS experiment | Design a matched-budget experiment comparing mutation search, random search and a fixed baseline. Keep data splits, seeds, proxy epochs and reporting fields identical. Do not claim superiority before repeated results. |
| UI direction | Design an exhibition-grade model-evolution interface for NEUROFORGE. Use paper, graphite, cobalt, coral and acid-lime. Avoid generic AI orbs. Make the architecture, lineage and evidence the hero. |
| Physics motion | Add one physically motivated interaction to the existing interface. Use spring damping, pointer response and a reduced-motion fallback. Preserve the current layout and describe the user action that the motion explains. |
| Viva practice | Act as a skeptical evaluator. Ask ten questions about DynamicCNN, audio log-mel features, mutation, metrics, data leakage and confidence. After each answer, ask one follow-up that tests understanding. |
| Error analysis | Create a per-class error report for this run. Include support, precision, recall, F1, confusion pairs, representative failures and a data-quality hypothesis. Separate measured findings from hypotheses. |
| Release note | Write a concise release note from these changed files and checks. Separate implemented behavior, evidence, limitations and next work. Never turn a roadmap item into a completed claim. |

### Prompt discipline

Every prompt should state the role, repository context, user outcome, constraints, evidence required, files allowed to change and exact verification command.

## 13. Ultimate roadmap

What NEUROFORGE can become after the local MVP

1. Finish the production audio intake, prediction endpoint, playback, waveform/log-mel view and audio-specific export bundle.
2. Add immutable run directories and a model registry; stop overwriting one mutable default bundle.
3. Repair ImageFolder evaluation with explicit train/validation/test manifests and grouped leakage checks.
4. Add real ESC-50 and UrbanSound8K benchmark runners with official fold aggregation.
5. Add per-class error galleries, calibration curves, confusion matrices and OOD evaluation.
6. Expand search to learning rate, resolution, augmentation, optimizer and scheduler through multi-fidelity promotion.
7. Add random-search and fixed-baseline ablations to test whether parent-guided mutation helps.
8. Add edge-friendly architectures, quantization, ONNX/Core ML export and measured device benchmarks.
9. Add optional multimodal fusion only after independent image/audio models are reliable.
10. Add authentication, shared experiments, artifact storage and deployment automation for a team product.

## 14. 30 / 60 / 90 day execution plan

A sequence that keeps science ahead of spectacle

| Window | Deliverables | Definition of done |
| --- | --- | --- |
| Next 30 days | Audio React/API flow; ESC-10 smoke benchmark; true image manifests; immutable bundle schema. | A user can mount, train and infer on both modalities with separate saved bundles. |
| Days 31–60 | ESC-50 fold runner; repeated seeds; error-analysis panels; random-search baseline; registry UI. | A benchmark report explains variance, per-class behavior and lineage across runs. |
| Days 61–90 | Multi-fidelity search; stronger backbones; export targets; deployment prototype; optional multimodal design. | A selected model can move from local experiment to a reproducible, portable inference service. |

### Priority order

Evidence first, audio product flow second, search breadth third, deployment fourth. A beautiful surface should reveal a trustworthy system underneath.

## 15. Viva and presentation story

Explain it in five minutes without overclaiming

1. Problem: manual architecture selection is slow and opaque.
2. Contribution: an inspectable mutation-driven CNN search workflow with lineage and local inference.
3. Mechanism: outer evolution changes blueprints; inner training learns weights.
4. Two modalities: images become RGB tensors; sounds become log-mel tensors; both use DynamicCNN separately.
5. Evidence: show the dataset, split, budget, candidate mutation, metrics and artifact identity together.
6. Limit: current ImageFolder testing needs stronger independence; audio needs a real benchmark and complete web path.
7. Next proof: repeated fair runs against random search and fixed baselines, plus per-class analysis.

### Best short answer

NEUROFORGE does not promise the globally best model. It makes a finite architecture search visible, measurable and reproducible enough to improve.

## 16. Definition of success

The project is complete when the claims are earned

| Success signal | Target |
| --- | --- |
| Image workflow | Independent test, repeated seeds, class-level reports and portable bundle. |
| Audio workflow | Real benchmark with official folds, audio UI, ranked prediction and exported preprocessing. |
| Search validity | Matched random-search baseline and ablations for mutation, fitness and budget. |
| User experience | A first-time user can understand what happened, why a model won and what to ship. |
| Viva readiness | Every major claim points to a file, a metric, a split or an explicitly labelled plan. |

### Final principle

A higher score is useful. A score whose data, model, split, budget and lineage are visible is useful and defensible.

