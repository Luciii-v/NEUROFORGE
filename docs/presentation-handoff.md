# NEUROFORGE presentation handoff

This document gives a teammate enough context to analyze the project and build a presentation without reverse-engineering the entire repository first.

## One-line story

NEUROFORGE turns model design into an observable evolutionary loop: load a dataset, generate candidate CNNs, mutate strong traits, select a winner, train it, and ship an inference-ready MVP.

## Recommended presentation arc

### 1. The problem

Choosing a neural-network architecture manually is slow, opaque, and easy to bias toward one familiar design.

### 2. The idea

NEUROFORGE treats architectures as candidates in a searchable population. Each candidate is described by a compact configuration containing:

- Number of convolutional layers.
- Filters per layer.
- Kernel sizes.
- Activation function.
- Batch normalization.
- Pooling.
- Dropout.

### 3. The evolutionary loop

```text
Generation 1 candidates
        ↓
Rank by multi-metric proxy score
        ↓
Select strong parents
        ↓
Mutate architecture traits
        ↓
Evaluate the next generation
        ↓
Retrain and test the selected architecture
```

The UI exposes this as a native model-lineage board rather than hiding the search behind one final number.

### 4. The product experience

The cinematic intro establishes the product identity. The dashboard continues the same language with a forge-control hero, live pipeline rail, candidate metrics, lineage, artifact downloads, and live image testing.

## Suggested live demo

1. Open the app and let the `NEUROFORGE` intro play.
2. Open **Dataset** and load an ImageFolder or Roboflow export.
3. Confirm the class count and split information.
4. Open **Search lab**.
5. Show the search budget: `5 × 8 = 40` candidate evaluations.
6. Start the search and point out generation updates, candidate metrics, and new-best notifications.
7. Show the candidate table and lineage board.
8. Open **Final model** and run full training with presentation mode off.
9. Show held-out accuracy, macro precision, recall, and F1.
10. Open **Live test** and upload a new image.

## Evidence available in the repository

- `search_history.json` contains candidate IDs, generation numbers, parent IDs, mutation operations, architecture configs, metrics, and scores.
- `manual_baseline.json` contains the fixed human-designed baseline.
- `best_model_config.json` contains the selected architecture snapshot.
- `best_model_metadata.json` contains dataset information, training mode, checkpoint metadata, and held-out metrics.
- `milestone_1_doc.md`, `neuroforge_pitch.docx`, and the project dossier contain earlier project framing and milestone material.

## Metrics and interpretation

The project reports both accuracy and macro metrics. Accuracy alone can look acceptable on an imbalanced dataset while minority classes perform poorly. Macro precision, recall, and F1 give every class equal weight and should be shown together in the presentation.

The original crop-pest run used 12 candidates and reached 47.54% accuracy / 30.05% macro F1 after final training. Controlled 40-candidate experiments improved macro F1 into the high 30s, but did not automatically improve raw accuracy. This is an important research result: search breadth helps exploration, but proxy depth, class imbalance, dataset size, and training strategy also affect the final model.

## Current limitations to explain honestly

- The current production path is image classification; audio is still roadmap work.
- The crop-pest dataset is small and imbalanced.
- The current architecture search is mutation-based rather than a full Bayesian or gradient-based NAS system.
- Proxy ranking and full-training ranking can disagree.
- The current UI runs locally and does not yet include authentication or a shared experiment database.

## Strong future-work slide

- Add audio loading and mel-spectrogram model search.
- Add repeated-seed evaluation and confidence intervals.
- Add confusion-matrix and per-class error analysis to the UI.
- Add experiment comparison across runs.
- Add collaborator-facing export of charts and presentation-ready reports.
