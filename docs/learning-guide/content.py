"""Editable content for the NEUROFORGE Mutation Playbook (4 November 2026)."""

PAGES = []


def page(title, subtitle, blocks, tag="FIELD NOTES", source=""):
    PAGES.append(dict(title=title, subtitle=subtitle, blocks=blocks, tag=tag, source=source))


def qa(question, answer):
    return ("qa", question, answer)


def note(title, body):
    return ("note", title, body)


def callout(title, body):
    return ("callout", title, body)


page("Your project.\nYour superpower.",
     "A teammate-friendly field guide to what NEUROFORGE does, how it learns, and how to explain it without sounding like a textbook.", [
    ("cover",),
], tag="THE MUTATION PLAYBOOK")

page("Pick your adventure", "Read once to understand. Revisit the short routes before a demo.", [
     callout("THE ONE-LINE PITCH", "NEUROFORGE is a React/FastAPI model-evolution studio: it mounts labelled image or audio data, evolves CNN architectures through selection and mutation, trains the chosen design, and exposes the evidence behind its predictions."),
    ("routes",),
    note("The analogy we will use", "Imagine an engineering tournament. Each model architecture is a robot design. Training teaches a robot its job. Mutation changes the design. A scoreboard chooses which designs get another round. The final robot still needs a proper exam."),
    note("How to read the status labels", "<b>IMPLEMENTED</b> means a code path exists. <b>RECORDED</b> means a saved result or earlier run is available. <b>PROPOSED</b> means future work. These are different levels of evidence, not interchangeable badges."),
    callout("THE FOUR QUESTIONS TO ASK ABOUT EVERY PERCENTAGE", "Which dataset? Which model? Which split? Which training budget? Without these, a percentage is just decoration."),
], tag="START HERE", source="Working source, production frontend and saved artifacts inspected on 3 Nov 2026; older runs are labelled historical.")

page("Your questions, decoded", "The fast answers to the questions you actually asked. The next pages unpack them.", [
    qa("Are we using any model?", "Yes: a PyTorch <b>DynamicCNN</b> whose architecture is generated and mutated. ResNet-18 is a separate, optional pretrained image baseline. No LLM chooses the architectures."),
    qa("Without mutation, what is special?", "Mutation and traceable parent-child evolution are the core. The UI visual can change; the search idea stays. The novelty claim is the integrated, observable workflow—not inventing evolutionary search itself."),
    qa("Why are results below 90%?", "Data quality, limited examples, visual difficulty, the training recipe and the restricted search space all matter. Three-epoch proxy results are early ranking signals. Mutation does not guarantee a target percentage."),
    qa("Why does training take so long?", "The production workbench starts at 3 generations × 4 candidates × 2 proxy epochs. Each new architecture trains from scratch, then the selected design can receive a declared full-training budget. Cache hits avoid some repeated evaluations."),
    qa("Can we use Kaggle, add audio and improve the UI?", "Yes, with format preparation and fair splits. The production UI now has a dedicated mutation workbench, while audio remains a backend/core foundation rather than a finished web workflow. Real audio benchmark results are still pending. The identity is <b>Build. Mutate. Ship.</b> with lineage and mutation evidence."),
    qa("Why the training warning, wrong apple label, and low crop score?", "These are three different issues: inconsistent UI/artifact state; an apple outside a pest classifier's label set; and genuinely weak crop performance. One fix cannot solve all three."),
], tag="YOUR Q&A", source="Answers reconciled with frontend/src, server.py, architect.py, neuroforge_core.py and audio_pipeline.py.")

page("The entire journey", "Dataset in. An evolved, trained classifier out.", [
    ("flow", [
        ("01  CONNECT", "A labelled image dataset arrives through a local path or ZIP in the production workbench."),
        ("02  PREPARE", "Read labels, form splits, resize images or build audio features."),
        ("03  BUILD", "Create CNN configurations and turn each into a trainable model."),
        ("04  AUDITION", "Train briefly; compare predictions with labels; calculate fitness."),
        ("05  MUTATE + SELECT", "Rank parents, preserve an elite, mutate children; repeat generations."),
        ("06  TRAIN THE WINNER", "Take the best architecture and train fresh weights for longer."),
         ("07  EVALUATE + SHIP", "Report metrics, download the blueprint/checkpoint, and try an in-scope image or WAV clip."),
    ]),
    callout("THE LOOP INSIDE THE ARROWS", "Steps 4–5 repeat. The winner means <b>best among the candidates evaluated under our objective</b>, not the best possible model in the world."),
     note("Two inputs, separate jobs", "Images become RGB tensors. Audio becomes one-channel log-mel tensors. They use the same search machinery, but are trained as separate classifiers; the project does not fuse image and sound in one model yet."),
], tag="THE BIG PICTURE", source="frontend/src/components/MutationLoopPage.jsx; server.py; architect.py; neuroforge_core.py.")

page("Two loops. Two jobs.", "If you understand this page, you understand the heart of the project.", [
    ("compare", [
        ("OUTER LOOP / EVOLUTION", "Changes the <b>blueprint</b>.<br/>Example: filters [32, 64] become [32, 128].<br/>Uses selection, mutation and a fitness score.<br/>Implemented in architect.py."),
        ("INNER LOOP / LEARNING", "Changes the <b>numbers inside the blueprint</b>.<br/>Example: convolution weights improve through gradient updates.<br/>Uses loss, backpropagation and an optimizer.<br/>Implemented in neuroforge_core.py."),
    ]),
    qa("Does a child inherit its parent's learned knowledge?", "Not currently. It inherits an architecture configuration. A newly evaluated child initializes fresh weights. The final training run also starts a fresh DynamicCNN. Weight sharing or weight inheritance would be a separate feature."),
    qa("Is mutation random, or does the model learn the mutation?", "The edits are randomized within allowed choices. Performance-based selection guides which parents contribute. The mutation operator itself is not a learned neural network."),
    qa("Why can a child be worse than its parent?", "Changing a design is an experiment, not an upgrade guarantee. The child may fit the task poorly or learn more slowly. Elitism keeps a strong existing design while other candidates explore."),
    callout("TEACH IT LIKE THIS", "“We evolve the recipe outside; we learn the weights inside.” Architecture search and ordinary model training solve two different problems."),
], tag="CORE CONCEPT", source="architect.py: mutate_config, run_architect_search; neuroforge_core.py: evaluate_candidate, full_train.")

page("Meet the model", "A CNN is a pattern detector, not a tiny human brain.", [
    note("The DynamicCNN assembly line", "<b>Input → [Convolution → optional BatchNorm → activation → optional MaxPool → optional Dropout2d] × layers → adaptive average pooling → linear classifier.</b><br/>The final layer outputs one number per class. Softmax converts those numbers into relative class probabilities during live inference."),
    ("table", ["Part", "What it does in plain English"], [
        ["Convolution / filter", "Slides learned detectors across the input to find useful local patterns."],
        ["Activation", "Adds non-linearity: the network can learn more than a straight-line rule."],
        ["Pooling", "Shrinks feature maps; trades spatial detail for efficiency."],
        ["Batch normalization", "Normalizes intermediate activations using batch/running statistics."],
        ["Dropout2d", "Randomly removes feature channels during training to discourage reliance on one shortcut."],
        ["Adaptive average pool", "Summarizes each channel to one value, regardless of spatial size."],
    ]),
     qa("What architecture snapshot is recorded right now?", "The current selected image configuration is a <b>3-layer DynamicCNN</b>: filters [32, 64, 128], kernels [3, 5, 3], GELU, BatchNorm on, pooling on and dropout 0.0. The classifier head expands to the dataset's nine fruit classes. The exact weights and metrics must always be read from the matching metadata bundle."),
    qa("And ResNet-18?", "An optional ImageNet-pretrained image model. The current code freezes its backbone parameters and trains a replacement classifier head. It does not mutate the ResNet backbone. Keep its results separate from NAS-generated CNN results."),
], tag="MODEL IDENTITY", source="neuroforge_core.py: DynamicCNN, _build_model; best_model_config.json; current selected search-history record.")

page("The mutation tournament", "How eight candidates become the next generation.", [
    note("Round rules", "<b>Generation 1:</b> one default design plus mutations of that design.<br/><b>Later generations:</b> sort the previous population by fitness; keep the top half as parents. Preserve the top design as one elite slot. Uniformly sample parents from the top half to create the remaining mutated children.<br/><b>Final choice:</b> keep the highest-fitness candidate found across all generations."),
    ("table", ["Trait", "Choices / mutation attempt"], [
        ["Layer count", "25% attempt: +1 or −1, clamped to 2–5 layers."],
        ["Filters per layer", "35% attempt per layer: 16, 32, 64 or 128."],
        ["Kernel per layer", "25% attempt per layer: 3×3 or 5×5."],
        ["Activation / dropout", "20%: ReLU/GELU/SiLU; 25%: dropout 0.0–0.4 in steps of 0.1."],
        ["BatchNorm / pooling", "Each has a 10% chance of toggling."],
    ]),
    note("Actual lineage from the saved run", "<b>G02-C03 → G03-C02</b><br/>Recorded mutations: <b>activation ReLU → GELU</b> and <b>dropout 0.2 → 0.0</b>.<br/>The child was selected with 74.19% proxy accuracy and 67.47% macro F1. This is a real recorded edit, not an animation invented for the interface."),
    callout("SMALL BUT IMPORTANT", "An attempt may draw the same value, so it is not always a change. If nothing changed, the operator forces a filter change. An <b>elite</b> is deliberately copied unchanged—it is preservation, not mutation. Crossover is not implemented."),
], tag="EVOLUTION RULES", source="architect.py: mutate_config, run_architect_search; search_history.json: G03-C02.")

page("How the scoreboard works", "Fitness is a design choice. It is not another accuracy percentage.", [
    ("formula", "S = 0.45A + 0.25P + 0.30F − N/1,000,000 − 100L", "A = accuracy %, P = macro precision %, F = macro F1 %, N = parameters, L = seconds per sample."),
    qa("Why these ingredients?", "Accuracy rewards overall correctness; macro precision and F1 give each class attention. Parameter and latency penalties mildly discourage expensive models. The weights are hand-chosen heuristics, not scientifically proven optimal settings."),
    note("Worked example / illustrative, not a run result", "Suppose A = 80, P = 75, F = 72, N = 100,000 and L = 0.002 seconds.<br/><b>S = 36 + 18.75 + 21.6 − 0.1 − 0.2 = 76.05.</b><br/>A score of 76.05 is a ranking value. It does not mean 76.05% probability of being correct."),
    qa("What is measured but not directly in the score?", "Recall and training time are reported but are not separate score terms. Recall influences F1 indirectly. The final-training checkpoint uses validation <b>accuracy</b>, not this search fitness. Those selection rules differ."),
    note("Real current comparison / same proxy budget", "Fixed manual CNN: <b>50.00% accuracy, 43.22% F1, score 48.46.</b><br/>Selected mutated CNN: <b>74.19% accuracy, 67.47% F1, score 71.13.</b><br/>That is +24.19 accuracy percentage points and +24.25 F1 points under this run's 2-epoch proxy budget. It is evidence for this run, not proof of universal superiority."),
    callout("EVALUATOR ANSWER", "“We scalarize several objectives into one fitness score. We will validate the weighting with ablations and repeated runs.”"),
], tag="SCORING", source="architect.py: _fitness; manual_baseline.json; selected record in search_history.json.")

page("Metrics without panic", "A tiny confusion matrix makes the formulas feel obvious.", [
    ("confusion",),
    note("Imagine 100 binary examples / illustrative", "For class A: <b>TP = 40</b> (correct A), <b>FP = 20</b> (B called A), <b>FN = 10</b> (A missed), <b>TN = 30</b> (correct non-A). Rows are truth; columns are predictions."),
    ("table", ["Metric", "Formula", "Example"], [
        ["Accuracy", "Correct / all examples", "(40 + 30) / 100 = 70%"],
        ["Precision for A", "TP / (TP + FP)", "40 / 60 = 66.67%"],
        ["Recall for A", "TP / (TP + FN)", "40 / 50 = 80%"],
        ["F1 for A", "2PR / (P + R)", "72.73%"],
    ]),
    note("Now average across classes", "Class B has precision 75%, recall 60%, and F1 66.67%.<br/><b>Macro precision = 70.83%; macro recall = 70%; macro F1 = 69.70%.</b><br/>Macro F1 is the average of the individual class F1 values. It is not generally the F1 of macro precision and macro recall."),
    callout("MEMORY HACK", "<b>Precision:</b> “When I say A, am I right?”<br/><b>Recall:</b> “Of all real A examples, how many did I catch?”<br/><b>F1:</b> “Can I do both reasonably well?”"),
], tag="METRICS LAB", source="neuroforge_core.py: classification_metrics. The 100-example matrix is a teaching example.")

page("Percentages can fool you", "Confidence, accuracy and F1 are three different conversations.", [
    qa("Is 96% confidence the same as 96% accuracy?", "No. Confidence is the largest softmax probability for <b>one input</b>. Accuracy summarizes correctness over labelled inputs. Softmax can be very confident and wrong, especially on unfamiliar data."),
    qa("Why was fruit macro F1 lower than accuracy?", "In the historical 89-image fruit test set, one of nine classes had no true examples. The original all-nine-class macro average assigned it zero. The recorded F1 was 77.10%; averaging only the eight present classes gives about 86.74%. These answer different questions."),
    callout("CHANGING THE AVERAGE DOES NOT TRAIN A BETTER MODEL", "77.10% → 86.74% here is a metric-definition change, not an improvement in predictions. Show the evaluated class count and name the averaging rule. A missing class still has unknown real performance."),
    qa("Does a 40% threshold detect unknown objects?", "No. The production UI warns below 40% as a deliberately modest low-confidence heuristic. With five classes, 20% is the uniform-random reference; a valid pest can score below 40%, and an apple can score above it. Proper unknown-input rejection requires validation on known and unknown samples."),
    qa("What else do we measure?", "Parameter count estimates model size. Training time records the candidate training loop. Reported per-sample inference latency is evaluation elapsed time divided by sample count; it includes loading/transfers and is not a pure warmed-up model-kernel benchmark."),
    note("Other averages you may hear", "<b>Macro:</b> equal weight per class. <b>Weighted:</b> class scores weighted by example count. <b>Micro:</b> pool decisions first; in ordinary single-label multiclass classification, micro F1 equals accuracy. NEUROFORGE's fitness uses its all-class macro fields."),
], tag="METRIC TRAPS", source="classification_metrics, evaluate_candidate, evaluate_checkpoint; frontend/src/components/InteractiveCrucible.jsx; historical fruit snapshot.")

page("What training really does", "The architecture is the recipe. These steps teach its weights.", [
    note("One update, in six moves", "<b>Batch → forward pass → weighted cross-entropy loss → backward pass → optimizer step → repeat.</b><br/>The prediction is compared with the correct label. Gradients describe how weights affect error. The optimizer updates those weights. Evaluation uses model.eval() and no_grad(), so weights are not learned from evaluation labels."),
    ("table", ["Setting", "Proxy search", "Final training"], [
        ["Optimizer", "Adam", "AdamW"],
        ["Initial learning rate", "0.001", "0.003"],
        ["Batch size through core", "64", "128"],
        ["Epochs", "3 by default", "20 by default in the mutation loop; user-declared budget; function default 50"],
        ["Weight decay", "Adam default: 0", "0.0001"],
        ["Schedule", "No scheduler", "CosineAnnealingLR"],
        ["Loss", "Class-weighted cross-entropy", "Class-weighted cross-entropy"],
    ]),
    qa("What is an epoch?", "One pass through the training set. A batch is one group of examples inside that pass. A generation is one population of architectures. An epoch and a generation are not the same unit."),
    qa("What do class weights do?", "Weights are based on inverse training-class frequency and normalized. Mistakes on underrepresented classes get more weight in the loss. This already exists; it does not create new images, oversample them, or guarantee balanced F1."),
    callout("THE TWO WINNERS", "Search chooses an <b>architecture by fitness</b>. Final training saves the <b>epoch checkpoint with highest validation accuracy</b>. Early stopping is a proposal, not part of the current training loop."),
], tag="TRAINING ENGINE", source="neuroforge_core.py: _class_weights, evaluate_candidate, full_train; frontend/src/components/MutationLoopPage.jsx; server.py.")

page("The dataset's first day", "Before learning, the files must become consistent numbers.", [
    note("Images: the actual preprocessing", "Local path or ZIP → discover class folders or Roboflow CSV → open RGB image → resize to <b>64×64</b> → training-only horizontal flip and up to 8° random rotation → tensor → normalize each RGB channel with mean/std 0.5.<br/>Evaluation and live image inference use deterministic resizing and normalization."),
    qa("How does the model know the classes?", "Folder names or CSV columns define the label vocabulary. Labels are mapped to indices; the final layer has one output per class. The ordered class list must travel with the checkpoint. Index 3 has no meaning without that mapping."),
     note("Formats the current production workflow understands", "<b>ImageFolder:</b> dataset/class-name/image.jpg.<br/><b>Roboflow classification:</b> train, valid and test directories with _classes.csv; only single-label rows are used. The React workbench accepts a local path or ZIP for image datasets. <b>Audio core:</b> WAV class folders plus ESC-50 and UrbanSound8K metadata layouts are detected; log-mel tensors and fold metadata are implemented, while production-web intake and benchmark reporting remain next work.<br/><b>Fallback:</b> CIFAR-10 remains available in the core; it uses 32×32 inputs."),
    qa("What about messy ZIPs?", "The image loader filters macOS __MACOSX folders, ._ sidecar files and non-image entries, and normalizes nested image roots. A dataset folder is not automatically a valid benchmark: labels, duplicates and split design still need checking."),
    callout("DO NOT MIX THESE UP", "<b>Normalization</b> changes numerical scale. <b>Augmentation</b> creates training variations. <b>Splitting</b> determines which examples may influence training and selection. None replaces the others."),
], tag="INPUT ADAPTERS", source="neuroforge_core.py: _image_transform, get_dataset_metadata, CleanImageFolder; server.py dataset endpoints; frontend/src/components/MutationLoopPage.jsx.")

page("The exam must stay sealed", "An important correction to the earlier conversation.", [
    ("compare", [
        ("WHAT A FAIR DESIGN NEEDS", "<b>Train:</b> learn weights.<br/><b>Validation:</b> choose architecture and checkpoint.<br/><b>Test:</b> score the finished decisions once.<br/>Keep related originals, crops and augmented copies in the same split."),
        ("WHAT CURRENT IMAGEFOLDER DOES", "Creates approximately <b>80% train / 20% evaluation</b> by class.<br/>Both valid and test requests use that same evaluation subset.<br/>It is held out from gradient updates, but not from model selection."),
    ]),
    callout("WHY THIS MATTERS FOR THE CROP RESULT", "The JSON key is named held_out_test_metrics, but the crop ImageFolder result is <b>not an independent final test</b>. Search and checkpoint selection already used those examples. A field name cannot make the evaluation independent."),
    note("Other paths", "<b>Roboflow:</b> separate supplied train/valid/test directories are used, although cross-split duplicate/source leakage is not automatically audited.<br/><b>CIFAR-10 fallback:</b> the official test partition is reused during candidate selection; it needs a separate validation split.<br/><b>Audio folds:</b> separate folds are used, but a full cross-validation runner is not implemented."),
    note("The improvement to make next", "Implement and persist a true train/valid/test manifest for ImageFolder; group by original source when needed. Use validation for every development choice. Then evaluate an untouched test set. On very small data, use a carefully designed nested/grouped validation protocol."),
    callout("SAY THIS TO THE EVALUATOR", "“The image workflow works, but the crop evaluation protocol still needs independent testing. We are fixing the protocol before making generalization claims.”"),
], tag="HONEST EVALUATION", source="neuroforge_core.py: get_split_indices, get_dataloaders (ImageFolder/CIFAR paths), full_train.")

page("The real scorecard", "A dated snapshot, not a promise about the next run.", [
    ("snapshot",),
     note("Current means the files inspected for this guide", "The latest saved run is <b>20261103T112330Z</b>, a 3-generation × 4-candidate fruit DynamicCNN search with 2 proxy epochs. Its selected DynamicCNN row is a proxy search result; the separately trained ResNet-18 comparison has a recorded final test artifact. The two rows must not be described as equal training conditions."),
    ("table", ["Model / budget", "Accuracy · Precision · Recall · F1 · MSE · RMSE", "How to interpret it"], [
        ["Manual CNN / 2-epoch proxy", "50.00 · 52.61 · 46.10 · 43.22 · 0.0797 · 0.2824", "Fixed reference; proxy metrics, not a final-trained comparison."],
        ["Selected DynamicCNN / 2-epoch proxy", "74.19 · 71.07 · 69.95 · 67.47 · 0.0417 · 0.2043", "Mutation winner in the latest saved search; not a final independent test."],
        ["ResNet-18 / 20-epoch final", "70.79 · 70.25 · 62.01 · 64.01 · 0.0479 · 0.2189", "Fixed transfer-learning baseline; same active fruit test split."],
    ]),
    callout("THE FAIR COMPARISON", "Accuracy, precision, recall and F1 are percentages. MSE and RMSE are probability-vector diagnostics in raw 0–1 units, so lower is better. Compare models on the <b>same dataset, split and declared budget</b>; the manual row is deliberately labelled proxy while the two final rows use 20 epochs."),
    note("Current artifact inventory", "The latest saved image search records 12 candidate slots, 10 unique candidate configurations plus the manual baseline, and 2 reused candidate evaluations. The comparison keeps a separate ResNet-18 checkpoint so training it does not overwrite the selected DynamicCNN. Audio needs modality-specific bundles before switching between image and sound runs. The project still has no model registry."),
], tag="RECORDED RESULTS", source="best_model_metadata.json, search_history.json, manual_baseline.json, resnet18_comparison_metadata.json and the /api/comparison response.")

page("Why is the crop score low?", "A historical crop-pest case study—not the current fruit checkpoint.", [
    qa("Why only 42.62% in the old crop Mutation lab?", "That was the selected candidate's <b>3-epoch proxy accuracy</b> in the historical crop-pest run. It ranked cheaply trained architectures. New children started from scratch, and some performed worse than their parents. That run's later full-trained generated model reported 44.26% on the reused evaluation set; it is not the current fruit checkpoint."),
    note("What the current evidence actually says", "There are <b>295 images across five classes</b>: 234 training examples and 61 evaluation examples. Evaluation class supports are 14, 7, 4, 20 and 16. The largest class accounts for 20/61 = <b>32.79%</b>; always predicting that class is a useful naive baseline. Uniform random guessing has 20% expected accuracy."),
    qa("What could be limiting performance?", "Tiny and uneven class coverage is confirmed. Small pest details at 64×64, background shortcuts, confusing labels, the selected two-layer design and heavy dropout may also matter. These are hypotheses to test—not causes proven just by looking at the score."),
    qa("Why doesn't more mutation automatically help?", "It explores more recipes, but it cannot invent missing training evidence. Shallow evaluations may choose fast starters instead of the best long-term learner. The search only explores a restricted CNN family and does not tune the whole training recipe."),
    callout("90% IS A TARGET, NOT A SWITCH", "On the old 89-image fruit set, two additional correct predictions would move 79/89 = 88.76% to 81/89 = 91.01%. That is small-test-set sensitivity, not a sound reason to tune against those test images."),
], tag="YOUR LOW-SCORE QUESTION", source="Historical crop split_indices and metrics; core image transforms; current fruit run is documented separately on the scorecard page.")

page("Make crop F1 better", "A prioritized experiment plan that keeps mutation central.", [
    ("steps", [
        ("01  FIX THE MEASURING TAPE", "Create independent train/validation/test splits first. Group duplicate crops and same-source images. Preserve class coverage; otherwise an apparent improvement may just be leakage."),
        ("02  INSPECT THE MISTAKES", "Use the confusion matrix and a gallery of misclassified examples. Check labels, background shortcuts and weak classes. Add genuinely different, correctly labelled images to the underrepresented cases."),
        ("03  KEEP MORE USEFUL DETAIL", "Compare 64 vs 128 pixels on validation, keeping the budget documented. If pests occupy tiny regions, consistent object cropping may help. Update training and inference preprocessing together."),
        ("04  IMPROVE THE INNER LOOP", "Try modest crop/brightness changes appropriate to pests, a smaller learning-rate range, and validation-based early stopping. Class-balanced loss already exists. More augmentation is not always better."),
        ("05  IMPROVE THE OUTER LOOP", "Continue evolving CNNs. Re-evaluate a short list of promising designs with deeper budgets and several seeds before choosing. This multi-fidelity approach needs implementation and quality checks."),
        ("06  COMPARE, THEN LOCK", "Fully train a fixed CNN and a separate transfer-learning baseline fairly. Choose with validation, then evaluate the locked winner once on the untouched test set."),
    ]),
    callout("IF YOU ONLY DO ONE THING", "Improve the evidence: honest splits plus more varied, balanced examples. Switching to ResNet can be a useful baseline, but its accuracy is not proof that the mutation engine improved."),
], tag="PROPOSED / QUALITY PLAN", source="Recommendations, not completed experiments. Existing loss/augmentation verified in neuroforge_core.py.")

page("Faster, without fooling us", "Remove wasted work first. Cutting the experiment is a separate trade-off.", [
    note("Where time goes", "The current saved run used <b>(12 candidates + 1 baseline) × 2 = 26 proxy training epochs</b> before final training; 2 candidate evaluations were reused, so 22 model-epochs actually ran. The production workbench starts at 3 generations × 4 candidates × 2 proxy epochs and lets the user declare a different budget. The older crop run used 40 candidates and 3 proxy epochs. Disk loading, transfers and validation also cost time."),
    ("table", ["Move", "Status", "Quality / runtime implication"], [
        ["Reuse duplicate architectures", "Implemented", "Same run, seed and budget; fewer repeated evaluations."],
        ["Cache audio log-mel features", "Implemented in core", "Avoid repeated STFTs; the production web workbench is currently image-focused."],
        ["Tune loaders / decoded-image cache", "Proposed", "Benchmark end-to-end; preserve train-only random augmentation."],
        ["Early stopping / staged budgets", "Proposed", "Can save work, but may discard slow-learning good candidates."],
        ["Fewer candidates / epochs", "Demo trade-off", "Faster, but less exploration or ranking reliability."],
    ]),
    qa("Can you promise equal quality and less time?", "Not before measurement. Cache reuse removes obvious repetition; other changes can alter optimization or ranking. Compare the same workload and seeds, report runtime and accuracy/F1 together, and keep the final evaluation protocol fixed."),
    note("Time-reporting trap", "A reused history entry copies the original training duration. Summing every history row can double-count work that did not run again. Use wall-clock search duration and unique evaluation counts. GPU timing also needs warm-up and synchronization for a clean benchmark."),
    callout("GOOD SPEED CLAIM", "“We avoided four duplicate candidate evaluations in this run.”<br/><b>Not yet supported:</b> “The whole app is 10× faster with identical quality.”"),
], tag="YOUR RUNTIME QUESTION", source="architect.py metrics_cache and reuse flags; core loaders; audio_pipeline.py _FEATURE_CACHE; frontend/src/components/MutationLoopPage.jsx.")

page("New dataset, better science", "Kaggle is a source, not a quality certificate.", [
    ("table", ["Dataset", "Why use it?", "Preparation / caveat"], [
        ["Crop-pest data", "Directly matches the agriculture story.", "295 examples; improve coverage and fix independent evaluation."],
        ["Oxford-IIIT Pet", "37 breeds, roughly 200 images/class; established image benchmark.", "Fine-grained breeds can be hard. Preserve official splits; adapter/conversion needed."],
        ["ESC-50 / ESC-10", "2,000 five-second clips, 50 classes; ESC-10 is a smaller subset.", "Preserve predefined folds and source groups. Code has one fixed fold split."],
        ["UrbanSound8K", "8,732 excerpts, 10 urban-sound classes; up to four seconds.", "Keep ten official folds. WAV encoding support must be checked."],
    ]),
    qa("Will a bigger dataset automatically give 90%?", "No. More classes or harder images can reduce accuracy while making the benchmark more valuable. Choose data that matches the intended problem, has reliable labels, and gives enough independent examples to evaluate each class."),
    note("Before downloading from Kaggle or another site", "Check label meanings, licence, class counts, sample quality, duplicates, original-source groups and the official split instructions. Prefer the original publisher over a repackaged mirror when possible. A clean-looking archive can still leak near-identical images across splits."),
    callout("RECOMMENDED DIRECTION", "Keep crop pests as the motivating task. Add a recognized image benchmark to test generality, and ESC-50/ESC-10 for the audio pipeline. Treat each as its own experiment—not interchangeable leaderboard numbers."),
], tag="DATASET CHOICES", source="Oxford VGG dataset page; ESC-50 official README; UrbanSound8K official dataset page. Links on the Sources page.")

page("New images: full steps", "A practical path from download to a result you can defend.", [
    ("steps", [
        ("1 / DOWNLOAD + KEEP THE ORIGINAL", "Get images and labels from the publisher. Keep the raw download unchanged. Record dataset version, source and licence. Keep local data out of Git."),
        ("2 / AUDIT + SPLIT", "Count labels, open examples, fix wrong labels, group near-duplicates by original source. Keep official test membership. Split only the development/training pool into train and validation."),
        ("3 / PREPARE A SUPPORTED FORMAT", "For a proper three-way image experiment, use a Roboflow-style classification export: train/valid/test directories, each with _classes.csv and the same class columns. Oxford Pets needs a converter/adapter; the app does not automatically read its annotations."),
        ("4 / CONNECT IN THE UI", "Run <font name='Mono'>./run_studio.sh</font>, open <font name='Mono'>http://127.0.0.1:8000/?view=loop</font>, then mount a local dataset path or upload a ZIP. Check the class names and sample count. A plain ImageFolder is usable for development but currently has the two-way split limitation."),
        ("5 / RUN THE MUTATION CYCLE", "Use a small run first only to check plumbing. For a benchmark, choose the budget beforehand, retain lineage and baseline results, and choose candidates using validation only."),
        ("6 / TRAIN + TEST + SAVE", "In the Ship panel, declare the full-training epoch budget and train the selected architecture. Download the checkpoint and blueprint, then return to the main studio for in-scope image inference. Preserve the JSON artifacts before changing datasets."),
    ]),
    note("CSV shape / illustration", "<b>filename,apple,orange</b><br/>sample-001.jpg,1,0<br/>sample-002.jpg,0,1<br/>Use the same column order in every split. The labels come from the dataset—not from model predictions."),
    callout("DO NOT ACCIDENTALLY TEST YOUR TRAINING SET", "A successful prediction on a training image proves the inference path runs. It is not evidence of generalization. Name which split your demo sample came from."),
], tag="DATASET RUNBOOK", source="server.py dataset endpoints; frontend/src/components/MutationLoopPage.jsx; conversion and independent split preparation are required work, not a built-in Oxford adapter.")

page("Audio: give sound a shape", "We turn a waveform into a pattern map, then reuse the mutation engine.", [
     ("audio",),
     note("The current audio recipe", "<b>WAV → mono → 16,000 samples/second → center-crop or pad to 4 seconds → STFT → power → 64 mel bands → logarithm → per-clip normalization → one-channel CNN.</b><br/>With the current settings, a clip becomes approximately <b>[1, 64, 251]</b>: channel, frequency bands, time frames."),
     qa("What are STFT and log-mel?", "STFT examines short overlapping windows to show how frequencies change over time. Mel bands group frequencies on a perception-inspired scale. The logarithm compresses large energy differences. Think of it as a map of the sound, not a transcript."),
    note("Actual settings and augmentation", "FFT/window length 512 samples; hop 256; Hann window. Training can scale the normalized features, add small noise, and mask a short time span. Base features are cached. Standard-library wave decodes audio; PyTorch performs the transforms. Librosa and torchaudio are not currently used."),
     qa("Is audio ready?", "The core data/search/train/predict path supports the audio tensor contract and passed synthetic-WAV checks earlier. A real ESC-50/UrbanSound8K benchmark, a FastAPI audio route and the React audio workflow remain pending. Audio is sound classification, not speech-to-text, conversation, or joint image/audio reasoning."),
    callout("QUALITY DETAIL TO REMEMBER", "The resampler is basic linear interpolation, not an anti-aliased audio resampler. ESC-50 has five-second clips; the current four-second center crop can discard events. Both deserve validation before a serious benchmark."),
], tag="AUDIO FOUNDATION", source="audio_pipeline.py constants, _resample, _load_features, waveform_to_log_mel, audio_sample_from_bytes.")

page("Audio: first real benchmark", "Keep the same mutation story, and preserve the dataset's rules.", [
    ("steps", [
        ("1 / GET ESC-50", "Download the official archive from its repository and extract it. Keep audio/ and meta/esc50.csv together. For ESC-10, filter the esc10 flag but preserve metadata and original folds; there is no built-in subset toggle."),
        ("2 / CHECK FILES + LABELS", "Confirm WAV encoding, category counts and fold coverage. Missing or unreadable files must be reported. The current loader supports a limited set of PCM encodings; test compatibility before launching a long search."),
         ("3 / CONNECT + SMOKE-CHECK", "Use the core/audio path or the preserved Streamlit backup for an audio smoke check; the production React workbench still needs the audio intake surface. Run a tiny mutation cycle, then a short training pass and an in-scope WAV upload to verify shapes, labels and artifact reload."),
        ("4 / UNDERSTAND TODAY'S SPLIT", "ESC-50 uses folds 1–3 for train, 4 for validation and 5 for test. UrbanSound8K uses 1–8 / 9 / 10. These are development holdouts, not full official cross-validation scores."),
        ("5 / COMPLETE THE BENCHMARK", "For a reportable benchmark, implement the publisher's fold protocol. Architecture and checkpoint choices must use only development folds for each outer test fold. Aggregate results across folds; do not choose the best test fold."),
        ("6 / SAVE THE WHOLE BUNDLE", "Export audio weights, architecture, ordered labels, preprocessing settings, splits and metrics. The project still reuses the same default artifact filenames across modalities, so save each run before switching."),
    ]),
    callout("THE SHORT AUDIO PITCH", "“The input adapter changes, but the evolutionary search idea stays. We mutate CNN architectures over log-mel features and measure the classifier on separate folds.”"),
], tag="AUDIO RUNBOOK", source="audio_pipeline.py fold handling; preserved Streamlit audio path; official ESC-50 and UrbanSound8K evaluation instructions.")

page("The apple became a bug", "A useful failure story—not something to hide from your teammate.", [
    callout("THE MENU ANALOGY", "Ask a restaurant that sells five pest labels to serve “apple.” It still has only five options on the menu. Softmax distributes probability across those options even when none is correct."),
    qa("Why did it predict Rice Bug at 62.62%?", "At that moment the saved model was a five-class crop-pest ResNet-18 baseline, not the earlier nine-class fruit DynamicCNN. Apple was outside its trained label set. The label was wrong for the image, but not proof that fruit recognition was randomly relabelled."),
    qa("Did the new warning solve unknown detection?", "It made the scope clearer. The production panel warns below 40% and explains that a five-class result near the 20% random baseline is uncertain. It does <b>not</b> make the model reliably reject apples. Known-versus-unknown detection and confidence calibration need separate data and evaluation."),
    qa("Why did “train first” keep returning?", "The old UI mixed session flags, selected dataset/architecture and saved metadata. The production site now separates the FastAPI workspace state from the saved checkpoint, exposes readiness through <font name='Mono'>/api/status</font>, and has a dedicated mutation route. Immutable per-run bundles and a registry are still future work."),
    note("The robust design we still want", "One immutable bundle per trained run: model type + config + weights + ordered labels + preprocessing + dataset/split identity. Inference should read that bundle, not whatever session classes happen to be active. A config hash alone does not identify trained weights or dataset contents."),
    callout("TEACH THE LESSON", "Model quality, UI state and input scope are different debugging layers. Check the loaded model's identity before interpreting a confidence number."),
], tag="FAILURE CLINIC", source="Earlier failure screenshots and checkpoint inspection; server.py workspace/checkpoint status; frontend/src/components/InteractiveCrucible.jsx.")

page("Meet the production stack", "Five teammates inside the codebase. Each has a different job.", [
    note("frontend/src / THE STUDIO", "React/Vite production UI: editorial home, Build/Mutate/Ship opening sequence, theme, topology specimen, live image inference and export. <font name='Mono'>MutationLoopPage.jsx</font> adds dataset intake, search controls, candidate streaming, training, model comparison and downloads."),
    note("server.py / THE ENGINE ROOM", "FastAPI serves the built frontend and exposes status/history, dataset mount/upload/unload, training, image prediction, comparison, downloads, <font name='Mono'>/health</font>, and the <font name='Mono'>/ws/search</font> stream."),
    note("architect.py / THE TOURNAMENT ORGANIZER", "Defines the seed design and manual baseline; mutates configurations; selects parents and elites; calculates fitness; caches duplicate evaluation results; records candidate IDs, ancestry, mutations and scores; writes the winning architecture and search history."),
    note("neuroforge_core.py / THE TRAINING ENGINE", "Defines DynamicCNN and the ResNet baseline path. Loads images, chooses MPS/CUDA/CPU, computes loss and metrics, trains proxy and final models, saves checkpoints, reloads them and performs inference. Audio loaders plug into this engine."),
    note("audio_pipeline.py / THE SOUND TRANSLATOR", "Discovers WAV layouts, decodes/resamples clips, builds log-mel features and selects folds. It remains a backend foundation; the production workbench is image-focused."),
    ("formula", "React/Vite → FastAPI → architect.py → neuroforge_core.py", "Audio preparation delegates to audio_pipeline.py. The original Streamlit app is preserved as app_streamlit_backup.py."),
    callout("IF THE EVALUATOR ASKS “WHERE IS MUTATION?”", "Open <b>architect.py → mutate_config()</b>, then <b>run_architect_search()</b>. For proof of a specific edit, show its mutation_operations and parent_id in search_history.json."),
], tag="CODE TOUR", source="frontend/src/App.jsx, frontend/src/components/MutationLoopPage.jsx, server.py, architect.py, neuroforge_core.py, audio_pipeline.py.")

page("Files that remember things", "The blueprint, the learned numbers, and the receipt must agree.", [
    ("table", ["File", "Exactly what it contains"], [
        ["best_model_config.json", "Selected architecture blueprint: layers, filters, kernels and other traits."],
        ["best_model_weights.pth + resnet18_comparison_weights.pth", "Separate PyTorch state_dicts for the selected DynamicCNN and the ResNet-18 comparison. Neither is a self-contained app."],
        ["best_model_metadata.json + resnet18_comparison_metadata.json", "Dataset/classes, run settings, model identity, training mode and reported metrics for each checkpoint."],
        ["search_history.json", "All candidate records: parents, generations, mutation operations, configs, metrics and fitness."],
        ["manual_baseline.json", "The fixed manually designed CNN's proxy results. Not the ResNet baseline."],
        ["candidate_config.json", "Transient config for the candidate currently being evaluated."],
        ["server.py", "FastAPI routes, WebSocket search stream, checkpoint status and static frontend serving."],
        ["frontend/src + frontend/dist", "Editable React/Vite studio source and the production bundle served by FastAPI."],
        ["run_studio.sh + app_streamlit_backup.py", "Production launcher plus the preserved legacy Streamlit implementation."],
    ]),
    note("The libraries / production stack", "<b>React 18 + Vite:</b> the production interface and build.<br/><b>Three.js:</b> interactive topology specimen.<br/><b>FastAPI + Uvicorn:</b> API, WebSocket search stream and static serving.<br/><b>PyTorch 2.14.0 + Torchvision 0.29.0:</b> models, training, image transforms and ResNet-18.<br/><b>Streamlit 1.64.0:</b> retained for the legacy backup path."),
    callout("THE SUPPORT CREW", "Python json, csv, pathlib, zipfile, wave, random, hashlib and time handle storage, labels, files, audio decoding, random choices, config hashes and timing. The browser uses fetch and WebSocket calls; metrics are implemented with PyTorch."),
], tag="ARTIFACTS + STACK", source="requirements.txt; module imports; artifact writers and loaders.")

page("Where the project stands", "A working prototype with a real differentiator and specific unfinished work.", [
    ("compare", [
        ("IMPLEMENTED", "React/Vite production studio served by FastAPI.<br/>Refresh-time Build/Mutate/Ship opening scene.<br/>Light/dark theme persistence.<br/>Image path mounting and ZIP upload/unmount/delete controls.<br/>WebSocket candidate/generation streaming.<br/>Generation, population and epoch controls.<br/>Final training and checkpoint/blueprint downloads.<br/>Live image inference with explicit low-confidence messaging.<br/>Side-by-side accuracy/precision/recall/F1/MSE/RMSE comparison.<br/>Separate ResNet-18 transfer baseline training and download."),
        ("NOT YET ESTABLISHED", "Consistent 90%+ results.<br/>Independent crop test performance.<br/>Multi-seed statistical reliability.<br/>Full real-audio web benchmark.<br/>Validated unknown-input rejection.<br/>NAS superiority to random search.<br/>Immutable model registry or deployment automation.<br/>Measured universal speed-up.<br/>A final-trained manual baseline under the same budget as the two champion rows."),
    ]),
    note("Current limitations worth knowing", "The search space is small and uses mutation only, without crossover or inherited weights. It does not tune learning rate/resolution/optimizer. Candidate evaluation is sequential. No validation early stopping or staged-budget search is implemented. The project still stores one mutable default best-model bundle; the production workspace and checkpoint status are clearer, but an immutable registry is not implemented."),
    note("Audio-specific limitations", "Fixed-length center cropping may lose important events. The resampler is basic. The feature cache is process-global, unbounded and keyed by path, so changed files can produce stale features. WAV decoding is limited. Audio adapters are implementation support, not proof of benchmark accuracy. The FastAPI route, React intake/playback/spectrogram surface and modality-specific artifact registry are still required."),
    callout("NEXT PRIORITIES", "1. Independent crop splits and immutable model bundles.<br/>2. Class-level data/error analysis and fair baseline studies.<br/>3. A measured runtime improvement with equal-quality checks.<br/>4. Real audio benchmarks and fold aggregation.<br/>5. Add a production audio workbench only after the benchmark protocol is defined."),
], tag="CURRENT STATE", source="Inspected production frontend, FastAPI service and current artifacts; synthetic audio checks are historical smoke evidence, not a fresh benchmark.")

page("Evaluator: round one", "Short answers that survive a follow-up question.", [
    qa("What problem are you solving?", "Choosing a CNN by hand is trial and error. We automate part of that design search and expose which architecture changed, who its parent was, and how the candidate performed."),
    qa("What is your contribution?", "An integrated mutation-driven NAS workflow with data input, dynamic model construction, multi-metric ranking, traceable lineage, final training and inference. Evolutionary NAS already exists; our project is its implemented, inspectable application."),
    qa("Why use mutation instead of just training ResNet?", "ResNet evaluates an existing fixed backbone. Our core question is whether evolving small CNN designs can find useful task-specific trade-offs. ResNet is a comparison baseline, not the source of our mutation novelty."),
    qa("How is this different from random search?", "Generation one explores around a default design; later children come from high-fitness parents and an elite is preserved. To prove that this beats random search, we need a matched-budget random-search baseline. We have not established that yet."),
    qa("What prevents a generation from getting worse?", "The best prior design is copied into an elite slot and the global best is retained. New children may be worse. Reused scores make elite ranking stable within a run; genuinely retraining can still vary."),
    qa("Do you guarantee the best architecture?", "No. We return the highest-scoring evaluated candidate in a finite search space under a specific data split, budget and score. That is a practical search result, not a global optimum."),
], tag="VIVA / CORE", source="Architecture-search behavior verified in architect.py.")

page("Evaluator: round two", "These questions separate understanding from memorizing a pitch.", [
    qa("Why not just optimize accuracy?", "A majority class can dominate accuracy. Macro metrics treat classes equally, while size and latency matter for deployment. The current weighted objective is a heuristic and needs sensitivity analysis."),
    qa("Are your final test results independent?", "For Roboflow, supplied validation/test directories are distinct, subject to a dataset leakage audit. For ImageFolder crop data, the current code reuses its evaluation subset for selection and final reporting. We must repair that before claiming independent test performance."),
    qa("What did 62.62% confidence mean for that apple?", "It was the largest relative probability among pest labels from the loaded pest model. It was neither a calibrated correctness estimate nor evidence that the image belonged to the pest domain."),
    qa("Why is seed 42 not enough for reproducibility?", "It controls several random choices and helps repeat a run. Hardware kernels, data changes and dependencies can still differ. Multiple seeds measure variability; dataset versions, splits, config and software versions must also be recorded."),
    qa("Is the baseline comparison fair?", "The fixed manual CNN and candidates receive the same proxy budget and evaluation split. The selected model gets additional final training. A strong final comparison still needs a fully trained manual baseline, comparable tuning budgets and repeated runs."),
    qa("Does the system deploy models automatically?", "It exports local model artifacts and runs local inference. “Ship” currently means an inference-capable MVP bundle, not automatic cloud deployment, a model registry or a production service."),
], tag="VIVA / DEPTH", source="Core data/training paths, score implementation, UI and exports. This page distinguishes current behavior from stronger claims.")

page("Teach it in ten minutes", "A smooth teammate handoff: story → mechanism → evidence → honest limits.", [
    ("steps", [
        ("00:00–01:00 / THE HOOK", "“What if choosing a CNN was a visible tournament instead of blind trial and error?” Ask your teammate to explain why architecture and weights are different."),
        ("01:00–03:00 / THE TWO LOOPS", "Draw outer evolution around inner training. Use the robot-blueprint analogy. Show one actual parent ID and the dropout 0.2 → 0.4 mutation."),
        ("03:00–05:00 / THE SCOREBOARD", "Show the fitness formula and the confusion-matrix example. Ask: “If I predict rice bug for everything, can accuracy look okay?” Then explain macro F1."),
        ("05:00–07:00 / THE APP", "Open the studio at <font name='Mono'>127.0.0.1:8000</font>, then enter <font name='Mono'>?view=loop</font> for the workbench. Mount the dataset, set a small budget, and show the live candidate stream before returning to the topology and inference views."),
        ("07:00–09:00 / THE PLOT TWIST", "Tell the apple-versus-pest story. Explain model scope, confidence and artifact identity. Then explain the crop split limitation; this is where technical honesty earns trust."),
        ("09:00–10:00 / THE TEACH-BACK", "Ask them to explain the entire pipeline in five arrows and answer: “What is the next experiment that would prove quality improved?”"),
    ]),
    callout("TWO-PERSON PRESENTATION", "<b>Person A:</b> problem, evolution and demo navigation.<br/><b>Person B:</b> training, metrics, evidence and limitations.<br/>Both should be able to answer the apple question and distinguish proxy from final performance."),
], tag="TEAMMATE MODE", source="Teaching script based on the current workflow; timing is a suggested presentation plan.")

page("Demo day: stay in control", "Make the audience follow the evidence, not the loading spinner.", [
    note("Start locally / from the repository", "<font name='Mono'>./run_studio.sh</font><br/>Open <b>http://127.0.0.1:8000/</b> for the studio or <b>http://127.0.0.1:8000/?view=loop</b> for the mutation workbench. <font name='Mono'>GET /health</font> proves the service responds; <font name='Mono'>GET /api/status</font> shows device, workspace and checkpoint state. The opening sequence replays on refresh and can be skipped."),
    ("steps", [
        ("BEFORE YOU PRESENT", "Read metadata for model type, dataset, ordered classes and run ID. Check that config and weights belong together. Export the current bundle before starting a new search or switching data."),
        ("SHOW THE DIFFERENTIATOR", "In the mutation workbench, mount the crop dataset, set a declared budget, and explain a parent, a child and one from→to edit as the WebSocket feed arrives. Explain that an elite is preserved. The full search_history.json contains all candidate records."),
        ("SHOW THE EVIDENCE", "Say whether the displayed metric is proxy, validation or independent test. Use the latest saved result, not a favourite old screenshot. Name the dataset and model with every number."),
        ("TRY AN IN-SCOPE SAMPLE", "Use an image from the model's actual class vocabulary and identify its split. A pest model is not a fruit model. An uncertain or wrong prediction is a chance to explain limitations, not to rename the output."),
        ("CLOSE WITH A CONCRETE NEXT STEP", "“Our next priority is independent crop evaluation and immutable run bundles, followed by fair repeated-seed and audio benchmarks.”"),
    ]),
    callout("30-SECOND RESCUE ANSWER", "“NEUROFORGE evolves CNN blueprints through parent selection and mutation. PyTorch trains their weights. We rank candidates using classification metrics plus cost penalties, retain lineage, and train the chosen design. The prototype works; evaluation reliability is our next engineering priority.”"),
], tag="PRESENTATION RUNBOOK", source="run_studio.sh; server.py; frontend/src/App.jsx; frontend/src/components/MutationLoopPage.jsx; current artifact behavior.")

page("Boss fight: test yourself", "Close the explanation pages. Answer these out loud.", [
    ("table", ["#", "Question to answer out loud"], [
        ["01", "A child changes filters[1] from 64 to 128. Which loop made this change? Does it inherit the parent's trained weights?"],
        ["02", "The saved run used three generations, four candidates and two proxy epochs. How many candidate slots? How many uncached model evaluations including one manual baseline?"],
        ["03", "A = 70%, P = 65%, F = 60%, N = 200,000 and L = 0.003 seconds. What fitness does our formula give?"],
        ["04", "Can a majority-class model get high accuracy and poor macro F1? Why?"],
        ["05", "A pest model assigns an apple 95% confidence. Is that in-scope?"],
        ["06", "If selection used the same data called “test,” is it independent?"],
        ["07", "The saved run has 12 candidate slots, two reused results and one manual baseline: how many actual evaluations?"],
        ["08", "What can a synthetic audio smoke test prove—and not prove—about ESC-50?"],
    ]),
    note("Bonus / your own words", "Give the one-line product pitch without saying “AI magic,” “guaranteed 90%,” or “globally best model.” Answers are on the next page."),
], tag="ACTIVE RECALL", source="Questions derived from actual code paths. Numerical fitness inputs are illustrative.")

page("Answer key + memory hooks", "If you can explain why, you own the concept.", [
    ("table", ["Question", "Answer"], [
        ["01", "Outer evolution changes the architecture. No learned-weight inheritance is currently implemented."],
        ["02", "12 candidate slots; 11 unique evaluations including the manual baseline, each at 2 proxy epochs, so 22 uncached model-epochs before final training."],
        ["03", "31.5 + 16.25 + 18 − 0.2 − 0.3 = 65.25 fitness points. Not 65.25% accuracy."],
        ["04", "Yes. Majority examples dominate accuracy. Macro F1 gives minority classes equal weight."],
        ["05", "No. Softmax scores only the known classes. High confidence can still be confidently wrong."],
        ["06", "No. Selection already used the examples; renaming the result does not create independence."],
        ["07", "11 actual evaluations including the manual baseline. The 2 reused candidate rows remain part of lineage history but did not retrain."],
        ["08", "The path runs on the synthetic fixture. That establishes plumbing, not real sound-recognition quality."],
    ]),
    callout("FIVE HOOKS TO KEEP", "<b>Blueprint vs weights:</b> recipe vs learned skill.<br/><b>Mutation vs training:</b> change the design vs teach the design.<br/><b>Precision vs recall:</b> trust the prediction vs find the real cases.<br/><b>Validation vs test:</b> choose the answer vs seal the final exam.<br/><b>Confidence vs accuracy:</b> one model belief vs many checked outcomes."),
    note("Bonus answer", "“NEUROFORGE is a mutation-based CNN architecture-search studio with visible lineage, measured candidate selection, final training and local inference for image and audio workflows.”"),
], tag="ANSWER KEY", source="Teaching examples; the cache count also matches the current saved run.")

page("The project story so far", "Your feedback, the repo, the diagram and the video all fit around one idea.", [
    qa("Where were we when I came back?", "The image prototype, private GitHub package, architecture diagram and 21-second brag video had been created. The current milestone is a React/Vite + FastAPI production studio with a dedicated mutation workbench, while the original Streamlit app is preserved as a backup."),
    qa("Had I given criticism or feedback?", "Yes. You approved the video, asked for easier file access, questioned low accuracy and runtime, disliked the neuron/orb visual, and insisted mutation remain the main identity. The repeated warning and apple prediction were also concrete corrective feedback."),
    qa("What do the diagram and brag video do?", "The Archify HTML explains runtime components and evidence links. The brag-slim video presents the product story. They are communication artifacts; neither trains models nor proves accuracy. Older visuals may show earlier run results."),
    qa("How does GitHub help our teammate?", "It packages source, setup instructions and documentation for analysis and presentation. The repository is private, so collaborator access matters. Local datasets and environments are ignored. Check the current branch/files rather than assuming every local change is already pushed."),
    note("Design decision to remember", "Keep <b>Build. Mutate. Ship.</b>, lineage and mutation operations visible. The production frontend now uses a paper/graphite materials language with cobalt, coral and acid-lime accents, a topology specimen and a refresh-time opening sequence. The physics layer adds spring cursor response, local node gravity, flexible synapses, drag inertia and impact feedback. The technical idea stays visible while the visual treatment remains replaceable."),
    callout("WHAT THIS GUIDE ADDS", "One consistent explanation of the project, with current facts separated from old results and planned upgrades. You can teach the system without relying on the order of the chat history."),
], tag="YOUR NON-TECHNICAL Q&A", source="Conversation history; frontend/src; server.py; repository package and presentation artifacts.")

page("Strong claims, clean wording", "Sound confident because you are precise.", [
    ("table", ["Avoid saying", "Say this instead"], [
        ["We invented evolutionary NAS.", "We built an observable mutation-driven NAS workflow and are evaluating its usefulness."],
        ["Mutation guarantees 90%.", "Mutation explores designs; performance depends on data, recipe and evaluation."],
        ["The crop model has a held-out final test.", "Current ImageFolder results reuse the evaluation split; independent testing is pending."],
        ["The child learns from parent weights.", "The child inherits architecture traits; weights are trained from scratch."],
        ["ResNet proves our mutation wins.", "ResNet is a separate transfer-learning baseline."],
        ["We detect every unknown image below 40%.", "We show a heuristic low-confidence warning; unknown detection is unvalidated."],
        ["Audio works at benchmark accuracy.", "Audio core plumbing passed synthetic checks; the production route and real benchmark remain pending."],
        ["F1 rose by excluding a missing class.", "The averaging definition changed; predictions did not improve."],
    ]),
    note("How to answer when you do not know", "“That is not measured yet. We would test it using [a matched baseline / independent split / several seeds], keeping [the budget / dataset / hardware] fixed.” A concrete experiment is a better answer than an invented number."),
    callout("THE PROJECT'S REAL STRENGTH", "It makes architecture search inspectable: design → parent → mutation → measured outcome. Explaining the failure cases honestly strengthens that story."),
], tag="EVALUATOR LANGUAGE", source="Claims checked against the inspected code; proposed experiments are not reported as completed.")

GLOSSARY = [
    ("The learning basics", [
        ("Machine learning (ML)", "Learning patterns from data instead of hand-writing every decision rule."),
        ("Deep learning", "ML using multi-layer neural networks; our CNN is one example."),
        ("Supervised learning", "Learning from inputs with known correct labels. That is this project's training setup."),
        ("Classification", "Choosing a label from a defined list. We assign one class per input."),
        ("Detection / segmentation", "Detection locates objects; segmentation labels pixels. NEUROFORGE currently classifies instead."),
        ("Model / architecture", "A model makes predictions; architecture is its structure or blueprint."),
        ("Weights / parameters", "The learnable numbers adjusted during training. More is not automatically better."),
        ("Hyperparameter", "A choice outside ordinary weight learning, such as learning rate or layer count."),
        ("Tensor", "A multi-dimensional array of numbers. Images and spectrograms become tensors."),
        ("Feature map", "The pattern responses produced by a convolutional layer."),
        ("Training / inference", "Training changes weights using labels. Inference uses trained weights to predict."),
        ("Generalization", "Working well on genuinely new examples, beyond memorized training inputs."),
        ("Overfitting / underfitting", "Overfitting memorizes training quirks; underfitting has not learned enough useful structure."),
        ("Baseline", "A reference method that lets you judge whether a more complex method helps."),
    ]),
    ("CNNs and optimization", [
        ("CNN", "Convolutional neural network: a model using sliding local pattern detectors."),
        ("Kernel / filter", "A learned small matrix. Kernel size is its spatial footprint; filter count is output-channel count."),
        ("Channel", "A feature dimension: RGB has three input channels; our log-mel tensor has one."),
        ("Activation / ReLU", "A non-linear transformation. ReLU replaces negative values with zero."),
        ("GELU / SiLU", "Smooth activation alternatives supported by the mutation search."),
        ("Pooling", "Spatial reduction. Max pooling keeps local maxima; average pooling keeps local averages."),
        ("Batch normalization", "Uses batch/running statistics to normalize intermediate activations."),
        ("Dropout / regularization", "Dropout removes features randomly in training. Regularization broadly discourages overfitting."),
        ("Epoch / batch", "Epoch: one training-data pass. Batch: the examples used in one update."),
        ("Loss / cross-entropy", "A training error signal. Cross-entropy penalizes low probability for the correct class."),
        ("Gradient / backpropagation", "Gradients describe sensitivity of loss; backprop efficiently computes them through layers."),
        ("Optimizer / Adam / AdamW", "The update rule. Adam adapts updates; AdamW adds decoupled weight decay."),
        ("Learning rate / scheduler", "Update step size; a scheduler changes it over training. We use cosine decay in final training."),
        ("Weight decay / early stopping", "Weight decay discourages large weights. Early stopping stops when validation stalls; not implemented here yet."),
    ]),
    ("The evolution vocabulary", [
        ("NAS", "Neural architecture search: automatically exploring neural-network designs."),
        ("AutoML", "A broader umbrella automating ML choices. NAS automates architecture choices specifically."),
        ("Candidate / population", "One proposed design / the collection of designs evaluated in a generation."),
        ("Generation", "One round of candidate evaluation and selection, not a training epoch."),
        ("Mutation", "A randomized architecture edit within allowed rules."),
        ("Parent / child", "The source design / a design derived from it. This does not imply inherited learned weights."),
        ("Lineage", "The recorded chain linking children to parents and their edits."),
        ("Fitness / objective", "The numerical rule used to rank candidate usefulness."),
        ("Selection / elitism", "Choose strong parents / preserve the strongest prior design unchanged."),
        ("Search space", "The set of allowed architectures and trait values the algorithm can explore."),
        ("Exploration / exploitation", "Try new designs / build around already promising designs."),
        ("Crossover", "Combine traits from two parents. Not implemented in the current algorithm."),
        ("Proxy budget / multi-fidelity", "A cheap evaluation budget / comparing at staged budgets. Staged promotion is proposed, not implemented."),
        ("Ablation / random-search baseline", "Remove one feature to test its value / compare with unguided sampled designs under a fair budget."),
    ]),
    ("Data, metrics and trust", [
        ("Train / validation / test", "Learn weights / make development choices / evaluate locked choices independently."),
        ("Stratification / group split", "Preserve class proportions / keep related source examples in the same partition."),
        ("Leakage", "Information from evaluation data influences training or model choices, making results optimistic."),
        ("Augmentation / normalization", "Make training variations / rescale numerical input values consistently."),
        ("Imbalance / class weighting", "Unequal class counts / make loss contributions depend on class frequency."),
        ("Confusion matrix / support", "Counts of truth versus predicted labels / number of true examples of a class."),
        ("TP / FP / FN / TN", "Correct positive / false alarm / missed positive / correct negative, for one class versus the rest."),
        ("Accuracy", "Correct predictions divided by all checked examples."),
        ("Precision / recall", "Correct among predicted members / found among actual members of a class."),
        ("F1 / macro average", "Harmonic balance of precision and recall / average class scores equally."),
        ("Micro / weighted average", "Pool decisions before scoring / weight class scores by their support."),
        ("Logits / softmax / confidence", "Raw class scores / normalize them to sum to one / often the largest resulting probability."),
        ("Calibration / OOD", "Whether confidence matches real correctness / out-of-distribution input unlike the intended data."),
        ("Cross-validation / confidence interval", "Evaluate across partitions / quantify uncertainty of an estimate; neither is automatic evidence here."),
    ]),
    ("Audio and engineering", [
        ("Waveform / sample rate", "Audio amplitude over time / how many samples are stored per second."),
        ("PCM / mono", "Pulse-code modulation stores sampled amplitudes / a single audio channel."),
        ("Resampling / aliasing", "Change sample rate / frequencies become distorted when downsampling without suitable filtering."),
        ("FFT / STFT", "Fast frequency decomposition / repeated decomposition over short time windows."),
        ("Window / hop", "The weighted audio chunk analyzed / how far the next chunk advances."),
        ("Spectrogram / mel / log-mel", "Time-frequency map / hearing-inspired frequency scale / logged mel-band energies."),
        ("Transfer learning / backbone", "Reuse knowledge from another task / the main feature-extraction network."),
        ("Frozen / fine-tuning", "Do not update selected parameters / adapt pretrained weights. Our baseline trains a replacement head."),
        ("Checkpoint / state_dict", "Saved model state / PyTorch's named tensors, including parameters and buffers."),
        ("Config hash / metadata", "A fingerprint of configuration text / the contextual record explaining a run."),
        ("Seed / reproducibility", "Initialize random choices / make a result repeatable with recorded data, code and environment."),
        ("Latency / throughput", "Time per request or sample / work completed per unit time; not the same metric."),
        ("MPS / CUDA / CPU", "Apple GPU backend / NVIDIA GPU backend / general-purpose processor."),
        ("Cache / session state / MVP", "Reuse computed data / UI values surviving reruns / minimum viable product, not a finished production service."),
    ]),
]

for title, entries in GLOSSARY:
    page(title, "Buzzwords translated into plain English. Explain the idea before naming it.", [
        ("glossary", entries),
    ], tag="POCKET DICTIONARY", source="Definitions tailored to this project's implementation; future features are explicitly labelled.")

page("Sources + fact-check trail", "Where to look when someone asks: “Show me in the code.”", [
    ("table", ["Question", "Source of truth"], [
        ["What mutates? Who becomes a parent?", "architect.py: mutate_config, run_architect_search"],
        ["How are scores calculated?", "architect.py: _fitness"],
        ["What is the model and training recipe?", "neuroforge_core.py: DynamicCNN, evaluate_candidate, full_train"],
        ["How are metrics/splits computed?", "neuroforge_core.py: classification_metrics, get_split_indices, get_dataloaders"],
        ["How does audio work?", "audio_pipeline.py: _resample, waveform_to_log_mel, get_split_metadata"],
        ["What model does the UI use?", "frontend/src/App.jsx, frontend/src/components/InteractiveCrucible.jsx, server.py /api/status"],
        ["How are comparison metrics shown?", "server.py /api/comparison; frontend/src/components/MutationLoopPage.jsx"],
        ["What happened in the current run?", "best_model_metadata.json, search_history.json, manual_baseline.json"],
    ]),
    ("links", [
        ("Oxford-IIIT Pet / official data and annotations", "https://www.robots.ox.ac.uk/~vgg/data/pets/"),
        ("ESC-50 / official dataset, folds and licence", "https://github.com/karolpiczak/ESC-50"),
        ("UrbanSound8K / official dataset and protocol", "https://urbansounddataset.weebly.com/urbansound8k.html"),
        ("PyTorch / training and API reference", "https://docs.pytorch.org/docs/stable/index.html"),
        ("Torchvision / datasets, transforms and models", "https://docs.pytorch.org/vision/stable/index.html"),
        ("FastAPI / service documentation", "https://fastapi.tiangolo.com/"),
        ("Vite / frontend build documentation", "https://vite.dev/guide/"),
    ]),
    note("Evidence boundary", "This guide describes the <b>working tree inspected on 4 November 2026</b>, including uncommitted edits. Older results are labelled historical; the guide build does not train a model. A sanitized snapshot and source fingerprints accompany it."),
], tag="TRACEABILITY", source="Dataset pages checked during guide creation. Libraries and behavior were checked from local source and requirements.txt.")

page("One page before you walk in", "Understand the project. Own the limits. Show the evidence.", [
    callout("SAY THIS FIRST", "“NEUROFORGE is an evolutionary CNN architecture-search studio. It generates candidate designs, selects strong parents, mutates their traits, records lineage, then trains and tests the chosen model.”"),
    note("The arrow answer", "<b>Labelled data → preprocessing → CNN population → proxy training + fitness → parent selection → mutation → next generation → best architecture → full training → evaluation + local inference/export.</b>"),
    ("compare", [
        ("REMEMBER", "Outer loop changes architecture.<br/>Inner loop learns weights.<br/>Fitness chooses architecture.<br/>Validation accuracy saves checkpoint.<br/>DynamicCNN is the main model.<br/>ResNet-18 is a separate baseline."),
        ("DO NOT CONFUSE", "Proxy and final performance.<br/>Fruit and crop datasets.<br/>Confidence and correctness.<br/>A metadata key and a fair split.<br/>Synthetic audio and a benchmark.<br/>Exporting and deploying."),
    ]),
     note("Latest fruit snapshot / 4 Nov 2026", "<b>2,960 images • 9 classes • 12 candidate slots • 2 proxy epochs.</b><br/>Selected DynamicCNN proxy: <b>74.19% accuracy / 71.07% precision / 69.95% recall / 67.47% F1 / 0.0417 MSE / 0.2043 RMSE.</b><br/>ResNet-18 recorded final comparison: <b>70.79% accuracy / 70.25% precision / 62.01% recall / 64.01% F1 / 0.0479 MSE / 0.2189 RMSE.</b><br/>The dynamic row is a proxy search result, not a claim of final independent test performance; crop evaluation remains a separate historical case study."),
    callout("END WITH THIS", "“Our strongest feature is traceable architecture evolution. Our next step is stronger evaluation and data quality, not just a bigger number on the screen.”"),
], tag="POCKET CHEAT SHEET", source="Current saved run and inspected implementation. Results can change when the single default artifact bundle is overwritten.")
