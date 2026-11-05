"""Build the NEUROFORGE ultimate system plan as a PDF and Markdown companion.

This is a planning artifact, not a benchmark report. Completed work is labelled
with evidence, while proposed work is kept visibly separate.
"""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PDF_PATH = OUT / "NEUROFORGE-Ultimate-System-Plan.pdf"
MD_PATH = OUT / "NEUROFORGE-Ultimate-System-Plan.md"

INK = colors.HexColor("#131719")
MUTED = colors.HexColor("#626863")
PAPER = colors.HexColor("#F4F0E7")
PAPER_LIGHT = colors.HexColor("#FBF9F4")
NIGHT = colors.HexColor("#101215")
COBALT = colors.HexColor("#2448DA")
CORAL = colors.HexColor("#FF5D42")
ACID = colors.HexColor("#D4ED42")
SAGE = colors.HexColor("#A7C97A")
LINE = colors.HexColor("#D8D5CB")


SECTIONS = [
    ("Executive direction", "What NEUROFORGE becomes", [
        ("p", "NEUROFORGE becomes a local-first model-evolution studio for labelled visual and acoustic data. A user mounts a dataset, inspects its vocabulary, evolves task-specific CNN blueprints, trains the selected design, evaluates it honestly, and exports a reproducible inference bundle."),
        ("callout", "North-star sentence", "Build the model family in public. Mutate the blueprint with evidence. Ship the weights with their context."),
        ("table", ["Outcome", "Meaning"], [
            ["Image classification", "Identify one known class from an image, with per-class metrics and live inference."],
            ["Audio classification", "Identify one known sound class from a WAV clip represented as a log-mel spectrogram."],
            ["Architecture search", "Compare finite CNN designs through parent selection, mutation, proxy training and lineage."],
            ["Evidence layer", "Keep labels, splits, configuration, weights, metrics, timing and limitations together."],
        ]),
    ]),
    ("Progress snapshot", "What exists today and how far the MVP has moved", [
        ("table", ["Area", "Current state", "Evidence / gap"], [
            ["Image MVP", "Implemented and demoable", "React studio, ImageFolder/Roboflow intake, DynamicCNN search, final training path, metrics, exports and live image inference exist."],
            ["Audio core", "Foundation implemented", "audio_pipeline.py handles WAV discovery, ESC-50/UrbanSound8K layouts, folds, log-mel tensors and audio prediction support."],
            ["Audio product flow", "Next implementation slice", "FastAPI prediction routing, React audio intake/playback/spectrogram and a real ESC-50 benchmark still need to be connected and verified."],
            ["Evolution engine", "Implemented", "architect.py records generations, parents, mutations, fitness and cached candidate evaluations."],
            ["Production UI", "Strong local MVP", "React/Vite/Three.js studio is served by FastAPI; motion, topology, mutation loop and export drawers are present."],
            ["Research rigor", "Partially implemented", "Roboflow has explicit splits; plain ImageFolder still needs a true independent test manifest and grouped leakage audit."],
            ["Deployment", "Future", "No registry, authentication, cloud deployment or multi-user experiment store yet."],
        ]),
        ("note", "Planning estimates", "Image classification is roughly 85% of the intended local MVP; audio is roughly 55% of its core path and 25% of its complete product path. These are planning estimates, not accuracy scores."),
    ]),
    ("The complete system architecture", "One platform, two modality adapters, one evolution idea", [
        ("diagram", "React/Vite/Three.js studio\n        ↓ REST + WebSocket\nFastAPI workspace service\n   ├── dataset intake + status + exports\n   ├── image adapter: RGB tensors\n   └── audio adapter: WAV → log-mel tensors\n        ↓\narchitect.py / candidate population + mutation + fitness\n        ↓\nneuroforge_core.py / DynamicCNN + training + metrics\n        ↓\nrun bundle: config + weights + labels + preprocessing + splits + metrics"),
        ("table", ["Boundary", "Contract"], [
            ["Frontend → API", "Dataset metadata, search controls, streamed candidate events, training status and ranked predictions."],
            ["Audio adapter → core", "One-channel feature tensor, ordered class names, fold-aware samples and preprocessing metadata."],
            ["Architect → core", "A JSON architecture configuration; the child receives fresh weights during evaluation."],
            ["Checkpoint → inference", "The model type, config, weights, class order and preprocessing must agree."],
        ]),
        ("callout", "Important boundary", "Image and audio use the same search machinery but remain separate classifiers. A future multimodal model would be a new experiment, not an accidental side effect of adding audio."),
    ]),
    ("Image classification plan", "From pixels to a defensible prediction", [
        ("steps", [
            "Mount ImageFolder or Roboflow data and produce an explicit class vocabulary.",
            "Audit labels, duplicates, class counts, corrupt files and source relationships.",
            "Create train/validation/test manifests; group related originals before splitting.",
            "Resize and normalize RGB images; apply train-only augmentation.",
            "Run a fixed manual CNN and optional ResNet-18 transfer baseline.",
            "Search DynamicCNN configurations with a declared proxy budget.",
            "Retrain the selected architecture from fresh weights using validation for checkpoint choice.",
            "Evaluate the locked model once on an untouched test split and export the bundle.",
        ]),
        ("table", ["Input", "Recommended first production profile"], [
            ["Tensor", "3 × 64 × 64 for the current MVP; compare 128 and 224 only through validation."],
            ["Model", "DynamicCNN search space: 2–5 convolution blocks, filters 16/32/64/128, kernels 3/5, activation, pooling, BatchNorm and dropout."],
            ["Loss", "Class-weighted cross-entropy for imbalance; document weights and class supports."],
            ["Metrics", "Accuracy plus macro precision, recall, F1, confusion matrix, parameter count and latency."],
        ]),
    ]),
    ("Audio classification plan", "Give every sound a measurable shape", [
        ("diagram", "WAV clip\n  ↓ decode + mono mix\n16 kHz resample\n  ↓ center crop / pad to 4 s\nSTFT: 512 FFT, 256 hop, Hann window\n  ↓ power spectrum\n64 mel filters → log energy → normalize\n  ↓\n1 × 64 × ~251 feature tensor\n  ↓\nDynamicCNN → class probabilities"),
        ("table", ["Stage", "Implementation plan"], [
            ["Input contract", "Start with 16-bit PCM WAV; add safe decoder support for common formats only after tests."],
            ["Dataset layouts", "Class folders first; ESC-50 and UrbanSound8K metadata/fold layouts as benchmark adapters."],
            ["Splits", "ESC-50 folds 1–3 train, 4 validation, 5 test for the first smoke benchmark; later aggregate official cross-validation."],
            ["Augmentation", "Feature gain, small noise and time masking initially; validate whether each improves macro F1."],
            ["Inference", "Upload WAV, play it back, show waveform/spectrogram, predicted class, ranked probabilities and latency."],
        ]),
        ("callout", "Audio quality rule", "A synthetic WAV test proves plumbing. Only a real, fold-respecting ESC-50 or UrbanSound8K run can support a sound-recognition quality claim."),
    ]),
    ("The Architect and the two learning loops", "Keep the concept simple enough to defend", [
        ("diagram", "OUTER LOOP / EVOLUTION\nseed → evaluate → rank → select parents → mutate → next generation\n                         ↓\nINNER LOOP / TRAINING\nbatch → forward pass → loss → backprop → optimizer update"),
        ("table", ["Question", "NEUROFORGE answer"], [
            ["What changes outside?", "Layer count, filters, kernels, activation, BatchNorm, pooling and dropout."],
            ["What changes inside?", "The trainable convolution and classifier weights."],
            ["Does a child inherit weights?", "No. It inherits a configuration lineage; candidate weights are trained fresh."],
            ["Why keep an elite?", "To preserve the best known design while exploration continues."],
            ["Is it globally optimal?", "No. It is the best evaluated candidate under a finite space, split, score and budget."],
        ]),
    ]),
    ("Model architecture book", "The blueprint the evaluator should be able to draw", [
        ("table", ["Layer / component", "Image path", "Audio path"], [
            ["Input", "RGB tensor, 3 channels", "Normalized log-mel map, 1 channel"],
            ["Feature blocks", "Conv2D → optional BatchNorm → activation → optional MaxPool/Dropout2d", "Same DynamicCNN block pattern over frequency-time features"],
            ["Compression", "Adaptive average pooling to 1 × 1", "Adaptive average pooling to 1 × 1"],
            ["Classifier", "Linear layer with N image classes", "Linear layer with N sound classes"],
            ["Output", "Logits → softmax probabilities", "Logits → softmax probabilities"],
        ]),
        ("note", "Current selected image blueprint", "The current JSON snapshot records 3 layers with filters [32, 64, 128], kernels [3, 5, 3], BatchNorm on, GELU activation, pooling on and dropout 0.0. The output head expands to the dataset's class count."),
        ("callout", "Future architecture families", "Add depthwise-separable CNNs for edge deployment, residual blocks for deeper image search, temporal convolution or lightweight CRNN options for audio, and an explicit architecture registry so every run can be reloaded without mutable default filenames."),
    ]),
    ("Training, scoring and evaluation", "Numbers must answer a precise question", [
        ("table", ["Phase", "Purpose", "Allowed data"], [
            ["Proxy search", "Cheaply rank candidate architectures", "Training data for updates; validation for metrics and ranking."],
            ["Final training", "Teach the selected architecture for the declared budget", "Training data; validation for checkpoint selection and early stopping."],
            ["Final test", "Measure the locked decision once", "Untouched test data only."],
        ]),
        ("p", "The current fitness combines accuracy, macro precision and macro F1 with small parameter and latency penalties. This is a ranking heuristic, not a probability and not a guarantee of deployment quality."),
        ("callout", "Required report line", "Every metric must name the dataset, model, split, class count, training budget, seed and averaging rule."),
        ("table", ["Reliability upgrade", "Why it matters"], [
            ["Grouped split", "Prevents near-duplicate frames, crops or recordings leaking across partitions."],
            ["Multiple seeds", "Shows whether a result is stable or lucky."],
            ["Per-class report", "Prevents a majority class from hiding weak classes."],
            ["Calibration/OOD set", "Tests whether confidence and in-scope detection can be trusted."],
        ]),
    ]),
    ("Artifact and reproducibility contract", "Ship context with the model", [
        ("diagram", "run_id/\n  ├── config.json\n  ├── weights.pth\n  ├── labels.json\n  ├── preprocessing.json\n  ├── split_manifest.json\n  ├── metrics.json\n  ├── lineage.json\n  └── README.md"),
        ("table", ["Artifact", "Must record"], [
            ["Configuration", "Modality, input shape, layers, filters, kernels, activation, pooling and dropout."],
            ["Preprocessing", "Image size/normalization or sample rate, clip length, FFT, hop, mel count and normalization."],
            ["Dataset identity", "Source, version/hash, classes, counts, skipped files and split membership."],
            ["Training", "Seed, device, epochs, optimizer, learning rate, scheduler, class weights and checkpoint rule."],
            ["Lineage", "Candidate IDs, parents, mutations, scores, cache reuse and selection reason."],
        ]),
    ]),
    ("UI theme and type system", "An instrument, not a generic dashboard", [
        ("callout", "Design direction", "NEUROFORGE is an exhibition-grade machine room: quiet paper surfaces carry the story, graphite instrument bays hold live computation, and cobalt/coral/acid-lime mark state changes."),
        ("table", ["Role", "Choice"], [
            ["Primary paper", "#EDE9DF — warm mineral paper for narrative surfaces."],
            ["Light paper", "#F7F4ED — readable panels and drawers."],
            ["Ink", "#111315 — near-black graphite, never a flat pure black default."],
            ["Muted text", "#656761 — evidence labels and supporting copy."],
            ["Cobalt", "#2448DA — active structure, links and search state."],
            ["Coral", "#FF5D42 — mutation, warning and physical offset shadow."],
            ["Acid lime", "#D4ED42 — success, live energy and confirmation."],
            ["Night bay", "#101215 — topology, spectrogram and model instruments."],
        ]),
        ("table", ["Type role", "Direction"], [
            ["Display", "Space Grotesk-like geometric sans; tight tracking and large editorial scale for decisive headings."],
            ["Narrative accent", "DM Serif Display-like italic serif for a single human, reflective counterpoint."],
            ["Telemetry", "IBM Plex Mono-like monospace for run IDs, labels, metrics, code and state."],
            ["Rules", "Sentence case for user-facing copy; reserved uppercase for telemetry and system labels."],
        ]),
    ]),
    ("Motion and interaction system", "Physics should explain the machine", [
        ("table", ["Motion", "Purpose"], [
            ["Spring cursor field", "Buttons and surfaces follow the pointer with damped return, making the material feel responsive."],
            ["Topology gravity", "Nearby 3D nodes repel the pointer and flex their synapses, turning architecture into a living specimen."],
            ["Drag inertia", "The 3D model keeps momentum after a gesture, then settles rather than stopping abruptly."],
            ["Impact ring", "Pointer-down creates a local shockwave so contact has a visible physical consequence."],
            ["Reduced motion", "Respect prefers-reduced-motion and keep the information legible without animation."],
        ]),
        ("callout", "Restraint", "One memorable motion per moment is stronger than ten unrelated effects. Motion should answer loading, selection, mutation, inference or contact."),
    ]),
    ("Prompt library", "Reusable prompts for the next build cycles", [
        ("table", ["Use", "Prompt"], [
            ["Architecture review", "Act as a senior ML architect. Inspect the current NEUROFORGE config, data shape and training loop. Return the smallest change that improves reliability, its acceptance criteria, and the exact checks to run."],
            ["Audio adapter", "Extend NEUROFORGE's existing audio path without changing the image contract. Support WAV intake, log-mel preprocessing, ordered labels, fold-aware metadata and a user-visible prediction response. Identify every caller and add a smoke check."],
            ["NAS experiment", "Design a matched-budget experiment comparing mutation search, random search and a fixed baseline. Keep data splits, seeds, proxy epochs and reporting fields identical. Do not claim superiority before repeated results."],
            ["UI direction", "Design an exhibition-grade model-evolution interface for NEUROFORGE. Use paper, graphite, cobalt, coral and acid-lime. Avoid generic AI orbs. Make the architecture, lineage and evidence the hero."],
            ["Physics motion", "Add one physically motivated interaction to the existing interface. Use spring damping, pointer response and a reduced-motion fallback. Preserve the current layout and describe the user action that the motion explains."],
            ["Viva practice", "Act as a skeptical evaluator. Ask ten questions about DynamicCNN, audio log-mel features, mutation, metrics, data leakage and confidence. After each answer, ask one follow-up that tests understanding."],
            ["Error analysis", "Create a per-class error report for this run. Include support, precision, recall, F1, confusion pairs, representative failures and a data-quality hypothesis. Separate measured findings from hypotheses."],
            ["Release note", "Write a concise release note from these changed files and checks. Separate implemented behavior, evidence, limitations and next work. Never turn a roadmap item into a completed claim."],
        ]),
        ("note", "Prompt discipline", "Every prompt should state the role, repository context, user outcome, constraints, evidence required, files allowed to change and exact verification command."),
    ]),
    ("Ultimate roadmap", "What NEUROFORGE can become after the local MVP", [
        ("steps", [
            "Finish the production audio intake, prediction endpoint, playback, waveform/log-mel view and audio-specific export bundle.",
            "Add immutable run directories and a model registry; stop overwriting one mutable default bundle.",
            "Repair ImageFolder evaluation with explicit train/validation/test manifests and grouped leakage checks.",
            "Add real ESC-50 and UrbanSound8K benchmark runners with official fold aggregation.",
            "Add per-class error galleries, calibration curves, confusion matrices and OOD evaluation.",
            "Expand search to learning rate, resolution, augmentation, optimizer and scheduler through multi-fidelity promotion.",
            "Add random-search and fixed-baseline ablations to test whether parent-guided mutation helps.",
            "Add edge-friendly architectures, quantization, ONNX/Core ML export and measured device benchmarks.",
            "Add optional multimodal fusion only after independent image/audio models are reliable.",
            "Add authentication, shared experiments, artifact storage and deployment automation for a team product.",
        ]),
    ]),
    ("30 / 60 / 90 day execution plan", "A sequence that keeps science ahead of spectacle", [
        ("table", ["Window", "Deliverables", "Definition of done"], [
            ["Next 30 days", "Audio React/API flow; ESC-10 smoke benchmark; true image manifests; immutable bundle schema.", "A user can mount, train and infer on both modalities with separate saved bundles."],
            ["Days 31–60", "ESC-50 fold runner; repeated seeds; error-analysis panels; random-search baseline; registry UI.", "A benchmark report explains variance, per-class behavior and lineage across runs."],
            ["Days 61–90", "Multi-fidelity search; stronger backbones; export targets; deployment prototype; optional multimodal design.", "A selected model can move from local experiment to a reproducible, portable inference service."],
        ]),
        ("callout", "Priority order", "Evidence first, audio product flow second, search breadth third, deployment fourth. A beautiful surface should reveal a trustworthy system underneath."),
    ]),
    ("Viva and presentation story", "Explain it in five minutes without overclaiming", [
        ("steps", [
            "Problem: manual architecture selection is slow and opaque.",
            "Contribution: an inspectable mutation-driven CNN search workflow with lineage and local inference.",
            "Mechanism: outer evolution changes blueprints; inner training learns weights.",
            "Two modalities: images become RGB tensors; sounds become log-mel tensors; both use DynamicCNN separately.",
            "Evidence: show the dataset, split, budget, candidate mutation, metrics and artifact identity together.",
            "Limit: current ImageFolder testing needs stronger independence; audio needs a real benchmark and complete web path.",
            "Next proof: repeated fair runs against random search and fixed baselines, plus per-class analysis.",
        ]),
        ("callout", "Best short answer", "NEUROFORGE does not promise the globally best model. It makes a finite architecture search visible, measurable and reproducible enough to improve."),
    ]),
    ("Definition of success", "The project is complete when the claims are earned", [
        ("table", ["Success signal", "Target"], [
            ["Image workflow", "Independent test, repeated seeds, class-level reports and portable bundle."],
            ["Audio workflow", "Real benchmark with official folds, audio UI, ranked prediction and exported preprocessing."],
            ["Search validity", "Matched random-search baseline and ablations for mutation, fitness and budget."],
            ["User experience", "A first-time user can understand what happened, why a model won and what to ship."],
            ["Viva readiness", "Every major claim points to a file, a metric, a split or an explicitly labelled plan."],
        ]),
        ("callout", "Final principle", "A higher score is useful. A score whose data, model, split, budget and lineage are visible is useful and defensible."),
    ]),
]


def styles():
    base = getSampleStyleSheet()
    base.add(ParagraphStyle("PlanTitle", parent=base["Title"], fontName="Helvetica-Bold", fontSize=30, leading=34, textColor=INK, alignment=TA_LEFT, spaceAfter=12))
    base.add(ParagraphStyle("PlanSubtitle", parent=base["BodyText"], fontName="Helvetica", fontSize=12, leading=17, textColor=MUTED, spaceAfter=12))
    base.add(ParagraphStyle("PlanH1", parent=base["Heading1"], fontName="Helvetica-Bold", fontSize=21, leading=25, textColor=INK, spaceAfter=7))
    base.add(ParagraphStyle("PlanBody", parent=base["BodyText"], fontName="Helvetica", fontSize=9.4, leading=13.4, textColor=INK, spaceAfter=7))
    base.add(ParagraphStyle("PlanSmall", parent=base["BodyText"], fontName="Helvetica", fontSize=8.1, leading=10.8, textColor=MUTED, spaceAfter=4))
    base.add(ParagraphStyle("PlanCalloutTitle", parent=base["BodyText"], fontName="Helvetica-Bold", fontSize=10.2, leading=12.5, textColor=COBALT, spaceAfter=4))
    base.add(ParagraphStyle("PlanCalloutBody", parent=base["BodyText"], fontName="Helvetica", fontSize=9.2, leading=13.1, textColor=INK))
    base.add(ParagraphStyle("PlanDiagram", parent=base["Code"], fontName="Courier", fontSize=8.2, leading=11, textColor=ACID, backColor=NIGHT, leftIndent=10, rightIndent=10, borderPadding=10, spaceAfter=9))
    base.add(ParagraphStyle("PlanCover", parent=base["Title"], fontName="Helvetica-Bold", fontSize=37, leading=42, textColor=PAPER_LIGHT, spaceAfter=14))
    base.add(ParagraphStyle("PlanCoverSub", parent=base["BodyText"], fontName="Helvetica", fontSize=13, leading=18, textColor=colors.HexColor("#D9E2DE"), spaceAfter=8))
    return base


def para(text, style):
    return Paragraph(str(text).replace("\n", "<br/>"), style)


def callout(title, body, s):
    table = Table([[para(title, s["PlanCalloutTitle"])], [para(body, s["PlanCalloutBody"])]], colWidths=[165 * mm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#E7F0E2")),
        ("BOX", (0, 0), (-1, -1), 0.5, LINE),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    return table


def data_table(headers, rows, s):
    data = [[para(h, s["PlanSmall"]) for h in headers]]
    data += [[para(cell, s["PlanBody"]) for cell in row] for row in rows]
    table = Table(data, colWidths=[165 * mm / len(headers)] * len(headers), repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), COBALT),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [PAPER_LIGHT, colors.HexColor("#E9E9E2")]),
        ("GRID", (0, 0), (-1, -1), 0.35, LINE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return table


def section_flow(section, s):
    title, subtitle, blocks = section
    flow = [Paragraph(title, s["PlanH1"]), Paragraph(subtitle, s["PlanSubtitle"]), HRFlowable(width="100%", thickness=1, color=LINE, spaceAfter=12)]
    for block in blocks:
        kind = block[0]
        if kind == "p":
            flow.append(Paragraph(block[1], s["PlanBody"]))
        elif kind == "note":
            flow += [callout(block[1], block[2], s), Spacer(1, 7)]
        elif kind == "callout":
            flow += [callout(block[1], block[2], s), Spacer(1, 7)]
        elif kind == "table":
            flow += [data_table(block[1], block[2], s), Spacer(1, 8)]
        elif kind == "steps":
            for index, item in enumerate(block[1], 1):
                flow.append(Paragraph(f"<b>{index:02d}.</b> {item}", s["PlanBody"]))
        elif kind == "diagram":
            flow += [Paragraph(block[1].replace("&", "&amp;"), s["PlanDiagram"]), Spacer(1, 4)]
    return flow


def cover_flow(s):
    return [
        Spacer(1, 31 * mm),
        Paragraph("NEUROFORGE", s["PlanCover"]),
        Paragraph("ULTIMATE SYSTEM PLAN", s["PlanCover"]),
        Paragraph("Image classification · audio classification · architecture search · product design · viva readiness", s["PlanCoverSub"]),
        Spacer(1, 15 * mm),
        callout("BUILD. MUTATE. SHIP.", "A complete plan for turning the current local model-evolution studio into a rigorous multimodal platform. Completed work, current evidence, design language, prompts and future prospects are separated clearly.", s),
        Spacer(1, 14 * mm),
        Paragraph("Working-tree planning snapshot / 04 November 2026", s["PlanCoverSub"]),
    ]


def on_page(canvas, doc):
    canvas.saveState()
    width, height = A4
    if doc.page == 1:
        canvas.setFillColor(NIGHT)
        canvas.rect(0, 0, width, height, fill=1, stroke=0)
        canvas.setStrokeColor(colors.HexColor("#24303A"))
        for x in range(0, int(width), 26):
            canvas.line(x, 0, x, height)
        for y in range(0, int(height), 26):
            canvas.line(0, y, width, y)
    else:
        canvas.setStrokeColor(LINE)
        canvas.line(18 * mm, 15 * mm, width - 18 * mm, 15 * mm)
        canvas.setFillColor(MUTED)
        canvas.setFont("Helvetica", 7.5)
        canvas.drawString(18 * mm, 10 * mm, "NEUROFORGE / ultimate system plan")
        canvas.drawRightString(width - 18 * mm, 10 * mm, str(doc.page))
    canvas.restoreState()


def markdown():
    lines = ["# NEUROFORGE — Ultimate System Plan", "", "**Build. Mutate. Ship.** · Working-tree snapshot: 04 November 2026", "", "This plan separates implemented behavior, recorded evidence, proposed work and design direction.", ""]
    for index, (title, subtitle, blocks) in enumerate(SECTIONS, 1):
        lines += [f"## {index:02d}. {title}", "", subtitle, ""]
        for block in blocks:
            kind = block[0]
            if kind in {"p", "note", "callout"}:
                if kind != "p":
                    lines += [f"### {block[1]}", "", block[2], ""]
                else:
                    lines += [block[1], ""]
            elif kind == "diagram":
                lines += ["```text", block[1], "```", ""]
            elif kind == "steps":
                lines += [f"{i}. {value}" for i, value in enumerate(block[1], 1)] + [""]
            elif kind == "table":
                headers, rows = block[1], block[2]
                lines += ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
                lines += ["| " + " | ".join(row) + " |" for row in rows] + [""]
    MD_PATH.write_text("\n".join(lines) + "\n")


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    s = styles()
    doc = BaseDocTemplate(str(PDF_PATH), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=18 * mm, bottomMargin=22 * mm, title="NEUROFORGE — Ultimate System Plan", author="NEUROFORGE project")
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="main")
    doc.addPageTemplates([PageTemplate(id="plan", frames=frame, onPage=on_page)])
    story = cover_flow(s)
    for section in SECTIONS:
        story.append(PageBreak())
        story.extend(section_flow(section, s))
    doc.build(story)
    markdown()
    print(f"Built {PDF_PATH.relative_to(ROOT)}")
    print(f"Built {MD_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    build()
