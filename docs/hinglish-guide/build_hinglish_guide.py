"""Build the NEUROFORGE Hinglish technical guide as PDF and Markdown."""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PDF_PATH = OUT / "NEUROFORGE-Hinglish-Implementation-Guide.pdf"
MD_PATH = OUT / "NEUROFORGE-Hinglish-Implementation-Guide.md"
DOWNLOAD_PATH = Path.home() / "Downloads" / PDF_PATH.name

INK = colors.HexColor("#14232D")
MUTED = colors.HexColor("#5B6970")
TEAL = colors.HexColor("#007D72")
CORAL = colors.HexColor("#F0644B")
MINT = colors.HexColor("#DFF4E8")
PAPER = colors.HexColor("#F7F5EF")
LINE = colors.HexColor("#D4DCD8")


SECTIONS = [
    {
        "title": "NEUROFORGE ka current model setup",
        "subtitle": "Abhi hum exactly kya use kar rahe hain?",
        "blocks": [
            ("callout", "Short answer", "NEUROFORGE mein main model ek PyTorch DynamicCNN hai. Architect iske architecture ko mutate karta hai. ResNet-18 ek separate transfer-learning comparison baseline hai."),
            ("table", ["Model", "Role", "Current status"], [
                ["DynamicCNN", "Mutation search ka selected winner", "74.19% latest saved proxy accuracy; 2 proxy epochs"],
                ["Manual CNN", "Fixed human-designed reference", "50.00% accuracy; 2-epoch proxy"],
                ["ResNet-18", "ImageNet-pretrained comparison baseline", "70.79% accuracy; 20 final epochs"],
            ]),
            ("p", "Decision tree, Random Forest, XGBoost, SVM, KNN ya Logistic Regression abhi project mein use nahi ho rahe. Raw images ke pixels par tree models usually best first choice nahi hote; unko CNN embeddings ke upar classical baseline ki tarah add kiya ja sakta hai."),
            ("callout", "Important", "ResNet mutation engine ka part nahi hai. Woh ek fixed backbone comparison hai. Isse hum poochte hain: evolved small CNN aur standard transfer-learning model ke results kaise compare hote hain?"),
        ],
    },
    {
        "title": "Audio classification ka complete flow",
        "subtitle": "Sound ko tensor mein kaise badalte hain, aur model kaise predict karta hai?",
        "blocks": [
            ("callout", "Short answer", "Audio mein hum waveform ko directly classifier ko nahi dete. Pehle WAV ko mono 16 kHz par laate hain, 4-second clip fit karte hain, phir log-mel spectrogram banate hain. Yeh one-channel image jaisa tensor DynamicCNN ko diya jata hai."),
            ("table", ["Stage", "Simple Hinglish meaning"], [
                ["WAV input", "16-bit PCM WAV clip; pehle audio file read hoti hai."],
                ["Mono + resample", "Multiple channels average hote hain aur sample rate 16,000 Hz par aata hai."],
                ["Crop / pad", "Clip ko 4 seconds ke fixed length mein center-crop ya zero-pad karte hain."],
                ["STFT", "Short windows par frequency analysis karke time ke saath frequency changes milte hain."],
                ["Mel filterbank", "Frequencies ko 64 hearing-inspired mel bands mein group karte hain."],
                ["Log + normalize", "Energy range compress aur per-clip scale normalize hoti hai."],
                ["DynamicCNN", "1 × 64 × roughly 251 tensor se sound class logits predict hote hain."],
            ]),
            ("callout", "Viva line", "Image mein pixels ka pattern hota hai; audio mein time-frequency pattern. Dono ko CNN ke liye tensor banakar alag classifiers train karte hain."),
        ],
    },
    {
        "title": "Audio architecture aur benchmark plan",
        "subtitle": "Architecture same idea follow karti hai, lekin input channel alag hota hai.",
        "blocks": [
            ("table", ["Architecture part", "Image path", "Audio path"], [
                ["Input", "3-channel RGB image", "1-channel log-mel spectrogram"],
                ["Feature blocks", "Conv2D → BatchNorm/activation → optional pooling/dropout", "Same DynamicCNN blocks over frequency-time features"],
                ["Compression", "Adaptive average pooling", "Adaptive average pooling"],
                ["Head", "Linear layer with image classes", "Linear layer with sound classes"],
                ["Output", "Softmax class probabilities", "Softmax class probabilities"],
            ]),
            ("steps", [
                "Pehle ESC-10 subset par plumbing smoke test karo.",
                "ESC-50 metadata aur official folds verify karo.",
                "Folds 1–3 train, fold 4 validation, fold 5 first test check ke liye use karo.",
                "Small mutation search run karke tensor shape, labels aur checkpoint reload verify karo.",
                "Real benchmark ke liye official cross-validation aggregation add karo.",
                "Audio weights, config, ordered labels, preprocessing aur fold metadata separate export karo.",
            ]),
            ("callout", "Current status", "Audio pipeline ka core foundation code mein hai. React audio upload/playback, FastAPI audio prediction route aur real ESC-50 benchmark abhi next work hain. Synthetic WAV check benchmark accuracy prove nahi karta."),
        ],
    },
    {
        "title": "Architect ka workflow",
        "subtitle": "Architect ka kaam: blueprint choose karna, weights learn karna nahi.",
        "blocks": [
            ("steps", [
                "Dataset mount hota hai: local path ya ZIP se labelled images read hote hain.",
                "Classes aur split metadata inspect hota hai.",
                "Manual CNN baseline run hota hai, taaki fixed reference mile.",
                "Generation 1 mein default blueprint aur uske mutated children bante hain.",
                "Har candidate ko short proxy budget par train kiya jata hai.",
                "Accuracy, precision, recall, F1, parameter count aur latency record hote hain.",
                "Fitness score candidates ko rank karta hai.",
                "Top-half parents select hote hain; best design ka elite copy preserve hota hai.",
                "Children parent architecture se mutate hote hain; trained weights inherit nahi hote.",
                "Duplicate architecture cache se repeated evaluation avoid hoti hai.",
                "Final winner ko longer full-training budget par fresh weights ke saath train kiya jata hai.",
                "DynamicCNN, manual CNN aur ResNet-18 ko same active dataset/split context mein compare kiya jata hai.",
            ]),
            ("callout", "Ek line mein", "Outer loop blueprint evolve karta hai. Inner loop us blueprint ke weights learn karta hai."),
        ],
    },
    {
        "title": "CNN layer-by-layer",
        "subtitle": "Convolutional layer kaun hai, aur har layer ka kya kaam hai?",
        "blocks": [
            ("table", ["Component", "Simple Hinglish meaning"], [
                ["Input", "Image ko numbers/tensor mein convert kiya jata hai; current image path 64×64 preprocessing use karta hai."],
                ["Convolutional layer", "Image par chhote windows slide karke local patterns detect karta hai: edges, texture, shape."],
                ["Filter", "Ek learned detector. Har filter kisi pattern ko detect karne ki koshish karta hai."],
                ["Kernel", "Filter ka small spatial window, jaise 3×3 ya 5×5. Kernel size batata hai detector kitna area dekhta hai."],
                ["Activation", "Non-linearity add karta hai. ReLU, GELU aur SiLU current search choices hain."],
                ["BatchNorm", "Intermediate activations ko stable scale par rakhta hai, training ko easier bana sakta hai."],
                ["Pooling", "Feature map ka spatial size reduce karta hai. MaxPool efficiency badhata hai, detail lose kar sakta hai."],
                ["Dropout2d", "Training ke time kuch feature channels temporarily remove karta hai, overfitting reduce karne ke liye."],
                ["Adaptive average pooling", "Har feature channel ko ek summary number mein convert karta hai, image size variations handle karne mein help karta hai."],
                ["Linear classifier", "Final layer har class ke liye ek score/logit deta hai."],
                ["Softmax", "Logits ko relative class probabilities mein convert karta hai; confidence correctness guarantee nahi hai."],
            ]),
        ],
    },
    {
        "title": "Filters, kernels, dropout aur normalization",
        "subtitle": "In words ko viva mein kaise explain karein?",
        "blocks": [
            ("callout", "Filter vs kernel", "Kernel window ka size hai: 3×3 ya 5×5. Filter poora learned detector hai. Ek convolutional layer mein multiple filters hote hain; har filter ek output feature map banata hai."),
            ("callout", "Dropout", "Dropout model ko ek hi shortcut par depend karne se rokne ki koshish karta hai. Yeh sirf training time par active hota hai; inference mein disabled hota hai."),
            ("callout", "Normalization", "Input normalization image values ko consistent numerical range mein laata hai. Batch normalization network ke beech ke activations ko stabilize karta hai. Dono same cheez nahi hain."),
            ("table", ["Example", "Meaning"], [
                ["filters [32, 64]", "Pehli convolutional layer ke 32 detectors, doosri ke 64 detectors."],
                ["kernel_sizes [3, 5]", "Pehli layer 3×3 window aur doosri 5×5 window use karegi."],
                ["dropout 0.4", "Training mein feature channels ka roughly 40% dropout probability."],
                ["batch_norm true", "Convolution ke baad BatchNorm enabled."],
                ["pooling false", "Architecture mein MaxPool branches disabled."],
            ]),
        ],
    },
    {
        "title": "Classification vs regression",
        "subtitle": "Hum kya kar rahe hain, aur MSE/RMSE ka actual meaning kya hai?",
        "blocks": [
            ("table", ["Topic", "Classification", "Regression"], [
                ["Output", "Known classes mein se ek label, jaise Apple ya Banana", "Ek continuous number, jaise price ya temperature"],
                ["Current NEUROFORGE use", "Yes: multiclass image classification", "No: current production task regression nahi hai"],
                ["Typical loss", "Cross-entropy", "MSE or Huber loss"],
                ["Typical metrics", "Accuracy, precision, recall, F1", "MSE, RMSE, MAE, R²"],
            ]),
            ("p", "Current comparison table mein MSE/RMSE classification probability vectors par calculate hote hain: predicted probability vector ko one-hot true-label vector se compare kiya jata hai. Yeh auxiliary diagnostics hain, regression proof nahi. MSE/RMSE lower hona better hai."),
            ("callout", "Viva answer", "Hum classification solve kar rahe hain. Accuracy aur macro F1 main metrics hain. MSE/RMSE ko probability calibration/error diagnostics ke roop mein show karte hain, regression metrics ke roop mein nahi."),
        ],
    },
    {
        "title": "Current results aur >90% roadmap",
        "subtitle": "Target possible ho sakta hai, guarantee nahi.",
        "blocks": [
            ("table", ["Model", "Acc", "Precision", "Recall", "F1", "MSE", "RMSE"], [
                ["DynamicCNN / 2 proxy", "74.19%", "71.07%", "69.95%", "67.47%", "0.0417", "0.2043"],
                ["Manual CNN / 2 proxy", "50.00%", "52.61%", "46.10%", "43.22%", "0.0797", "0.2824"],
                ["ResNet-18 / 20 final", "70.79%", "70.25%", "62.01%", "64.01%", "0.0479", "0.2189"],
            ]),
            ("steps", [
                "64×64 se 128×128 ya 224×224 resolution compare karo; small visual details preserve honge.",
                "ResNet-18 ka final block unfreeze karke low learning rate par fine-tune karo.",
                "EfficientNet, ResNet-50 ya ConvNeXt jaise stronger pretrained baselines add karo.",
                "Architect ko learning rate, weight decay, resolution aur augmentation bhi mutate/tune karne do.",
                "Labels, duplicates, imbalance aur source leakage audit karo.",
                "At least 3 seeds par repeat karo; same split, budget aur hardware report karo.",
                "Ensemble tabhi use karo jab validation aur untouched test par improvement prove ho.",
            ]),
            ("callout", "Important math", "Latest DynamicCNN row 2-epoch proxy result hai, final independent test nahi. 89-image split par har ek prediction percentage ko noticeably move kar sakti hai. 90% ko target samjho, guarantee nahi; pehle independent split aur repeated seeds complete karo."),
        ],
    },
    {
        "title": "Kya Decision Tree ya Random Forest add karna chahiye?",
        "subtitle": "Classical ML ka sensible comparison kaise banega?",
        "blocks": [
            ("p", "Raw image pixels directly tree models ko dena usually weak baseline hoga: dimensionality bahut high hoti hai aur spatial structure lose ho jata hai."),
            ("steps", [
                "Pretrained CNN se har image ka embedding vector extract karo.",
                "Embedding ko train/validation/test split ke saath store karo.",
                "Random Forest, SVM, Logistic Regression aur XGBoost ko same embedding par train karo.",
                "Same class labels, same test split aur same metrics use karo.",
                "Result ko DynamicCNN/ResNet se alag classical baseline ke roop mein label karo.",
            ]),
            ("callout", "Current status", "Decision Tree, Random Forest aur SVM abhi implement nahi hain. Inko add karna possible hai, lekin pehle current CNN/ResNet comparison ko stable aur fair rakhna better hai."),
        ],
    },
    {
        "title": "Implementation timeline",
        "subtitle": "11 baje phone kholkar jo idea aaya tha, uska status.",
        "blocks": [
            ("callout", "Clarification note", "“11 baje phone kholkar aaya hua idea” ka exact feature name repository notes mein recorded nahi hai. Isliye us feature ke baare mein fake technical claim nahi karenge."),
            ("table", ["Phase", "Kab / priority", "What happens"], [
                ["Current release", "Abhi complete", "Metric comparison, ResNet-18 baseline, MSE/RMSE, checkpoint downloads and stale-dataset guard."],
                ["Next experiment", "Next implementation sprint", "224px resolution, ResNet fine-tuning, learning-rate/augmentation tuning and repeated seeds."],
                ["Phone-open idea", "Next sprint after exact feature confirmation", "One-line acceptance criteria define karke prototype, test and demo integration."],
                ["Classical baselines", "After CNN comparison is stable", "CNN embeddings par Random Forest/SVM/Logistic Regression comparison."],
            ]),
            ("callout", "Short answer", "Pehle current comparison ko fair baseline banayenge. Uske baad >90% improvement experiments. Phone-open idea ka exact naam/expected behavior confirm hote hi usko next sprint mein implement karenge."),
        ],
    },
    {
        "title": "Viva cheat sheet",
        "subtitle": "Short answers jo confidently bol sakte ho.",
        "blocks": [
            ("table", ["Question", "Answer"], [
                ["What is NEUROFORGE?", "Mutation-based CNN architecture-search studio with visible lineage, training evidence and local inference."],
                ["Architect kya karta hai?", "Candidate CNN blueprints generate, mutate, rank and select karta hai."],
                ["Weights inherit hote hain?", "Nahi. Child architecture inherit karta hai; weights fresh train hote hain."],
                ["ResNet-18 kyun?", "Fixed pretrained transfer-learning baseline, mutation winner prove karne ke liye nahi."],
                ["Random Forest use ho raha hai?", "Nahi. Current system DynamicCNN, manual CNN aur ResNet-18 use karta hai."],
                ["Classification ya regression?", "Multiclass image classification. MSE/RMSE probability-vector diagnostics hain."],
                [">90% kaise?", "Better data, higher resolution, fine-tuning, stronger baselines, hyperparameter search and repeated fair evaluation."],
            ]),
            ("callout", "Remember", "Accuracy ka number akela enough nahi. Dataset, model, split, training budget aur all metrics saath mein bolne hain."),
        ],
    },
]


def build_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("GuideTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=29, leading=34, textColor=INK, alignment=TA_LEFT, spaceAfter=12))
    styles.add(ParagraphStyle("GuideSubtitle", parent=styles["Normal"], fontName="Helvetica", fontSize=12, leading=17, textColor=MUTED, spaceAfter=14))
    styles.add(ParagraphStyle("GuideSection", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=20, leading=24, textColor=INK, spaceAfter=7))
    styles.add(ParagraphStyle("GuideBody", parent=styles["BodyText"], fontName="Helvetica", fontSize=9.7, leading=14, textColor=INK, spaceAfter=8))
    styles.add(ParagraphStyle("GuideSmall", parent=styles["BodyText"], fontName="Helvetica", fontSize=8.2, leading=11, textColor=MUTED, spaceAfter=5))
    styles.add(ParagraphStyle("GuideCalloutTitle", parent=styles["BodyText"], fontName="Helvetica-Bold", fontSize=10.5, leading=13, textColor=TEAL, spaceAfter=4))
    styles.add(ParagraphStyle("GuideCalloutBody", parent=styles["BodyText"], fontName="Helvetica", fontSize=9.3, leading=13.5, textColor=INK))
    styles.add(ParagraphStyle("GuideCover", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=36, leading=40, textColor=INK, alignment=TA_CENTER, spaceAfter=16))
    styles.add(ParagraphStyle("GuideCoverSub", parent=styles["BodyText"], fontName="Helvetica", fontSize=13, leading=19, textColor=MUTED, alignment=TA_CENTER))
    return styles


def para(text, style):
    return Paragraph(str(text).replace("\n", "<br/>") , style)


def callout(title, body, styles):
    table = Table([[para(title, styles["GuideCalloutTitle"])], [para(body, styles["GuideCalloutBody"])]], colWidths=[165 * mm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), MINT),
        ("BOX", (0, 0), (-1, -1), 0.5, LINE),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    return table


def table(headers, rows, styles):
    data = [[para(header, styles["GuideSmall"]) for header in headers]]
    data += [[para(cell, styles["GuideBody"]) for cell in row] for row in rows]
    widths = [165 * mm / len(headers)] * len(headers)
    rendered = Table(data, colWidths=widths, repeatRows=1)
    rendered.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), TEAL),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#EDF0EC")]),
        ("GRID", (0, 0), (-1, -1), 0.35, LINE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return rendered


def section_flow(section, styles):
    flow = [Paragraph(section["title"], styles["GuideSection"]), Paragraph(section["subtitle"], styles["GuideSubtitle"])]
    for block in section["blocks"]:
        kind = block[0]
        if kind == "p":
            flow.append(Paragraph(block[1], styles["GuideBody"]))
        elif kind == "callout":
            flow += [callout(block[1], block[2], styles), Spacer(1, 8)]
        elif kind == "table":
            flow += [table(block[1], block[2], styles), Spacer(1, 8)]
        elif kind == "steps":
            for index, item in enumerate(block[1], 1):
                flow.append(Paragraph(f"<b>{index:02d}.</b> {item}", styles["GuideBody"]))
        else:
            raise ValueError(f"Unknown block type: {kind}")
    return flow


def markdown_block(block):
    kind = block[0]
    if kind == "p":
        return [block[1], ""]
    if kind == "callout":
        return [f"### {block[1]}", "", block[2], ""]
    if kind == "table":
        headers, rows = block[1], block[2]
        lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
        lines += ["| " + " | ".join(row) + " |" for row in rows]
        return lines + [""]
    if kind == "steps":
        return [f"{index}. {item}" for index, item in enumerate(block[1], 1)] + [""]
    return []


def write_markdown():
    lines = [
        "# NEUROFORGE — The Builders' Playbook",
        "",
        "**Build. Mutate. Ship.** · 4 November 2026",
        "",
        "This guide explains the current models, Architect workflow, CNN concepts, metrics and implementation roadmap in Hinglish.",
        "",
    ]
    for index, section in enumerate(SECTIONS, 1):
        lines += [f"## {index:02d}. {section['title']}", "", section["subtitle"], ""]
        for block in section["blocks"]:
            lines += markdown_block(block)
    MD_PATH.write_text("\n".join(lines) + "\n")


def on_page(canvas, doc):
    canvas.saveState()
    width, height = A4
    canvas.setStrokeColor(LINE)
    canvas.line(18 * mm, 15 * mm, width - 18 * mm, 15 * mm)
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 7.5)
    canvas.drawString(18 * mm, 10 * mm, "NEUROFORGE / field playbook")
    canvas.drawRightString(width - 18 * mm, 10 * mm, f"{doc.page}")
    canvas.restoreState()


def build_pdf():
    styles = build_styles()
    doc = BaseDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=22 * mm,
        title="NEUROFORGE — The Builders' Playbook",
        author="NEUROFORGE project",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="main")
    doc.addPageTemplates([PageTemplate(id="guide", frames=frame, onPage=on_page)])

    story = [
        Spacer(1, 35 * mm),
        Paragraph("NEUROFORGE", styles["GuideCover"]),
        Paragraph("THE BUILDERS' PLAYBOOK", styles["GuideCover"]),
        Paragraph("Architecture, image + audio classification, metrics, classical ML and the >90% roadmap", styles["GuideCoverSub"]),
        Spacer(1, 15 * mm),
        callout("Current snapshot", "Fruit dataset: 2,960 images, 9 classes. Latest saved DynamicCNN search result: 74.19% proxy accuracy. ResNet-18 comparison: 70.79%. MSE/RMSE are probability-vector diagnostics.", styles),
        Spacer(1, 12 * mm),
        Paragraph("Practical notes for viva, teammate handoff and implementation planning.", styles["GuideCoverSub"]),
    ]
    for section in SECTIONS:
        story.append(PageBreak())
        story.extend(section_flow(section, styles))
    doc.build(story)
    write_markdown()
    DOWNLOAD_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOWNLOAD_PATH.write_bytes(PDF_PATH.read_bytes())
    print(f"Built {PDF_PATH}")
    print(f"Copied {DOWNLOAD_PATH}")


if __name__ == "__main__":
    build_pdf()
