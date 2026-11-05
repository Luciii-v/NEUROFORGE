"""Build the illustrated PDF and Markdown companion without touching app artifacts.

Build dependencies: reportlab, pymupdf (verification only).
Run from anywhere: python docs/learning-guide/build.py
"""

import hashlib
import html
import json
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle

from content import PAGES

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PDF = OUT / "NEUROFORGE-Mutation-Playbook.pdf"
GUIDE_DATE_SHORT = "04 NOV 2026"
GUIDE_DATE_LONG = "04 NOVEMBER 2026"
GUIDE_DATE_ISO = "2026-11-04"
W, H = 595.276, 841.89
M = 43
CW = W - 2 * M
BOTTOM = 72
INK = "#162734"
MUTED = "#536570"
PAPER = "#F6F5F0"
TEAL = "#007D72"
MINT = "#A7F0CD"
VIOLET = "#6751C9"
CORAL = "#CA4E36"
NAVY = "#10232F"
DARK_PAGES = {5, 8, 20, 40}
STATS = []


def color(value):
    return colors.HexColor(value)


def register_fonts():
    font_dir = Path("/System/Library/Fonts/Supplemental")
    import reportlab
    fallback = Path(reportlab.__file__).parent / "fonts"
    for name, local, bundled in [
        ("Body", "Arial.ttf", "Vera.ttf"),
        ("BodyBold", "Arial Bold.ttf", "VeraBd.ttf"),
        ("BodyItalic", "Arial Italic.ttf", "VeraIt.ttf"),
        ("Display", "Trebuchet MS Bold.ttf", "VeraBd.ttf"),
        ("Mono", "Courier New.ttf", "Vera.ttf"),
    ]:
        path = font_dir / local
        if not path.exists():
            path = fallback / bundled
        pdfmetrics.registerFont(TTFont(name, str(path)))
    pdfmetrics.registerFontFamily("Body", normal="Body", bold="BodyBold", italic="BodyItalic", boldItalic="BodyBold")


def markup(text):
    """Keep only the small intentional markup vocabulary used by content.py."""
    escaped = html.escape(str(text), quote=False)
    for tag in ["b", "/b", "i", "/i", "br/", "br", "/font"]:
        escaped = escaped.replace(html.escape(f"<{tag}>"), f"<{tag}>")
    escaped = escaped.replace("&lt;font name='Mono'&gt;", "<font name='Mono'>")
    return escaped


def plain(text):
    return html.unescape(re.sub(r"<[^>]+>", "", str(text).replace("<br/>", "\n")))


def capture_snapshot():
    """Allowlist public facts; never copy raw metadata paths into the guide."""
    target = OUT / "snapshot.json"
    meta = json.loads((ROOT / "best_model_metadata.json").read_text())
    history = json.loads((ROOT / "search_history.json").read_text())
    baseline = json.loads((ROOT / "manual_baseline.json").read_text())
    resnet_path = ROOT / "resnet18_comparison_metadata.json"
    resnet_meta = json.loads(resnet_path.read_text()) if resnet_path.exists() else {}
    selected = next(row for row in history if row["candidate_id"] == meta["best_candidate_id"])
    final = (meta.get("held_out_test_metrics") or meta.get("best_test_metrics") or meta.get("best_metrics") or {})
    fields = ["accuracy", "precision", "recall", "f1", "mse", "rmse", "parameters", "score"]
    snapshot = {
        "as_of": GUIDE_DATE_ISO,
        "run_id": meta["run_id"],
        "training_mode": meta["training_mode"],
        "training_complete": meta.get("training_complete"),
        "epochs": meta.get("epochs"),
        "generations": meta.get("generations"),
        "population_size": meta.get("population_size"),
        "proxy_epochs": meta.get("proxy_epochs"),
        "dataset": {k: meta["dataset"].get(k) for k in ["format", "modality", "num_classes", "num_samples", "class_names"]},
        "selected_config": json.loads((ROOT / "best_model_config.json").read_text()),
        "selected_candidate": selected["candidate_id"],
        "selected_parent": selected["parent_id"],
        "selected_mutation": selected["mutation_operations"],
        "proxy": {k: selected.get(k) for k in fields},
        "baseline_proxy": {k: baseline.get(k) for k in fields},
        "final_evaluation": {k: final.get(k) for k in fields + ["confusion_matrix", "sample_count"]},
        "resnet_evaluation": {k: resnet_meta.get("held_out_test_metrics", {}).get(k) for k in fields + ["confusion_matrix", "sample_count"]},
        "resnet_training": {k: resnet_meta.get(k) for k in ["architecture", "epochs", "training_mode", "comparison_role"]},
        "evaluation_caveat": "ImageFolder reuses the same evaluation subset for selection and final reporting; not an independent test.",
        "candidate_slots": len(history),
        "unique_candidate_configs": len({x["config_hash"] for x in history}),
        "unique_evaluations_including_baseline": meta.get("unique_evaluations"),
        "reused_evaluations": meta.get("reused_evaluations"),
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in [
            "architect.py", "neuroforge_core.py", "audio_pipeline.py", "server.py", "run_studio.sh",
            "frontend/package.json", "frontend/src/App.jsx", "frontend/src/components/MutationLoopPage.jsx",
            "best_model_config.json", "best_model_metadata.json", "search_history.json", "manual_baseline.json",
            "resnet18_comparison_metadata.json",
        ] if (ROOT / name).exists()},
    }
    target.write_text(json.dumps(snapshot, indent=2) + "\n")
    return snapshot


class Layout:
    def __init__(self, cv, page_num, dark=False, scale=1.0, dry=False):
        self.c = cv
        self.page_num = page_num
        self.dark = dark
        self.scale = scale
        self.dry = dry
        self.fg = "#F0F4EF" if dark else INK
        self.muted = "#B8CCCE" if dark else MUTED
        self.accent = MINT if dark else TEAL

    def rect(self, x, top, w, h, fill, stroke=None, radius=0):
        if self.dry:
            return
        self.c.setFillColor(color(fill))
        self.c.setStrokeColor(color(stroke or fill))
        if radius:
            self.c.roundRect(x, H - top - h, w, h, radius, fill=1, stroke=int(bool(stroke)))
        else:
            self.c.rect(x, H - top - h, w, h, fill=1, stroke=int(bool(stroke)))

    def line(self, x1, y1, x2, y2, stroke, width=1):
        if not self.dry:
            self.c.setStrokeColor(color(stroke))
            self.c.setLineWidth(width)
            self.c.line(x1, H - y1, x2, H - y2)

    def text(self, text, x, top, w=CW, size=10.7, fg=None, font="Body", leading=None):
        size *= self.scale
        style = ParagraphStyle("guide", fontName=font, fontSize=size, leading=(leading or size * 1.36),
                               textColor=color(fg or self.fg), alignment=TA_LEFT,
                               spaceBefore=0, spaceAfter=0, allowWidows=0, allowOrphans=0)
        p = Paragraph(markup(text).replace("\n", "<br/>"), style)
        _, height = p.wrap(w, H)
        if not self.dry:
            p.drawOn(self.c, x, H - top - height)
        return height

    def arrow(self, x1, y1, x2, y2, stroke=None):
        import math
        stroke = stroke or self.accent
        self.line(x1, y1, x2, y2, stroke, 1.6)
        angle = math.atan2(y2 - y1, x2 - x1)
        for offset in [-0.5, 0.5]:
            self.line(x2, y2, x2 - 6 * math.cos(angle + offset), y2 - 6 * math.sin(angle + offset), stroke, 1.6)

    def header(self, spec):
        self.rect(0, 0, W, H, NAVY if self.dark else PAPER)
        self.rect(M, 30, 30, 4, self.accent)
        self.text("NEUROFORGE / FIELD GUIDE", M + 42, 25, 300, 8, self.muted, "BodyBold")
        self.text(GUIDE_DATE_SHORT, W - 121, 25, 90, 8, self.muted)
        self.text(spec["tag"], M, 62, CW, 9, self.accent, "BodyBold")
        th = self.text(spec["title"], M, 86, CW, 29, self.fg, "Display", 32)
        sh = self.text(spec["subtitle"], M, 93 + th, CW, 11, self.muted)
        return 111 + th + sh

    def footer(self, spec):
        source = spec.get("source", "")
        self.line(M, H - 58, W - M, H - 58, "#37505B" if self.dark else "#D2D9D5")
        self.text(source, M, H - 51, CW - 42, 7.2, self.muted)
        self.text(f"{self.page_num:02d} / {len(PAGES):02d}", W - 82, H - 35, 55, 8, self.accent, "BodyBold")

    def box(self, title, body, top, kind="note", x=M, width=CW):
        pad = 15
        title_size = 11.4 if kind == "qa" else 10
        # Measurement and drawing use exactly the same wrapping and line heights.
        measure = Layout(self.c, self.page_num, self.dark, self.scale, dry=True)
        ht = measure.text(title, x + pad, top + pad, width - pad * 2, title_size, font="BodyBold")
        hb = measure.text(body, x + pad, top + pad + ht + 6, width - pad * 2)
        height = pad * 2 + ht + 6 + hb
        if kind == "callout":
            bg = "#214538" if self.dark else "#E1F0E7"
            head_color = self.accent
        elif kind == "qa":
            bg = "#1A303D" if self.dark else "#FFFFFF"
            head_color = "#D6C9FF" if self.dark else VIOLET
        else:
            bg = "#1B3340" if self.dark else "#EBEEE8"
            head_color = self.accent
        self.rect(x, top, width, height, bg, radius=8)
        self.text(title, x + pad, top + pad, width - pad * 2, title_size, head_color, "BodyBold")
        self.text(body, x + pad, top + pad + ht + 6, width - pad * 2)
        return height + 11

    def table(self, headers, rows, top):
        cols = len(headers)
        widths = [CW * .31, CW * .69] if cols == 2 else [CW * .29, CW * .31, CW * .40]
        style = ParagraphStyle("cell", fontName="Body", fontSize=10 * self.scale,
                               leading=13.2 * self.scale, textColor=color(self.fg))
        header_style = ParagraphStyle("head", parent=style, fontName="BodyBold", textColor=color("#FFFFFF"))
        data = [[Paragraph(markup(v), header_style) for v in headers]]
        data += [[Paragraph(markup(v), style) for v in row] for row in rows]
        table = Table(data, colWidths=widths, hAlign="LEFT")
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), color("#304856" if self.dark else TEAL)),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [color("#1B3340" if self.dark else "#FFFFFF"), color("#233C48" if self.dark else "#EBEEE8")]),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 11),
            ("RIGHTPADDING", (0, 0), (-1, -1), 11),
            ("TOPPADDING", (0, 0), (-1, -1), 9 * self.scale),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 9 * self.scale),
        ]))
        _, height = table.wrap(CW, H)
        if not self.dry:
            table.drawOn(self.c, M, H - top - height)
        return height + 13

    def blocks(self, blocks, top):
        y = top
        for block in blocks:
            kind = block[0]
            if kind in {"qa", "note", "callout"}:
                y += self.box(block[1], block[2], y, kind)
            elif kind == "table":
                y += self.table(block[1], block[2], y)
            elif kind == "compare":
                width = (CW - 13) / 2
                heights = [self.box(t, b, y, "note", M + i * (width + 13), width)
                           for i, (t, b) in enumerate(block[1])]
                y += max(heights)
            elif kind == "formula":
                y += self.box(block[1], block[2], y, "callout")
            elif kind == "steps":
                for i, (title, body) in enumerate(block[1]):
                    self.rect(M, y + 2, 23, 23, self.accent, radius=5)
                    self.text(str(i + 1), M + 7, y + 5, 16, 10, NAVY if self.dark else "#FFFFFF", "BodyBold")
                    th = self.text(title, M + 36, y, CW - 36, 10.5, self.accent, "BodyBold")
                    bh = self.text(body, M + 36, y + th + 4, CW - 36)
                    y += th + bh + 17
            elif kind == "glossary":
                for term, definition in block[1]:
                    th = self.text(term, M, y, CW, 11.3, self.accent, "BodyBold")
                    bh = self.text(definition, M, y + th + 3, CW, 10.7)
                    y += th + bh + 12
            elif kind == "flow":
                for i, (title, body) in enumerate(block[1]):
                    compact = Layout(self.c, self.page_num, self.dark, self.scale, dry=True)
                    th = compact.text(title, M + 28, y, CW - 28, 9.3, self.accent, "BodyBold")
                    bh = compact.text(body, M + 28, y + th + 2, CW - 28, 9.4, self.muted)
                    h = th + bh + 9
                    self.rect(M + 28, y, CW - 28, h, "#1B3340" if self.dark else "#FFFFFF", radius=5)
                    self.text(title, M + 40, y + 7, CW - 52, 9.3, self.accent, "BodyBold")
                    self.text(body, M + 40, y + 7 + th + 2, CW - 52, 9.4, self.muted)
                    self.rect(M, y + 4, 16, 16, self.accent, radius=3)
                    self.text(str(i + 1), M + 4, y + 6, 10, 7.5, NAVY if self.dark else "#FFFFFF", "BodyBold")
                    if i < len(block[1]) - 1:
                        self.line(M + 8, y + h + 1, M + 8, y + h + 8, self.accent, 1)
                    y += h + 5
            elif kind == "links":
                for label, url in block[1]:
                    lh = self.text(label, M, y, CW, 10, self.accent, "BodyBold")
                    uh = self.text(url, M, y + lh + 2, CW, 9.1, self.muted)
                    if not self.dry:
                        self.c.linkURL(url, (M, H - y - lh - uh - 2, W - M, H - y), relative=0)
                    y += lh + uh + 10
            elif kind == "routes":
                y += self.routes(y)
            elif kind == "confusion":
                y += self.confusion(y)
            elif kind == "snapshot":
                y += self.snapshot(y)
            elif kind == "audio":
                y += self.audio(y)
            else:
                raise ValueError(f"Unknown block: {kind}")
        return y

    def routes(self, top):
        routes = [
            ("01 / Get the idea", "Workflow, models, mutation", 3, 7),
            ("02 / Explain the numbers", "Score, metrics, training, splits", 8, 13),
            ("03 / Improve the results", "Evidence, crop quality, runtime", 14, 17),
            ("04 / Change the inputs", "Dataset steps and audio", 18, 21),
            ("05 / Know the machine", "Failures, files, project status", 22, 25),
            ("06 / Teach and defend it", "Viva, demo, quiz and feedback", 26, 33),
            ("07 / Decode the buzzwords", "Five pocket-dictionary pages", 34, 38),
            ("08 / Last-minute mode", "Sources and one-page recap", 39, 40),
        ]
        width = (CW - 12) / 2
        row_h = 66 * self.scale
        for i, (label, desc, first, last) in enumerate(routes):
            x = M + (i % 2) * (width + 12)
            y = top + (i // 2) * row_h
            self.rect(x, y, width, row_h - 7, "#FFFFFF", radius=7)
            self.text(label, x + 12, y + 9, width - 24, 10.5, TEAL, "BodyBold")
            self.text(f"{desc}<br/><b>Pages {first:02d}–{last:02d} →</b>", x + 12, y + 25 * self.scale, width - 24, 9.1, MUTED)
            if not self.dry:
                self.c.linkRect("Jump to chapter", f"p{first}", (x, H-y-row_h+7, x+width, H-y), relative=0, thickness=0)
        return row_h * 4 + 6

    def confusion(self, top):
        self.rect(M, top, CW, 145, "#E3EBE9", radius=8)
        self.text("TRUTH ↓ / PREDICTION →", M + 18, top + 16, 215, 9, TEAL, "BodyBold")
        x0, y0 = M + 230, top + 31
        for j, label in enumerate(["Predicted A", "Predicted B"]):
            self.text(label, x0 + j * 111, top + 10, 108, 9, MUTED)
        for i, label in enumerate(["Actual A", "Actual B"]):
            self.text(label, M + 95, y0 + i * 48 + 13, 110, 10.7, INK, "BodyBold")
            for j, value in enumerate([[40, 10], [20, 30]][i]):
                self.rect(x0 + j * 111, y0 + i * 48, 104, 42, TEAL if i == j else "#D7DDE0", radius=4)
                self.text(str(value), x0 + j * 111 + 35, y0 + i * 48 + 6, 66, 21,
                          "#FFFFFF" if i == j else INK, "Display")
        return 158

    def snapshot(self, top):
        for i, (value, title) in enumerate([("2,960", "FRUIT IMAGES"), ("9", "FRUIT CLASSES"), ("20", "FINAL EPOCHS")]):
            width = (CW - 20) / 3
            x = M + i * (width + 10)
            self.rect(x, top, width, 72, TEAL if i == 0 else "#E3EBE9", radius=8)
            self.text(value, x + 14, top + 7, width - 28, 30, "#FFFFFF" if i == 0 else INK, "Display")
            self.text(title, x + 14, top + 50, width - 24, 8, "#FFFFFF" if i == 0 else MUTED, "BodyBold")
        return 84

    def audio(self, top):
        import math
        self.rect(M, top, CW, 110, "#1B3340", radius=8)
        self.text("WAVEFORM", M + 12, top + 12, 130, 8, MINT, "BodyBold")
        for i in range(115):
            x = M + 13 + i * 1.35
            amp = math.sin(i * .5) * math.sin(i * .074) * 23
            self.line(x, top + 64 - amp, x, top + 64 + amp, MINT, .8)
        self.arrow(M + 178, top + 62, M + 213, top + 62, MINT)
        self.text("LOG-MEL MAP", M + 231, top + 12, 140, 8, MINT, "BodyBold")
        for row in range(10):
            for col in range(29):
                value = (math.sin(col * .39 + row * .64) + 1) / 2
                shade = colors.Color(.18 + .36 * value, .26 + .50 * value, .39 + .19 * value)
                if not self.dry:
                    self.c.setFillColor(shade)
                    self.c.rect(M + 231 + col * 8, H - (top + 34 + row * 5.5) - 5, 7, 5, fill=1, stroke=0)
        self.text("Illustration only · no real audio shown", M + 231, top + 92, 240, 7.5, "#B8CCCE")
        return 122


def cover(cv):
    l = Layout(cv, 1, True)
    l.rect(0, 0, W, H, NAVY)
    for x in range(0, int(W), 26):
        l.line(x, 0, x, H, "#18323F", .35)
    for y in range(0, int(H), 26):
        l.line(0, y, W, y, "#18323F", .35)
    l.rect(M, 43, 141, 25, MINT, radius=4)
    l.text("FIELD NOTES / 2026", M + 10, 49, 135, 8, NAVY, "BodyBold")
    l.text("NEUROFORGE", M, 96, CW, 39, "#F0F4EF", "Display")
    l.text("THE MUTATION\nPLAYBOOK", M - 1, 161, CW + 6, 48, MINT, "Display", 53)
    l.text("Build. Mutate. Ship.", M, 290, CW, 20, "#D2C2FF", "Display")
    l.text("Your questions. Clear answers.\nThe real code. The honest numbers.", M, 337, CW, 15, "#DDE8E5")
    # A blueprint family tree—not biological imagery.
    y = 442
    for i, label in enumerate(["PARENT", "MUTATION", "CHILD"]):
        x = M + i * 174
        l.rect(x, y, 157, 100, "#1B3340", "#42616B", 8)
        l.text(label, x + 13, y + 12, 140, 9, MINT, "BodyBold")
        text = ["CNN blueprint\nfilters [32, 64]", "filters[1]\n64 → 128", "New blueprint\nfilters [32, 128]"][i]
        l.text(text, x + 13, y + 40, 137, 12, "#F0F4EF")
        if i < 2:
            l.arrow(x + 158, y + 50, x + 171, y + 50, MINT)
    l.text("Illustrative architecture edit", M, y + 109, CW, 8, "#AAC2C8")
    l.text("FROM “WHAT IS A CNN?”\nTO “LET ME EXPLAIN OUR SYSTEM.”", M, 618, CW, 19, "#F0F4EF", "Display", 26)
    l.line(M, 719, W - M, 719, "#42616B")
    l.text("40 visual pages • worked examples • viva practice\n70 glossary entries • project-specific failure stories", M, 739, CW, 10, "#B8CCCE")
    l.text(f"WORKING-TREE SNAPSHOT / {GUIDE_DATE_LONG}", M, 799, CW, 8, MINT, "BodyBold")


def markdown_companion():
    lines = ["# NEUROFORGE — The Mutation Playbook", "", f"**Build. Mutate. Ship.** · Working-tree snapshot: {GUIDE_DATE_LONG.title()}", "",
             "Editable plain-text companion to the illustrated PDF. Page references match the PDF.", ""]
    for number, spec in enumerate(PAGES, 1):
        lines += [f"## {number:02d}. {spec['title'].replace(chr(10), ' ')}", "", spec["subtitle"], ""]
        for block in spec["blocks"]:
            kind = block[0]
            if kind in {"qa", "note", "callout", "formula"}:
                lines += [f"### {plain(block[1])}", "", plain(block[2]), ""]
            elif kind in {"steps", "flow", "compare", "glossary", "links"}:
                for term, body in block[1]:
                    lines += [f"- **{plain(term)}:** {plain(body)}"]
                lines.append("")
            elif kind == "table":
                lines += ["| " + " | ".join(block[1]) + " |", "| " + " | ".join("---" for _ in block[1]) + " |"]
                lines += ["| " + " | ".join(plain(cell) for cell in row) + " |" for row in block[2]]
                lines.append("")
            elif kind == "confusion":
                lines += ["Illustrative matrix: actual A → predicted A=40, B=10; actual B → predicted A=20, B=30.", ""]
            elif kind == "snapshot":
                lines += ["Current fruit snapshot: 2,960 images, 9 classes, 20 final epochs; comparison includes DynamicCNN, manual CNN and ResNet-18 metrics.", ""]
            elif kind == "audio":
                lines += ["Diagram: waveform → illustrative log-mel map.", ""]
            elif kind == "cover":
                lines += ["From ‘What is a CNN?’ to ‘Let me explain our system.’", ""]
            elif kind == "routes":
                lines += ["Routes: idea 3–7; metrics 8–13; improvements 14–17; datasets/audio 18–21; files/state 22–25; practice 26–33; glossary 34–38; sources/recap 39–40.", ""]
        if spec.get("source"):
            lines += [f"*Evidence: {spec['source']}*", ""]
    (OUT / "NEUROFORGE-Mutation-Playbook.md").write_text("\n".join(lines))


def main():
    assert len(PAGES) == 40, f"Update page routes/cover: got {len(PAGES)} pages"
    register_fonts()
    capture_snapshot()
    cv = canvas.Canvas(str(PDF), pagesize=(W, H), pageCompression=1)
    cv.setTitle("NEUROFORGE — The Mutation Playbook")
    cv.setAuthor("NEUROFORGE project")
    cv.setSubject("A plain-English project guide: questions, metrics, mutation, training, audio, limitations and viva practice")
    cv.setKeywords("NEUROFORGE, mutation, CNN, NAS, project guide, viva, training, metrics, audio")
    cv.setViewerPreference("DisplayDocTitle", "true")
    for n, spec in enumerate(PAGES, 1):
        cv.bookmarkPage(f"p{n}")
        cv.addOutlineEntry(f"{n:02d}  {spec['title'].replace(chr(10), ' ')}", f"p{n}", level=0)
        if n == 1:
            cover(cv)
            STATS.append({"page": n, "scale": 1, "content_bottom": 814})
        else:
            chosen = None
            for scale in [1, .97, .94, .91, .88, .85]:
                draft = Layout(cv, n, n in DARK_PAGES, scale, dry=True)
                start = draft.header(spec)
                bottom = draft.blocks(spec["blocks"], start)
                if bottom <= H - BOTTOM:
                    chosen = scale
                    break
            if chosen is None:
                raise ValueError(f"Page {n} overflows: {bottom:.1f} > {H-BOTTOM:.1f}. Shorten content or layout.")
            layout = Layout(cv, n, n in DARK_PAGES, chosen)
            start = layout.header(spec)
            bottom = layout.blocks(spec["blocks"], start)
            layout.footer(spec)
            STATS.append({"page": n, "scale": chosen, "content_bottom": round(bottom, 2)})
        cv.showPage()
    cv.save()
    markdown_companion()
    (OUT / "layout-check.json").write_text(json.dumps(STATS, indent=2) + "\n")
    print(f"Built {len(PAGES)} pages: {PDF.relative_to(ROOT)}")
    print("Smallest body font:", round(10.7 * min(x["scale"] for x in STATS), 2))


if __name__ == "__main__":
    main()
