import streamlit as st
import os
import json
import shutil
import time
import zipfile
from pathlib import Path

import neuroforge_core
import architect


PROJECT_ROOT = Path(__file__).resolve().parent
WEIGHTS_PATH = PROJECT_ROOT / "best_model_weights.pth"
CONFIG_PATH = PROJECT_ROOT / "best_model_config.json"
METADATA_PATH = PROJECT_ROOT / "best_model_metadata.json"
HISTORY_PATH = PROJECT_ROOT / "search_history.json"


def load_json(path):
    try:
        with open(path) as handle:
            return json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def load_saved_search_result():
    """Restore the last persisted search so the dashboard survives a rerun."""
    history = load_json(HISTORY_PATH)
    metadata = load_json(METADATA_PATH)
    config = load_json(CONFIG_PATH)
    if not history or not metadata or not config:
        return None

    best_candidate_id = metadata.get("best_candidate_id")
    best_result = next(
        (item for item in history if item.get("candidate_id") == best_candidate_id),
        history[0],
    )
    return {
        "run_id": metadata.get("run_id"),
        "best_config": config,
        "best_result": best_result,
        "manual_baseline": metadata.get("manual_baseline"),
        "history": history,
        "metadata": metadata,
    }


def register_dataset(path, metadata):
    st.session_state["dataset_dir"] = path
    st.session_state["dataset_metadata"] = metadata
    st.session_state["dataset_classes"] = metadata["num_classes"]
    st.session_state["class_names"] = metadata["class_names"]
    st.session_state["model_trained"] = False
    st.session_state["use_tl_for_inference"] = False


def render_lineage_board(search_result):
    """Render the model family tree as a presentation-friendly native UI board."""
    history = search_result.get("history", [])
    if not history:
        return

    by_id = {item["candidate_id"]: item for item in history}
    generation_one = [item for item in history if item["generation"] == 1]
    generation_two = [item for item in history if item["generation"] == 2]
    if not generation_one or not generation_two:
        return

    selected_parent_ids = []
    for item in generation_two:
        parent_id = item.get("parent_id")
        if parent_id and parent_id not in selected_parent_ids:
            selected_parent_ids.append(parent_id)
    if not selected_parent_ids:
        selected_parent_ids = [max(generation_one, key=lambda item: item["score"])["candidate_id"]]
    selected_parents = [by_id[parent_id] for parent_id in selected_parent_ids if parent_id in by_id]
    final_result = search_result.get("best_result") or max(history, key=lambda item: item["score"])

    with st.container(border=True):
        st.subheader("Model lineage", icon=":material/account_tree:")
        st.caption("The strongest candidates become parents. Their characteristics are mutated into the next generation.")

        st.markdown("**Generation 1 · Candidate population**", text_alignment="center")
        first_columns = st.columns(min(4, len(generation_one)))
        for column, item in zip(first_columns, generation_one):
            with column:
                with st.container(border=True):
                    st.markdown(f"**{item['candidate_id'].split('-')[-1]}**")
                    st.caption(f"Accuracy {item['accuracy']:.1f}%")
                    st.caption(f"F1 {item['f1']:.1f}% · {item['parameters']:,} params")

        st.markdown(":material/arrow_downward: **Selected parents**", text_alignment="center")
        parent_columns = st.columns(min(4, len(selected_parents)))
        for column, item in zip(parent_columns, selected_parents):
            with column:
                with st.container(border=True):
                    st.badge("Selected parent", icon=":material/star:", color="violet")
                    st.markdown(f"**{item['candidate_id'].split('-')[-1]}**")
                    st.caption(f"F1 {item['f1']:.1f}% · {item['config']['activation']}")

        st.markdown(":material/arrow_downward: **Mutation produces the next generation**", text_alignment="center")
        st.markdown("**Generation 2 · Mutated candidates**", text_alignment="center")
        second_columns = st.columns(min(4, len(generation_two)))
        for column, item in zip(second_columns, generation_two):
            with column:
                with st.container(border=True):
                    st.markdown(f"**{item['candidate_id'].split('-')[-1]}**")
                    st.caption(f"Parent {item['parent_id'].split('-')[-1] if item.get('parent_id') else '—'}")
                    st.caption(f"Accuracy {item['accuracy']:.1f}% · F1 {item['f1']:.1f}%")

        st.markdown(":material/arrow_downward: **Selection continues until the final architecture emerges**", text_alignment="center")
        with st.container(border=True):
            st.badge("Final selected architecture", icon=":material/emoji_events:", color="green")
            final_columns = st.columns(4)
            with final_columns[0]:
                st.metric("Model", final_result["candidate_id"].split("-")[-1])
            with final_columns[1]:
                st.metric("Proxy accuracy", f"{final_result['accuracy']:.1f}%")
            with final_columns[2]:
                st.metric("Macro F1", f"{final_result['f1']:.1f}%")
            with final_columns[3]:
                st.metric("Architecture", f"{final_result['config']['num_layers']} layers")
            st.caption(
                f"{final_result['config']['activation']} activation · "
                f"filters {final_result['config']['filters']} · "
                f"kernels {final_result['config']['kernel_sizes']}"
            )


persisted_search = load_saved_search_result()
persisted_metadata = load_json(METADATA_PATH) or {}
if "search_result" not in st.session_state and persisted_search:
    st.session_state["search_result"] = persisted_search
if "model_trained" not in st.session_state:
    st.session_state["model_trained"] = (
        WEIGHTS_PATH.exists()
        and persisted_metadata.get("training_mode") in {"dynamic_cnn", "transfer_learning_resnet18"}
    )
if "use_tl_for_inference" not in st.session_state:
    st.session_state["use_tl_for_inference"] = persisted_metadata.get("training_mode") == "transfer_learning_resnet18"
if "dataset_metadata" not in st.session_state and persisted_metadata.get("dataset"):
    saved_dataset = persisted_metadata["dataset"]
    st.session_state["dataset_metadata"] = saved_dataset
    st.session_state["dataset_dir"] = saved_dataset.get("dataset_root")
    st.session_state["dataset_classes"] = saved_dataset.get("num_classes")
    st.session_state["class_names"] = saved_dataset.get("class_names", [])
if st.session_state.get("dataset_dir"):
    try:
        refreshed_dataset = neuroforge_core.get_dataset_metadata(st.session_state["dataset_dir"])
    except (FileNotFoundError, OSError, ValueError):
        refreshed_dataset = None
    if refreshed_dataset:
        st.session_state["dataset_metadata"] = refreshed_dataset
        st.session_state["dataset_dir"] = refreshed_dataset.get("dataset_root", st.session_state["dataset_dir"])
        st.session_state["dataset_classes"] = refreshed_dataset.get("num_classes")
        st.session_state["class_names"] = refreshed_dataset.get("class_names", [])

st.set_page_config(
    page_title="NEUROFORGE",
    page_icon=":material/memory:",
    layout="wide"
)

if "intro_seen" not in st.session_state:
    st.session_state["intro_seen"] = True
    st.html(
        """
        <div class="nf-cinematic-splash">
            <div class="nf-splash-grid"></div>
            <div class="nf-splash-scanline"></div>
            <div class="nf-splash-orbit nf-splash-orbit-one"></div>
            <div class="nf-splash-orbit nf-splash-orbit-two"></div>
            <div class="nf-splash-core"></div>
            <div class="nf-splash-content">
                <div class="nf-splash-kicker">NEURAL ARCHITECTURE // DISCOVERY SYSTEM</div>
                <div class="nf-splash-wordmark" aria-label="NEUROFORGE">
                    <span>N</span><span>E</span><span>U</span><span>R</span><span>O</span><span>F</span><span>O</span><span>R</span><span>G</span><span>E</span>
                </div>
                <div class="nf-splash-tagline"><span>BUILD</span><b>·</b><span>MUTATE</span><b>·</b><span>SHIP</span></div>
                <div class="nf-splash-progress"><span></span></div>
                <div class="nf-splash-status">INITIALIZING THE FORGE <i>///</i></div>
            </div>
        </div>
        <style>
            .nf-cinematic-splash {
                position: relative;
                min-height: 88vh;
                overflow: hidden;
                display: grid;
                place-items: center;
                color: #eef2ff;
                background:
                    radial-gradient(circle at 50% 47%, rgba(139, 92, 246, .22), transparent 18%),
                    radial-gradient(circle at 50% 50%, rgba(34, 211, 238, .11), transparent 34%),
                    #05060b;
                isolation: isolate;
            }
            .nf-cinematic-splash::before,
            .nf-cinematic-splash::after {
                content: "";
                position: absolute;
                inset: 0;
                pointer-events: none;
            }
            .nf-cinematic-splash::before {
                z-index: 4;
                opacity: .18;
                background: repeating-linear-gradient(0deg, rgba(255,255,255,.08) 0 1px, transparent 1px 5px);
                mix-blend-mode: overlay;
            }
            .nf-cinematic-splash::after {
                z-index: 5;
                background: linear-gradient(90deg, transparent, rgba(139, 92, 246, .18), transparent);
                transform: translateX(-110%);
                animation: nf-splash-sweep 2.8s ease-in-out .2s infinite;
            }
            .nf-splash-grid {
                position: absolute;
                inset: -30%;
                opacity: .25;
                transform: perspective(420px) rotateX(64deg) translateY(32%);
                background-image:
                    linear-gradient(rgba(139, 92, 246, .22) 1px, transparent 1px),
                    linear-gradient(90deg, rgba(34, 211, 238, .18) 1px, transparent 1px);
                background-size: 52px 52px;
                animation: nf-splash-grid-drift 7s linear infinite;
            }
            .nf-splash-scanline {
                position: absolute;
                z-index: 3;
                left: 0;
                right: 0;
                height: 2px;
                background: linear-gradient(90deg, transparent, #a78bfa, #67e8f9, transparent);
                box-shadow: 0 0 24px 4px rgba(103, 232, 249, .45);
                transform: translateY(-48vh);
                animation: nf-splash-scan 2.6s ease-in-out .2s infinite;
            }
            .nf-splash-orbit {
                position: absolute;
                z-index: 1;
                width: min(58vw, 640px);
                aspect-ratio: 1;
                border: 1px solid rgba(167, 139, 250, .25);
                border-radius: 50%;
                box-shadow: 0 0 60px rgba(139, 92, 246, .08), inset 0 0 60px rgba(34, 211, 238, .06);
            }
            .nf-splash-orbit-one { transform: rotate(20deg) scaleY(.34); animation: nf-splash-orbit-one 5s ease-in-out infinite; }
            .nf-splash-orbit-two { transform: rotate(-35deg) scaleY(.55); border-color: rgba(103, 232, 249, .16); animation: nf-splash-orbit-two 7s ease-in-out infinite; }
            .nf-splash-core {
                position: absolute;
                z-index: 2;
                width: 9px;
                height: 9px;
                border-radius: 50%;
                background: #f5f3ff;
                box-shadow: 0 0 18px 7px #a78bfa, 0 0 70px 18px rgba(34, 211, 238, .5);
                animation: nf-splash-pulse 1.8s ease-in-out infinite;
            }
            .nf-splash-content {
                position: relative;
                z-index: 6;
                width: min(92%, 1120px);
                text-align: center;
                animation: nf-splash-arrive 1.3s cubic-bezier(.2,.8,.2,1) both;
            }
            .nf-splash-kicker {
                margin-bottom: 22px;
                color: #a5b4fc;
                font: 600 clamp(9px, 1.1vw, 13px)/1.2 ui-monospace, SFMono-Regular, Menlo, monospace;
                letter-spacing: .42em;
                text-transform: uppercase;
                animation: nf-splash-flicker 2.4s steps(2, end) infinite;
            }
            .nf-splash-wordmark {
                display: flex;
                justify-content: center;
                gap: clamp(1px, .45vw, 8px);
                color: #fafaff;
                font: 800 clamp(42px, 9vw, 132px)/.92 ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                letter-spacing: .04em;
                text-shadow: 0 0 8px rgba(255,255,255,.75), 0 0 30px rgba(139,92,246,.65), 0 0 80px rgba(34,211,238,.25);
            }
            .nf-splash-wordmark span { display: inline-block; opacity: 0; transform: translateY(24px) scale(.82); animation: nf-splash-letter .85s cubic-bezier(.2,.8,.2,1) forwards; }
            .nf-splash-wordmark span:nth-child(1) { animation-delay: .18s; }
            .nf-splash-wordmark span:nth-child(2) { animation-delay: .24s; }
            .nf-splash-wordmark span:nth-child(3) { animation-delay: .30s; }
            .nf-splash-wordmark span:nth-child(4) { animation-delay: .36s; }
            .nf-splash-wordmark span:nth-child(5) { animation-delay: .42s; }
            .nf-splash-wordmark span:nth-child(6) { animation-delay: .48s; }
            .nf-splash-wordmark span:nth-child(7) { animation-delay: .54s; }
            .nf-splash-wordmark span:nth-child(8) { animation-delay: .60s; }
            .nf-splash-wordmark span:nth-child(9) { animation-delay: .66s; }
            .nf-splash-wordmark span:nth-child(10) { animation-delay: .72s; }
            .nf-splash-tagline {
                display: flex;
                justify-content: center;
                gap: 14px;
                margin-top: 28px;
                color: #c4b5fd;
                font: 700 12px/1 ui-monospace, SFMono-Regular, Menlo, monospace;
                letter-spacing: .36em;
            }
            .nf-splash-tagline b { color: #67e8f9; font-weight: 400; }
            .nf-splash-progress {
                width: min(360px, 62vw);
                height: 2px;
                margin: 48px auto 15px;
                overflow: hidden;
                background: rgba(148, 163, 184, .2);
            }
            .nf-splash-progress span {
                display: block;
                width: 38%;
                height: 100%;
                background: linear-gradient(90deg, #8b5cf6, #67e8f9);
                box-shadow: 0 0 14px #67e8f9;
                animation: nf-splash-progress 2.3s cubic-bezier(.2,.8,.2,1) forwards;
            }
            .nf-splash-status {
                color: rgba(226, 232, 240, .58);
                font: 600 10px/1.2 ui-monospace, SFMono-Regular, Menlo, monospace;
                letter-spacing: .28em;
            }
            .nf-splash-status i { color: #67e8f9; font-style: normal; animation: nf-splash-dots 1s steps(4, end) infinite; }
            @keyframes nf-splash-arrive { from { opacity: 0; transform: scale(.96); filter: blur(10px); } to { opacity: 1; transform: scale(1); filter: blur(0); } }
            @keyframes nf-splash-letter { to { opacity: 1; transform: translateY(0) scale(1); } }
            @keyframes nf-splash-sweep { 0%, 20% { transform: translateX(-110%); } 65%, 100% { transform: translateX(110%); } }
            @keyframes nf-splash-scan { 0%, 10% { transform: translateY(-48vh); opacity: 0; } 20% { opacity: 1; } 85% { opacity: .8; } 100% { transform: translateY(48vh); opacity: 0; } }
            @keyframes nf-splash-grid-drift { to { background-position: 0 52px, 52px 0; } }
            @keyframes nf-splash-orbit-one { 50% { transform: rotate(200deg) scaleY(.34); } }
            @keyframes nf-splash-orbit-two { 50% { transform: rotate(145deg) scaleY(.55); } }
            @keyframes nf-splash-pulse { 50% { transform: scale(1.7); opacity: .72; } }
            @keyframes nf-splash-progress { from { width: 0; } to { width: 100%; } }
            @keyframes nf-splash-flicker { 0%, 91%, 100% { opacity: 1; } 92% { opacity: .32; } 94% { opacity: .85; } }
            @keyframes nf-splash-dots { 0% { opacity: .2; } 50% { opacity: 1; } 100% { opacity: .2; } }
            @media (prefers-reduced-motion: reduce) {
                .nf-cinematic-splash *, .nf-cinematic-splash::after { animation-duration: .01ms !important; animation-iteration-count: 1 !important; }
            }
        </style>
        """,
        width="stretch",
    )
    time.sleep(5.35)
    st.rerun()

st.html(
    """
    <div class="nf-dashboard-ambient"></div>
    <style>
        div[data-testid="stAppViewContainer"] {
            background:
                radial-gradient(circle at 82% 7%, rgba(139, 92, 246, .13), transparent 24%),
                radial-gradient(circle at 12% 42%, rgba(34, 211, 238, .07), transparent 26%),
                #05060b !important;
        }
        div[data-testid="stHeader"] { background: transparent !important; }
        div[data-testid="stMainBlockContainer"] { position: relative; z-index: 2; }
        div[data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(12, 16, 30, .98), rgba(5, 6, 11, .98)) !important;
            border-right: 1px solid rgba(139, 92, 246, .22);
        }
        .nf-dashboard-ambient {
            position: fixed;
            z-index: 0;
            inset: 0;
            pointer-events: none;
            opacity: .32;
            background-image:
                linear-gradient(rgba(139, 92, 246, .035) 1px, transparent 1px),
                linear-gradient(90deg, rgba(34, 211, 238, .028) 1px, transparent 1px);
            background-size: 72px 72px;
            mask-image: linear-gradient(to bottom, black, transparent 82%);
        }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-color: rgba(148, 163, 184, .18) !important;
            background: linear-gradient(145deg, rgba(15, 18, 33, .72), rgba(5, 6, 11, .58)) !important;
            box-shadow: 0 16px 60px rgba(0, 0, 0, .18), inset 0 1px rgba(255,255,255,.025);
        }
        div[data-testid="stMetric"] {
            border-color: rgba(139, 92, 246, .21) !important;
            background: linear-gradient(145deg, rgba(22, 20, 43, .78), rgba(7, 9, 17, .88)) !important;
            box-shadow: 0 0 0 1px rgba(139, 92, 246, .035), 0 12px 35px rgba(0, 0, 0, .22);
        }
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: #f5f3ff !important;
            text-shadow: 0 0 18px rgba(167, 139, 250, .24);
        }
        button[kind="primary"] {
            border: 1px solid rgba(196, 181, 253, .55) !important;
            background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
            box-shadow: 0 0 22px rgba(124, 58, 237, .28);
        }
        button[kind="primary"]:hover { box-shadow: 0 0 32px rgba(103, 232, 249, .36); }
        button[data-baseweb="tab"] { color: rgba(226, 232, 240, .65) !important; }
        button[data-baseweb="tab"][aria-selected="true"] { color: #c4b5fd !important; text-shadow: 0 0 14px rgba(167, 139, 250, .45); }
        div[data-baseweb="input"]:focus-within { border-color: #8b5cf6 !important; box-shadow: 0 0 0 1px #8b5cf6, 0 0 18px rgba(139, 92, 246, .22); }
        @media (prefers-reduced-motion: reduce) {
            .nf-dashboard-ambient { opacity: .16; }
            .nf-hero-beam, .nf-forge-rail::before { animation: none !important; }
        }
    </style>
    """,
    width="stretch",
)

# Sidebar system status
with st.sidebar:
    st.title("NEUROFORGE", icon=":material/auto_awesome:")
    st.caption("Generative ML studio · v0.2 MVP")
    st.space("large")
    st.subheader("System status", icon=":material/memory:")
    st.badge("Engine online", icon=":material/check_circle:", color="green")
    st.badge(f"Device · {str(neuroforge_core.device).upper()}", icon=":material/developer_board:", color="blue")
    st.badge(
        "Final model ready" if st.session_state.get("model_trained") else "Awaiting training",
        icon=":material/model_training:",
        color="green" if st.session_state.get("model_trained") else "orange",
    )
    st.space("medium")
    st.caption("Pipeline")
    st.markdown(":green-badge[1 · Load]  →  :blue-badge[2 · Mutate]  →  :violet-badge[3 · Ship]")
    st.space("large")
    st.caption("Current workspace")
    dataset_metadata = st.session_state.get("dataset_metadata") or {}
    st.caption(dataset_metadata.get("format", "No dataset linked").replace("_", " ").title())
    if dataset_metadata.get("num_classes"):
        st.caption(f"{dataset_metadata['num_classes']} classes · {dataset_metadata.get('num_samples', '—')} samples")


final_test_metrics = persisted_metadata.get("held_out_test_metrics", {})
dataset_snapshot = persisted_metadata.get("dataset", {})
dataset_label = dataset_snapshot.get("format", "image classification").replace("_", " ").upper()
run_label = persisted_metadata.get("run_id", "NO ACTIVE RUN")
has_dataset = bool(st.session_state.get("dataset_metadata") or dataset_snapshot)
has_search = bool(st.session_state.get("search_result"))
has_model = bool(st.session_state.get("model_trained"))
initial_search_count = len(st.session_state.get("search_result", {}).get("history", [])) if st.session_state.get("search_result") else 0
active_pipeline_stage = 4 if has_model else 3 if has_search else 2 if has_dataset else 1
pipeline_status = "SHIP READY" if has_model else "SEARCH COMPLETE" if has_search else "DATA LOADED" if has_dataset else "AWAITING DATA"
rail_steps = [
    ("01", "LOAD DATA", "Images now · audio next"),
    ("02", "GENERATE", "Build a model population"),
    ("03", "MUTATE", "Keep strong traits · evolve"),
    ("04", "SHIP MVP", "Train · test · download"),
]
rail_track_html = ""
for index, (number, title, copy) in enumerate(rail_steps, start=1):
    active_class = " nf-rail-active" if index == active_pipeline_stage else ""
    rail_track_html += (
        f'<div class="nf-rail-step{active_class}">'
        f'<div class="nf-rail-number">{number}</div>'
        f'<div class="nf-rail-title">{title}</div>'
        f'<div class="nf-rail-copy">{copy}</div>'
        "</div>"
    )
    if index < len(rail_steps):
        rail_track_html += '<div class="nf-rail-connector">›</div>'
st.html(
    """
    <section class="nf-dashboard-hero">
        <div class="nf-hero-signal"><span></span> NEUROFORGE // FORGE CONTROL <b>● __PIPELINE_STATUS__</b></div>
        <div class="nf-hero-title"><span>BUILD.</span> <span>MUTATE.</span> <em>SHIP.</em></div>
        <div class="nf-hero-copy">A generative ML studio that turns data into a measured, evolvable model MVP.</div>
        <div class="nf-hero-meta"><span>DATASET // __DATASET__</span><span>RUN // __RUN__</span><span>ENGINE // MPS READY</span></div>
        <div class="nf-hero-beam"></div>
        <div class="nf-hero-visual" aria-hidden="true">
            <div class="nf-hero-visual-grid"></div>
            <div class="nf-hero-orbit nf-hero-orbit-one"></div>
            <div class="nf-hero-orbit nf-hero-orbit-two"></div>
            <div class="nf-hero-orbit nf-hero-orbit-three"></div>
            <div class="nf-hero-core"><span></span></div>
            <div class="nf-hero-node nf-hero-node-a">GEN</div>
            <div class="nf-hero-node nf-hero-node-b">MUT</div>
            <div class="nf-hero-node nf-hero-node-c">SHIP</div>
        </div>
    </section>
    <style>
        .nf-dashboard-hero {
            position: relative;
            overflow: hidden;
            margin: 18px 0 22px;
            padding: clamp(26px, 4vw, 54px) clamp(20px, 4vw, 58px);
            border: 1px solid rgba(139, 92, 246, .28);
            background:
                radial-gradient(circle at 82% 35%, rgba(139, 92, 246, .2), transparent 24%),
                linear-gradient(115deg, rgba(12, 15, 29, .96), rgba(8, 8, 16, .72));
            box-shadow: inset 0 1px rgba(255,255,255,.05), 0 22px 80px rgba(0,0,0,.24);
        }
        .nf-dashboard-hero::before {
            content: "";
            position: absolute;
            inset: 0;
            opacity: .22;
            background-image: linear-gradient(rgba(139,92,246,.08) 1px, transparent 1px), linear-gradient(90deg, rgba(34,211,238,.06) 1px, transparent 1px);
            background-size: 42px 42px;
            mask-image: linear-gradient(90deg, black, transparent 76%);
        }
         .nf-hero-signal, .nf-hero-copy, .nf-hero-meta, .nf-hero-title { position: relative; z-index: 1; }
        .nf-hero-signal {
            color: #a5b4fc;
            font: 700 10px/1.2 ui-monospace, SFMono-Regular, Menlo, monospace;
            letter-spacing: .25em;
        }
        .nf-hero-signal span { display: inline-block; width: 7px; height: 7px; margin-right: 8px; border-radius: 50%; background: #67e8f9; box-shadow: 0 0 14px #67e8f9; }
        .nf-hero-signal b { margin-left: 18px; color: #86efac; font-weight: 700; letter-spacing: .12em; }
        .nf-hero-title {
            margin: 20px 0 14px;
            color: #f8fafc;
            font: 850 clamp(42px, 7vw, 94px)/.88 ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            letter-spacing: -.06em;
            text-shadow: 0 0 24px rgba(167,139,250,.18);
        }
        .nf-hero-title span:nth-child(2) { color: #c4b5fd; }
        .nf-hero-title em { color: #67e8f9; font-style: normal; text-shadow: 0 0 28px rgba(103,232,249,.3); }
        .nf-hero-copy { max-width: 680px; color: rgba(226,232,240,.75); font-size: 15px; letter-spacing: .01em; }
         .nf-hero-meta { display: flex; flex-wrap: wrap; gap: 16px 26px; margin-top: 28px; color: rgba(148,163,184,.68); font: 600 9px/1.2 ui-monospace, SFMono-Regular, Menlo, monospace; letter-spacing: .18em; }
         .nf-hero-meta span::before { content: "// "; color: #8b5cf6; }
         .nf-hero-beam { position: absolute; top: 0; bottom: 0; left: -30%; width: 20%; transform: skewX(-18deg); background: linear-gradient(90deg, transparent, rgba(103,232,249,.1), transparent); animation: nf-hero-beam 7s ease-in-out infinite; }
         .nf-hero-visual { position: absolute; z-index: 0; top: 50%; right: clamp(24px, 8vw, 120px); width: min(31vw, 370px); aspect-ratio: 1; transform: translateY(-50%); opacity: .82; pointer-events: none; }
         .nf-hero-visual::before { content: ""; position: absolute; inset: 16%; border-radius: 50%; background: radial-gradient(circle, rgba(103,232,249,.2), rgba(139,92,246,.1) 34%, transparent 70%); filter: blur(16px); }
         .nf-hero-visual-grid { position: absolute; inset: 8%; border: 1px solid rgba(103,232,249,.13); background: linear-gradient(rgba(103,232,249,.08) 1px, transparent 1px), linear-gradient(90deg, rgba(139,92,246,.08) 1px, transparent 1px); background-size: 24px 24px; transform: rotate(45deg); mask-image: radial-gradient(circle, black 18%, transparent 70%); opacity: .55; }
         .nf-hero-orbit { position: absolute; border: 1px solid rgba(167,139,250,.55); border-radius: 50%; box-shadow: 0 0 18px rgba(139,92,246,.12); }
         .nf-hero-orbit-one { inset: 12%; transform: rotateX(67deg) rotateZ(22deg); animation: nf-hero-orbit-one 10s linear infinite; }
         .nf-hero-orbit-two { inset: 22%; border-color: rgba(103,232,249,.52); transform: rotateY(67deg) rotateZ(-28deg); animation: nf-hero-orbit-two 8s linear infinite reverse; }
         .nf-hero-orbit-three { inset: 31%; border-style: dashed; border-color: rgba(196,181,253,.38); transform: rotateX(70deg) rotateZ(84deg); animation: nf-hero-orbit-three 13s linear infinite; }
         .nf-hero-core { position: absolute; inset: 37%; display: grid; place-items: center; border: 1px solid rgba(103,232,249,.62); border-radius: 50%; background: radial-gradient(circle at 38% 34%, #f5f3ff 0 2%, #67e8f9 5%, #8b5cf6 26%, rgba(76,29,149,.5) 54%, rgba(5,6,11,.12) 72%); box-shadow: 0 0 30px rgba(103,232,249,.32), 0 0 70px rgba(139,92,246,.28); animation: nf-hero-core-pulse 3.2s ease-in-out infinite; }
         .nf-hero-core span { width: 14%; aspect-ratio: 1; border-radius: 50%; background: #fff; box-shadow: 0 0 16px #67e8f9, 0 0 32px #8b5cf6; }
         .nf-hero-node { position: absolute; color: rgba(226,232,240,.65); font: 700 8px/1 ui-monospace, SFMono-Regular, Menlo, monospace; letter-spacing: .18em; text-shadow: 0 0 12px rgba(103,232,249,.7); }
         .nf-hero-node-a { top: 11%; right: 10%; color: #67e8f9; }
         .nf-hero-node-b { bottom: 12%; left: 7%; color: #c4b5fd; }
         .nf-hero-node-c { top: 49%; right: -2%; color: #86efac; }
         @keyframes nf-hero-beam { 0%, 58% { left: -30%; } 85%, 100% { left: 125%; } }
         @keyframes nf-hero-orbit-one { to { transform: rotateX(67deg) rotateZ(382deg); } }
         @keyframes nf-hero-orbit-two { to { transform: rotateY(67deg) rotateZ(-388deg); } }
         @keyframes nf-hero-orbit-three { to { transform: rotateX(70deg) rotateZ(444deg); } }
         @keyframes nf-hero-core-pulse { 50% { transform: scale(1.08); filter: brightness(1.18); } }
         @media (max-width: 800px) {
             .nf-dashboard-hero { margin: 10px 0 16px; padding: 28px 22px; }
             .nf-hero-title { font-size: clamp(40px, 15vw, 72px); line-height: .94; }
             .nf-hero-copy { font-size: 13px; }
             .nf-hero-meta { gap: 10px 16px; letter-spacing: .12em; }
             .nf-hero-visual { right: -90px; width: 260px; opacity: .22; }
         }
         @media (prefers-reduced-motion: reduce) { .nf-hero-orbit, .nf-hero-core { animation: none !important; } }
     </style>
     """.replace("__DATASET__", dataset_label).replace("__RUN__", run_label).replace("__PIPELINE_STATUS__", pipeline_status),
    width="stretch",
)
with st.container(horizontal=True):
    top_accuracy_metric = st.empty()
    top_accuracy_metric.metric("Final test accuracy", f"{final_test_metrics.get('accuracy', 0):.2f}%" if final_test_metrics else "—", border=True)
    top_f1_metric = st.empty()
    top_f1_metric.metric("Macro F1", f"{final_test_metrics.get('f1', 0):.2f}%" if final_test_metrics else "—", border=True)
    models_searched_metric = st.empty()
    models_searched_metric.metric("Models searched", str(initial_search_count) if initial_search_count else "—", border=True)
    st.metric("Classes", str(dataset_snapshot.get("num_classes", "—")), border=True)

st.html(
    """
    <section class="nf-forge-rail">
        <div class="nf-rail-header"><span>THE NEUROFORGE LOOP</span><b>LIVE PIPELINE</b></div>
        <div class="nf-rail-track">
            __RAIL_TRACK__
        </div>
    </section>
    <style>
        .nf-forge-rail {
            position: relative;
            overflow: hidden;
            margin: 4px 0 28px;
            padding: 22px 26px 26px;
            border: 1px solid rgba(103, 232, 249, .16);
            background: linear-gradient(135deg, rgba(10, 17, 31, .9), rgba(15, 10, 29, .84));
            box-shadow: inset 0 1px rgba(255,255,255,.035), 0 18px 60px rgba(0,0,0,.18);
        }
        .nf-forge-rail::before { content: ""; position: absolute; inset: 0; opacity: .22; background: linear-gradient(90deg, transparent, rgba(103,232,249,.09), transparent); transform: translateX(-100%); animation: nf-rail-sweep 8s ease-in-out infinite; }
        .nf-rail-header { position: relative; display: flex; justify-content: space-between; align-items: center; color: #c4b5fd; font: 700 11px/1.2 ui-monospace, SFMono-Regular, Menlo, monospace; letter-spacing: .22em; }
        .nf-rail-header b { color: #86efac; font-size: 9px; letter-spacing: .16em; }
        .nf-rail-header b::before { content: ""; display: inline-block; width: 6px; height: 6px; margin-right: 8px; border-radius: 50%; background: #86efac; box-shadow: 0 0 12px #86efac; vertical-align: 1px; }
        .nf-rail-track { position: relative; display: flex; align-items: stretch; gap: 12px; margin-top: 22px; }
        .nf-rail-step { position: relative; flex: 1; min-height: 96px; padding: 15px 16px; border: 1px solid rgba(148,163,184,.17); background: rgba(5,6,11,.52); }
        .nf-rail-step::after { content: ""; position: absolute; left: 0; right: 0; bottom: 0; height: 2px; background: linear-gradient(90deg, transparent, rgba(139,92,246,.45), transparent); }
        .nf-rail-active { border-color: rgba(139,92,246,.55); background: linear-gradient(135deg, rgba(76,29,149,.2), rgba(5,6,11,.58)); box-shadow: 0 0 28px rgba(139,92,246,.12); }
        .nf-rail-number { color: #67e8f9; font: 700 10px/1 ui-monospace, SFMono-Regular, Menlo, monospace; letter-spacing: .18em; }
        .nf-rail-title { margin-top: 17px; color: #f5f3ff; font: 800 13px/1 ui-sans-serif, system-ui, sans-serif; letter-spacing: .1em; }
        .nf-rail-copy { margin-top: 10px; color: rgba(226,232,240,.56); font-size: 11px; }
        .nf-rail-connector { display: grid; place-items: center; color: #8b5cf6; font: 400 31px/1 ui-sans-serif, system-ui, sans-serif; text-shadow: 0 0 16px rgba(139,92,246,.5); }
        @keyframes nf-rail-sweep { 0%, 58% { transform: translateX(-100%); } 82%, 100% { transform: translateX(100%); } }
        @media (max-width: 800px) { .nf-rail-track { flex-wrap: wrap; } .nf-rail-step { min-width: 42%; } .nf-rail-connector { display: none; } }
    </style>
     """.replace("__RAIL_TRACK__", rail_track_html),
    width="stretch",
)

st.space("small")

# Create tabs for the product flow.
tab1, tab2, tab3, tab4 = st.tabs(["Dataset", "Search lab", "Final model", "Live test"])

with tab1:
    with st.container(border=True):
        st.header("Upload Dataset", icon=":material/cloud_upload:")
        
        data_type = st.pills(
            "Select modality",
            options=["Images", "Audio (coming soon)"],
            default="Images",
        )
        
        st.caption("Use class folders (ImageFolder) or a Roboflow export containing train/valid/test folders with _classes.csv.")
        
        if data_type != "Images":
            st.info("Audio is part of the NEUROFORGE roadmap. The current MVP supports image classification with ImageFolder and Roboflow exports.", icon=":material/graphic_eq:")
        upload_mode = st.radio("How do you want to load data?", ["Enter Local Path (best for large datasets)", "Upload ZIP file"])
        
        if data_type == "Images" and upload_mode == "Upload ZIP file":
            uploaded_file = st.file_uploader("Upload image ZIP", type=["zip"])
            
            if uploaded_file is not None:
                extract_dir = f"./custom_data/{data_type.lower()}"
                if os.path.exists(extract_dir):
                    shutil.rmtree(extract_dir)
                os.makedirs(extract_dir, exist_ok=True)
                
                with st.spinner(f"Extracting {uploaded_file.name}..."):
                    with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
                        zip_ref.extractall(extract_dir)
                
                search_dir = extract_dir
                if not neuroforge_core.find_roboflow_dataset_root(search_dir):
                    search_dir = str(neuroforge_core.find_imagefolder_dataset_root(search_dir))
                try:
                    dataset_metadata = neuroforge_core.get_dataset_metadata(search_dir)
                except (ValueError, FileNotFoundError) as exc:
                    st.error(f"Dataset format not recognized: {exc}")
                    dataset_metadata = None

                if dataset_metadata:
                    class_names = dataset_metadata['class_names']
                    st.session_state['dataset_classes'] = dataset_metadata['num_classes']
                    st.session_state['class_names'] = class_names
                    st.session_state['dataset_dir'] = dataset_metadata.get('dataset_root', search_dir)
                    st.session_state['dataset_metadata'] = dataset_metadata
                    st.session_state['model_trained'] = False

                    st.toast(f"Successfully extracted {len(class_names)} classes!", icon=":material/check_circle:")
                    st.success(f"Dataset ready! Found {len(class_names)} classes: {', '.join(class_names[:5])}{'...' if len(class_names)>5 else ''}. Proceed to Architecture Search.", icon=":material/arrow_forward:")
                    if dataset_metadata.get('splits'):
                        st.caption(f"Usable samples — train: {dataset_metadata['splits']['train']}, validation: {dataset_metadata['splits']['valid']}, test: {dataset_metadata['splits']['test']}.")
                    quality_report = dataset_metadata.get('quality_report', {})
                    skipped_rows = sum(report.get('skipped_rows', 0) for report in quality_report.values() if isinstance(report, dict))
                    if skipped_rows:
                        st.warning(f"Skipped {skipped_rows} label rows with no label or multiple labels; they are excluded from classification training.")
                
        elif data_type == "Images":
            # Local Path Mode
            st.info("Since NEUROFORGE is running locally, you can just point it directly to the folder on your Mac!")
            local_path = st.text_input("Enter absolute path to dataset folder (e.g. ~/Downloads/crop pest dataset):")
            
            if st.button("Load Local Dataset", type="primary") and local_path:
                # Clean up the path (remove accidental spaces and expand ~ to /Users/...)
                clean_path = os.path.expanduser(local_path.strip())
                
                if os.path.exists(clean_path) and os.path.isdir(clean_path):
                    try:
                        dataset_metadata = neuroforge_core.get_dataset_metadata(clean_path)
                    except (ValueError, FileNotFoundError) as exc:
                        st.error(f"Dataset format not recognized: {exc}")
                        dataset_metadata = None

                    if dataset_metadata and dataset_metadata['num_classes'] > 0:
                        class_names = dataset_metadata['class_names']
                        st.session_state['dataset_classes'] = dataset_metadata['num_classes']
                        st.session_state['dataset_dir'] = dataset_metadata.get('dataset_root', clean_path)
                        st.session_state['class_names'] = class_names
                        st.session_state['dataset_metadata'] = dataset_metadata
                        st.session_state['model_trained'] = False
                        
                        st.toast(f"Successfully linked {len(class_names)} classes!", icon=":material/check_circle:")
                        st.success(f"Dataset ready! Found {len(class_names)} classes: {', '.join(class_names[:5])}{'...' if len(class_names)>5 else ''}. Proceed to Architecture Search.", icon=":material/arrow_forward:")
                        if dataset_metadata.get('splits'):
                            st.caption(f"Usable samples — train: {dataset_metadata['splits']['train']}, validation: {dataset_metadata['splits']['valid']}, test: {dataset_metadata['splits']['test']}.")
                        quality_report = dataset_metadata.get('quality_report', {})
                        skipped_rows = sum(report.get('skipped_rows', 0) for report in quality_report.values() if isinstance(report, dict))
                        if skipped_rows:
                            st.warning(f"Skipped {skipped_rows} label rows with no label or multiple labels; they are excluded from classification training.")
                    else:
                        st.error(f"No supported classes found inside '{clean_path}'. Use class folders or a supported Roboflow export.")
                else:
                    st.error(f"Path does not exist: '{clean_path}'. Please check the path and try again.")

with tab2:
    with st.container(border=True):
        st.header("Search lab", icon=":material/auto_fix_high:")
        st.caption("Watch NEUROFORGE generate, evaluate, and mutate a population of model architectures.")

        active_dataset = st.session_state.get("dataset_metadata") or {}
        with st.container(horizontal=True):
            st.badge(
                active_dataset.get("format", "No dataset linked").replace("_", " ").title(),
                icon=":material/database:",
                color="blue",
            )
            st.badge(
                f"{active_dataset.get('num_classes', '—')} classes",
                icon=":material/category:",
                color="violet",
            )
            st.badge(
                f"Seed 42 · fixed split",
                icon=":material/lock:",
                color="green",
            )
        
        with st.container(border=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                generations = st.number_input("Generations to evolve", min_value=1, max_value=50, value=5)
            with col2:
                population = st.number_input("Models per generation", min_value=2, max_value=16, value=8)
            with col3:
                proxy_epochs = st.number_input("Proxy epochs per model", min_value=1, max_value=8, value=3)

        st.caption(
            f"Search budget: {int(generations * population)} candidate evaluations + 1 manual baseline. "
            "Larger populations explore more architectural mutations; proxy epochs control evaluation depth."
        )
        
        # Placeholders for live updates
        status_text = st.empty()
        progress_bar = st.empty()
        metrics_col1, metrics_col2 = st.columns(2)
        best_acc_metric = metrics_col1.empty()
        current_model_metric = metrics_col2.empty()
        
        def architect_callback(update):
            if update["type"] == "generation":
                status_text.info(f"🧬 Generation {update['current']} of {update['total']}...", icon=":material/science:")
                progress_bar.progress((update['current'] - 1) / update['total'])
            elif update["type"] == "evaluating":
                current_model_metric.metric("Currently Evaluating", f"Candidate {update['candidate']}/{update['total']}")
            elif update["type"] == "result":
                pass # Can add more detailed logs here if needed
            elif update["type"] == "new_best":
                best_acc_metric.metric("Best Proxy Accuracy", f"{update['accuracy']:.2f}%", delta="New Best!")
                st.toast(f"New best architecture found: {update['accuracy']:.2f}%", icon=":material/emoji_events:")
            elif update["type"] == "complete":
                status_text.success("✅ Architect Search Complete!")
                current_model_metric.metric("Search status", "Complete")
                progress_bar.progress(1.0)
                st.balloons()

        with st.container(horizontal=True, horizontal_alignment="right"):
            if st.button("Start Search", icon=":material/play_arrow:", type="primary"):
                # Initialize empty states
                best_acc_metric.metric("Best Proxy Accuracy", "0.00%")
                current_model_metric.metric("Currently Evaluating", "Initializing...")
                progress_bar.progress(0)
                
                # Run the search and retain its auditable candidate history.
                dataset_dir = st.session_state.get('dataset_dir')
                with st.spinner("🧠 The Architect is mutating models in the background..."):
                    search_result = architect.run_architect_search(
                        generations=generations, 
                        population_size=population, 
                        st_callback=architect_callback,
                        dataset_dir=dataset_dir,
                        proxy_epochs=proxy_epochs,
                    )
                st.session_state['search_result'] = search_result
                st.session_state['model_trained'] = False
                st.session_state['use_tl_for_inference'] = False
                models_searched_metric.metric("Models searched", str(len(search_result["history"])), border=True)
                st.success(
                    f"Built and evaluated {len(search_result['history'])} candidate models "
                    f"across {generations} generations, plus one manual baseline."
                )

        search_result = st.session_state.get('search_result')
        if search_result:
            baseline = search_result.get("manual_baseline")
            best_result = search_result.get("best_result")
            if baseline and best_result:
                st.subheader("Baseline comparison", icon=":material/compare_arrows:")
                comparison_rows = [
                    {
                        "Model": "Manual baseline",
                        "Accuracy": f"{baseline['accuracy']:.2f}%",
                        "Precision": f"{baseline['precision']:.2f}%",
                        "Recall": f"{baseline['recall']:.2f}%",
                        "F1": f"{baseline['f1']:.2f}%",
                        "Parameters": baseline["parameters"],
                        "Architecture": "2 layers · 32/64 filters",
                    },
                    {
                        "Model": "NEUROFORGE selected",
                        "Accuracy": f"{best_result['accuracy']:.2f}%",
                        "Precision": f"{best_result['precision']:.2f}%",
                        "Recall": f"{best_result['recall']:.2f}%",
                        "F1": f"{best_result['f1']:.2f}%",
                        "Parameters": best_result["parameters"],
                        "Architecture": f"Generation {best_result['generation']} · {best_result['config']['num_layers']} layers",
                    },
                ]
                st.dataframe(comparison_rows, width="stretch", hide_index=True)
                st.caption("This is a controlled proxy-budget comparison. The held-out test comparison comes after final training.")

            render_lineage_board(search_result)

            st.subheader("Candidate Models", icon=":material/table_chart:")
            history_rows = [
                {
                    "Model": item["candidate_id"],
                    "Generation": item["generation"],
                    "Parent": item["parent_id"] or "Initial population",
                    "Accuracy": f"{item['accuracy']:.2f}%",
                    "Precision": f"{item['precision']:.2f}%",
                    "Recall": f"{item['recall']:.2f}%",
                    "F1": f"{item['f1']:.2f}%",
                    "Parameters": item["parameters"],
                    "Score": f"{item['score']:.2f}",
                }
                for item in search_result["history"]
            ]
            st.dataframe(history_rows, width="stretch", hide_index=True)
            generation_scores = []
            for generation in sorted({item["generation"] for item in search_result["history"]}):
                generation_scores.append(
                    max(item["accuracy"] for item in search_result["history"] if item["generation"] == generation)
                )
            with st.container(border=True):
                st.subheader("Search progression", icon=":material/trending_up:")
                st.caption("Best proxy accuracy discovered in each generation")
                st.line_chart(generation_scores, height=180)
            with st.expander("Final architecture characteristics"):
                st.json(search_result["best_config"])
                st.caption(
                    "The characteristics shown here are the architecture decisions that survived "
                    "selection and mutation. Learned feature maps will be added after the search "
                    "pipeline is validated."
                )
            history_download = json.dumps(search_result["history"], indent=2)
            st.download_button(
                "Download search history",
                data=history_download,
                file_name="neuroforge_search_history.json",
                mime="application/json",
                icon=":material/download:",
            )
        
with tab3:
    with st.container(border=True):
        st.header("Final model", icon=":material/model_training:")
        st.caption("This is the model MVP produced by the search. Retrain it, inspect its held-out metrics, and download the artifacts.")

        artifact_metadata = st.session_state.get("artifact_metadata") or load_json(METADATA_PATH) or {}
        artifact_test_metrics = artifact_metadata.get("held_out_test_metrics", {})
        ship_snapshot_slots = {}
        if artifact_metadata:
            with st.container(border=True):
                st.subheader("Ship-ready snapshot", icon=":material/verified:")
                final_columns = st.columns(4)
                with final_columns[0]:
                    ship_snapshot_slots["accuracy"] = st.empty()
                    ship_snapshot_slots["accuracy"].metric("Test accuracy", f"{artifact_test_metrics.get('accuracy', 0):.2f}%" if artifact_test_metrics else "Not evaluated")
                with final_columns[1]:
                    ship_snapshot_slots["f1"] = st.empty()
                    ship_snapshot_slots["f1"].metric("Macro F1", f"{artifact_test_metrics.get('f1', 0):.2f}%" if artifact_test_metrics else "Not evaluated")
                with final_columns[2]:
                    ship_snapshot_slots["precision"] = st.empty()
                    ship_snapshot_slots["precision"].metric("Precision", f"{artifact_test_metrics.get('precision', 0):.2f}%" if artifact_test_metrics else "Not evaluated")
                with final_columns[3]:
                    artifact_class_count = artifact_metadata.get("num_classes") or artifact_metadata.get("dataset", {}).get("num_classes", "—")
                    st.metric("Classes", str(artifact_class_count))
                st.caption(
                    f"Model type: {artifact_metadata.get('training_mode', 'unknown').replace('_', ' ')} · "
                    f"Run: {artifact_metadata.get('run_id', '—')} · "
                    f"Weights: {artifact_metadata.get('weights_path', 'not trained yet')}"
                )

            config_for_display = load_json(CONFIG_PATH)
            if config_for_display:
                with st.expander("Inspect final architecture", icon=":material/account_tree:"):
                    st.json(config_for_display)

            download_columns = st.columns(3)
            with download_columns[0]:
                if WEIGHTS_PATH.exists():
                    st.download_button(
                        "Download weights",
                        data=WEIGHTS_PATH.read_bytes(),
                        file_name="neuroforge_final_model.pth",
                        mime="application/octet-stream",
                        icon=":material/download:",
                    )
            with download_columns[1]:
                if CONFIG_PATH.exists():
                    st.download_button(
                        "Download architecture",
                        data=CONFIG_PATH.read_bytes(),
                        file_name="neuroforge_final_config.json",
                        mime="application/json",
                        icon=":material/code:",
                    )
            with download_columns[2]:
                if METADATA_PATH.exists():
                    st.download_button(
                        "Download report",
                        data=METADATA_PATH.read_bytes(),
                        file_name="neuroforge_final_report.json",
                        mime="application/json",
                        icon=":material/description:",
                    )

        st.space("small")
        with st.container(border=True):
            st.subheader("Retrain the selected model", icon=":material/rocket_launch:")
            st.caption("Use the generated architecture by default. ResNet-18 remains available only as a comparison baseline.")
        
        train_status = st.empty()
        train_progress = st.empty()
        metrics_col1, metrics_col2, metrics_col3, metrics_col4, metrics_col5 = st.columns(5)
        train_acc_metric = metrics_col1.empty()
        val_acc_metric = metrics_col2.empty()
        precision_metric = metrics_col3.empty()
        f1_metric = metrics_col4.empty()
        eta_metric = metrics_col5.empty()
        
        def train_callback(update):
            train_status.info(f"Epoch {update['epoch']} of {update['total_epochs']} running...", icon=":material/model_training:")
            train_progress.progress(update['epoch'] / update['total_epochs'])
            train_acc_metric.metric("Training Accuracy", f"{update['train_acc']:.2f}%")
            val_acc_metric.metric("Validation Accuracy", f"{update['val_acc']:.2f}%")
            precision_metric.metric("Validation Precision", f"{update.get('val_precision', 0):.2f}%")
            f1_metric.metric("Validation F1", f"{update.get('val_f1', 0):.2f}%")
            eta_metric.metric("Estimated Time Remaining", update.get('eta', 'Calculating...'))
        
        with st.container(horizontal=True):
            training_epochs = st.number_input("Training epochs", min_value=1, max_value=100, value=20)
            demo_mode = st.checkbox("Presentation mode (1 epoch)", value=False)
            use_tl = st.checkbox("Use ResNet-18 baseline", value=False)
            
            if st.button("Run Full Training", icon=":material/rocket_launch:", type="primary"):
                train_status.info("Loading best architecture...", icon=":material/hourglass_top:")
                train_progress.progress(0)
                train_acc_metric.metric("Training Accuracy", "0.00%")
                val_acc_metric.metric("Validation Accuracy", "0.00%")
                precision_metric.metric("Validation Precision", "0.00%")
                f1_metric.metric("Validation F1", "0.00%")
                eta_metric.metric("Estimated Time Remaining", "Calculating...")
                
                try:
                    with open("best_model_config.json", "r") as f:
                        best_config = json.load(f)
                        
                    epochs_to_run = 1 if demo_mode else training_epochs
                    
                    # Run the full deep training loop
                    dataset_dir = st.session_state.get('dataset_dir')
                    with st.spinner(f"🔥 Training on Apple Silicon MPS for {epochs_to_run} epochs..."):
                        final_acc = neuroforge_core.full_train(
                            best_config,
                            epochs=epochs_to_run,
                            st_callback=train_callback,
                            dataset_dir=dataset_dir,
                            use_transfer_learning=use_tl,
                        )
                    
                    st.session_state['use_tl_for_inference'] = use_tl
                    st.session_state['model_trained'] = True
                    st.session_state['artifact_metadata'] = load_json(METADATA_PATH) or {}
                    refreshed_metadata = st.session_state['artifact_metadata']
                    refreshed_test_metrics = refreshed_metadata.get("held_out_test_metrics", {})
                    if refreshed_test_metrics:
                        if ship_snapshot_slots:
                            ship_snapshot_slots["accuracy"].metric("Test accuracy", f"{refreshed_test_metrics['accuracy']:.2f}%")
                            ship_snapshot_slots["f1"].metric("Macro F1", f"{refreshed_test_metrics['f1']:.2f}%")
                            ship_snapshot_slots["precision"].metric("Precision", f"{refreshed_test_metrics['precision']:.2f}%")
                        top_accuracy_metric.metric("Final test accuracy", f"{refreshed_test_metrics['accuracy']:.2f}%", border=True)
                        top_f1_metric.metric("Macro F1", f"{refreshed_test_metrics['f1']:.2f}%", border=True)
                    
                    train_status.success(f"🎉 Training Complete! Final Validation Accuracy: {final_acc:.2f}%", icon=":material/verified:")
                    train_progress.progress(1.0)
                    st.balloons()
                    
                except FileNotFoundError:
                    train_status.error("No architecture found! Please run the Architect Search first so it can generate 'best_model_config.json'.", icon=":material/error:")

# --- TAB 4: Live Testing ---
with tab4:
    st.header("Live test", icon=":material/visibility:")
    st.caption("Upload one image and inspect the final model's prediction and confidence.")
    
    if not st.session_state.get('model_trained', False):
        st.warning("Train a final model in the Final model tab before testing.", icon=":material/warning:")
    else:
        uploaded_test_file = st.file_uploader("Upload an image to test", type=["png", "jpg", "jpeg"], key="test_upload")
        
        if uploaded_test_file is not None:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.image(uploaded_test_file, caption="Input Image", width="stretch")
                
            with col2:
                with st.spinner("Analyzing image..."):
                    try:
                        from PIL import Image
                        import torchvision.transforms as transforms
                        
                        # Open and transform image
                        image = Image.open(uploaded_test_file).convert('RGB')
                        transform = transforms.Compose([
                            transforms.Resize((64, 64)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                        ])
                        img_tensor = transform(image)
                        
                        # Get model config and classes
                        with open(CONFIG_PATH, "r") as f:
                            best_config = json.load(f)
                        saved_dataset = st.session_state.get("dataset_metadata") or persisted_metadata.get("dataset", {})
                        num_classes = st.session_state.get('dataset_classes') or saved_dataset.get('num_classes', 10)
                        class_names = st.session_state.get('class_names') or saved_dataset.get('class_names', [f"Class {i}" for i in range(num_classes)])
                        use_tl = st.session_state.get('use_tl_for_inference', False)
                        
                        # Predict
                        pred_idx, confidence = neuroforge_core.predict_image(img_tensor, best_config, num_classes, use_transfer_learning=use_tl)
                        predicted_label = class_names[pred_idx]
                        
                        st.success(f"**Prediction:** {predicted_label.title()}", icon=":material/target:")
                        st.metric("Confidence Score", f"{confidence * 100:.2f}%")
                        
                    except Exception as e:
                        st.error(f"Prediction failed: {str(e)}", icon=":material/error:")
