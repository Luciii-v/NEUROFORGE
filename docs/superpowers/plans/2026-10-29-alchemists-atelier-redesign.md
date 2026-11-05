# NEUROFORGE — "The Alchemist’s Atelier" Redesign Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform NEUROFORGE from a generic dark SaaS layout into a high-craft Awwwards-caliber standalone web application ("The Alchemist's Atelier") featuring an expressive animated companion ("Forge"), tactile walnut/amber/sage aesthetics, and real-time streaming for "Build. Mutate. Ship.", preserving the iconic cinematic opening sequence.

**Architecture:** 
- **Backend**: Lightweight FastAPI server (`server.py`) wrapping existing PyTorch modules ([`neuroforge_core.py`](file:///Users/mahathavivaanveer/Developer/NEUROFORGE/neuroforge_core.py), [`architect.py`](file:///Users/mahathavivaanveer/Developer/NEUROFORGE/architect.py)) with WebSocket & SSE endpoints for live mutation streaming and model inference.
- **Frontend**: High-craft React + Vite + Tailwind CSS application (`frontend/`) featuring fluid animations, custom tactile controls, vacuum-tube gauges, an interactive lineage board, and an animated SVG companion ("Forge").

**Tech Stack:** FastAPI, Uvicorn, Python 3.12, PyTorch, React, Vite, Tailwind CSS, Lucide Icons, Canvas/SVG animations.

**Spec:** [Design Document: The Alchemist's Atelier](#design-document-neuroforge--the-alchemists-atelier)

---

## Global Constraints
- **Preserve Core PyTorch Backend**: Do not break or alter the training/evaluation logic in [`neuroforge_core.py`](file:///Users/mahathavivaanveer/Developer/NEUROFORGE/neuroforge_core.py) or mutation loop in [`architect.py`](file:///Users/mahathavivaanveer/Developer/NEUROFORGE/architect.py).
- **Cinematic Opening Preserved**: The animated wordmark ("NEUROFORGE"), orbital rings, scanline sweep, and perspective grid must remain the grand entry sequence.
- **"Build. Mutate. Ship." Narrative**: The entire studio layout must be anchored around these three chambers.
- **Warm Atelier Palette**: Dominated by deep charcoal/walnut (`#0c0b0d`, `#161312`), sage green (`#789d78`), amber glow (`#f59e0b`), royal amethyst (`#8b5cf6`), and electric cyan (`#06b6d4`).

---

## Tasks

- [ ] **Task 1: Build the FastAPI Backend Engine (`server.py`)**
  - [ ] Implement REST endpoints: `/api/status`, `/api/dataset/load`, `/api/dataset/upload-zip`, `/api/model/active`, `/api/predict`.
  - [ ] Implement WebSocket endpoint `/ws/search` that executes [`architect.run_architect_search`](file:///Users/mahathavivaanveer/Developer/NEUROFORGE/architect.py) and streams candidate evaluations in real time.
  - [ ] Verify with an automated curl / python test script.

- [ ] **Task 2: Scaffold the High-Craft Vite + React Frontend (`frontend/`)**
  - [ ] Initialize Vite React template with Tailwind CSS in `frontend/`.
  - [ ] Configure custom color tokens (walnut, sage, amber, amethyst, cyan, charcoal) and luxury typography in `tailwind.config.js`.
  - [ ] Port the exact cinematic splash sequence (`CinematicSplash.jsx`) with letter stagger, orbital rings, and audio/visual studio tagline.
  - [ ] Test that the splash plays smoothly and transitions into the studio.

- [ ] **Task 3: Build the "Forge" Companion Component (`ForgeCompanion.jsx`)**
  - [ ] Create responsive SVG avatar with expressive eye/tool animations (curious magnifying loupe, welding goggles with glowing sparks, victorious beaker).
  - [ ] Build speech bubble system that reflects backend pipeline states (idle, dataset inspected, mutating, victory).
  - [ ] Wire up conversational speech hints for user actions.

- [ ] **Task 4: Build the 3 Chambers ("Build. Mutate. Ship.")**
  - [ ] **Chamber 1 (BUILD)**: Dataset directory input, ZIP drag-and-drop, class badges, and split preview.
  - [ ] **Chamber 2 (MUTATE)**: Real-time search launcher with glowing amber controls, live candidate feed, and the interactive Lineage Tree where parent-child mutations pulse with amethyst energy.
  - [ ] **Chamber 3 (SHIP)**: Vacuum-tube analog confidence gauges, confusion matrix / performance radar, and live single-image classification with instant alchemical liquid meters.

- [ ] **Task 5: End-to-End Integration, Verification & Launch Script**
  - [ ] Connect frontend API client to the FastAPI backend.
  - [ ] Create a single launch script `run_studio.sh` that boots both backend and frontend.
  - [ ] Test with the held-out fruit image and existing model checkpoints to verify 100% feature parity and high-craft aesthetic.
