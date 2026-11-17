"""NEUROFORGE Studio API.

FastAPI service that wraps the search (architect.py) and training/inference
core (neuroforge_core.py), and serves the React studio from frontend/dist.

Evaluation contract (see docs/neuroforge-ultimate-plan, section 08):
  * search candidates are ranked on the validation split
  * final training picks its checkpoint on validation
  * the test split is measured once, on the locked checkpoint
Every number this API returns says which split and budget produced it. Models
that have not been trained on the active dataset are reported as "not_run"
with no metrics — never with placeholder figures.
"""

import asyncio
import base64
import io
import json
import os
import shutil
import threading
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

import architect
import neuroforge_core

PROJECT_ROOT = Path(__file__).resolve().parent
WEIGHTS_PATH = PROJECT_ROOT / "best_model_weights.pth"
CONFIG_PATH = PROJECT_ROOT / "best_model_config.json"
METADATA_PATH = PROJECT_ROOT / "best_model_metadata.json"
HISTORY_PATH = PROJECT_ROOT / "search_history.json"
LATEST_SEARCH_PATH = PROJECT_ROOT / "latest_search.json"
MANUAL_BASELINE_PATH = PROJECT_ROOT / "manual_baseline.json"
DATASETS_DIR = PROJECT_ROOT / "datasets"
RUNS_DIR = Path(architect.RUNS_DIR)
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"
MAX_UPLOAD_BYTES = 2 * 1024 ** 3

BASELINE_SPECS = {
    "manual_cnn": {
        "name": "Manual CNN",
        "kind": "hand-designed 2-layer reference",
        "transfer": False,
        "note": "Fixed human-designed CNN trained with the same final-training protocol as the winner.",
    },
    "resnet18": {
        "name": "ResNet-18",
        "kind": "pretrained residual backbone",
        "transfer": True,
        "note": "ImageNet-pretrained, frozen backbone, retrained linear head, 224px input.",
    },
    "mobilenet_v3": {
        "name": "MobileNet-V3 Small",
        "kind": "pretrained edge backbone",
        "transfer": True,
        "note": "Depthwise-separable mobile backbone, frozen, retrained head, 224px input.",
    },
    "shufflenet_v2": {
        "name": "ShuffleNet-V2 0.5x",
        "kind": "pretrained compact backbone",
        "transfer": True,
        "note": "Channel-shuffle compact backbone, frozen, retrained head, 224px input.",
    },
    "efficientnet_b0": {
        "name": "EfficientNet-B0",
        "kind": "pretrained compound-scaled backbone",
        "transfer": True,
        "note": "Compound-scaled backbone, frozen, retrained head, 224px input.",
    },
}


def _baseline_paths(model_key):
    return (
        PROJECT_ROOT / f"{model_key}_comparison_weights.pth",
        PROJECT_ROOT / f"{model_key}_comparison_metadata.json",
    )


app = FastAPI(title="NEUROFORGE Studio Engine", version="3.0.0")

_default_origins = "http://localhost:5173,http://127.0.0.1:5173,http://localhost:8000,http://127.0.0.1:8000"
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.environ.get("NEUROFORGE_CORS_ORIGINS", _default_origins).split(",") if o.strip()],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# ---------------------------------------------------------------------------
# Workspace + job state
# ---------------------------------------------------------------------------

workspace_state: Dict[str, Any] = {
    "dataset_dir": None,
    "dataset_metadata": None,
    "dataset_owned": False,
}

# One heavy job (search or training) at a time: they share the accelerator and
# both write experiment artifacts.
job_lock = threading.Lock()
job_state: Dict[str, Any] = {"kind": None, "label": None, "started_at": None, "progress": [], "result": None, "error": None}


def _begin_job(kind, label):
    if not job_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail=f"Busy: {job_state.get('label') or job_state.get('kind')} is still running.")
    job_state.update({"kind": kind, "label": label, "started_at": time.time(), "progress": [], "result": None, "error": None})


def _end_job(result=None, error=None):
    job_state.update({"kind": None, "result": result, "error": error, "finished_at": time.time()})
    job_lock.release()


def load_json(path: Path) -> Optional[Any]:
    try:
        with open(path, "r") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def _same_path(a, b):
    if not a or not b:
        return False
    try:
        return os.path.realpath(str(a)) == os.path.realpath(str(b))
    except (OSError, TypeError):
        return str(a) == str(b)


def _active_dataset_dir():
    if workspace_state["dataset_dir"]:
        return workspace_state["dataset_dir"]
    return ((load_json(METADATA_PATH) or {}).get("dataset") or {}).get("dataset_root")


def _active_dataset_metadata():
    if workspace_state["dataset_metadata"]:
        return workspace_state["dataset_metadata"]
    dataset_dir = _active_dataset_dir()
    if dataset_dir and Path(dataset_dir).exists():
        try:
            return neuroforge_core.get_dataset_metadata(dataset_dir)
        except Exception:
            return None
    return None


def resolve_active_model() -> Dict[str, Any]:
    """Single source of truth for the checkpoint live inference serves.

    ``trained_config`` in the metadata is written in the same promotion step as
    the weights, so it is authoritative. ``best_model_config.json`` is only a
    fallback for checkpoints produced before that contract existed.
    """
    metadata = load_json(METADATA_PATH) or {}
    config = metadata.get("trained_config") or metadata.get("config") or load_json(CONFIG_PATH)
    ready = bool(WEIGHTS_PATH.exists() and config and metadata.get("training_complete"))
    test = metadata.get("held_out_test_metrics") or {}
    return {
        "ready": ready,
        "config": config,
        "metadata": metadata,
        "protocol_version": metadata.get("schema_version", 1),
        # Schema 1 checkpoints selected epochs on the same split they reported as "test".
        "test_is_independent": metadata.get("schema_version", 1) >= 2,
        "test_metrics": test,
        "validation_metrics": metadata.get("validation_metrics") or {},
    }


# ---------------------------------------------------------------------------
# Status, history, runs
# ---------------------------------------------------------------------------


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "neuroforge-studio"}


@app.get("/api/status")
def get_status():
    active = resolve_active_model()
    dataset_metadata = _active_dataset_metadata()
    return {
        "status": "online",
        "device": str(neuroforge_core.device).upper(),
        "workspace": {
            "dataset_dir": _active_dataset_dir(),
            "dataset_metadata": dataset_metadata,
            "dataset_owned": workspace_state["dataset_owned"],
            "dataset_mounted": bool(workspace_state["dataset_dir"]),
            "model_trained": active["ready"],
            "model_matches_dataset": _same_path(
                (active["metadata"].get("dataset") or {}).get("dataset_root"),
                (dataset_metadata or {}).get("dataset_root"),
            ),
        },
        "job": _job_snapshot(),
        "active_model": {k: active[k] for k in ("ready", "config", "protocol_version", "test_is_independent", "test_metrics", "validation_metrics")}
        | {
            "bundle_id": active["metadata"].get("bundle_id"),
            "candidate_id": active["metadata"].get("candidate_id") or active["metadata"].get("best_candidate_id"),
            "class_names": active["metadata"].get("class_names"),
            "parameters": active["metadata"].get("parameters"),
            "best_epoch": active["metadata"].get("best_epoch"),
            "epochs_run": active["metadata"].get("epochs_run"),
            "learning_curve": active["metadata"].get("learning_curve"),
            "training": active["metadata"].get("training"),
            "preprocessing": active["metadata"].get("preprocessing"),
        },
        "latest_search": load_json(LATEST_SEARCH_PATH),
        # Shape read by the studio's hero, ribbon and specimen canvas.
        "checkpoint": {
            "ready": active["ready"],
            "config": active["config"],
            "metadata": active["metadata"],
            "best_test_metrics": active["test_metrics"],
            "history_count": len(load_json(HISTORY_PATH) or []),
        },
    }


def _job_snapshot():
    return {
        "running": job_state["kind"] is not None,
        "kind": job_state["kind"],
        "label": job_state["label"],
        "started_at": job_state["started_at"],
        "progress": job_state["progress"][-200:],
        "result": job_state["result"],
        "error": job_state["error"],
    }


@app.get("/api/job")
def get_job():
    return _job_snapshot()


@app.get("/api/history")
def get_history():
    active = resolve_active_model()
    return {
        "history": load_json(HISTORY_PATH) or [],
        "search": load_json(LATEST_SEARCH_PATH),
        "manual_baseline": load_json(MANUAL_BASELINE_PATH),
        "metadata": active["metadata"],
        "config": active["config"],
    }


@app.get("/api/runs")
def list_runs():
    """Immutable run registry: every search and every trained bundle."""
    runs = []
    if RUNS_DIR.exists():
        for run_dir in sorted(RUNS_DIR.iterdir(), reverse=True):
            if not run_dir.is_dir():
                continue
            summary = load_json(run_dir / "search_summary.json") or {}
            bundles = []
            for bundle in sorted(run_dir.glob("model-*")):
                metrics = load_json(bundle / "metrics.json") or {}
                test = metrics.get("held_out_test_metrics") or {}
                bundles.append({
                    "bundle_id": metrics.get("bundle_id") or f"{run_dir.name}/{bundle.name}",
                    "training_mode": metrics.get("training_mode"),
                    "test_accuracy": test.get("accuracy"),
                    "test_f1": test.get("f1"),
                    "valid_f1": (metrics.get("validation_metrics") or {}).get("f1"),
                    "best_epoch": metrics.get("best_epoch"),
                })
            runs.append({
                "run_id": run_dir.name,
                "kind": "search" if summary else "baseline",
                "dataset_root": (summary.get("dataset") or {}).get("dataset_root"),
                "best_candidate_id": summary.get("best_candidate_id"),
                "best_metrics": summary.get("best_metrics"),
                "candidates": (summary.get("unique_evaluations") or 0) + (summary.get("reused_evaluations") or 0),
                "bundles": bundles,
            })
    return {"runs": runs}


# ---------------------------------------------------------------------------
# Dataset intake
# ---------------------------------------------------------------------------


class LoadDatasetRequest(BaseModel):
    path: str


def _mount(path: Path, owned: bool):
    meta = neuroforge_core.get_dataset_metadata(str(path))
    workspace_state.update({"dataset_dir": str(path), "dataset_metadata": meta, "dataset_owned": owned})
    return meta


@app.post("/api/dataset/load")
def load_dataset(req: LoadDatasetRequest):
    if job_state["kind"]:
        raise HTTPException(status_code=409, detail="Wait for the running job to finish before changing datasets.")
    p = Path(req.path).expanduser().resolve()
    if not p.is_dir():
        raise HTTPException(status_code=400, detail=f"Directory not found: {req.path}")
    try:
        return {"success": True, "metadata": _mount(p, owned=False)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to inspect dataset: {e}")


@app.post("/api/dataset/upload-zip")
async def upload_zip(file: UploadFile = File(...)):
    if job_state["kind"]:
        raise HTTPException(status_code=409, detail="Wait for the running job to finish before changing datasets.")
    if not (file.filename or "").lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only .zip files are supported.")

    target_dir = (DATASETS_DIR / Path(file.filename).stem).resolve()
    if DATASETS_DIR.resolve() not in target_dir.parents:
        raise HTTPException(status_code=400, detail="Invalid archive name.")
    target_dir.mkdir(parents=True, exist_ok=True)
    temp_zip = target_dir / ".upload.zip"
    try:
        written = 0
        with open(temp_zip, "wb") as buffer:
            while chunk := await file.read(1024 * 1024):
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="Archive is larger than 2 GB; mount it by local path instead.")
                buffer.write(chunk)
        with zipfile.ZipFile(temp_zip, "r") as archive:
            for member in archive.namelist():
                destination = (target_dir / member).resolve()
                if destination != target_dir and target_dir not in destination.parents:
                    raise HTTPException(status_code=400, detail=f"Archive entry escapes the dataset folder: {member}")
            archive.extractall(target_dir)
        meta = _mount(target_dir, owned=True)
        return {"success": True, "metadata": meta, "extracted_to": str(target_dir)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to extract and inspect zip: {e}")
    finally:
        temp_zip.unlink(missing_ok=True)


class UnloadDatasetRequest(BaseModel):
    delete_uploaded: bool = False


@app.post("/api/dataset/unload")
def unload_dataset(req: UnloadDatasetRequest):
    """Clear the active workspace dataset; only remove app-owned ZIP extracts."""
    if job_state["kind"]:
        raise HTTPException(status_code=409, detail="Wait for the running job to finish before changing datasets.")
    dataset_dir = workspace_state.get("dataset_dir")
    deleted = False
    if req.delete_uploaded and workspace_state.get("dataset_owned") and dataset_dir:
        dataset_path = Path(dataset_dir).resolve()
        managed_root = DATASETS_DIR.resolve()
        if managed_root == dataset_path or managed_root not in dataset_path.parents:
            raise HTTPException(status_code=400, detail="Only uploaded datasets inside the local datasets folder can be deleted.")
        shutil.rmtree(dataset_path)
        deleted = True
    workspace_state.update({"dataset_dir": None, "dataset_metadata": None, "dataset_owned": False})
    return {"success": True, "deleted": deleted, "unmounted": True}


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

COMPARISON_METRICS = ("accuracy", "precision", "recall", "f1", "mse", "rmse")


def _metric_bundle(metrics):
    if not metrics:
        return None
    return {key: metrics.get(key) for key in COMPARISON_METRICS}


@app.get("/api/comparison")
def get_model_comparison():
    """Comparable metrics for every model trained on the active dataset.

    Rows are only "ready" when the checkpoint was trained on the active dataset.
    Test metrics are only labelled independent for protocol v2 checkpoints.
    """
    dataset_metadata = _active_dataset_metadata()
    dataset_root = (dataset_metadata or {}).get("dataset_root")
    active = resolve_active_model()
    rows = []

    def row_for(model_id, label, kind, metadata, note, ready):
        test = (metadata or {}).get("held_out_test_metrics") or {}
        matches = _same_path((metadata or {}).get("dataset", {}).get("dataset_root"), dataset_root)
        status = "ready" if ready and matches else ("other_dataset" if ready else "not_run")
        independent = (metadata or {}).get("schema_version", 1) >= 2
        return {
            "id": model_id,
            "label": label,
            "kind": kind,
            "status": status,
            "metrics": _metric_bundle(test) if status == "ready" else None,
            "validation": _metric_bundle((metadata or {}).get("validation_metrics")) if status == "ready" else None,
            "split": "test" if independent else "test (legacy: also used for selection)",
            "test_is_independent": independent,
            "sample_count": test.get("sample_count"),
            "parameters": (metadata or {}).get("parameters"),
            "trainable_parameters": (metadata or {}).get("trainable_parameters"),
            "latency_ms": (test.get("inference_latency_per_sample_seconds") or 0) * 1000 or None,
            "budget": f"{metadata.get('epochs_run', metadata.get('epochs'))} epochs (best {metadata.get('best_epoch')})" if status == "ready" and metadata.get("best_epoch") else (f"{metadata.get('epochs')} epochs" if status == "ready" else None),
            "config": (metadata or {}).get("trained_config"),
            "note": note,
        }

    rows.append(row_for(
        "dynamic_cnn", "NEUROFORGE winner", "evolved DynamicCNN", active["metadata"],
        "Architecture selected by mutation search on validation, then fully trained.", active["ready"],
    ))

    for model_key, spec in BASELINE_SPECS.items():
        weights_path, metadata_path = _baseline_paths(model_key)
        meta = load_json(metadata_path) or {}
        ready = bool(weights_path.exists() and meta.get("training_complete"))
        rows.append(row_for(model_key, spec["name"], spec["kind"], meta, spec["note"], ready))

    # The proxy-search baseline is a different (shorter) budget; report it separately.
    proxy_baseline = load_json(MANUAL_BASELINE_PATH)
    latest_search = load_json(LATEST_SEARCH_PATH) or {}
    proxy = None
    if proxy_baseline and _same_path((latest_search.get("dataset") or {}).get("dataset_root"), dataset_root):
        proxy = {
            "manual_baseline": _metric_bundle(proxy_baseline),
            "search_winner": latest_search.get("best_metrics"),
            "budget": latest_search.get("proxy_epochs"),
            "split": proxy_baseline.get("evaluation_split", "valid"),
        }

    return {
        "success": True,
        "dataset": dataset_metadata,
        "metric_definitions": {
            "accuracy": "Percent of correctly classified samples.",
            "precision": "Macro precision across classes, in percent.",
            "recall": "Macro recall across classes, in percent.",
            "f1": "Macro F1 across classes, in percent.",
            "mse": "Mean squared error between probability vector and one-hot target; lower is better.",
            "rmse": "Square root of probability-vector MSE; lower is better.",
        },
        "models": rows,
        "proxy_comparison": proxy,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


class FullTrainRequest(BaseModel):
    epochs: int = Field(30, ge=1, le=200)
    candidate_id: Optional[str] = None


def _record_progress(update):
    job_state["progress"].append(update)


def _resolve_training_config(candidate_id, dataset_dir):
    history = load_json(HISTORY_PATH) or []
    summary = load_json(LATEST_SEARCH_PATH) or {}
    summary_matches = _same_path((summary.get("dataset") or {}).get("dataset_root"), (neuroforge_core.get_dataset_metadata(dataset_dir) or {}).get("dataset_root"))
    if candidate_id:
        for candidate in history:
            if candidate.get("candidate_id") == candidate_id:
                return candidate["config"], candidate_id, summary if summary_matches else None, history
        raise HTTPException(status_code=404, detail=f"Candidate {candidate_id} is not in the latest search.")
    if summary_matches and summary.get("best_config"):
        return summary["best_config"], summary.get("best_candidate_id"), summary, history
    raise HTTPException(status_code=400, detail="Run a search on this dataset first, or pass a candidate_id.")


@app.post("/api/train/full")
def run_full_training(req: FullTrainRequest):
    dataset_dir = _active_dataset_dir()
    if not dataset_dir:
        raise HTTPException(status_code=400, detail="No dataset loaded to train on.")
    config, candidate_id, summary, history = _resolve_training_config(req.candidate_id, dataset_dir)
    _begin_job("train", f"Training {candidate_id or 'selected architecture'}")
    try:
        result = architect.train_final_model(
            config=config,
            dataset_dir=dataset_dir,
            epochs=req.epochs,
            output_weights_path=str(WEIGHTS_PATH),
            output_metadata_path=str(METADATA_PATH),
            output_config_path=str(CONFIG_PATH),
            search_summary=summary,
            search_history=history,
            candidate_id=candidate_id,
            progress_callback=_record_progress,
        )
        _end_job(result=result)
        return {"success": True, "metrics": result}
    except Exception as e:
        _end_job(error=str(e))
        raise HTTPException(status_code=500, detail=f"Full training failed: {e}")


class BaselineTrainRequest(BaseModel):
    model_type: str = "resnet18"
    epochs: int = Field(20, ge=1, le=200)


@app.post("/api/train/baseline")
def run_baseline_training(req: BaselineTrainRequest):
    model_key = req.model_type.lower()
    spec = BASELINE_SPECS.get(model_key)
    if not spec:
        raise HTTPException(status_code=400, detail=f"Unknown baseline '{req.model_type}'. Options: {', '.join(BASELINE_SPECS)}")
    dataset_dir = _active_dataset_dir()
    if not dataset_dir:
        raise HTTPException(status_code=400, detail=f"No dataset loaded to compare with {spec['name']}.")
    dataset_metadata = neuroforge_core.get_dataset_metadata(dataset_dir)
    if spec["transfer"] and dataset_metadata.get("modality") != "image":
        raise HTTPException(status_code=400, detail=f"{spec['name']} is available for image datasets only.")

    weights_path, metadata_path = _baseline_paths(model_key)
    _begin_job("train", f"Training baseline {spec['name']}")
    try:
        result = architect.train_final_model(
            config=architect.MANUAL_BASELINE_CONFIG,
            dataset_dir=dataset_dir,
            epochs=req.epochs,
            output_weights_path=str(weights_path),
            output_metadata_path=str(metadata_path),
            use_transfer_learning=spec["transfer"],
            transfer_model=model_key if spec["transfer"] else "resnet18",
            progress_callback=_record_progress,
        )
        meta = load_json(metadata_path) or {}
        meta.update({"model_type": f"{model_key}_baseline", "architecture": spec["name"], "comparison_role": f"fixed_{model_key}_baseline"})
        architect._write_json(str(metadata_path), meta)
        _end_job(result=result)
        return {"success": True, "model": spec["name"], "model_type": model_key, "metrics": result, "comparison": get_model_comparison()}
    except Exception as e:
        _end_job(error=str(e))
        raise HTTPException(status_code=500, detail=f"{spec['name']} comparison failed: {e}")


class EvaluateConfigRequest(BaseModel):
    config: Dict[str, Any]
    proxy_epochs: int = Field(3, ge=1, le=5)


@app.post("/api/evaluate-config")
def evaluate_config(req: EvaluateConfigRequest):
    """Proxy-train a hand-built design under the search budget (validation split).

    Lets the DNA sequencer compare a human design with the search winner on
    equal terms instead of with a made-up estimate.
    """
    dataset_dir = _active_dataset_dir()
    if not dataset_dir:
        raise HTTPException(status_code=400, detail="Mount a dataset first.")
    try:
        neuroforge_core.validate_config(req.config)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    _begin_job("evaluate", "Evaluating a hand-built design")
    try:
        split = neuroforge_core.get_dataset_split_metadata(dataset_dir)
        metrics = neuroforge_core.evaluate_candidate(req.config, epochs=req.proxy_epochs, dataset_dir=dataset_dir, split_indices=split)
        _end_job()
    except Exception as e:
        _end_job(error=str(e))
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {e}")
    summary = load_json(LATEST_SEARCH_PATH) or {}
    same_dataset = _same_path((summary.get("dataset") or {}).get("dataset_root"), (neuroforge_core.get_dataset_metadata(dataset_dir) or {}).get("dataset_root"))
    return {
        "metrics": {k: metrics[k] for k in ("accuracy", "precision", "recall", "f1", "parameters", "training_time_seconds")},
        "split": "valid",
        "proxy_epochs": req.proxy_epochs,
        "search_winner": summary.get("best_metrics") if same_dataset and summary.get("proxy_epochs") == req.proxy_epochs else None,
    }


# Backwards-compatible aliases for the previous studio build.
@app.post("/api/train/transfer")
def run_transfer_training(req: BaselineTrainRequest):
    return run_baseline_training(req)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def compute_gradcam_overlay(model, input_tensor, target_class_idx: int, target_size: tuple = (256, 256)) -> Optional[str]:
    """Grad-CAM on the last convolution; returns a turbo-coloured RGBA PNG data URL."""
    last_conv = None
    for module in getattr(model, "features", torch.nn.Sequential()).modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = module
    if last_conv is None:
        return None

    captured = {}
    h1 = last_conv.register_forward_hook(lambda m, i, o: captured.__setitem__("activations", o))
    h2 = last_conv.register_full_backward_hook(lambda m, gi, go: captured.__setitem__("gradients", go[0]))
    try:
        x = input_tensor.clone().detach().requires_grad_(True)
        model.zero_grad()
        model(x)[0, target_class_idx].backward()
    finally:
        h1.remove()
        h2.remove()
    if "gradients" not in captured or "activations" not in captured:
        return None

    weights = captured["gradients"].mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * captured["activations"]).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=(target_size[1], target_size[0]), mode="bilinear", align_corners=False)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam_np = cam.squeeze().detach().cpu().numpy()

    r = np.clip(1.5 - np.abs(4.0 * cam_np - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * cam_np - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * cam_np - 1.0), 0.0, 1.0)
    a = np.clip(cam_np * 1.8, 0.15, 0.90)
    rgba = np.stack([r * 255, g * 255, b * 255, a * 255], axis=-1).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("utf-8")


_model_cache: Dict[str, Any] = {"key": None, "model": None}


def _load_active_model(config, num_classes, dataset_dir):
    key = (WEIGHTS_PATH.stat().st_mtime_ns, neuroforge_core.config_hash(config), num_classes)
    if _model_cache["key"] != key:
        model = neuroforge_core._build_model(config, num_classes=num_classes, dataset_dir=dataset_dir)
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=neuroforge_core.device, weights_only=True))
        model.to(neuroforge_core.device).eval()
        _model_cache.update({"key": key, "model": model})
    return _model_cache["model"]


@app.post("/api/predict")
async def predict_sample(file: UploadFile = File(...)):
    active = resolve_active_model()
    if not active["ready"]:
        raise HTTPException(status_code=400, detail="No trained model yet. Run a search and train the winner first.")
    config, metadata = active["config"], active["metadata"]
    dataset_dir = (metadata.get("dataset") or {}).get("dataset_root") or _active_dataset_dir()
    modality = (metadata.get("dataset") or {}).get("modality", "image")
    class_names = metadata.get("class_names") or []
    num_classes = metadata.get("num_classes") or len(class_names)
    if not num_classes:
        raise HTTPException(status_code=400, detail="Checkpoint metadata is missing its class list; retrain the model.")

    contents = await file.read()
    is_wav = (file.filename or "").lower().endswith(".wav")
    if is_wav and modality != "audio":
        raise HTTPException(status_code=400, detail="The active model was trained on images. Mount an audio dataset (ESC-50 / UrbanSound8K) and train it to classify WAV files.")
    if not is_wav and modality == "audio":
        raise HTTPException(status_code=400, detail="The active model was trained on audio. Upload a 16-bit PCM WAV file.")

    try:
        start_time = time.perf_counter()
        saliency_map = None
        feature_shape = None
        if modality == "audio":
            probs, feature_shape = neuroforge_core.predict_audio_distribution(
                contents, config, num_classes=num_classes, weights_path=str(WEIGHTS_PATH), dataset_dir=dataset_dir,
            )
            probs = probs.numpy()
            latency_ms = (time.perf_counter() - start_time) * 1000
        else:
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            model = _load_active_model(config, num_classes, dataset_dir)
            transform = neuroforge_core._image_transform(dataset_dir=dataset_dir, train=False, image_size=neuroforge_core.input_size_for(config))
            input_tensor = transform(image).unsqueeze(0).to(neuroforge_core.device)
            with torch.no_grad():
                probs = torch.softmax(model(input_tensor), dim=1)[0].cpu().numpy()
            latency_ms = (time.perf_counter() - start_time) * 1000
            try:
                saliency_map = compute_gradcam_overlay(model, input_tensor, int(probs.argmax()), target_size=image.size)
            except Exception as cam_error:
                print("Grad-CAM skipped:", cam_error)

        ranked = sorted(
            (
                {
                    "class_index": idx,
                    "class_name": class_names[idx] if idx < len(class_names) else f"Class_{idx}",
                    "confidence": float(prob),
                    "percentage": round(float(prob) * 100, 2),
                }
                for idx, prob in enumerate(probs)
            ),
            key=lambda item: item["confidence"],
            reverse=True,
        )
        top = ranked[0]
        response = {
            "success": True,
            "modality": modality,
            "prediction": top["class_name"],
            "confidence": top["confidence"],
            "percentage": top["percentage"],
            "chance_percentage": round(100 / num_classes, 2),
            "latency_ms": round(latency_ms, 2),
            "all_classes": ranked,
            "saliency_map": saliency_map,
        }
        if feature_shape:
            response.update({"feature_shape": feature_shape, "preprocessing": "16 kHz mono · 4 s · 64-band log-mel"})
        return response
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")


@app.post("/api/audio/predict")
async def predict_audio_sample(file: UploadFile = File(...)):
    return await predict_sample(file)


# ---------------------------------------------------------------------------
# Live search
# ---------------------------------------------------------------------------


@app.websocket("/ws/search")
async def websocket_search(websocket: WebSocket):
    await websocket.accept()
    if not job_lock.acquire(blocking=False):
        await websocket.send_json({"type": "error", "message": f"Busy: {job_state.get('label')} is still running."})
        await websocket.close()
        return
    stop_event = threading.Event()
    loop = asyncio.get_running_loop()

    def send(payload):
        future = asyncio.run_coroutine_threadsafe(websocket.send_json(payload), loop)
        try:
            future.result(timeout=10)
        except Exception:
            stop_event.set()  # client is gone; stop spending compute

    async def listen_for_stop():
        try:
            while True:
                message = await websocket.receive_json()
                if message.get("type") == "stop":
                    stop_event.set()
        except Exception:
            stop_event.set()

    listener = None
    try:
        init_data = await websocket.receive_json()
        generations = int(init_data.get("generations", 3))
        population = int(init_data.get("population", 4))
        proxy_epochs = int(init_data.get("proxy_epochs", 2))
        dataset_dir = _active_dataset_dir()
        if not dataset_dir:
            await websocket.send_json({"type": "error", "message": "Mount a dataset before starting a search."})
            return
        if not (1 <= generations <= 20 and 2 <= population <= 24 and 1 <= proxy_epochs <= 5):
            await websocket.send_json({"type": "error", "message": "Budget out of range: generations 1–20, population 2–24, proxy epochs 1–5."})
            return

        job_state.update({"kind": "search", "label": f"Search {generations}×{population}", "started_at": time.time(), "progress": [], "result": None, "error": None})
        await websocket.send_json({
            "type": "started",
            "config": {"generations": generations, "population": population, "proxy_epochs": proxy_epochs},
            "evaluation_split": "valid",
        })
        listener = asyncio.create_task(listen_for_stop())

        def on_candidate(candidate):
            job_state["progress"].append({"candidate_id": candidate["candidate_id"], "score": candidate["score"]})
            send({"type": "candidate_evaluated", "data": candidate})

        def on_generation(gen_idx, summary):
            send({"type": "generation_completed", "generation": gen_idx, "summary": summary})

        result = await loop.run_in_executor(None, lambda: architect.run_architect_search(
            dataset_dir=dataset_dir,
            generations=generations,
            population_size=population,
            proxy_epochs=proxy_epochs,
            output_dir=str(PROJECT_ROOT),
            progress_callback=on_candidate,
            generation_callback=on_generation,
            should_stop=stop_event.is_set,
        ))
        job_state["result"] = {"run_id": result["run_id"], "best_candidate_id": result["best_candidate_id"]}
        await websocket.send_json({
            "type": "search_completed",
            "run_id": result["run_id"],
            "stopped_early": result["metadata"].get("stopped_early"),
            "best_candidate_id": result["best_candidate_id"],
            "best_result": result["best_result"],
            "manual_baseline": result["manual_baseline"],
        })
    except WebSocketDisconnect:
        stop_event.set()
    except Exception as e:
        job_state["error"] = str(e)
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        if listener:
            listener.cancel()
        job_state.update({"kind": None, "finished_at": time.time()})
        job_lock.release()


# ---------------------------------------------------------------------------
# Samples + downloads
# ---------------------------------------------------------------------------


def _contained_file(root: Path, rel_path: str, suffixes) -> Path:
    """Resolve rel_path under root and refuse anything that escapes it."""
    root = root.resolve()
    candidate = (root / rel_path).resolve()
    if root not in candidate.parents or candidate.suffix.lower() not in suffixes or not candidate.is_file():
        raise HTTPException(status_code=404, detail="Sample not found.")
    return candidate


@app.get("/api/samples")
def get_sample_files():
    """One sample per class (up to 8), preferring the test/valid partitions."""
    dataset_dir = _active_dataset_dir()
    if not dataset_dir or not Path(dataset_dir).exists():
        return {"samples": []}
    dataset_metadata = _active_dataset_metadata() or {}
    root = Path(dataset_dir).resolve()
    samples, seen = [], set()

    def add(path, class_name, modality):
        if class_name in seen or len(samples) >= 8:
            return
        seen.add(class_name)
        samples.append({"class_name": class_name, "filename": Path(path).name, "rel_path": str(Path(path).resolve().relative_to(root)), "modality": modality})

    if dataset_metadata.get("modality") == "audio":
        descriptor = neuroforge_core.audio_pipeline.detect_audio_dataset(dataset_dir)
        for path, _, class_name, _ in (descriptor or {}).get("samples", []):
            add(path, class_name, "audio")
        return {"samples": samples}

    roboflow_root = neuroforge_core.find_roboflow_dataset_root(dataset_dir)
    if roboflow_root:
        for split in ("test", "valid"):
            try:
                class_names, split_samples, _ = neuroforge_core._read_roboflow_labels(roboflow_root / split)
            except Exception:
                continue
            for img_path, label_idx in split_samples:
                add(img_path, class_names[label_idx], "image")
        return {"samples": samples}

    # ImageFolder: draw from the sealed test partition so demos aren't training images.
    image_root = neuroforge_core.find_imagefolder_dataset_root(dataset_dir)
    dataset = neuroforge_core.CleanImageFolder(root=image_root, is_valid_file=neuroforge_core._is_usable_image_file)
    split = neuroforge_core.get_split_indices(len(dataset), labels=dataset.targets)
    for index in split["test"]:
        path, label = dataset.samples[index]
        add(path, dataset.classes[label], "image")
    return {"samples": samples}


@app.get("/api/sample-image")
def get_sample_image(rel_path: str):
    dataset_dir = _active_dataset_dir()
    if not dataset_dir:
        raise HTTPException(status_code=404, detail="Dataset directory not configured.")
    return FileResponse(_contained_file(Path(dataset_dir), rel_path, neuroforge_core.SUPPORTED_IMAGE_EXTENSIONS))


@app.get("/api/sample-audio")
def get_sample_audio(rel_path: str):
    dataset_dir = _active_dataset_dir()
    if not dataset_dir:
        raise HTTPException(status_code=404, detail="Dataset directory not configured.")
    return FileResponse(_contained_file(Path(dataset_dir), rel_path, {".wav"}), media_type="audio/wav")


@app.get("/api/download/weights")
def download_weights():
    if not WEIGHTS_PATH.exists():
        raise HTTPException(status_code=404, detail="Weights file not found.")
    return FileResponse(WEIGHTS_PATH, filename="best_model_weights.pth")


@app.get("/api/download/config")
def download_config():
    active = resolve_active_model()
    if not active["config"]:
        raise HTTPException(status_code=404, detail="Config not found.")
    path = RUNS_DIR / ".export_config.json"
    RUNS_DIR.mkdir(exist_ok=True)
    architect._write_json(str(path), active["config"])
    return FileResponse(path, filename="best_model_config.json")


@app.get("/api/download/bundle")
def download_bundle():
    """Zip of the active model's immutable bundle (config, weights, labels, preprocessing, splits, metrics, lineage)."""
    bundle_dir = resolve_active_model()["metadata"].get("bundle_dir")
    if not bundle_dir or not Path(bundle_dir).is_dir():
        raise HTTPException(status_code=404, detail="The active model predates run bundles. Retrain it to produce one.")
    archive = shutil.make_archive(str(RUNS_DIR / ".export_bundle"), "zip", bundle_dir)
    return FileResponse(archive, filename=f"neuroforge-{Path(bundle_dir).name}.zip")


@app.get("/api/download/transfer-weights")
def download_transfer_weights(model: str = "resnet18"):
    if model.lower() not in BASELINE_SPECS:
        raise HTTPException(status_code=400, detail="Unknown baseline.")
    weights_path, _ = _baseline_paths(model.lower())
    if not weights_path.exists():
        raise HTTPException(status_code=404, detail="That baseline has not been trained yet.")
    return FileResponse(weights_path, filename=weights_path.name)


if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host=os.environ.get("NEUROFORGE_HOST", "127.0.0.1"), port=int(os.environ.get("NEUROFORGE_PORT", 8000)))
