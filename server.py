import asyncio
import json
import os
import shutil
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from PIL import Image
import io

import neuroforge_core
import architect

from fastapi.staticfiles import StaticFiles

PROJECT_ROOT = Path(__file__).resolve().parent
WEIGHTS_PATH = PROJECT_ROOT / "best_model_weights.pth"
CONFIG_PATH = PROJECT_ROOT / "best_model_config.json"
METADATA_PATH = PROJECT_ROOT / "best_model_metadata.json"
HISTORY_PATH = PROJECT_ROOT / "search_history.json"
MANUAL_BASELINE_PATH = PROJECT_ROOT / "manual_baseline.json"
RESNET_WEIGHTS_PATH = PROJECT_ROOT / "resnet18_comparison_weights.pth"
RESNET_METADATA_PATH = PROJECT_ROOT / "resnet18_comparison_metadata.json"
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"

app = FastAPI(title="NEUROFORGE Studio Engine", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "neuroforge-studio"}

# Global in-memory workspace state
workspace_state = {
    "dataset_dir": None,
    "dataset_metadata": None,
    "dataset_owned": False,
    "model_trained": False,
    "search_in_progress": False,
    "current_run_id": None,
}


def load_json(path: Path) -> Optional[Any]:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def get_checkpoint_status() -> Dict[str, Any]:
    has_weights = WEIGHTS_PATH.exists()
    config = load_json(CONFIG_PATH)
    metadata = load_json(METADATA_PATH) or {}
    history = load_json(HISTORY_PATH) or []
    
    is_ready = bool(has_weights and config and metadata.get("training_complete"))
    
    return {
        "ready": is_ready,
        "has_weights": has_weights,
        "config": config,
        "metadata": metadata,
        "history_count": len(history),
        "device": str(neuroforge_core.device),
        "best_test_metrics": metadata.get("held_out_test_metrics", {}),
    }


@app.get("/api/status")
def get_status():
    ckpt = get_checkpoint_status()
    return {
        "status": "online",
        "device": str(neuroforge_core.device).upper(),
        "workspace": {
            "dataset_dir": workspace_state["dataset_dir"],
            "dataset_metadata": workspace_state["dataset_metadata"],
            "dataset_owned": workspace_state["dataset_owned"],
            "model_trained": ckpt["ready"] or workspace_state["model_trained"],
            "search_in_progress": workspace_state["search_in_progress"],
        },
        "checkpoint": ckpt,
    }


class LoadDatasetRequest(BaseModel):
    path: str


@app.post("/api/dataset/load")
def load_dataset(req: LoadDatasetRequest):
    p = Path(req.path).expanduser().resolve()
    if not p.exists() or not p.is_dir():
        raise HTTPException(status_code=400, detail=f"Directory not found: {req.path}")
    
    try:
        meta = neuroforge_core.get_dataset_metadata(str(p))
        workspace_state["dataset_dir"] = str(p)
        workspace_state["dataset_metadata"] = meta
        workspace_state["dataset_owned"] = False
        return {"success": True, "metadata": meta}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to inspect dataset: {str(e)}")


@app.post("/api/dataset/upload-zip")
async def upload_zip(file: UploadFile = File(...)):
    if not file.filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only .zip files are supported.")
    
    target_dir = PROJECT_ROOT / "datasets" / Path(file.filename).stem
    target_dir.mkdir(parents=True, exist_ok=True)
    
    temp_zip = target_dir / "temp_upload.zip"
    try:
        with open(temp_zip, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        with zipfile.ZipFile(temp_zip, "r") as zip_ref:
            zip_ref.extractall(target_dir)
        
        temp_zip.unlink(missing_ok=True)
        
        meta = neuroforge_core.get_dataset_metadata(str(target_dir))
        workspace_state["dataset_dir"] = str(target_dir)
        workspace_state["dataset_metadata"] = meta
        workspace_state["dataset_owned"] = True
        return {"success": True, "metadata": meta, "extracted_to": str(target_dir)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to extract and inspect zip: {str(e)}")


class UnloadDatasetRequest(BaseModel):
    delete_uploaded: bool = False


@app.post("/api/dataset/unload")
def unload_dataset(req: UnloadDatasetRequest):
    """Clear the active workspace dataset; only remove app-owned ZIP extracts."""
    dataset_dir = workspace_state.get("dataset_dir")
    was_owned = bool(workspace_state.get("dataset_owned"))
    deleted = False

    if req.delete_uploaded and was_owned and dataset_dir:
        dataset_path = Path(dataset_dir).resolve()
        managed_root = (PROJECT_ROOT / "datasets").resolve()
        if managed_root == dataset_path or managed_root not in dataset_path.parents:
            raise HTTPException(status_code=400, detail="Only uploaded datasets managed inside the local datasets folder can be deleted.")
        shutil.rmtree(dataset_path, ignore_errors=False)
        deleted = True

    workspace_state["dataset_dir"] = None
    workspace_state["dataset_metadata"] = None
    workspace_state["dataset_owned"] = False
    workspace_state["model_trained"] = False
    return {"success": True, "deleted": deleted, "unmounted": True}


@app.get("/api/history")
def get_history():
    history = load_json(HISTORY_PATH) or []
    metadata = load_json(METADATA_PATH) or {}
    config = load_json(CONFIG_PATH) or {}
    return {
        "history": history,
        "metadata": metadata,
        "config": config,
    }


COMPARISON_METRICS = ("accuracy", "precision", "recall", "f1", "mse", "rmse")


def _metric_bundle(metrics):
    return {key: metrics.get(key) for key in COMPARISON_METRICS}


def _active_dataset_dir():
    if workspace_state["dataset_dir"]:
        return workspace_state["dataset_dir"]
    metadata = load_json(METADATA_PATH) or {}
    return metadata.get("dataset", {}).get("dataset_root")


def _dataset_matches_model(model_metadata, dataset_metadata, dataset_dir):
    model_root = (model_metadata.get("dataset") or {}).get("dataset_root")
    active_root = (dataset_metadata or {}).get("dataset_root") or dataset_dir
    if not model_root or not active_root:
        return False
    try:
        return os.path.realpath(str(model_root)) == os.path.realpath(str(active_root))
    except (OSError, TypeError):
        return str(model_root) == str(active_root)


def _evaluate_saved_checkpoint(metadata, config, dataset_dir, weights_path, use_transfer_learning=False):
    metrics = dict(metadata.get("held_out_test_metrics") or {})
    if metrics.get("mse") is not None and metrics.get("rmse") is not None:
        return metrics
    if not (config and dataset_dir and Path(weights_path).exists()):
        return metrics
    try:
        evaluated = neuroforge_core.evaluate_checkpoint(
            config,
            int(metadata.get("num_classes") or len(metadata.get("class_names") or [])),
            dataset_dir=dataset_dir,
            evaluation_split="test",
            use_transfer_learning=use_transfer_learning,
            weights_path=str(weights_path),
            seed=int(metadata.get("seed", 42)),
        )
        metrics.update(evaluated)
    except Exception:
        # A comparison card can still show the persisted classification metrics
        # when an older or mismatched checkpoint cannot be re-evaluated.
        pass
    return metrics


@app.get("/api/comparison")
def get_model_comparison():
    """Return comparable metrics for the mutated CNN, manual CNN and ResNet-18."""
    dataset_dir = _active_dataset_dir()
    metadata = load_json(METADATA_PATH) or {}
    config = load_json(CONFIG_PATH) or {}
    dataset_metadata = workspace_state["dataset_metadata"] or metadata.get("dataset")

    dynamic_matches = _dataset_matches_model(metadata, dataset_metadata, dataset_dir)
    dynamic_metrics = _evaluate_saved_checkpoint(
        metadata,
        config,
        dataset_dir,
        WEIGHTS_PATH,
        use_transfer_learning=False,
    ) if dynamic_matches else {}

    manual = load_json(MANUAL_BASELINE_PATH) or metadata.get("manual_baseline") or {}
    manual_metrics = dict(manual)
    if dataset_dir:
        try:
            split_metadata = neuroforge_core.get_dataset_split_metadata(
                dataset_dir,
                seed=int(metadata.get("seed", 42)),
            )
            manual_metrics.update(neuroforge_core.evaluate_candidate(
                architect.MANUAL_BASELINE_CONFIG,
                epochs=int(metadata.get("proxy_epochs", 3)),
                dataset_dir=dataset_dir,
                seed=int(metadata.get("seed", 42)),
                split_indices=split_metadata,
            ))
        except Exception:
            pass

    resnet_metadata = load_json(RESNET_METADATA_PATH) or {}
    resnet_matches = _dataset_matches_model(resnet_metadata, dataset_metadata, dataset_dir)
    resnet_metrics = _evaluate_saved_checkpoint(
        resnet_metadata,
        config,
        dataset_dir,
        RESNET_WEIGHTS_PATH,
        use_transfer_learning=True,
    ) if resnet_matches else {}

    return {
        "success": True,
        "dataset": dataset_metadata,
        "metric_definitions": {
            "accuracy": "Percent of correctly classified samples.",
            "precision": "Macro precision across the class vocabulary, in percent.",
            "recall": "Macro recall across the class vocabulary, in percent.",
            "f1": "Macro F1 across the class vocabulary, in percent.",
            "mse": "Mean squared error between the probability vector and one-hot target; lower is better.",
            "rmse": "Square root of probability-vector MSE; lower is better.",
        },
        "models": [
            {
                "id": "dynamic_cnn",
                "label": "Selected DynamicCNN",
                "kind": "mutation winner",
                "status": "ready" if dynamic_metrics else ("stale" if metadata.get("training_complete") else "unavailable"),
                "metrics": _metric_bundle(dynamic_metrics),
                "config": config,
                "run_id": metadata.get("run_id"),
                "budget": f"{metadata.get('epochs', '—')} final epochs",
                "note": "Architecture selected by mutation search; final checkpoint is used by live inference." if dynamic_matches else "Saved checkpoint belongs to a different dataset. Run a search or training cycle for the active dataset before comparing it.",
            },
            {
                "id": "manual_cnn",
                "label": "Manual CNN baseline",
                "kind": "fixed reference",
                "status": "ready" if manual_metrics else "unavailable",
                "metrics": _metric_bundle(manual_metrics),
                "config": manual.get("config", architect.MANUAL_BASELINE_CONFIG),
                "run_id": manual.get("run_id"),
                "budget": f"{metadata.get('proxy_epochs', '—')} proxy epochs",
                "note": "Fixed human-designed CNN evaluated under the proxy search budget.",
            },
            {
                "id": "resnet18",
                "label": "ResNet-18 transfer baseline",
                "kind": "comparison baseline",
                "status": "ready" if resnet_metadata.get("training_complete") and resnet_metrics else ("stale" if resnet_metadata.get("training_complete") else "not_run"),
                "metrics": _metric_bundle(resnet_metrics),
                "config": {"backbone": "ResNet-18", "pretrained": True, "trainable_head": True},
                "run_id": resnet_metadata.get("run_id"),
                "budget": f"{resnet_metadata.get('epochs', '—')} final epochs" if resnet_metadata else "not run",
                "note": "ImageNet-pretrained frozen backbone with a trainable replacement classifier head." if resnet_matches else "Saved ResNet-18 checkpoint belongs to a different dataset.",
            },
        ],
    }


class FullTrainRequest(BaseModel):
    epochs: int = 15


@app.post("/api/train/full")
def run_full_training(req: FullTrainRequest):
    if req.epochs < 1 or req.epochs > 100:
        raise HTTPException(status_code=400, detail="epochs must be between 1 and 100")
    dataset_dir = workspace_state["dataset_dir"]
    if not dataset_dir:
        meta = load_json(METADATA_PATH) or {}
        dataset_dir = meta.get("dataset", {}).get("dataset_root")
        if not dataset_dir:
            raise HTTPException(status_code=400, detail="No dataset loaded to train on.")
    
    config = load_json(CONFIG_PATH)
    if not config:
        raise HTTPException(status_code=400, detail="No active architecture configuration found to train.")
    
    try:
        result = architect.train_final_model(
            config=config,
            dataset_dir=dataset_dir,
            epochs=req.epochs,
            output_weights_path=str(WEIGHTS_PATH),
            output_metadata_path=str(METADATA_PATH),
        )
        workspace_state["model_trained"] = True
        return {"success": True, "metrics": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Full training failed: {str(e)}")


class ResNetTrainRequest(BaseModel):
    epochs: int = 20


@app.post("/api/train/resnet")
def run_resnet_training(req: ResNetTrainRequest):
    if req.epochs < 1 or req.epochs > 100:
        raise HTTPException(status_code=400, detail="epochs must be between 1 and 100")

    dataset_dir = _active_dataset_dir()
    if not dataset_dir:
        raise HTTPException(status_code=400, detail="No dataset loaded to compare with ResNet-18.")

    try:
        dataset_metadata = neuroforge_core.get_dataset_metadata(dataset_dir)
        if dataset_metadata.get("modality") != "image":
            raise HTTPException(status_code=400, detail="ResNet-18 comparison is available for image datasets only.")
        config = load_json(CONFIG_PATH) or architect.MANUAL_BASELINE_CONFIG
        result = architect.train_final_model(
            config=config,
            dataset_dir=dataset_dir,
            epochs=req.epochs,
            output_weights_path=str(RESNET_WEIGHTS_PATH),
            output_metadata_path=str(RESNET_METADATA_PATH),
            use_transfer_learning=True,
        )
        resnet_metadata = load_json(RESNET_METADATA_PATH) or {}
        resnet_metadata.update({
            "model_type": "resnet18_transfer_baseline",
            "architecture": "ResNet-18",
            "dataset": dataset_metadata,
            "comparison_role": "fixed_transfer_learning_baseline",
        })
        with open(RESNET_METADATA_PATH, "w") as handle:
            json.dump(resnet_metadata, handle, indent=2)
        return {
            "success": True,
            "model": "ResNet-18",
            "metrics": result,
            "comparison": get_model_comparison(),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ResNet-18 comparison failed: {str(e)}")


@app.post("/api/predict")
async def predict_sample(file: UploadFile = File(...)):
    if not WEIGHTS_PATH.exists() or not CONFIG_PATH.exists():
        raise HTTPException(status_code=400, detail="Trained model checkpoint not found.")

    config = load_json(CONFIG_PATH)
    metadata = load_json(METADATA_PATH) or {}
    dataset_dir = workspace_state["dataset_dir"] or metadata.get("dataset", {}).get("dataset_root")
    dataset_metadata = workspace_state.get("dataset_metadata")
    if not dataset_metadata and dataset_dir:
        try:
            dataset_metadata = neuroforge_core.get_dataset_metadata(dataset_dir)
        except Exception:
            dataset_metadata = None

    modality = (dataset_metadata or metadata.get("dataset") or {}).get("modality", "image")
    class_names = (dataset_metadata or {}).get("class_names") or metadata.get("class_names") or metadata.get("dataset", {}).get("class_names")
    num_classes = (dataset_metadata or {}).get("num_classes") or metadata.get("num_classes") or (len(class_names) if class_names else 2)
    if not class_names:
        class_names = [f"Class_{i}" for i in range(num_classes)]

    try:
        contents = await file.read()
        start_time = time.perf_counter()
        if modality == "audio":
            if not file.filename.lower().endswith(".wav"):
                raise ValueError("Audio inference currently accepts 16-bit PCM WAV files.")
            probs, feature_shape = neuroforge_core.predict_audio_distribution(
                contents,
                config,
                num_classes=num_classes,
                weights_path=str(WEIGHTS_PATH),
                dataset_dir=dataset_dir,
            )
            probs = probs.numpy()
        else:
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            model = neuroforge_core._build_model(config, num_classes=num_classes, dataset_dir=dataset_dir)
            state_dict = torch.load(WEIGHTS_PATH, map_location=neuroforge_core.device, weights_only=True)
            model.load_state_dict(state_dict)
            model.to(neuroforge_core.device)
            model.eval()
            transform = neuroforge_core._image_transform(dataset_dir=dataset_dir, train=False)
            input_tensor = transform(image).unsqueeze(0).to(neuroforge_core.device)
            with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
        latency_ms = (time.perf_counter() - start_time) * 1000

        ranked_predictions = []
        for idx, prob in enumerate(probs):
            name = class_names[idx] if idx < len(class_names) else f"Class_{idx}"
            ranked_predictions.append({
                "class_index": idx,
                "class_name": name,
                "confidence": float(prob),
                "percentage": round(float(prob) * 100, 2),
            })
        ranked_predictions.sort(key=lambda x: x["confidence"], reverse=True)
        
        top_pred = ranked_predictions[0]
        response = {
            "success": True,
            "modality": modality,
            "prediction": top_pred["class_name"],
            "confidence": top_pred["confidence"],
            "percentage": top_pred["percentage"],
            "latency_ms": round(latency_ms, 2),
            "all_classes": ranked_predictions,
        }
        if modality == "audio":
            response.update({
                "feature_shape": feature_shape,
                "preprocessing": "16 kHz mono · 4 s · 64-band log-mel",
            })
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@app.websocket("/ws/search")
async def websocket_search(websocket: WebSocket):
    await websocket.accept()
    workspace_state["search_in_progress"] = True
    
    try:
        init_data = await websocket.receive_json()
        generations = init_data.get("generations", 3)
        population = init_data.get("population", 4)
        proxy_epochs = init_data.get("proxy_epochs", 2)
        dataset_dir = workspace_state["dataset_dir"] or init_data.get("dataset_dir")
        
        if not dataset_dir:
            await websocket.send_json({"type": "error", "message": "No dataset specified or loaded."})
            return
        
        await websocket.send_json({
            "type": "started",
            "message": f"Starting Forge Search: {generations} gens × {population} candidates, {proxy_epochs} proxy epochs",
            "config": {"generations": generations, "population": population, "proxy_epochs": proxy_epochs}
        })
        
        # Define callback to stream candidate completions over websocket
        loop = asyncio.get_event_loop()
        
        def on_candidate_evaluated(candidate_data):
            asyncio.run_coroutine_threadsafe(
                websocket.send_json({
                    "type": "candidate_evaluated",
                    "data": candidate_data
                }),
                loop
            )
            
        def on_generation_completed(gen_idx, generation_summary):
            asyncio.run_coroutine_threadsafe(
                websocket.send_json({
                    "type": "generation_completed",
                    "generation": gen_idx,
                    "summary": generation_summary
                }),
                loop
            )
        
        # Run search in background thread so websocket event loop is unblocked
        search_result = await loop.run_in_executor(
            None,
            lambda: architect.run_architect_search(
                dataset_dir=dataset_dir,
                generations=generations,
                population_size=population,
                proxy_epochs=proxy_epochs,
                progress_callback=on_candidate_evaluated,
                generation_callback=on_generation_completed,
            )
        )
        
        await websocket.send_json({
            "type": "search_completed",
            "best_candidate_id": search_result.get("best_candidate_id"),
            "best_result": search_result.get("best_result"),
            "manual_baseline": search_result.get("manual_baseline"),
        })
        
    except WebSocketDisconnect:
        print("Search WebSocket client disconnected.")
    except Exception as e:
        await websocket.send_json({"type": "error", "message": str(e)})
    finally:
        workspace_state["search_in_progress"] = False


@app.get("/api/samples")
def get_sample_files():
    """Return sample files from the active modality for instant inference."""
    dataset_dir = workspace_state["dataset_dir"]
    if not dataset_dir:
        meta = load_json(METADATA_PATH) or {}
        dataset_dir = meta.get("dataset", {}).get("dataset_root")
    
    if not dataset_dir or not Path(dataset_dir).exists():
        return {"samples": []}

    dataset_metadata = workspace_state.get("dataset_metadata")
    if not dataset_metadata:
        try:
            dataset_metadata = neuroforge_core.get_dataset_metadata(dataset_dir)
        except Exception:
            dataset_metadata = None

    if (dataset_metadata or {}).get("modality") == "audio":
        descriptor = neuroforge_core.audio_pipeline.detect_audio_dataset(dataset_dir)
        if not descriptor:
            return {"samples": []}
        samples = []
        seen_classes = set()
        for path, _, class_name, _ in descriptor["samples"]:
            if class_name in seen_classes:
                continue
            seen_classes.add(class_name)
            samples.append({
                "class_name": class_name,
                "filename": Path(path).name,
                "rel_path": str(Path(path).resolve().relative_to(Path(dataset_dir).resolve())),
                "modality": "audio",
            })
        return {"samples": samples[:6]}

    p = Path(dataset_dir)
    samples = []
    # Search for up to 6 diverse class samples
    for class_dir in sorted(p.iterdir()):
        if class_dir.is_dir() and not class_dir.name.startswith((".", "_")):
            images = [f for f in class_dir.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]]
            if images:
                chosen = images[0]
                samples.append({
                    "class_name": class_dir.name,
                    "filename": chosen.name,
                    "rel_path": f"{class_dir.name}/{chosen.name}",
                    "modality": "image",
                })
    return {"samples": samples[:6]}


@app.get("/api/sample-image")
def get_sample_image(rel_path: str):
    """Serve a sample image file."""
    dataset_dir = workspace_state["dataset_dir"]
    if not dataset_dir:
        meta = load_json(METADATA_PATH) or {}
        dataset_dir = meta.get("dataset", {}).get("dataset_root")
    
    if not dataset_dir:
        raise HTTPException(status_code=404, detail="Dataset directory not configured.")
    
    img_path = (Path(dataset_dir) / rel_path).resolve()
    if not img_path.exists() or not img_path.is_file():
        raise HTTPException(status_code=404, detail="Image not found.")
    
    return FileResponse(img_path)


@app.get("/api/sample-audio")
def get_sample_audio(rel_path: str):
    """Serve a WAV sample from the active audio dataset."""
    dataset_dir = workspace_state["dataset_dir"]
    if not dataset_dir:
        meta = load_json(METADATA_PATH) or {}
        dataset_dir = meta.get("dataset", {}).get("dataset_root")
    if not dataset_dir:
        raise HTTPException(status_code=404, detail="Dataset directory not configured.")

    audio_path = (Path(dataset_dir) / rel_path).resolve()
    dataset_root = Path(dataset_dir).resolve()
    if dataset_root not in audio_path.parents or audio_path.suffix.lower() != ".wav":
        raise HTTPException(status_code=400, detail="Invalid audio sample path.")
    if not audio_path.exists() or not audio_path.is_file():
        raise HTTPException(status_code=404, detail="Audio sample not found.")
    return FileResponse(audio_path, media_type="audio/wav")


@app.get("/api/download/weights")
def download_weights():
    if not WEIGHTS_PATH.exists():
        raise HTTPException(status_code=404, detail="Weights file not found.")
    return FileResponse(WEIGHTS_PATH, filename="best_model_weights.pth")


@app.get("/api/download/config")
def download_config():
    if not CONFIG_PATH.exists():
        raise HTTPException(status_code=404, detail="Config file not found.")
    return FileResponse(CONFIG_PATH, filename="best_model_config.json")


@app.get("/api/download/resnet-weights")
def download_resnet_weights():
    if not RESNET_WEIGHTS_PATH.exists():
        raise HTTPException(status_code=404, detail="ResNet-18 comparison checkpoint has not been trained yet.")
    return FileResponse(RESNET_WEIGHTS_PATH, filename="resnet18_comparison_weights.pth")


if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
