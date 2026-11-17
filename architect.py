import copy
import json
import math
import os
import random
import shutil
from datetime import datetime, timezone

import neuroforge_core
from neuroforge_core import evaluate_candidate


DEFAULT_CONFIG = {
    "num_layers": 3,
    "filters": [32, 64, 128],
    "kernel_sizes": [3, 3, 3],
    "batch_norm": True,
    "activation": "ReLU",
    "pooling": True,
    "dropout": 0.2,
}

# Fixed topology used as the human-designed reference for every experiment.
MANUAL_BASELINE_CONFIG = {
    "num_layers": 2,
    "filters": [32, 64],
    "kernel_sizes": [3, 3],
    "batch_norm": True,
    "activation": "ReLU",
    "pooling": True,
    "dropout": 0.2,
}

FILTER_CHOICES = [16, 32, 64, 128]
KERNEL_CHOICES = [3, 5]
ACTIVATION_CHOICES = ["ReLU", "GELU", "SiLU"]
BLOCK_CHOICES = ["plain", "residual", "depthwise"]
INPUT_SIZE_CHOICES = [64, 96, 128]

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(PROJECT_ROOT, "runs")


def mutate_config(base_config, rng=None, return_operations=False, modality="image"):
    """Make small, traceable changes to one parent architecture."""
    rng = rng or random
    new_config = copy.deepcopy(base_config)
    operations = []

    if rng.random() < 0.25:
        old_layers = new_config["num_layers"]
        new_layers = max(2, min(5, old_layers + rng.choice([-1, 1])))
        if new_layers != old_layers:
            new_config["num_layers"] = new_layers
            if new_layers > old_layers:
                for _ in range(new_layers - old_layers):
                    new_config["filters"].append(rng.choice(FILTER_CHOICES))
                    new_config["kernel_sizes"].append(rng.choice(KERNEL_CHOICES))
            else:
                new_config["filters"] = new_config["filters"][:new_layers]
                new_config["kernel_sizes"] = new_config["kernel_sizes"][:new_layers]
            operations.append({"field": "num_layers", "from": old_layers, "to": new_layers})

    for index in range(new_config["num_layers"]):
        if rng.random() < 0.35:
            old_value = new_config["filters"][index]
            new_value = rng.choice(FILTER_CHOICES)
            new_config["filters"][index] = new_value
            if new_value != old_value:
                operations.append({"field": f"filters[{index}]", "from": old_value, "to": new_value})
        if rng.random() < 0.25:
            old_value = new_config["kernel_sizes"][index]
            new_value = rng.choice(KERNEL_CHOICES)
            new_config["kernel_sizes"][index] = new_value
            if new_value != old_value:
                operations.append({"field": f"kernel_sizes[{index}]", "from": old_value, "to": new_value})

    if rng.random() < 0.2:
        old_value = new_config["activation"]
        new_value = rng.choice(ACTIVATION_CHOICES)
        new_config["activation"] = new_value
        if new_value != old_value:
            operations.append({"field": "activation", "from": old_value, "to": new_value})

    if rng.random() < 0.25:
        old_value = new_config["dropout"]
        new_value = rng.choice([0.0, 0.1, 0.2, 0.3, 0.4])
        new_config["dropout"] = new_value
        if new_value != old_value:
            operations.append({"field": "dropout", "from": old_value, "to": new_value})

    if rng.random() < 0.1:
        new_config["batch_norm"] = not new_config.get("batch_norm", True)
        operations.append({"field": "batch_norm", "to": new_config["batch_norm"]})

    if rng.random() < 0.1:
        new_config["pooling"] = not new_config.get("pooling", True)
        operations.append({"field": "pooling", "to": new_config["pooling"]})

    if rng.random() < 0.2:
        old_value = new_config.get("block_type", "plain")
        new_value = rng.choice(BLOCK_CHOICES)
        if new_value != old_value:
            new_config["block_type"] = new_value
            operations.append({"field": "block_type", "from": old_value, "to": new_value})

    # Input resolution only applies to images; log-mel audio tensors have a fixed shape.
    if modality == "image" and rng.random() < 0.15:
        old_value = new_config.get("input_size", 64)
        new_value = rng.choice(INPUT_SIZE_CHOICES)
        if new_value != old_value:
            new_config["input_size"] = new_value
            operations.append({"field": "input_size", "from": old_value, "to": new_value})

    # Ensure a child is meaningfully different while keeping mutation controlled.
    if not operations:
        index = rng.randrange(new_config["num_layers"])
        old_value = new_config["filters"][index]
        choices = [value for value in FILTER_CHOICES if value != old_value]
        new_value = rng.choice(choices)
        new_config["filters"][index] = new_value
        operations.append({"field": f"filters[{index}]", "from": old_value, "to": new_value})

    neuroforge_core.validate_config(new_config)
    if return_operations:
        return new_config, operations
    return new_config


def _write_json(path, payload):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary_path = f"{path}.tmp"
    with open(temporary_path, "w") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(temporary_path, path)


def _read_json(path, default=None):
    try:
        with open(path) as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return default


def _fitness(metrics):
    """V1 multi-metric objective on VALIDATION metrics (percentages)."""
    return (
        0.45 * metrics["accuracy"]
        + 0.25 * metrics["precision"]
        + 0.30 * metrics["f1"]
        - metrics["parameters"] / 1_000_000
        - metrics["inference_latency_per_sample_seconds"] * 100
    )


OBJECTIVE = "0.45*accuracy + 0.25*precision + 0.30*f1 - params/1e6 - latency_per_sample*100 (validation split)"


def run_architect_search(
    generations=5,
    population_size=8,
    st_callback=None,
    progress_callback=None,
    generation_callback=None,
    dataset_dir=None,
    seed=42,
    proxy_epochs=3,
    output_dir=".",
    runs_dir=None,
    should_stop=None,
):
    """Run mutation-based NAS and persist an auditable search history.

    Everything is written to ``runs/<run_id>/``. The latest search history and
    baseline are mirrored into ``output_dir`` for the UI, but the active model
    files (best_model_*) are never touched: a search must not invalidate the
    checkpoint that live inference is serving.
    """
    if generations < 1 or population_size < 2:
        raise ValueError("generations must be >= 1 and population_size must be >= 2")
    if proxy_epochs < 1 or proxy_epochs > 5:
        raise ValueError("proxy_epochs must be between 1 and 5")

    neuroforge_core.set_random_seed(seed)
    rng = random.Random(seed)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = os.path.join(runs_dir or RUNS_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    dataset_metadata = neuroforge_core.get_dataset_metadata(dataset_dir)
    modality = dataset_metadata.get("modality", "image")
    split_indices = neuroforge_core.get_dataset_split_metadata(dataset_dir, seed=seed) if dataset_dir else None
    _write_json(os.path.join(run_dir, "split_manifest.json"), split_indices or {"strategy": "cifar10_official_test_plus_5k_valid", "seed": seed})

    metrics_cache = {}
    cache_hits = 0
    evaluations_run = 0
    stopped_early = False

    neuroforge_core.validate_config(MANUAL_BASELINE_CONFIG)
    evaluations_run += 1
    baseline_metrics = evaluate_candidate(
        MANUAL_BASELINE_CONFIG,
        epochs=proxy_epochs,
        dataset_dir=dataset_dir,
        seed=seed,
        split_indices=split_indices,
    )
    metrics_cache[neuroforge_core.config_hash(MANUAL_BASELINE_CONFIG)] = dict(baseline_metrics)
    manual_baseline = {
        "model_type": "manual_baseline",
        "run_id": run_id,
        "budget": f"{proxy_epochs} proxy epochs",
        "evaluation_split": "valid",
        "config": copy.deepcopy(MANUAL_BASELINE_CONFIG),
        **baseline_metrics,
        "score": _fitness(baseline_metrics),
        "status": "evaluated",
    }
    _write_json(os.path.join(run_dir, "manual_baseline.json"), manual_baseline)

    best_config = copy.deepcopy(DEFAULT_CONFIG)
    best_result = None
    history = []
    previous_generation = []

    for generation in range(1, generations + 1):
        if should_stop and should_stop():
            stopped_early = True
            break
        if st_callback:
            st_callback({"type": "generation", "current": generation, "total": generations})

        if generation == 1:
            candidates = [{"config": copy.deepcopy(DEFAULT_CONFIG), "parent_id": None, "operations": []}]
            while len(candidates) < population_size:
                child, operations = mutate_config(DEFAULT_CONFIG, rng=rng, return_operations=True, modality=modality)
                candidates.append({"config": child, "parent_id": None, "operations": operations})
        else:
            ranked_parents = sorted(previous_generation, key=lambda item: item["score"], reverse=True)
            parent_pool = ranked_parents[:max(1, math.ceil(population_size / 2))]
            candidates = [{
                "config": copy.deepcopy(ranked_parents[0]["config"]),
                "parent_id": ranked_parents[0]["candidate_id"],
                "operations": [{"field": "elite", "to": "preserved"}],
            }]
            while len(candidates) < population_size:
                parent = rng.choice(parent_pool)
                child, operations = mutate_config(parent["config"], rng=rng, return_operations=True, modality=modality)
                candidates.append({"config": child, "parent_id": parent["candidate_id"], "operations": operations})

        generation_results = []
        for index, candidate in enumerate(candidates, start=1):
            candidate_id = f"{run_id}-G{generation:02d}-C{index:02d}"
            config = candidate["config"]
            if st_callback:
                st_callback({"type": "evaluating", "candidate": index, "total": len(candidates)})

            cache_key = neuroforge_core.config_hash(config)
            cache_hit = cache_key in metrics_cache
            if cache_hit:
                cache_hits += 1
                metrics = dict(metrics_cache[cache_key])
            else:
                evaluations_run += 1
                metrics = evaluate_candidate(
                    config,
                    epochs=proxy_epochs,
                    dataset_dir=dataset_dir,
                    seed=seed,
                    split_indices=split_indices,
                )
                metrics_cache[cache_key] = dict(metrics)
            result = {
                "run_id": run_id,
                "candidate_id": candidate_id,
                "generation": generation,
                "parent_id": candidate["parent_id"],
                "mutation_operations": candidate["operations"],
                "config": copy.deepcopy(config),
                **metrics,
                "evaluation_split": "valid",
                "score": _fitness(metrics),
                "status": "evaluated",
            }
            if cache_hit:
                result["evaluation_reused"] = True
            generation_results.append(result)
            history.append(result)
            if st_callback:
                st_callback({"type": "result", **{k: result[k] for k in ("candidate_id", "generation", "parent_id", "accuracy", "precision", "f1", "parameters", "score")}, "candidate": index, "params": result["parameters"], "config": copy.deepcopy(config), "mutation_operations": result["mutation_operations"]})
            if progress_callback:
                progress_callback(copy.deepcopy(result))

        generation_results.sort(key=lambda item: item["score"], reverse=True)
        for rank, result in enumerate(generation_results, start=1):
            result["rank"] = rank

        generation_best = generation_results[0]
        if best_result is None or generation_best["score"] > best_result["score"]:
            best_result = generation_best
            best_config = copy.deepcopy(generation_best["config"])
            if st_callback:
                st_callback({"type": "new_best", "gen": generation, "accuracy": generation_best["accuracy"], "precision": generation_best["precision"], "f1": generation_best["f1"], "config": best_config})
        previous_generation = generation_results
        _write_json(os.path.join(run_dir, "search_history.json"), history)
        if generation_callback:
            generation_callback(generation, {
                "best_candidate_id": generation_best["candidate_id"],
                "accuracy": generation_best["accuracy"],
                "precision": generation_best["precision"],
                "recall": generation_best["recall"],
                "f1": generation_best["f1"],
                "score": generation_best["score"],
                "candidate_count": len(generation_results),
            })

    if best_result is None:
        raise RuntimeError("Search was stopped before any candidate was evaluated")

    summary = {
        "schema_version": 2,
        "run_id": run_id,
        "run_dir": run_dir,
        "seed": seed,
        "generations": generations,
        "generations_completed": max(item["generation"] for item in history),
        "stopped_early": stopped_early,
        "population_size": population_size,
        "proxy_epochs": proxy_epochs,
        "unique_evaluations": evaluations_run,
        "reused_evaluations": cache_hits,
        "dataset": dataset_metadata,
        "split_strategy": (split_indices or {}).get("strategy", "cifar10"),
        "evaluation_split": "valid",
        "objective": OBJECTIVE,
        "best_candidate_id": best_result["candidate_id"],
        "best_config": best_config,
        "best_metrics": {key: best_result[key] for key in ("accuracy", "precision", "recall", "f1", "parameters", "score")},
        "best_config_hash": best_result["config_hash"],
        "manual_baseline": manual_baseline,
        "comparison": {
            "accuracy_delta": best_result["accuracy"] - manual_baseline["accuracy"],
            "precision_delta": best_result["precision"] - manual_baseline["precision"],
            "recall_delta": best_result["recall"] - manual_baseline["recall"],
            "f1_delta": best_result["f1"] - manual_baseline["f1"],
            "parameter_delta": best_result["parameters"] - manual_baseline["parameters"],
        },
        "training_mode": "dynamic_cnn_proxy_search",
    }
    _write_json(os.path.join(run_dir, "search_summary.json"), summary)
    _write_json(os.path.join(run_dir, "selected_config.json"), best_config)

    # Mirror the latest search for the UI. These files describe the search,
    # not the deployed model, so overwriting them is safe.
    _write_json(os.path.join(output_dir, "search_history.json"), history)
    _write_json(os.path.join(output_dir, "manual_baseline.json"), manual_baseline)
    _write_json(os.path.join(output_dir, "latest_search.json"), summary)

    if st_callback:
        st_callback({"type": "complete", "best_accuracy": best_result["accuracy"], "best_precision": best_result["precision"], "best_f1": best_result["f1"], "best_config": best_config, "history_count": len(history)})
    return {
        "run_id": run_id,
        "run_dir": run_dir,
        "best_candidate_id": best_result["candidate_id"],
        "best_config": best_config,
        "best_result": best_result,
        "manual_baseline": manual_baseline,
        "history": history,
        "metadata": summary,
    }


def candidate_lineage(history, candidate_id):
    """Walk parent links from a candidate back to its generation-1 ancestor."""
    by_id = {item["candidate_id"]: item for item in history or []}
    chain = []
    current = by_id.get(candidate_id)
    while current is not None and len(chain) < 64:
        chain.append({
            key: current.get(key)
            for key in ("candidate_id", "generation", "parent_id", "mutation_operations", "config", "accuracy", "f1", "score")
        })
        current = by_id.get(current.get("parent_id"))
    return list(reversed(chain))


def _bundle_readme(metadata):
    test = metadata.get("held_out_test_metrics", {})
    valid = metadata.get("validation_metrics", {})
    dataset = metadata.get("dataset", {})
    return "\n".join([
        f"# NEUROFORGE model bundle {metadata.get('bundle_id')}",
        "",
        f"- Dataset: `{dataset.get('dataset_root')}` ({metadata.get('num_classes')} classes, format {dataset.get('format')})",
        f"- Model: {metadata.get('training_mode')} · config hash `{metadata.get('config_hash')}`",
        f"- Training: up to {metadata.get('epochs')} epochs, best epoch {metadata.get('best_epoch')}, seed {metadata.get('seed')}, device {metadata.get('device')}",
        f"- Selection: {metadata.get('training', {}).get('selection_rule')}",
        f"- Validation: accuracy {valid.get('accuracy', 0):.2f}% · macro F1 {valid.get('f1', 0):.2f}%",
        f"- Test (evaluated once, n={test.get('sample_count')}): accuracy {test.get('accuracy', 0):.2f}% · macro F1 {test.get('f1', 0):.2f}%",
        "",
        "Files: `config.json`, `weights.pth`, `labels.json`, `preprocessing.json`, `split_manifest.json`, `metrics.json`, `lineage.json`.",
        "",
    ])


def _atomic_copy(source, destination):
    temporary_path = f"{destination}.tmp"
    shutil.copyfile(source, temporary_path)
    os.replace(temporary_path, destination)


def train_final_model(
    config,
    dataset_dir,
    epochs=30,
    output_weights_path="best_model_weights.pth",
    output_metadata_path="best_model_metadata.json",
    seed=42,
    use_transfer_learning=False,
    transfer_model="resnet18",
    search_summary=None,
    search_history=None,
    candidate_id=None,
    output_config_path=None,
    progress_callback=None,
):
    """Train a locked architecture into an immutable bundle, then promote it.

    The bundle is written under ``runs/<run_id>/model-<timestamp>/``. Only when
    training and the single test evaluation succeed are its weights, config and
    metadata copied over the output paths, so a failed or interrupted run can
    never leave the active checkpoint half-written or mismatched.
    """
    neuroforge_core.validate_config(config)
    run_id = (search_summary or {}).get("run_id") or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    label = transfer_model if use_transfer_learning else "dynamic"
    bundle_dir = os.path.join(RUNS_DIR, run_id, f"model-{label}-{stamp}")
    os.makedirs(bundle_dir, exist_ok=True)
    weights_path = os.path.join(bundle_dir, "weights.pth")
    metrics_path = os.path.join(bundle_dir, "metrics.json")

    split_indices = neuroforge_core.get_dataset_split_metadata(dataset_dir, seed=seed) if dataset_dir else None
    best_validation_accuracy = neuroforge_core.full_train(
        config=config,
        epochs=epochs,
        dataset_dir=dataset_dir,
        use_transfer_learning=use_transfer_learning,
        transfer_model=transfer_model,
        seed=seed,
        weights_path=weights_path,
        metadata_path=metrics_path,
        split_indices=split_indices,
        st_callback=progress_callback,
    )
    metadata = _read_json(metrics_path, {})
    metadata["bundle_id"] = f"{run_id}/{os.path.basename(bundle_dir)}"
    metadata["bundle_dir"] = bundle_dir
    metadata["run_id"] = run_id
    if search_summary and not use_transfer_learning:
        for key in ("objective", "best_candidate_id", "manual_baseline", "comparison", "generations", "population_size", "proxy_epochs", "unique_evaluations", "reused_evaluations"):
            if key in search_summary:
                metadata[key] = search_summary[key]
        metadata["search_best_metrics"] = search_summary.get("best_metrics")
    if candidate_id:
        metadata["candidate_id"] = candidate_id
    lineage = candidate_lineage(search_history, candidate_id or (search_summary or {}).get("best_candidate_id"))

    _write_json(metrics_path, metadata)
    _write_json(os.path.join(bundle_dir, "config.json"), config)
    _write_json(os.path.join(bundle_dir, "labels.json"), metadata.get("class_names", []))
    _write_json(os.path.join(bundle_dir, "preprocessing.json"), metadata.get("preprocessing", {}))
    _write_json(os.path.join(bundle_dir, "split_manifest.json"), split_indices or {})
    _write_json(os.path.join(bundle_dir, "lineage.json"), lineage)
    with open(os.path.join(bundle_dir, "README.md"), "w") as handle:
        handle.write(_bundle_readme(metadata))

    # Promote: weights first, then config, then metadata (metadata marks readiness).
    metadata["weights_path"] = os.path.abspath(output_weights_path)
    _atomic_copy(weights_path, output_weights_path)
    if output_config_path:
        _write_json(output_config_path, config)
    _write_json(output_metadata_path, metadata)

    return {
        "best_validation_accuracy": best_validation_accuracy,
        "validation_metrics": metadata.get("validation_metrics", {}),
        "held_out_test_metrics": metadata.get("held_out_test_metrics", {}),
        "training_complete": metadata.get("training_complete", False),
        "bundle_dir": bundle_dir,
        "epochs_run": metadata.get("epochs_run"),
        "best_epoch": metadata.get("best_epoch"),
    }


if __name__ == "__main__":
    run_architect_search(generations=5, population_size=8, proxy_epochs=3)
