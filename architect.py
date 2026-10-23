import copy
import json
import math
import os
import random
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


def mutate_config(base_config, rng=None, return_operations=False):
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
    temporary_path = f"{path}.tmp"
    with open(temporary_path, "w") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(temporary_path, path)


def _fitness(metrics):
    """V1 multi-metric objective; all classification metrics are percentages."""
    return (
        0.45 * metrics["accuracy"]
        + 0.25 * metrics["precision"]
        + 0.30 * metrics["f1"]
        - metrics["parameters"] / 1_000_000
        - metrics["inference_latency_per_sample_seconds"] * 100
    )


def run_architect_search(
    generations=5,
    population_size=8,
    st_callback=None,
    dataset_dir=None,
    seed=42,
    proxy_epochs=3,
    output_dir=".",
):
    """Run mutation-based NAS and persist an auditable search history."""
    if generations < 1 or population_size < 2:
        raise ValueError("generations must be >= 1 and population_size must be >= 2")
    if proxy_epochs < 1 or proxy_epochs > 5:
        raise ValueError("proxy_epochs must be between 1 and 5")

    os.makedirs(output_dir, exist_ok=True)
    neuroforge_core.set_random_seed(seed)
    rng = random.Random(seed)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dataset_metadata = neuroforge_core.get_dataset_metadata(dataset_dir)
    split_indices = None
    if dataset_dir:
        split_indices = neuroforge_core.get_image_split_indices(dataset_dir, seed=seed)

    neuroforge_core.validate_config(MANUAL_BASELINE_CONFIG)
    baseline_metrics = evaluate_candidate(
        MANUAL_BASELINE_CONFIG,
        epochs=proxy_epochs,
        dataset_dir=dataset_dir,
        seed=seed,
        split_indices=split_indices,
    )
    manual_baseline = {
        "model_type": "manual_baseline",
        "config": copy.deepcopy(MANUAL_BASELINE_CONFIG),
        **baseline_metrics,
        "score": _fitness(baseline_metrics),
        "status": "evaluated",
    }
    _write_json(os.path.join(output_dir, "manual_baseline.json"), manual_baseline)

    best_config = copy.deepcopy(DEFAULT_CONFIG)
    best_result = None
    history = []
    previous_generation = [{"config": copy.deepcopy(best_config), "candidate_id": None}]

    for generation in range(1, generations + 1):
        if st_callback:
            st_callback({"type": "generation", "current": generation, "total": generations})

        if generation == 1:
            candidates = [{"config": copy.deepcopy(best_config), "parent_id": None, "operations": []}]
            while len(candidates) < population_size:
                child, operations = mutate_config(DEFAULT_CONFIG, rng=rng, return_operations=True)
                candidates.append({"config": child, "parent_id": None, "operations": operations})
        else:
            ranked_parents = sorted(
                previous_generation,
                key=lambda item: item["score"],
                reverse=True,
            )
            parent_pool = ranked_parents[:max(1, math.ceil(population_size / 2))]
            candidates = [{
                "config": copy.deepcopy(ranked_parents[0]["config"]),
                "parent_id": ranked_parents[0]["candidate_id"],
                "operations": [{"field": "elite", "to": "preserved"}],
            }]
            while len(candidates) < population_size:
                parent = rng.choice(parent_pool)
                child, operations = mutate_config(parent["config"], rng=rng, return_operations=True)
                candidates.append({
                    "config": child,
                    "parent_id": parent["candidate_id"],
                    "operations": operations,
                })

        generation_results = []
        for index, candidate in enumerate(candidates, start=1):
            candidate_id = f"{run_id}-G{generation:02d}-C{index:02d}"
            config = candidate["config"]
            if st_callback:
                st_callback({"type": "evaluating", "candidate": index, "total": len(candidates)})

            _write_json(os.path.join(output_dir, "candidate_config.json"), config)
            metrics = evaluate_candidate(
                config,
                epochs=proxy_epochs,
                dataset_dir=dataset_dir,
                seed=seed,
                split_indices=split_indices,
            )
            result = {
                "run_id": run_id,
                "candidate_id": candidate_id,
                "generation": generation,
                "parent_id": candidate["parent_id"],
                "mutation_operations": candidate["operations"],
                "config": copy.deepcopy(config),
                **metrics,
                "score": _fitness(metrics),
                "status": "evaluated",
            }
            generation_results.append(result)
            history.append(result)
            if st_callback:
                st_callback({
                    "type": "result",
                    "candidate": index,
                    "accuracy": result["accuracy"],
                    "precision": result["precision"],
                    "f1": result["f1"],
                    "params": result["parameters"],
                })

        generation_results.sort(key=lambda item: item["score"], reverse=True)
        for rank, result in enumerate(generation_results, start=1):
            result["rank"] = rank

        generation_best = generation_results[0]
        if best_result is None or generation_best["score"] > best_result["score"]:
            best_result = generation_best
            best_config = copy.deepcopy(generation_best["config"])
            _write_json(os.path.join(output_dir, "best_model_config.json"), best_config)
            if st_callback:
                st_callback({
                    "type": "new_best",
                    "gen": generation,
                    "accuracy": generation_best["accuracy"],
                    "precision": generation_best["precision"],
                    "f1": generation_best["f1"],
                    "config": best_config,
                })
        previous_generation = generation_results
        _write_json(os.path.join(output_dir, "search_history.json"), history)

    metadata = {
        "schema_version": 1,
        "run_id": run_id,
        "seed": seed,
        "generations": generations,
        "population_size": population_size,
        "proxy_epochs": proxy_epochs,
        "dataset": dataset_metadata,
        "split_indices": split_indices,
        "objective": "0.45*accuracy + 0.25*precision + 0.30*f1 - params/1e6 - latency_per_sample*100",
        "best_candidate_id": best_result["candidate_id"],
        "best_metrics": {
            key: best_result[key]
            for key in ("accuracy", "precision", "recall", "f1", "parameters", "score")
        },
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
    _write_json(os.path.join(output_dir, "best_model_metadata.json"), metadata)

    if st_callback:
        st_callback({
            "type": "complete",
            "best_accuracy": best_result["accuracy"],
            "best_precision": best_result["precision"],
            "best_f1": best_result["f1"],
            "best_config": best_config,
            "history_count": len(history),
        })
    return {
        "run_id": run_id,
        "best_config": best_config,
        "best_result": best_result,
        "manual_baseline": manual_baseline,
        "history": history,
        "metadata": metadata,
    }


if __name__ == "__main__":
    run_architect_search(generations=5, population_size=8, proxy_epochs=3)
