"""Fast regression tests for the evaluation contract and the API.

Everything runs on a tiny synthetic ImageFolder dataset inside a temp dir and
never touches the real checkpoints, runs or datasets in the project folder.

    python -m pytest tests -q
"""

import json
import random
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import architect  # noqa: E402
import neuroforge_core  # noqa: E402

CLASSES = ("alpha", "beta", "gamma")


@pytest.fixture(scope="session")
def tiny_dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("data") / "tiny"
    rng = random.Random(0)
    for class_index, name in enumerate(CLASSES):
        folder = root / name
        folder.mkdir(parents=True)
        for i in range(14):
            # Each class gets a distinct dominant colour so the task is learnable.
            colour = [rng.randint(0, 60) for _ in range(3)]
            colour[class_index] = 200 + rng.randint(0, 55)
            Image.new("RGB", (40, 40), tuple(colour)).save(folder / f"{i:02d}.png")
    # macOS archive junk that must be ignored
    (root / "__MACOSX").mkdir()
    (root / "alpha" / "._00.png").write_bytes(b"junk")
    return root


@pytest.fixture()
def sandbox(tmp_path, monkeypatch):
    """Point every artifact path at a temp folder."""
    import server

    monkeypatch.setattr(architect, "RUNS_DIR", str(tmp_path / "runs"))
    for name, filename in {
        "PROJECT_ROOT": None,
        "WEIGHTS_PATH": "best_model_weights.pth",
        "CONFIG_PATH": "best_model_config.json",
        "METADATA_PATH": "best_model_metadata.json",
        "HISTORY_PATH": "search_history.json",
        "LATEST_SEARCH_PATH": "latest_search.json",
        "MANUAL_BASELINE_PATH": "manual_baseline.json",
        "DATASETS_DIR": "datasets",
        "RUNS_DIR": "runs",
    }.items():
        monkeypatch.setattr(server, name, tmp_path if filename is None else tmp_path / filename)
    monkeypatch.setattr(server, "_baseline_paths", lambda key: (tmp_path / f"{key}_w.pth", tmp_path / f"{key}_m.json"))
    server.workspace_state.update({"dataset_dir": None, "dataset_metadata": None, "dataset_owned": False})
    return tmp_path


# --- splits -----------------------------------------------------------------


def test_split_is_three_way_disjoint_and_complete():
    labels = [i % 4 for i in range(103)]
    split = neuroforge_core.get_split_indices(len(labels), seed=7, labels=labels)
    train, valid, test = set(split["train"]), set(split["valid"]), set(split["test"])
    assert not (train & valid) and not (train & test) and not (valid & test)
    assert train | valid | test == set(range(len(labels)))
    # every class appears in valid and test
    assert {labels[i] for i in valid} == {labels[i] for i in test} == {0, 1, 2, 3}


def test_split_is_deterministic():
    labels = [i % 3 for i in range(60)]
    assert neuroforge_core.get_split_indices(60, seed=1, labels=labels) == neuroforge_core.get_split_indices(60, seed=1, labels=labels)


def test_legacy_two_way_split_is_regenerated(tiny_dataset):
    legacy = {"train": list(range(30)), "test": list(range(30, 42)), "seed": 42}
    assert not neuroforge_core.split_is_current(legacy)
    _, loader, _ = neuroforge_core.get_dataloaders(dataset_dir=str(tiny_dataset), split_indices=legacy, evaluation_split="valid")
    assert len(loader.dataset) < 12  # valid partition, not the leaked legacy "test"


def test_imagefolder_ignores_macos_junk(tiny_dataset):
    meta = neuroforge_core.get_dataset_metadata(str(tiny_dataset))
    assert meta["class_names"] == list(CLASSES)
    assert meta["num_samples"] == 42
    assert sum(meta["splits"].values()) == 42


# --- search space ---------------------------------------------------------------


@pytest.mark.parametrize("seed", range(25))
def test_mutations_always_produce_valid_configs(seed):
    rng = random.Random(seed)
    config = dict(architect.DEFAULT_CONFIG)
    for _ in range(10):
        config, operations = architect.mutate_config(config, rng=rng, return_operations=True)
        neuroforge_core.validate_config(config)
        assert operations


def test_audio_mutations_never_touch_input_size():
    rng = random.Random(3)
    config = dict(architect.DEFAULT_CONFIG)
    for _ in range(200):
        config = architect.mutate_config(config, rng=rng, modality="audio")
        assert "input_size" not in config


@pytest.mark.parametrize("block_type", ["plain", "residual", "depthwise"])
@pytest.mark.parametrize("size", [64, 96, 128])
def test_every_block_type_runs_forward(block_type, size):
    config = {**architect.DEFAULT_CONFIG, "block_type": block_type, "input_size": size, "filters": [16, 32, 32]}
    model = neuroforge_core.DynamicCNN(config, num_classes=5)
    assert model(torch.randn(2, 3, size, size)).shape == (2, 5)


def test_plain_blocks_keep_legacy_state_dict_layout():
    legacy_keys = {"features.0.weight", "features.1.running_mean", "classifier.weight"}
    keys = set(neuroforge_core.DynamicCNN(architect.DEFAULT_CONFIG, num_classes=3).state_dict())
    assert legacy_keys <= keys


# --- search + training protocol -------------------------------------------------


def test_search_never_touches_active_model(sandbox, tiny_dataset):
    sentinel = b"active-weights"
    (sandbox / "best_model_weights.pth").write_bytes(sentinel)
    (sandbox / "best_model_metadata.json").write_text('{"keep": true}')
    result = architect.run_architect_search(
        generations=2, population_size=2, proxy_epochs=1,
        dataset_dir=str(tiny_dataset), output_dir=str(sandbox),
    )
    assert (sandbox / "best_model_weights.pth").read_bytes() == sentinel
    assert json.loads((sandbox / "best_model_metadata.json").read_text()) == {"keep": True}
    run_dir = Path(result["run_dir"])
    for name in ("search_history.json", "search_summary.json", "split_manifest.json", "manual_baseline.json", "selected_config.json"):
        assert (run_dir / name).exists(), name
    assert all(c["evaluation_split"] == "valid" for c in result["history"])


def test_search_can_be_stopped(sandbox, tiny_dataset):
    calls = {"n": 0}

    def stop_after_first_generation():
        calls["n"] += 1
        return calls["n"] > 1

    result = architect.run_architect_search(
        generations=5, population_size=2, proxy_epochs=1,
        dataset_dir=str(tiny_dataset), output_dir=str(sandbox), should_stop=stop_after_first_generation,
    )
    assert result["metadata"]["stopped_early"] is True
    assert result["metadata"]["generations_completed"] == 1


def test_final_training_selects_on_valid_and_writes_bundle(sandbox, tiny_dataset):
    search = architect.run_architect_search(generations=1, population_size=2, proxy_epochs=1, dataset_dir=str(tiny_dataset), output_dir=str(sandbox))
    out = architect.train_final_model(
        config=search["best_config"], dataset_dir=str(tiny_dataset), epochs=3,
        output_weights_path=str(sandbox / "best_model_weights.pth"),
        output_metadata_path=str(sandbox / "best_model_metadata.json"),
        output_config_path=str(sandbox / "best_model_config.json"),
        search_summary=search["metadata"], search_history=search["history"],
    )
    bundle = Path(out["bundle_dir"])
    for name in ("config.json", "weights.pth", "labels.json", "preprocessing.json", "split_manifest.json", "metrics.json", "lineage.json", "README.md"):
        assert (bundle / name).exists(), name
    meta = json.loads((sandbox / "best_model_metadata.json").read_text())
    assert meta["schema_version"] == 2 and meta["training_complete"]
    assert meta["trained_config"] == search["best_config"]
    assert meta["held_out_test_metrics"]["split"] == "test"
    assert "validation" in meta["training"]["selection_rule"]
    assert json.loads((bundle / "lineage.json").read_text())[-1]["candidate_id"] == search["best_candidate_id"]


# --- API -----------------------------------------------------------------------


@pytest.fixture()
def client(sandbox):
    from fastapi.testclient import TestClient

    import server

    return TestClient(server.app)


def test_sample_image_rejects_path_traversal(client, tiny_dataset):
    client.post("/api/dataset/load", json={"path": str(tiny_dataset)})
    assert client.get("/api/sample-image", params={"rel_path": "../../../../etc/hosts"}).status_code == 404
    assert client.get("/api/sample-image", params={"rel_path": "alpha/00.png"}).status_code == 200


def test_comparison_has_no_placeholder_metrics(client, tiny_dataset):
    client.post("/api/dataset/load", json={"path": str(tiny_dataset)})
    rows = client.get("/api/comparison").json()["models"]
    assert all(row["status"] == "not_run" and row["metrics"] is None for row in rows)
    assert not any("vit" in row["id"] for row in rows)


def test_full_flow_search_train_predict(client, tiny_dataset):
    assert client.post("/api/dataset/load", json={"path": str(tiny_dataset)}).status_code == 200
    with client.websocket_connect("/ws/search") as ws:
        ws.send_json({"generations": 1, "population": 2, "proxy_epochs": 1})
        while (message := ws.receive_json())["type"] != "search_completed":
            assert message["type"] != "error", message
    response = client.post("/api/train/full", json={"epochs": 2})
    assert response.status_code == 200, response.text
    status = client.get("/api/status").json()
    assert status["active_model"]["ready"] and status["active_model"]["test_is_independent"]

    samples = client.get("/api/samples").json()["samples"]
    assert {s["class_name"] for s in samples} == set(CLASSES)
    image = client.get("/api/sample-image", params={"rel_path": samples[0]["rel_path"]}).content
    prediction = client.post("/api/predict", files={"file": ("x.png", image, "image/png")}).json()
    assert prediction["prediction"] in CLASSES
    assert abs(sum(c["confidence"] for c in prediction["all_classes"]) - 1) < 1e-3

    rows = {row["id"]: row for row in client.get("/api/comparison").json()["models"]}
    assert rows["dynamic_cnn"]["status"] == "ready" and rows["dynamic_cnn"]["test_is_independent"]
    assert client.get("/api/download/bundle").status_code == 200


def test_training_requires_a_search_for_this_dataset(client, tiny_dataset):
    client.post("/api/dataset/load", json={"path": str(tiny_dataset)})
    assert client.post("/api/train/full", json={"epochs": 1}).status_code == 400


def test_evaluate_config_trains_a_hand_built_design(client, tiny_dataset):
    client.post("/api/dataset/load", json={"path": str(tiny_dataset)})
    config = {"num_layers": 2, "filters": [16, 32], "kernel_sizes": [3, 3], "batch_norm": True, "activation": "ReLU", "pooling": True, "dropout": 0.0}
    response = client.post("/api/evaluate-config", json={"config": config, "proxy_epochs": 1})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["split"] == "valid" and 0 <= body["metrics"]["accuracy"] <= 100
    bad = client.post("/api/evaluate-config", json={"config": {**config, "kernel_sizes": [7, 3]}})
    assert bad.status_code == 400
