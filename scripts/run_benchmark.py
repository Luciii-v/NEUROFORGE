"""Reproducible end-to-end benchmark: search -> train winner -> train baselines.

Every model uses the same split, seed and final-training protocol
(validation-selected checkpoint, test evaluated once). Results are printed as a
table and saved to runs/<run_id>/benchmark.json.

    python scripts/run_benchmark.py "datasets/crop pest dataset" \
        --generations 5 --population 8 --proxy-epochs 3 --epochs 40 \
        --baselines manual_cnn efficientnet_b0 resnet18 mobilenet_v3 shufflenet_v2
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import architect  # noqa: E402
import neuroforge_core  # noqa: E402

TRANSFER = {"efficientnet_b0", "resnet18", "mobilenet_v3", "shufflenet_v2"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset")
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--proxy-epochs", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--baseline-epochs", type=int, default=None, help="defaults to --epochs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--baselines", nargs="*", default=["manual_cnn", "efficientnet_b0", "resnet18"])
    parser.add_argument("--no-promote", action="store_true", help="keep the winner out of best_model_* (bundle only)")
    args = parser.parse_args()

    dataset_dir = str(neuroforge_core.find_imagefolder_dataset_root(Path(args.dataset).expanduser().resolve()))
    meta = neuroforge_core.get_dataset_metadata(dataset_dir)
    print(f"Dataset: {meta['dataset_root']} · {meta['num_classes']} classes · splits {meta.get('splits')}")

    started = time.time()
    search = architect.run_architect_search(
        generations=args.generations,
        population_size=args.population,
        proxy_epochs=args.proxy_epochs,
        dataset_dir=dataset_dir,
        seed=args.seed,
        output_dir=str(ROOT),
    )
    print(f"\nSearch {search['run_id']}: winner {search['best_candidate_id']} {json.dumps(search['best_config'])}")

    scratch = Path(search["run_dir"])
    winner_paths = (
        (ROOT / "best_model_weights.pth", ROOT / "best_model_metadata.json", ROOT / "best_model_config.json")
        if not args.no_promote
        else (scratch / "_winner.pth", scratch / "_winner.json", scratch / "_winner_config.json")
    )
    results = {}
    results["neuroforge_winner"] = architect.train_final_model(
        config=search["best_config"],
        dataset_dir=dataset_dir,
        epochs=args.epochs,
        seed=args.seed,
        output_weights_path=str(winner_paths[0]),
        output_metadata_path=str(winner_paths[1]),
        output_config_path=str(winner_paths[2]),
        search_summary=search["metadata"],
        search_history=search["history"],
    )

    for key in args.baselines:
        transfer = key in TRANSFER
        out = architect.train_final_model(
            config=architect.MANUAL_BASELINE_CONFIG,
            dataset_dir=dataset_dir,
            epochs=args.baseline_epochs or args.epochs,
            seed=args.seed,
            output_weights_path=str(ROOT / f"{key}_comparison_weights.pth"),
            output_metadata_path=str(ROOT / f"{key}_comparison_metadata.json"),
            use_transfer_learning=transfer,
            transfer_model=key if transfer else "resnet18",
        )
        metadata_path = ROOT / f"{key}_comparison_metadata.json"
        metadata = json.loads(metadata_path.read_text())
        metadata.update({"model_type": f"{key}_baseline", "comparison_role": f"fixed_{key}_baseline"})
        metadata_path.write_text(json.dumps(metadata, indent=2))
        results[key] = out

    print(f"\n{'model':<20}{'valid acc':>10}{'valid F1':>10}{'test acc':>10}{'test F1':>10}{'best ep':>9}")
    for name, out in results.items():
        v, t = out["validation_metrics"], out["held_out_test_metrics"]
        print(f"{name:<20}{v['accuracy']:>10.1f}{v['f1']:>10.1f}{t['accuracy']:>10.1f}{t['f1']:>10.1f}{out['best_epoch']:>9}")
    print(f"\ntest n={results['neuroforge_winner']['held_out_test_metrics']['sample_count']} · seed {args.seed} · {time.time() - started:.0f}s total")

    (scratch / "benchmark.json").write_text(json.dumps({"args": vars(args), "search_run": search["run_id"], "results": results}, indent=2, default=str))


if __name__ == "__main__":
    main()
