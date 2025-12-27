#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from classifier.utils.vqc_training import main as vqc_main


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _base_metrics(args, seed: int) -> dict:
    return {
        "status": None,
        "best_val_acc": None,
        "best_val_loss": None,
        "best_epoch": None,
        "best_val_loss_scaled": None,
        "runtime_seconds": None,
        "n_params": None,
        "best_val_loss_batchmean_unscaled": None,
        "best_val_acc_batchmean_unscaled": None,
        "best_epoch_batchmean_unscaled": None,
        "train_acc_at_best": None,
        "train_acc_at_best_batchmean_unscaled": None,
        "val_size": None,
        "val_idx_hash": None,
        "seed": seed,
        "fold": args.fold,
        "dataset": args.dataset,
        "model": args.model,
        "building_block_tag": args.building_block_tag,
        "depth": args.depth,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "optimizer": args.optimizer,
        "lr": args.lr,
        "temperature": args.temperature,
        "error": None,
    }


def _run_once(args, model_name: str, run_dir: Path, seed: int) -> dict:
    data_dir = Path(args.data_dir) if args.data_dir else ROOT / "data"
    config = {
        "dataset_name": args.dataset,
        "data_dir": os.fspath(data_dir),
        "basepath": os.fspath(run_dir),
        "fold": args.fold,
        "seed": seed,
        "model_name": model_name,
        "building_block_tag": args.building_block_tag,
        "n_qubits": args.n_qubits,
        "depth": args.depth,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "optimizer": args.optimizer,
        "learning_rate": args.lr,
        "temperature": args.temperature,
        "compression_depth": 0,
        "early_stopping_patience": args.early_stopping_patience,
        "min_delta": args.min_delta,
        "n_patches": 1,
    }

    start = time.time()
    metrics = _base_metrics(args, seed)
    try:
        summary = vqc_main(config, use_ray=False) or {}
        metrics.update({"status": "SUCCESS"})
        metrics.update(summary)
    except Exception as exc:
        metrics["status"] = "FAILED"
        metrics["error"] = str(exc)
        metrics["traceback"] = traceback.format_exc()
    metrics["runtime_seconds"] = time.time() - start
    return metrics


def run(args) -> int:
    trial_dir = Path(args.trial_dir)
    model_map = {
        "linear": "LinearVQC",
        "nonlinear": "NonLinearVQC",
    }
    model_name = model_map[args.model]

    restarts = max(1, args.restarts)
    restart_metrics = []

    for restart_idx in range(restarts):
        seed = args.seed + restart_idx
        run_dir = trial_dir if restarts == 1 else trial_dir / f"restart_{restart_idx}"
        run_dir.mkdir(parents=True, exist_ok=True)
        metrics = _run_once(args, model_name, run_dir, seed)
        metrics["restart_idx"] = restart_idx
        restart_metrics.append(metrics)

    _write_json(trial_dir / "metrics_restarts.json", {"restarts": restart_metrics})

    successes = [m for m in restart_metrics if m.get("status") == "SUCCESS" and m.get("best_val_acc") is not None]
    if successes:
        best = max(successes, key=lambda m: m["best_val_acc"])
        final = _base_metrics(args, best["seed"])
        final.update(best)
        final["status"] = "SUCCESS"
        final["error"] = None
        _write_json(trial_dir / "metrics.json", final)
        return 0

    error_msg = "all restarts failed"
    if restart_metrics:
        error_msg = restart_metrics[0].get("error") or error_msg
    final = _base_metrics(args, args.seed)
    final.update(
        {
            "status": "FAILED",
            "runtime_seconds": sum(m.get("runtime_seconds", 0.0) or 0.0 for m in restart_metrics),
            "error": error_msg,
        }
    )
    _write_json(trial_dir / "metrics.json", final)
    return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run VQC training with a simple CLI.")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trial_dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="linear", choices=["linear", "nonlinear"])
    parser.add_argument("--building_block_tag", type=str, default="su4")
    parser.add_argument("--n_qubits", type=int, default=None)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "bfgs", "lbfgs"])
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--temperature", type=float, default=1.0 / 128.0)
    parser.add_argument("--early_stopping_patience", type=int, default=10)
    parser.add_argument("--min_delta", type=float, default=0.0)
    parser.add_argument("--restarts", type=int, default=1)

    sys.exit(run(parser.parse_args()))
