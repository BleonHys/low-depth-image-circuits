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


def _write_metrics(trial_dir: Path, payload: dict) -> None:
    trial_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = trial_dir / "metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2))


def run(args) -> int:
    root = Path(__file__).resolve().parents[1]
    data_dir = Path(args.data_dir) if args.data_dir else root / "data"
    trial_dir = Path(args.trial_dir)

    config = {
        "dataset_name": args.dataset,
        "data_dir": os.fspath(data_dir),
        "basepath": os.fspath(trial_dir),
        "fold": args.fold,
        "seed": args.seed,
        "model_name": args.model_name,
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
    }

    start = time.time()
    try:
        summary = vqc_main(config, use_ray=False) or {}
        metrics = {
            "status": "SUCCESS",
            "best_val_acc": summary.get("best_val_acc"),
            "best_val_loss": summary.get("best_val_loss"),
            "best_epoch": summary.get("best_epoch"),
            "train_acc_at_best": summary.get("train_acc_at_best"),
            "runtime_seconds": time.time() - start,
            "n_params": summary.get("n_params"),
            "config": config,
        }
        _write_metrics(trial_dir, metrics)
        return 0
    except Exception as exc:
        metrics = {
            "status": "FAILED",
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "runtime_seconds": time.time() - start,
            "config": config,
        }
        _write_metrics(trial_dir, metrics)
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run VQC training with a simple CLI.")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_name", type=str, default="LinearVQC",
                        choices=["LinearVQC", "NonLinearVQC", "NonLinearVQC_shadow"])
    parser.add_argument("--building_block_tag", type=str, default="su4")
    parser.add_argument("--n_qubits", type=int, default=None)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "bfgs", "lbfgs"])
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--temperature", type=float, default=128)
    parser.add_argument("--early_stopping_patience", type=int, default=10)
    parser.add_argument("--min_delta", type=float, default=0.0)
    parser.add_argument("--trial_dir", type=str, required=True)

    sys.exit(run(parser.parse_args()))
