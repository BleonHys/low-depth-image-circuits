#!/usr/bin/env python3
import argparse
import json
import math
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from circuit_optimization.encodings.registry import get_permutation
except ImportError:
    get_permutation = None


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _get_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def _tail_lines(path: Path, n: int = 40) -> list[str]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        return [line.rstrip("\n") for line in lines[-n:]]
    except Exception:
        return []


def _run_subprocess(cmd, stdout_path: Path, stderr_path: Path, timeout: int, cwd: Path | None = None) -> dict:
    start = time.time()
    try:
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            result = subprocess.run(
                cmd,
                cwd=cwd or ROOT,
                stdout=stdout,
                stderr=stderr,
                text=True,
                timeout=timeout,
            )
        runtime = time.time() - start
        status = "SUCCESS" if result.returncode == 0 else "FAILED"
        return {"status": status, "runtime": runtime, "returncode": result.returncode}
    except subprocess.TimeoutExpired:
        runtime = time.time() - start
        return {"status": "TIMEOUT", "runtime": runtime, "returncode": None}


def _read_dataset_config(dataset_dir: Path) -> dict | None:
    config_path = dataset_dir / "dataset_config.json"
    if not config_path.exists():
        return None
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return None


_DATASET_SHAPES = {
    "imagenette/320px": (128, 128),
    "cifar10": (32, 32),
    "mnist": (32, 32),
    "fashion_mnist": (32, 32),
}


def _infer_patch_shape(tfds_name: str, n_patches: int, dataset_dir: Path | None = None) -> tuple[int, int]:
    config = _read_dataset_config(dataset_dir) if dataset_dir else None
    shape = None
    if config and config.get("shape"):
        shape = config["shape"]
    elif tfds_name in _DATASET_SHAPES:
        shape = _DATASET_SHAPES[tfds_name]
    if not shape or len(shape) < 2:
        raise ValueError(f"Unknown image shape for tfds_name '{tfds_name}'.")

    height = int(shape[0])
    width = int(shape[1])
    grid = int(math.isqrt(n_patches))
    if grid * grid != n_patches:
        raise ValueError(f"n_patches must be a perfect square, got {n_patches}.")
    if height % grid != 0 or width % grid != 0:
        raise ValueError(f"n_patches {n_patches} does not evenly divide image shape {height}x{width}.")
    return height // grid, width // grid


def _parse_svm_metrics(result_dir: Path, factoring: str, factors: int, compression_level: int) -> dict | None:
    csv_path = result_dir / f"{factoring}_f{factors}_c{compression_level}.csv"
    if not csv_path.exists():
        return None
    try:
        import csv

        with csv_path.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return None
        row = rows[-1]
        return {
            "train_accuracy": float(row.get("train_accuracy", "nan")),
            "val_accuracy": float(row.get("val_accuracy", "nan")),
        }
    except Exception:
        return None


def _parse_tn_metrics(run_dir: Path) -> dict | None:
    metrics_path = None
    for candidate in ("training_metrics.csv", "metrics.csv"):
        candidate_path = run_dir / candidate
        if candidate_path.exists():
            metrics_path = candidate_path
            break
    if metrics_path is None:
        return None
    try:
        import pandas as pd

        df = pd.read_csv(metrics_path)
        if df.empty or "val_acc" not in df.columns:
            return None
        last = df.iloc[-1]
        return {
            "train_accuracy": float(last.get("train_acc", float("nan"))),
            "val_accuracy": float(last.get("val_acc", float("nan"))),
        }
    except Exception:
        return None


def _parse_vqc_metrics(run_dir: Path) -> dict | None:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return None
    try:
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        if data.get("status") != "SUCCESS":
            return None
        return {
            "val_accuracy": data.get("best_val_acc"),
        }
    except Exception:
        return None


def _dataset_ready(dataset_dir: Path, n_patches: int, model: str) -> bool:
    if not (dataset_dir / "labels.npy").exists():
        return False
    if model == "svm":
        return (dataset_dir / f"states_p{n_patches}.npy").exists()
    if model in {"tn_mps", "tn_mpo"}:
        return (dataset_dir / f"mps_p{n_patches}.pkl").exists()
    if model in {"vqc_linear", "vqc_nonlinear"}:
        return (dataset_dir / f"states_p{n_patches}.npy").exists()
    return True


def run_experiments(args) -> None:
    results_dir = ROOT / args.results_dir
    _ensure_dir(results_dir)
    git_commit = _get_git_commit()

    dataset_cache = {}

    for encoding in args.indexings:
        for model in args.models:
            for fold in args.folds:
                for seed in args.seeds:
                    dataset_id = (
                        f"{args.base_dataset}__idx-{encoding}__k{args.max_per_class}__p{args.n_patches}__s{seed}"
                    )
                    run_dir = results_dir / dataset_id / model / f"fold{fold}" / f"seed{seed}"
                    _ensure_dir(run_dir)

                    run_json_path = run_dir / "run.json"
                    if args.skip_if_done and run_json_path.exists():
                        try:
                            existing = json.loads(run_json_path.read_text(encoding="utf-8"))
                            if existing.get("status") == "SUCCESS":
                                continue
                        except Exception:
                            pass

                    stdout_path = run_dir / "stdout.txt"
                    stderr_path = run_dir / "stderr.txt"

                    record = {
                        "config": {
                            "tfds_name": args.tfds_name,
                            "base_dataset": args.base_dataset,
                            "dataset_id": dataset_id,
                            "encoding": encoding,
                            "model": model,
                            "fold": fold,
                            "seed": seed,
                            "restarts": args.restarts if model in {"vqc_linear", "vqc_nonlinear"} else 1,
                            "max_per_class": args.max_per_class,
                            "n_patches": args.n_patches,
                            "timestamp": datetime.utcnow().isoformat() + "Z",
                            "git_commit": git_commit,
                        },
                        "status": "FAILED",
                        "metrics": None,
                        "runtime_seconds": None,
                        "logs": {
                            "stdout": str(stdout_path),
                            "stderr": str(stderr_path),
                        },
                        "error": None,
                        "artifacts": {},
                    }

                    try:
                        if model in {"vqc_linear", "vqc_nonlinear"} and args.n_patches != 1:
                            record["error"] = "invalid_n_patches: VQC requires n_patches=1"
                            run_json_path.write_text(json.dumps(record, indent=2))
                            continue

                        dataset_dir = ROOT / "data" / dataset_id
                        try:
                            patch_m, patch_n = _infer_patch_shape(args.tfds_name, args.n_patches, dataset_dir)
                            record["config"]["patch_shape"] = [patch_m, patch_n]
                        except Exception as exc:
                            record["error"] = f"encoding_validation_failed: {exc}"
                            run_json_path.write_text(json.dumps(record, indent=2))
                            continue

                        # Validate encoding early to ensure failure is recorded.
                        if get_permutation is not None:
                            try:
                                _ = get_permutation(patch_m, patch_n, encoding)
                            except Exception as exc:
                                record["error"] = f"invalid_encoding: {exc}"
                                run_json_path.write_text(json.dumps(record, indent=2))
                                continue

                        if not _dataset_ready(dataset_dir, args.n_patches, model):
                            cached = dataset_cache.get(dataset_id)
                            if cached is None or cached.get("status") in {"SUCCESS", "READY"}:
                                gen_stdout = run_dir / "dataset_stdout.txt"
                                gen_stderr = run_dir / "dataset_stderr.txt"
                                cmd = [
                                    sys.executable,
                                    "prepare_data.py",
                                    "--dataset_name",
                                    args.tfds_name,
                                    "--dataset_id",
                                    dataset_id,
                                    "--indexing",
                                    encoding,
                                    "--n_patches",
                                    str(args.n_patches),
                                    "--max_per_class",
                                    str(args.max_per_class),
                                    "--seed",
                                    str(seed),
                                ]
                                gen_res = _run_subprocess(cmd, gen_stdout, gen_stderr, args.timeout_seconds, cwd=ROOT)
                                cached = {
                                    "status": gen_res["status"],
                                    "stdout": str(gen_stdout),
                                    "stderr": str(gen_stderr),
                                }
                                dataset_cache[dataset_id] = cached
                            else:
                                dataset_cache[dataset_id] = cached
                        else:
                            dataset_cache[dataset_id] = {"status": "READY"}

                        if dataset_cache[dataset_id]["status"] not in {"SUCCESS", "READY"} or not _dataset_ready(
                            dataset_dir, args.n_patches, model
                        ):
                            record["error"] = "dataset_generation_failed"
                            record["logs"]["dataset_stdout"] = dataset_cache[dataset_id].get("stdout")
                            record["logs"]["dataset_stderr"] = dataset_cache[dataset_id].get("stderr")
                            run_json_path.write_text(json.dumps(record, indent=2))
                            continue

                        dataset_config = _read_dataset_config(dataset_dir)
                        if dataset_config:
                            record["config"]["image_shape"] = dataset_config.get("shape")
                            record["config"]["color_mode"] = dataset_config.get("color_mode")

                        if model == "svm":
                            factoring = "multicopy"
                            factors = 1
                            compression_level = 0
                            timestamp = f"{dataset_id}__f{fold}__s{seed}"
                            cmd = [
                                sys.executable,
                                "utils/svm_training.py",
                                "--timestamp",
                                timestamp,
                                "--foldindex",
                                str(fold),
                                "--dataset",
                                dataset_id,
                                "--factors",
                                str(factors),
                                "--compression_level",
                                str(compression_level),
                                "--factoring",
                                factoring,
                            ]
                            result = _run_subprocess(
                                cmd, stdout_path, stderr_path, args.timeout_seconds, cwd=ROOT / "classifier"
                            )
                            record["runtime_seconds"] = result["runtime"]
                            if result["status"] == "SUCCESS":
                                svm_dir = ROOT / "classifier" / "_results" / "svm" / f"{dataset_id}_{timestamp}"
                                record["artifacts"]["svm_results_dir"] = str(svm_dir)
                                metrics = _parse_svm_metrics(svm_dir, factoring, factors, compression_level)
                                record["metrics"] = metrics
                            record["status"] = result["status"]
                        elif model in {"tn_mps", "tn_mpo"}:
                            tn_model = "mps" if model == "tn_mps" else "mpo"
                            cmd = [
                                sys.executable,
                                "scripts/run_tn_training.py",
                                "--dataset",
                                dataset_id,
                                "--model",
                                tn_model,
                                "--fold",
                                str(fold),
                                "--basepath",
                                str(run_dir),
                                "--data_dir",
                                str(ROOT / "data"),
                                "--patched",
                                "true" if args.n_patches > 1 else "false",
                                "--n_factors",
                                str(args.n_patches),
                                "--warmstart",
                                "false",
                                "--compression_depth",
                                "0",
                                "--n_samples_warm_start",
                                str(min(args.max_per_class, 100)),
                                "--batch_size",
                                "50",
                                "--learning_rate",
                                "1e-4",
                                "--epochs",
                                "10",
                                "--chi_final",
                                "16",
                            ]
                            result = _run_subprocess(cmd, stdout_path, stderr_path, args.timeout_seconds, cwd=ROOT)
                            record["runtime_seconds"] = result["runtime"]
                            if result["status"] == "SUCCESS":
                                metrics = _parse_tn_metrics(run_dir)
                                record["metrics"] = metrics
                            record["status"] = result["status"]
                        elif model in {"vqc_linear", "vqc_nonlinear"}:
                            vqc_model = "linear" if model == "vqc_linear" else "nonlinear"
                            vqc_depth = 2
                            vqc_epochs = 30
                            vqc_batch_size = 16
                            vqc_optimizer = "adam"
                            vqc_lr = 0.01
                            vqc_temperature = 1.0
                            vqc_building_block = "su4"
                            vqc_patience = 10
                            vqc_min_delta = 0.0
                            record["config"].update(
                                {
                                    "vqc_model": vqc_model,
                                    "vqc_depth": vqc_depth,
                                    "vqc_epochs": vqc_epochs,
                                    "vqc_batch_size": vqc_batch_size,
                                    "vqc_optimizer": vqc_optimizer,
                                    "vqc_lr": vqc_lr,
                                    "vqc_temperature": vqc_temperature,
                                    "vqc_building_block": vqc_building_block,
                                    "vqc_patience": vqc_patience,
                                    "vqc_min_delta": vqc_min_delta,
                                    "vqc_restarts": args.restarts,
                                }
                            )
                            cmd = [
                                sys.executable,
                                "scripts/run_vqc_training.py",
                                "--dataset",
                                dataset_id,
                                "--data_dir",
                                str(ROOT / "data"),
                                "--fold",
                                str(fold),
                                "--seed",
                                str(seed),
                                "--model",
                                vqc_model,
                                "--building_block_tag",
                                vqc_building_block,
                                "--depth",
                                str(vqc_depth),
                                "--epochs",
                                str(vqc_epochs),
                                "--batch_size",
                                str(vqc_batch_size),
                                "--optimizer",
                                vqc_optimizer,
                                "--lr",
                                str(vqc_lr),
                                "--temperature",
                                str(vqc_temperature),
                                "--early_stopping_patience",
                                str(vqc_patience),
                                "--min_delta",
                                str(vqc_min_delta),
                                "--restarts",
                                str(args.restarts),
                                "--trial_dir",
                                str(run_dir),
                            ]
                            result = _run_subprocess(cmd, stdout_path, stderr_path, args.timeout_seconds, cwd=ROOT)
                            record["runtime_seconds"] = result["runtime"]
                            metrics = _parse_vqc_metrics(run_dir)
                            record["metrics"] = metrics
                            record["status"] = result["status"]
                        else:
                            record["error"] = f"unknown_model: {model}"

                        if record["status"] != "SUCCESS":
                            record["error"] = record["error"] or "run_failed"
                            record["error_tail"] = _tail_lines(stderr_path)
                    except Exception as exc:
                        record["status"] = "FAILED"
                        record["error"] = f"runner_exception: {exc}"
                        record["traceback"] = traceback.format_exc()
                    finally:
                        run_json_path.write_text(json.dumps(record, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run encoding ablation experiments.")
    parser.add_argument("--tfds_name", type=str, default="imagenette/320px")
    parser.add_argument("--base_dataset", type=str, default="imagenette_128")
    parser.add_argument("--indexings", type=str, nargs="+", default=["row_major"])
    parser.add_argument("--models", type=str, nargs="+", default=["svm"])
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--restarts", type=int, default=1)
    parser.add_argument("--max_per_class", type=int, default=50)
    parser.add_argument("--n_patches", type=int, default=1)
    parser.add_argument("--timeout_seconds", type=int, default=3600)
    parser.add_argument("--results_dir", type=str, default="results/encoding_ablation")
    parser.add_argument("--skip_if_done", action=argparse.BooleanOptionalAction, default=True)

    run_experiments(parser.parse_args())
