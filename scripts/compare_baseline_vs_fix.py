#!/usr/bin/env python3
import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(cmd, cwd, env=None):
    result = subprocess.run(cmd, cwd=cwd, env=env, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(cmd)}")


def _git(args, cwd=ROOT):
    result = subprocess.run(["git"] + args, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result.stdout.strip()


def _ensure_worktree(path: Path, ref: str) -> None:
    if path.exists():
        _git(["-C", str(path), "rev-parse", "--git-dir"])
        status = _git(["-C", str(path), "status", "--porcelain"])
        if status:
            raise RuntimeError(f"Worktree {path} has uncommitted changes.")
        _git(["-C", str(path), "checkout", ref])
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    _git(["worktree", "add", str(path), ref])


def _worktree_is_clean(path: Path) -> bool:
    status = _git(["-C", str(path), "status", "--porcelain", "--untracked-files=no"])
    return status == ""


def _dataset_ready(dataset_dir: Path, n_patches: int) -> bool:
    if not (dataset_dir / "labels.npy").exists():
        return False
    return (dataset_dir / f"states_p{n_patches}.npy").exists()


def _dataset_id(base_dataset: str, encoding: str, max_per_class: int, n_patches: int, seed: int) -> str:
    return f"{base_dataset}__idx-{encoding}__k{max_per_class}__p{n_patches}__s{seed}"


def _read_dataset_config(dataset_dir: Path) -> dict:
    config_path = dataset_dir / "dataset_config.json"
    if not config_path.exists():
        return {}
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _infer_n_qubits(dataset_dir: Path, n_patches: int) -> int:
    import numpy as np

    states_path = dataset_dir / f"states_p{n_patches}.npy"
    states = np.load(states_path)
    if states.ndim == 3:
        states = states[0]
    dim = int(states.shape[1])
    n_qubits = int(np.log2(dim))
    if 2**n_qubits != dim:
        raise ValueError(f"State dimension {dim} is not a power of two.")
    return n_qubits


def _run_logged(cmd, cwd: Path, stdout_path: Path, stderr_path: Path, timeout: int, env=None) -> int:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            stdout=stdout,
            stderr=stderr,
            text=True,
            timeout=timeout,
        )
    return result.returncode


def _prepare_dataset(
    worktree_path: Path,
    data_dir: Path | None,
    tfds_name: str,
    dataset_id: str,
    encoding: str,
    max_per_class: int,
    n_patches: int,
    seed: int,
    timeout: int,
    env,
    run_dir: Path,
) -> dict:
    dataset_dir = (data_dir or (worktree_path / "data")) / dataset_id
    if _dataset_ready(dataset_dir, n_patches):
        return {"status": "READY"}
    if data_dir is not None:
        return {"status": "MISSING"}
    stdout_path = run_dir / "dataset_stdout.txt"
    stderr_path = run_dir / "dataset_stderr.txt"
    cmd = [
        sys.executable,
        "prepare_data.py",
        "--dataset_name",
        tfds_name,
        "--dataset_id",
        dataset_id,
        "--indexing",
        encoding,
        "--n_patches",
        str(n_patches),
        "--max_per_class",
        str(max_per_class),
        "--seed",
        str(seed),
    ]
    returncode = _run_logged(cmd, worktree_path, stdout_path, stderr_path, timeout, env=env)
    status = "SUCCESS" if returncode == 0 else "FAILED"
    return {
        "status": status,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }


def _derive_metrics_from_csv(csv_path: Path) -> dict:
    if not csv_path.exists():
        return {}
    try:
        import pandas as pd

        df = pd.read_csv(csv_path)
    except Exception:
        return {}

    metrics = {}
    if "loss_val" in df.columns and not df["loss_val"].isna().all():
        idx_base = int(df["loss_val"].idxmin())
        metrics["best_val_loss_batchmean_unscaled"] = float(df.loc[idx_base, "loss_val"])
        if "accuracy_val" in df.columns:
            metrics["best_val_acc_batchmean_unscaled"] = float(df.loc[idx_base, "accuracy_val"])
        metrics["best_epoch_batchmean_unscaled"] = idx_base + 1
        if "accuracy_train" in df.columns:
            metrics["train_acc_at_best_batchmean_unscaled"] = float(df.loc[idx_base, "accuracy_train"])

    if "loss_val_scaled" in df.columns and not df["loss_val_scaled"].isna().all():
        idx_scaled = int(df["loss_val_scaled"].idxmin())
        metrics["best_val_loss"] = float(df.loc[idx_scaled, "loss_val_scaled"])
        if "accuracy_val_sampleweighted" in df.columns:
            metrics["best_val_acc"] = float(df.loc[idx_scaled, "accuracy_val_sampleweighted"])
        elif "accuracy_val" in df.columns:
            metrics["best_val_acc"] = float(df.loc[idx_scaled, "accuracy_val"])
        metrics["best_epoch"] = idx_scaled + 1
        if "accuracy_train" in df.columns:
            metrics["train_acc_at_best"] = float(df.loc[idx_scaled, "accuracy_train"])
    elif "loss_val" in df.columns and not df["loss_val"].isna().all():
        idx_base = int(df["loss_val"].idxmin())
        metrics["best_val_loss"] = float(df.loc[idx_base, "loss_val"])
        if "accuracy_val" in df.columns:
            metrics["best_val_acc"] = float(df.loc[idx_base, "accuracy_val"])
        metrics["best_epoch"] = idx_base + 1
        if "accuracy_train" in df.columns:
            metrics["train_acc_at_best"] = float(df.loc[idx_base, "accuracy_train"])

    return metrics


def _run_vqc_training(
    worktree_path: Path,
    config: dict,
    metrics_path: Path,
    stdout_path: Path,
    stderr_path: Path,
    timeout: int,
    env,
) -> int:
    config_path = metrics_path.parent / "config_compare.json"
    config_path.write_text(json.dumps(config, indent=2))
    script = (
        "import json, sys, time, traceback\n"
        "from pathlib import Path\n"
        "from utils.vqc_training import main as vqc_main\n"
        "config = json.loads(Path(sys.argv[1]).read_text())\n"
        "metrics_path = Path(sys.argv[2])\n"
        "start = time.time()\n"
        "metrics = {\"status\": None, \"error\": None}\n"
        "try:\n"
        "    summary = vqc_main(config, use_ray=False)\n"
        "    metrics[\"status\"] = \"SUCCESS\"\n"
        "    if isinstance(summary, dict):\n"
        "        metrics.update(summary)\n"
        "except Exception as exc:\n"
        "    metrics[\"status\"] = \"FAILED\"\n"
        "    metrics[\"error\"] = str(exc)\n"
        "    metrics[\"traceback\"] = traceback.format_exc()\n"
        "metrics[\"runtime_seconds\"] = time.time() - start\n"
        "metrics_path.write_text(json.dumps(metrics, indent=2))\n"
    )
    cmd = [sys.executable, "-c", script, str(config_path), str(metrics_path)]
    return _run_logged(cmd, worktree_path / "classifier", stdout_path, stderr_path, timeout, env=env)


def _run_vqc_job(
    worktree_path: Path,
    results_dir: Path,
    data_dir: Path | None,
    tfds_name: str,
    base_dataset: str,
    encoding: str,
    fold: int,
    seed: int,
    max_per_class: int,
    n_patches: int,
    vqc_depth: int,
    vqc_epochs: int,
    vqc_batch_size: int,
    vqc_lr: float,
    vqc_temperature: float,
    vqc_building_block: str,
    vqc_patience: int,
    vqc_min_delta: float,
    timeout: int,
    skip_if_done: bool,
    env,
) -> None:
    dataset_id = _dataset_id(base_dataset, encoding, max_per_class, n_patches, seed)
    run_dir = results_dir / dataset_id / "vqc_linear" / f"fold{fold}" / f"seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    run_json_path = run_dir / "run.json"

    if skip_if_done and run_json_path.exists():
        try:
            existing = json.loads(run_json_path.read_text(encoding="utf-8"))
            if existing.get("status") == "SUCCESS":
                return
        except Exception:
            pass

    record = {
        "config": {
            "tfds_name": tfds_name,
            "base_dataset": base_dataset,
            "dataset_id": dataset_id,
            "encoding": encoding,
            "model": "vqc_linear",
            "fold": fold,
            "seed": seed,
            "restarts": 1,
            "max_per_class": max_per_class,
            "n_patches": n_patches,
            "vqc_model": "linear",
            "vqc_depth": vqc_depth,
            "vqc_epochs": vqc_epochs,
            "vqc_batch_size": vqc_batch_size,
            "vqc_optimizer": "adam",
            "vqc_lr": vqc_lr,
            "vqc_temperature": vqc_temperature,
            "vqc_building_block": vqc_building_block,
            "vqc_patience": vqc_patience,
            "vqc_min_delta": vqc_min_delta,
            "git_commit": _git(["-C", str(worktree_path), "rev-parse", "HEAD"]),
        },
        "status": "FAILED",
        "metrics": None,
        "runtime_seconds": None,
        "logs": {
            "stdout": str(run_dir / "stdout.txt"),
            "stderr": str(run_dir / "stderr.txt"),
        },
        "error": None,
    }

    dataset_result = _prepare_dataset(
        worktree_path,
        data_dir,
        tfds_name,
        dataset_id,
        encoding,
        max_per_class,
        n_patches,
        seed,
        timeout,
        env,
        run_dir,
    )
    if dataset_result["status"] not in {"READY", "SUCCESS"}:
        record["error"] = (
            "dataset_missing_in_data_dir" if dataset_result["status"] == "MISSING" else "dataset_generation_failed"
        )
        record["logs"]["dataset_stdout"] = dataset_result.get("stdout")
        record["logs"]["dataset_stderr"] = dataset_result.get("stderr")
        run_json_path.write_text(json.dumps(record, indent=2))
        return

    dataset_dir = (data_dir or (worktree_path / "data")) / dataset_id
    dataset_config = _read_dataset_config(dataset_dir)
    if dataset_config:
        record["config"]["image_shape"] = dataset_config.get("shape")
        record["config"]["color_mode"] = dataset_config.get("color_mode")

    try:
        n_qubits = _infer_n_qubits(dataset_dir, n_patches)
    except Exception as exc:
        record["error"] = f"invalid_n_qubits: {exc}"
        run_json_path.write_text(json.dumps(record, indent=2))
        return

    config = {
        "dataset_name": dataset_id,
        "data_dir": os.fspath(data_dir or (worktree_path / "data")),
        "basepath": os.fspath(run_dir),
        "fold": fold,
        "seed": seed,
        "model_name": "LinearVQC",
        "building_block_tag": vqc_building_block,
        "n_qubits": n_qubits,
        "depth": vqc_depth,
        "epochs": vqc_epochs,
        "batch_size": vqc_batch_size,
        "optimizer": "adam",
        "learning_rate": vqc_lr,
        "temperature": vqc_temperature,
        "compression_depth": 0,
        "early_stopping_patience": vqc_patience,
        "min_delta": vqc_min_delta,
        "n_patches": n_patches,
    }

    stdout_path = run_dir / "stdout.txt"
    stderr_path = run_dir / "stderr.txt"
    metrics_path = run_dir / "metrics.json"
    returncode = _run_vqc_training(
        worktree_path,
        config,
        metrics_path,
        stdout_path,
        stderr_path,
        timeout,
        env,
    )
    metrics = _load_metrics(run_dir)
    derived = _derive_metrics_from_csv(run_dir / "training_data_epoch.csv")
    metrics.update(derived)
    metrics_path.write_text(json.dumps(metrics, indent=2))

    record["runtime_seconds"] = metrics.get("runtime_seconds")
    record["metrics"] = {"val_accuracy": metrics.get("best_val_acc")}
    record["status"] = "SUCCESS" if returncode == 0 and metrics.get("status") == "SUCCESS" else "FAILED"
    if record["status"] != "SUCCESS":
        record["error"] = metrics.get("error") or "run_failed"
    run_json_path.write_text(json.dumps(record, indent=2))


def _run_jobs(worktree_path: Path, results_dir: Path, args, env) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir) if args.data_dir else None
    for encoding in args.indexings:
        for fold in args.folds:
            for seed in args.seeds:
                _run_vqc_job(
                    worktree_path=worktree_path,
                    results_dir=results_dir,
                    data_dir=data_dir,
                    tfds_name=args.tfds_name,
                    base_dataset=args.base_dataset,
                    encoding=encoding,
                    fold=fold,
                    seed=seed,
                    max_per_class=args.max_per_class,
                    n_patches=1,
                    vqc_depth=args.vqc_depth,
                    vqc_epochs=args.vqc_epochs,
                    vqc_batch_size=args.vqc_batch_size,
                    vqc_lr=args.vqc_lr,
                    vqc_temperature=args.vqc_temperature,
                    vqc_building_block=args.vqc_building_block,
                    vqc_patience=args.vqc_patience,
                    vqc_min_delta=args.vqc_min_delta,
                    timeout=args.timeout_seconds,
                    skip_if_done=args.skip_if_done,
                    env=env,
                )

def _load_metrics(run_dir: Path) -> dict:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return {}
    try:
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_training_curve(run_dir: Path, epochs: int) -> list[float]:
    csv_path = run_dir / "training_data_epoch.csv"
    if not csv_path.exists():
        return []
    try:
        import pandas as pd

        df = pd.read_csv(csv_path)
        if "loss_train" not in df.columns:
            return []
        return [float(x) for x in df["loss_train"].head(epochs).tolist()]
    except Exception:
        return []


def _collect_runs(results_dir: Path, loss_epochs: int) -> dict:
    runs = {}
    for run_path in results_dir.rglob("run.json"):
        try:
            data = json.loads(run_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        cfg = data.get("config", {})
        encoding = cfg.get("encoding")
        fold = cfg.get("fold")
        seed = cfg.get("seed")
        if encoding is None or fold is None or seed is None:
            continue
        key = (encoding, int(fold), int(seed))
        run_dir = run_path.parent
        metrics = _load_metrics(run_dir)
        runs[key] = {
            "run_dir": str(run_dir),
            "status": data.get("status"),
            "metrics": metrics,
            "train_loss_curve": _load_training_curve(run_dir, loss_epochs),
        }
    return runs


def _comparison_rows(baseline_runs: dict, fix_runs: dict, loss_epochs: int) -> list[dict]:
    rows = []
    keys = sorted(set(baseline_runs) | set(fix_runs))
    for key in keys:
        encoding, fold, seed = key
        base = baseline_runs.get(key) or {}
        fix = fix_runs.get(key) or {}
        base_metrics = base.get("metrics", {}) if base else {}
        fix_metrics = fix.get("metrics", {}) if fix else {}
        rows.append(
            {
                "encoding": encoding,
                "fold": fold,
                "seed": seed,
                "baseline_status": base.get("status"),
                "fix_status": fix.get("status"),
                "baseline_best_val_acc": base_metrics.get("best_val_acc"),
                "fix_best_val_acc": fix_metrics.get("best_val_acc"),
                "baseline_best_val_loss": base_metrics.get("best_val_loss"),
                "fix_best_val_loss": fix_metrics.get("best_val_loss"),
                "baseline_best_epoch": base_metrics.get("best_epoch"),
                "fix_best_epoch": fix_metrics.get("best_epoch"),
                "baseline_best_val_acc_batchmean_unscaled": base_metrics.get("best_val_acc_batchmean_unscaled"),
                "fix_best_val_acc_batchmean_unscaled": fix_metrics.get("best_val_acc_batchmean_unscaled"),
                "baseline_best_val_loss_batchmean_unscaled": base_metrics.get("best_val_loss_batchmean_unscaled"),
                "fix_best_val_loss_batchmean_unscaled": fix_metrics.get("best_val_loss_batchmean_unscaled"),
                "baseline_train_loss_firstN": ";".join(
                    f"{v:.6f}" for v in (base.get("train_loss_curve") or [])[:loss_epochs]
                ),
                "fix_train_loss_firstN": ";".join(
                    f"{v:.6f}" for v in (fix.get("train_loss_curve") or [])[:loss_epochs]
                ),
                "baseline_run_dir": base.get("run_dir"),
                "fix_run_dir": fix.get("run_dir"),
            }
        )
    return rows


def main() -> int:
    default_data_dir = ROOT / "data"
    parser = argparse.ArgumentParser(description="Run baseline vs fix VQC experiments and compare results.")
    parser.add_argument("--baseline_ref", type=str, default="baseline_eval")
    parser.add_argument("--fix_ref", type=str, default="fix_eval")
    parser.add_argument("--worktree_base", type=str, default=str(ROOT / ".worktrees"))
    parser.add_argument("--report_dir", type=str, default=str(ROOT / "results" / "baseline_vs_fix"))
    parser.add_argument("--tfds_name", type=str, default="imagenette/320px")
    parser.add_argument("--base_dataset", type=str, default="imagenette_128")
    parser.add_argument("--max_per_class", type=int, default=10)
    parser.add_argument("--indexings", type=str, nargs="+", default=["row_major", "hilbert"])
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    parser.add_argument("--restarts", type=int, default=1)
    parser.add_argument("--timeout_seconds", type=int, default=3600)
    parser.add_argument("--loss_curve_epochs", type=int, default=5)
    parser.add_argument("--skip_if_done", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--data_dir", type=str, default=str(default_data_dir) if default_data_dir.exists() else None)
    parser.add_argument("--vqc_depth", type=int, default=2)
    parser.add_argument("--vqc_epochs", type=int, default=30)
    parser.add_argument("--vqc_batch_size", type=int, default=16)
    parser.add_argument("--vqc_lr", type=float, default=0.01)
    parser.add_argument("--vqc_temperature", type=float, default=1.0)
    parser.add_argument("--vqc_building_block", type=str, default="su4")
    parser.add_argument("--vqc_patience", type=int, default=10)
    parser.add_argument("--vqc_min_delta", type=float, default=0.0)
    args = parser.parse_args()

    baseline_path = Path(args.worktree_base) / "baseline_eval"
    fix_path = Path(args.worktree_base) / "fix_eval"
    fix_head = _git(["rev-parse", args.fix_ref])
    current_head = _git(["rev-parse", "HEAD"])
    use_current_fix = fix_head == current_head

    if use_current_fix:
        if not _worktree_is_clean(ROOT):
            raise RuntimeError("Current worktree has tracked changes; commit or stash before compare.")
        fix_path = ROOT

    _ensure_worktree(baseline_path, args.baseline_ref)
    if fix_path != ROOT:
        _ensure_worktree(fix_path, args.fix_ref)

    env = os.environ.copy()
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    baseline_results_dir = baseline_path / "results" / "baseline_eval"
    fix_results_dir = fix_path / "results" / "fix_eval"

    if args.restarts != 1:
        raise RuntimeError("compare_baseline_vs_fix only supports restarts=1 for now.")

    _run_jobs(baseline_path, baseline_results_dir, args, env)
    _run_jobs(fix_path, fix_results_dir, args, env)

    _run(
        [sys.executable, "scripts/summarize_results.py", "--results_dir", str(baseline_results_dir)],
        cwd=ROOT,
        env=env,
    )
    _run(
        [sys.executable, "scripts/summarize_results.py", "--results_dir", str(fix_results_dir)],
        cwd=ROOT,
        env=env,
    )

    baseline_runs = _collect_runs(baseline_results_dir, args.loss_curve_epochs)
    fix_runs = _collect_runs(fix_results_dir, args.loss_curve_epochs)
    rows = _comparison_rows(baseline_runs, fix_runs, args.loss_curve_epochs)

    comparison_csv = report_dir / "comparison.csv"
    comparison_json = report_dir / "comparison.json"

    if rows:
        with comparison_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    comparison_json.write_text(
        json.dumps(
            {
                "baseline_ref": args.baseline_ref,
                "fix_ref": args.fix_ref,
                "baseline_results_dir": str(baseline_results_dir),
                "fix_results_dir": str(fix_results_dir),
                "rows": rows,
            },
            indent=2,
        )
    )
    print(f"Wrote {comparison_csv}")
    print(f"Wrote {comparison_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
