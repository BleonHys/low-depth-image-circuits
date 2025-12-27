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

    baseline_results_dir = "results/baseline_eval"
    fix_results_dir = "results/fix_eval"

    base_cmd = [
        sys.executable,
        "scripts/experiment_runner.py",
        "--tfds_name",
        args.tfds_name,
        "--base_dataset",
        args.base_dataset,
        "--indexings",
        *args.indexings,
        "--models",
        "vqc_linear",
        "--folds",
        *[str(f) for f in args.folds],
        "--seeds",
        *[str(s) for s in args.seeds],
        "--restarts",
        str(args.restarts),
        "--max_per_class",
        str(args.max_per_class),
        "--n_patches",
        "1",
        "--timeout_seconds",
        str(args.timeout_seconds),
    ]
    if args.skip_if_done:
        base_cmd.append("--skip_if_done")
    else:
        base_cmd.append("--no-skip_if_done")

    _run(base_cmd + ["--results_dir", baseline_results_dir], cwd=baseline_path, env=env)
    _run(
        base_cmd
        + [
            "--results_dir",
            fix_results_dir,
            "--vqc_depth",
            str(args.vqc_depth),
            "--vqc_epochs",
            str(args.vqc_epochs),
            "--vqc_batch_size",
            str(args.vqc_batch_size),
            "--vqc_lr",
            str(args.vqc_lr),
            "--vqc_temperature",
            str(args.vqc_temperature),
            "--vqc_building_block",
            args.vqc_building_block,
            "--vqc_patience",
            str(args.vqc_patience),
            "--vqc_min_delta",
            str(args.vqc_min_delta),
        ],
        cwd=fix_path,
        env=env,
    )

    _run(
        [sys.executable, "scripts/summarize_results.py", "--results_dir", baseline_results_dir],
        cwd=baseline_path,
        env=env,
    )
    _run(
        [sys.executable, "scripts/summarize_results.py", "--results_dir", fix_results_dir],
        cwd=fix_path,
        env=env,
    )

    baseline_runs = _collect_runs(baseline_path / baseline_results_dir, args.loss_curve_epochs)
    fix_runs = _collect_runs(fix_path / fix_results_dir, args.loss_curve_epochs)
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
                "baseline_results_dir": str(baseline_path / baseline_results_dir),
                "fix_results_dir": str(fix_path / fix_results_dir),
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
