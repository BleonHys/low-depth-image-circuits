#!/usr/bin/env python3
import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from circuit_optimization.encodings.registry import get_permutation
except ImportError:
    get_permutation = None


def _load_metrics_json(run_dir: Path) -> dict | None:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return None
    try:
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        if data.get("status") != "SUCCESS":
            return None
        primary_val_acc = data.get("best_val_acc")
        legacy_val_acc = data.get("best_val_acc_batchmean_unscaled")
        if primary_val_acc is None:
            primary_val_acc = legacy_val_acc
        primary_train_acc = data.get("train_acc_at_best")
        legacy_train_acc = data.get("train_acc_at_best_batchmean_unscaled")
        if primary_train_acc is None:
            primary_train_acc = legacy_train_acc
        return {
            "val_accuracy": primary_val_acc,
            "train_accuracy": primary_train_acc,
            "val_accuracy_legacy_batchmean_unscaled": legacy_val_acc,
            "train_accuracy_legacy_batchmean_unscaled": legacy_train_acc,
            "runtime_seconds": data.get("runtime_seconds"),
        }
    except Exception:
        return None


def _safe_mean(values):
    return float(statistics.mean(values)) if values else None


def _safe_std(values):
    return float(statistics.pstdev(values)) if values else None


def _compute_lps(m: int, n: int, encoding: str) -> float | None:
    if get_permutation is None:
        return None
    try:
        idx = get_permutation(m, n, encoding)
    except Exception:
        return None
    inv = np.empty_like(idx)
    inv[idx] = np.arange(len(idx))

    distances = []
    for r in range(m):
        for c in range(n):
            pos = inv[r * n + c]
            if r + 1 < m:
                distances.append(abs(pos - inv[(r + 1) * n + c]))
            if c + 1 < n:
                distances.append(abs(pos - inv[r * n + (c + 1)]))
    return float(np.mean(distances)) if distances else None


def summarize(results_dir: Path) -> None:
    run_files = list(results_dir.rglob("run.json"))
    if not run_files:
        print(f"No run.json files found under {results_dir}")
        return

    records = []
    for path in run_files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            data["_run_dir"] = str(path.parent)
            records.append(data)
        except Exception:
            print(f"Warning: failed to parse {path}")

    grouped = {}
    shapes_by_encoding = {}

    for rec in records:
        cfg = rec.get("config", {})
        encoding = cfg.get("encoding")
        model = cfg.get("model")
        if not encoding or not model:
            continue
        key = (encoding, model)
        grouped.setdefault(key, []).append(rec)
        shape = cfg.get("patch_shape") or cfg.get("image_shape")
        if shape:
            shapes_by_encoding.setdefault(encoding, []).append(tuple(shape))

    summaries = []
    lps_cache = {}

    for (encoding, model), recs in grouped.items():
        n_total = len(recs)
        successes = [r for r in recs if r.get("status") == "SUCCESS"]
        n_success = len(successes)
        success_rate = n_success / n_total if n_total else 0.0

        val_accs = []
        legacy_val_accs = []
        runtimes = []
        for r in successes:
            run_dir = Path(r.get("_run_dir", ""))
            metrics = _load_metrics_json(run_dir) or {}
            if not metrics:
                metrics = r.get("metrics") or {}
            val = metrics.get("val_accuracy")
            if val is not None and not np.isnan(val):
                val_accs.append(val)
            legacy_val = metrics.get("val_accuracy_legacy_batchmean_unscaled")
            if legacy_val is not None and not np.isnan(legacy_val):
                legacy_val_accs.append(legacy_val)
            runtime = r.get("runtime_seconds")
            if runtime is not None:
                runtimes.append(runtime)
            elif metrics.get("runtime_seconds") is not None:
                runtimes.append(metrics.get("runtime_seconds"))

        mean_val = _safe_mean(val_accs)
        std_val = _safe_std(val_accs)
        mean_legacy_val = _safe_mean(legacy_val_accs)
        std_legacy_val = _safe_std(legacy_val_accs)
        mean_runtime = _safe_mean(runtimes)
        stability_score = success_rate * mean_val if mean_val is not None else None

        if encoding not in lps_cache:
            shape_list = shapes_by_encoding.get(encoding) or []
            if shape_list:
                m, n = shape_list[0]
                if any(shape != (m, n) for shape in shape_list):
                    print(f"Warning: multiple shapes for encoding {encoding}, using {m}x{n}")
                lps_cache[encoding] = _compute_lps(m, n, encoding)
            else:
                lps_cache[encoding] = None

        summaries.append({
            "encoding": encoding,
            "model": model,
            "n_total_runs": n_total,
            "n_success": n_success,
            "success_rate": success_rate,
            "mean_val_acc": mean_val,
            "std_val_acc": std_val,
            "mean_val_acc_legacy_batchmean_unscaled": mean_legacy_val,
            "std_val_acc_legacy_batchmean_unscaled": std_legacy_val,
            "legacy_metric_note": "legacy comparability only" if legacy_val_accs else None,
            "mean_runtime": mean_runtime,
            "stability_score": stability_score,
            "lps": lps_cache.get(encoding),
        })

    summary_csv = results_dir / "summary.csv"
    summary_json = results_dir / "summary.json"

    import csv
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)

    summary_json.write_text(json.dumps(summaries, indent=2))
    print(f"Wrote {summary_csv}")
    print(f"Wrote {summary_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize encoding ablation results.")
    parser.add_argument("--results_dir", type=str, default="results/encoding_ablation")
    summarize(Path(parser.parse_args().results_dir))
