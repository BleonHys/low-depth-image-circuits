#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# ---- CONFIG ----
TFDS_NAME="imagenette/320px"
BASE_DATASET="imagenette_128"
RESULTS_DIR="results/encoding_ablation_vqc_all9"
ENCODINGS=(row_major column_major snake vertical_snake diagonal diagonal_zigzag corner_spiral morton hilbert)

FOLDS=(0 1 2 3 4)
SEEDS=(42 43)
MAX_PER_CLASS=50
N_PATCHES=1

# VQC settings
RESTARTS_LINEAR=1
RESTARTS_NONLINEAR=1
TIMEOUT_LINEAR=21600     # 6h per job
TIMEOUT_NONLINEAR=21600  # 6h per job

# ----------------

banner () {
  echo
  echo "================================================================================"
  echo "$1"
  echo "================================================================================"
}

progress () {
  python - <<'PY'
import json, glob
from collections import Counter

runs = glob.glob("results/encoding_ablation_vqc_all9/**/run.json", recursive=True)
c = Counter()
for p in runs:
    d = json.load(open(p, "r", encoding="utf-8"))
    cfg = d.get("config", {})
    c[(cfg.get("model"), d.get("status"))] += 1

print("Run.json count:", len(runs))
for k,v in sorted(c.items()):
    print(f"{k}: {v}")
PY
}

summarize () {
  python scripts/summarize_results.py --results_dir "$RESULTS_DIR"
  echo
  echo "Top of summary.csv:"
  column -s, -t "$RESULTS_DIR/summary.csv" | head -n 40 || true
}

sanity () {
  banner "SANITY CHECKS"
  echo "Repo: $REPO_ROOT"
  echo "Python: $(python -V)"
  echo "GPU:"
  nvidia-smi | head -n 20 || true
  echo "JAX devices:"
  python -c "import jax; print(jax.devices())"
  echo "Available encodings:"
  python -c "from circuit_optimization.encodings.registry import list_encodings; print(list_encodings())"
  echo "Data dir exists?"; ls -ld data || true
  echo "Results dir: $RESULTS_DIR"
  mkdir -p "$RESULTS_DIR"
}

pilot () {
  banner "PILOT (small, but full stack): 2 encodings, 1 fold, 1 seed, both VQC models"
  python scripts/experiment_runner.py \
    --tfds_name "$TFDS_NAME" \
    --base_dataset "$BASE_DATASET" \
    --indexings row_major morton \
    --models vqc_linear vqc_nonlinear \
    --folds 0 \
    --seeds 42 \
    --restarts 1 \
    --max_per_class 10 \
    --n_patches "$N_PATCHES" \
    --timeout_seconds 7200 \
    --results_dir "$RESULTS_DIR"

  summarize
  progress
}

run_linear_all9 () {
  banner "FULL BLOCK A: VQC_LINEAR on ALL 9 encodings (5 folds, 2 seeds)"
  python scripts/experiment_runner.py \
    --tfds_name "$TFDS_NAME" \
    --base_dataset "$BASE_DATASET" \
    --indexings "${ENCODINGS[@]}" \
    --models vqc_linear \
    --folds "${FOLDS[@]}" \
    --seeds "${SEEDS[@]}" \
    --restarts "$RESTARTS_LINEAR" \
    --max_per_class "$MAX_PER_CLASS" \
    --n_patches "$N_PATCHES" \
    --timeout_seconds "$TIMEOUT_LINEAR" \
    --results_dir "$RESULTS_DIR"

  summarize
  progress
}

run_nonlinear_all9 () {
  banner "FULL BLOCK B: VQC_NONLINEAR on ALL 9 encodings (5 folds, 2 seeds)"
  python scripts/experiment_runner.py \
    --tfds_name "$TFDS_NAME" \
    --base_dataset "$BASE_DATASET" \
    --indexings "${ENCODINGS[@]}" \
    --models vqc_nonlinear \
    --folds "${FOLDS[@]}" \
    --seeds "${SEEDS[@]}" \
    --restarts "$RESTARTS_NONLINEAR" \
    --max_per_class "$MAX_PER_CLASS" \
    --n_patches "$N_PATCHES" \
    --timeout_seconds "$TIMEOUT_NONLINEAR" \
    --results_dir "$RESULTS_DIR"

  summarize
  progress
}

main () {
  sanity
  pilot
  run_linear_all9
  run_nonlinear_all9

  banner "DONE (final summarize)"
  summarize
  progress
  echo "Final results in: $RESULTS_DIR"
}

main
