#!/usr/bin/env bash
# ASTRA ML Pipeline Runner
# ─────────────────────────────────────────────────────────────────────────────
# Usage:
#   ./ml/run_all.sh --train /path/to/train.csv --test /path/to/test.csv
#
# Or set environment variables before running:
#   export ASTRA_TRAIN_CSV=/path/to/train.csv
#   export ASTRA_TEST_CSV=/path/to/test.csv
#   ./ml/run_all.sh
#
# Results are written to ml/results/
# The best CatBoost model is saved to ml/models/ and backend/model/
# ─────────────────────────────────────────────────────────────────────────────

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

TRAIN_CSV="${ASTRA_TRAIN_CSV:-}"
TEST_CSV="${ASTRA_TEST_CSV:-}"

# Parse CLI args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --train) TRAIN_CSV="$2"; shift 2;;
    --test)  TEST_CSV="$2";  shift 2;;
    *) echo "Unknown argument: $1"; exit 1;;
  esac
done

if [[ -z "$TRAIN_CSV" || -z "$TEST_CSV" ]]; then
  echo "ERROR: Provide --train and --test paths or set ASTRA_TRAIN_CSV / ASTRA_TEST_CSV"
  exit 1
fi

export ASTRA_TRAIN_CSV="$TRAIN_CSV"
export ASTRA_TEST_CSV="$TEST_CSV"

echo "========================================================"
echo "ASTRA ML Pipeline"
echo "Train: $TRAIN_CSV"
echo "Test:  $TEST_CSV"
echo "========================================================"
echo ""

cd "$ROOT_DIR"

echo "Step 1/6: Dataset Audit"
python ml/audit/01_dataset_audit.py --train "$TRAIN_CSV" --test "$TEST_CSV"
echo ""

echo "Step 2/6: Full Pipeline (5-model comparison + CatBoost train)"
python ml/training/full_pipeline.py --train "$TRAIN_CSV" --test "$TEST_CSV"
echo ""

echo "Step 3/6: Ablation Study"
python ml/experiments/ablation_study.py --train "$TRAIN_CSV" --test "$TEST_CSV"
echo ""

echo "Step 4/6: Early Warning Evaluation"
python ml/experiments/early_warning.py --train "$TRAIN_CSV" --test "$TEST_CSV"
echo ""

echo "Step 5/6: Chronological Evaluation"
python ml/experiments/chronological_eval.py --train "$TRAIN_CSV" --test "$TEST_CSV"
echo ""

echo "Step 6/9: Calibration Analysis"
python ml/experiments/calibration.py --train "$TRAIN_CSV" --test "$TEST_CSV"
echo ""

echo "Step 7/9: SHAP Explainability"
python ml/experiments/explainability_shap.py --train "$TRAIN_CSV" --test "$TEST_CSV"
echo ""

echo "Step 8/9: Multi-Horizon Early Warning"
python ml/experiments/multi_horizon_warning.py --train "$TRAIN_CSV" --test "$TEST_CSV"
echo ""

echo "Step 9/9: Mission & Regime Cross-Validation"
python ml/experiments/mission_validation.py --train "$TRAIN_CSV" --test "$TEST_CSV"
echo ""

echo "========================================================"
echo "ALL STEPS COMPLETE"
echo "Results saved in: ml/results/"
ls -la ml/results/
echo "========================================================"
