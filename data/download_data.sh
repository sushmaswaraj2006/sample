#!/usr/bin/env bash
# ASTRA Dataset Download & Verification Helper
set -e

DATA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DATA_DIR"

echo "=========================================================="
echo "ASTRA Dataset Download & Verification Helper"
echo "Target directory: $DATA_DIR"
echo "=========================================================="

TRAIN_FILE="train_data.csv"
TEST_FILE="test_data.csv"

# 1. Check if files exist locally or in Downloads
if [[ -f "$TRAIN_FILE" && -f "$TEST_FILE" ]]; then
    echo "[OK] Found $TRAIN_FILE and $TEST_FILE in $DATA_DIR"
    exit 0
fi

# Check ~/Downloads fallback
USER_DOWNLOADS="$HOME/Downloads"
if [[ -f "$USER_DOWNLOADS/$TRAIN_FILE" && -f "$USER_DOWNLOADS/$TEST_FILE" ]]; then
    echo "Found datasets in $USER_DOWNLOADS! Symlinking to $DATA_DIR..."
    ln -sf "$USER_DOWNLOADS/$TRAIN_FILE" "$DATA_DIR/$TRAIN_FILE"
    ln -sf "$USER_DOWNLOADS/$TEST_FILE" "$DATA_DIR/$TEST_FILE"
    echo "[OK] Symlinked successfully."
    exit 0
fi

echo ""
echo "Datasets not found in $DATA_DIR or $USER_DOWNLOADS."
echo ""
echo "To obtain the official European Space Agency (ESA) SDC-II benchmark dataset:"
echo "1. Register/Login at ESA Kelvins: https://kelvins.esa.int/collision-avoidance-challenge/"
echo "2. Download 'train_data.csv' and 'test_data.csv'"
echo "3. Move them into $DATA_DIR"
echo ""
echo "Or specify external paths using environment variables:"
echo "  export ASTRA_TRAIN_CSV=/path/to/train_data.csv"
echo "  export ASTRA_TEST_CSV=/path/to/test_data.csv"
echo "=========================================================="
