#!/usr/bin/env bash
set -e

#########################
# ARGUMENTS
#########################
EXP_GROUP=$1    # ví dụ: exp1, exp2, exp3
SCENE=$2        # ví dụ: pulling hoặc cutting

if [ -z "$EXP_GROUP" ] || [ -z "$SCENE" ]; then
    echo "Usage: ./run.sh <exp_group> <scene>"
    echo "Example: ./run.sh exp7 pulling"
    exit 1
fi

#########################
# PATHS
#########################

DATA_ROOT="/media/dial2/Ubuntu Volume/dataset/EndoNeRF-EC"
DATA_PATH="$DATA_ROOT/$SCENE"

EXP_NAME="$EXP_GROUP/endonerf-ec/$SCENE"
MODEL_PATH="output/$EXP_NAME"

echo "======================================"
echo "Experiment group : $EXP_GROUP"
echo "Scene            : $SCENE"
echo "Dataset          : $DATA_PATH"
echo "Exp name         : $EXP_NAME"
echo "Model output     : $MODEL_PATH"
echo "======================================"

#########################
# 1. TRAIN
#########################

echo "[1/2] Training..."
python train.py \
    -s "$DATA_PATH" \
    --expname "$EXP_NAME"

#########################
# 2. RENDER
#########################

echo "[2/2] Rendering..."
python render.py \
    --model_path "$MODEL_PATH" \
    --skip_test \
    --skip_train

echo "DONE."
