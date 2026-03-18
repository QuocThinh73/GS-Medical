#!/usr/bin/env bash
set -e

#########################
# ARGUMENTS
#########################
EXP_GROUP=$1    # exp1, exp2, exp3

if [ -z "$EXP_GROUP" ]; then
    echo "Usage: ./pipeline.sh <exp_group>"
    exit 1
fi

#########################
# CONFIG
#########################

DATASET="EndoNeRF"
DATA_ROOT="/media/dial2/Ubuntu Volume/dataset/EndoNeRF"

SCENES=(
    "cutting_tissues_twice"
    "pulling_soft_tissues"
)

#########################
# LOOP OVER SCENES
#########################

for SCENE in "${SCENES[@]}"
do
    echo "======================================"
    echo "Running scene: $SCENE"
    echo "======================================"

    DATA_PATH="$DATA_ROOT/$SCENE"
    EXP_NAME="$EXP_GROUP/$DATASET/$SCENE"
    MODEL_PATH="output/$EXP_NAME"

    #########################
    # 1. TRAIN
    #########################
    echo "[1/3] Training..."
    python train.py \
        -s "$DATA_PATH" \
        --expname "$EXP_NAME"

    #########################
    # 2. RENDER
    #########################
    echo "[2/3] Rendering..."
    for iter in $(seq 0 1000 30000)
    do
        echo "Rendering iteration $iter"
        python render.py \
            --model_path "$MODEL_PATH" \
            --iteration $iter \
            --skip_train \
            --skip_video
    done

    #########################
    # 3. EVALUATION
    #########################
    echo "[3/3] Evaluating..."
    python metrics.py \
        --model_path "$MODEL_PATH"

done