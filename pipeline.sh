#!/usr/bin/env bash
set -e

#########################
# ARGUMENTS
#########################
EXP_GROUP=$1    # exp1, exp2, exp3
DATASET=$2      # EndoNeRF-EC, StereoMIS
SCENE=$3        # pulling_soft_tissues, cutting_tissues_twice, P1_1, P1_2

if [ -z "$EXP_GROUP" ] || [ -z "$DATASET" ] || [ -z "$SCENE" ]; then
    echo "Usage: ./pipeline.sh <exp_group> <dataset> <scene>"
    echo "Example: ./pipeline.sh exp1 EndoNeRF pulling_soft_tissues"
    exit 1
fi

#########################
# PATHS
#########################

DATA_ROOT="/media/dial2/Ubuntu Volume/dataset"
DATA_PATH="$DATA_ROOT/$DATASET/$SCENE"

EXP_NAME="$EXP_GROUP/$DATASET/$SCENE"
MODEL_PATH="output/$EXP_NAME"

echo "======================================"
echo "Experiment group : $EXP_GROUP"
echo "Scene            : $SCENE"
echo "Dataset          : $DATASET"
echo "Exp name         : $EXP_NAME"
echo "Model output     : $MODEL_PATH"
echo "======================================"

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
# python render.py \
#     --model_path "$MODEL_PATH"

for iter in $(seq 1000 1000 30000)
do
    echo "Rendering iteration $iter"
    python render.py \
        --model_path "$MODEL_PATH" \
        --iteration $iter \
        --quiet
done

#########################
# 3. EVALUATION
#########################

echo "[3/3] Evaluating..."
python metrics.py \
    --model_path "$MODEL_PATH" 
