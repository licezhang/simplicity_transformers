#!/bin/bash

# run_experiment.sh - Full pipeline for ICL experiment

set -e  # Exit on error

echo "=============================================="
echo "In-Context Learning Experiment"
echo "=============================================="
echo ""

# Configuration
TASK_TYPE="single_feature"  # Options: single_feature, xor
N_FEATURES=4
N_EXAMPLES=10
TRAIN_SIZE=5000
EVAL_SIZE=500
P_NOISE=0.0
EPOCHS=20
BATCH_SIZE=256
SEED=42
OUTPUT_DIR="outputs"

# Create output directory
mkdir -p $OUTPUT_DIR

echo "Configuration:"
echo "  Task type: $TASK_TYPE"
echo "  Features: $N_FEATURES"
echo "  Examples per prompt: $N_EXAMPLES"
echo "  Train size: $TRAIN_SIZE"
echo "  Eval size: $EVAL_SIZE"
echo "  Noise: $P_NOISE"
echo "  Epochs: $EPOCHS"
echo ""

# Step 1: Generate and inspect data
echo "=============================================="
echo "Step 1: Generating sample data..."
echo "=============================================="
python scripts/generate_data.py \
    --task_type $TASK_TYPE \
    --n_features $N_FEATURES \
    --n_examples $N_EXAMPLES \
    --num_samples 5 \
    --p_noise $P_NOISE \
    --seed $SEED \
    --show 3
echo ""

# Step 2: Train model
echo "=============================================="
echo "Step 2: Training model..."
echo "=============================================="
python -m src.train \
    --task_type $TASK_TYPE \
    --n_features $N_FEATURES \
    --n_examples $N_EXAMPLES \
    --train_size $TRAIN_SIZE \
    --eval_size $EVAL_SIZE \
    --p_noise $P_NOISE \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --seed $SEED \
    --output_dir $OUTPUT_DIR
echo ""

# Step 3: Evaluate few-shot generalization
echo "=============================================="
echo "Step 3: Evaluating few-shot generalization..."
echo "=============================================="
python -m src.evaluate \
    --task_type $TASK_TYPE \
    --n_features $N_FEATURES \
    --p_noise $P_NOISE \
    --checkpoint $OUTPUT_DIR/best_model.pt \
    --k_shots 5 10 15 \
    --eval_size 500 \
    --seed $SEED
echo ""

echo "=============================================="
echo "Experiment complete!"
echo "Results saved in: $OUTPUT_DIR"
echo "=============================================="
