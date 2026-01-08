# In-Context Learning Transformer

A minimal PyTorch implementation for studying in-context learning with synthetic boolean tasks.

## Overview

This repository provides a minimal, modular framework for training tiny Transformers on synthetic in-context learning tasks. The model learns to predict labels based on examples provided in the prompt context.

## Setup

```bash
pip install -r requirements.txt
```

## Quick Start

Run the full experiment pipeline:

```bash
./scripts/run_experiment.sh
```

This will:
1. Generate and display sample data
2. Train a Transformer model
3. Evaluate few-shot generalization (k=5, 10, 50)

## Data Format

Each training example is a sequence of labeled stimuli followed by one unlabeled query stimulus:

```
ACEG:0;BDFH:1;ACFG:0;ADEH:
```

- Each stimulus has N features (default N=4)
- Each feature has two possible values represented by unique symbols:
  - Feature 0: A, B
  - Feature 1: C, D
  - Feature 2: E, F
  - Feature 3: G, H
- `;` separates stimuli
- `:` separates stimulus from label
- Target: predict the label (0 or 1) of the final stimulus

## Task Types

### 1. Single-Feature Rule
One feature determines the label with optional noise.

```bash
python -m src.train --task_type single_feature --p_noise 0.1
```

### 2. XOR Rule
Label is the XOR of two features.

```bash
python -m src.train --task_type xor
```

## Usage

### Generate and Inspect Data

```bash
python scripts/generate_data.py \
    --task_type single_feature \
    --n_features 4 \
    --n_examples 10 \
    --num_samples 5 \
    --show 3
```

### Train a Model

```bash
python -m src.train \
    --task_type single_feature \
    --n_features 4 \
    --n_examples 10 \
    --train_size 5000 \
    --eval_size 500 \
    --epochs 20 \
    --batch_size 32 \
    --seed 42 \
    --output_dir outputs
```

Key training arguments:
- `--task_type`: Task type (`single_feature` or `xor`)
- `--n_features`: Number of features per stimulus
- `--n_examples`: Number of labeled examples in context
- `--p_noise`: Label noise probability (for single_feature)
- `--train_size`: Number of training prompts
- `--epochs`: Number of training epochs
- `--d_model`: Model dimension (default: 64)
- `--n_layers`: Number of Transformer layers (default: 2)
- `--n_heads`: Number of attention heads (default: 4)

### Evaluate Few-Shot Generalization

```bash
python -m src.evaluate \
    --task_type single_feature \
    --n_features 4 \
    --checkpoint outputs/best_model.pt \
    --k_shots 5 10 50 \
    --eval_size 500 \
    --seed 42
```

This evaluates the model with different numbers of in-context examples (5, 10, 50) to measure few-shot generalization.

## Project Structure

```
.
├── src/
│   ├── data.py        # Data generation and Dataset class
│   ├── model.py       # Tiny Transformer model
│   ├── train.py       # Training loop
│   ├── evaluate.py    # Few-shot evaluation
│   └── utils.py       # Utilities (tokenization, seeding, etc.)
├── scripts/
│   ├── generate_data.py      # Standalone data generation
│   └── run_experiment.sh     # Full pipeline demo
├── requirements.txt
└── README.md
```

## Model Architecture

- Tiny Transformer encoder
- Default configuration:
  - `d_model = 64`
  - `n_layers = 2`
  - `n_heads = 4`
  - `dim_feedforward = 256`
- Input: tokenized sequence of labeled stimuli + query
- Output: binary classification logits

## Examples

### Train on XOR task

```bash
python -m src.train \
    --task_type xor \
    --n_features 4 \
    --n_examples 20 \
    --train_size 10000 \
    --epochs 30
```

### Train with noisy labels

```bash
python -m src.train \
    --task_type single_feature \
    --n_features 4 \
    --p_noise 0.2 \
    --train_size 5000 \
    --epochs 25
```

### Evaluate with custom k-shot settings

```bash
python -m src.evaluate \
    --checkpoint outputs/best_model.pt \
    --task_type single_feature \
    --k_shots 3 7 15 30 100 \
    --eval_size 1000
```

## Customization

The codebase is designed to be minimal and easy to modify:

- **Add new task types**: Extend `TaskGenerator` in `src/data.py`
- **Change model architecture**: Modify `TinyTransformer` in `src/model.py`
- **Adjust training**: Edit hyperparameters in `src/train.py` or via command-line arguments
- **Custom evaluation**: Modify `src/evaluate.py` for different evaluation metrics

## Deterministic Experiments

All experiments are fully deterministic when using the same seed:

```bash
python -m src.train --seed 42
python -m src.evaluate --seed 42 --checkpoint outputs/best_model.pt
```

## Notes

- No explicit tokenization library needed; characters are tokenized directly
- Vocabulary is built automatically from the data
- Data is generated programmatically; no files are saved
- Minimal dependencies: only PyTorch and NumPy

## Citation

If you use this code, please cite appropriately or adapt for your own research.
