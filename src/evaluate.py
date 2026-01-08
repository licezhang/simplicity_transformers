"""
Evaluation script for few-shot generalization.
"""

import torch
from torch.utils.data import DataLoader
import argparse
import os

from .data import ICLDataset
from .model import create_model
from .utils import set_seed, collate_fn, load_checkpoint


def evaluate_few_shot(model, task_config, vocab, k_shots, eval_size, device):
    """
    Evaluate model with k-shot prompts.

    Args:
        model: Trained model
        task_config: Configuration for the task
        vocab: Token vocabulary
        k_shots: Number of labeled examples in context
        eval_size: Number of evaluation prompts
        device: torch device

    Returns:
        accuracy: Accuracy on the evaluation set
    """
    # Create dataset with k examples per prompt
    eval_config = task_config.copy()
    eval_config['n_examples_per_prompt'] = k_shots
    eval_config['dataset_size'] = eval_size

    eval_dataset = ICLDataset(**eval_config)

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, vocab)
    )

    model.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for input_ids, labels, padding_mask in eval_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            padding_mask = padding_mask.to(device)

            logits = model(input_ids, padding_mask)
            predictions = torch.argmax(logits, dim=1)
            total_correct += (predictions == labels).sum().item()
            total_samples += input_ids.size(0)

    accuracy = total_correct / total_samples
    return accuracy


def main(args):
    # Set seed
    set_seed(args.seed)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Task configuration (should match training)
    task_config = {
        'task_type': args.task_type,
        'n_features': args.n_features,
        'p_noise': args.p_noise,
        'seed': args.seed + 1000  # Different seed for evaluation
    }

    # Create a dummy dataset to get vocab
    dummy_dataset = ICLDataset(**task_config, n_examples_per_prompt=5, dataset_size=1)
    vocab = dummy_dataset.get_vocab()

    # Create model
    model_config = {
        'd_model': args.d_model,
        'n_heads': args.n_heads,
        'n_layers': args.n_layers,
        'dim_feedforward': args.dim_feedforward,
        'dropout': args.dropout
    }

    print("Creating model...")
    model = create_model(len(vocab), model_config)
    model = model.to(device)

    # Load checkpoint
    checkpoint_path = args.checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint from {checkpoint_path}...")
    load_checkpoint(model, None, checkpoint_path, device)

    # Evaluate with different k-shot settings
    print(f"\nEvaluating few-shot generalization...")
    print(f"Task type: {args.task_type}")
    print(f"Number of features: {args.n_features}")
    print("-" * 50)

    k_shots_list = args.k_shots
    results = {}

    for k in k_shots_list:
        print(f"\nEvaluating with k={k} labeled examples...")
        accuracy = evaluate_few_shot(
            model, task_config, vocab, k, args.eval_size, device
        )
        results[k] = accuracy
        print(f"k={k}: Accuracy = {accuracy:.4f}")

    # Summary
    print("\n" + "=" * 50)
    print("Few-shot Evaluation Results")
    print("=" * 50)
    for k, acc in results.items():
        print(f"k={k:3d} shots: {acc:.4f}")
    print("=" * 50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate ICL Transformer')

    # Task parameters (should match training)
    parser.add_argument('--task_type', type=str, default='single_feature',
                        choices=['single_feature', 'xor'],
                        help='Type of task')
    parser.add_argument('--n_features', type=int, default=4,
                        help='Number of features per stimulus')
    parser.add_argument('--p_noise', type=float, default=0.0,
                        help='Label noise probability')

    # Model parameters (should match training)
    parser.add_argument('--d_model', type=int, default=64,
                        help='Model dimension')
    parser.add_argument('--n_heads', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--n_layers', type=int, default=2,
                        help='Number of transformer layers')
    parser.add_argument('--dim_feedforward', type=int, default=256,
                        help='Feedforward dimension')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout probability')

    # Evaluation parameters
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--k_shots', type=int, nargs='+', default=[5, 10, 50],
                        help='Number of shots to evaluate (e.g., 5 10 50)')
    parser.add_argument('--eval_size', type=int, default=500,
                        help='Number of evaluation prompts per k-shot setting')

    # Other
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    main(args)
