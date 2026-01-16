"""
Evaluation script for few-shot generalization.
"""

import torch
from torch.utils.data import DataLoader
import argparse
import os
import pandas as pd
from pathlib import Path

from .data import ICLDataset
from .model import create_model
from .utils import set_seed, collate_fn, load_checkpoint


def evaluate_few_shot(model, task_config, vocab, k_shots, eval_size, device, batch_size=32):
    """
    Evaluate model with k-shot prompts.

    Returns:
        dict with aggregated metrics and a list of per-example dicts:
        {
          'summary': {overall, rule, exception, n_rule, n_exception},
          'examples': [ {'correct':0/1, 'is_exception':0/1}, ... ]
        }
    """
    eval_config = task_config.copy()
    eval_config['n_examples_per_prompt'] = k_shots
    eval_config['dataset_size'] = eval_size

    eval_dataset = ICLDataset(**eval_config)

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, vocab)
    )

    model.eval()
    rule_correct = rule_total = 0
    exc_correct = exc_total = 0

    examples = []  # per-example records

    with torch.no_grad():
        for input_ids, labels, padding_mask, is_exception in eval_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            padding_mask = padding_mask.to(device)
            is_exception = is_exception.to(device)

            logits = model(input_ids, padding_mask)
            preds = torch.argmax(logits, dim=1)

            # Move to CPU numpy for per-example bookkeeping
            preds_np = preds.cpu().numpy()
            labels_np = labels.cpu().numpy()
            is_exc_np = is_exception.cpu().numpy().astype(bool)

            for p, l, ie in zip(preds_np, labels_np, is_exc_np):
                correct = int(p == l)
                examples.append({'correct': correct, 'is_exception': int(ie)})

                if ie:
                    exc_correct += correct
                    exc_total += 1
                else:
                    rule_correct += correct
                    rule_total += 1

    overall = (rule_correct + exc_correct) / max(1, (rule_total + exc_total))
    rule_acc = rule_correct / max(1, rule_total)
    exc_acc = exc_correct / max(1, exc_total)

    summary = {
        'overall': overall,
        'rule': rule_acc,
        'exception': exc_acc,
        'n_rule': int(rule_total),
        'n_exception': int(exc_total)
    }

    return {'summary': summary, 'examples': examples}


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
        'n_examples_per_prompt': args.n_examples,
        'dataset_size': args.eval_size,
        'curriculum': args.curriculum,
        'p_noise': args.p_noise,
        'p_query_in_context': args.p_query_in_context,
        'seed': args.seed + 1000  # Different seed for evaluation
    }

    # Create a dummy dataset to get vocab
    dummy_dataset = ICLDataset(**task_config)
    vocab = dummy_dataset.get_vocab()
    query_token_id = vocab['?']

    # Create model
    model_config = {
        'd_model': args.d_model,
        'n_heads': args.n_heads,
        'n_layers': args.n_layers,
        'dim_feedforward': args.dim_feedforward,
        'dropout': args.dropout,
        'query_token_id': query_token_id,
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

       # Ensure results_dir exists
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    k_shots_list = args.k_shots
    summary_rows = []
    # We'll also save per-k example-level CSVs (one file per k)
    for k in k_shots_list:
        print(f"\nEvaluating with k={k} labeled examples...")
        out = evaluate_few_shot(
            model, task_config, vocab, k, args.eval_size, device, batch_size=32
        )
        summary = out['summary']
        examples = out['examples']

        summary_row = {
            'k_shots': k,
            'overall_acc': summary['overall'],
            'rule_acc': summary['rule'],
            'exception_acc': summary['exception'],
            'n_rule': summary['n_rule'],
            'n_exception': summary['n_exception']
        }
        summary_rows.append(summary_row)

        # Save per-example CSV for this k
        per_example_df = pd.DataFrame(examples)
        per_example_df['k_shots'] = k
        per_example_csv = os.path.join(args.results_dir, f'per_example_k={k}.csv')
        per_example_df.to_csv(per_example_csv, index=False)
        print(f"Saved per-example results to {per_example_csv}")

        print(f"k={k}:")
        print(f"  overall acc   = {summary['overall']:.3f}")
        print(f"  rule acc      = {summary['rule']:.3f}  (n={summary['n_rule']})")
        print(f"  exception acc = {summary['exception']:.3f}  (n={summary['n_exception']})")

    # Save summary CSV
    results_df = pd.DataFrame(summary_rows)
    summary_csv_path = os.path.join(args.results_dir, 'few_shot_summary.csv')
    results_df.to_csv(summary_csv_path, index=False)
    print(f"\nFew-shot summary saved to {summary_csv_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate ICL Transformer')

    # Task parameters (should match training)
    parser.add_argument('--task_type', type=str, default='single_feature',
                        choices=['single_feature', 'xor', 'simple_rule_with_exception'],
                        help='Type of task')
    parser.add_argument('--n_features', type=int, default=4,
                        help='Number of features per stimulus')
    parser.add_argument('--n_examples', type=int, default=16,
                        help='Number of labeled examples in context')
    parser.add_argument('--p_noise', type=float, default=0.0,
                        help='Label noise probability')
    parser.add_argument('--p_query_in_context', type=float, default=1,
                        help='Probability that the query stimulus is included among the in-context examples')
    parser.add_argument('--curriculum', type=bool, default=True,
                        help='Whether to use curriculum learning during eval')

    # Model parameters (should match training)
    parser.add_argument('--d_model', type=int, default=128,
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
    parser.add_argument('--k_shots', type=int, nargs='+', default=[2, 5, 10, 16, 18],
                        help='Number of shots to evaluate (e.g., 5 10 50)')
    parser.add_argument('--eval_size', type=int, default=500,
                        help='Number of evaluation prompts per k-shot setting')
    parser.add_argument('--use_causal_mask', action='store_true',
                help='Use causal attention mask (recommended for ICL)')
    parser.add_argument('--results_dir', type=str, default='outputs/eval_results',
                        help='Directory to save evaluation results')

    # Other
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    main(args)
