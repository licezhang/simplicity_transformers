#!/usr/bin/env python3
"""
Standalone script to generate and inspect ICL data.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data import ICLDataset
import argparse


def main(args):
    print(f"Generating {args.task_type} task data...")
    print(f"Features: {args.n_features}, Examples per prompt: {args.n_examples}")
    print(f"Noise: {args.p_noise}")
    print("-" * 60)

    # Create dataset
    dataset = ICLDataset(
        task_type=args.task_type,
        n_features=args.n_features,
        n_examples_per_prompt=args.n_examples,
        dataset_size=args.num_samples,
        p_noise=args.p_noise,
        seed=args.seed,
        curriculum=args.curriculum
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Vocabulary: {dataset.get_vocab()}")
    print(f"Relevant features: {dataset.task_gen.relevant_features}")
    print("-" * 60)

    # Show some examples
    print(f"\nShowing first {min(args.show, len(dataset))} examples:\n")

    for i in range(min(args.show, len(dataset))):
        prompt, label, _ = dataset[i]
        print(f"Example {i+1}:")
        print(f"Prompt: {prompt}")
        print(f"Label:  {label}")
        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate ICL data')

    parser.add_argument('--task_type', type=str, default='single_feature',
                        choices=['single_feature', 'xor', 'simple_rule_with_exception'],
                        help='Type of task')
    parser.add_argument('--n_features', type=int, default=4,
                        help='Number of features')
    parser.add_argument('--n_examples', type=int, default=10,
                        help='Number of labeled examples per prompt')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of prompts to generate')
    parser.add_argument('--p_noise', type=float, default=0.0,
                        help='Label noise probability')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--show', type=int, default=5,
                        help='Number of examples to display')
    parser.add_argument('--curriculum', action='store_true',
                        help='Order examples with rule-consistent before exceptions (for simple_rule_with_exception)')

    args = parser.parse_args()
    main(args)
