"""
Data generation for in-context learning experiments.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import List, Tuple, Dict


class StimulusGenerator:
    """Generates stimuli with N features, each having 2 possible values."""

    def __init__(self, n_features: int = 4, seed: int = 42):
        self.n_features = n_features
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

        alphabet = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')

        # Sample 2 symbols PER feature
        # Prefer sampling without replacement when possible; fall back to replacement if n_features is large.
        n_needed = 2 * n_features
        if n_needed <= len(alphabet):
            chosen = list(np.random.choice(alphabet, size=n_needed, replace=False))
        else:
            chosen = list(np.random.choice(alphabet, size=n_needed, replace=True))

        self.feature_symbols = []
        for f in range(n_features):
            self.feature_symbols.append([chosen[2*f], chosen[2*f + 1]])

    def generate_stimulus(self) -> str:
        """Generate a single stimulus (e.g., 'ACEG')."""
        stimulus = ''
        for feature_idx in range(self.n_features):
            value = np.random.randint(0, 2)
            stimulus += self.feature_symbols[feature_idx][value]
        return stimulus

    def stimulus_to_feature_values(self, stimulus: str) -> List[int]:
        """Convert stimulus string to feature values [0 or 1 for each feature]."""
        values = []
        for i, char in enumerate(stimulus):
            if char == self.feature_symbols[i][0]:
                values.append(0)
            else:
                values.append(1)
        return values


class TaskGenerator:
    """Generates tasks (rules) for labeling stimuli."""

    def __init__(self, task_type: str, n_features: int = 4,
                 relevant_features: List[int] = None,
                 p_noise: float = 0.0, seed: int = None):
        """
        Args:
            task_type: 'single_feature', 'xor', or 'simple_rule_with_exception'
            n_features: Number of features per stimulus
            relevant_features: Which features determine the label
            p_noise: Probability of flipping label (for single_feature only)
            seed: Random seed (if None, won't set seed for randomization)
        """
        self.task_type = task_type
        self.n_features = n_features
        self.p_noise = p_noise
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

        # Choose relevant features if not specified
        if relevant_features is None:
            if task_type == 'single_feature':
                self.relevant_features = [np.random.randint(0, n_features)]
            elif task_type == 'xor':
                # Choose 2 features for XOR
                self.relevant_features = list(np.random.choice(n_features, size=2, replace=False))
            elif task_type == 'simple_rule_with_exception':
                # Choose 1 feature for simple rule, 2 other features for XOR exception
                # Format: [simple_rule_feature, exception_feature_1, exception_feature_2]
                all_features = list(range(n_features))
                np.random.shuffle(all_features)
                self.relevant_features = all_features[:3]  # First is simple rule, next 2 are XOR exception
            else:
                raise ValueError(f"Unknown task_type: {task_type}")
        else:
            self.relevant_features = relevant_features

    def get_label(self, feature_values: List[int], apply_noise: bool = True) -> int:
        """Compute label based on feature values and task rule.

        Args:
            feature_values: Binary feature values
            apply_noise: Whether to apply label noise (should be False for query labels)
        """
        if self.task_type == 'single_feature':
            # Label is determined by single feature
            label = feature_values[self.relevant_features[0]]
            # Add noise only if requested (not for query labels)
            if apply_noise and np.random.random() < self.p_noise:
                label = 1 - label
        elif self.task_type == 'xor':
            # Label is XOR of two features
            label = feature_values[self.relevant_features[0]] ^ feature_values[self.relevant_features[1]]
        elif self.task_type == 'simple_rule_with_exception':
            # Label is based on simple rule feature, but with exceptions for specific feature combination
            # Simple rule: label = relevant_features[0]
            # Exception: when BOTH relevant_features[1] and relevant_features[2] = 1, flip the label
            # This makes exactly 1/4 of stimuli exceptions (only the (1,1) combination)
            simple_rule_label = feature_values[self.relevant_features[0]]
            exception_condition = (feature_values[self.relevant_features[1]] == 1 and
                                  feature_values[self.relevant_features[2]] == 1)

            if exception_condition:
                label = 1 - simple_rule_label  # Exception: flip the simple rule
            else:
                label = simple_rule_label  # Follow simple rule
        else:
            raise ValueError(f"Unknown task_type: {self.task_type}")

        return label


class ICLDataset(Dataset):
    """
    In-context learning dataset.
    Each example is a sequence of k labeled stimuli + 1 unlabeled query stimulus.
    Format: "ACEG:0;BDFH:1;ACFG:0;ADEH:"
    Target is the label of the query stimulus.
    """

    def __init__(self,
                 task_type: str = 'single_feature',
                 n_features: int = 4,
                 n_examples_per_prompt: int = 10,
                 dataset_size: int = 1000,
                 p_noise: float = 0.0,
                 relevant_features: List[int] = None,
                 seed: int = 42,
                 p_query_in_context: float = 0.0,
                 curriculum: bool = False):
        """
        Args:
            task_type: 'single_feature' or 'xor'
            n_features: Number of features per stimulus
            n_examples_per_prompt: Number of labeled examples in context
            dataset_size: Total number of prompts to generate
            p_noise: Label noise probability (single_feature only)
            relevant_features: Which features determine labels
            seed: Random seed
            p_query_in_context: Probability that query stimulus also appears in context
            curriculum: If True, order examples for simple_rule_with_exception so rule-consistent come before exceptions
        """
        self.task_type = task_type
        self.n_features = n_features
        self.n_examples_per_prompt = n_examples_per_prompt
        self.dataset_size = dataset_size
        self.seed = seed
        self.p_query_in_context = p_query_in_context
        self.curriculum = curriculum

        # Initialize generators
        self.stimulus_gen = StimulusGenerator(n_features, seed)
        self.task_gen = TaskGenerator(task_type, n_features, relevant_features, p_noise, seed)

        # Generate all prompts upfront for determinism
        np.random.seed(seed)
        self.prompts = []
        self.labels = []
        self.is_exception = []

        for _ in range(dataset_size):
            prompt, label, is_exc = self._generate_prompt()
            self.prompts.append(prompt)
            self.labels.append(label)
            self.is_exception.append(is_exc)


    def _generate_prompt(self) -> Tuple[str, int]:
        """Generate a single prompt with k labeled examples + 1 query."""
        stimulus_gen = StimulusGenerator(
            n_features=self.n_features,
            seed=None
        )

        task_gen = TaskGenerator(
            task_type=self.task_type,
            n_features=self.n_features,
            relevant_features=None,
            p_noise=self.task_gen.p_noise,
            seed=None
        )

        examples = []
        used_stimuli = set()

        if self.p_query_in_context == 0:
            query_stimulus = stimulus_gen.generate_stimulus()

        for _ in range(self.n_examples_per_prompt):
            stimulus = stimulus_gen.generate_stimulus()
            # check if stimulus is the query stimulus when p_query_in_context == 0
            while self.p_query_in_context == 0 and stimulus == query_stimulus:
                stimulus = stimulus_gen.generate_stimulus()
            feature_values = stimulus_gen.stimulus_to_feature_values(stimulus)
            label = task_gen.get_label(feature_values, apply_noise=True)
            examples.append(f"{stimulus}:{label}")
            used_stimuli.add(stimulus)

        # Apply curriculum ordering if enabled for simple_rule_with_exception
        if self.curriculum and self.task_type == 'simple_rule_with_exception':
            # Separate examples into rule-consistent and exceptions
            rule_consistent = []
            exceptions = []

            for example in examples:
                stimulus = example.split(':')[0]
                feature_values = stimulus_gen.stimulus_to_feature_values(stimulus)

                # Check if this is an exception (both exception features are 1)
                exception_condition = (
                    feature_values[task_gen.relevant_features[1]] == 1 and
                    feature_values[task_gen.relevant_features[2]] == 1
                )

                if exception_condition:
                    exceptions.append(example)
                else:
                    rule_consistent.append(example)

            # Reorder: rule-consistent first, then exceptions
            examples = rule_consistent + exceptions

        # With probability p_query_in_context, force the query to be one of the demos (pure copy task)
        if self.p_query_in_context == 0:
            pass
        elif np.random.random() < self.p_query_in_context and len(used_stimuli) > 0:
            query_stimulus = np.random.choice(list(used_stimuli))
        else:
            query_stimulus = stimulus_gen.generate_stimulus()

        query_feature_values = stimulus_gen.stimulus_to_feature_values(query_stimulus)
        query_label = task_gen.get_label(query_feature_values, apply_noise=False)

        # Query marker AFTER stimulus, then ':' (no label token after)
        examples.append(f"{query_stimulus}?:")

        is_exception = self._is_exception(query_feature_values, task_gen)

        prompt = ";".join(examples)
        return prompt, query_label, is_exception
    

    def _is_exception(self, feature_values, task_gen):
        """
        Returns True if feature_values correspond to an exception
        for simple_rule_with_exception tasks.
        """
        if self.task_type != 'simple_rule_with_exception':
            return False

        return (
            feature_values[task_gen.relevant_features[1]] == 1 and
            feature_values[task_gen.relevant_features[2]] == 1
        )


    def __len__(self) -> int:
        return self.dataset_size

    def __getitem__(self, idx: int) -> Tuple[str, int]:
        return self.prompts[idx], self.labels[idx], self.is_exception[idx]


    def get_vocab(self) -> Dict[str, int]:
        """
        Build vocabulary for the dataset.
        Vocab includes: all possible alphabet characters (A-Z), digits 0-1, separator ;, colon :
        Since each prompt uses a randomly selected pair of symbols from the alphabet,
        we need to include all possible symbols.
        """
        vocab = {'<PAD>': 0}
        idx = 1

        # Add all possible alphabet characters (A-Z) that could be used as feature symbols
        alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        for char in alphabet:
            vocab[char] = idx
            idx += 1

        # Add label tokens and separators
        for token in ['0', '1', ':', ';', '?']:
            vocab[token] = idx
            idx += 1

        return vocab


def create_dataloaders(train_config: dict, eval_config: dict = None):
    """
    Create train and eval dataloaders.

    Args:
        train_config: Config dict with keys like task_type, n_features, etc.
        eval_config: Optional separate config for eval set

    Returns:
        train_dataset, eval_dataset, vocab
    """
    train_dataset = ICLDataset(**train_config)

    # Use same config for eval if not specified
    if eval_config is None:
        eval_config = train_config.copy()
        eval_config['seed'] = train_config['seed'] + 1  # Different seed for eval
        eval_config['dataset_size'] = train_config.get('eval_size', 200)

    eval_dataset = ICLDataset(**eval_config)
    vocab = train_dataset.get_vocab()

    return train_dataset, eval_dataset, vocab
