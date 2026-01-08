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

        # Randomly sample 2 symbols to use for all features
        alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        self.symbols = list(np.random.choice(list(alphabet), size=2, replace=False))

        # Use the same 2 symbols for all features
        self.feature_symbols = [self.symbols for _ in range(n_features)]

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
            # Find which value (0 or 1) this character represents for feature i
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
            task_type: 'single_feature' or 'xor'
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
                 seed: int = 42):
        """
        Args:
            task_type: 'single_feature' or 'xor'
            n_features: Number of features per stimulus
            n_examples_per_prompt: Number of labeled examples in context
            dataset_size: Total number of prompts to generate
            p_noise: Label noise probability (single_feature only)
            relevant_features: Which features determine labels
            seed: Random seed
        """
        self.task_type = task_type
        self.n_features = n_features
        self.n_examples_per_prompt = n_examples_per_prompt
        self.dataset_size = dataset_size
        self.seed = seed

        # Initialize generators
        self.stimulus_gen = StimulusGenerator(n_features, seed)
        self.task_gen = TaskGenerator(task_type, n_features, relevant_features, p_noise, seed)

        # Generate all prompts upfront for determinism
        np.random.seed(seed)
        self.prompts = []
        self.labels = []

        for _ in range(dataset_size):
            prompt, label = self._generate_prompt()
            self.prompts.append(prompt)
            self.labels.append(label)

    def _generate_prompt(self) -> Tuple[str, int]:
        """Generate a single prompt with k labeled examples + 1 query."""
        # Create a new stimulus generator for each prompt to use different symbols
        stimulus_gen = StimulusGenerator(
            n_features=self.n_features,
            seed=None  # Don't set seed to get different symbols each time
        )

        # Create a new task generator for each prompt to randomize relevant features
        task_gen = TaskGenerator(
            task_type=self.task_type,
            n_features=self.n_features,
            relevant_features=None,  # Will be randomly chosen
            p_noise=self.task_gen.p_noise,
            seed=None  # Don't set seed to allow randomization
        )

        examples = []
        used_stimuli = set()

        # Generate k labeled examples (noise can be applied here)
        for _ in range(self.n_examples_per_prompt):
            stimulus = stimulus_gen.generate_stimulus()
            feature_values = stimulus_gen.stimulus_to_feature_values(stimulus)
            label = task_gen.get_label(feature_values, apply_noise=True)
            examples.append(f"{stimulus}:{label}")
            used_stimuli.add(stimulus)

        # Generate query stimulus (NO noise applied to ground truth label)
        # Ensure query stimulus is not one that was already shown with a label
        query_stimulus = stimulus_gen.generate_stimulus()
        while query_stimulus in used_stimuli:
            query_stimulus = stimulus_gen.generate_stimulus()

        query_feature_values = stimulus_gen.stimulus_to_feature_values(query_stimulus)
        query_label = task_gen.get_label(query_feature_values, apply_noise=False)
        examples.append(f"{query_stimulus}:")

        # Join with separator
        prompt = ";".join(examples)

        return prompt, query_label

    def __len__(self) -> int:
        return self.dataset_size

    def __getitem__(self, idx: int) -> Tuple[str, int]:
        return self.prompts[idx], self.labels[idx]

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
        for token in ['0', '1', ':', ';']:
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
