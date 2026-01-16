"""
Utility functions for the ICL experiment.
"""

import torch
import numpy as np
import random
from typing import List, Tuple, Dict


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # Make PyTorch deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def tokenize_prompt(prompt: str, vocab: Dict[str, int]) -> List[int]:
    """
    Tokenize a prompt string into token indices.

    Args:
        prompt: String like "ACEG:0;BDFH:1;ACFG:"
        vocab: Dictionary mapping tokens to indices

    Returns:
        List of token indices
    """
    tokens = []
    for char in prompt:
        if char in vocab:
            tokens.append(vocab[char])
        else:
            raise ValueError(f"Unknown token: {char}")
    return tokens


def collate_fn(batch: List[Tuple[str, int]], vocab: Dict[str, int]):
    """
    Collate function for DataLoader.
    Tokenizes prompts and pads sequences.

    Args:
        batch: List of (prompt, label) tuples
        vocab: Token vocabulary

    Returns:
        input_ids: [batch_size, max_seq_len] - padded token indices
        labels: [batch_size] - target labels
        padding_mask: [batch_size, max_seq_len] - True for padding positions
    """
    prompts, labels, is_exception = zip(*batch)

    # Tokenize all prompts
    tokenized = [tokenize_prompt(p, vocab) for p in prompts]

    # Find max length
    max_len = max(len(t) for t in tokenized)

    # Pad sequences
    pad_idx = vocab['<PAD>']
    input_ids = []
    padding_mask = []

    for tokens in tokenized:
        # Pad to max_len
        padding_length = max_len - len(tokens)
        padded_tokens = tokens + [pad_idx] * padding_length
        mask = [False] * len(tokens) + [True] * padding_length

        input_ids.append(padded_tokens)
        padding_mask.append(mask)

    # Convert to tensors
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)
    padding_mask = torch.tensor(padding_mask, dtype=torch.bool)

    return input_ids, labels, padding_mask, torch.tensor(is_exception, dtype=torch.bool)


def save_checkpoint(model, optimizer, epoch, loss, path):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(model, optimizer, path, device):
    """Load model checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Checkpoint loaded from {path} (epoch {epoch}, loss {loss:.4f})")
    return epoch, loss


def count_parameters(model):
    """Count the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
