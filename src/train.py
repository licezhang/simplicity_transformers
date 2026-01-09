"""
Training script for the ICL transformer.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
from pathlib import Path

# set the current working directory to /mnt/big/jirko/simplicity_transformers
os.chdir('/mnt/big/jirko/simplicity_transformers') 
from .data import ICLDataset
from .model import create_model
from .utils import set_seed, collate_fn, save_checkpoint, count_parameters


def train_epoch(model, dataloader, criterion, optimizer, device, vocab, aux_loss_weight: float = 0.0):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
       
    colon_id = vocab[':']
    zero_id = vocab['0']
    one_id = vocab['1']

    for batch_idx, (input_ids, labels, padding_mask) in enumerate(dataloader):
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        padding_mask = padding_mask.to(device)

        # Forward pass
        optimizer.zero_grad()
        logits = model(input_ids, padding_mask)
        loss = criterion(logits, labels)

        # Optional auxiliary loss: predict each demo label from preceding ':'
        if aux_loss_weight > 0.0:
            with torch.set_grad_enabled(True):
                h = model.encode(input_ids, padding_mask=padding_mask)  # [B,T,D]
        
            # positions t where token is ':' and t+1 is 0 or 1
            if input_ids.size(1) >= 2:
                tok_t = input_ids[:, :-1]
                tok_next = input_ids[:, 1:]

                colon_pos = (tok_t == colon_id)
                next_is_label = (tok_next == zero_id) | (tok_next == one_id)
                mask = colon_pos & next_is_label  # [B,T-1]

                if mask.any():
                    # gather representations at ':' positions
                    idx_b, idx_t = mask.nonzero(as_tuple=True)
                    reps = h[idx_b, idx_t, :]  # [N,D]

                    # targets are next token (0/1) mapped to class index 0/1
                    targets_tok = tok_next[idx_b, idx_t]
                    targets = (targets_tok == one_id).long()

                    aux_logits = model.output_head(reps)  # [N,2]
                    aux_loss = criterion(aux_logits, targets)
                    loss = loss + aux_loss_weight * aux_loss

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track metrics
        total_loss += loss.item() * input_ids.size(0)
        predictions = torch.argmax(logits, dim=1)
        total_correct += (predictions == labels).sum().item()
        total_samples += input_ids.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for input_ids, labels, padding_mask in dataloader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            padding_mask = padding_mask.to(device)

            logits = model(input_ids, padding_mask)
            loss = criterion(logits, labels)

            total_loss += loss.item() * input_ids.size(0)
            predictions = torch.argmax(logits, dim=1)
            total_correct += (predictions == labels).sum().item()
            total_samples += input_ids.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return avg_loss, accuracy


def main(args):
    # Set seed for reproducibility
    set_seed(args.seed)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create datasets
    train_config = {
        'task_type': args.task_type,
        'n_features': args.n_features,
        'n_examples_per_prompt': args.n_examples,
        'dataset_size': args.train_size,
        'p_noise': args.p_noise,
        'seed': args.seed,
        'p_query_in_context': args.p_query_in_context,  # New parameter to control query inclusion
    }

    eval_config = train_config.copy()
    eval_config['dataset_size'] = args.eval_size
    eval_config['seed'] = args.seed + 1  # Different seed for eval

    print("Creating datasets...")
    train_dataset = ICLDataset(**train_config)
    eval_dataset = ICLDataset(**eval_config)
    vocab = train_dataset.get_vocab()
    query_token_id = vocab['?']

    print(f"Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Task type: {args.task_type}")
    print(f"Relevant features: {train_dataset.task_gen.relevant_features}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, vocab)
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, vocab)
    )

    # Create model
    model_config = {
        'd_model': args.d_model,
        'n_heads': args.n_heads,
        'n_layers': args.n_layers,
        'dim_feedforward': args.dim_feedforward,
        'dropout': args.dropout,
        'query_token_id': query_token_id,
        'use_causal_mask': args.use_causal_mask,
    }

    print("Creating model...")
    model = create_model(len(vocab), model_config)
    model = model.to(device)

    n_params = count_parameters(model)
    print(f"Number of trainable parameters: {n_params:,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    print(f"\nTraining for {args.epochs} epochs...")
    best_eval_acc = 0

    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, vocab=vocab, aux_loss_weight=args.aux_loss_weight)
        eval_loss, eval_acc = evaluate(model, eval_loader, criterion, device)

        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Eval Loss: {eval_loss:.4f} Acc: {eval_acc:.4f}")

        # Save best model
        if eval_acc > best_eval_acc:
            best_eval_acc = eval_acc
            save_path = os.path.join(args.output_dir, 'best_model.pt')
            save_checkpoint(model, optimizer, epoch, eval_loss, save_path)

    # Save final model
    final_path = os.path.join(args.output_dir, 'final_model.pt')
    save_checkpoint(model, optimizer, args.epochs, eval_loss, final_path)

    print(f"\nTraining complete! Best eval accuracy: {best_eval_acc:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ICL Transformer')

    # Data parameters
    parser.add_argument('--task_type', type=str, default='single_feature',
                        choices=['single_feature', 'xor'],
                        help='Type of task')
    parser.add_argument('--n_features', type=int, default=4,
                        help='Number of features per stimulus')
    parser.add_argument('--n_examples', type=int, default=20,
                        help='Number of labeled examples in context')
    parser.add_argument('--train_size', type=int, default=5000,
                        help='Number of training prompts')
    parser.add_argument('--eval_size', type=int, default=500,
                        help='Number of evaluation prompts')
    parser.add_argument('--p_noise', type=float, default=0.0,
                        help='Label noise probability (single_feature only)')
    parser.add_argument('--p_query_in_context', type=float, default=0,
                        help='Probability that the query stimulus is included among the in-context examples')

    # Model parameters
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

    # Training parameters
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='Learning rate')

    # Other
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Output directory for checkpoints')
    parser.add_argument('--use_causal_mask', action='store_true',
                    help='Use causal attention mask (recommended for ICL)')
    parser.add_argument('--aux_loss_weight', type=float, default=0.0,
                        help='Weight for auxiliary demo-label loss (0 disables)')

    args = parser.parse_args()

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)
