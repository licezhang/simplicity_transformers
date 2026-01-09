"""
Tiny Transformer model for in-context learning.
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]
        """
        return x + self.pe[:, :x.size(1), :]


# class TinyTransformer(nn.Module):
#     """
#     Small Transformer encoder for in-context learning.
#     Predicts the label of the final (query) stimulus in a sequence.
#     """

#     def __init__(self,
#                  vocab_size: int,
#                  d_model: int = 64,
#                  n_heads: int = 8,
#                  n_layers: int = 4,
#                  dim_feedforward: int = 256,
#                  dropout: float = 0.1,
#                  max_seq_len: int = 200):
#         """
#         Args:
#             vocab_size: Size of token vocabulary
#             d_model: Embedding dimension
#             n_heads: Number of attention heads
#             n_layers: Number of transformer layers
#             dim_feedforward: Dimension of feedforward network
#             dropout: Dropout probability
#             max_seq_len: Maximum sequence length
#         """
#         super().__init__()

#         self.d_model = d_model

#         # Token embedding
#         self.embedding = nn.Embedding(vocab_size, d_model)

#         # Positional encoding
#         self.pos_encoder = PositionalEncoding(d_model, max_seq_len)

#         # Transformer encoder
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=d_model,
#             nhead=n_heads,
#             dim_feedforward=dim_feedforward,
#             dropout=dropout,
#             batch_first=True
#         )
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

#         # Output head for binary classification
#         self.output_head = nn.Linear(d_model, 2)

#         # Initialize weights
#         self._init_weights()

#     def _init_weights(self):
#         """Initialize weights."""
#         initrange = 0.1
#         self.embedding.weight.data.uniform_(-initrange, initrange)
#         self.output_head.bias.data.zero_()
#         self.output_head.weight.data.uniform_(-initrange, initrange)

#     def forward(self, x, padding_mask=None):
#         """
#         Args:
#             x: [batch_size, seq_len] - token indices
#             padding_mask: [batch_size, seq_len] - True for padding positions

#         Returns:
#             logits: [batch_size, 2] - logits for binary classification
#         """
#         # Embedding: [batch_size, seq_len, d_model]
#         x = self.embedding(x) * math.sqrt(self.d_model)

#         # Add positional encoding
#         x = self.pos_encoder(x)

#         # Transformer encoding
#         # padding_mask: True for positions to mask
#         x = self.transformer(x, src_key_padding_mask=padding_mask)

#         # Use the last token's representation for prediction
#         # (corresponds to the query stimulus)
#         if padding_mask is not None:
#             # Find the last non-padding position for each sequence
#             seq_lengths = (~padding_mask).sum(dim=1) - 1  # [batch_size]
#             last_token_repr = x[torch.arange(x.size(0)), seq_lengths]
#         else:
#             last_token_repr = x[:, -1, :]  # [batch_size, d_model]

#         # Classification head
#         logits = self.output_head(last_token_repr)  # [batch_size, 2]

#         return logits


class TinyTransformer(nn.Module):
    """
    Small Transformer encoder for in-context learning.
    Predicts the label of the final (query) stimulus in a sequence.
    """

    def __init__(self,
                 vocab_size: int,
                 d_model: int = 64,
                 n_heads: int = 8,
                 n_layers: int = 4,
                 dim_feedforward: int = 256,
                 dropout: float = 0.1,
                 max_seq_len: int = 200,
                 query_token_id: int = None,
                 use_causal_mask: bool = True):
        super().__init__()

        self.d_model = d_model
        self.query_token_id = query_token_id
        self.use_causal_mask = use_causal_mask

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.output_head = nn.Linear(d_model, 2)
        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.output_head.bias.data.zero_()
        self.output_head.weight.data.uniform_(-initrange, initrange)

    def _causal_mask(self, seq_len: int, device: torch.device):
        # True = masked (disallowed attention)
        return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)

    def encode(self, x, padding_mask=None):
        """
        Returns per-token hidden states: [B, T, d_model]
        """
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)

        attn_mask = None
        if self.use_causal_mask:
            attn_mask = self._causal_mask(x.size(1), x.device)

        x = self.transformer(x, mask=attn_mask, src_key_padding_mask=padding_mask)
        return x

    def forward(self, x, padding_mask=None):
        """
        Args:
            x: [batch_size, seq_len]
            padding_mask: [batch_size, seq_len] - True for padding positions
        Returns:
            logits: [batch_size, 2]
        """
        h = self.encode(x, padding_mask=padding_mask)  # [B,T,D]

        # Prefer query marker representation if provided/found
        if self.query_token_id is not None:
            qmask = (x == self.query_token_id)  # [B,T]
            if qmask.any():
                # take first occurrence per sequence
                qpos = torch.argmax(qmask.int(), dim=1)  # [B]
                query_repr = h[torch.arange(h.size(0), device=h.device), qpos]  # [B,D]
                return self.output_head(query_repr)

        # Fallback: last non-padding token
        if padding_mask is not None:
            seq_lengths = (~padding_mask).sum(dim=1) - 1
            last_token_repr = h[torch.arange(h.size(0), device=h.device), seq_lengths]
        else:
            last_token_repr = h[:, -1, :]

        return self.output_head(last_token_repr)



def create_model(vocab_size: int, model_config: dict = None):
    """
    Create a TinyTransformer model.

    Args:
        vocab_size: Size of vocabulary
        model_config: Optional dict with model hyperparameters

    Returns:
        model: TinyTransformer instance
    """
    if model_config is None:
            model_config = {}

    default_config = {
        'd_model': 64,
        'n_heads': 4,
        'n_layers': 2,
        'dim_feedforward': 256,
        'dropout': 0.1,
        'max_seq_len': 1000,
        'query_token_id': None,
        'use_causal_mask': True,
    }

    default_config.update(model_config)
    model = TinyTransformer(vocab_size=vocab_size, **default_config)
    return model