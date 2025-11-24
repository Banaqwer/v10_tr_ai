# ================================================================
#  TRANSFORMER ENGINE — V10-TR (CCT-90)
#  Full 4-layer Transformer Encoder for FX AI
#  Render-safe, CPU optimized
# ================================================================

import torch
import torch.nn as nn
import math

from utils import get_device


# ================================================================
#  POSITIONAL ENCODING (for chunk embeddings)
# ================================================================

class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding.
    Adds sequence order information for transformer.
    """

    def __init__(self, d_model, max_len=100):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        positions = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        denom = torch.exp(torch.arange(0, d_model, 2).float() *
                          (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(positions * denom)
        pe[:, 1::2] = torch.cos(positions * denom)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x shape: [batch, seq_len, d_model]
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


# ================================================================
#  MULTI-HEAD ATTENTION
# ================================================================

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()

        assert d_model % num_heads == 0, "d_model must divide evenly by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Linear projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Output projection
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x shape: [batch, seq_len, d_model]
        """

        B, T, D = x.size()

        # project Q, K, V
        q = self.W_q(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.W_k(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # attention weights
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # weighted sum of values
        context = torch.matmul(attn, v)

        # reshape and project
        context = context.transpose(1, 2).reshape(B, T, D)

        return self.W_o(context)


# ================================================================
#  FEED FORWARD NETWORK
# ================================================================

class FeedForward(nn.Module):
    def __init__(self, d_model, hidden_dim, dropout=0.1):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model)
        )

    def forward(self, x):
        return self.net(x)


# ================================================================
#  TRANSFORMER ENCODER LAYER
# ================================================================

class TransformerEncoderLayer(nn.Module):
    """
    One full transformer block = attention + residual + layernorm + MLP
    """

    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1):
        super().__init__()

        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.ff = FeedForward(d_model, ff_dim, dropout)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention block
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))

        # Feed-forward block
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))

        return x


# ================================================================
#  FULL TRANSFORMER (stacked encoder layers)
# ================================================================

class TransformerEncoder(nn.Module):
    """
    Full 4-layer transformer encoder.
    Input:  sequence of chunk embeddings (CCT-90: 6 tokens)
    Output: context vector (averaged)
    """

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.d_model = cfg.transformer_model_dim
        self.num_heads = cfg.transformer_heads
        self.dropout = cfg.transformer_dropout

        self.device = get_device()

        # 1) Expand chunk embeddings to Transformer model dimension
        self.input_proj = nn.Linear(cfg.cct_embed_dim, self.d_model)

        # 2) Positional encoding
        self.pos_encoder = PositionalEncoding(self.d_model, max_len=20)

        # 3) Stack encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=self.d_model,
                num_heads=self.num_heads,
                ff_dim=cfg.transformer_feedforward_dim,
                dropout=self.dropout
            )
            for _ in range(cfg.transformer_layers)
        ])

        # 4) Final normalization
        self.final_norm = nn.LayerNorm(self.d_model)

    # ============================================================
    #  FORWARD PASS
    # ============================================================

    def forward(self, embeddings):
        """
        embeddings shape: [num_chunks=6, embed_dim]
        """

        # Add batch dimension: [1, 6, embed_dim]
        x = embeddings.unsqueeze(0).to(self.device)

        # Project to transformer dim
        x = self.input_proj(x)  # shape: [1, 6, d_model]

        # Add positional encodings
        x = self.pos_encoder(x)

        # Pass through N transformer layers
        for layer in self.layers:
            x = layer(x)

        # Normalize
        x = self.final_norm(x)

        # Reduce sequence → single context vector
        # (mean pooling because CPU-safe, stable)
        context_vector = x.mean(dim=1)  # shape: [1, d_model]

        return context_vector.squeeze(0).cpu()
