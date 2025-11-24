# ================================================================
#  CCT-90 EMBEDDING ENGINE
#  Converts 90-day sequences → compressed embeddings for Transformer
#  Author: Jack (V10-TR System)
# ================================================================

import torch
import torch.nn as nn
import numpy as np

from utils import chunk_sequence, get_device, zscore_scale


# ================================================================
#  BOTTLENECK EMBEDDING NETWORK
#  Each 15-day chunk → embedding vector (dim = config.cct_embed_dim)
# ================================================================

class ChunkEmbeddingNet(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super().__init__()

        # Simple bottleneck MLP, Render-safe (light CPU usage)
        self.net = nn.Sequential(
            nn.Linear(input_dim * 15, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim)
        )

    def forward(self, x):
        """
        x shape: [batch, chunk_size, feature_dim]
        """
        b, t, f = x.size()
        flat = x.reshape(b, t * f)
        return self.net(flat)  # [batch, embed_dim]


# ================================================================
#  EMBED ENGINE — MAIN CCT-90 MODULE
# ================================================================

class CCT90Embedder:
    """
    Takes a rolling 90-day window of features:
       shape: [90, n_features]

    Splits into 6 chunks of 15 days.

    Each chunk is compressed into a dense embedding vector:
       shape: [6, embed_dim]

    Output goes into the Transformer.

    This engine is CPU-safe and renders fast.
    """

    def __init__(self, cfg, n_features):
        self.cfg = cfg
        self.chunk_size = cfg.cct_chunk_size
        self.embed_dim = cfg.cct_embed_dim
        self.device = get_device()

        # Bottleneck encoder
        self.encoder = ChunkEmbeddingNet(
            input_dim=n_features,
            embed_dim=self.embed_dim
        ).to(self.device)

    # ============================================================
    #  EXTRACT 90-DAY WINDOW AND GENERATE EMBEDDINGS
    # ============================================================

    def transform(self, full_feature_matrix):
        """
        full_feature_matrix: np.array of shape [N_days, n_features]
        Returns: torch.tensor of shape [num_chunks, embed_dim]
        """

        # Step 1: take last 90 days
        window_len = self.cfg.cct_window_days
        if full_feature_matrix.shape[0] < window_len:
            # pad missing days with zeros
            pad = np.zeros((window_len - full_feature_matrix.shape[0],
                            full_feature_matrix.shape[1]))
            mat = np.vstack([pad, full_feature_matrix])
        else:
            mat = full_feature_matrix[-window_len:]

        # Step 2: normalize each feature via z-score
        mat = np.array([zscore_scale(col) for col in mat.T]).T

        # Step 3: chunk into 6 groups of 15 days
        chunks_np = chunk_sequence(mat, self.chunk_size)  # [6, 15, n_features]

        # Convert to torch tensor
        chunks = torch.tensor(chunks_np, dtype=torch.float32).to(self.device)

        # Step 4: feed each chunk through encoder
        embeddings = self.encoder(chunks)  # shape: [6, embed_dim]

        return embeddings.detach().cpu()  # keep CPU-friendly for transformer
