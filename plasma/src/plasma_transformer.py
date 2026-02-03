"""
Simple transformer for learning plasma state evolution.

The model learns to predict the next diagnostic state from the sequence so far.
We extract embeddings and analyze their geometry for disruption precursors.
"""

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class PlasmaTransformer(nn.Module):
    """Transformer for plasma diagnostic sequences.

    Architecture:
    - Input projection: diagnostic_dim -> embed_dim
    - Positional encoding
    - Transformer encoder layers
    - Output projection: embed_dim -> diagnostic_dim

    For geometry analysis, we extract the hidden states (embeddings)
    at each timestep.
    """

    def __init__(
        self,
        diagnostic_dim: int,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
        max_seq_len: int = 1000,
    ):
        super().__init__()

        self.diagnostic_dim = diagnostic_dim
        self.embed_dim = embed_dim

        # Input projection
        self.input_proj = nn.Linear(diagnostic_dim, embed_dim)

        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
        nn.init.normal_(self.pos_encoding, std=0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection
        self.output_proj = nn.Linear(embed_dim, diagnostic_dim)

        # Causal mask cache
        self._causal_mask = None

    def _get_causal_mask(self, seq_len: int, device) -> torch.Tensor:
        """Generate causal attention mask."""
        if self._causal_mask is None or self._causal_mask.size(0) < seq_len:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
            self._causal_mask = mask.bool()
        return self._causal_mask[:seq_len, :seq_len]

    def forward(self, x: torch.Tensor, return_embeddings: bool = False):
        """Forward pass.

        Args:
            x: [batch, seq_len, diagnostic_dim] input sequence
            return_embeddings: if True, also return hidden states

        Returns:
            predictions: [batch, seq_len, diagnostic_dim] next-state predictions
            embeddings: [batch, seq_len, embed_dim] if return_embeddings=True
        """
        batch_size, seq_len, _ = x.shape

        # Project to embedding space
        h = self.input_proj(x)

        # Add positional encoding
        h = h + self.pos_encoding[:, :seq_len, :]

        # Causal mask for autoregressive prediction
        mask = self._get_causal_mask(seq_len, x.device)

        # Transform
        h = self.transformer(h, mask=mask)

        # Project to output
        predictions = self.output_proj(h)

        if return_embeddings:
            return predictions, h
        return predictions

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Extract embeddings without computing predictions."""
        _, embeddings = self.forward(x, return_embeddings=True)
        return embeddings


def train_on_shots(
    model: PlasmaTransformer,
    trajectories: list[np.ndarray],
    epochs: int = 50,
    lr: float = 1e-3,
    seq_len: int = 100,
    batch_size: int = 32,
    device: str = "cpu",
    verbose: bool = True,
):
    """Train model on plasma trajectories.

    Args:
        model: PlasmaTransformer instance
        trajectories: list of [T, D] arrays (different length shots)
        epochs: training epochs
        lr: learning rate
        seq_len: sequence length for training chunks
        batch_size: batch size
        device: 'cpu' or 'cuda' or 'mps'
        verbose: print progress

    Returns:
        losses: list of epoch losses
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Create training chunks from trajectories
    chunks = []
    for traj in trajectories:
        # Normalize
        mean = traj.mean(axis=0, keepdims=True)
        std = traj.std(axis=0, keepdims=True) + 1e-10
        traj_norm = (traj - mean) / std

        # Create overlapping chunks
        for i in range(0, len(traj_norm) - seq_len, seq_len // 2):
            chunk = traj_norm[i:i + seq_len]
            chunks.append(chunk)

    chunks = np.array(chunks, dtype=np.float32)
    if verbose:
        print(f"Training on {len(chunks)} chunks of length {seq_len}")

    losses = []

    for epoch in range(epochs):
        # Shuffle
        perm = np.random.permutation(len(chunks))
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, len(chunks), batch_size):
            batch_idx = perm[i:i + batch_size]
            batch = torch.tensor(chunks[batch_idx], device=device)

            # Input is t=0..T-1, target is t=1..T
            x = batch[:, :-1, :]
            y = batch[:, 1:, :]

            # Forward
            pred = model(x)

            # MSE loss
            loss = F.mse_loss(pred, y)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        epoch_loss /= n_batches
        losses.append(epoch_loss)

        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch + 1}/{epochs}: loss = {epoch_loss:.6f}")

    return losses


def extract_embedding_trajectory(
    model: PlasmaTransformer,
    trajectory: np.ndarray,
    device: str = "cpu",
    chunk_size: int = 200,
) -> np.ndarray:
    """Extract embedding trajectory for a shot.

    Args:
        model: trained PlasmaTransformer
        trajectory: [T, D] diagnostic trajectory
        device: compute device
        chunk_size: process in chunks to manage memory

    Returns:
        embeddings: [T, embed_dim] embedding trajectory
    """
    model = model.to(device)
    model.eval()

    # Normalize
    mean = trajectory.mean(axis=0, keepdims=True)
    std = trajectory.std(axis=0, keepdims=True) + 1e-10
    traj_norm = (trajectory - mean) / std

    embeddings = []

    with torch.no_grad():
        # Process in overlapping chunks
        for i in range(0, len(traj_norm), chunk_size // 2):
            chunk = traj_norm[max(0, i - chunk_size // 2):i + chunk_size]
            if len(chunk) < 10:
                continue

            x = torch.tensor(chunk[np.newaxis], dtype=torch.float32, device=device)
            emb = model.get_embeddings(x)

            # Take embeddings from the non-overlapping part
            if i == 0:
                embeddings.append(emb[0].cpu().numpy())
            else:
                # Skip the overlapping portion
                start = chunk_size // 4
                embeddings.append(emb[0, start:].cpu().numpy())

    return np.concatenate(embeddings, axis=0)[:len(trajectory)]


if not TORCH_AVAILABLE:
    print("PyTorch not available. Install with: pip install torch")
