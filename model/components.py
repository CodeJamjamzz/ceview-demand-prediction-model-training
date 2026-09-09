"""Shared PyTorch layers for the frozen initial forecasting comparison."""

import torch
from torch import nn

from .experiment import DEFAULT_PROTOCOL, load_protocol


def _shared(path):
    return load_protocol(path)["architecture"]["shared"]


def _sequence(x, length, width):
    if x.ndim != 3 or x.shape[1:] != (length, width) or x.shape[0] == 0:
        raise ValueError(f"Expected a nonempty [batch, {length}, {width}] tensor")
    if not torch.is_floating_point(x) or not torch.isfinite(x).all():
        raise ValueError("Sequence must contain finite floating-point values")


class SharedEmbeddings(nn.Module):
    """Append country/category embeddings to each historical numerical row."""

    def __init__(self, protocol_path=DEFAULT_PROTOCOL):
        super().__init__()
        protocol = load_protocol(protocol_path)
        spec = protocol["architecture"]["shared"]
        self.lookback = protocol["dataset"]["lookback"]
        self.numeric_features = len(protocol["dataset"]["features"])
        self.market = nn.Embedding(len(protocol["dataset"]["vocabulary"]["market"]),
                                   spec["identity_embedding_dim"])
        self.category = nn.Embedding(len(protocol["dataset"]["vocabulary"]["category"]),
                                     spec["identity_embedding_dim"])

    def forward(self, history, market_id, category_id):
        _sequence(history, self.lookback, self.numeric_features)
        for ids, embedding in ((market_id, self.market), (category_id, self.category)):
            if ids.shape != (history.shape[0],) or ids.dtype != torch.int64:
                raise ValueError("Identity IDs must be int64 vectors matching the batch")
            if ids.device != history.device or embedding.weight.device != history.device:
                raise ValueError("History, identities, and embeddings must share a device")
            if ((ids < 0) | (ids >= embedding.num_embeddings)).any():
                raise ValueError("Identity ID is outside the frozen vocabulary")
        identities = torch.cat((self.market(market_id), self.category(category_id)), dim=-1)
        return torch.cat((history, identities[:, None, :].expand(-1, self.lookback, -1)), dim=-1)


class LearnedPositions(nn.Module):
    """Add learned historical positions, followed by the shared input dropout."""

    def __init__(self, protocol_path=DEFAULT_PROTOCOL):
        super().__init__()
        protocol = load_protocol(protocol_path)
        spec = protocol["architecture"]["shared"]
        self.lookback = protocol["dataset"]["lookback"]
        self.width = spec["width"]
        self.embedding = nn.Embedding(self.lookback, self.width)
        self.dropout = nn.Dropout(spec["dropout"])

    def forward(self, encoded):
        _sequence(encoded, self.lookback, self.width)
        positions = torch.arange(self.lookback, device=encoded.device)
        return self.dropout(encoded + self.embedding(positions)[None, :, :])


class PredictionHead(nn.Module):
    """Pool encoded historical positions and predict 12 raw scaled values."""

    def __init__(self, protocol_path=DEFAULT_PROTOCOL):
        super().__init__()
        protocol = load_protocol(protocol_path)
        spec = protocol["architecture"]["shared"]
        self.lookback = protocol["dataset"]["lookback"]
        self.width = spec["width"]
        self.layers = nn.Sequential(
            nn.Linear(self.width, spec["head_hidden"]), nn.ReLU(),
            nn.Dropout(spec["dropout"]),
            nn.Linear(spec["head_hidden"], protocol["dataset"]["horizon"]),
        )

    def forward(self, encoded):
        _sequence(encoded, self.lookback, self.width)
        return self.layers(encoded.mean(dim=1))


class PreNormTransformerBlock(nn.Module):
    """Historical self-attention and feed-forward residuals, with no causal mask.

    The architecture-level final LayerNorm belongs after the encoder stack.
    """

    def __init__(self, protocol_path=DEFAULT_PROTOCOL):
        super().__init__()
        protocol = load_protocol(protocol_path)
        spec = protocol["architecture"]["shared"]
        self.lookback = protocol["dataset"]["lookback"]
        self.width = spec["width"]
        self.attention_norm = nn.LayerNorm(self.width, eps=spec["layer_norm_eps"])
        self.attention = nn.MultiheadAttention(
            self.width, spec["attention_heads"], dropout=spec["dropout"], batch_first=True,
        )
        self.attention_dropout = nn.Dropout(spec["dropout"])
        self.feedforward_norm = nn.LayerNorm(self.width, eps=spec["layer_norm_eps"])
        self.feedforward = nn.Sequential(
            nn.Linear(self.width, spec["feedforward_hidden"]), nn.GELU(approximate="none"),
            nn.Dropout(spec["dropout"]), nn.Linear(spec["feedforward_hidden"], self.width),
            nn.Dropout(spec["dropout"]),
        )

    def forward(self, encoded):
        _sequence(encoded, self.lookback, self.width)
        normalized = self.attention_norm(encoded)
        attended, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        residual = encoded + self.attention_dropout(attended)
        return residual + self.feedforward(self.feedforward_norm(residual))
