"""Complete A/B/C forecasters assembled from the shared frozen components."""

from torch import nn

from .components import LearnedPositions, PredictionHead, PreNormTransformerBlock, SharedEmbeddings
from .experiment import DEFAULT_PROTOCOL, load_protocol


class Forecaster(nn.Module):
    def __init__(self, architecture, protocol_path=DEFAULT_PROTOCOL):
        super().__init__()
        if architecture not in ("A", "B", "C"):
            raise ValueError("Architecture must be A, B, or C")
        protocol = load_protocol(protocol_path)
        spec = protocol["architecture"][architecture]
        shared = protocol["architecture"]["shared"]
        self.architecture = architecture
        self.identities = SharedEmbeddings(protocol_path)
        input_width = len(protocol["dataset"]["features"]) + 2 * shared["identity_embedding_dim"]
        self.recurrent = None
        if architecture in ("A", "C"):
            self.recurrent = nn.LSTM(
                input_width, spec["hidden_per_direction"], num_layers=spec["recurrent_layers"],
                batch_first=True, bidirectional=spec["bidirectional"],
                dropout=spec["recurrent_dropout"],
            )
            self.projection = nn.Identity()
        else:
            self.projection = nn.Linear(*spec["projection"])
        self.norm = nn.LayerNorm(shared["width"], eps=shared["layer_norm_eps"])
        if architecture == "A":
            self.encoder = nn.Identity()
            self.dropout = nn.Dropout(shared["dropout"])
        else:
            self.encoder = nn.Sequential(
                LearnedPositions(protocol_path),
                *[PreNormTransformerBlock(protocol_path) for _ in range(spec["encoder_blocks"])],
            )
            self.dropout = nn.Identity()
        self.head = PredictionHead(protocol_path)

    def forward(self, history, market_id, category_id):
        sequence = self.identities(history, market_id, category_id)
        if self.recurrent is not None:
            sequence, _ = self.recurrent(sequence)
        else:
            sequence = self.projection(sequence)
        return self.head(self.dropout(self.norm(self.encoder(sequence))))
