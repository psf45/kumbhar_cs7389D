import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyGPT(nn.Module):
    def __init__(
        self,
        vocab_size=5000,
        block_size=128,
        d_model=256,
        n_heads=4,
        n_layers=4,
        dropout=0.1,
    ):
        super().__init__()
        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x, targets=None):
        B, T = x.shape
        pos = torch.arange(0, T, device=x.device).unsqueeze(0)
        tok = self.token_emb(x)
        pos = self.pos_emb(pos)
        h = tok + pos
        h = self.transformer(h)
        h = self.ln_f(h)
        logits = self.head(h)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1)
            )
        return logits, loss