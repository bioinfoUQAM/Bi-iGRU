from dataclasses import dataclass
import torch
import torch.nn as nn

@dataclass
class Configs:
    seq_len: int
    pred_len: int
    enc_in: int
    d_model: int = 128
    num_layers: int = 2
    dropout: float = 0.1
    bidirectional: bool = True

class CustomModel(nn.Module):
    """
    A lightweight bidirectional-GRU forecaster to stand in for your Bi-iGRU.
    Input:  (B, seq_len, D)
    Output: (B, pred_len, D)
    """
    def __init__(self, cfg: Configs):
        super().__init__()
        self.cfg = cfg
        self.gru = nn.GRU(
            input_size=cfg.enc_in,
            hidden_size=cfg.d_model,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
            bidirectional=cfg.bidirectional,
            batch_first=True,
        )
        out_dim = cfg.d_model * (2 if cfg.bidirectional else 1)
        self.head = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, cfg.pred_len * cfg.enc_in),
        )

    def forward(self, x):
        # x: (B, seq_len, D)
        y, _ = self.gru(x)
        # take last timestep
        last = y[:, -1, :]  # (B, out_dim)
        out = self.head(last)  # (B, pred_len*D)
        return out.view(-1, self.cfg.pred_len, self.cfg.enc_in)
