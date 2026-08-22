# coding: utf-8
import torch
from torch import nn


class SpaceFeatureEncoder(nn.Module):
    def __init__(
            self,
            input_dim: int = 4 * 4,
            hidden_dim: int = 256,
            num_layers: int = 2,
            num_heads: int = 8,
            output_num: int = 24,
            output_dim: int = 128
        ):
        super().__init__()
        self.name = "SpaceFeatureNet(Transformer-Encoder)"

        self.proj = nn.Linear(input_dim, hidden_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                batch_first=True
            ),
            num_layers=num_layers
        )
        self.extractor = nn.Sequential(
            nn.Conv1d(
                in_channels=hidden_dim,
                out_channels=output_dim,
                kernel_size=3,
                padding=1,
                padding_mode="replicate"
            ),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(output_num),
        )


    def forward(
            self,
            down_feat: torch.Tensor,
            up_feat: torch.Tensor,
            in_feat: torch.Tensor,
            out_feat: torch.Tensor
        ) -> torch.Tensor:
        x = torch.cat([down_feat, up_feat, in_feat, out_feat], dim=-1)
        # x.shape = (batch_size, seq_len, input_dim)
        x = self.proj(x)
        # x.shape = (batch_size, seq_len, d_model)
        encoded: torch.Tensor = self.encoder(x)
        # encoded.shape = (batch_size, seq_len, d_model)
        encoded = encoded.permute(0, 2, 1)
        out: torch.Tensor = self.extractor(encoded)
        out = out.permute(0, 2, 1)
        # out.shape = (batch_size, output_num, output_dim)
        return out
