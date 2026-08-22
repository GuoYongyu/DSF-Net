# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialConv(nn.Module):
    def __init__(
            self,
            input_dim: int = 4,
            hidden_dim: int = 256,
            kernel_size: int = 3,
            output_num: int = 24,
            output_dim: int = 128,
        ):
        super(SpatialConv, self).__init__()
        self.name = "SpaceFeatureNet(SpatialConv)"

        self.proj = nn.Conv2d(
            in_channels=4,                  # 4 channels: [down, up, in, out]
            out_channels=hidden_dim,
            kernel_size=(kernel_size, 1),   # convolute on time, not space
            padding=(kernel_size // 2, 0),  # keep time sequence length
        )
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(
                hidden_dim, hidden_dim * 2,
                kernel_size=(1, kernel_size),
                padding=(0, kernel_size // 2)
            ),
            nn.GELU(),
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=1)
        )
        self.reduction = nn.Sequential(
            nn.Linear(hidden_dim * input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)  # stablize training
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
        # input: (batch_size, seq_len, input_dim)
        batch_size, seq_len, _ = down_feat.shape

        x = torch.stack([down_feat, up_feat, in_feat, out_feat], dim=1)
        # x: (batch_size, 4, seq_len, input_dim)

        # get local time feature of each space channel
        x: torch.Tensor = self.proj(x)
        # x: (batch_size, hidden_dim, seq_len, input_dim)

        # spatial interaction of groups
        x: torch.Tensor = self.spatial_conv(x)
        # x: (batch_size, hidden_dim, seq_len, input_dim)

        # map dimension to target
        x = x.permute(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        # x: (batch_size, seq_len, hidden_dim * input_dim)
        red_out: torch.Tensor = self.reduction(x)
        red_out = red_out.permute(0, 2, 1)
        out: torch.Tensor = self.extractor(red_out)
        out = out.permute(0, 2, 1)
        # out: (batch_size, seq_len, output_dim)
        return out
