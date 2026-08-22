# coding: utf-8
import torch
from torch import nn


class TimeFeatureLSTM(nn.Module):
    def __init__(
            self,
            input_dim: int = 32,
            hidden_dim: int = 256,
            num_layers: int = 2,
            output_num: int = 24,
            output_dim: int = 128
        ):
        super().__init__()
        self.name = "TimeFeatureNet(LSTM)"

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
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


    @staticmethod
    def augment(x: torch.Tensor) -> torch.Tensor:
        """
        Augment x by adding cosine and sine encoding to the last dimension
        """
        # x.shape = (batch_size, seq_len, input_dim)
        batch_size, seq_len, _ = x.shape
        pos = torch.arange(seq_len, device=x.device, dtype=torch.float32)
        pos = pos.unsqueeze(0).repeat(batch_size, 1)
        sin_enc = torch.sin(pos * 0.01)
        cos_enc = torch.cos(pos * 0.01)
        return torch.cat([x, sin_enc.unsqueeze(-1), cos_enc.unsqueeze(-1)], dim=-1)  


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape = (batch_size, seq_len, input_dim)
        x = self.augment(x)
        # x.shape = (batch_size, seq_len, input_dim + 2)
        lstm_out, _ = self.lstm(x)
        # lstm_out.shape = (batch_size, seq_len, hidden_dim)
        lstm_out: torch.Tensor = lstm_out.permute(0, 2, 1)
        out: torch.Tensor = self.extractor(lstm_out)
        out = out.permute(0, 2, 1)
        # out.shape = (batch_size, pred_len, output_dim)
        return out
