# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLSTM(nn.Module):
    def __init__(
            self,
            input_dim: int = 32,
            hidden_dim: int = 256,
            num_layers: int = 2,
            output_num: int = 24,
            output_dim: int = 128,
            bidirectional: bool = True
        ):
        super().__init__()
        self.name = "TimeFeatureNet(Attention-LSTM)"

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True
        )

        lstm_out_dim = hidden_dim * 2 if bidirectional else hidden_dim

        # Bahdanau Attention
        self.attention = nn.Sequential(
            # decrease dimension to calculate attention scores
            nn.Linear(lstm_out_dim, lstm_out_dim),
            nn.Tanh(),
            # get attention weight of each time step
            nn.Linear(lstm_out_dim, 1)
        )

        self.reduction = nn.Sequential(
            # middle layer
            nn.Linear(lstm_out_dim, 256),
            nn.ReLU(),
            # avoid overfitting
            nn.Dropout(0.2),
            # get features output
            nn.Linear(256, hidden_dim)
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
        x = self.augment(x)
        # x: (batch_size, seq_len, feat_dim)
        lstm_out, _ = self.lstm(x)
        # lstm_out: (batch_size, seq_len, lstm_out_dim)
        attn_weights: torch.Tensor = self.attention(lstm_out).squeeze(-1)
        attn_weights = F.softmax(attn_weights, dim=1)
        # attn_weights: (batch_size, seq_len)
        weighted_feat = lstm_out * attn_weights.unsqueeze(-1)
        # weighted_feat: (batch_size, seq_len, lstm_out_dim)
        out: torch.Tensor = self.reduction(weighted_feat)
        # out: (batch_size, seq_len, output_dim)
        out = out.permute(0, 2, 1)
        out = self.extractor(out)
        out = out.permute(0, 2, 1)
        # out: (batch_size, output_num, output_dim)
        return out
