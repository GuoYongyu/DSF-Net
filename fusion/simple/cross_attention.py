# coding: utf-8
import torch
from torch import nn
import torch.nn.functional as F

from sim.config import *


class CrossAttention(nn.Module):
    def __init__(
            self,
            fusion_way: str = ["plus", "multiply", "concat"][0],
            space_dim: int = 128,
            time_dim: int = 128,
            output_dim: int = 128,
        ):
        super().__init__()
        assert space_dim == time_dim, "space_dim and time_dim must be equal"

        self.name = "CorssAttention"
        self.fusion_way = fusion_way

        if fusion_way == "concat":
            input_dim = space_dim + time_dim
        else:
            input_dim = space_dim
        self.output_dim = output_dim
              
        self.attention_fc = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()


    @staticmethod
    def attention(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x.shape = (batch_size, seq_len, feat_dim)
        # y.shape = (batch_size, seq_len, feat_dim)
        d = torch.tensor(x.shape[-1], dtype=torch.float32, device=x.device)
        att = torch.matmul(x, y.transpose(-2, -1)) / torch.sqrt(d)
        att = F.softmax(att, dim=-1)
        # att.shape = (batch_size, output_dim)
        return att


    def forward(self, space: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        # space.shape = (batch_size, seq_len, space_dim)
        # time.shape = (batch_size, seq_len, time_dim)
        space_att = torch.matmul(self.attention(time, space), space)
        time_att = torch.matmul(self.attention(space, time), time)
        # space_att.shape = (batch_size, seq_len, space_dim)
        # time_att.shape = (batch_size, seq_len, time_dim)
        if self.fusion_way == "concat":
            fused = torch.cat([space_att, time_att], dim=-1)
        elif self.fusion_way == "plus":
            fused = space_att + time_att
        elif self.fusion_way == "multiply":
            fused = space_att * time_att
        else:
            raise ValueError(f"fusion_way must be 'plus', 'multiply' or 'concat', got {self.fusion_way}")
        # fused.shape = (batch_size, seq_len, output_dim)
        fused = self.attention_fc(fused)
        fused = self.relu(fused)
        # fused.shape = (batch_size, seq_len, output_dim)
        return fused
        