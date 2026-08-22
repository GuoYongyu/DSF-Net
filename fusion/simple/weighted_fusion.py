# coding: utf-8
import torch
from torch import nn
import torch.nn.functional as F

from sim.config import *


class WeightedFusion(nn.Module):
    def __init__(
            self,
            fusion_way: str = ["plus", "multiply"][0],
            space_dim: int = 128,
            time_dim: int = 128,
            output_dim: int = 128
        ):
        super().__init__()
        assert space_dim == time_dim, "space_dim and time_dim must be equal"

        self.name = "WeightedFusion"
        self.fusion_way = fusion_way
        self.output_dim = output_dim

        self.space_fc = nn.Linear(space_dim, output_dim)
        self.time_fc = nn.Linear(time_dim, output_dim)
        self.relu = nn.ReLU()


    def forward(self, space: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        # space.shape = (batch_size, seq_len, space_dim)
        # time.shape = (batch_size, seq_len, time_dim)
        space = self.space_fc(space)
        time = self.time_fc(time)
        # space.shape = (batch_size, seq_len, output_dim)
        # time.shape = (batch_size, seq_len, output_dim)
        if self.fusion_way == "plus":
            fused = space + time
        elif self.fusion_way == "multiply":
            fused = space * time
        else:
            raise ValueError(f"weighted_way must be 'plus' or 'multiply', but got {self.fusion_way}")
        fused = self.relu(fused)
        # fused.shape = (batch_size, seq_len, output_dim)
        return fused
        