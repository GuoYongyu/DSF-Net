# coding: utf-8
import torch
from torch import nn
import torch.nn.functional as F

from sim.config import *


class WeightedConcat(nn.Module):
    def __init__(
            self,
            space_dim: int = 128,
            time_dim: int = 128,
            output_dim: int = 256
        ):
        super().__init__()
        assert 2 * space_dim == 2 * time_dim == output_dim, \
            "space_dim, time_dim must be equal, and output_dim must be equal to 2 times of them"
        
        self.name = "WeightedConcat"
        self.output_dim = output_dim * 2

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
        fused = torch.cat([space, time], dim=-1)
        fused = self.relu(fused)
        # fused.shape = (batch_size, seq_len, output_dim * 2)
        return fused
        