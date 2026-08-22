# coding: utf-8
import torch
from torch import nn
import torch.nn.functional as F

from sim.config import *


class FeatureMapping(nn.Module):
    def __init__(
            self,
            fusion_way: str = ["plus", "multiply", "concat"][0],
            space_dim: int = 256,
            time_dim: int = 256,
            mapping_dim: int = 128,
        ):
        super().__init__()
        self.name = "FeatureMapping"
        self.fusion_way = fusion_way
        
        if self.fusion_way == "concat":
            self.output_dim = 2 * mapping_dim
        else:
            self.output_dim = mapping_dim

        self.space_fc = nn.Linear(space_dim, mapping_dim)
        self.time_fc = nn.Linear(time_dim, mapping_dim)
        self.space_relu = nn.ReLU()
        self.time_relu = nn.ReLU()

    
    def forward(self, space: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        # space.shape = (batch_size, seq_len, space_dim)
        # time.shape = (batch_size, seq_len, time_dim)
        space = self.space_fc(space)
        time = self.time_fc(time)
        # space.shape = (batch_size, seq_len, mapping_dim)
        # time.shape = (batch_size, seq_len, mapping_dim)
        space = self.space_relu(space)
        time = self.time_relu(time)
        # space.shape = (batch_size, seq_len, mapping_dim)
        # time.shape = (batch_size, seq_len, mapping_dim)
        if self.fusion_way == "concat":
            fused = torch.cat([space, time], dim=-1)
        elif self.fusion_way == "plus":
            fused = space + time
        elif self.fusion_way == "multiply":
            fused = space * time
        else:
            raise ValueError(f"fusion_way must be 'plus', 'multiply' or 'concat', but got {self.fusion_way}")
        # fused.shape = (batch_size, seq_len, output_dim)
        return fused
        