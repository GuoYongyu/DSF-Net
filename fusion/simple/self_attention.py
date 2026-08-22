# coding: utf-8
import torch
from torch import nn
import torch.nn.functional as F

from sim.config import *


class SelfAttention(nn.Module):
    def __init__(
            self,
            space_dim: int = 128,
            time_dim: int = 128,
        ):
        super().__init__()
        self.name = "SelfAttention"
        assert space_dim == time_dim, "space_dim and time_dim must be equal"
        self.output_dim = space_dim + time_dim


    @staticmethod
    def attention(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x.shape = (batch_size, seq_len, feat_dim)
        # y.shape = (batch_size, seq_len, feat_dim)
        d = torch.tensor(x.shape[-1], dtype=torch.float32)
        att = torch.matmul(x, y.transpose(-2, -1)) / torch.sqrt(d)
        att = F.softmax(att, dim=-1)
        # att.shape = (batch_size, output_dim)
        return att


    def forward(self, space: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        # space.shape = (batch_size, seq_len, space_dim)
        # time.shape = (batch_size, seq_len, time_dim)
        joint = torch.cat([space, time], dim=-1)
        joint_att = self.attention(joint, joint)
        joint_att = torch.matmul(joint_att, joint)
        # joint.shape = (batch_size, seq_len, output_dim)
        out = F.softmax(joint_att, dim=-1)
        # out.shape = (batch_size, seq_len, output_dim)
        return out