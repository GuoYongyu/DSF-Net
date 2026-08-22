# coding: utf-8
import itertools

import torch
from torch import nn
import torch.nn.functional as F


class DCPFusion(nn.Module):
    """Directional Congestion-Pressure Fusion.

    The module keeps the same temporal/spatial feature contract as the existing
    fusion layers, and only uses raw directional spatial sequences when the
    caller provides them.
    """

    DIRECTION_ORDER = ("down", "up", "in", "out")

    def __init__(
            self,
            time_dim: int = 128,
            space_dim: int = 128,
            direction_input_dim: int = 4,
            hidden_dim: int = 128,
            output_dim: int = 256,
            direction_num: int = 4,
            pressure_eta: float = 1.0,
            dropout: float = 0.1,
            include_shared_spatial_in_direction_tokens: bool = True,
            direction_mode: str = "adaptive",
            direction_shuffle_seed: int = 42,
        ):
        super().__init__()
        assert time_dim == space_dim, "space_dim and time_dim must be equal"
        if direction_mode not in {"adaptive", "uniform", "shuffled", "shared-only"}:
            raise ValueError("direction_mode must be adaptive, uniform, shuffled, or shared-only")

        self.name = "DCPFusion"
        self.output_dim = output_dim
        self.direction_num = direction_num
        self.pressure_eta = nn.Parameter(torch.tensor(pressure_eta, dtype=torch.float32))
        self.include_shared_spatial_in_direction_tokens = include_shared_spatial_in_direction_tokens
        self.direction_mode = direction_mode
        self.direction_shuffle_seed = int(direction_shuffle_seed)
        self.capture_diagnostics = False
        self.captured_alpha: list[torch.Tensor] = []
        permutations = torch.tensor(
            list(itertools.permutations(range(direction_num))),
            dtype=torch.long,
        )
        self.register_buffer("direction_permutations", permutations, persistent=False)

        self.time_proj = nn.Linear(time_dim, hidden_dim)
        self.space_proj = nn.Linear(space_dim, hidden_dim)
        self.direction_proj = nn.Linear(direction_input_dim, hidden_dim)
        self.direction_embed = nn.Embedding(direction_num, hidden_dim)

        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.pressure_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.direction_bias = nn.Parameter(torch.zeros(direction_num))

        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Sigmoid(),
        )
        self.out_proj = nn.Linear(hidden_dim * 2, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)


    def _shuffle_direction_feats(
            self,
            direction_feats: tuple[torch.Tensor, ...],
        ) -> tuple[torch.Tensor, ...]:
        """Destroy direction-slot semantics without changing tensor values."""
        stacked = torch.stack(direction_feats, dim=1)
        batch_size = stacked.shape[0]
        permutation_ids = (
            torch.arange(batch_size, device=stacked.device) + self.direction_shuffle_seed
        ) % self.direction_permutations.shape[0]
        permutations = self.direction_permutations[permutation_ids]
        gather_index = permutations.view(
            batch_size, self.direction_num, 1, 1
        ).expand_as(stacked)
        shuffled = torch.gather(stacked, dim=1, index=gather_index)
        return tuple(shuffled[:, idx] for idx in range(self.direction_num))

    def _build_direction_tokens(
            self,
            time: torch.Tensor,
            space: torch.Tensor,
            direction_feats: tuple[torch.Tensor, ...] | None,
        ) -> torch.Tensor:
        batch_size, output_len, _ = time.shape
        device = time.device

        space_feat = self.space_proj(space).unsqueeze(2) # shape: (batch_size, output_len, 1, hidden_dim)

        if direction_feats is None:
            tokens = space_feat.repeat(1, 1, self.direction_num, 1)
        else:
            if self.direction_mode == "shuffled":
                direction_feats = self._shuffle_direction_feats(direction_feats)
            encoded = []
            for idx, feat in enumerate(direction_feats):
                x = self.direction_proj(feat)
                x = x.transpose(1, 2)
                x = F.adaptive_avg_pool1d(x, output_len)
                x = x.transpose(1, 2)
                encoded.append(x + self.direction_embed(torch.tensor(idx, device=device)))
            tokens = torch.stack(encoded, dim=2)
            if self.include_shared_spatial_in_direction_tokens:
                tokens = tokens + space_feat

        if tokens.shape[0] != batch_size:
            raise ValueError("direction token batch size must match time feature batch size")
        return tokens


    def forward(
            self,
            time: torch.Tensor,
            space: torch.Tensor,
            down_feat: torch.Tensor | None = None,
            up_feat: torch.Tensor | None = None,
            in_feat: torch.Tensor | None = None,
            out_feat: torch.Tensor | None = None,
            return_diagnostics: bool = False,
            alpha_override: torch.Tensor | None = None,
        ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # time/space: (batch_size, pred_len, feature_dim)
        time_token = self.time_proj(time)

        direction_feats = None
        if all(feat is not None for feat in (down_feat, up_feat, in_feat, out_feat)):
            direction_feats = (down_feat, up_feat, in_feat, out_feat)
        if self.direction_mode == "shared-only":
            direction_feats = None
        direction_tokens = self._build_direction_tokens(time, space, direction_feats)

        query = self.query_proj(time_token).unsqueeze(2)
        key = self.key_proj(direction_tokens)
        value = self.value_proj(direction_tokens)

        scale = torch.sqrt(torch.tensor(key.shape[-1], dtype=torch.float32, device=key.device))
        pressure = self.pressure_mlp(direction_tokens).squeeze(-1)
        score = (query * key).sum(dim=-1) / scale
        score = score + self.direction_bias.view(1, 1, -1) + self.pressure_eta * pressure
        if alpha_override is not None:
            alpha = alpha_override.to(device=score.device, dtype=score.dtype)
            if alpha.ndim == 2:
                alpha = alpha.unsqueeze(1)
            if alpha.ndim != 3 or alpha.shape[-1] != self.direction_num:
                raise ValueError(
                    "alpha_override must have shape (batch, horizon, direction_num) "
                    "or (batch, direction_num)"
                )
            if alpha.shape[0] not in {1, score.shape[0]}:
                raise ValueError("alpha_override batch dimension does not match input")
            if alpha.shape[1] not in {1, score.shape[1]}:
                raise ValueError("alpha_override horizon dimension does not match input")
            if alpha.shape[0] == 1:
                alpha = alpha.expand(score.shape[0], -1, -1)
            if alpha.shape[1] == 1:
                alpha = alpha.expand(-1, score.shape[1], -1)
            if not torch.isfinite(alpha).all() or torch.any(alpha < 0):
                raise ValueError("alpha_override must be finite and non-negative")
            normalizer = alpha.sum(dim=-1, keepdim=True)
            if torch.any(normalizer <= 0):
                raise ValueError("alpha_override rows must have positive sums")
            alpha = alpha / normalizer
        elif self.direction_mode == "uniform":
            alpha = torch.full_like(score, 1.0 / float(self.direction_num))
        else:
            alpha = F.softmax(score, dim=-1)
        if self.capture_diagnostics:
            self.captured_alpha.append(alpha.detach().cpu())
        spatial_pressure = (alpha.unsqueeze(-1) * value).sum(dim=2)

        gate_input = torch.cat([
            time_token,
            spatial_pressure,
            time_token * spatial_pressure,
            torch.abs(time_token - spatial_pressure),
        ], dim=-1)
        gate = self.gate(gate_input)
        adaptive = gate * time_token + (1.0 - gate) * spatial_pressure
        fused = torch.cat([adaptive, time_token * spatial_pressure], dim=-1)
        fused = self.out_proj(self.dropout(fused))
        fused = self.norm(fused)
        if not return_diagnostics:
            return fused
        diagnostics = {
            "alpha": alpha,
            "gate": gate,
            "pressure": pressure,
            "logits": score,
            "spatial_pressure": spatial_pressure,
        }
        return fused, diagnostics
