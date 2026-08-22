# coding: utf-8
import torch
from torch import nn


class PredictionHead(nn.Module):
    def __init__(
            self,
            input_dim: int = 256,
            task_num: int = 2,
            output_dim: int = 24,
        ):
        super().__init__()
        self.name = "PredictionHead"
        self.output_dim = output_dim
        self.task_num = task_num

        # shared is selectable
        self.shared = nn.Sequential(
            nn.Conv1d(input_dim, input_dim * 2, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(input_dim * 2),
        )
        self.towers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim * 2, input_dim),
                nn.Linear(input_dim, input_dim // 2),
                nn.ReLU(),
            )
            for _ in range(task_num)
        ])
        self.preds = nn.ModuleList([
            nn.Conv1d(input_dim // 2, 1, kernel_size=1)
            for _ in range(task_num)
        ])

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape = (batch_size, seq_len, feat_dim)
        # seq_len = output_len
        _, seq_len, _ = x.shape
        assert seq_len == self.output_dim, \
            f"sequence length must be equal to prediction length ({self.output_dim})"
        
        # ret.shape = (batch_size, task_num, seq_len / output_dim)
        # fc(x).shape = (batch_size, seq_len, 1)
        # .squeeze(-1) -> (batch_size, seq_len)
        x = x.permute(0, 2, 1)
        x = self.shared(x)
        x = x.permute(0, 2, 1)
        task_outputs: list[torch.Tensor] = list()
        for i in range(self.task_num):
            t: torch.Tensor = self.towers[i](x)
            t = t.permute(0, 2, 1)
            task_outputs.append(self.preds[i](t).permute(0, 2, 1).squeeze(-1))

        # (batch_size, output_dim, 2)
        ret = torch.stack(task_outputs, dim=2)
        return ret


class TaskSpecificPredictionHead(nn.Module):
    """Decode one task-specific fused representation per prediction target."""

    def __init__(
            self,
            input_dim: int = 256,
            task_num: int = 2,
            output_dim: int = 24,
        ):
        super().__init__()
        self.name = "TaskSpecificPredictionHead"
        self.output_dim = output_dim
        self.task_num = task_num
        self.heads = nn.ModuleList([
            PredictionHead(
                input_dim=input_dim,
                task_num=1,
                output_dim=output_dim,
            )
            for _ in range(task_num)
        ])

    def forward(self, task_features: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> torch.Tensor:
        if len(task_features) != self.task_num:
            raise ValueError(
                f"expected {self.task_num} task feature tensors, got {len(task_features)}"
            )

        task_outputs = [
            head(feature)[..., 0]
            for head, feature in zip(self.heads, task_features)
        ]
        return torch.stack(task_outputs, dim=2)


class _PointwiseExpert(nn.Module):
    """Transform one fused feature vector into the decoder feature space."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class _BottleneckExpert(nn.Module):
    """Parameter-matched nonlinear transform used by deeper candidate heads."""

    def __init__(self, input_dim: int, bottleneck_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, bottleneck_dim),
            nn.ReLU(),
            nn.Linear(bottleneck_dim, input_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class _RoutedPredictionHead(nn.Module):
    """Shared decoding behaviour for heads that route fused features by task."""

    def __init__(self, input_dim: int, task_num: int, output_dim: int):
        super().__init__()
        if task_num != 2:
            raise ValueError("routed multi-task prediction heads require exactly two tasks")
        if input_dim < 2:
            raise ValueError("input_dim must be at least 2")
        self.input_dim = input_dim
        self.task_num = task_num
        self.output_dim = output_dim
        self.towers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim * 2, input_dim),
                nn.Linear(input_dim, input_dim // 2),
                nn.ReLU(),
            )
            for _ in range(task_num)
        ])
        self.preds = nn.ModuleList([
            nn.Conv1d(input_dim // 2, 1, kernel_size=1)
            for _ in range(task_num)
        ])

    def _validate_input(self, x: torch.Tensor) -> None:
        if x.ndim != 3:
            raise ValueError("fused features must have shape (batch, horizon, feature)")
        if x.shape[1] != self.output_dim:
            raise ValueError(
                f"sequence length must equal prediction length ({self.output_dim})"
            )
        if x.shape[2] != self.input_dim:
            raise ValueError(f"feature dimension must equal input_dim ({self.input_dim})")

    def _decode(self, task_features: list[torch.Tensor]) -> torch.Tensor:
        if len(task_features) != self.task_num:
            raise ValueError(f"expected {self.task_num} task feature tensors")
        task_outputs = []
        for task_index, feature in enumerate(task_features):
            hidden = self.towers[task_index](feature).permute(0, 2, 1)
            task_outputs.append(
                self.preds[task_index](hidden).permute(0, 2, 1).squeeze(-1)
            )
        return torch.stack(task_outputs, dim=2)


class MMoEPredictionHead(_RoutedPredictionHead):
    """Multi-gate mixture-of-experts decoder for the two DSF-Net targets."""

    def __init__(
            self,
            input_dim: int = 256,
            task_num: int = 2,
            output_dim: int = 24,
            num_experts: int = 4,
        ):
        if num_experts < 1:
            raise ValueError("num_experts must be positive")
        super().__init__(input_dim=input_dim, task_num=task_num, output_dim=output_dim)
        self.name = "MMoEPredictionHead"
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            _PointwiseExpert(input_dim)
            for _ in range(num_experts)
        ])
        self.gates = nn.ModuleList([
            nn.Linear(input_dim, num_experts)
            for _ in range(task_num)
        ])

    def routing_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-step task gates with shape (batch, horizon, task, expert)."""
        self._validate_input(x)
        return torch.stack([
            torch.softmax(gate(x), dim=-1)
            for gate in self.gates
        ], dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_input(x)
        expert_features = torch.stack([expert(x) for expert in self.experts], dim=2)
        routes = self.routing_weights(x)
        routed_features = torch.einsum("btke,bted->btkd", routes, expert_features)
        return self._decode(list(routed_features.unbind(dim=2)))


class PLEPredictionHead(_RoutedPredictionHead):
    """Progressive layered extraction decoder with shared and private experts."""

    def __init__(
            self,
            input_dim: int = 256,
            task_num: int = 2,
            output_dim: int = 24,
            num_shared_experts: int = 2,
            num_private_experts: int = 1,
            num_levels: int = 1,
        ):
        if num_shared_experts < 1 or num_private_experts < 1 or num_levels < 1:
            raise ValueError("PLE requires positive shared and private expert counts")
        super().__init__(input_dim=input_dim, task_num=task_num, output_dim=output_dim)
        self.name = "PLEPredictionHead"
        self.num_shared_experts = num_shared_experts
        self.num_private_experts = num_private_experts
        self.num_levels = num_levels
        self.shared_experts = nn.ModuleList([
            _PointwiseExpert(input_dim)
            for _ in range(num_shared_experts)
        ])
        self.private_experts = nn.ModuleList([
            nn.ModuleList([
                _PointwiseExpert(input_dim)
                for _ in range(num_private_experts)
            ])
            for _ in range(task_num)
        ])
        expert_count_per_task = num_shared_experts + num_private_experts
        self.gates = nn.ModuleList([
            nn.Linear(input_dim, expert_count_per_task)
            for _ in range(task_num)
        ])
        self.refinement_shared_experts = nn.ModuleList([
            nn.ModuleList([
                _BottleneckExpert(input_dim * 2, max(input_dim // 8, 1))
                for _ in range(num_shared_experts)
            ])
            for _ in range(num_levels - 1)
        ])
        self.refinement_private_experts = nn.ModuleList([
            nn.ModuleList([
                nn.ModuleList([
                    _BottleneckExpert(input_dim * 2, max(input_dim // 8, 1))
                    for _ in range(num_private_experts)
                ])
                for _ in range(task_num)
            ])
            for _ in range(num_levels - 1)
        ])
        self.refinement_gates = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(input_dim * 2, expert_count_per_task)
                for _ in range(task_num)
            ])
            for _ in range(num_levels - 1)
        ])

    def routing_weights(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Return one normalized gate tensor per task over its eligible experts."""
        self._validate_input(x)
        return tuple(torch.softmax(gate(x), dim=-1) for gate in self.gates)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_input(x)
        shared_features = [expert(x) for expert in self.shared_experts]
        routes = self.routing_weights(x)
        task_features = []
        for task_index, task_routes in enumerate(routes):
            eligible_features = shared_features + [
                expert(x) for expert in self.private_experts[task_index]
            ]
            stacked_features = torch.stack(eligible_features, dim=2)
            task_features.append(
                torch.einsum("bte,bted->btd", task_routes, stacked_features)
            )
        for shared_experts, private_experts, gates in zip(
                self.refinement_shared_experts,
                self.refinement_private_experts,
                self.refinement_gates,
        ):
            shared_input = torch.stack(task_features, dim=2).mean(dim=2)
            shared_features = [expert(shared_input) for expert in shared_experts]
            next_features = []
            for task_index, feature in enumerate(task_features):
                eligible_features = shared_features + [
                    expert(feature) for expert in private_experts[task_index]
                ]
                routes = torch.softmax(gates[task_index](feature), dim=-1)
                next_features.append(
                    torch.einsum("bte,bted->btd", routes, torch.stack(eligible_features, dim=2))
                )
            task_features = next_features
        return self._decode(task_features)


class SeparateTowersPredictionHead(_RoutedPredictionHead):
    """Fully separate task towers after a shared DCAFusion representation."""

    def __init__(
            self,
            input_dim: int = 256,
            task_num: int = 2,
            output_dim: int = 24,
            tower_depth: int = 1,
        ):
        if tower_depth < 1:
            raise ValueError("tower_depth must be positive")
        super().__init__(input_dim=input_dim, task_num=task_num, output_dim=output_dim)
        self.name = "SeparateTowersPredictionHead"
        self.tower_depth = tower_depth
        self.feature_extractors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, input_dim * 2),
                nn.ReLU(),
                *[
                    module
                    for _ in range(tower_depth - 1)
                    for module in (
                        _BottleneckExpert(input_dim * 2, max(input_dim // 4, 1)),
                    )
                ],
            )
            for _ in range(task_num)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_input(x)
        return self._decode([extractor(x) for extractor in self.feature_extractors])


class _CrossStitchUnit(nn.Module):
    """Channel-wise, identity-initialized mixing of two task feature streams."""

    def __init__(self, feature_dim: int):
        super().__init__()
        self.weights = nn.Parameter(torch.eye(2).repeat(feature_dim, 1, 1))

    def forward(self, task_features: list[torch.Tensor]) -> list[torch.Tensor]:
        if len(task_features) != 2:
            raise ValueError("Cross-Stitch requires two task feature tensors")
        stacked = torch.stack(task_features, dim=2)
        mixed = torch.einsum("btih,hij->btjh", stacked, self.weights)
        return list(mixed.unbind(dim=2))


class CrossStitchPredictionHead(_RoutedPredictionHead):
    """Two task streams with one or more learnable Cross-Stitch exchanges."""

    def __init__(
            self,
            input_dim: int = 256,
            task_num: int = 2,
            output_dim: int = 24,
            num_stitch_layers: int = 1,
        ):
        if num_stitch_layers < 1:
            raise ValueError("num_stitch_layers must be positive")
        super().__init__(input_dim=input_dim, task_num=task_num, output_dim=output_dim)
        self.name = "CrossStitchPredictionHead"
        self.num_stitch_layers = num_stitch_layers
        self.input_extractors = nn.ModuleList([
            nn.Sequential(nn.Linear(input_dim, input_dim * 2), nn.ReLU())
            for _ in range(task_num)
        ])
        self.task_layers = nn.ModuleList([
            nn.ModuleList([
                _BottleneckExpert(input_dim * 2, max(input_dim // 4, 1))
                for _ in range(task_num)
            ])
            for _ in range(num_stitch_layers)
        ])
        self.stitches = nn.ModuleList([
            _CrossStitchUnit(input_dim * 2)
            for _ in range(num_stitch_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_input(x)
        task_features = [extractor(x) for extractor in self.input_extractors]
        for task_layers, stitch in zip(self.task_layers, self.stitches):
            task_features = [layer(feature) for layer, feature in zip(task_layers, task_features)]
            task_features = stitch(task_features)
        return self._decode(task_features)
