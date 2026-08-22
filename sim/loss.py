# coding: utf-8
from typing import Type, Any

import torch
from torch import nn
from torch.nn import functional as F


class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()


    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return self.mse(y_pred, y_true)


class HuberLoss(nn.Module):
    """
    计算Huber损失（Huber Loss）
    """
    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.delta = delta


    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return nn.HuberLoss(delta=self.delta)(y_pred, y_true)


class MAPELoss(nn.Module):
    """
    计算平均绝对百分比误差（MAPE）
    """
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps


    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        denominator = torch.clamp(torch.abs(y_true), min=self.eps)
        ape = torch.abs((y_true - y_pred) / denominator)
        return torch.mean(ape)


class MultiTasksLoss(nn.Module):
    def __init__(
            self,
            num_tasks: int = 2,
            loss_type: Type[Any] = HuberLoss,
            weighting: str = "dynamic",
            fixed_weights: list[float] | None = None,
            warmup_epochs: int = 0,
            warmup_fixed_weights: list[float] | None = None,
            mape_lambda: float = 0.1,
            task_scales: list[float] | torch.Tensor | None = None,
        ):
        super().__init__()
        assert weighting in ["dynamic", "fixed", "uncertainty"], "weighting must be dynamic/fixed/uncertainty"
        assert warmup_epochs >= 0, "warmup_epochs must be >= 0"
        assert 0.0 <= mape_lambda <= 0.2, "mape_lambda must be in [0.0, 0.2]"

        self.num_tasks = num_tasks
        self.weighting = weighting
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 1
        self.mape_lambda = mape_lambda
        self.mape_loss = MAPELoss()
        self.criterions = [
            loss_type() for _ in range(num_tasks)
        ]
        if task_scales is None:
            task_scales = torch.ones(num_tasks, dtype=torch.float32)
        task_scales_tensor = torch.as_tensor(task_scales, dtype=torch.float32)
        assert task_scales_tensor.numel() == num_tasks, "task_scales length must equal num_tasks"
        assert torch.all(task_scales_tensor > 0), "task_scales must be positive"
        self.register_buffer("task_scales", task_scales_tensor)

        if fixed_weights is None:
            fixed_weights = [1.0 / num_tasks] * num_tasks
        if warmup_fixed_weights is not None:
            fixed_weights = warmup_fixed_weights
        self.register_buffer("fixed_weights", self._normalize_weights(fixed_weights))

        if weighting == "uncertainty":
            self.log_vars = nn.Parameter(torch.zeros(num_tasks, dtype=torch.float32))


    def _normalize_weights(self, weights: list[float] | torch.Tensor) -> torch.Tensor:
        tensor = torch.tensor(weights, dtype=torch.float32) if not isinstance(weights, torch.Tensor) else weights.to(torch.float32)
        assert tensor.numel() == self.num_tasks, "weights length must equal num_tasks"
        assert tensor.sum().item() > 0, "sum of weights must be positive"
        return tensor / tensor.sum()


    def set_fixed_weights(self, weights: list[float] | torch.Tensor):
        self.fixed_weights = self._normalize_weights(weights).to(self.fixed_weights.device)


    def set_epoch(self, epoch: int):
        assert epoch >= 1, "epoch must be >= 1"
        self.current_epoch = epoch


    def _effective_weighting(self) -> str:
        if self.warmup_epochs > 0 and self.current_epoch <= self.warmup_epochs:
            return "fixed"
        return self.weighting


    def get_current_weights(self, task_loss: torch.Tensor | None = None) -> torch.Tensor:
        if self.weighting == "dynamic":
            assert task_loss is not None, "task_loss is required for dynamic weighting"
            return nn.functional.softmax(task_loss, dim=0)
        if self.weighting == "uncertainty":
            inv_var = torch.exp(-self.log_vars)
            return inv_var / inv_var.sum()
        return self.fixed_weights


    def forward(
            self,
            y_pred: torch.Tensor,
            y_true: torch.Tensor,
        ) -> tuple[torch.Tensor, dict]:
        task_losses: list[torch.Tensor] = []
        loss_list: list[float] = []
        for i in range(self.num_tasks):
            t_pred = y_pred[..., i]
            t_true = y_true[..., i]
            scale = torch.clamp(self.task_scales[i].to(t_pred.device), min=1e-8)
            t_loss: torch.Tensor = self.criterions[i](t_pred / scale, t_true / scale)
            task_losses.append(t_loss)
            loss_list.append(t_loss.detach().cpu().item())

        task_loss = torch.stack(task_losses, dim=0)
        effective_mode = self._effective_weighting()
        if effective_mode == "uncertainty":
            log_vars = self.log_vars.to(task_loss.device)
            precision = torch.exp(-log_vars)
            base_loss = torch.mean(precision * task_loss + log_vars)
            weights = precision / precision.sum()
        elif effective_mode == "dynamic":
            weights = self.get_current_weights(task_loss).to(task_loss.device)
            base_loss = torch.sum(weights * task_loss)
        else:
            weights = self.fixed_weights.to(task_loss.device)
            base_loss = torch.sum(weights * task_loss)

        mape_loss: torch.Tensor = self.mape_loss(y_pred, y_true)
        total_loss = base_loss + self.mape_lambda * mape_loss

        return total_loss, {
            "mode": self.weighting,
            "configured_mode": self.weighting,
            "effective_mode": effective_mode,
            "warmup_epochs": self.warmup_epochs,
            "loss list": loss_list,
            "weights": weights.detach().cpu().numpy().tolist(),
            "weight_tensor": weights.detach(),
            "base_loss": base_loss.detach().cpu().item(),
            "mape_loss": mape_loss.detach().cpu().item(),
            "mape_lambda": self.mape_lambda,
            "task_loss_tensor": task_loss,
            "mape_loss_tensor": mape_loss,
        }


class ScenarioCompositeLoss(nn.Module):
    def __init__(
            self,
            num_tasks: int = 2,
            thresholds: list[float] | torch.Tensor | None = None,
            scales: list[float] | torch.Tensor | None = None,
            event_pos_weights: list[float] | torch.Tensor | None = None,
            reg_weights: list[float] | torch.Tensor | None = None,
            event_weights: list[float] | torch.Tensor | None = None,
            task_weights: list[float] | torch.Tensor | None = None,
            wape_weight: float = 0.1,
            peak_weight: float = 0.0,
            joint_event_weight: float = 0.0,
            joint_event_pos_weight: float | torch.Tensor = 1.0,
            focal_gamma: float = 2.0,
            normalization_decay: float = 0.98,
            eps: float = 1e-6,
        ):
        super().__init__()
        assert num_tasks >= 1, "num_tasks must be >= 1"
        assert wape_weight >= 0, "wape_weight must be >= 0"
        assert peak_weight >= 0, "peak_weight must be >= 0"
        assert joint_event_weight >= 0, "joint_event_weight must be >= 0"
        assert focal_gamma >= 0, "focal_gamma must be >= 0"
        assert 0.0 <= normalization_decay < 1.0, "normalization_decay must be in [0, 1)"

        self.num_tasks = num_tasks
        self.wape_weight = wape_weight
        self.peak_weight = peak_weight
        self.joint_event_weight = joint_event_weight
        self.focal_gamma = focal_gamma
        self.normalization_decay = normalization_decay
        self.eps = eps
        self.current_epoch = 1

        if thresholds is None:
            thresholds = [float("nan")] * num_tasks
        if scales is None:
            scales = [1.0] * num_tasks
        if event_pos_weights is None:
            event_pos_weights = [1.0] * num_tasks
        if reg_weights is None:
            reg_weights = [0.35, 0.35] if num_tasks == 2 else [0.7]
        if event_weights is None:
            event_weights = [0.1, 0.1] if num_tasks == 2 else [0.2]
        if task_weights is None:
            task_weights = [1.0] * num_tasks

        self.register_buffer("thresholds", self._as_float_vector(thresholds, "thresholds"))
        self.register_buffer("scales", torch.clamp(self._as_float_vector(scales, "scales"), min=eps))
        self.register_buffer("event_pos_weights", torch.clamp(self._as_float_vector(event_pos_weights, "event_pos_weights"), min=1.0, max=20.0))
        self.register_buffer("reg_weights", self._as_float_vector(reg_weights, "reg_weights"))
        self.register_buffer("event_weights", self._as_float_vector(event_weights, "event_weights"))
        joint_pos_weight_tensor = torch.as_tensor(joint_event_pos_weight, dtype=torch.float32).reshape(())
        if not torch.isfinite(joint_pos_weight_tensor) or joint_pos_weight_tensor <= 0:
            raise ValueError("joint_event_pos_weight must be finite and positive")
        self.register_buffer("joint_event_pos_weight", joint_pos_weight_tensor)
        task_weight_tensor = self._as_float_vector(task_weights, "task_weights")
        if torch.any(task_weight_tensor < 0) or torch.sum(task_weight_tensor) <= 0:
            raise ValueError("task_weights must be non-negative and sum to a positive value")
        self.register_buffer("task_weights", task_weight_tensor / torch.sum(task_weight_tensor))
        self.register_buffer("task_composite_ema", torch.ones(num_tasks, dtype=torch.float32))
        self.register_buffer("task_composite_ema_initialized", torch.tensor(0.0, dtype=torch.float32))


    def _as_float_vector(self, values: list[float] | torch.Tensor, name: str) -> torch.Tensor:
        tensor = torch.tensor(values, dtype=torch.float32) if not isinstance(values, torch.Tensor) else values.to(torch.float32)
        assert tensor.numel() == self.num_tasks, f"{name} length must equal num_tasks"
        return tensor.reshape(self.num_tasks)


    def set_epoch(self, epoch: int):
        assert epoch >= 1, "epoch must be >= 1"
        self.current_epoch = epoch


    def _normalize_task_composite(
            self,
            task_composite: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.training:
            with torch.no_grad():
                current = task_composite.detach().to(dtype=self.task_composite_ema.dtype)
                if self.task_composite_ema_initialized.item() < 0.5:
                    self.task_composite_ema.copy_(current)
                    self.task_composite_ema_initialized.fill_(1.0)
                else:
                    self.task_composite_ema.mul_(self.normalization_decay)
                    self.task_composite_ema.add_(current, alpha=1.0 - self.normalization_decay)

        normalizer = torch.clamp(
            self.task_composite_ema.to(device=task_composite.device, dtype=task_composite.dtype),
            min=self.eps,
        )
        return task_composite / normalizer.detach(), normalizer


    def _event_focal_loss(
            self,
            pred: torch.Tensor,
            true: torch.Tensor,
            task_id: int,
        ) -> torch.Tensor:
        tau = self.thresholds[task_id].to(pred.device)
        if not torch.isfinite(tau):
            return pred.sum() * 0.0

        scale = torch.clamp(self.scales[task_id].to(pred.device), min=self.eps)
        logits = (pred - tau) / scale
        labels = (true >= tau).to(dtype=pred.dtype)
        pos_weight = self.event_pos_weights[task_id].to(pred.device)
        bce = F.binary_cross_entropy_with_logits(
            logits,
            labels,
            reduction="none",
            pos_weight=pos_weight,
        )
        prob = torch.sigmoid(logits)
        pt = prob * labels + (1.0 - prob) * (1.0 - labels)
        return (((1.0 - pt) ** self.focal_gamma) * bce).mean()


    def _peak_loss(
            self,
            y_pred: torch.Tensor,
            y_true: torch.Tensor,
        ) -> torch.Tensor:
        if self.num_tasks < 1:
            return y_pred.sum() * 0.0
        tau = self.thresholds[0].to(y_pred.device)
        if not torch.isfinite(tau):
            return y_pred.sum() * 0.0

        pred = y_pred[..., 0]
        true = y_true[..., 0]
        has_peak = true.max(dim=1).values >= tau
        if not torch.any(has_peak):
            return pred.sum() * 0.0

        scale = torch.clamp(self.scales[0].to(y_pred.device), min=self.eps)
        positions = torch.arange(pred.shape[1], device=pred.device, dtype=pred.dtype)
        soft_pos = torch.sum(torch.softmax(pred[has_peak] / scale, dim=1) * positions, dim=1)
        true_pos = torch.argmax(true[has_peak], dim=1).to(dtype=pred.dtype)
        denom = float(max(1, pred.shape[1] - 1))
        return torch.mean(torch.abs(soft_pos - true_pos) / denom)


    def _joint_event_focal_loss(
            self,
            y_pred: torch.Tensor,
            y_true: torch.Tensor,
        ) -> torch.Tensor:
        if self.num_tasks < 2 or y_pred.shape[-1] < 2 or y_true.shape[-1] < 2:
            return y_pred.sum() * 0.0
        tau_c = self.thresholds[0].to(y_pred.device)
        tau_f = self.thresholds[1].to(y_pred.device)
        if not torch.isfinite(tau_c) or not torch.isfinite(tau_f):
            return y_pred.sum() * 0.0

        scale_c = torch.clamp(self.scales[0].to(y_pred.device), min=self.eps)
        scale_f = torch.clamp(self.scales[1].to(y_pred.device), min=self.eps)
        z_c = (y_pred[..., 0] - tau_c) / scale_c
        z_f = (y_pred[..., 1] - tau_f) / scale_f
        logits = torch.minimum(z_c, z_f)
        labels = torch.logical_and(
            y_true[..., 0] >= tau_c,
            y_true[..., 1] >= tau_f,
        ).to(dtype=y_pred.dtype)
        bce = F.binary_cross_entropy_with_logits(
            logits,
            labels,
            reduction="none",
            pos_weight=self.joint_event_pos_weight.to(y_pred.device),
        )
        prob = torch.sigmoid(logits)
        pt = prob * labels + (1.0 - prob) * (1.0 - labels)
        return (((1.0 - pt) ** self.focal_gamma) * bce).mean()


    def forward(
            self,
            y_pred: torch.Tensor,
            y_true: torch.Tensor,
        ) -> tuple[torch.Tensor, dict]:
        reg_losses: list[torch.Tensor] = []
        wape_losses: list[torch.Tensor] = []
        event_losses: list[torch.Tensor] = []

        for task_id in range(self.num_tasks):
            pred = y_pred[..., task_id]
            true = y_true[..., task_id]
            scale = torch.clamp(self.scales[task_id].to(pred.device), min=self.eps)
            norm_error = (pred - true) / scale
            reg_losses.append(F.huber_loss(norm_error, torch.zeros_like(norm_error), delta=1.0, reduction="mean"))
            wape_losses.append(torch.sum(torch.abs(pred - true)) / (torch.sum(torch.abs(true)) + self.eps))
            event_losses.append(self._event_focal_loss(pred, true, task_id))

        reg_tensor = torch.stack(reg_losses, dim=0)
        wape_tensor = torch.stack(wape_losses, dim=0)
        event_tensor = torch.stack(event_losses, dim=0)
        reg_weights = self.reg_weights.to(y_pred.device)
        event_weights = self.event_weights.to(y_pred.device)
        task_weights = self.task_weights.to(y_pred.device)

        reg_loss = torch.sum(reg_weights * reg_tensor)
        wape_loss = torch.mean(wape_tensor)
        event_loss = torch.sum(event_weights * event_tensor)
        peak_loss = self._peak_loss(y_pred, y_true)
        joint_event_loss = self._joint_event_focal_loss(y_pred, y_true)
        task_composite_tensor = (
            reg_weights * reg_tensor
            + self.wape_weight * wape_tensor
            + event_weights * event_tensor
        )
        if self.peak_weight > 0:
            peak_terms = torch.zeros_like(task_composite_tensor)
            peak_terms[0] = self.peak_weight * peak_loss
            task_composite_tensor = task_composite_tensor + peak_terms

        normalized_task_tensor, task_normalizers = self._normalize_task_composite(task_composite_tensor)
        total_loss = torch.sum(task_weights * normalized_task_tensor)
        if self.joint_event_weight > 0:
            total_loss = total_loss + self.joint_event_weight * joint_event_loss

        pcgrad_objectives: list[torch.Tensor] = [
            normalized_task_tensor[i] for i in range(self.num_tasks)
        ]
        if self.joint_event_weight > 0:
            pcgrad_objectives.append(self.joint_event_weight * joint_event_loss)

        return total_loss, {
            "mode": "scenario-composite",
            "configured_mode": "scenario-composite",
            "effective_mode": "scenario-composite",
            "loss list": task_composite_tensor.detach().cpu().numpy().tolist(),
            "weights": task_weights.detach().cpu().numpy().tolist(),
            "weight_tensor": task_weights.detach(),
            "base_loss": torch.mean(task_composite_tensor).detach().cpu().item(),
            "reg_loss": reg_loss.detach().cpu().item(),
            "wape_loss": wape_loss.detach().cpu().item(),
            "event_loss": event_loss.detach().cpu().item(),
            "peak_loss": peak_loss.detach().cpu().item(),
            "joint_event_loss": joint_event_loss.detach().cpu().item(),
            "joint_event_weight": self.joint_event_weight,
            "joint_event_pos_weight": self.joint_event_pos_weight.detach().cpu().item(),
            "normalized_loss": total_loss.detach().cpu().item(),
            "total_loss": total_loss.detach().cpu().item(),
            "task_composite_loss_list": task_composite_tensor.detach().cpu().numpy().tolist(),
            "normalized_task_loss_list": normalized_task_tensor.detach().cpu().numpy().tolist(),
            "task_normalizers": task_normalizers.detach().cpu().numpy().tolist(),
            "reg_loss_list": reg_tensor.detach().cpu().numpy().tolist(),
            "wape_loss_list": wape_tensor.detach().cpu().numpy().tolist(),
            "event_loss_list": event_tensor.detach().cpu().numpy().tolist(),
            "reg_weights": reg_weights.detach().cpu().numpy().tolist(),
            "event_weights": event_weights.detach().cpu().numpy().tolist(),
            "thresholds": self.thresholds.detach().cpu().numpy().tolist(),
            "scales": self.scales.detach().cpu().numpy().tolist(),
            "event_pos_weights": self.event_pos_weights.detach().cpu().numpy().tolist(),
            "wape_weight": self.wape_weight,
            "peak_weight": self.peak_weight,
            "focal_gamma": self.focal_gamma,
            "normalization_decay": self.normalization_decay,
            "task_loss_tensor": normalized_task_tensor,
            "wape_loss_tensor": wape_loss,
            "event_loss_tensor": event_tensor,
            "peak_loss_tensor": peak_loss,
            "joint_event_loss_tensor": joint_event_loss,
            "pcgrad_objectives": pcgrad_objectives,
        }
