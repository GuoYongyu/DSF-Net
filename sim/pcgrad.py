# coding: utf-8

import torch
from torch import nn


def _dot(grads_a: list[torch.Tensor | None], grads_b: list[torch.Tensor | None]) -> torch.Tensor:
    value = None
    for ga, gb in zip(grads_a, grads_b):
        if ga is None or gb is None:
            continue
        part = torch.sum(ga * gb)
        value = part if value is None else value + part
    if value is None:
        return torch.tensor(0.0)
    return value


def _project_if_conflict(
        grads_a: list[torch.Tensor | None],
        grads_b: list[torch.Tensor | None],
    ) -> list[torch.Tensor | None]:
    dot_ab = _dot(grads_a, grads_b)
    if dot_ab >= 0:
        return grads_a

    dot_bb = _dot(grads_b, grads_b)
    if dot_bb <= 0:
        return grads_a

    scale = dot_ab / dot_bb
    projected: list[torch.Tensor | None] = []
    for ga, gb in zip(grads_a, grads_b):
        if ga is None:
            projected.append(None)
        elif gb is None:
            projected.append(ga)
        else:
            projected.append(ga - scale * gb)
    return projected


def pcgrad_backward(task_losses: list[torch.Tensor], model: nn.Module):
    params = [p for p in model.parameters() if p.requires_grad]
    if len(params) == 0 or len(task_losses) == 0:
        return

    task_grads: list[list[torch.Tensor | None]] = []
    for idx, task_loss in enumerate(task_losses):
        grads = torch.autograd.grad(
            task_loss,
            params,
            retain_graph=idx < len(task_losses) - 1,
            allow_unused=True,
        )
        task_grads.append([g.clone() if g is not None else None for g in grads])

    projected_grads: list[list[torch.Tensor | None]] = []
    for i in range(len(task_grads)):
        grad_i = task_grads[i]
        for j in torch.randperm(len(task_grads)).tolist():
            if i == j:
                continue
            grad_i = _project_if_conflict(grad_i, task_grads[j])
        projected_grads.append(grad_i)

    for param_idx, param in enumerate(params):
        valid = [g[param_idx] for g in projected_grads if g[param_idx] is not None]
        if len(valid) == 0:
            param.grad = None
            continue
        param.grad = torch.stack(valid, dim=0).mean(dim=0)


def make_weighted_task_losses(loss_info: dict) -> list[torch.Tensor]:
    objectives = loss_info.get("pcgrad_objectives")
    if objectives is not None:
        return [objective for objective in objectives]

    task_loss: torch.Tensor = loss_info["task_loss_tensor"]
    weights: torch.Tensor | None = loss_info.get("weight_tensor")
    if weights is None:
        return [t_loss for t_loss in task_loss]

    weights = weights.to(device=task_loss.device, dtype=task_loss.dtype).detach()
    if weights.numel() != task_loss.numel():
        raise ValueError("loss weight count must match task loss count")
    return [weights[i] * task_loss[i] for i in range(task_loss.shape[0])]


def collect_module_grads(
        loss: torch.Tensor,
        module: nn.Module,
        retain_graph: bool = True,
    ) -> list[tuple[torch.nn.Parameter, torch.Tensor | None]]:
    params = [p for p in module.parameters() if p.requires_grad]
    if len(params) == 0:
        return []
    grads = torch.autograd.grad(
        loss,
        params,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    return list(zip(params, grads))


def assign_module_grads(param_grads: list[tuple[torch.nn.Parameter, torch.Tensor | None]]) -> None:
    for param, grad in param_grads:
        param.grad = None if grad is None else grad.detach().clone()
