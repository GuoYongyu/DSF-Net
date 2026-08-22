# coding: utf-8
import math

import torch
from torch import nn


def _flatten(
        y_pred: torch.Tensor,
        y_true: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    task_num = y_pred.shape[-1]
    pred_list, true_list = [], []
    for task_id in range(task_num):
        pred_flat = y_pred[..., task_id].reshape(-1)
        true_flat = y_true[..., task_id].reshape(-1)
        pred_list.append(pred_flat)
        true_list.append(true_flat)
    return pred_list, true_list


def mse(y_true: torch.Tensor, y_pred: torch.Tensor) -> tuple[float, list[float]]:
    pred_task_list, true_task_list = _flatten(y_pred, y_true)
    task_mse = []
    for pred, true in zip(pred_task_list, true_task_list):
        mse = torch.mean((true - pred) ** 2)
        task_mse.append(mse.detach().cpu().item())
    return sum(task_mse) / len(task_mse), task_mse


def mape(y_true: torch.Tensor, y_pred: torch.Tensor) -> tuple[float, list[float]]:
    pred_task_list, true_task_list = _flatten(y_pred, y_true)
    task_mape = []
    for pred, true in zip(pred_task_list, true_task_list):
        abs_error = torch.abs(true - pred)
        true_denominator = torch.where(true == 0, torch.tensor(1e-8, device=true.device), true)
        mape = torch.mean(abs_error / torch.abs(true_denominator))
        task_mape.append(mape.detach().cpu().item() * 100)
    return sum(task_mape) / len(task_mape), task_mape


def smape(y_true: torch.Tensor, y_pred: torch.Tensor) -> tuple[float, list[float]]:
    pred_task_list, true_task_list = _flatten(y_pred, y_true)
    task_smape = []
    for pred, true in zip(pred_task_list, true_task_list):
        abs_error = torch.abs(true - pred)
        denominator = (torch.abs(true) + torch.abs(pred)) / 2
        denominator = torch.where(denominator == 0, torch.tensor(1e-8, device=denominator.device), denominator)
        smape = torch.mean(abs_error / denominator)
        task_smape.append(smape.detach().cpu().item() * 100)
    return sum(task_smape) / len(task_smape), task_smape


def r2_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> tuple[float, list[float]]:
    pred_task_list, true_task_list = _flatten(y_pred, y_true)
    task_r2 = []
    for pred, true in zip(pred_task_list, true_task_list):
        ss_res = torch.sum((true - pred) ** 2)
        true_mean = torch.mean(true)
        ss_tot = torch.sum((true - true_mean) ** 2)

        if ss_tot < 1e-8:
            r2 = 0.0
        else:
            r2 = 1 - (ss_res / ss_tot)
        task_r2.append(r2.detach().cpu().item())
    return sum(task_r2) / len(task_r2), task_r2


def rmse(y_true: torch.Tensor, y_pred: torch.Tensor) -> tuple[float, list[float]]:
    pred_task_list, true_task_list = _flatten(y_pred, y_true)
    task_rmse = []
    for pred, true in zip(pred_task_list, true_task_list):
        mse = torch.mean((true - pred) ** 2)
        rmse = math.sqrt(mse.detach().cpu().item())
        task_rmse.append(rmse)
    return sum(task_rmse) / len(task_rmse), task_rmse


def huber_loss(
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        delta: float = 1.0
    ) -> tuple[float, list[float]]:
    pred_task_list, true_task_list = _flatten(y_pred, y_true)
    task_huber = []
    for pred, true in zip(pred_task_list, true_task_list):
        loss: torch.Tensor = nn.HuberLoss(delta=delta)(true, pred)
        task_huber.append(loss.detach().cpu().item())
    return sum(task_huber) / len(task_huber), task_huber
