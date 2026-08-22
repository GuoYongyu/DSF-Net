# coding: utf-8
import csv
import os
import time
import warnings
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from sim.config import *
from sim.pred_dump import save_test_predictions
from sim.net import DSFNet
from sim.loss import MSELoss, HuberLoss, MultiTasksLoss, ScenarioCompositeLoss
from sim.loss_func import mse, rmse, r2_score, mape, smape, huber_loss
from sim.pcgrad import (
    assign_module_grads,
    collect_module_grads,
    make_weighted_task_losses,
    pcgrad_backward,
)
from sim.utils import save_training_process
from logger.logger import LOG


EPS = 1e-12


def _metric_is_better(
    value: float,
    best: float,
    mode: str,
    tie_breaker: float | None = None,
    best_tie_breaker: float | None = None,
    tolerance: float = 1e-12,
) -> bool:
    if np.isnan(value):
        return False
    if mode == "min":
        if value < best - tolerance:
            return True
        tied = abs(value - best) <= tolerance
    elif mode == "max":
        if value > best + tolerance:
            return True
        tied = abs(value - best) <= tolerance
    else:
        raise ValueError("mode must be 'min' or 'max'")
    if not tied or tie_breaker is None or best_tie_breaker is None:
        return False
    if np.isnan(tie_breaker):
        return False
    return tie_breaker < best_tie_breaker - tolerance


def _checkpoint_metric_specs(
        model_dir: str,
        weight_tag: str,
    ) -> dict[str, tuple[str, str, float]]:
    modes = {
        "huber": "min",
        "rmse": "min",
        "balanced-rmse": "min",
        "f1-hc": "max",
        "f1-hf": "max",
        "f1-joint": "max",
        "pte": "min",
        "scenario-priority": "max",
    }
    return {
        metric_name: (
            os.path.join(model_dir, f"best-{metric_name}_{weight_tag}.pth"),
            mode,
            float("inf") if mode == "min" else float("-inf"),
        )
        for metric_name, mode in modes.items()
    }


def _validate_joint_checkpoint_mode(
        target_mode: str,
        best_checkpoint_metric: str,
        checkpoint_metrics: tuple[str, ...] | list[str] | None,
    ) -> None:
    requested = set(checkpoint_metrics or ())
    if best_checkpoint_metric == "f1-joint":
        requested.add("f1-joint")
    if target_mode != "multi" and "f1-joint" in requested:
        raise ValueError("f1-joint requires target_mode=multi")


def _flatten_batches(batches: list[torch.Tensor]) -> np.ndarray:
    if len(batches) == 0:
        return np.empty((0,), dtype=np.float32)
    arrays = [b.detach().cpu().numpy() for b in batches]
    return np.concatenate(arrays, axis=0)


def _f1_at_threshold(y_true: np.ndarray, y_pred: np.ndarray, tau: float) -> float:
    true_flag = y_true >= tau
    pred_flag = y_pred >= tau
    tp = float(np.logical_and(true_flag, pred_flag).sum())
    fp = float(np.logical_and(~true_flag, pred_flag).sum())
    fn = float(np.logical_and(true_flag, ~pred_flag).sum())
    precision = tp / (tp + fp + EPS)
    recall = tp / (tp + fn + EPS)
    return 2.0 * precision * recall / (precision + recall + EPS)


def _pte_hours(y_true: np.ndarray, y_pred: np.ndarray, tau: float) -> float:
    has_peak = np.max(y_true, axis=1) >= tau
    if not np.any(has_peak):
        return float("nan")
    true_arg = np.argmax(y_true[has_peak], axis=1)
    pred_arg = np.argmax(y_pred[has_peak], axis=1)
    abs_err = np.abs(pred_arg - true_arg).astype(np.float64)
    return float(np.mean(abs_err))


def _scenario_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        target_mode: str,
        tau_c: float,
        tau_f: float,
        metric_weights: tuple[float, float, float],
        prediction_dim: int,
    ) -> dict[str, float]:
    if y_true.ndim == 2:
        y_true = y_true[:, :, np.newaxis]
    if y_pred.ndim == 2:
        y_pred = y_pred[:, :, np.newaxis]
    if y_true.ndim != 3 or y_pred.ndim != 3:
        return {
            "F1_hc": float("nan"),
            "PTE": float("nan"),
            "F1_hf": float("nan"),
            "Precision_joint": float("nan"),
            "Recall_joint": float("nan"),
            "F1_joint": float("nan"),
            "PTE_score": float("nan"),
            "scenario_score": float("nan"),
        }

    w_hc, w_hf, w_pte = metric_weights
    f1_hc = float("nan")
    pte = float("nan")
    f1_hf = float("nan")
    precision_joint = float("nan")
    recall_joint = float("nan")
    f1_joint = float("nan")

    if target_mode == "multi":
        if y_true.shape[-1] >= 1 and not np.isnan(tau_c):
            f1_hc = _f1_at_threshold(y_true[..., 0], y_pred[..., 0], tau=tau_c)
            pte = _pte_hours(y_true[..., 0], y_pred[..., 0], tau=tau_c)
        if y_true.shape[-1] >= 2 and not np.isnan(tau_f):
            f1_hf = _f1_at_threshold(y_true[..., 1], y_pred[..., 1], tau=tau_f)
        if (
            y_true.shape[-1] >= 2
            and y_pred.shape[-1] >= 2
            and np.isfinite(tau_c)
            and np.isfinite(tau_f)
        ):
            true_joint = np.logical_and(y_true[..., 0] >= tau_c, y_true[..., 1] >= tau_f)
            pred_joint = np.logical_and(y_pred[..., 0] >= tau_c, y_pred[..., 1] >= tau_f)
            tp = float(np.logical_and(true_joint, pred_joint).sum())
            fp = float(np.logical_and(~true_joint, pred_joint).sum())
            fn = float(np.logical_and(true_joint, ~pred_joint).sum())
            precision_joint = tp / (tp + fp + EPS)
            recall_joint = tp / (tp + fn + EPS)
            f1_joint = (
                2.0 * precision_joint * recall_joint
                / (precision_joint + recall_joint + EPS)
            )
    elif target_mode == "crowding":
        if y_true.shape[-1] >= 1 and not np.isnan(tau_c):
            f1_hc = _f1_at_threshold(y_true[..., 0], y_pred[..., 0], tau=tau_c)
            pte = _pte_hours(y_true[..., 0], y_pred[..., 0], tau=tau_c)
    elif target_mode == "equiv_flow":
        if y_true.shape[-1] >= 1 and not np.isnan(tau_f):
            f1_hf = _f1_at_threshold(y_true[..., 0], y_pred[..., 0], tau=tau_f)

    horizon_denom = float(max(1, prediction_dim - 1))
    pte_score = float("nan")
    if not np.isnan(pte):
        pte_score = max(0.0, 1.0 - (pte / horizon_denom))

    weighted_terms: list[tuple[float, float]] = []
    if not np.isnan(f1_hc) and w_hc > 0:
        weighted_terms.append((w_hc, f1_hc))
    if not np.isnan(f1_hf) and w_hf > 0:
        weighted_terms.append((w_hf, f1_hf))
    if not np.isnan(pte_score) and w_pte > 0:
        weighted_terms.append((w_pte, pte_score))

    if weighted_terms:
        weight_sum = sum(w for w, _ in weighted_terms)
        scenario_score = sum(w * value for w, value in weighted_terms) / weight_sum
    else:
        scenario_score = float("nan")

    return {
        "F1_hc": float(f1_hc),
        "PTE": float(pte),
        "F1_hf": float(f1_hf),
        "Precision_joint": float(precision_joint),
        "Recall_joint": float(recall_joint),
        "F1_joint": float(f1_joint),
        "PTE_score": float(pte_score),
        "scenario_score": float(scenario_score),
    }


def _selection_diagnostics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        tau_c: float,
        target_mode: str = "multi",
    ) -> dict[str, float]:
    """Validation-only diagnostics used to choose a head configuration.

    This mirrors the manuscript definitions for MAE, Pearson correlation and
    peak-time accuracy, without importing the final test-table machinery.
    """
    if y_true.ndim != 3 or y_pred.ndim != 3 or y_true.shape != y_pred.shape:
        raise ValueError("selection diagnostics require matched (sample, horizon, task) arrays")
    output: dict[str, float] = {
        "crowding_mae": float("nan"),
        "crowding_correlation": float("nan"),
        "flow_mae": float("nan"),
        "flow_correlation": float("nan"),
    }
    if target_mode == "multi":
        task_specs = ((0, "crowding"), (1, "flow"))
    elif target_mode == "crowding":
        task_specs = ((0, "crowding"),)
    elif target_mode == "equiv_flow":
        task_specs = ((0, "flow"),)
    else:
        raise ValueError(f"Unsupported target_mode for selection diagnostics: {target_mode}")
    for task_index, task_name in task_specs:
        truth = y_true[..., task_index].reshape(-1)
        prediction = y_pred[..., task_index].reshape(-1)
        output[f"{task_name}_mae"] = float(np.mean(np.abs(prediction - truth)))
        if np.std(truth) <= EPS or np.std(prediction) <= EPS:
            output[f"{task_name}_correlation"] = float("nan")
        else:
            output[f"{task_name}_correlation"] = float(np.corrcoef(truth, prediction)[0, 1])
    if target_mode == "equiv_flow" or not np.isfinite(tau_c):
        output["PTA@3"] = 0.0
        output["PTA@6"] = 0.0
        return output
    crowd_truth = y_true[..., 0]
    crowd_prediction = y_pred[..., 0]
    peak_rows = np.max(crowd_truth, axis=1) >= tau_c
    for window in (3, 6):
        if not np.any(peak_rows):
            output[f"PTA@{window}"] = 0.0
            continue
        true_peak = np.argmax(crowd_truth[peak_rows], axis=1)
        predicted_peak = np.argmax(crowd_prediction[peak_rows], axis=1)
        output[f"PTA@{window}"] = float(np.mean(np.abs(true_peak - predicted_peak) <= window))
    return output


def _train_thresholds(
        train_loader: DataLoader,
        target_mode: str,
        percentile: float,
    ) -> tuple[float, float]:
    y_batches: list[np.ndarray] = []
    for batch in train_loader:
        _, yb, _ = _unpack_batch(batch)
        y_batches.append(yb.detach().cpu().numpy())
    if len(y_batches) == 0:
        return float("nan"), float("nan")

    y_train = np.concatenate(y_batches, axis=0)
    if y_train.ndim == 2:
        y_train = y_train[:, :, np.newaxis]

    tau_c = float("nan")
    tau_f = float("nan")
    if target_mode == "multi":
        if y_train.shape[-1] >= 1:
            tau_c = float(np.percentile(y_train[..., 0].reshape(-1), percentile))
        if y_train.shape[-1] >= 2:
            tau_f = float(np.percentile(y_train[..., 1].reshape(-1), percentile))
    elif target_mode == "crowding":
        if y_train.shape[-1] >= 1:
            tau_c = float(np.percentile(y_train[..., 0].reshape(-1), percentile))
    elif target_mode == "equiv_flow":
        if y_train.shape[-1] >= 1:
            tau_f = float(np.percentile(y_train[..., 0].reshape(-1), percentile))
    return tau_c, tau_f


def _fmt_float(v: float) -> str:
    if np.isnan(v):
        return "nan"
    return f"{v:.4f}"


def _thresholds_for_target_mode(
        target_mode: str,
        tau_c: float,
        tau_f: float,
        task_num: int,
    ) -> list[float]:
    if target_mode == "multi":
        thresholds = [tau_c, tau_f]
    elif target_mode == "crowding":
        thresholds = [tau_c]
    elif target_mode == "equiv_flow":
        thresholds = [tau_f]
    else:
        thresholds = [float("nan")] * task_num
    if len(thresholds) < task_num:
        thresholds.extend([float("nan")] * (task_num - len(thresholds)))
    return thresholds[:task_num]


def _robust_scale(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 1.0
    q75, q25 = np.percentile(values, [75, 25])
    scale = float(q75 - q25)
    if scale <= EPS:
        scale = float(np.std(values))
    if scale <= EPS:
        scale = float(np.mean(np.abs(values)))
    if scale <= EPS:
        scale = 1.0
    return scale


def _event_pos_weight(values: np.ndarray, tau: float) -> float:
    if not np.isfinite(tau):
        return 1.0
    flags = values >= tau
    pos = float(flags.sum())
    neg = float(flags.size - flags.sum())
    if pos <= 0 or neg <= 0:
        return 1.0
    return float(np.clip(neg / pos, 1.0, 20.0))


def _train_target_profile(
        train_loader: DataLoader,
        target_mode: str,
        percentile: float,
        include_joint_pos_weight: bool = False,
    ) -> (
        tuple[float, float, list[float], list[float]]
        | tuple[float, float, list[float], list[float], float]
    ):
    y_batches: list[np.ndarray] = []
    for batch in train_loader:
        _, yb, _ = _unpack_batch(batch)
        y_batches.append(yb.detach().cpu().numpy())
    if len(y_batches) == 0:
        empty_profile = (float("nan"), float("nan"), [1.0], [1.0])
        return (*empty_profile, 1.0) if include_joint_pos_weight else empty_profile

    y_train = np.concatenate(y_batches, axis=0)
    if y_train.ndim == 2:
        y_train = y_train[:, :, np.newaxis]

    tau_c = float("nan")
    tau_f = float("nan")
    if target_mode == "multi":
        if y_train.shape[-1] >= 1:
            tau_c = float(np.percentile(y_train[..., 0].reshape(-1), percentile))
        if y_train.shape[-1] >= 2:
            tau_f = float(np.percentile(y_train[..., 1].reshape(-1), percentile))
    elif target_mode == "crowding" and y_train.shape[-1] >= 1:
        tau_c = float(np.percentile(y_train[..., 0].reshape(-1), percentile))
    elif target_mode == "equiv_flow" and y_train.shape[-1] >= 1:
        tau_f = float(np.percentile(y_train[..., 0].reshape(-1), percentile))

    thresholds = _thresholds_for_target_mode(target_mode, tau_c, tau_f, y_train.shape[-1])
    scales: list[float] = []
    pos_weights: list[float] = []
    for task_id in range(y_train.shape[-1]):
        values = y_train[..., task_id].reshape(-1)
        scales.append(_robust_scale(values))
        pos_weights.append(_event_pos_weight(values, thresholds[task_id]))
    profile = (tau_c, tau_f, scales, pos_weights)
    if include_joint_pos_weight:
        return (*profile, _joint_event_pos_weight_from_targets(
            y_train,
            target_mode=target_mode,
            tau_c=tau_c,
            tau_f=tau_f,
        ))
    return profile


def _joint_event_pos_weight_from_targets(
        y_train: np.ndarray,
        target_mode: str,
        tau_c: float,
        tau_f: float,
    ) -> float:
    if (
        target_mode != "multi"
        or y_train.ndim != 3
        or y_train.shape[-1] < 2
        or not np.isfinite(tau_c)
        or not np.isfinite(tau_f)
    ):
        return 1.0
    labels = np.logical_and(y_train[..., 0] >= tau_c, y_train[..., 1] >= tau_f)
    positive_count = int(labels.sum())
    negative_count = int(labels.size - positive_count)
    if positive_count <= 0 or negative_count <= 0:
        return 1.0
    return float(np.clip(negative_count / positive_count, 1.0, 20.0))


def _loss_info_scalars(loss_info: dict) -> dict[str, float]:
    scalars: dict[str, float] = {}
    skip_keys = {
        "pcgrad_objectives",
        "task_loss_tensor",
        "weight_tensor",
        "mape_loss_tensor",
        "wape_loss_tensor",
        "event_loss_tensor",
        "peak_loss_tensor",
        "joint_event_loss_tensor",
        "residual_loss_tensor",
    }
    for key, value in loss_info.items():
        if key in skip_keys or isinstance(value, str):
            continue
        if isinstance(value, (int, float, np.integer, np.floating)):
            scalars[key] = float(value)
        elif isinstance(value, (list, tuple)):
            for idx, item in enumerate(value, start=1):
                if isinstance(item, (int, float, np.integer, np.floating)):
                    scalars[f"{key}_{idx}"] = float(item)
    return scalars


def _grad_norm(parameters) -> float:
    total = 0.0
    for param in parameters:
        if param.grad is None:
            continue
        value = float(param.grad.detach().data.norm(2).item())
        total += value * value
    return float(total ** 0.5)


def _write_loss_breakdown_csv(path: str, rows: list[dict[str, float]]) -> None:
    if len(rows) == 0:
        return
    fields: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _unpack_batch(batch) -> tuple[tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor | None]:
    if len(batch) == 3:
        xb, yb, baseline = batch
        return xb, yb, baseline
    xb, yb = batch
    return xb, yb, None


def _move_batch_to_device(
        batch,
        device: str | int,
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor | None]:
    xb, yb, baseline = _unpack_batch(batch)
    xb = tuple(x.to(device) for x in xb)
    yb = yb.to(device)
    if baseline is not None:
        baseline = baseline.to(device)
    return xb, yb, baseline


def _apply_residual_baseline(
        raw_output: torch.Tensor,
        baseline: torch.Tensor | None,
    ) -> torch.Tensor:
    if baseline is None:
        return raw_output
    return baseline + raw_output


def _normalized_residual_loss(
        raw_output: torch.Tensor,
        y_true: torch.Tensor,
        baseline: torch.Tensor | None,
        target_scales: list[float],
    ) -> torch.Tensor:
    if baseline is None:
        return raw_output.sum() * 0.0
    scale = torch.tensor(target_scales, dtype=raw_output.dtype, device=raw_output.device)
    while scale.ndim < raw_output.ndim:
        scale = scale.unsqueeze(0)
    scale = torch.clamp(scale, min=EPS)
    residual_target = y_true - baseline
    return F.smooth_l1_loss(raw_output / scale, residual_target / scale)


def _evaluate_dsfnet(
        model: DSFNet,
        data_loader: DataLoader,
        device: str | int,
    ) -> tuple[dict[str, float], dict[str, list[float]], list[torch.Tensor], list[torch.Tensor]]:
    task_num = len(data_loader.dataset[0][1][0])
    total_r2, total_mse, total_rmse, total_mape, total_smape, total_huber = 0, 0, 0, 0, 0, 0
    task_r2 = [0.0] * task_num
    task_mse = [0.0] * task_num
    task_rmse = [0.0] * task_num
    task_mape = [0.0] * task_num
    task_smape = [0.0] * task_num
    task_huber = [0.0] * task_num
    y_true_batches: list[torch.Tensor] = []
    y_pred_batches: list[torch.Tensor] = []

    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            xb, yb, baseline = _move_batch_to_device(batch, device)
            curb, upb, downb, inb, outb = xb
            raw_output = model(curb, upb, downb, inb, outb)
            output = _apply_residual_baseline(raw_output, baseline)
            y_true_batches.append(yb.detach().cpu())
            y_pred_batches.append(output.detach().cpu())

            value, task_values = r2_score(yb, output)
            total_r2 += value * yb.size(0)
            for i, task_value in enumerate(task_values):
                task_r2[i] += task_value * yb.size(0)

            value, task_values = mse(yb, output)
            total_mse += value * yb.size(0)
            for i, task_value in enumerate(task_values):
                task_mse[i] += task_value * yb.size(0)

            value, task_values = rmse(yb, output)
            total_rmse += value * yb.size(0)
            for i, task_value in enumerate(task_values):
                task_rmse[i] += task_value * yb.size(0)

            value, task_values = mape(yb, output)
            total_mape += value * yb.size(0)
            for i, task_value in enumerate(task_values):
                task_mape[i] += task_value * yb.size(0)

            value, task_values = smape(yb, output)
            total_smape += value * yb.size(0)
            for i, task_value in enumerate(task_values):
                task_smape[i] += task_value * yb.size(0)

            value, task_values = huber_loss(yb, output)
            total_huber += value * yb.size(0)
            for i, task_value in enumerate(task_values):
                task_huber[i] += task_value * yb.size(0)

    dataset_size = len(data_loader.dataset)
    metrics = {
        "R2": total_r2 / dataset_size,
        "MSE": total_mse / dataset_size,
        "RMSE": total_rmse / dataset_size,
        "MAPE": total_mape / dataset_size,
        "SMAPE": total_smape / dataset_size,
        "Huber": total_huber / dataset_size,
    }
    task_metrics = {
        "R2": [value / dataset_size for value in task_r2],
        "MSE": [value / dataset_size for value in task_mse],
        "RMSE": [value / dataset_size for value in task_rmse],
        "MAPE": [value / dataset_size for value in task_mape],
        "SMAPE": [value / dataset_size for value in task_smape],
        "Huber": [value / dataset_size for value in task_huber],
    }
    return metrics, task_metrics, y_true_batches, y_pred_batches


def _print_metrics_block(
        header: str,
        metrics: dict[str, float],
        task_metrics: dict[str, list[float]],
        task_num: int,
    ) -> None:
    print(
        f"{header}\n"
        + f"       Loss(R2):      {metrics['R2']:.4f}\n"
        + f"       Loss(MSE):     {metrics['MSE']:.4f}\n"
        + f"       Loss(RMSE):    {metrics['RMSE']:.4f}\n"
        + f"       Loss(MAPE):    {metrics['MAPE']:.4f}\n"
        + f"       Loss(SMAPE):   {metrics['SMAPE']:.4f}\n"
        + f"       Loss(Huber):   {metrics['Huber']:.4f}\n"
        + f"       Mean(Task R2):    {sum(task_metrics['R2']) / task_num:.4f}\n"
        + f"       Mean(Task MSE):   {sum(task_metrics['MSE']) / task_num:.4f}\n"
        + f"       Mean(Task RMSE):  {sum(task_metrics['RMSE']) / task_num:.4f}\n"
        + f"       Mean(Task MAPE):  {sum(task_metrics['MAPE']) / task_num:.4f}\n"
        + f"       Mean(Task SMAPE): {sum(task_metrics['SMAPE']) / task_num:.4f}\n"
        + f"       Mean(Task Huber): {sum(task_metrics['Huber']) / task_num:.4f}"
    )
    for task_id in range(task_num):
        print(
            f"       Task-{task_id + 1}: "
            + f"R2={task_metrics['R2'][task_id]:.4f}, "
            + f"MSE={task_metrics['MSE'][task_id]:.4f}, "
            + f"RMSE={task_metrics['RMSE'][task_id]:.4f}, "
            + f"MAPE={task_metrics['MAPE'][task_id]:.4f}, "
            + f"SMAPE={task_metrics['SMAPE'][task_id]:.4f}, "
            + f"Huber={task_metrics['Huber'][task_id]:.4f}"
        )


def train(
        p_id: int,  # process id
        segment: str,
        weights: tuple[float, float],
        train_loader: DataLoader,
        valid_loader: DataLoader,
        test_loader: DataLoader,
        device: str | int,
        target_mode: str = "multi",
        prediction_dim: int = OUTPUT_DIM,
        time_net: str = ["lstm", "attention_lstm", "timesnet"][0],
        time_net_kwargs: dict = TIME_LSTM_ARGS,
        space_net: str = ["encoder", "cross_attention", "spatial_conv"][0],
        space_net_kwargs: dict = SPACE_ENCODER_ARGS,
        fusion_net: str = [
            "weighted_concat",
            "weighted_fusion",
            "feature_mapping",
            "cross_attention",
            "self_attention"
        ][0],
        fusion_net_kwargs: dict = FUSION_WEIGHTED_CONCAT_ARGS,
        epochs: int = 100,
        optimizer_type: str = ["adam", "sgd", "rmsprop", "adamw"][0],
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        grad_clip_norm: float | None = None,
        loss_weighting: str = "dynamic",
        fixed_task_weights: list[float] | None = None,
        loss_warmup_epochs: int = 0,
        loss_warmup_fixed_task_weights: list[float] | None = None,
        stage_task_weights_end: list[float] | None = None,
        stage_start_epoch: int | None = None,
        stage_end_epoch: int | None = None,
        use_pcgrad: bool = False,
        pcgrad_include_mape: bool = True,
        criterion_type: str = ["mse", "huber", "scenario"][0],
        mape_lambda: float = 0.1,
        normalize_task_losses: bool = False,
        scenario_loss_reg_weights: list[float] | None = None,
        scenario_loss_event_weights: list[float] | None = None,
        scenario_loss_wape_weight: float = 0.1,
        scenario_loss_peak_weight: float = 0.0,
        scenario_loss_joint_weight: float = 0.0,
        scenario_loss_focal_gamma: float = 2.0,
        scheduler_type: str = ["annealing", "plateau", "linear", "polynomial"][0],
        save_charts: bool = True,
        dump_test_preds: bool = True,
        run_tag: str | None = None,
        split_mode: str = "random",
        input_ablation: str = "full",
        residual_baseline: str = "none",
        residual_loss_weight: float = 0.0,
        best_checkpoint_metric: str = "huber",
        checkpoint_metrics: tuple[str, ...] | None = None,
        scenario_metric_weights: tuple[float, float, float] = (0.4, 0.4, 0.2),
        scenario_threshold_percentile: float = SCENARIO_THRESHOLD_PERCENTILE,
        evaluate_test: bool = True,
        decouple_space: bool = False,
        multi_task_architecture: str = "cross-stitch-2",
        multitask_head_config: dict | None = None,
        experiment_meta: dict[str, Any] | None = None,
    ) -> dict[str, float]:
    time_net_kwargs["feature_num"] = prediction_dim
    space_net_kwargs["feature_num"] = prediction_dim

    best_valid_loss = float("inf")
    best_valid_scenario_score = float("-inf")
    best_valid_scenario_tiebreak_loss = float("inf")
    train_loss = dict()
    valid_r2_loss = dict()
    valid_mse_loss = dict()
    valid_rmse_loss = dict()
    valid_mape_loss = dict()
    valid_smape_loss = dict()
    valid_huber_loss = dict()
    train_time = 0

    task_num = len(train_loader.dataset[0][1][0])
    tau_c, tau_f, target_scales, event_pos_weights, joint_event_pos_weight = _train_target_profile(
        train_loader=train_loader,
        target_mode=target_mode,
        percentile=scenario_threshold_percentile,
        include_joint_pos_weight=True,
    )
    target_thresholds = _thresholds_for_target_mode(target_mode, tau_c, tau_f, task_num)
    loss_breakdown_history: list[dict[str, float]] = []
    valid_r2_task_loss = [dict() for _ in range(task_num)]
    valid_mse_task_loss = [dict() for _ in range(task_num)]
    valid_rmse_task_loss = [dict() for _ in range(task_num)]
    valid_mape_task_loss = [dict() for _ in range(task_num)]
    valid_smape_task_loss = [dict() for _ in range(task_num)]
    valid_huber_task_loss = [dict() for _ in range(task_num)]

    model = DSFNet(
        time_net=time_net,
        time_kwargs=time_net_kwargs,
        space_net=space_net,
        space_kwargs=space_net_kwargs,
        fusion_net=fusion_net,
        fusion_kwargs=fusion_net_kwargs,
        task_num=task_num,
        output_dim=prediction_dim,
        decouple_space=decouple_space,
        multi_task_architecture=multi_task_architecture,
        multitask_head_config=multitask_head_config,
    )
    model.to(device)
    model_name = f"tn-{time_net}_sn-{space_net}_fn-{fusion_net}"
    if residual_baseline != "none":
        if residual_baseline not in {"ha", "last"}:
            raise ValueError("residual_baseline must be one of ['none', 'ha', 'last']")
        if residual_loss_weight < 0:
            raise ValueError("residual_loss_weight must be >= 0")
        model_name = f"{model_name}_rb-{residual_baseline}"
    if input_ablation != "full":
        if input_ablation not in {"no-tributary", "no-upstream", "no-downstream"}:
            raise ValueError(
                "input_ablation must be one of "
                "['full', 'no-tributary', 'no-upstream', 'no-downstream']"
            )
        model_name = f"{model_name}_ia-{input_ablation.replace('-', '_')}"
    if model.multi_task_architecture != "shared":
        model_name = (
            f"{model_name}_{model.multi_task_architecture.replace('-', '_')}"
        )

    if optimizer_type == "adam":
        optimizer: torch.optim.Optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay,
        )
    elif optimizer_type == "sgd":
        optimizer: torch.optim.Optimizer = torch.optim.SGD(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay,
        )
    elif optimizer_type == "rmsprop":
        optimizer: torch.optim.Optimizer = torch.optim.RMSprop(
            model.parameters(), lr=learning_rate, alpha=0.99, momentum=0.9, weight_decay=weight_decay,
        )
    elif optimizer_type == "adamw":
        optimizer: torch.optim.Optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"optimizer {optimizer} is not supported")

    if criterion_type == "mse":
        criterion: nn.Module = MultiTasksLoss(
            num_tasks=task_num,
            loss_type=MSELoss,
            weighting=loss_weighting,
            fixed_weights=fixed_task_weights,
            warmup_epochs=loss_warmup_epochs,
            warmup_fixed_weights=loss_warmup_fixed_task_weights,
            mape_lambda=mape_lambda,
            task_scales=target_scales if normalize_task_losses else None,
        )
    elif criterion_type == "huber":
        criterion: nn.Module = MultiTasksLoss(
            num_tasks=task_num,
            loss_type=HuberLoss,
            weighting=loss_weighting,
            fixed_weights=fixed_task_weights,
            warmup_epochs=loss_warmup_epochs,
            warmup_fixed_weights=loss_warmup_fixed_task_weights,
            mape_lambda=mape_lambda,
            task_scales=target_scales if normalize_task_losses else None,
        )
    elif criterion_type in {"scenario", "scenario-composite"}:
        criterion = ScenarioCompositeLoss(
            num_tasks=task_num,
            thresholds=target_thresholds,
            scales=target_scales,
            event_pos_weights=event_pos_weights,
            reg_weights=scenario_loss_reg_weights,
            event_weights=scenario_loss_event_weights,
            task_weights=fixed_task_weights,
            wape_weight=scenario_loss_wape_weight,
            peak_weight=scenario_loss_peak_weight,
            joint_event_weight=scenario_loss_joint_weight,
            joint_event_pos_weight=joint_event_pos_weight,
            focal_gamma=scenario_loss_focal_gamma,
        )
    else:
        raise ValueError(f"criterion {criterion_type} is not supported")

    criterion.to(device)
    criterion_params = [p for p in criterion.parameters() if p.requires_grad]
    if criterion_params:
        optimizer.add_param_group({"params": criterion_params})

    if loss_warmup_epochs < 0 or loss_warmup_epochs > epochs:
        raise ValueError("loss_warmup_epochs must be in [0, epochs]")
    if loss_warmup_fixed_task_weights is not None and len(loss_warmup_fixed_task_weights) != task_num:
        raise ValueError("loss_warmup_fixed_task_weights length must equal task_num")

    if stage_task_weights_end is not None:
        if loss_warmup_epochs > 0:
            raise ValueError("staged task weights cannot be used with warmup weighting")
        if fixed_task_weights is None:
            raise ValueError("fixed_task_weights must be set when using staged task weights")
        if loss_weighting != "fixed":
            raise ValueError("staged task weights only support loss_weighting=fixed")
        if stage_start_epoch is None or stage_end_epoch is None:
            raise ValueError("stage_start_epoch and stage_end_epoch must be set when using staged task weights")
        if len(stage_task_weights_end) != task_num or len(fixed_task_weights) != task_num:
            raise ValueError("staged task weights length must equal task_num")
        if stage_start_epoch < 1 or stage_end_epoch < stage_start_epoch or stage_end_epoch > epochs:
            raise ValueError("invalid staged epoch range")

    checkpoint_metric_choices = {"huber", "rmse", "balanced-rmse", "f1-hc", "f1-hf", "f1-joint", "pte", "scenario-priority"}
    if best_checkpoint_metric not in checkpoint_metric_choices:
        raise ValueError(f"best_checkpoint_metric must be one of {sorted(checkpoint_metric_choices)}")
    _validate_joint_checkpoint_mode(
        target_mode=target_mode,
        best_checkpoint_metric=best_checkpoint_metric,
        checkpoint_metrics=checkpoint_metrics,
    )
    if len(scenario_metric_weights) != 3:
        raise ValueError("scenario_metric_weights must contain exactly 3 values [w_hc, w_hf, w_pte]")
    if any(w < 0 for w in scenario_metric_weights):
        raise ValueError("scenario_metric_weights must be non-negative")
    if sum(scenario_metric_weights) <= 0:
        raise ValueError("scenario_metric_weights sum must be > 0")
    if scenario_threshold_percentile <= 0 or scenario_threshold_percentile >= 100:
        raise ValueError("scenario_threshold_percentile must be in (0, 100)")

    if scheduler_type == "annealing":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6,
        )
    elif scheduler_type == "plateau": 
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode="min",   # monitor the minimum validation loss
            factor=0.5,   # reduce learning rate by 0.5
            patience=10,  # stop training if no improvement for 10 epochs
        )
    elif scheduler_type == "linear":
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=5
        )
    elif scheduler_type == "polynomial":
        scheduler = torch.optim.lr_scheduler.PolynomialLR(
            optimizer, total_iters=epochs, power=1.0
        )
    else:
        raise ValueError(f"scheduler {scheduler_type} is not supported")
    
    # model weights dir
    if run_tag is not None and "small_test" in MODEL_WEIGHTS_DIR:
        model_dir = os.path.join(MODEL_WEIGHTS_DIR, segment, f"{model_name}_{run_tag}", target_mode)
    else:
        model_dir = os.path.join(MODEL_WEIGHTS_DIR, segment, model_name, target_mode)
    os.makedirs(model_dir, exist_ok=True)
    weight_tag = f"aw-{weights[0]}_vw-{weights[1]}"
    checkpoint_specs = _checkpoint_metric_specs(model_dir, weight_tag)
    checkpoint_files = {name: spec[0] for name, spec in checkpoint_specs.items()}
    best_huber_file = checkpoint_files["huber"]
    selected_checkpoint_metrics = set(checkpoint_files if checkpoint_metrics is None else checkpoint_metrics)
    if target_mode != "multi" and checkpoint_metrics is None:
        selected_checkpoint_metrics.discard("f1-joint")
    unknown_checkpoint_metrics = selected_checkpoint_metrics - set(checkpoint_files)
    if unknown_checkpoint_metrics:
        raise ValueError(
            f"Unknown checkpoint metrics: {sorted(unknown_checkpoint_metrics)}"
        )
    if best_checkpoint_metric not in selected_checkpoint_metrics:
        raise ValueError("best_checkpoint_metric must be included in checkpoint_metrics")
    best_model_file = checkpoint_files[best_checkpoint_metric]
    best_metric_values = {name: spec[2] for name, spec in checkpoint_specs.items()}
    LOG.info(
        f"checkpoint metric={best_checkpoint_metric}, "
        f"scenario_tau_c={tau_c}, scenario_tau_f={tau_f}, "
        f"scenario_weights={scenario_metric_weights}, "
        f"target_scales={target_scales}, event_pos_weights={event_pos_weights}, "
        f"joint_event_weight={scenario_loss_joint_weight}, "
        f"joint_event_pos_weight={joint_event_pos_weight}, "
        f"residual_baseline={residual_baseline}, residual_loss_weight={residual_loss_weight}, "
        f"multi_task_architecture={model.multi_task_architecture}"
    )

    model.train()
    loss: torch.Tensor
    for epoch in range(epochs):
        criterion.set_epoch(epoch + 1)
        if stage_task_weights_end is not None:
            start_tensor = torch.tensor(fixed_task_weights, dtype=torch.float32)
            end_tensor = torch.tensor(stage_task_weights_end, dtype=torch.float32)
            if epoch + 1 <= stage_start_epoch:
                alpha = 0.0
            elif epoch + 1 >= stage_end_epoch:
                alpha = 1.0
            else:
                alpha = (epoch + 1 - stage_start_epoch) / (stage_end_epoch - stage_start_epoch)
            current_weights = (1 - alpha) * start_tensor + alpha * end_tensor
            criterion.set_fixed_weights(current_weights)
            print(f"(Process {p_id}) Epoch {epoch + 1}/{epochs}, staged task weights: {criterion.fixed_weights.detach().cpu().numpy().tolist()}")

        start_time = time.time_ns()
        batch_loss = 0.0
        epoch_loss_info: dict[str, float] = {}
        epoch_grad_norm = 0.0
        for batch in train_loader:
            xb, yb, baseline = _move_batch_to_device(batch, device)
            curb, upb, downb, inb, outb = xb
            optimizer.zero_grad()
            raw_output = model(curb, upb, downb, inb, outb)
            output = _apply_residual_baseline(raw_output, baseline)
            loss, loss_info = criterion(output, yb)
            residual_loss = _normalized_residual_loss(raw_output, yb, baseline, target_scales)
            if baseline is not None and residual_loss_weight > 0:
                loss = loss + residual_loss_weight * residual_loss
                loss_info["residual_loss"] = residual_loss.detach().cpu().item()
                loss_info["residual_loss_weight"] = residual_loss_weight
                loss_info["residual_loss_tensor"] = residual_loss
                if "pcgrad_objectives" in loss_info:
                    loss_info["pcgrad_objectives"] = list(loss_info["pcgrad_objectives"]) + [residual_loss_weight * residual_loss]
            if use_pcgrad:
                criterion_param_grads = collect_module_grads(loss, criterion, retain_graph=True)
                task_losses = make_weighted_task_losses(loss_info)
                if pcgrad_include_mape and "mape_loss_tensor" in loss_info:
                    task_losses.append(loss_info["mape_lambda"] * loss_info["mape_loss_tensor"])
                pcgrad_backward(task_losses, model)
                assign_module_grads(criterion_param_grads)
            else:
                loss.backward()
            if grad_clip_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            batch_grad_norm = _grad_norm(model.parameters())
            optimizer.step()
            batch_size = yb.size(0)
            batch_loss += loss.item() * yb.size(0)
            epoch_grad_norm += batch_grad_norm * batch_size
            for key, value in _loss_info_scalars(loss_info).items():
                epoch_loss_info[key] = epoch_loss_info.get(key, 0.0) + value * batch_size
        train_loss[epoch + 1] = batch_loss / len(train_loader.dataset)
        loss_breakdown_row: dict[str, float] = {
            "epoch": float(epoch + 1),
            "train_loss": train_loss[epoch + 1],
            "lr": float(optimizer.param_groups[0]["lr"]),
            "grad_norm": epoch_grad_norm / len(train_loader.dataset),
        }
        for key, value in epoch_loss_info.items():
            loss_breakdown_row[key] = value / len(train_loader.dataset)
        loss_breakdown_history.append(loss_breakdown_row)
        if scheduler_type in ["annealing", "linear", "polynomial"]:
            scheduler.step()
        end_time = time.time_ns()
        train_time += (end_time - start_time) / 1e9
        print(f"(Process {p_id}) Epoch {epoch + 1}/{epochs}, Loss: {train_loss[epoch + 1]}, Time: {(end_time - start_time) / 1e9}s")

        # evaluate
        if (epoch + 1) % 10 == 0:
            valid_metrics, valid_task_metrics, valid_true_batches, valid_pred_batches = _evaluate_dsfnet(
                model, valid_loader, device
            )
            v_r2_loss = valid_metrics["R2"]
            v_mse_loss = valid_metrics["MSE"]
            v_rmse_loss = valid_metrics["RMSE"]
            v_mape_loss = valid_metrics["MAPE"]
            v_smape_loss = valid_metrics["SMAPE"]
            v_huber_loss = valid_metrics["Huber"]
            v_r2_task = valid_task_metrics["R2"]
            v_mse_task = valid_task_metrics["MSE"]
            v_rmse_task = valid_task_metrics["RMSE"]
            v_mape_task = valid_task_metrics["MAPE"]
            v_smape_task = valid_task_metrics["SMAPE"]
            v_huber_task = valid_task_metrics["Huber"]
            v_balanced_rmse = float(np.mean([
                v_rmse_task[task_id] / max(float(target_scales[task_id]), EPS)
                for task_id in range(task_num)
            ]))

            valid_r2_loss[epoch + 1] = v_r2_loss
            valid_mse_loss[epoch + 1] = v_mse_loss
            valid_rmse_loss[epoch + 1] = v_rmse_loss
            valid_mape_loss[epoch + 1] = v_mape_loss
            valid_smape_loss[epoch + 1] = v_smape_loss
            valid_huber_loss[epoch + 1] = v_huber_loss
            valid_scenario = _scenario_metrics(
                y_true=_flatten_batches(valid_true_batches),
                y_pred=_flatten_batches(valid_pred_batches),
                target_mode=target_mode,
                tau_c=tau_c,
                tau_f=tau_f,
                metric_weights=scenario_metric_weights,
                prediction_dim=prediction_dim,
            )
            valid_diagnostics = _selection_diagnostics(
                _flatten_batches(valid_true_batches),
                _flatten_batches(valid_pred_batches),
                tau_c=tau_c,
                target_mode=target_mode,
            )
            for task_id in range(task_num):
                valid_r2_task_loss[task_id][epoch + 1] = v_r2_task[task_id]
                valid_mse_task_loss[task_id][epoch + 1] = v_mse_task[task_id]
                valid_rmse_task_loss[task_id][epoch + 1] = v_rmse_task[task_id]
                valid_mape_task_loss[task_id][epoch + 1] = v_mape_task[task_id]
                valid_smape_task_loss[task_id][epoch + 1] = v_smape_task[task_id]
                valid_huber_task_loss[task_id][epoch + 1] = v_huber_task[task_id]

            _print_metrics_block(
                f"(Process {p_id}, eval) Epoch {epoch + 1}/{epochs}",
                valid_metrics,
                valid_task_metrics,
                task_num,
            )
            print(
                f"       ScenarioScore={_fmt_float(valid_scenario['scenario_score'])}, "
                f"F1_hc={_fmt_float(valid_scenario['F1_hc'])}, "
                f"PTE={_fmt_float(valid_scenario['PTE'])}, "
                f"F1_hf={_fmt_float(valid_scenario['F1_hf'])}, "
                f"Precision_joint={_fmt_float(valid_scenario['Precision_joint'])}, "
                f"Recall_joint={_fmt_float(valid_scenario['Recall_joint'])}, "
                f"F1_joint={_fmt_float(valid_scenario['F1_joint'])}"
            )
            if evaluate_test:
                test_metrics_epoch, test_task_metrics_epoch, _, _ = _evaluate_dsfnet(
                    model, test_loader, device
                )
                _print_metrics_block(
                    f"(Process {p_id}, test) Epoch {epoch + 1}/{epochs}",
                    test_metrics_epoch,
                    test_task_metrics_epoch,
                    task_num,
                )
            model.train()

            if scheduler_type == "plateau":
                # schedule learning rate
                scheduler.step(v_huber_loss)

            current_scenario_score = float(valid_scenario["scenario_score"])
            current_f1_hc = float(valid_scenario["F1_hc"])
            current_f1_hf = float(valid_scenario["F1_hf"])
            current_f1_joint = float(valid_scenario["F1_joint"])
            current_pte = float(valid_scenario["PTE"])
            checkpoint_common = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_huber": v_huber_loss,
                "val_rmse": v_rmse_loss,
                "val_balanced_rmse": v_balanced_rmse,
                "val_f1_hc": current_f1_hc,
                "val_f1_hf": current_f1_hf,
                "val_precision_joint": float(valid_scenario["Precision_joint"]),
                "val_recall_joint": float(valid_scenario["Recall_joint"]),
                "val_f1_joint": current_f1_joint,
                "val_pte": current_pte,
                "scenario_score": current_scenario_score,
                "scenario_weights": scenario_metric_weights,
                "scenario_threshold_percentile": scenario_threshold_percentile,
                "scenario_tau_c": tau_c,
                "scenario_tau_f": tau_f,
                "val_crowding_mae": valid_diagnostics["crowding_mae"],
                "val_crowding_correlation": valid_diagnostics["crowding_correlation"],
                "val_flow_mae": valid_diagnostics["flow_mae"],
                "val_flow_correlation": valid_diagnostics["flow_correlation"],
                "val_pta_at_3": valid_diagnostics["PTA@3"],
                "val_pta_at_6": valid_diagnostics["PTA@6"],
                "target_scales": target_scales,
                "event_pos_weights": event_pos_weights,
                "joint_event_pos_weight": joint_event_pos_weight,
                "scenario_loss_joint_weight": scenario_loss_joint_weight,
                "criterion": criterion_type,
                "residual_baseline": residual_baseline,
                "residual_loss_weight": residual_loss_weight,
                "multi_task_architecture": model.multi_task_architecture,
                "run_tag": run_tag or "",
                "split_mode": split_mode,
                "selection_scope": "validation-only",
                "fusion_net": fusion_net,
                "direction_mode": getattr(model.fusion, "direction_mode", "not-applicable"),
                "num_parameters": sum(p.numel() for p in model.parameters()),
                "num_trainable_parameters": sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                ),
                "experiment_meta": dict(experiment_meta or {}),
            }

            def save_metric_checkpoint(
                metric_name: str,
                metric_value: float,
                mode: str,
                path: str,
                tie_breaker_value: float | None = None,
            ) -> None:
                nonlocal best_valid_scenario_tiebreak_loss
                if metric_name not in selected_checkpoint_metrics:
                    return
                current_best = best_metric_values[metric_name]
                current_tie_breaker = (
                    best_valid_scenario_tiebreak_loss
                    if metric_name == "scenario-priority"
                    else None
                )
                improved = _metric_is_better(
                    value=metric_value,
                    best=current_best,
                    mode=mode,
                    tie_breaker=tie_breaker_value,
                    best_tie_breaker=current_tie_breaker,
                )
                if not improved:
                    return
                best_metric_values[metric_name] = metric_value
                if metric_name == "scenario-priority" and tie_breaker_value is not None:
                    best_valid_scenario_tiebreak_loss = tie_breaker_value
                if not SAVE_MODEL_WEIGHTS:
                    return
                os.makedirs(os.path.dirname(path), exist_ok=True)
                torch.save({
                    **checkpoint_common,
                    "best_metric_value": metric_value,
                    "best_metric_tie_breaker": tie_breaker_value,
                    "checkpoint_metric": metric_name,
                }, path)
                LOG.info(f"best {metric_name} model has been saved to {path}")

            metric_values = {
                "huber": v_huber_loss,
                "rmse": v_rmse_loss,
                "balanced-rmse": v_balanced_rmse,
                "f1-hc": current_f1_hc,
                "f1-hf": current_f1_hf,
                "f1-joint": current_f1_joint,
                "pte": current_pte,
            }
            for metric_name, metric_value in metric_values.items():
                path, mode, _ = checkpoint_specs[metric_name]
                save_metric_checkpoint(metric_name, metric_value, mode, path)
            save_metric_checkpoint(
                "scenario-priority",
                current_scenario_score,
                checkpoint_specs["scenario-priority"][1],
                checkpoint_specs["scenario-priority"][0],
                tie_breaker_value=v_balanced_rmse,
            )

            best_valid_loss = best_metric_values["huber"]
            best_valid_scenario_score = best_metric_values["scenario-priority"]

    if not evaluate_test:
        torch.cuda.empty_cache()
        del model
        del optimizer
        del criterion
        last_valid_epoch = max(valid_r2_loss.keys()) if valid_r2_loss else None
        return {
            "Epochs": epochs,
            "Train Loss (MSE)": train_loss[epochs],
            "Valid Loss (R2)": valid_r2_loss[last_valid_epoch] if last_valid_epoch is not None else None,
            "Valid Loss (MAPE)": valid_mape_loss[last_valid_epoch] if last_valid_epoch is not None else None,
            "Valid Loss (SMAPE)": valid_smape_loss[last_valid_epoch] if last_valid_epoch is not None else None,
            "Valid Loss (Huber)": valid_huber_loss[last_valid_epoch] if last_valid_epoch is not None else None,
            "Best Checkpoint Metric": best_checkpoint_metric,
            "Best Scenario Score": best_valid_scenario_score if best_valid_scenario_score > float("-inf") else None,
            "Train Time(s)": train_time,
        }

    # final test evaluation
    model_file = best_model_file
    selected_checkpoint_path: str | None = None
    selected_checkpoint_epoch: int | None = None
    if not os.path.exists(model_file) and best_checkpoint_metric != "huber":
        if os.path.exists(best_huber_file):
            model_file = best_huber_file
    if SAVE_MODEL_WEIGHTS and os.path.exists(model_file):
        checkpoint = torch.load(model_file)
        model.load_state_dict(checkpoint["model_state_dict"])
        selected_checkpoint_path = os.path.abspath(model_file)
        selected_checkpoint_epoch = int(checkpoint.get("epoch", 0))
    elif SAVE_MODEL_WEIGHTS:
        warnings.warn(f"Best model not found, use current in-memory model for test: {model_file}")
    test_metrics, test_task_metrics, test_true_batches, test_pred_batches = _evaluate_dsfnet(
        model, test_loader, device
    )
    t_r2_loss = test_metrics["R2"]
    t_mse_loss = test_metrics["MSE"]
    t_rmse_loss = test_metrics["RMSE"]
    t_mape_loss = test_metrics["MAPE"]
    t_smape_loss = test_metrics["SMAPE"]
    t_huber_loss = test_metrics["Huber"]
    t_r2_task = test_task_metrics["R2"]
    t_mse_task = test_task_metrics["MSE"]
    t_rmse_task = test_task_metrics["RMSE"]
    t_mape_task = test_task_metrics["MAPE"]
    t_smape_task = test_task_metrics["SMAPE"]
    t_huber_task = test_task_metrics["Huber"]
    _print_metrics_block(f"(Process {p_id}, test)", test_metrics, test_task_metrics, task_num)

    charts_dir = f"{model_dir}/charts"
    if dump_test_preds:
        os.makedirs(charts_dir, exist_ok=True)
        pred_path = save_test_predictions(
            charts_dir=charts_dir,
            segment=segment,
            target_mode=target_mode,
            model_name=model_name,
            aw=weights[0],
            vw=weights[1],
            pred_len=prediction_dim,
            test_loader=test_loader,
            y_true_batches=test_true_batches,
            y_pred_batches=test_pred_batches,
            run_tag=run_tag,
            split_mode=split_mode,
            extra_meta={
                "criterion": criterion_type,
                "loss_weighting": loss_weighting,
                "input_ablation": input_ablation,
                "residual_baseline": residual_baseline,
                "residual_loss_weight": residual_loss_weight,
                "best_checkpoint_metric": best_checkpoint_metric,
                "scenario_threshold_percentile": scenario_threshold_percentile,
                "selection_scope": "validation-only",
                "selected_checkpoint_path": selected_checkpoint_path or "",
                "selected_checkpoint_epoch": selected_checkpoint_epoch if selected_checkpoint_epoch is not None else -1,
                "scenario_tau_c": tau_c,
                "scenario_tau_f": tau_f,
                "target_scales": target_scales,
                "event_pos_weights": event_pos_weights,
                "joint_event_pos_weight": joint_event_pos_weight,
                "scenario_loss_joint_weight": scenario_loss_joint_weight,
                "mape_lambda": mape_lambda,
                "normalize_task_losses": normalize_task_losses,
                "multi_task_architecture": model.multi_task_architecture,
                "num_parameters": sum(p.numel() for p in model.parameters()),
                "num_trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                **(experiment_meta or {}),
            },
        )
        LOG.info(f"test predictions have been saved to {pred_path}")

    # save charts
    if save_charts:
        os.makedirs(charts_dir, exist_ok=True)
        breakdown_path = os.path.join(charts_dir, f"loss_breakdown_{weight_tag}.csv")
        _write_loss_breakdown_csv(breakdown_path, loss_breakdown_history)
        LOG.info(f"loss breakdown has been saved to {breakdown_path}")
        valid_loss_dump = {
            "R2":       valid_r2_loss,
            "MSE":      valid_mse_loss,
            "RMSE":     valid_rmse_loss,
            "MAPE":     valid_mape_loss,
            "SMAPE":    valid_smape_loss,
            "Huber":    valid_huber_loss,
        }
        if task_num > 0:
            valid_loss_dump["Mean-Task-R2"] = {
                epoch: sum(valid_r2_task_loss[i][epoch] for i in range(task_num)) / task_num
                for epoch in valid_r2_loss
            }
            valid_loss_dump["Mean-Task-MSE"] = {
                epoch: sum(valid_mse_task_loss[i][epoch] for i in range(task_num)) / task_num
                for epoch in valid_mse_loss
            }
            valid_loss_dump["Mean-Task-RMSE"] = {
                epoch: sum(valid_rmse_task_loss[i][epoch] for i in range(task_num)) / task_num
                for epoch in valid_rmse_loss
            }
            valid_loss_dump["Mean-Task-MAPE"] = {
                epoch: sum(valid_mape_task_loss[i][epoch] for i in range(task_num)) / task_num
                for epoch in valid_mape_loss
            }
            valid_loss_dump["Mean-Task-SMAPE"] = {
                epoch: sum(valid_smape_task_loss[i][epoch] for i in range(task_num)) / task_num
                for epoch in valid_smape_loss
            }
            valid_loss_dump["Mean-Task-Huber"] = {
                epoch: sum(valid_huber_task_loss[i][epoch] for i in range(task_num)) / task_num
                for epoch in valid_huber_loss
            }
            for task_id in range(task_num):
                task_idx = task_id + 1
                valid_loss_dump[f"Task-{task_idx}-R2"] = valid_r2_task_loss[task_id]
                valid_loss_dump[f"Task-{task_idx}-MSE"] = valid_mse_task_loss[task_id]
                valid_loss_dump[f"Task-{task_idx}-RMSE"] = valid_rmse_task_loss[task_id]
                valid_loss_dump[f"Task-{task_idx}-MAPE"] = valid_mape_task_loss[task_id]
                valid_loss_dump[f"Task-{task_idx}-SMAPE"] = valid_smape_task_loss[task_id]
                valid_loss_dump[f"Task-{task_idx}-Huber"] = valid_huber_task_loss[task_id]

        test_loss_dump = {
            "R2":       t_r2_loss,
            "MSE":      t_mse_loss,
            "RMSE":     t_rmse_loss,
            "MAPE":     t_mape_loss,
            "SMAPE":    t_smape_loss,
            "Huber":    t_huber_loss,
        }
        if task_num > 0:
            test_loss_dump["Mean-Task-R2"] = sum(t_r2_task) / task_num
            test_loss_dump["Mean-Task-MSE"] = sum(t_mse_task) / task_num
            test_loss_dump["Mean-Task-RMSE"] = sum(t_rmse_task) / task_num
            test_loss_dump["Mean-Task-MAPE"] = sum(t_mape_task) / task_num
            test_loss_dump["Mean-Task-SMAPE"] = sum(t_smape_task) / task_num
            test_loss_dump["Mean-Task-Huber"] = sum(t_huber_task) / task_num
            for task_id in range(task_num):
                task_idx = task_id + 1
                test_loss_dump[f"Task-{task_idx}-R2"] = t_r2_task[task_id]
                test_loss_dump[f"Task-{task_idx}-MSE"] = t_mse_task[task_id]
                test_loss_dump[f"Task-{task_idx}-RMSE"] = t_rmse_task[task_id]
                test_loss_dump[f"Task-{task_idx}-MAPE"] = t_mape_task[task_id]
                test_loss_dump[f"Task-{task_idx}-SMAPE"] = t_smape_task[task_id]
                test_loss_dump[f"Task-{task_idx}-Huber"] = t_huber_task[task_id]

        save_training_process(
            dir=charts_dir,
            segment=segment,
            epochs=epochs,
            crowding_weights=weight_tag,
            model_name=model_name,
            optimizer=optimizer_type,
            scheduler=scheduler_type,
            criterion=criterion_type,
            loss_weighting=loss_weighting,
            task_weights=fixed_task_weights,
            train_time=train_time,
            train_loss=train_loss,
            valid_loss=valid_loss_dump,
            test_loss=test_loss_dump,
        )

    # clear cache
    torch.cuda.empty_cache()
    del model
    del optimizer
    del criterion

    last_valid_epoch = max(valid_r2_loss.keys()) if len(valid_r2_loss) > 0 else None

    return {
        "Epochs":               epochs,
        "Train Loss (MSE)":     train_loss[epochs],
        "Valid Loss (R2)":      valid_r2_loss[last_valid_epoch] if last_valid_epoch is not None else None,
        "Valid Loss (MAPE)":    valid_mape_loss[last_valid_epoch] if last_valid_epoch is not None else None,
        "Valid Loss (SMAPE)":   valid_smape_loss[last_valid_epoch] if last_valid_epoch is not None else None,
        "Valid Loss (Huber)":   valid_huber_loss[last_valid_epoch] if last_valid_epoch is not None else None,
        "Best Checkpoint Metric": best_checkpoint_metric,
        "Best Scenario Score":  best_valid_scenario_score if best_valid_scenario_score > float("-inf") else None,
        "Train Time(s)":        train_time,
    }
