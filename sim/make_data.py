# coding: utf-8
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from publication_protocol import assert_publication_matrix_segment
from sim.config import *
from logger.logger import LOG


EPS = 1e-12
TARGET_MODES = ("multi", "crowding", "equiv_flow")


class TimeSpaceDataset(Dataset):
    def __init__(
            self,
            cur_data: np.ndarray | torch.Tensor,
            up_data: np.ndarray | torch.Tensor,
            down_data: np.ndarray | torch.Tensor,
            in_data: np.ndarray | torch.Tensor,
            out_data: np.ndarray | torch.Tensor,
            y: np.ndarray | torch.Tensor,
            baseline: np.ndarray | torch.Tensor | None = None,
            context: np.ndarray | torch.Tensor | None = None,
        ):
        self.num_samples = cur_data.shape[0]
        inputs_list = [cur_data, up_data, down_data, in_data, out_data]
        for i, inp in enumerate(inputs_list):
            if inp.shape[0] != self.num_samples:
                raise ValueError(f"Input {i} has different shape {inp.shape} than expected {self.num_samples}.")
        if y.shape[0] != self.num_samples:
            raise ValueError(f"Output has different shape {y.shape} than expected {self.num_samples}.")
        if baseline is not None and baseline.shape[0] != self.num_samples:
            raise ValueError(f"Baseline has different shape {baseline.shape} than expected {self.num_samples}.")
        if context is not None and context.shape[0] != self.num_samples:
            raise ValueError(f"Context has different shape {context.shape} than expected {self.num_samples}.")

        self.X = [torch.as_tensor(inp, dtype=torch.float32) for inp in inputs_list]
        if context is not None:
            self.X.append(torch.as_tensor(context, dtype=torch.float32))
        self.y = torch.as_tensor(y, dtype=torch.float32)
        self.baseline = torch.as_tensor(baseline, dtype=torch.float32) if baseline is not None else None

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index: int) -> tuple:
        sample_inputs = tuple(inp[index] for inp in self.X)
        sample_output = self.y[index]
        if self.baseline is not None:
            return sample_inputs, sample_output, self.baseline[index]
        return sample_inputs, sample_output


def _split_indices(
        total_samples: int,
        train_ratio: float = TRAIN_SCALE_RATIO,
        test_ratio: float = TEST_SCALE_RATIO,
        random_state: int = 42,
        split_mode: str = "random",
        purge_steps: int = 0,
        valid_ratio: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert split_mode in ["random", "temporal", "final"], "split_mode must be random, temporal, or final"
    if purge_steps < 0:
        raise ValueError("purge_steps must be >= 0")
    if split_mode == "final":
        # Locked protocol used for the paper: the first period is split in
        # chronological order (1,955 train, 192 embargo, 380 validation),
        # followed by the complete second period (897 test origins).
        if purge_steps != 0:
            raise ValueError("purge_steps must be zero for split_mode=final")
        if total_samples != 3424:
            raise ValueError(
                "split_mode=final expects 3424 forecast origins "
                "(1955 train + 192 embargo + 380 validation + 897 test)"
            )
        return (
            np.arange(0, 1955, dtype=np.int64),
            np.arange(2147, 2527, dtype=np.int64),
            np.arange(2527, 3424, dtype=np.int64),
        )
    if split_mode != "temporal" and purge_steps > 0:
        raise ValueError("purge_steps is only supported with split_mode=temporal")

    if valid_ratio is not None:
        ratios = (train_ratio, valid_ratio, test_ratio)
        if any(ratio <= 0.0 or ratio >= 1.0 for ratio in ratios):
            raise ValueError("train/valid/test ratios must each be in (0, 1)")
        if not np.isclose(sum(ratios), 1.0):
            raise ValueError("train_ratio + valid_ratio + test_ratio must equal 1")
        # With an explicit validation ratio, ratios describe the complete
        # dataset (the paper's 7:1:2 split), not the legacy pre-test subset.
        train_len = int(total_samples * train_ratio)
    else:
        train_len = None

    indices = np.arange(total_samples, dtype=np.int64)
    test_len = int(total_samples * test_ratio)
    test_start = total_samples - test_len if test_len > 0 else total_samples
    train_valid_indices = indices[:test_start]

    if split_mode == "random":
        train_valid_indices = np.random.RandomState(random_state).permutation(train_valid_indices)
        resolved_train_len = (
            train_len
            if train_len is not None
            else int(len(train_valid_indices) * train_ratio)
        )
        train_indices = np.asarray(train_valid_indices[:resolved_train_len], dtype=np.int64)
        valid_indices = np.asarray(train_valid_indices[resolved_train_len:], dtype=np.int64)
        test_indices = indices[test_start:]
        return train_indices, valid_indices, test_indices

    train_boundary = (
        train_len if train_len is not None else int(len(train_valid_indices) * train_ratio)
    )
    train_stop = max(0, train_boundary - purge_steps)
    valid_stop = max(train_boundary, test_start - purge_steps)
    train_indices = indices[:train_stop]
    valid_indices = indices[train_boundary:valid_stop]
    test_indices = indices[test_start:]
    if train_indices.size == 0 or valid_indices.size == 0 or test_indices.size == 0:
        raise ValueError(
            "temporal split produced an empty partition; reduce purge_steps or adjust split ratios"
        )
    return train_indices, valid_indices, test_indices


def free_flow_speed_from_train(y_components: np.ndarray, train_indices: np.ndarray) -> float:
    if y_components.ndim != 3 or y_components.shape[-1] < 1:
        return 0.0
    if train_indices.size == 0:
        return 0.0
    speed = y_components[train_indices, :, 0].reshape(-1)
    speed = speed[np.isfinite(speed) & (speed > 0)]
    if speed.size == 0:
        return 0.0
    return float(np.percentile(speed, 80))


def free_domain_occupancy_from_train(
        y_components: np.ndarray,
        train_indices: np.ndarray,
) -> float:
    """Training-only 80th percentile used to normalize raw domain occupancy."""
    if y_components.ndim != 3 or y_components.shape[-1] < 2 or train_indices.size == 0:
        return 1.0
    occupancy = y_components[train_indices, :, 1].reshape(-1)
    occupancy = occupancy[np.isfinite(occupancy) & (occupancy > 0)]
    if occupancy.size == 0:
        return 1.0
    return max(float(np.percentile(occupancy, 80)), EPS)


def make_prediction_vector_from_components(
        components: np.ndarray,
        weights: tuple[float, float],
        target_mode: str,
        free_flow_speed: float,
        free_domain_occupancy: float | None = None,
    ) -> np.ndarray:
    if target_mode not in TARGET_MODES:
        raise ValueError(f"target_mode must be one of {list(TARGET_MODES)}, but got {target_mode}")
    if components.ndim != 3 or components.shape[-1] < 3:
        raise ValueError(
            "y.npy must have shape (samples, horizon, >=3) with "
            "[space_mean_speed, domain_occupancy, equiv_flow]."
        )

    speed = components[:, :, 0]
    domain_occupancy = components[:, :, 1]
    equiv_flow = components[:, :, 2]
    if free_flow_speed > EPS:
        speed_degradation = np.clip(1.0 - speed / free_flow_speed, 0.0, 1.0)
    else:
        speed_degradation = np.zeros_like(speed, dtype=float)
    if free_domain_occupancy is None or free_domain_occupancy <= EPS:
        normalized_occupancy = np.clip(domain_occupancy, 0.0, 1.0)
    else:
        normalized_occupancy = np.clip(domain_occupancy / free_domain_occupancy, 0.0, 1.0)
    crowding = weights[0] * normalized_occupancy + weights[1] * speed_degradation

    if target_mode == "multi":
        return np.stack((crowding, equiv_flow), axis=2)
    if target_mode == "crowding":
        return crowding[..., np.newaxis]
    return equiv_flow[..., np.newaxis]


def make_prediction_vector(
        data_file: os.PathLike | str,
        weights: tuple[float, float],
        target_mode: str = "multi",
        pred_len: int = 24,
        train_indices: np.ndarray | None = None,
    ) -> np.ndarray:
    assert os.path.exists(data_file), f"Data file {data_file} does not exist."
    data = np.load(data_file)
    if data.shape[1] < pred_len:
        raise ValueError(
            f"Prediction length {pred_len} exceeds y.npy time dimension {data.shape[1]}. "
            "Please regenerate matrices with a larger prediction window (>= requested pred_len)."
        )
    components = data[:, :pred_len, :]
    if train_indices is None:
        train_indices, _, _ = _split_indices(total_samples=components.shape[0], split_mode="random")
    v_free = free_flow_speed_from_train(components, train_indices)
    d_free = free_domain_occupancy_from_train(components, train_indices)
    return make_prediction_vector_from_components(
        components, weights, target_mode, v_free, free_domain_occupancy=d_free
    )


def make_residual_baseline(
        hist_components: np.ndarray,
        weights: tuple[float, float],
        target_mode: str,
        free_flow_speed: float,
        pred_len: int,
        baseline: str,
        free_domain_occupancy: float | None = None,
    ) -> np.ndarray | None:
    if baseline == "none":
        return None
    if baseline not in {"ha", "last"}:
        raise ValueError("residual baseline must be one of ['none', 'ha', 'last']")
    if hist_components.ndim != 3 or hist_components.shape[-1] < 3:
        raise ValueError(
            "hist_y.npy must have shape (samples, history, >=3) with "
            "[space_mean_speed, domain_occupancy, equiv_flow]."
        )
    if hist_components.shape[1] < pred_len:
        raise ValueError("hist_y.npy history length must be >= pred_len")

    hist_targets = make_prediction_vector_from_components(
        hist_components,
        weights=weights,
        target_mode=target_mode,
        free_flow_speed=free_flow_speed,
        free_domain_occupancy=free_domain_occupancy,
    )
    if baseline == "last":
        return np.repeat(hist_targets[:, -1:, :], pred_len, axis=1)

    daily_period = 24
    rows: list[np.ndarray] = []
    for horizon_idx in range(pred_len):
        aligned = hist_targets[:, horizon_idx % daily_period::daily_period, :]
        rows.append(np.mean(aligned, axis=1))
    return np.stack(rows, axis=1)


def make_train_test_data(
        dataset: TimeSpaceDataset,
        train_ratio: float = TRAIN_SCALE_RATIO,
        valid_ratio: float = VALID_SCALE_RATIO,
        test_ratio: float = TEST_SCALE_RATIO,
        random_state: int = 42,
        split_mode: str = "random",
        purge_steps: int = 0,
    ) -> tuple[Subset, Subset, Subset]:
    assert train_ratio + valid_ratio == 1.0, "Train, valid scale ratios must sum up to 1.0."
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    train_indices, valid_indices, test_indices = _split_indices(
        total_samples=len(dataset),
        train_ratio=train_ratio,
        test_ratio=test_ratio,
        random_state=random_state,
        split_mode=split_mode,
        purge_steps=purge_steps,
    )
    return Subset(dataset, train_indices), Subset(dataset, valid_indices), Subset(dataset, test_indices)


def make_loader(
        segment: str,
        weights: tuple[float, float],
        batch_size: int = 32,
        split_mode: str = "random",
        target_mode: str = "multi",
        pred_len: int = 24,
        random_state: int = 42,
        loader_seed: int | None = None,
        purge_steps: int = 0,
        input_ablation: str = "full",
        residual_baseline: str = "none",
        train_ratio: float = TRAIN_SCALE_RATIO,
        valid_ratio: float | None = None,
        test_ratio: float = TEST_SCALE_RATIO,
        context_data: np.ndarray | torch.Tensor | None = None,
    ) -> tuple[DataLoader, DataLoader, DataLoader | None]:
    LOG.info(f"Make loader for segment {segment} with batch size {batch_size}.")
    if split_mode == "final":
        assert_publication_matrix_segment(MATRICES_DIR, segment)
    cur_data = np.load(os.path.join(MATRICES_DIR, segment, "cur.npy"))
    up_data = np.load(os.path.join(MATRICES_DIR, segment, "up.npy"))
    down_data = np.load(os.path.join(MATRICES_DIR, segment, "down.npy"))
    in_data = np.load(os.path.join(MATRICES_DIR, segment, "in.npy"))
    out_data = np.load(os.path.join(MATRICES_DIR, segment, "out.npy"))

    if input_ablation == "no-upstream":
        up_data = np.zeros_like(up_data)
    elif input_ablation == "no-downstream":
        down_data = np.zeros_like(down_data)
    elif input_ablation == "no-tributary":
        in_data = np.zeros_like(in_data)
        out_data = np.zeros_like(out_data)
    elif input_ablation != "full":
        raise ValueError("input_ablation must be one of ['full', 'no-upstream', 'no-downstream', 'no-tributary']")

    train_indices, valid_indices, test_indices = _split_indices(
        total_samples=cur_data.shape[0],
        train_ratio=train_ratio,
        test_ratio=test_ratio,
        random_state=random_state,
        split_mode=split_mode,
        purge_steps=purge_steps,
        valid_ratio=valid_ratio,
    )
    y_components = np.load(os.path.join(MATRICES_DIR, segment, "y.npy"))
    if y_components.shape[1] < pred_len:
        raise ValueError(
            f"Prediction length {pred_len} exceeds y.npy time dimension {y_components.shape[1]}. "
            "Please regenerate matrices with a larger prediction window (>= requested pred_len)."
        )
    y_components = y_components[:, :pred_len, :]
    v_free = free_flow_speed_from_train(y_components, train_indices)
    d_free = free_domain_occupancy_from_train(y_components, train_indices)
    y_data = make_prediction_vector_from_components(
        components=y_components,
        weights=weights,
        target_mode=target_mode,
        free_flow_speed=v_free,
        free_domain_occupancy=d_free,
    )
    baseline_data = None
    if residual_baseline != "none":
        hist_file = os.path.join(MATRICES_DIR, segment, "hist_y.npy")
        if not os.path.exists(hist_file):
            raise ValueError(f"Raw history file {hist_file} not found. Regenerate matrices.")
        baseline_data = make_residual_baseline(
            hist_components=np.load(hist_file),
            weights=weights,
            target_mode=target_mode,
            free_flow_speed=v_free,
            free_domain_occupancy=d_free,
            pred_len=pred_len,
            baseline=residual_baseline,
        )
    full_dataset = TimeSpaceDataset(
        cur_data,
        up_data,
        down_data,
        in_data,
        out_data,
        y_data,
        baseline=baseline_data,
        context=context_data,
    )
    train_dataset = Subset(full_dataset, train_indices)
    valid_dataset = Subset(full_dataset, valid_indices)
    test_dataset = Subset(full_dataset, test_indices)

    num_workers = 0 if sys.platform == "win32" else 4
    train_generator = torch.Generator().manual_seed(
        random_state if loader_seed is None else loader_seed
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=train_generator,
    )
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, valid_loader, test_loader
