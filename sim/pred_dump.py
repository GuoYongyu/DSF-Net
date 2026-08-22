# coding: utf-8
from __future__ import annotations

import os
from datetime import datetime
from typing import Any

import numpy as np
from torch.utils.data import DataLoader, Subset

from publication_protocol import (
    PREDICTION_HORIZON,
    SCENARIO_THRESHOLD_PERCENTILE,
    TARGET_MODE,
    PublicationProtocolError,
    expected_test_indices,
    publication_metadata_for_segment,
)


def _to_numpy_batches(batches: list[Any]) -> np.ndarray:
    if len(batches) == 0:
        return np.empty((0,), dtype=np.float32)

    np_batches: list[np.ndarray] = []
    for item in batches:
        if hasattr(item, "detach"):
            arr = item.detach().cpu().numpy()
        else:
            arr = np.asarray(item)
        np_batches.append(arr)
    return np.concatenate(np_batches, axis=0)


def _to_scalar(value: Any) -> Any:
    array = np.asarray(value)
    return array.item() if array.shape == () else value


def get_test_indices(test_loader: DataLoader) -> np.ndarray:
    dataset = test_loader.dataset
    if isinstance(dataset, Subset):
        return np.asarray(dataset.indices, dtype=np.int64)
    return np.arange(len(dataset), dtype=np.int64)


def save_test_predictions(
        charts_dir: str,
        segment: str,
        target_mode: str,
        model_name: str,
        aw: float,
        vw: float,
        pred_len: int,
        test_loader: DataLoader,
        y_true_batches: list[Any],
        y_pred_batches: list[Any],
        run_tag: str | None = None,
        split_mode: str | None = None,
        extra_meta: dict[str, Any] | None = None,
    ) -> str:
    preds_dir = os.path.join(charts_dir, "test_preds")
    os.makedirs(preds_dir, exist_ok=True)

    y_true = _to_numpy_batches(y_true_batches).astype(np.float32)
    y_pred = _to_numpy_batches(y_pred_batches).astype(np.float32)
    test_indices = get_test_indices(test_loader)

    publication_meta: dict[str, Any] = {}
    if split_mode == "final":
        if target_mode != TARGET_MODE:
            raise PublicationProtocolError(
                f"Final publication artifacts require target_mode={TARGET_MODE!r}."
            )
        if int(pred_len) != PREDICTION_HORIZON:
            raise PublicationProtocolError(
                f"Final publication artifacts require pred_len={PREDICTION_HORIZON}."
            )
        if not np.array_equal(test_indices, expected_test_indices()):
            raise PublicationProtocolError(
                "Final publication artifacts require exactly the 897 locked "
                "second-period test origins."
            )
        stored_percentile = (
            None
            if extra_meta is None
            else extra_meta.get("scenario_threshold_percentile")
        )
        try:
            percentile_matches = np.isclose(
                float(stored_percentile),
                SCENARIO_THRESHOLD_PERCENTILE,
                atol=1e-8,
                rtol=0.0,
            )
        except (TypeError, ValueError):
            percentile_matches = False
        if not percentile_matches:
            raise PublicationProtocolError(
                "Final publication artifacts must prove that scenario thresholds "
                "were computed at the training-split P70."
            )
        publication_meta = publication_metadata_for_segment(segment)

    now_str = datetime.now().strftime("%Y%m%d%H%M%S")
    safe_tag = (run_tag or "default").replace(" ", "_")
    file_name = f"{now_str}_{safe_tag}_aw-{aw}_vw-{vw}_test_preds.npz"
    out_path = os.path.join(preds_dir, file_name)

    payload: dict[str, Any] = {
        "y_true": y_true,
        "y_pred": y_pred,
        "test_indices": test_indices,
        "segment": np.array(segment),
        "target_mode": np.array(target_mode),
        "model_name": np.array(model_name),
        "aw": np.array(aw, dtype=np.float32),
        "vw": np.array(vw, dtype=np.float32),
        "pred_len": np.array(pred_len, dtype=np.int64),
        "run_tag": np.array(run_tag or ""),
    }
    if split_mode is not None:
        payload["split_mode"] = np.array(split_mode)
    if extra_meta is not None:
        for key, value in extra_meta.items():
            if key in publication_meta and _to_scalar(value) != publication_meta[key]:
                raise PublicationProtocolError(
                    f"extra_meta cannot override reserved publication field {key!r}."
                )
            payload[key] = np.array(value)
    for key, value in publication_meta.items():
        payload[key] = np.array(value)

    np.savez_compressed(out_path, **payload)
    return out_path
