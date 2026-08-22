# coding: utf-8
"""Single source of truth for the publication overall-comparison protocol."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from publication_protocol import (
    EMBARGO_ORIGIN_COUNT,
    EVALUATION_SEGMENTS,
    FINAL_EXPERIMENT_SPEC,
    FINAL_EMBARGO_RATIO,
    FINAL_TEST_RATIO,
    FINAL_TRAIN_RATIO,
    FINAL_VALID_RATIO,
    PROTOCOL_ID,
    TEST_ORIGIN_COUNT,
    TRAIN_ORIGIN_COUNT,
    VALID_ORIGIN_COUNT,
)


METRIC_COLUMNS = (
    "NRMSE",
    "NMAE",
    "MeanPearsonR",
    "F1hc",
    "F1hf",
    "JWP@24h",
)
LOWER_IS_BETTER = frozenset({"NRMSE", "NMAE"})
DEFAULT_SINGLE_SEEDS = tuple(FINAL_EXPERIMENT_SPEC["main_seeds"])
DEFAULT_MULTI_SEEDS = tuple(FINAL_EXPERIMENT_SPEC["stability_seeds"])
DSF_DEFAULT_MULTI_TASK_ARCHITECTURE = "cross-stitch-2"
DSF_MULTI_TASK_ARCHITECTURES = (
    "shared",
    "spatial-private",
    "shared-private",
    "separate-towers",
    "separate-towers-1",
    "separate-towers-2",
    "cross-stitch",
    "cross-stitch-1",
    "cross-stitch-2",
    "mmoe",
    "mmoe-2",
    "mmoe-4",
    "ple",
    "ple-1",
    "ple-2",
)
DSF_BASE_MODEL_DIR = "tn-timesnet_sn-spatial_conv_fn-dcp_fusion_rb-ha"


def dsf_model_dir(multi_task_architecture: str) -> str:
    if multi_task_architecture not in DSF_MULTI_TASK_ARCHITECTURES:
        raise ValueError(
            f"Unknown DSF multi-task architecture: {multi_task_architecture!r}"
        )
    if multi_task_architecture == "shared":
        return DSF_BASE_MODEL_DIR
    return f"{DSF_BASE_MODEL_DIR}_{multi_task_architecture.replace('-', '_')}"


@dataclass(frozen=True)
class ExperimentProtocol:
    split_mode: str = "final"
    train_ratio: float = FINAL_TRAIN_RATIO
    valid_ratio: float = FINAL_VALID_RATIO
    test_ratio: float = FINAL_TEST_RATIO
    embargo_ratio: float = FINAL_EMBARGO_RATIO
    train_origin_count: int = TRAIN_ORIGIN_COUNT
    embargo_origin_count: int = EMBARGO_ORIGIN_COUNT
    valid_origin_count: int = VALID_ORIGIN_COUNT
    test_origin_count: int = TEST_ORIGIN_COUNT
    split_seed: int = 42
    target_mode: str = "multi"
    pred_len: int = 24
    aw: float = 0.5
    vw: float = 0.5

    @property
    def tag_prefix(self) -> str:
        if self.split_mode == "final":
            return PROTOCOL_ID
        train = round(self.train_ratio * 100)
        test = round(self.test_ratio * 100)
        return f"fair-r{train:02d}{test:02d}-split{self.split_seed}"

    @property
    def ratios(self) -> tuple[float, float, float]:
        return self.train_ratio, self.valid_ratio, self.test_ratio


FAIR_OVERALL_PROTOCOL = ExperimentProtocol()


@dataclass(frozen=True)
class MethodSpec:
    key: str
    label: str
    model_dir: str
    entrypoint: str = ""
    deterministic: bool = False
    rerunnable: bool = True
    protocol_note: str = ""

    @property
    def prediction_subdir(self) -> Path:
        return Path(self.model_dir) / "multi" / "charts" / "test_preds"


def canonical_run_tag(
    method: MethodSpec,
    seed: int | None,
    protocol: ExperimentProtocol = FAIR_OVERALL_PROTOCOL,
) -> str:
    """Return the artifact tag used by a validated overall-comparison run."""
    if not method.deterministic and seed is None:
        raise ValueError(f"A training seed is required for {method.label}.")
    suffix = "deterministic" if method.deterministic else f"seed{seed}"
    return f"{protocol.tag_prefix}-{suffix}"


DEFAULT_METHODS = (
    MethodSpec("ha", "HA", "ha", "main_ha.py", deterministic=True),
    MethodSpec("arima", "ARIMA", "arima", "main_arima.py", deterministic=True),
    MethodSpec(
        "attention-lstm",
        "Attention-LSTM",
        "(ablation)tn-attention_lstm/ablate",
        "main_ablate.py",
        protocol_note="main_ablate.py must run with target_mode=multi",
    ),
    MethodSpec(
        "timesnet",
        "TimesNet",
        "(ablation)tn-timesnet/ablate",
        "main_ablate.py",
        protocol_note="main_ablate.py must run with target_mode=multi",
    ),
    MethodSpec("dcrnn", "DCRNN", "dcrnn", "main_dcrnn.py"),
    MethodSpec("stgcn", "STGCN", "stgcn", "main_stgcn.py"),
    MethodSpec(
        "graph-wavenet",
        "GraphWaveNet",
        "graph-wavenet",
        "main_wavenet.py",
    ),
    MethodSpec(
        "psta",
        "PSTA",
        "psta",
        "main_psta.py",
        protocol_note="legacy default was temporal 0.70/0.10/0.20",
    ),
    MethodSpec(
        "imfcp",
        "ImFCP",
        "imfcp",
        "main_imfcp.py",
        protocol_note="legacy default was temporal 0.70/0.20/0.10",
    ),
    MethodSpec(
        "st-moe",
        "ST-MoE-RMQRN",
        "st-moe-rmqrn",
        "main_st_moe.py",
        protocol_note="legacy default was temporal 0.70/0.10/0.20",
    ),
    MethodSpec(
        "dsf-net",
        "DSF-Net (ours)",
        "tn-timesnet_sn-spatial_conv_fn-dcp_fusion_rb-ha",
        "main_sim.py",
        protocol_note=(
            "main_sim.py now separates training seed from fixed split_seed=42"
        ),
    ),
)

DETERMINISTIC_METHOD_KEYS = frozenset(
    method.key for method in DEFAULT_METHODS if method.deterministic
)
METHOD_ALIASES = {
    "ptsa": "psta",
    "st-moe-rmqrn": "st-moe",
    "dcafusion": "dsf-net",
    "dcpfusion": "dsf-net",
    "wavenet": "graph-wavenet",
    "graph_wavenet": "graph-wavenet",
    "attention_lstm": "attention-lstm",
    "st_moe": "st-moe",
}


def resolve_methods(keys: Sequence[str] | None) -> tuple[MethodSpec, ...]:
    if keys is None:
        return DEFAULT_METHODS

    by_key = {method.key: method for method in DEFAULT_METHODS}
    resolved: list[MethodSpec] = []
    unknown: list[str] = []
    for requested_key in keys:
        normalized = requested_key.strip().lower()
        normalized = METHOD_ALIASES.get(normalized, normalized)
        method = by_key.get(normalized)
        if method is None:
            unknown.append(requested_key)
        elif method not in resolved:
            resolved.append(method)
    if unknown:
        valid = ", ".join(by_key)
        raise ValueError(f"Unknown method key(s): {', '.join(unknown)}. Valid keys: {valid}")
    if not resolved:
        raise ValueError("At least one method must be selected.")
    return tuple(resolved)
