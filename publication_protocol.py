# coding: utf-8
"""Auditable provenance gate for the final TRC experiments.

The final comparison must never accept a prediction artifact merely because
its filename or run tag contains ``final``.  This module binds each NPZ to a
canonical experiment protocol and to the exact matrix bytes used by its
segment.  It also validates the locked temporal split and reconstructs the
dual targets from ``y.npy`` when requested.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PROTOCOL_SCHEMA_VERSION = "dsf-publication-protocol/v2"
PROTOCOL_ID = "trc-final-topodir-dfree80-v2"
FINAL_MATRIX_DIRECTORY_NAME = "matrices_trc_final_topodir_v2"
MATRIX_MANIFEST_NAME = "publication_matrix_manifest.json"
SEGMENT_PROVENANCE_NAME = "matrix_provenance.json"
MATRIX_FILE_NAMES = (
    "cur.npy",
    "hist_y.npy",
    "up.npy",
    "down.npy",
    "in.npy",
    "out.npy",
    "y.npy",
    "cur_time_mask.npy",
    "direction_time_mask.npy",
    "role_mask.npy",
    "target_mask.npy",
    "forecast_origin_time.npy",
    "target_time.npy",
    "training_profile.npy",
    "input_profile.npy",
    "context_profile.npy",
)
EVALUATION_SEGMENTS = tuple(f"M-{index:02d}" for index in range(1, 18))

HISTORY_LENGTH = 168
CONTEXT_HISTORY_LENGTH = 48

TRAIN_START = 0
TRAIN_STOP = 1955
EMBARGO_START = TRAIN_STOP
EMBARGO_STOP = 2147
VALID_START = EMBARGO_STOP
VALID_STOP = 2527
TEST_START = VALID_STOP
TEST_STOP = 3424
TOTAL_ORIGINS = TEST_STOP

TRAIN_ORIGIN_COUNT = TRAIN_STOP - TRAIN_START
EMBARGO_ORIGIN_COUNT = EMBARGO_STOP - EMBARGO_START
VALID_ORIGIN_COUNT = VALID_STOP - VALID_START
TEST_ORIGIN_COUNT = TEST_STOP - TEST_START

FINAL_TRAIN_RATIO = TRAIN_ORIGIN_COUNT / TOTAL_ORIGINS
FINAL_EMBARGO_RATIO = EMBARGO_ORIGIN_COUNT / TOTAL_ORIGINS
FINAL_VALID_RATIO = VALID_ORIGIN_COUNT / TOTAL_ORIGINS
FINAL_TEST_RATIO = TEST_ORIGIN_COUNT / TOTAL_ORIGINS

PREDICTION_HORIZON = 24
TARGET_MODE = "multi"
CROWDING_WEIGHT = 0.5
SPEED_WEIGHT = 0.5
FREE_STATE_PERCENTILE = 80.0
SCENARIO_THRESHOLD_PERCENTILE = 70.0
SPLIT_SEED = 42

DIRECTION_DEFINITION_ID = "mainstream-topology_tributary-cog-junction/v1"
TARGET_DEFINITION_ID = "ci-dfree80-vfree80-w050_flow-sef/v1"
SPLIT_DEFINITION_ID = "p1-train1955-gap192-valid380_p2-test897/v1"
METRIC_DEFINITION_ID = "macro17-equal-task-trainstd-p70-jwp24/v1"


class PublicationProtocolError(RuntimeError):
    """Raised when a publication artifact cannot prove its provenance."""


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_value,
    )


def object_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def file_sha256(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


PUBLICATION_PROTOCOL: dict[str, Any] = {
    "schema_version": PROTOCOL_SCHEMA_VERSION,
    "protocol_id": PROTOCOL_ID,
    "direction_definition_id": DIRECTION_DEFINITION_ID,
    "target_definition_id": TARGET_DEFINITION_ID,
    "split_definition_id": SPLIT_DEFINITION_ID,
    "metric_definition_id": METRIC_DEFINITION_ID,
    "segments": list(EVALUATION_SEGMENTS),
    "split_mode": "final",
    "split_seed": SPLIT_SEED,
    "train_range": [TRAIN_START, TRAIN_STOP],
    "embargo_range": [EMBARGO_START, EMBARGO_STOP],
    "valid_range": [VALID_START, VALID_STOP],
    "test_range": [TEST_START, TEST_STOP],
    "total_origins": TOTAL_ORIGINS,
    "train_origin_count": TRAIN_ORIGIN_COUNT,
    "embargo_origin_count": EMBARGO_ORIGIN_COUNT,
    "valid_origin_count": VALID_ORIGIN_COUNT,
    "test_origin_count": TEST_ORIGIN_COUNT,
    "train_ratio": FINAL_TRAIN_RATIO,
    "embargo_ratio": FINAL_EMBARGO_RATIO,
    "valid_ratio": FINAL_VALID_RATIO,
    "test_ratio": FINAL_TEST_RATIO,
    "target_mode": TARGET_MODE,
    "prediction_horizon": PREDICTION_HORIZON,
    "history_length": HISTORY_LENGTH,
    "context_history_length": CONTEXT_HISTORY_LENGTH,
    "crowding_weight": CROWDING_WEIGHT,
    "speed_weight": SPEED_WEIGHT,
    "free_state_percentile": FREE_STATE_PERCENTILE,
    "scenario_threshold_percentile": SCENARIO_THRESHOLD_PERCENTILE,
}
PROTOCOL_SHA256 = object_sha256(PUBLICATION_PROTOCOL)

FINAL_EXPERIMENT_SPEC: dict[str, Any] = {
    "schema_version": "dsf-final-experiment-spec/v1",
    "publication_protocol_sha256": PROTOCOL_SHA256,
    "evaluation_segments": list(EVALUATION_SEGMENTS),
    "learning_methods": [
        "attention-lstm",
        "timesnet",
        "dcrnn",
        "stgcn",
        "graph-wavenet",
        "psta",
        "imfcp",
        "st-moe",
        "dsf-net",
    ],
    "deterministic_references": ["ha", "arima"],
    "learning_rates": [1e-4, 3e-4, 1e-3],
    "screening_segments": ["M-02", "M-10", "M-16"],
    "stability_segments": ["M-01", "M-06", "M-10", "M-12", "M-17"],
    "main_seeds": [42],
    "stability_seeds": [42, 43, 44],
    "screening_epochs": 80,
    "formal_epochs": 150,
    "early_stopping": {
        "warmup_epochs": 30,
        "patience": 20,
        "min_delta": 1e-4,
    },
    "optimizer": {
        "name": "adamw",
        "weight_decay": 1e-4,
        "effective_batch_size": 32,
        "gradient_clip_norm": 1.0,
    },
    "scheduler": {
        "name": "plateau",
        "factor": 0.5,
        "patience": 10,
        "min_lr": 1e-6,
    },
    "direction_variants": [
        "full",
        "pairwise-role-shuffled",
        "fully-role-shuffled",
    ],
    "fusion_variants": [
        "dcafusion",
        "self-attention",
        "cross-attention",
        "concat-mlp",
    ],
    "arima_candidate_count": 17,
    "gpu_max_parallel": 3,
    "arima_cpu_max_parallel": 6,
    "min_free_vram_mb": 12_000,
    "launch_stagger_seconds": 8,
    "time_gates_hours": {
        "screening": 7,
        "main": 32,
        "stability": 39,
        "analyses": 46,
        "final": 48,
    },
}
FINAL_EXPERIMENT_SPEC_SHA256 = object_sha256(FINAL_EXPERIMENT_SPEC)


def final_experiment_task_counts(
    spec: Mapping[str, Any] = FINAL_EXPERIMENT_SPEC,
) -> dict[str, int]:
    """Return the locked task counts derived from the formal experiment spec."""
    learning_methods = len(spec["learning_methods"])
    segments = len(spec["evaluation_segments"])
    learning_rates = len(spec["learning_rates"])
    screening_segments = len(spec["screening_segments"])
    main_seeds = len(spec["main_seeds"])
    stability_segments = len(spec["stability_segments"])
    additional_stability_seeds = len(spec["stability_seeds"]) - main_seeds
    additional_direction_variants = len(spec["direction_variants"]) - 1
    additional_fusion_variants = len(spec["fusion_variants"]) - 1
    deterministic_references = len(spec["deterministic_references"])

    main_training = learning_methods * segments * main_seeds
    stability_training = 2 * stability_segments * additional_stability_seeds
    direction_training = segments * additional_direction_variants
    fusion_training = screening_segments * additional_fusion_variants
    return {
        "learning_rate_screening": (
            learning_methods * learning_rates * screening_segments
        ),
        "arima_validation": int(spec["arima_candidate_count"]) * segments,
        "main_training": main_training,
        "stability_training": stability_training,
        "direction_training": direction_training,
        "fusion_training": fusion_training,
        "final_evaluation": (
            main_training
            + stability_training
            + direction_training
            + fusion_training
            + deterministic_references * segments
        ),
    }


def validate_final_experiment_spec(
    spec: Mapping[str, Any] = FINAL_EXPERIMENT_SPEC,
) -> None:
    """Fail closed if a formal experiment spec differs from the locked design."""
    exact_fields: dict[str, Any] = {
        "publication_protocol_sha256": PROTOCOL_SHA256,
        "evaluation_segments": list(EVALUATION_SEGMENTS),
        "learning_methods": FINAL_EXPERIMENT_SPEC["learning_methods"],
        "deterministic_references": ["ha", "arima"],
        "learning_rates": [1e-4, 3e-4, 1e-3],
        "screening_segments": ["M-02", "M-10", "M-16"],
        "stability_segments": ["M-01", "M-06", "M-10", "M-12", "M-17"],
        "main_seeds": [42],
        "stability_seeds": [42, 43, 44],
        "screening_epochs": 80,
        "formal_epochs": 150,
        "early_stopping": {
            "warmup_epochs": 30,
            "patience": 20,
            "min_delta": 1e-4,
        },
        "optimizer": {
            "name": "adamw",
            "weight_decay": 1e-4,
            "effective_batch_size": 32,
            "gradient_clip_norm": 1.0,
        },
        "scheduler": {
            "name": "plateau",
            "factor": 0.5,
            "patience": 10,
            "min_lr": 1e-6,
        },
        "direction_variants": [
            "full",
            "pairwise-role-shuffled",
            "fully-role-shuffled",
        ],
        "fusion_variants": [
            "dcafusion",
            "self-attention",
            "cross-attention",
            "concat-mlp",
        ],
        "arima_candidate_count": 17,
        "gpu_max_parallel": 3,
        "arima_cpu_max_parallel": 6,
        "min_free_vram_mb": 12_000,
        "launch_stagger_seconds": 8,
        "time_gates_hours": {
            "screening": 7,
            "main": 32,
            "stability": 39,
            "analyses": 46,
            "final": 48,
        },
    }
    for field, expected in exact_fields.items():
        if spec.get(field) != expected:
            raise PublicationProtocolError(
                f"final experiment {field} differs from the locked protocol: "
                f"expected={expected!r}, actual={spec.get(field)!r}"
            )

    expected_counts = {
        "learning_rate_screening": 81,
        "arima_validation": 289,
        "main_training": 153,
        "stability_training": 20,
        "direction_training": 34,
        "fusion_training": 9,
        "final_evaluation": 250,
    }
    if final_experiment_task_counts(spec) != expected_counts:
        raise PublicationProtocolError(
            "final experiment task counts differ from the locked protocol"
        )



def expected_test_indices() -> np.ndarray:
    return np.arange(TEST_START, TEST_STOP, dtype=np.int64)


def _matrix_file_record(path: Path) -> dict[str, Any]:
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    return {
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
    }


def matrix_builder_source_sha256() -> str:
    root = Path(__file__).resolve().parent
    sources = (
        root / "cleaning" / "publication_matrix.py",
        root / "publication_build_sql.py",
        root / "publication_matrix_builder.py",
        root / "publication_matrix_resumable.py",
    )
    missing = [str(source) for source in sources if not source.is_file()]
    if missing:
        raise PublicationProtocolError(f"Matrix builder sources not found: {missing}")
    records = {
        source.relative_to(root).as_posix(): file_sha256(source)
        for source in sources
    }
    return object_sha256(records)


def write_segment_provenance(
    segment_dir: os.PathLike[str] | str,
    segment: str,
) -> Path:
    """Seal matrices immediately after one segment is generated."""
    directory = Path(segment_dir).resolve()
    missing = [name for name in MATRIX_FILE_NAMES if not (directory / name).is_file()]
    if missing:
        raise PublicationProtocolError(
            f"Cannot seal {segment}; missing matrix files: {', '.join(missing)}"
        )
    files = {name: _matrix_file_record(directory / name) for name in MATRIX_FILE_NAMES}
    shape_failures = _validate_segment_shapes(segment, files)
    if shape_failures:
        raise PublicationProtocolError("; ".join(shape_failures))
    core: dict[str, Any] = {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": PROTOCOL_SHA256,
        "direction_definition_id": DIRECTION_DEFINITION_ID,
        "target_definition_id": TARGET_DEFINITION_ID,
        "segment": segment,
        "matrix_builder_source_sha256": matrix_builder_source_sha256(),
        "files": files,
        "bundle_sha256": object_sha256(files),
    }
    payload = {**core, "provenance_sha256": object_sha256(core)}
    destination = directory / SEGMENT_PROVENANCE_NAME
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def _validate_segment_provenance(
    path: Path,
    segment: str,
    files: Mapping[str, Any],
) -> tuple[list[str], str]:
    failures: list[str] = []
    if not path.is_file():
        return (
            [
                f"{segment} has no generation-time {SEGMENT_PROVENANCE_NAME}; "
                "legacy matrices cannot be promoted by adding a root manifest"
            ],
            "",
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return [f"cannot read {segment} matrix provenance: {error}"], ""
    expected_scalars = {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": PROTOCOL_SHA256,
        "direction_definition_id": DIRECTION_DEFINITION_ID,
        "target_definition_id": TARGET_DEFINITION_ID,
        "segment": segment,
        "matrix_builder_source_sha256": matrix_builder_source_sha256(),
        "bundle_sha256": object_sha256(files),
    }
    for key, expected in expected_scalars.items():
        if payload.get(key) != expected:
            failures.append(
                f"{segment} provenance {key}={payload.get(key)!r}, expected {expected!r}"
            )
    if payload.get("files") != files:
        failures.append(f"{segment} provenance file records differ from current matrices")
    core = dict(payload)
    stored_sha = str(core.pop("provenance_sha256", ""))
    if not stored_sha or object_sha256(core) != stored_sha:
        failures.append(f"{segment} provenance content fingerprint differs")
    return failures, stored_sha


def _validate_segment_shapes(segment: str, files: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    for name in MATRIX_FILE_NAMES:
        shape = tuple(int(value) for value in files[name]["shape"])
        if name not in {"training_profile.npy", "input_profile.npy", "context_profile.npy"} and (
            not shape or shape[0] != TOTAL_ORIGINS
        ):
            failures.append(
                f"{segment}/{name} has {shape[0] if shape else 0} origins; "
                f"expected {TOTAL_ORIGINS}"
            )

    expected_shapes = {
        "cur_time_mask.npy": (TOTAL_ORIGINS, HISTORY_LENGTH),
        "direction_time_mask.npy": (
            TOTAL_ORIGINS,
            4,
            CONTEXT_HISTORY_LENGTH,
        ),
        "role_mask.npy": (TOTAL_ORIGINS, 5),
        "target_mask.npy": (
            TOTAL_ORIGINS,
            PREDICTION_HORIZON,
            2,
        ),
        "forecast_origin_time.npy": (TOTAL_ORIGINS,),
        "target_time.npy": (TOTAL_ORIGINS, PREDICTION_HORIZON),
        "training_profile.npy": (6,),
        "input_profile.npy": (24, 2),
        "context_profile.npy": (4, 2, 2),
    }
    for name, expected in expected_shapes.items():
        shape = tuple(int(value) for value in files[name]["shape"])
        if shape != expected:
            failures.append(
                f"{segment}/{name} shape {shape}, expected {expected}"
            )

    for name in ("cur.npy", "hist_y.npy"):
        shape = tuple(int(value) for value in files[name]["shape"])
        if len(shape) != 3 or shape[1] != HISTORY_LENGTH:
            failures.append(
                f"{segment}/{name} shape {shape} does not use the locked "
                f"{HISTORY_LENGTH}-hour history"
            )

    y_shape = tuple(int(value) for value in files["y.npy"]["shape"])
    if (
        len(y_shape) != 3
        or y_shape[1] != PREDICTION_HORIZON
        or y_shape[2] < 3
    ):
        failures.append(
            f"{segment}/y.npy shape {y_shape} is incompatible with "
            "[speed, occupancy, equivalent flow] and a 24-hour horizon"
        )
    for name in ("up.npy", "down.npy", "in.npy", "out.npy"):
        shape = tuple(int(value) for value in files[name]["shape"])
        if len(shape) != 3 or shape[1:] != (CONTEXT_HISTORY_LENGTH, 2):
            failures.append(
                f"{segment}/{name} shape {shape} does not contain the required "
                f"{CONTEXT_HISTORY_LENGTH}-hour two-channel directional state"
            )
    return failures


def build_matrix_manifest(
    matrices_root: os.PathLike[str] | str,
    segments: Sequence[str] = EVALUATION_SEGMENTS,
    *,
    require_versioned_directory: bool = True,
) -> dict[str, Any]:
    """Hash and describe a complete final-protocol matrix directory."""
    root = Path(matrices_root).resolve()
    if require_versioned_directory and root.name != FINAL_MATRIX_DIRECTORY_NAME:
        raise PublicationProtocolError(
            f"Final matrices must live in a versioned directory named "
            f"{FINAL_MATRIX_DIRECTORY_NAME!r}, got {root.name!r}."
        )

    segment_records: dict[str, Any] = {}
    failures: list[str] = []
    for segment in segments:
        segment_dir = root / segment
        missing = [name for name in MATRIX_FILE_NAMES if not (segment_dir / name).is_file()]
        if missing:
            failures.append(f"{segment} missing matrix files: {', '.join(missing)}")
            continue
        files = {
            name: _matrix_file_record(segment_dir / name)
            for name in MATRIX_FILE_NAMES
        }
        failures.extend(_validate_segment_shapes(segment, files))
        provenance_failures, provenance_sha = _validate_segment_provenance(
            segment_dir / SEGMENT_PROVENANCE_NAME,
            segment,
            files,
        )
        failures.extend(provenance_failures)
        segment_records[segment] = {
            "files": files,
            "bundle_sha256": object_sha256(files),
            "segment_provenance_sha256": provenance_sha,
        }
    if failures:
        raise PublicationProtocolError("; ".join(failures))

    core: dict[str, Any] = {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "protocol": PUBLICATION_PROTOCOL,
        "protocol_sha256": PROTOCOL_SHA256,
        "matrix_directory_name": root.name,
        "segments": segment_records,
    }
    return {**core, "manifest_sha256": object_sha256(core)}


def write_matrix_manifest(
    matrices_root: os.PathLike[str] | str,
    output_path: os.PathLike[str] | str | None = None,
    segments: Sequence[str] = EVALUATION_SEGMENTS,
) -> Path:
    root = Path(matrices_root).resolve()
    manifest = build_matrix_manifest(root, segments=segments)
    destination = (
        Path(output_path).resolve()
        if output_path is not None
        else root / MATRIX_MANIFEST_NAME
    )
    if destination.parent != root:
        raise PublicationProtocolError(
            "The publication manifest must be stored inside the fingerprinted "
            "matrix root so later verification cannot resolve a different directory."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def validate_matrix_manifest(
    manifest: Mapping[str, Any],
    matrices_root: os.PathLike[str] | str | None = None,
    *,
    verify_files: bool = False,
) -> list[str]:
    failures: list[str] = []
    if manifest.get("schema_version") != PROTOCOL_SCHEMA_VERSION:
        failures.append("matrix manifest schema version is not the publication schema")
    if manifest.get("protocol") != PUBLICATION_PROTOCOL:
        failures.append("matrix manifest embeds a different publication protocol")
    if manifest.get("protocol_sha256") != PROTOCOL_SHA256:
        failures.append("matrix manifest protocol fingerprint differs")

    core = dict(manifest)
    stored_manifest_sha = str(core.pop("manifest_sha256", ""))
    if not stored_manifest_sha or object_sha256(core) != stored_manifest_sha:
        failures.append("matrix manifest content fingerprint differs")

    segment_records = manifest.get("segments")
    if not isinstance(segment_records, Mapping) or not segment_records:
        failures.append("matrix manifest contains no segment records")
        return failures

    for segment, record in segment_records.items():
        if not isinstance(record, Mapping) or not isinstance(record.get("files"), Mapping):
            failures.append(f"matrix manifest has an invalid record for {segment}")
            continue
        files = record["files"]
        if set(files) != set(MATRIX_FILE_NAMES):
            failures.append(f"matrix manifest has an incomplete file set for {segment}")
            continue
        if record.get("bundle_sha256") != object_sha256(files):
            failures.append(f"matrix bundle fingerprint differs for {segment}")
        failures.extend(_validate_segment_shapes(str(segment), files))

    if verify_files:
        if matrices_root is None:
            failures.append("matrices_root is required when verify_files=True")
        else:
            root = Path(matrices_root).resolve()
            if root.name != FINAL_MATRIX_DIRECTORY_NAME:
                failures.append("matrix root is not the canonical versioned directory")
            for segment, record in segment_records.items():
                for name, expected in record["files"].items():
                    path = root / str(segment) / str(name)
                    if not path.is_file():
                        failures.append(f"missing matrix file {path}")
                        continue
                    if file_sha256(path) != expected.get("sha256"):
                        failures.append(f"matrix content fingerprint differs for {segment}/{name}")
    return failures


def load_matrix_manifest(
    path: os.PathLike[str] | str,
    *,
    verify_files: bool = False,
) -> dict[str, Any]:
    manifest_path = Path(path).resolve()
    if not manifest_path.is_file():
        raise PublicationProtocolError(f"Publication matrix manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures = validate_matrix_manifest(
        manifest,
        matrices_root=manifest_path.parent,
        verify_files=verify_files,
    )
    if failures:
        raise PublicationProtocolError("; ".join(failures))
    return manifest


def default_manifest_path() -> Path:
    explicit = os.environ.get("DSF_PUBLICATION_MATRIX_MANIFEST")
    if explicit:
        return Path(explicit).resolve()
    matrices_root = Path(
        os.environ.get(
            "MATRICES_DIR",
            str(Path("datasets") / FINAL_MATRIX_DIRECTORY_NAME),
        )
    )
    return (matrices_root / MATRIX_MANIFEST_NAME).resolve()


def publication_metadata_for_segment(
    segment: str,
    manifest_path: os.PathLike[str] | str | None = None,
) -> dict[str, Any]:
    path = Path(manifest_path).resolve() if manifest_path is not None else default_manifest_path()
    manifest = load_matrix_manifest(path)
    segment_record = manifest["segments"].get(segment)
    if segment_record is None:
        raise PublicationProtocolError(
            f"Publication matrix manifest has no record for {segment}."
        )
    return {
        "publication_schema_version": PROTOCOL_SCHEMA_VERSION,
        "publication_protocol_id": PROTOCOL_ID,
        "publication_protocol_sha256": PROTOCOL_SHA256,
        "matrix_manifest_sha256": manifest["manifest_sha256"],
        "matrix_bundle_sha256": segment_record["bundle_sha256"],
        "direction_definition_id": DIRECTION_DEFINITION_ID,
        "target_definition_id": TARGET_DEFINITION_ID,
        "split_definition_id": SPLIT_DEFINITION_ID,
        "metric_definition_id": METRIC_DEFINITION_ID,
        "split_seed": SPLIT_SEED,
        "train_origin_count": TRAIN_ORIGIN_COUNT,
        "embargo_origin_count": EMBARGO_ORIGIN_COUNT,
        "valid_origin_count": VALID_ORIGIN_COUNT,
        "test_origin_count": TEST_ORIGIN_COUNT,
    }


def assert_publication_matrix_segment(
    matrices_root: os.PathLike[str] | str,
    segment: str,
    manifest_path: os.PathLike[str] | str | None = None,
) -> dict[str, Any]:
    """Reject a final-mode training job unless its segment bytes are sealed.

    This deliberately runs before a loader materializes the training arrays.
    It checks only the requested segment's file bytes, avoiding an unnecessary
    17-segment hash scan for every independent training process.
    """
    root = Path(matrices_root).resolve()
    manifest_file = (
        Path(manifest_path).resolve()
        if manifest_path is not None
        else default_manifest_path()
    )
    failures: list[str] = []
    if root.name != FINAL_MATRIX_DIRECTORY_NAME:
        failures.append(
            f"matrix root {root} is not the canonical {FINAL_MATRIX_DIRECTORY_NAME} directory"
        )
    if manifest_file.parent != root:
        failures.append(
            f"matrix root {root} differs from manifest root {manifest_file.parent}"
        )
    if failures:
        raise PublicationProtocolError("; ".join(failures))

    manifest = load_matrix_manifest(manifest_file, verify_files=False)
    record = manifest["segments"].get(segment)
    if record is None:
        raise PublicationProtocolError(
            f"Publication matrix manifest has no record for {segment}."
        )
    for name, expected in record["files"].items():
        path = root / segment / name
        if not path.is_file():
            failures.append(f"missing matrix file {path}")
        elif file_sha256(path) != expected.get("sha256"):
            failures.append(f"matrix content fingerprint differs for {segment}/{name}")
    if failures:
        raise PublicationProtocolError("; ".join(failures))
    return publication_metadata_for_segment(segment, manifest_file)


def reconstruct_all_targets(
    matrices_root: os.PathLike[str] | str,
    segment: str,
) -> np.ndarray:
    """Reconstruct every locked dual target from raw ``y.npy`` components."""
    components = np.load(
        Path(matrices_root) / segment / "y.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    if components.shape[0] != TOTAL_ORIGINS or components.shape[1] < PREDICTION_HORIZON:
        raise PublicationProtocolError(
            f"{segment}/y.npy shape {components.shape} is incompatible with the final split."
        )
    values = np.asarray(components[:, :PREDICTION_HORIZON, :3], dtype=np.float64)
    profile_path = Path(matrices_root) / segment / "training_profile.npy"
    profile = np.asarray(
        np.load(profile_path, allow_pickle=False),
        dtype=np.float64,
    )
    if profile.shape != (6,) or not np.isfinite(profile).all():
        raise PublicationProtocolError(
            f"{segment}/training_profile.npy must contain six finite train-only values."
        )
    d_free, v_free = float(profile[0]), float(profile[1])
    if v_free <= 0.0 or d_free <= 0.0:
        raise PublicationProtocolError(f"Invalid free-state scale for {segment}.")
    speed_degradation = np.clip(1.0 - values[:, :, 0] / v_free, 0.0, 1.0)
    normalized_occupancy = np.clip(values[:, :, 1] / d_free, 0.0, 1.0)
    crowding = (
        CROWDING_WEIGHT * normalized_occupancy
        + SPEED_WEIGHT * speed_degradation
    )
    # A covered hour with no vessel has D=0 and all traffic indicators equal zero.
    crowding[values[:, :, 1] <= 0.0] = 0.0
    targets = np.stack((crowding, values[:, :, 2]), axis=-1)
    return targets


def reconstruct_final_targets(
    matrices_root: os.PathLike[str] | str,
    segment: str,
) -> np.ndarray:
    return reconstruct_all_targets(matrices_root, segment)[TEST_START:TEST_STOP]


def final_training_profile(
    matrices_root: os.PathLike[str] | str,
    segment: str,
) -> tuple[float, float, float, float]:
    """Return train-only P70 thresholds and scales from unique hourly states."""
    profile = np.asarray(
        np.load(
            Path(matrices_root) / segment / "training_profile.npy",
            allow_pickle=False,
        ),
        dtype=np.float64,
    )
    if profile.shape != (6,) or not np.isfinite(profile).all():
        raise PublicationProtocolError(f"Invalid training profile for {segment}.")
    return (
        float(profile[4]),
        float(profile[5]),
        float(profile[2]),
        float(profile[3]),
    )


def _npz_scalar(data: np.lib.npyio.NpzFile, key: str, default: Any = None) -> Any:
    if key not in data.files:
        return default
    value = np.asarray(data[key])
    return value.item() if value.shape == () else default


def validate_publication_npz(
    npz_path: os.PathLike[str] | str,
    manifest_path: os.PathLike[str] | str,
    *,
    verify_ground_truth: bool = True,
) -> list[str]:
    """Return every reason a prediction artifact is inadmissible."""
    path = Path(npz_path).resolve()
    manifest_file = Path(manifest_path).resolve()
    try:
        manifest = load_matrix_manifest(manifest_file)
    except (OSError, ValueError, PublicationProtocolError) as error:
        return [str(error)]

    failures: list[str] = []
    try:
        with np.load(path, allow_pickle=False) as data:
            required_arrays = {"y_true", "y_pred", "test_indices"}
            missing_arrays = sorted(required_arrays - set(data.files))
            if missing_arrays:
                return ["missing arrays: " + ", ".join(missing_arrays)]

            segment = str(_npz_scalar(data, "segment", ""))
            segment_record = manifest["segments"].get(segment)
            if segment_record is None:
                failures.append(f"segment {segment!r} is absent from the matrix manifest")

            expected_scalars: dict[str, Any] = {
                "publication_schema_version": PROTOCOL_SCHEMA_VERSION,
                "publication_protocol_id": PROTOCOL_ID,
                "publication_protocol_sha256": PROTOCOL_SHA256,
                "matrix_manifest_sha256": manifest["manifest_sha256"],
                "direction_definition_id": DIRECTION_DEFINITION_ID,
                "target_definition_id": TARGET_DEFINITION_ID,
                "split_definition_id": SPLIT_DEFINITION_ID,
                "metric_definition_id": METRIC_DEFINITION_ID,
                "split_mode": "final",
                "split_seed": SPLIT_SEED,
                "target_mode": TARGET_MODE,
                "pred_len": PREDICTION_HORIZON,
                "train_origin_count": TRAIN_ORIGIN_COUNT,
                "embargo_origin_count": EMBARGO_ORIGIN_COUNT,
                "valid_origin_count": VALID_ORIGIN_COUNT,
                "test_origin_count": TEST_ORIGIN_COUNT,
            }
            if segment_record is not None:
                expected_scalars["matrix_bundle_sha256"] = segment_record["bundle_sha256"]
            for key, expected in expected_scalars.items():
                actual = _npz_scalar(data, key)
                if actual != expected:
                    failures.append(f"{key}={actual!r}, expected {expected!r}")

            for key, expected in {
                "aw": CROWDING_WEIGHT,
                "vw": SPEED_WEIGHT,
                "scenario_threshold_percentile": SCENARIO_THRESHOLD_PERCENTILE,
            }.items():
                actual = _npz_scalar(data, key)
                try:
                    matches = np.isclose(float(actual), expected, atol=1e-8, rtol=0.0)
                except (TypeError, ValueError):
                    matches = False
                if not matches:
                    failures.append(f"{key}={actual!r}, expected {expected!r}")

            if segment_record is not None:
                expected_tau_c, expected_tau_f, _, _ = final_training_profile(
                    manifest_file.parent,
                    segment,
                )
                for key, expected in {
                    "scenario_tau_c": expected_tau_c,
                    "scenario_tau_f": expected_tau_f,
                }.items():
                    actual = _npz_scalar(data, key)
                    try:
                        matches = np.isclose(
                            float(actual), expected, atol=1e-6, rtol=1e-6
                        )
                    except (TypeError, ValueError):
                        matches = False
                    if not matches:
                        failures.append(
                            f"{key}={actual!r}, expected training-split P70 {expected!r}"
                        )

            test_indices = np.asarray(data["test_indices"], dtype=np.int64)
            if not np.array_equal(test_indices, expected_test_indices()):
                failures.append(
                    "test_indices do not equal the locked 897-origin second-period test set"
                )

            y_true = np.asarray(data["y_true"])
            y_pred = np.asarray(data["y_pred"])
            expected_shape = (TEST_ORIGIN_COUNT, PREDICTION_HORIZON, 2)
            if y_true.shape != expected_shape:
                failures.append(f"y_true shape {y_true.shape}, expected {expected_shape}")
            if y_pred.shape != y_true.shape:
                failures.append("y_true/y_pred shape mismatch")
            if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
                failures.append("prediction artifact contains non-finite values")

            if (
                verify_ground_truth
                and segment_record is not None
                and y_true.shape == expected_shape
            ):
                expected_truth = reconstruct_final_targets(manifest_file.parent, segment)
                if not np.allclose(
                    y_true,
                    expected_truth,
                    rtol=1e-5,
                    atol=1e-6,
                    equal_nan=False,
                ):
                    failures.append(
                        "y_true does not match the D_free80/V_free80 target reconstructed "
                        "from the fingerprinted matrices"
                    )
    except Exception as error:
        return [f"cannot read publication NPZ {path}: {error}"]
    return failures


def assert_publication_npz(
    npz_path: os.PathLike[str] | str,
    manifest_path: os.PathLike[str] | str,
    *,
    verify_ground_truth: bool = True,
) -> None:
    failures = validate_publication_npz(
        npz_path,
        manifest_path,
        verify_ground_truth=verify_ground_truth,
    )
    if failures:
        raise PublicationProtocolError("; ".join(failures))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build-manifest")
    build.add_argument("--matrices-dir", type=Path, required=True)
    build.add_argument("--output", type=Path, default=None)
    build.add_argument("--segments", nargs="+", default=list(EVALUATION_SEGMENTS))
    validate = subparsers.add_parser("validate-npz")
    validate.add_argument("--manifest", type=Path, required=True)
    validate.add_argument("npz", type=Path, nargs="+")
    validate.add_argument("--skip-ground-truth", action="store_true")
    verify = subparsers.add_parser("verify-manifest")
    verify.add_argument("--manifest", type=Path, default=default_manifest_path())
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "build-manifest":
        destination = write_matrix_manifest(
            args.matrices_dir,
            output_path=args.output,
            segments=args.segments,
        )
        print(destination)
        return
    if args.command == "verify-manifest":
        manifest = load_matrix_manifest(args.manifest, verify_files=True)
        print(
            f"PASS {Path(args.manifest).resolve()} "
            f"segments={len(manifest['segments'])} "
            f"sha256={manifest['manifest_sha256']}"
        )
        return

    has_failures = False
    for path in args.npz:
        failures = validate_publication_npz(
            path,
            args.manifest,
            verify_ground_truth=not args.skip_ground_truth,
        )
        if failures:
            has_failures = True
            print(f"INVALID {path}")
            for failure in failures:
                print(f"  - {failure}")
        else:
            print(f"PASS {path}")
    if has_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
