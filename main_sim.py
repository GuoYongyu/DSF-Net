# utf-8
import argparse
import json
import random
import traceback
import warnings
from typing import Any
from multiprocessing import Process, set_start_method

import numpy as np
import torch

from fair_experiment_protocol import FAIR_OVERALL_PROTOCOL
from publication_protocol import publication_metadata_for_segment

from sim.config import *
from sim.train import train
from sim.make_data import make_loader
from logger.logger import LOG


warnings.filterwarnings("ignore")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Time-Space Fusion")
    parser.add_argument(
        "-a", "--all",
        action="store_true",
        help="Train each network combination automatically"
    )
    parser.add_argument(
        "-c", "--custom",
        action="store_true",
        help="Use custom network combination"
    )
    parser.add_argument(
        "-s", "--segment",
        type=str,
        default="M-02",
        help="Segment to train"
    )
    parser.add_argument(
        "-tn", "--time-net",
        type=str,
        default="lstm",
        choices=["lstm", "attention-lstm", "timesnet"],
        help="Time network type"
    )
    parser.add_argument(
        "-sn", "--space-net",
        type=str,
        default="encoder",
        choices=["encoder", "cross-attention", "spatial-conv"],
        help="Space network type"
    )
    parser.add_argument(
        "-fn", "--fusion-net",
        type=str,
        default="weighted-concat",
        choices=[
            "weighted-concat",
            "weighted-fusion",
            "feature-mapping",
            "cross-attention",
            "self-attention",
            "dcp-fusion",
            "concat-mlp",
            "gmu",
        ],
        help="Fusion network type"
    )
    parser.add_argument(
        "-d", "--device",
        type=int,
        default=0,
        help="Device to use for training, if not available, use 'cpu'"
    )
    parser.add_argument(
        "-e", "--epochs",
        type=int,
        default=100,
        help="Number of epochs to train"
    )
    parser.add_argument(
        "-w", "--weights",
        action="store_true",
        help="Use mutiple weights combination for training"
    )
    parser.add_argument(
        "--aw",
        type=float,
        default=FAIR_OVERALL_PROTOCOL.aw,
        help="Crowding-index domain-occupancy weight",
    )
    parser.add_argument(
        "--vw",
        type=float,
        default=FAIR_OVERALL_PROTOCOL.vw,
        help="Crowding-index speed-degradation weight",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adam", "sgd", "rmsprop", "adamw"],
        help="Optimizer type"
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="annealing",
        choices=["annealing", "plateau", "linear", "polynomial"],
        help="Learning rate scheduler type"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate"
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay"
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=None,
        help="Max grad norm for clipping"
    )
    parser.add_argument(
        "--criterion",
        type=str,
        default="huber",
        choices=["mse", "huber", "scenario"],
        help="Training criterion"
    )
    parser.add_argument(
        "--loss-weighting",
        type=str,
        default="fixed",
        choices=["dynamic", "fixed", "uncertainty"],
        help="Multi-task loss weighting mode"
    )
    parser.add_argument(
        "--task-weights",
        type=float,
        nargs="+",
        default=None,
        help="Task weights when --loss-weighting=fixed, e.g. --task-weights 0.7 0.3"
    )
    parser.add_argument(
        "--loss-warmup-epochs",
        type=int,
        default=25,
        help="Warmup epochs using fixed task weights before switching to configured loss weighting"
    )
    parser.add_argument(
        "--loss-warmup-task-weights",
        type=float,
        nargs="+",
        default=None,
        help="Fixed task weights used during warmup, e.g. --loss-warmup-task-weights 0.5 0.5"
    )
    parser.add_argument(
        "--task-weights-end",
        type=float,
        nargs="+",
        default=None,
        help="End task weights for staged fixed weighting, e.g. --task-weights-end 0.3 0.7"
    )
    parser.add_argument(
        "--stage-start-epoch",
        type=int,
        default=None,
        help="Start epoch (1-based) for staged task weight interpolation"
    )
    parser.add_argument(
        "--stage-end-epoch",
        type=int,
        default=None,
        help="End epoch (1-based) for staged task weight interpolation"
    )
    parser.add_argument(
        "--split-mode",
        type=str,
        default=FAIR_OVERALL_PROTOCOL.split_mode,
        choices=["random", "temporal", "final"],
        help="Train/valid split mode"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for model initialization and training stochasticity"
    )
    parser.add_argument("--split-seed", type=int, default=FAIR_OVERALL_PROTOCOL.split_seed)
    parser.add_argument("--train-ratio", type=float, default=FAIR_OVERALL_PROTOCOL.train_ratio)
    parser.add_argument("--valid-ratio", type=float, default=FAIR_OVERALL_PROTOCOL.valid_ratio)
    parser.add_argument("--test-ratio", type=float, default=FAIR_OVERALL_PROTOCOL.test_ratio)
    parser.add_argument(
        "--purge-steps",
        type=int,
        default=0,
        help="Number of forecast origins removed before temporal split boundaries"
    )
    parser.add_argument(
        "--target-mode",
        type=str,
        default=FAIR_OVERALL_PROTOCOL.target_mode,
        choices=["multi", "crowding", "equiv_flow"],
        help="Training target mode: multi-task or single-task"
    )
    parser.add_argument(
        "--use-pcgrad",
        action="store_true",
        help="Enable PCGrad for multi-objective gradient conflict handling"
    )
    parser.add_argument(
        "--pcgrad-include-mape",
        action="store_true",
        help="Include lambda * MAPE loss as an extra PCGrad objective"
    )
    parser.add_argument(
        "--scenario-loss-reg-weights",
        type=float,
        nargs="+",
        default=None,
        help="Regression weights for --criterion scenario, e.g. 0.35 0.35"
    )
    parser.add_argument(
        "--scenario-loss-event-weights",
        type=float,
        nargs="+",
        default=None,
        help="Event focal weights for --criterion scenario, e.g. 0.10 0.10"
    )
    parser.add_argument(
        "--scenario-loss-wape-weight",
        type=float,
        default=0.1,
        help="WAPE term weight for --criterion scenario"
    )
    parser.add_argument(
        "--scenario-loss-peak-weight",
        type=float,
        default=0.0,
        help="Soft peak timing loss weight for --criterion scenario"
    )
    parser.add_argument(
        "--scenario-loss-joint-weight",
        type=float,
        default=0.0,
        help="Joint high-state focal loss weight for --criterion scenario"
    )
    parser.add_argument(
        "--scenario-loss-focal-gamma",
        type=float,
        default=2.0,
        help="Focal gamma for --criterion scenario event terms"
    )
    parser.add_argument(
        "--multi-process",
        action="store_true",
        help="use multiple processes to train models"
    )
    parser.add_argument(
        "--use-test",
        action="store_true",
        help="use test set for validation"
    )
    parser.add_argument(
        "--pred-len",
        type=int,
        default=FAIR_OVERALL_PROTOCOL.pred_len,
        choices=[6, 12, 24, 48],
        help="Prediction horizon length in hours"
    )
    parser.add_argument(
        "--mape-lambda",
        type=float,
        default=0.1,
        help="Weight of the auxiliary MAPE term in MSE/Huber training"
    )
    parser.add_argument(
        "--normalize-task-losses",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Normalize each task residual by its train-split robust scale before MSE/Huber loss"
    )
    parser.add_argument(
        "--input-ablation",
        type=str,
        default="full",
        choices=["full", "no-tributary", "no-upstream", "no-downstream"],
        help="Mask selected DSFNet spatial inputs while keeping the model architecture unchanged"
    )
    parser.add_argument(
        "--residual-baseline",
        type=str,
        default="none",
        choices=["none", "ha", "last"],
        help="Train DSFNet as a residual model over a target-space baseline"
    )
    parser.add_argument(
        "--residual-loss-weight",
        type=float,
        default=0.0,
        help="Weight for normalized residual supervision when --residual-baseline is enabled"
    )
    parser.add_argument(
        "--dump-test-preds",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Dump per-sample test predictions to NPZ (default: enabled)"
    )
    parser.add_argument(
        "--run-tag",
        type=str,
        default=None,
        help="Optional run tag stored in prediction NPZ metadata"
    )
    parser.add_argument(
        "--best-checkpoint-metric",
        type=str,
        default="huber",
        choices=["huber", "rmse", "balanced-rmse", "f1-hc", "f1-hf", "f1-joint", "pte", "scenario-priority"],
        help="Checkpoint selection metric from validation metrics"
    )
    parser.add_argument(
        "--checkpoint-metrics",
        nargs="+",
        default=None,
        choices=["huber", "rmse", "balanced-rmse", "f1-hc", "f1-hf", "f1-joint", "pte", "scenario-priority"],
        help="Checkpoint metrics to materialize; defaults to all metrics"
    )
    parser.add_argument(
        "--scenario-metric-weights",
        type=float,
        nargs=3,
        default=[0.4, 0.4, 0.2],
        metavar=("F1_HC", "F1_HA", "PTE"),
        help="Weights for validation scenario-priority checkpoint score"
    )
    parser.add_argument(
        "--scenario-threshold-percentile",
        type=float,
        default=SCENARIO_THRESHOLD_PERCENTILE,
        help="Train-split percentile used to derive scenario thresholds"
    )
    parser.add_argument(
        "--evaluation-stage",
        choices=["screening", "final"],
        default="final",
        help="screening evaluates checkpoints on validation only; final also evaluates test data",
    )
    parser.add_argument(
        "--multi-task-architecture",
        type=str,
        default="cross-stitch-2",
        choices=[
            "shared", "spatial-private", "shared-private",
            "separate-towers", "separate-towers-1", "separate-towers-2",
            "cross-stitch", "cross-stitch-1", "cross-stitch-2",
            "mmoe", "mmoe-2", "mmoe-4", "ple", "ple-1", "ple-2",
        ],
        help="Multi-task prediction-head architecture; default is the selected Cross-Stitch-2 structure",
    )
    parser.add_argument(
        "--decouple-space",
        action="store_true",
        help="Legacy alias for --multi-task-architecture spatial-private"
    )
    parser.add_argument(
        "--fusion-dropout",
        type=float,
        default=None,
        help="Override dropout rate in fusion networks"
    )
    parser.add_argument(
        "--dca-mode",
        type=str,
        default="adaptive",
        choices=["adaptive", "uniform", "shuffled", "shared-only"],
        help="Directional weighting mode used by DCAFusion",
    )
    return parser.parse_args()


def train_process(
        p_id: int,
        seg_id: str,
        weights: tuple[float, float],
        device: str | int,
        time_net: str,
        time_args: dict,
        space_net: str,
        space_args: dict,
        fusion_net: str,
        fusion_args: dict,
        # use_test: bool,
        epochs: int,
        optimizer: str,
        scheduler: str,
        learning_rate: float,
        weight_decay: float,
        grad_clip_norm: float | None,
        loss_weighting: str,
        fixed_task_weights: list[float] | None,
        loss_warmup_epochs: int,
        loss_warmup_fixed_task_weights: list[float] | None,
        stage_task_weights_end: list[float] | None,
        stage_start_epoch: int | None,
        stage_end_epoch: int | None,
        use_pcgrad: bool,
        pcgrad_include_mape: bool,
        criterion_type: str,
        mape_lambda: float,
        normalize_task_losses: bool,
        scenario_loss_reg_weights: list[float] | None,
        scenario_loss_event_weights: list[float] | None,
        scenario_loss_wape_weight: float,
        scenario_loss_peak_weight: float,
        scenario_loss_joint_weight: float,
        scenario_loss_focal_gamma: float,
        split_mode: str,
        seed: int,
        purge_steps: int,
        target_mode: str,
        pred_len: int,
        input_ablation: str,
        residual_baseline: str,
        residual_loss_weight: float,
        dump_test_preds: bool,
        run_tag: str | None,
        best_checkpoint_metric: str,
        checkpoint_metrics: list[str] | None,
        scenario_metric_weights: list[float],
        scenario_threshold_percentile: float,
        evaluation_stage: str,
        multi_task_architecture: str,
        decouple_space: bool,
        fusion_dropout: float | None,
        split_seed: int,
        train_ratio: float,
        valid_ratio: float,
        test_ratio: float,
    ):
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if fusion_dropout is not None:
            fusion_args = dict(fusion_args)
            fusion_args["dropout"] = fusion_dropout
        print(f"Start train: tn = {time_net}, sn = {space_net}, fn = {fusion_net}, segment = {seg_id}")
        train_loader, valid_loader, test_loader = make_loader(
            segment=seg_id,
            weights=weights,
            split_mode=split_mode,
            target_mode=target_mode,
            pred_len=pred_len,
            random_state=split_seed,
            train_ratio=train_ratio,
            valid_ratio=valid_ratio,
            test_ratio=test_ratio,
            purge_steps=purge_steps,
            input_ablation=input_ablation,
            residual_baseline=residual_baseline,
        )
        publication_meta = (
            publication_metadata_for_segment(seg_id)
            if split_mode == "final"
            else {}
        )
        train(
            p_id=p_id,
            segment=seg_id,
            weights=weights,
            train_loader=train_loader,
            valid_loader=valid_loader,
            test_loader=test_loader,
            device=device,
            target_mode=target_mode,
            prediction_dim=pred_len,
            time_net=time_net,
            time_net_kwargs=time_args,
            space_net=space_net,
            space_net_kwargs=space_args,
            fusion_net=fusion_net,
            fusion_net_kwargs=fusion_args,
            epochs=epochs,
            optimizer_type=optimizer,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            grad_clip_norm=grad_clip_norm,
            loss_weighting=loss_weighting,
            fixed_task_weights=fixed_task_weights,
            loss_warmup_epochs=loss_warmup_epochs,
            loss_warmup_fixed_task_weights=loss_warmup_fixed_task_weights,
            stage_task_weights_end=stage_task_weights_end,
            stage_start_epoch=stage_start_epoch,
            stage_end_epoch=stage_end_epoch,
            use_pcgrad=use_pcgrad,
            pcgrad_include_mape=pcgrad_include_mape,
            criterion_type=criterion_type,
            mape_lambda=mape_lambda,
            normalize_task_losses=normalize_task_losses,
            scenario_loss_reg_weights=scenario_loss_reg_weights,
            scenario_loss_event_weights=scenario_loss_event_weights,
            scenario_loss_wape_weight=scenario_loss_wape_weight,
            scenario_loss_peak_weight=scenario_loss_peak_weight,
            scenario_loss_joint_weight=scenario_loss_joint_weight,
            scenario_loss_focal_gamma=scenario_loss_focal_gamma,
            scheduler_type=scheduler,
            dump_test_preds=dump_test_preds,
            run_tag=run_tag,
            split_mode=split_mode,
            input_ablation=input_ablation,
            residual_baseline=residual_baseline,
            residual_loss_weight=residual_loss_weight,
            best_checkpoint_metric=best_checkpoint_metric,
            checkpoint_metrics=None if checkpoint_metrics is None else tuple(checkpoint_metrics),
            scenario_metric_weights=tuple(float(v) for v in scenario_metric_weights),
            scenario_threshold_percentile=scenario_threshold_percentile,
            evaluate_test=evaluation_stage == "final",
            decouple_space=decouple_space,
            multi_task_architecture=multi_task_architecture,
            multitask_head_config={},
            experiment_meta={
                "random_seed": seed,
                "split_seed": split_seed,
                "train_ratio": train_ratio,
                "valid_ratio": valid_ratio,
                "test_ratio": test_ratio,
                "purge_steps": purge_steps,
                "use_pcgrad": use_pcgrad,
                "pcgrad_include_mape": pcgrad_include_mape,
                **publication_meta,
            },
            # scheduler_type,
        )
        # clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        del train_loader
        del test_loader
        del valid_loader
    except Exception as e:
        LOG.error(f"Train segment failed: {e}")
        traceback.print_exc()
        LOG.error(f"Segment: {seg_id}, Weights: {weights}, " + \
                  f"Time Net: {time_net}, Space Net: {space_net}, Fusion Net: {fusion_net}")
        raise


def __train(
        device: str | int,
        time_net: str,
        time_args: dict,
        space_net: str,
        space_args: dict,
        fusion_net: str,
        fusion_args: dict,
        epochs: int,
        optimizer: str,
        scheduler: str,
        learning_rate: float,
        weight_decay: float,
        grad_clip_norm: float | None,
        loss_weighting: str,
        fixed_task_weights: list[float] | None,
        loss_warmup_epochs: int = 25,
        loss_warmup_fixed_task_weights: list[float] | None = None,
        stage_task_weights_end: list[float] | None = None,
        stage_start_epoch: int | None = None,
        stage_end_epoch: int | None = None,
        use_pcgrad: bool = False,
        pcgrad_include_mape: bool = False,
        criterion_type: str = "huber",
        mape_lambda: float = 0.1,
        normalize_task_losses: bool = False,
        scenario_loss_reg_weights: list[float] | None = None,
        scenario_loss_event_weights: list[float] | None = None,
        scenario_loss_wape_weight: float = 0.1,
        scenario_loss_peak_weight: float = 0.0,
        scenario_loss_joint_weight: float = 0.0,
        scenario_loss_focal_gamma: float = 2.0,
        split_mode: str = "random",
        seed: int = 42,
        purge_steps: int = 0,
        target_mode: str = "multi",
        pred_len: int = 24,
        input_ablation: str = "full",
        residual_baseline: str = "none",
        residual_loss_weight: float = 0.0,
        dump_test_preds: bool = True,
        run_tag: str | None = None,
        best_checkpoint_metric: str = "huber",
        checkpoint_metrics: list[str] | None = None,
        scenario_metric_weights: list[float] | None = None,
        scenario_threshold_percentile: float = SCENARIO_THRESHOLD_PERCENTILE,
        evaluation_stage: str = "final",
        multi_task_architecture: str = "cross-stitch-2",
        decouple_space: bool = False,
        fusion_dropout: float | None = None,
        split_seed: int = FAIR_OVERALL_PROTOCOL.split_seed,
        train_ratio: float = FAIR_OVERALL_PROTOCOL.train_ratio,
        valid_ratio: float = FAIR_OVERALL_PROTOCOL.valid_ratio,
        test_ratio: float = FAIR_OVERALL_PROTOCOL.test_ratio,
        segments_list: list[str] | None = None,
        weights_list: list[tuple] = WEIGHTS_LIST,
        multi_process: bool = False,
    ):
    with open(CONFIG_MAINSTREAM_FILE, "r", encoding="utf-8") as f:
        seg_config: dict[str, dict[str, Any]] = json.load(f)

    time_net = time_net.replace("-", "_")
    space_net = space_net.replace("-", "_")
    fusion_net = fusion_net.replace("-", "_")

    if segments_list is not None:
        segments = segments_list
    else:
        segments = []
        for seg_id, seg_info in seg_config.items():
            if seg_id.startswith(MAINSTREAM_ID_PREFIX) and seg_info["used"]:
                segments.append(seg_id)

    process_step = 3 if multi_process else 1
    if len(weights_list) > 1:
        for seg_id in segments:
            for i in range(0, len(weights_list), process_step):
                processes: list[Process] = list()
                p_id: int = 0
                for weights in weights_list[i: i + process_step]:
                    p_id += 1
                    p = Process(
                        target=train_process,
                        args=(
                            p_id, seg_id, weights, device, time_net, time_args,
                            space_net, space_args, fusion_net, fusion_args,
                            epochs, optimizer, scheduler, learning_rate, weight_decay,
                            grad_clip_norm, loss_weighting, fixed_task_weights,
                            loss_warmup_epochs, loss_warmup_fixed_task_weights,
                            stage_task_weights_end, stage_start_epoch, stage_end_epoch,
                            use_pcgrad, pcgrad_include_mape, criterion_type,
                            mape_lambda, normalize_task_losses,
                            scenario_loss_reg_weights, scenario_loss_event_weights,
                            scenario_loss_wape_weight, scenario_loss_peak_weight,
                            scenario_loss_joint_weight,
                            scenario_loss_focal_gamma, split_mode, seed, purge_steps, target_mode,
                            pred_len, input_ablation, residual_baseline, residual_loss_weight,
                            dump_test_preds, run_tag, best_checkpoint_metric,
                            checkpoint_metrics,
                            scenario_metric_weights if scenario_metric_weights is not None else [0.4, 0.4, 0.2],
                            scenario_threshold_percentile,
                            evaluation_stage,
                            multi_task_architecture,
                            decouple_space,
                            fusion_dropout,
                            split_seed,
                            train_ratio,
                            valid_ratio,
                            test_ratio,
                        )
                    )
                    p.start()
                    processes.append(p)
                for p in processes:
                    p.join()
                    if p.exitcode != 0:
                        raise RuntimeError(f"Training subprocess failed with exit code {p.exitcode} for segment {seg_id}")
    else:
        for weights in weights_list:
            for i in range(0, len(segments), process_step):
                processes: list[Process] = list()
                p_id: int = 0
                for seg_id in segments[i: i + process_step]:
                    p_id += 1
                    p = Process(
                        target=train_process,
                        args=(
                            p_id, seg_id, weights, device, time_net, time_args,
                            space_net, space_args, fusion_net, fusion_args,
                            epochs, optimizer, scheduler, learning_rate, weight_decay,
                            grad_clip_norm, loss_weighting, fixed_task_weights,
                            loss_warmup_epochs, loss_warmup_fixed_task_weights,
                            stage_task_weights_end, stage_start_epoch, stage_end_epoch,
                            use_pcgrad, pcgrad_include_mape, criterion_type,
                            mape_lambda, normalize_task_losses,
                            scenario_loss_reg_weights, scenario_loss_event_weights,
                            scenario_loss_wape_weight, scenario_loss_peak_weight,
                            scenario_loss_joint_weight,
                            scenario_loss_focal_gamma, split_mode, seed, purge_steps, target_mode,
                            pred_len, input_ablation, residual_baseline, residual_loss_weight,
                            dump_test_preds, run_tag, best_checkpoint_metric,
                            checkpoint_metrics,
                            scenario_metric_weights if scenario_metric_weights is not None else [0.4, 0.4, 0.2],
                            scenario_threshold_percentile,
                            evaluation_stage,
                            multi_task_architecture,
                            decouple_space,
                            fusion_dropout,
                            split_seed,
                            train_ratio,
                            valid_ratio,
                            test_ratio,
                        )
                    )
                    p.start()
                    processes.append(p)
                for p in processes:
                    p.join()
                    if p.exitcode != 0:
                        raise RuntimeError(f"Training subprocess failed with exit code {p.exitcode} for segment {seg_id}")


def train_single_combination(args: argparse.Namespace):
    if args.time_net == "lstm":
        time_args = dict(TIME_LSTM_ARGS)
    elif args.time_net == "attention-lstm":
        time_args = dict(TIME_ATTENTION_LSTM_ARGS)
    elif args.time_net == "timesnet":
        time_args = dict(TIME_TIMESNET_ARGS)
    else:
        raise ValueError(f"Unknown time net type: {args.time_net}")

    if args.space_net == "encoder":
        space_args = dict(SPACE_ENCODER_ARGS)
    elif args.space_net == "cross-attention":
        space_args = dict(SPACE_CROSS_ATTENTION_ARGS)
    elif args.space_net == "spatial-conv":
        space_args = dict(SPACE_SPATIAL_CONV_ARGS)
    else:
        raise ValueError(f"Unknown space net type: {args.space_net}")
    
    if args.fusion_net == "weighted-concat":
        fusion_args = dict(FUSION_WEIGHTED_CONCAT_ARGS)
    elif args.fusion_net == "weighted-fusion":
        fusion_args = dict(FUSION_WEIGHTED_FUSION_ARGS)
    elif args.fusion_net == "feature-mapping":
        fusion_args = dict(FUSION_FEATURE_MAPPING_ARGS)
    elif args.fusion_net == "cross-attention":
        fusion_args = dict(FUSION_CROSS_ATTENTION_ARGS)
    elif args.fusion_net == "self-attention":
        fusion_args = dict(FUSION_SELF_ATTENTION_ARGS)
    elif args.fusion_net == "dcp-fusion":
        fusion_args = dict(FUSION_DCP_FUSION_ARGS)
    elif args.fusion_net == "concat-mlp":
        fusion_args = dict(FUSION_CONCAT_MLP_ARGS)
    elif args.fusion_net == "gmu":
        fusion_args = dict(FUSION_GMU_ARGS)
    else:
        raise ValueError(f"Unknown fusion net type: {args.fusion_net}")

    if args.fusion_net == "dcp-fusion":
        fusion_args["direction_mode"] = args.dca_mode
        fusion_args["direction_shuffle_seed"] = args.seed

    time_args["feature_num"] = args.pred_len
    space_args["feature_num"] = args.pred_len

    if args.weights:
        weights_list = WEIGHTS_LIST
    else:
        weights_list = [(args.aw, args.vw)]

    device = args.device if torch.cuda.is_available() else "cpu"
    epochs = int(args.epochs)
    segment_list = [args.segment] if args.segment else None

    __train(
        device=device,
        time_net=args.time_net,
        time_args=time_args,
        space_net=args.space_net,
        space_args=space_args,
        fusion_net=args.fusion_net,
        fusion_args=fusion_args,
        epochs=epochs,
        optimizer=args.optimizer,
        scheduler=args.scheduler,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip,
        loss_weighting=args.loss_weighting,
        fixed_task_weights=args.task_weights,
        loss_warmup_epochs=args.loss_warmup_epochs,
        loss_warmup_fixed_task_weights=args.loss_warmup_task_weights,
        stage_task_weights_end=args.task_weights_end,
        stage_start_epoch=args.stage_start_epoch,
        stage_end_epoch=args.stage_end_epoch,
        use_pcgrad=args.use_pcgrad,
        pcgrad_include_mape=args.pcgrad_include_mape,
        criterion_type=args.criterion,
        mape_lambda=args.mape_lambda,
        normalize_task_losses=args.normalize_task_losses,
        scenario_loss_reg_weights=args.scenario_loss_reg_weights,
        scenario_loss_event_weights=args.scenario_loss_event_weights,
        scenario_loss_wape_weight=args.scenario_loss_wape_weight,
        scenario_loss_peak_weight=args.scenario_loss_peak_weight,
        scenario_loss_joint_weight=args.scenario_loss_joint_weight,
        scenario_loss_focal_gamma=args.scenario_loss_focal_gamma,
        split_mode=args.split_mode,
        seed=args.seed,
        purge_steps=args.purge_steps,
        target_mode=args.target_mode,
        pred_len=args.pred_len,
        input_ablation=args.input_ablation,
        residual_baseline=args.residual_baseline,
        residual_loss_weight=args.residual_loss_weight,
        dump_test_preds=args.dump_test_preds,
        run_tag=args.run_tag,
        best_checkpoint_metric=args.best_checkpoint_metric,
        checkpoint_metrics=args.checkpoint_metrics,
        scenario_metric_weights=args.scenario_metric_weights,
        scenario_threshold_percentile=args.scenario_threshold_percentile,
        evaluation_stage=args.evaluation_stage,
        multi_task_architecture=args.multi_task_architecture,
        segments_list=segment_list,
        weights_list=weights_list,
        multi_process=args.multi_process,
        decouple_space=args.decouple_space,
        fusion_dropout=args.fusion_dropout,
        split_seed=args.split_seed,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
    )


def train_custom_combination(args: argparse.Namespace):
    time_net_pair = {
        "lstm":           TIME_LSTM_ARGS,
        # "attention-lstm": TIME_ATTENTION_LSTM_ARGS,
        # "timesnet":       TIME_TIMESNET_ARGS,
    }
    space_net_pair = {
        # "encoder":         SPACE_ENCODER_ARGS,
        # "cross-attention": SPACE_CROSS_ATTENTION_ARGS,
        "spatial-conv":    SPACE_SPATIAL_CONV_ARGS,
    }
    fusion_net_pair = {
        # "weighted-fusion": FUSION_WEIGHTED_FUSION_ARGS,
        "cross-attention": FUSION_CROSS_ATTENTION_ARGS,
        # "self-attention":  FUSION_SELF_ATTENTION_ARGS,
    }
    if args.loss_weighting == "fixed":
        weights_list = [(0.5, 0.5)]
    elif args.weights:
        weights_list = WEIGHTS_LIST
    else:
        weights_list = [(0.3, 0.7), (0.5, 0.5), (0.7, 0.3)]
        # weights_list = [(0.5, 0.5)]
    segment_list = ["M-02", "M-04", "M-07", "M-08", "M-10"]
    device = args.device if torch.cuda.is_available() else "cpu"
    epochs = int(args.epochs)
    for time_net, time_args in time_net_pair.items():
        for space_net, space_args in space_net_pair.items():
            for fusion_net, fusion_args in fusion_net_pair.items():
                time_args = dict(time_args)
                space_args = dict(space_args)
                fusion_args = dict(fusion_args)
                time_args["feature_num"] = args.pred_len
                space_args["feature_num"] = args.pred_len
                __train(
                    device=device,
                    time_net=time_net,
                    time_args=time_args,
                    space_net=space_net,
                    space_args=space_args,
                    fusion_net=fusion_net,
                    fusion_args=fusion_args,
                    epochs=epochs,
                    optimizer=args.optimizer,
                    scheduler=args.scheduler,
                    learning_rate=args.lr,
                    weight_decay=args.weight_decay,
                    grad_clip_norm=args.grad_clip,
                    loss_weighting=args.loss_weighting,
                    fixed_task_weights=args.task_weights,
                    loss_warmup_epochs=args.loss_warmup_epochs,
                    loss_warmup_fixed_task_weights=args.loss_warmup_task_weights,
                    use_pcgrad=args.use_pcgrad,
                    pcgrad_include_mape=args.pcgrad_include_mape,
                    criterion_type=args.criterion,
                    scenario_loss_reg_weights=args.scenario_loss_reg_weights,
                    scenario_loss_event_weights=args.scenario_loss_event_weights,
                    scenario_loss_wape_weight=args.scenario_loss_wape_weight,
                    scenario_loss_peak_weight=args.scenario_loss_peak_weight,
                    scenario_loss_joint_weight=args.scenario_loss_joint_weight,
                    scenario_loss_focal_gamma=args.scenario_loss_focal_gamma,
                    split_mode=args.split_mode,
                    target_mode=args.target_mode,
                    pred_len=args.pred_len,
                    input_ablation=args.input_ablation,
                    residual_baseline=args.residual_baseline,
                    residual_loss_weight=args.residual_loss_weight,
                    dump_test_preds=args.dump_test_preds,
                    run_tag=args.run_tag,
                    best_checkpoint_metric=args.best_checkpoint_metric,
                    scenario_metric_weights=args.scenario_metric_weights,
                    scenario_threshold_percentile=args.scenario_threshold_percentile,
                    evaluation_stage=args.evaluation_stage,
                    multi_task_architecture=args.multi_task_architecture,
                    segments_list=segment_list,
                    weights_list=weights_list,
                    multi_process=args.multi_process,
                    decouple_space=args.decouple_space,
                    fusion_dropout=args.fusion_dropout,
                    split_seed=args.split_seed,
                    train_ratio=args.train_ratio,
                    valid_ratio=args.valid_ratio,
                    test_ratio=args.test_ratio,
                )


def train_all_combinations(args: argparse.Namespace):
    time_net_pair = {
        "lstm":           TIME_LSTM_ARGS,
        "attention-lstm": TIME_ATTENTION_LSTM_ARGS,
        "timesnet":       TIME_TIMESNET_ARGS,
    }
    space_net_pair = {
        "encoder":         SPACE_ENCODER_ARGS,
        "cross-attention": SPACE_CROSS_ATTENTION_ARGS,
        "spatial-conv":    SPACE_SPATIAL_CONV_ARGS,
    }
    fusion_net_pair = {
        # "weighted-concat": FUSION_WEIGHTED_CONCAT_ARGS,
        "weighted-fusion": FUSION_WEIGHTED_FUSION_ARGS,
        # "feature-mapping": FUSION_FEATURE_MAPPING_ARGS,
        "cross-attention": FUSION_CROSS_ATTENTION_ARGS,
        "self-attention":  FUSION_SELF_ATTENTION_ARGS,
    }
    if args.loss_weighting == "fixed":
        weights_list = [(0.5, 0.5)]
    elif args.weights:
        weights_list = WEIGHTS_LIST
    else:
        weights_list = [(0.3, 0.7), (0.5, 0.5), (0.7, 0.3)]
    device = args.device if torch.cuda.is_available() else "cpu"
    epochs = int(args.epochs)
    for time_net, time_args in time_net_pair.items():
        for space_net, space_args in space_net_pair.items():
            for fusion_net, fusion_args in fusion_net_pair.items():
                time_args = dict(time_args)
                space_args = dict(space_args)
                fusion_args = dict(fusion_args)
                time_args["feature_num"] = args.pred_len
                space_args["feature_num"] = args.pred_len
                __train(
                    device=device,
                    time_net=time_net,
                    time_args=time_args,
                    space_net=space_net,
                    space_args=space_args,
                    fusion_net=fusion_net,
                    fusion_args=fusion_args,
                    epochs=epochs,
                    optimizer=args.optimizer,
                    scheduler=args.scheduler,
                    learning_rate=args.lr,
                    weight_decay=args.weight_decay,
                    grad_clip_norm=args.grad_clip,
                    loss_weighting=args.loss_weighting,
                    fixed_task_weights=args.task_weights,
                    loss_warmup_epochs=args.loss_warmup_epochs,
                    loss_warmup_fixed_task_weights=args.loss_warmup_task_weights,
                    use_pcgrad=args.use_pcgrad,
                    pcgrad_include_mape=args.pcgrad_include_mape,
                    criterion_type=args.criterion,
                    scenario_loss_reg_weights=args.scenario_loss_reg_weights,
                    scenario_loss_event_weights=args.scenario_loss_event_weights,
                    scenario_loss_wape_weight=args.scenario_loss_wape_weight,
                    scenario_loss_peak_weight=args.scenario_loss_peak_weight,
                    scenario_loss_joint_weight=args.scenario_loss_joint_weight,
                    scenario_loss_focal_gamma=args.scenario_loss_focal_gamma,
                    split_mode=args.split_mode,
                    target_mode=args.target_mode,
                    pred_len=args.pred_len,
                    input_ablation=args.input_ablation,
                    residual_baseline=args.residual_baseline,
                    residual_loss_weight=args.residual_loss_weight,
                    dump_test_preds=args.dump_test_preds,
                    run_tag=args.run_tag,
                    best_checkpoint_metric=args.best_checkpoint_metric,
                    scenario_metric_weights=args.scenario_metric_weights,
                    scenario_threshold_percentile=args.scenario_threshold_percentile,
                    evaluation_stage=args.evaluation_stage,
                    multi_task_architecture=args.multi_task_architecture,
                    weights_list=weights_list,
                    multi_process=args.multi_process,
                    decouple_space=args.decouple_space,
                    fusion_dropout=args.fusion_dropout,
                    split_seed=args.split_seed,
                    train_ratio=args.train_ratio,
                    valid_ratio=args.valid_ratio,
                    test_ratio=args.test_ratio,
                )


if __name__ == "__main__":
    try:
        set_start_method("spawn")
    except RuntimeError:
        pass

    args = parse_args()

    if args.all:
        train_all_combinations(args)
    elif args.custom:
        train_custom_combination(args)
    else:
        train_single_combination(args)
