# coding: utf-8
import copy

import torch
from torch import nn

from fusion.advance.dcp_fusion import DCPFusion
from fusion.simple.cross_attention import CrossAttention
from fusion.simple.feature_mapping import FeatureMapping
from fusion.simple.self_attention import SelfAttention
from fusion.simple.weighted_concat import WeightedConcat
from fusion.simple.weighted_fusion import WeightedFusion
from timenet.lstm import TimeFeatureLSTM
from timenet.attention_lstm import AttentionLSTM
from timenet.timesnet import TimesNet
from spacenet.encoder import SpaceFeatureEncoder
from spacenet.cross_attention import CrossAttentionEncoder
from spacenet.spatial_conv import SpatialConv
from sim.prediction_head import (
    CrossStitchPredictionHead,
    MMoEPredictionHead,
    PLEPredictionHead,
    PredictionHead,
    SeparateTowersPredictionHead,
    TaskSpecificPredictionHead,
)


MULTI_TASK_ARCHITECTURES = (
    "shared",
    "spatial-private",
    "shared-private",
    "separate-towers",
    "cross-stitch",
    "mmoe",
    "ple",
    "separate-towers-1",
    "separate-towers-2",
    "cross-stitch-1",
    "cross-stitch-2",
    "mmoe-2",
    "mmoe-4",
    "ple-1",
    "ple-2",
)
DECOUPLED_SPATIAL_ARCHITECTURES = ("spatial-private", "shared-private")


class DSFNet(nn.Module):
    def __init__(
            self,
            time_kwargs: dict,
            space_kwargs: dict,
            fusion_kwargs: dict,
            time_net: str = ["lstm", "attention_lstm", "timesnet"][0],
            space_net: str = ["encoder", "cross-attention", "spatial_conv"][0],
            fusion_net: str = [
                "weighted_concat", 
                "weighted_fusion", 
                "feature_mapping", 
                "cross-attention", 
                "self_attention",
                "dcp_fusion",
                "concat_mlp",
                "gmu",
            ][0],
            task_num: int = 2,
            output_dim: int = 24,
            decouple_space: bool = False,
            multi_task_architecture: str | None = None,
            multitask_head_config: dict | None = None,
        ):
        super().__init__()
        self.name = "DSFNet"
        if multi_task_architecture is None:
            multi_task_architecture = "spatial-private" if decouple_space else "cross-stitch-2"
        if multi_task_architecture not in MULTI_TASK_ARCHITECTURES:
            raise ValueError(
                "multi_task_architecture must be one of "
                f"{list(MULTI_TASK_ARCHITECTURES)}"
            )
        if decouple_space:
            if multi_task_architecture == "shared":
                multi_task_architecture = "spatial-private"
            elif multi_task_architecture != "spatial-private":
                raise ValueError(
                    "decouple_space is a legacy alias for "
                    "multi_task_architecture='spatial-private'"
                )
        if multi_task_architecture != "shared" and task_num != 2:
            raise ValueError(
                f"{multi_task_architecture} requires exactly two prediction tasks"
            )

        self.multi_task_architecture = multi_task_architecture
        self.multitask_head_config = dict(multitask_head_config or {})
        self.decouple_space = multi_task_architecture in DECOUPLED_SPATIAL_ARCHITECTURES

        if time_net == "lstm":
            self.time_model = TimeFeatureLSTM(
                input_dim=time_kwargs["input_dim"],
                hidden_dim=time_kwargs["hidden_dim"],
                num_layers=time_kwargs["num_layers"],
                output_num=time_kwargs["feature_num"],
                output_dim=time_kwargs["feature_dim"],
            )
        elif time_net == "attention_lstm":
            self.time_model = AttentionLSTM(
                input_dim=time_kwargs["input_dim"],
                hidden_dim=time_kwargs["hidden_dim"],
                num_layers=time_kwargs["num_layers"],
                output_num=time_kwargs["feature_num"],
                output_dim=time_kwargs["feature_dim"],
                bidirectional=time_kwargs["bidirectional"],
            )
        elif time_net == "timesnet":
            self.time_model = TimesNet(
                input_dim=time_kwargs["input_dim"],
                hidden_dim=time_kwargs["hidden_dim"],
                num_layers=time_kwargs["num_layers"],
                output_num=time_kwargs["feature_num"],
                output_dim=time_kwargs["feature_dim"],
                top_k=time_kwargs["top_k"],
                num_kernels=time_kwargs["num_kernels"],
                dropout=time_kwargs["dropout"],
            )

        if space_net == "encoder":
            self.space_model = SpaceFeatureEncoder(
                input_dim=space_kwargs["input_dim"],
                hidden_dim=space_kwargs["hidden_dim"],
                num_layers=space_kwargs["num_layers"],
                num_heads=space_kwargs["num_heads"],
                output_num=space_kwargs["feature_num"],
                output_dim=space_kwargs["feature_dim"],
            )
        elif space_net == "cross_attention":
            self.space_model = CrossAttentionEncoder(
                input_dim=space_kwargs["input_dim"],
                hidden_dim=space_kwargs["hidden_dim"],
                num_layers=space_kwargs["num_layers"],
                num_heads=space_kwargs["num_heads"],
                output_num=space_kwargs["feature_num"],
                output_dim=space_kwargs["feature_dim"],
            )
        elif space_net == "spatial_conv":
            self.space_model = SpatialConv(
                input_dim=space_kwargs["input_dim"],
                hidden_dim=space_kwargs["hidden_dim"],
                kernel_size=space_kwargs["kernel_size"],
                output_num=space_kwargs["feature_num"],
                output_dim=space_kwargs["feature_dim"],
            )

        if fusion_net == "weighted_concat":
            self.fusion = WeightedConcat(
                time_dim=time_kwargs["feature_dim"],
                space_dim=space_kwargs["feature_dim"],
                output_dim=fusion_kwargs["feature_dim"],
            )
        elif fusion_net == "weighted_fusion":
            self.fusion = WeightedFusion(
                fusion_way=fusion_kwargs["fusion_way"],
                time_dim=time_kwargs["feature_dim"],
                space_dim=space_kwargs["feature_dim"],
                output_dim=fusion_kwargs["feature_dim"],
            )
        elif fusion_net == "feature_mapping":
            self.fusion = FeatureMapping(
                fusion_way=fusion_kwargs["fusion_way"],
                time_dim=time_kwargs["feature_dim"],
                space_dim=space_kwargs["feature_dim"],
                mapping_dim=fusion_kwargs["mapping_dim"],
            )
        elif fusion_net == "cross_attention":
            self.fusion = CrossAttention(
                fusion_way=fusion_kwargs["fusion_way"],
                time_dim=time_kwargs["feature_dim"],
                space_dim=space_kwargs["feature_dim"],
                output_dim=fusion_kwargs["feature_dim"],
            )
        elif fusion_net == "self_attention":
            self.fusion = SelfAttention(
                time_dim=time_kwargs["feature_dim"],
                space_dim=space_kwargs["feature_dim"],
            )
        elif fusion_net == "dcp_fusion":
            self.fusion = DCPFusion(
                time_dim=time_kwargs["feature_dim"],
                space_dim=space_kwargs["feature_dim"],
                direction_input_dim=fusion_kwargs["direction_input_dim"],
                hidden_dim=fusion_kwargs["hidden_dim"],
                output_dim=fusion_kwargs["feature_dim"],
                pressure_eta=fusion_kwargs["pressure_eta"],
                dropout=fusion_kwargs["dropout"],
                direction_mode=fusion_kwargs.get("direction_mode", "adaptive"),
                direction_shuffle_seed=fusion_kwargs.get("direction_shuffle_seed", 42),
            )

        if self.decouple_space:
            self.space_model_2 = copy.deepcopy(self.space_model)
            self.fusion_2 = copy.deepcopy(self.fusion)

        if self.multi_task_architecture == "shared-private":
            self.predition = TaskSpecificPredictionHead(
                input_dim=self.fusion.output_dim,
                task_num=task_num,
                output_dim=output_dim,
            )
        elif self.multi_task_architecture in {"separate-towers", "separate-towers-1", "separate-towers-2"}:
            self.predition = SeparateTowersPredictionHead(
                input_dim=self.fusion.output_dim,
                task_num=task_num,
                output_dim=output_dim,
                tower_depth=2 if self.multi_task_architecture == "separate-towers-2" else 1,
            )
        elif self.multi_task_architecture in {"cross-stitch", "cross-stitch-1", "cross-stitch-2"}:
            self.predition = CrossStitchPredictionHead(
                input_dim=self.fusion.output_dim,
                task_num=task_num,
                output_dim=output_dim,
                num_stitch_layers=2 if self.multi_task_architecture == "cross-stitch-2" else 1,
            )
        elif self.multi_task_architecture in {"mmoe", "mmoe-2", "mmoe-4"}:
            self.predition = MMoEPredictionHead(
                input_dim=self.fusion.output_dim,
                task_num=task_num,
                output_dim=output_dim,
                num_experts=2 if self.multi_task_architecture == "mmoe-2" else 4,
            )
        elif self.multi_task_architecture in {"ple", "ple-1", "ple-2"}:
            self.predition = PLEPredictionHead(
                input_dim=self.fusion.output_dim,
                task_num=task_num,
                output_dim=output_dim,
                num_shared_experts=2,
                num_private_experts=1,
                num_levels=2 if self.multi_task_architecture == "ple-2" else 1,
            )
        else:
            self.predition = PredictionHead(
                input_dim=self.fusion.output_dim,
                task_num=task_num,
                output_dim=output_dim,
            )
    

    def forward(
            self,
            cur_feat: torch.Tensor,
            up_feat: torch.Tensor,
            down_feat: torch.Tensor,
            in_feat: torch.Tensor,
            out_feat: torch.Tensor,
        ) -> torch.Tensor:
        time_feature: torch.Tensor = self.time_model(cur_feat)
        if self.decouple_space:
            space_feature_1 = self.space_model(down_feat, up_feat, in_feat, out_feat)
            space_feature_2 = self.space_model_2(down_feat, up_feat, in_feat, out_feat)
            if isinstance(self.fusion, DCPFusion):
                fused_1 = self.fusion(time_feature, space_feature_1, down_feat, up_feat, in_feat, out_feat)
                fused_2 = self.fusion_2(time_feature, space_feature_2, down_feat, up_feat, in_feat, out_feat)
            else:
                fused_1 = self.fusion(time_feature, space_feature_1)
                fused_2 = self.fusion_2(time_feature, space_feature_2)

            if self.multi_task_architecture == "shared-private":
                return self.predition((fused_1, fused_2))

            # Apply the shared stem once so BatchNorm receives both tasks in a
            # single, order-independent update before the task towers split.
            batch_sizes = (fused_1.shape[0], fused_2.shape[0])
            shared_input = torch.cat((fused_1, fused_2), dim=0).permute(0, 2, 1)
            shared_output = self.predition.shared(shared_input).permute(0, 2, 1)
            x1, x2 = torch.split(shared_output, batch_sizes, dim=0)

            # Predict Task-1 (Crowding)
            t1 = self.predition.towers[0](x1).permute(0, 2, 1)
            pred_1 = self.predition.preds[0](t1).permute(0, 2, 1).squeeze(-1)

            # Predict Task-2 (Flow)
            t2 = self.predition.towers[1](x2).permute(0, 2, 1)
            pred_2 = self.predition.preds[1](t2).permute(0, 2, 1).squeeze(-1)
            
            return torch.stack([pred_1, pred_2], dim=2)
        else:
            space_feature: torch.Tensor = self.space_model(down_feat, up_feat, in_feat, out_feat)
            if isinstance(self.fusion, DCPFusion):
                fused_feature: torch.Tensor = self.fusion(
                    time_feature,
                    space_feature,
                    down_feat,
                    up_feat,
                    in_feat,
                    out_feat,
                )
            else:
                fused_feature: torch.Tensor = self.fusion(time_feature, space_feature)
            output: torch.Tensor = self.predition(fused_feature)
            return output
