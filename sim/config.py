# coding: utf-8
import os


# static arguments

WEIGHTS_LIST = [
    (0.2, 0.8),
    (0.3, 0.7),
    (0.4, 0.6),
    (0.5, 0.5),
    (0.6, 0.4),
    (0.7, 0.3),
    (0.8, 0.2),
]

CONFIG_MAINSTREAM_FILE = "./datasets/segment/mainstream.json"
CONFIG_TRIBUTARY_FILE  = "./datasets/segment/tributary.json"

MAINSTREAM_ID_PREFIX = "M-"
TRIBUTARY_ID_PREFIX  = "T-"

MATRICES_DIR: str = os.environ.get("MATRICES_DIR", "./datasets/matrices/")

MODEL_WEIGHTS_DIR: str = os.environ.get(
    "MODEL_WEIGHTS_DIR",
    "/root/autodl-tmp/CrowdingShippingPath-newdef/models/"
    # "./models/"
)

# The final manuscript protocol uses training-split P70 event thresholds for
# both the scenario-aware loss and validation-based checkpoint selection.  An
# environment override is retained for lightweight smoke tests.
SCENARIO_THRESHOLD_PERCENTILE: float = float(
    os.environ.get("SCENARIO_THRESHOLD_PERCENTILE", "70.0")
)

# Checkpoints are part of the auditable final protocol: the selected validation
# checkpoint, rather than the last in-memory epoch, supplies the test output.
# Set SAVE_MODEL_WEIGHTS=0 only for explicitly non-publication smoke tests.
SAVE_MODEL_WEIGHTS: bool = os.environ.get(
    "SAVE_MODEL_WEIGHTS", "1"
).strip().lower() not in {"0", "false", "no", "off"}


# customized arguments

# use last 10% data as test set
TEST_SCALE_RATIO  = 0.1
# use 70% of remaining data as train set, 30% as valid set
TRAIN_SCALE_RATIO = 0.7
VALID_SCALE_RATIO = 0.3

# the dimension of input (7 days = 168 hours)
TIME_INPUT_DIM  = 26
SPACE_INPUT_DIM = 2

# the dimension of time/space feature
FEATURE_DIM = 128

# the dimension of output (24 hour)
OUTPUT_DIM = 24

# arguments of time feature net
TIME_LSTM_ARGS = {
    "input_dim":     TIME_INPUT_DIM + 2,
    "hidden_dim":    256,
    "num_layers":    2,
    # "seq_len":       TIME_HISTORY_STEP,
    "feature_num":   OUTPUT_DIM,
    "feature_dim":   FEATURE_DIM,
}
TIME_ATTENTION_LSTM_ARGS = {
    "input_dim":     TIME_INPUT_DIM + 2,
    "hidden_dim":    256,
    "num_layers":    2,
    # "seq_len":       TIME_HISTORY_STEP,
    "feature_num":   OUTPUT_DIM,
    "feature_dim":   FEATURE_DIM,
    "bidirectional": True,
}
TIME_TIMESNET_ARGS = {
    # TimesNet appends two positional channels in forward, like the recurrent time nets.
    "input_dim":     TIME_INPUT_DIM + 2,
    "hidden_dim":    64,
    "num_layers":    2,    # num of time block layers
    # "seq_len":       TIME_HISTORY_STEP,
    "feature_num":   OUTPUT_DIM,
    "feature_dim":   FEATURE_DIM,
    # arguments below can be fixed
    "top_k":         3,    # select top_k periods in fft
    "num_kernels":   6,    # num of conv kernels in inception block
    "dropout":       0.1
}

# arguments of space feature net
SPACE_ENCODER_ARGS = {
    "input_dim":   4 * SPACE_INPUT_DIM,  # 4 types of spatial features will be concatenated
    "hidden_dim":  256,
    "num_layers":  2,
    "num_heads":   8,
    # "seq_len":     SPACE_HISTORY_STEP,
    "feature_num": OUTPUT_DIM,
    "feature_dim": FEATURE_DIM,
}
SPACE_CROSS_ATTENTION_ARGS = {
    "input_dim":   SPACE_INPUT_DIM,
    "hidden_dim":  256,
    "num_layers":  2,
    "num_heads":   8,
    # "seq_len":     SPACE_HISTORY_STEP,
    "feature_num": OUTPUT_DIM,
    "feature_dim": FEATURE_DIM,
}
SPACE_SPATIAL_CONV_ARGS = {
    "input_dim":   SPACE_INPUT_DIM,
    "hidden_dim":  256,
    "kernel_size": 3,    # integer only
    # "seq_len":     SPACE_HISTORY_STEP,
    "feature_num": OUTPUT_DIM,
    "feature_dim": FEATURE_DIM,
}

# arguments of fusion net
FUSION_WEIGHTED_CONCAT_ARGS = {
    "feature_dim": 256
}
FUSION_WEIGHTED_FUSION_ARGS = {
    "feature_dim": 256,
    # choice: [
    #   "plus", 
    #   "multiply"
    # ]
    "fusion_way":  "multiply"
}
FUSION_FEATURE_MAPPING_ARGS = {
    "mapping_dim": 128,
    "feature_dim": 256,
    # choice: [
    #   "plus", 
    #   "multiply", 
    #   "concat"
    # ]
    "fusion_way":  "concat"
}
FUSION_CROSS_ATTENTION_ARGS = {
    "feature_dim": 256,
    # choice: [
    #   "plus", 
    #   "multiply", 
    #   "concat"
    # ]
    "fusion_way":  "concat"
}
FUSION_SELF_ATTENTION_ARGS = {
    "feature_dim": 256,
}
FUSION_DCP_FUSION_ARGS = {
    "direction_input_dim": SPACE_INPUT_DIM,
    "hidden_dim":          128,
    "feature_dim":         256,
    "pressure_eta":        1.0,
    "dropout":             0.1,
}
FUSION_CONCAT_MLP_ARGS = {
    "hidden_dim":  256,
    "feature_dim": 256,
}
FUSION_GMU_ARGS = {
    "feature_dim": 256,
}
