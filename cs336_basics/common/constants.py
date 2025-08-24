# Common
DEFAULT_ENV_LOG_CONIFG = "LOG_CFG"

# Tokenization
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
DEFAULT_TOKEN_SPLIT = "<|endoftext|>"


# Module
VALID_DEVICE = ["cpu", "cuda", "mps"]
VALID_DTYPE = ["float32", "bfloat16", "float8"]  # avoid float16
DEFAULT_DEVICE = "cpu"
DEFAULT_DTYPE = "float32"

DEFAULT_ROPE_THETA = 10_000
DEFAULT_MAX_SEQ_LEN = 2048


# Pipeline
DEFAULT_MODEL_STATE_DICT_KEY = "model_state_dict"
DEFAULT_OPTIMIZER_STATE_DICT_KEY = "optimizer_state_dict"
DEFAULT_ITERATION_KEY = "iteration"
