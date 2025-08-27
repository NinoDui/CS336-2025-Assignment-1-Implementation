import os

# Common
DEFAULT_ENV_LOG_CONIFG = "LOG_CFG"

# IO
DEFAULT_MAX_NUM_PROCESSES = os.cpu_count() or 1
SENTINEL = None
DEFAULT_MAX_CHUNK_SIZE = (1 << 10) << 10  # 1MB
DEFAULT_MIN_CHUNK_SIZE = (1 << 10) << 8  # 256KB

# Tokenization
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
DEFAULT_TOKEN_SPLIT = "<|endoftext|>"
DEFAULT_BYTE_SIZE = 256

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
