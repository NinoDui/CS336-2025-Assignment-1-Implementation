import logging.config
import os

import yaml

from .constants import DEFAULT_ENV_LOG_CONIFG


def setup_logging(path="logging.yaml", env_key=DEFAULT_ENV_LOG_CONIFG):
    env_value = os.getenv(env_key, None)
    if env_value is not None:
        path = env_value

    if os.path.exists(path):
        with open(path, "rt") as f:
            cfg = yaml.safe_load(f.read())
        logging.config.dictConfig(cfg)
    else:
        logging.basicConfig()
