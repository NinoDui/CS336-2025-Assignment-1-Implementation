import importlib.metadata

__version__ = importlib.metadata.version("cs336_basics")

from .common import logging

logging.setup_logging()
