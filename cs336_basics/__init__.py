import importlib.metadata

from .common.logging import setup_logging

setup_logging()

__version__ = importlib.metadata.version("cs336_basics")
