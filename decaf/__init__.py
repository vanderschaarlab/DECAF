import sys

from . import logger  # noqa: F401
from .data import DataModule  # noqa: F401
from .DECAF import DECAF  # noqa: F401

logger.add(sink=sys.stderr, level="CRITICAL")
