try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from .training import train
from .environment import Env

__all__ = [
    "train",
    "Env",
]