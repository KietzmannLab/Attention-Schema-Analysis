try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from .training_with_as import training_with_as
from .training_without_as import training_without_as
from .environment import Train
from .emergent_environment import Train
from .emergent_training_with_as import emergent_training_with_as

__all__ = [
    "training_with_as",
    "training_with_as_no_ar",
    "training_without_as",
    "training_without_as_no_ar",
    "Train",
]