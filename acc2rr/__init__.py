"""ACC2RR respiratory-rate estimation package."""

from .core import Config
from .pipeline import analyze_recording

__all__ = ["Config", "analyze_recording"]
