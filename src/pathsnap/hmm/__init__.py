"""HMM package: emissions, transitions, Viterbi decoder."""
from .viterbi import decode  # noqa: F401

__all__ = ["decode"]
