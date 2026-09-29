"""Explicit fan-in/fan-out based weight initialization."""

import numpy as np
from numpy.typing import NDArray


def initialize(
    fan_in: int,
    fan_out: int,
    strategy: str = "he",
    rng: np.random.Generator | None = None,
    dtype: str = "float64",
) -> NDArray:
    """Draw zero-centered normal weights; biases are initialized separately."""
    if not isinstance(fan_in, int) or not isinstance(fan_out, int) or min(fan_in, fan_out) < 1:
        raise ValueError("fan_in and fan_out must be positive integers")
    scales = {"random": 0.01, "xavier": np.sqrt(2 / (fan_in + fan_out)), "he": np.sqrt(2 / fan_in)}
    if strategy not in scales:
        raise ValueError(f"Unknown initialization {strategy!r}; choose {tuple(scales)}")
    if np.dtype(dtype) not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError("dtype must be float32 or float64")
    rng = np.random.default_rng() if rng is None else rng
    return (rng.standard_normal((fan_in, fan_out)) * scales[strategy]).astype(dtype)
