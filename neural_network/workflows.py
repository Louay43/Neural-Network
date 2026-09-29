"""Small shared helpers for reproducible command-line runs."""

import json
import os
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

from . import Adam, Dense, Dropout, Momentum, NeuralNetwork, ReLU, SGD, Sigmoid, Softmax


@dataclass(frozen=True)
class ModelConfig:
    """Serializable architecture, regularization, and optimizer settings."""

    hidden: tuple[int, ...] = (128, 64)
    activation: str = "relu"
    initialization: str = "he"
    dropout: float = 0.2
    l2: float = 1e-4
    optimizer: str = "adam"
    learning_rate: float = 0.001
    momentum: float = 0.9
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    dtype: str = "float64"


def build_model(config: ModelConfig, seed: int = 42) -> NeuralNetwork:
    """Derive independent layer seeds from one run seed and select numeric precision."""
    if config.activation not in ("relu", "sigmoid"):
        raise ValueError("activation must be relu or sigmoid")
    activation = ReLU if config.activation == "relu" else Sigmoid
    rng, layers = np.random.default_rng(seed), []
    sizes = (784, *config.hidden, 10)
    layer_seeds = rng.integers(2**31, size=(len(sizes) - 1, 2))
    for index, (input_size, output_size) in enumerate(zip(sizes[:-1], sizes[1:])):
        layers.append(Dense(input_size, output_size, initialization=config.initialization,
                            l2=config.l2, seed=int(layer_seeds[index, 0]), dtype=config.dtype))
        if index < len(sizes) - 2:
            layers.append(activation())
            if config.dropout:
                layers.append(Dropout(config.dropout, seed=int(layer_seeds[index, 1])))
    return NeuralNetwork([*layers, Softmax()], seed=seed)


def build_optimizer(config: ModelConfig) -> SGD:
    if config.optimizer == "adam":
        return Adam(config.learning_rate, config.beta1, config.beta2, config.epsilon)
    if config.optimizer == "momentum":
        return Momentum(config.learning_rate, config.momentum)
    if config.optimizer == "sgd":
        return SGD(config.learning_rate)
    raise ValueError("optimizer must be sgd, momentum, or adam")


def save_json(path: str | Path, content: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run_metadata(config: ModelConfig | None = None) -> dict:
    """Record provenance without machine-specific user paths."""
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "numpy": np.__version__,
        "matplotlib": version("matplotlib"),
        "platform": platform.system(), "machine": platform.machine(),
        "command": "python " + " ".join(sys.argv),
        "blas_thread_environment": {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
    }
    if config is not None:
        metadata["model_config"] = asdict(config)
    return metadata
