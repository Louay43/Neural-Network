"""Trainable affine layers and inverted dropout, with explicit derivatives."""

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from .initializers import initialize


class Layer(Protocol):
    """Minimal contract shared by trainable and non-trainable layers."""

    def forward(self, inputs: NDArray, training: bool = False) -> NDArray: ...

    def backward(self, gradient: NDArray) -> NDArray: ...


class Dense:
    """Affine transformation Z = XW + b, with optional L2 weight penalty.

    The loss is already averaged over the batch. Do not average dW/db again.
    L2 contributes (l2 / 2) * sum(W**2) to the objective and l2 * W to dW.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        initialization: str = "he",
        l2: float = 0.0,
        seed: int | None = None,
        dtype: str = "float64",
    ) -> None:
        if not np.isfinite(l2) or l2 < 0:
            raise ValueError("l2 must be finite and nonnegative")
        self.input_size, self.output_size = input_size, output_size
        self.initialization, self.l2 = initialization, float(l2)
        self.weights = initialize(input_size, output_size, initialization, np.random.default_rng(seed), dtype)
        self.biases = np.zeros((1, output_size), dtype=dtype)
        self.dW = np.zeros_like(self.weights)
        self.db = np.zeros_like(self.biases)
        self.dX: NDArray | None = None
        self._inputs: NDArray | None = None

    def forward(self, inputs: NDArray, training: bool = False) -> NDArray:
        """Transform a (batch, input_size) matrix; cache only during training."""
        inputs = np.asarray(inputs, dtype=self.weights.dtype)
        if inputs.ndim != 2 or inputs.shape[1] != self.input_size:
            raise ValueError(f"Expected inputs with shape (batch, {self.input_size})")
        self._inputs = inputs if training else None
        return inputs @ self.weights + self.biases

    def backward(self, gradient: NDArray) -> NDArray:
        """Calculate dW = X.T @ dZ, db = sum(dZ), dX = dZ @ W.T."""
        if self._inputs is None:
            raise RuntimeError("Call forward(..., training=True) before backward")
        if gradient.shape != (self._inputs.shape[0], self.output_size):
            raise ValueError("Gradient shape does not match Dense output")
        self.dW = self._inputs.T @ gradient + self.l2 * self.weights
        self.db = np.sum(gradient, axis=0, keepdims=True)
        self.dX = gradient @ self.weights.T
        return self.dX

    def regularization_loss(self) -> float:
        """Weight-only L2 penalty (biases are not penalized)."""
        return float(0.5 * self.l2 * np.sum(self.weights ** 2))


class Dropout:
    """Inverted dropout: divide retained activations by the keep probability."""

    def __init__(self, rate: float = 0.2, seed: int | None = None) -> None:
        if not np.isfinite(rate) or not 0 <= rate < 1:
            raise ValueError("Dropout rate must be in [0, 1)")
        self.rate = float(rate)
        self.rng = np.random.default_rng(seed)
        self._mask: NDArray | None = None

    def forward(self, inputs: NDArray, training: bool = False) -> NDArray:
        """Use independent Bernoulli masks in training and identity at inference."""
        self._mask = None
        if not training:
            return inputs
        self._mask = (self.rng.random(inputs.shape) >= self.rate).astype(inputs.dtype) / (1 - self.rate)
        return inputs * self._mask

    def backward(self, gradient: NDArray) -> NDArray:
        """Reuse exactly the mask sampled during the training forward pass."""
        if self._mask is None:
            raise RuntimeError("Call forward(..., training=True) before backward")
        if gradient.shape != self._mask.shape:
            raise ValueError("Gradient shape does not match Dropout output")
        return gradient * self._mask
