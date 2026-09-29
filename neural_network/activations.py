"""Activation functions and their handwritten vector-Jacobian products."""

import numpy as np
from numpy.typing import NDArray


class ReLU:
    """Rectifier; use the conventional subgradient zero at the origin."""

    def __init__(self) -> None:
        self._positive: NDArray | None = None

    def forward(self, inputs: NDArray, training: bool = False) -> NDArray:
        self._positive = inputs > 0 if training else None
        return np.maximum(inputs, 0)

    def backward(self, gradient: NDArray) -> NDArray:
        if self._positive is None:
            raise RuntimeError("Call forward(..., training=True) before backward")
        if gradient.shape != self._positive.shape:
            raise ValueError("Gradient shape does not match ReLU output")
        return gradient * self._positive


class Sigmoid:
    """Overflow-safe logistic function, with derivative s * (1 - s)."""

    def __init__(self) -> None:
        self._output: NDArray | None = None

    def forward(self, inputs: NDArray, training: bool = False) -> NDArray:
        inputs = np.asarray(inputs)
        output = np.empty_like(inputs, dtype=np.result_type(inputs.dtype, np.float32))
        positive = inputs >= 0
        output[positive] = 1 / (1 + np.exp(-inputs[positive]))
        exp_x = np.exp(inputs[~positive])
        output[~positive] = exp_x / (1 + exp_x)
        self._output = output if training else None
        return output

    def backward(self, gradient: NDArray) -> NDArray:
        if self._output is None:
            raise RuntimeError("Call forward(..., training=True) before backward")
        if gradient.shape != self._output.shape:
            raise ValueError("Gradient shape does not match Sigmoid output")
        return gradient * self._output * (1 - self._output)


class Softmax:
    """Row-wise softmax; subtract each row's maximum before exponentiation."""

    def __init__(self) -> None:
        self._output: NDArray | None = None

    def forward(self, inputs: NDArray, training: bool = False) -> NDArray:
        if inputs.ndim != 2 or inputs.shape[1] < 2:
            raise ValueError("Softmax expects a (batch, classes) matrix with at least 2 classes")
        exp_x = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        output = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        self._output = output if training else None
        return output

    def backward(self, gradient: NDArray) -> NDArray:
        """Compute J.T @ gradient without allocating a batch of C-by-C matrices."""
        if self._output is None:
            raise RuntimeError("Call forward(..., training=True) before backward")
        if gradient.shape != self._output.shape:
            raise ValueError("Gradient shape does not match Softmax output")
        p = self._output
        return p * (gradient - np.sum(gradient * p, axis=1, keepdims=True))
