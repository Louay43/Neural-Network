"""Finite-difference verification of analytical parameter and input gradients."""

import numpy as np
from numpy.typing import NDArray

from .layers import Dropout
from .network import NeuralNetwork


def check_gradients(
    model: NeuralNetwork, X: NDArray, y: NDArray, epsilon: float = 1e-5,
    max_checks: int | None = None, seed: int = 0,
    atol: float = 1e-7, rtol: float = 1e-5,
) -> list[dict[str, str | float | int | bool]]:
    """Compare derivatives against [L(theta+h)-L(theta-h)]/(2h).

    Use a tiny float64 network away from ReLU kinks. Nonzero dropout must be
    removed/disabled: different masks would make the two objectives incomparable.
    L2 penalties are included. All perturbed values are restored, even on error.
    """
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be positive and finite")
    if max_checks is not None and (not isinstance(max_checks, int) or max_checks < 1):
        raise ValueError("max_checks must be a positive integer or None")
    if any(layer.weights.dtype != np.float64 for layer in model.dense_layers):
        raise ValueError("Gradient checking requires float64 parameters")
    if any(isinstance(layer, Dropout) and layer.rate > 0 for layer in model.layers):
        raise ValueError("Disable stochastic dropout before gradient checking")
    X = np.asarray(X, dtype=np.float64).copy()
    model.forward(X, training=True)
    input_gradient = model.backward(y).copy()
    tensors = []
    for index, layer in enumerate(model.dense_layers):
        tensors.extend([(f"dense_{index}.weights", layer.weights, layer.dW.copy()),
                        (f"dense_{index}.biases", layer.biases, layer.db.copy())])
    tensors.append(("inputs", X, input_gradient))
    rng = np.random.default_rng(seed)

    def objective() -> float:
        model.forward(X)
        return model.loss.forward_logits(model._logits, y) + model.regularization_loss()

    report = []
    for name, parameter, analytical in tensors:
        indices = np.arange(parameter.size)
        if max_checks is not None and parameter.size > max_checks:
            indices = rng.choice(indices, max_checks, replace=False)
        numeric = np.empty(len(indices))
        for position, index in enumerate(indices):
            old = parameter.flat[index]
            try:
                parameter.flat[index] = old + epsilon
                plus = objective()
                parameter.flat[index] = old - epsilon
                minus = objective()
                numeric[position] = (plus - minus) / (2 * epsilon)
            finally:
                parameter.flat[index] = old
        actual = analytical.ravel()[indices]
        difference = np.linalg.norm(actual - numeric)
        relative = difference / max(1e-12, np.linalg.norm(actual) + np.linalg.norm(numeric))
        report.append({"tensor": name, "checked": len(indices), "max_absolute_error": float(np.max(np.abs(actual - numeric))),
                       "relative_l2_error": float(relative), "passed": bool(np.allclose(actual, numeric, atol=atol, rtol=rtol))})
    return report
