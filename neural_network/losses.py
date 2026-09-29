"""Mean categorical cross-entropy, in probability and stable logit forms."""

import numpy as np
from numpy.typing import NDArray


def targets(y: NDArray, n: int, classes: int, dtype: np.dtype) -> NDArray:
    """Validate sparse class indices or one-hot/soft target distributions."""
    y = np.asarray(y)
    if y.shape == (n,):
        if not np.issubdtype(y.dtype, np.integer) or np.any(y < 0) or np.any(y >= classes):
            raise ValueError(f"Sparse labels must be integers in [0, {classes})")
        return np.eye(classes, dtype=dtype)[y]
    if y.shape == (n, classes):
        if not np.all(np.isfinite(y)) or np.any(y < 0) or not np.allclose(y.sum(axis=1), 1, atol=1e-6):
            raise ValueError("Dense targets must be nonnegative distributions summing to 1")
        return y.astype(dtype, copy=False)
    raise ValueError(f"Labels must have shape ({n},) or ({n}, {classes})")


class CategoricalCrossEntropy:
    """Cross-entropy averaged once, over examples, never over classes.

    Probability mode uses log(p + epsilon); backward differentiates that exact
    smoothed expression. Logit mode uses log-sum-exp and is preferred in training
    because it retains useful gradients even when a softmax value underflows.
    """

    def __init__(self, epsilon: float = 1e-12) -> None:
        if not np.isfinite(epsilon) or not 0 < epsilon < 1:
            raise ValueError("epsilon must be in (0, 1)")
        self.epsilon = epsilon

    def _validate(self, values: NDArray, y: NDArray) -> NDArray:
        if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] < 2 or not np.all(np.isfinite(values)):
            raise ValueError("Predictions must be a nonempty finite (batch, classes) matrix")
        # Integer logits are valid inputs; do not truncate soft targets to ints.
        return targets(y, *values.shape, np.result_type(values.dtype, np.float32))

    def forward(self, probabilities: NDArray, y: NDArray) -> float:
        """Compute -mean(sum(target * log(p + epsilon)))."""
        target = self._validate(probabilities, y)
        if np.any(probabilities < 0) or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-6):
            raise ValueError("Probabilities must be nonnegative and sum to 1 per row")
        return float(-np.sum(target * np.log(probabilities + self.epsilon)) / len(probabilities))

    def backward(self, probabilities: NDArray, y: NDArray) -> NDArray:
        """Return dL/dp; pass this through Softmax.backward for separate stages."""
        target = self._validate(probabilities, y)
        return -target / (probabilities + self.epsilon) / len(probabilities)

    def forward_logits(self, logits: NDArray, y: NDArray) -> float:
        """Stable fused softmax-cross-entropy, still implemented entirely by hand."""
        target = self._validate(logits, y)
        shifted = logits - logits.max(axis=1, keepdims=True)
        log_probabilities = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
        return float(-np.sum(target * log_probabilities) / len(logits))

    def backward_logits(self, logits: NDArray, y: NDArray) -> NDArray:
        """Analytic simplification of softmax + cross-entropy: (p - target) / N."""
        target = self._validate(logits, y)
        exps = np.exp(logits - logits.max(axis=1, keepdims=True))
        return (exps / exps.sum(axis=1, keepdims=True) - target) / len(logits)
