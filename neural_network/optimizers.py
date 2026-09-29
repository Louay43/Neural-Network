"""Parameter updates written directly in NumPy; no autodiff or training engine."""

from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray

from .layers import Dense


class SGD:
    """Stochastic gradient descent, optionally with classical momentum.

    v_t = momentum * v_(t-1) + gradient; parameter -= learning_rate * v_t.
    State belongs to this optimizer instance. Use a new instance for a new run.
    """

    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0) -> None:
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not np.isfinite(momentum) or not 0 <= momentum < 1:
            raise ValueError("momentum must be in [0, 1)")
        self.learning_rate, self.momentum = float(learning_rate), float(momentum)
        self._velocity: dict[tuple[Dense, str], NDArray] = {}

    @staticmethod
    def _parameters(layers: Iterable[Dense]) -> list[tuple[tuple[Dense, str], NDArray, NDArray]]:
        parameters = []
        for layer in layers:
            for name, gradient in (("weights", layer.dW), ("biases", layer.db)):
                parameter = getattr(layer, name)
                if parameter.shape != gradient.shape or not np.all(np.isfinite(gradient)):
                    raise ValueError("All parameter gradients must have matching shapes and be finite")
                parameters.append(((layer, name), parameter, gradient))
        return parameters

    def step(self, layers: Iterable[Dense]) -> None:
        """Update weights and biases once after a complete backward pass."""
        for key, parameter, gradient in self._parameters(layers):
            if self.momentum:
                velocity = self._velocity.setdefault(key, np.zeros_like(parameter))
                velocity *= self.momentum
                velocity += gradient
                parameter -= self.learning_rate * velocity
            else:
                parameter -= self.learning_rate * gradient


class Momentum(SGD):
    """Convenience name for SGD with momentum enabled."""

    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9) -> None:
        super().__init__(learning_rate, momentum)


class Adam(SGD):
    """Adaptive moment estimation, with first/second-moment bias correction."""

    def __init__(
        self, learning_rate: float = 0.001, beta1: float = 0.9,
        beta2: float = 0.999, epsilon: float = 1e-8,
    ) -> None:
        super().__init__(learning_rate)
        if not all(np.isfinite(b) and 0 <= b < 1 for b in (beta1, beta2)):
            raise ValueError("beta1 and beta2 must be in [0, 1)")
        if not np.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")
        self.beta1, self.beta2, self.epsilon = beta1, beta2, epsilon
        self.t = 0
        self._moments: dict[tuple[Dense, str], tuple[NDArray, NDArray]] = {}

    def step(self, layers: Iterable[Dense]) -> None:
        parameters = self._parameters(layers)
        self.t += 1  # One time step per mini-batch, not per parameter.
        for key, parameter, gradient in parameters:
            m, v = self._moments.setdefault(key, (np.zeros_like(parameter), np.zeros_like(parameter)))
            m *= self.beta1
            m += (1 - self.beta1) * gradient
            v *= self.beta2
            v += (1 - self.beta2) * gradient ** 2
            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)
            parameter -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
