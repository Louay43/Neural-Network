"""Sequential classifier and explicit mini-batch training loop."""

import json
import time
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .activations import ReLU, Sigmoid, Softmax
from .layers import Dense, Dropout, Layer
from .losses import CategoricalCrossEntropy, targets
from .optimizers import Adam, SGD


class NeuralNetwork:
    """A reusable dense classifier ending in Softmax.

    ``seed`` controls batch order. Layer constructors independently control their
    weight/mask seeds; the CLI builder sets all of them from one experiment seed.
    Forward caches support one outstanding training pass (not thread safe).
    """

    def __init__(self, layers: list[Layer], seed: int = 42) -> None:
        if len(layers) < 2 or not isinstance(layers[0], Dense) or not isinstance(layers[-1], Softmax):
            raise ValueError("A classifier must start with Dense and end with Softmax")
        if any(isinstance(layer, Softmax) for layer in layers[:-1]):
            raise ValueError("Softmax is supported only as the final classifier layer")
        if len({id(layer) for layer in layers}) != len(layers):
            raise ValueError("Layer instances cannot be reused within a network")
        width = layers[0].input_size
        for layer in layers:
            if isinstance(layer, Dense):
                if layer.input_size != width:
                    raise ValueError("Adjacent Dense layer dimensions do not agree")
                width = layer.output_size
        if width < 2:
            raise ValueError("The classifier needs at least two output classes")
        self.layers = list(layers)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.loss = CategoricalCrossEntropy()
        self.history: dict[str, list[float]] = {}
        self._logits: NDArray | None = None
        self._training_forward = False

    @property
    def dense_layers(self) -> list[Dense]:
        return [layer for layer in self.layers if isinstance(layer, Dense)]

    def _validate_data(self, X: NDArray, y: NDArray | None = None) -> NDArray:
        X = np.asarray(X, dtype=self.dense_layers[0].weights.dtype)
        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] != self.dense_layers[0].input_size:
            raise ValueError("X must be a nonempty matrix matching the network input size")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite numbers")
        if y is not None:
            targets(y, len(X), self.dense_layers[-1].output_size, X.dtype)
        return X

    def forward(self, X: NDArray, training: bool = False) -> NDArray:
        """Propagate activations in layer order; inference disables dropout."""
        output = X
        for layer in self.layers[:-1]:
            output = layer.forward(output, training=training)
        self._logits = output
        self._training_forward = training
        return self.layers[-1].forward(output, training=training)

    def backward(self, y: NDArray) -> NDArray:
        """Explicit chain rule, from loss/logits through each hidden layer to X."""
        if not self._training_forward or self._logits is None:
            raise RuntimeError("Call forward(..., training=True) before backward")
        gradient = self.loss.backward_logits(self._logits, y)
        # The fused derivative already includes the final Softmax Jacobian.
        for layer in reversed(self.layers[:-1]):
            gradient = layer.backward(gradient)
        return gradient

    def regularization_loss(self) -> float:
        return sum(layer.regularization_loss() for layer in self.dense_layers)

    def evaluate(self, X: NDArray, y: NDArray, batch_size: int = 1024) -> dict[str, float]:
        """Sample-weighted inference metrics; `loss` includes the L2 penalty."""
        X = self._validate_data(X, y)
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        y = np.asarray(y)
        total_loss, correct = 0.0, 0
        labels = y if y.ndim == 1 else y.argmax(axis=1)
        for start in range(0, len(X), batch_size):
            stop = start + batch_size
            probabilities = self.forward(X[start:stop])
            total_loss += self.loss.forward_logits(self._logits, y[start:stop]) * len(probabilities)
            correct += int(np.sum(probabilities.argmax(axis=1) == labels[start:stop]))
        data_loss = total_loss / len(X)
        return {"loss": data_loss + self.regularization_loss(), "data_loss": data_loss, "accuracy": correct / len(X)}

    def predict_proba(self, X: NDArray, batch_size: int = 1024) -> NDArray:
        X = self._validate_data(X)
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        return np.concatenate([self.forward(X[i:i + batch_size]) for i in range(0, len(X), batch_size)])

    def predict(self, X: NDArray, batch_size: int = 1024) -> NDArray:
        return self.predict_proba(X, batch_size).argmax(axis=1)

    def fit(
        self, X: NDArray, y: NDArray, epochs: int = 20, batch_size: int = 64,
        optimizer: SGD | None = None, validation_data: tuple[NDArray, NDArray] | None = None,
        verbose: bool = True,
    ) -> dict[str, list[float]]:
        """Shuffle every epoch, backpropagate every batch, then measure full splits.

        Epoch metrics use the final weights of that epoch, in inference mode, so
        training and validation losses are directly comparable. A new fit call
        resets history; pass the same optimizer to preserve its moment state.
        """
        X = self._validate_data(X, y)
        y = np.asarray(y)
        if not isinstance(epochs, int) or epochs < 1 or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("epochs and batch_size must be positive integers")
        if validation_data is not None:
            self._validate_data(*validation_data)
        optimizer = Adam() if optimizer is None else optimizer
        self.history = {"train_loss": [], "train_accuracy": [], "epoch_seconds": []}
        if validation_data is not None:
            self.history.update({"val_loss": [], "val_accuracy": []})
        for epoch in range(epochs):
            started = time.perf_counter()
            order = self.rng.permutation(len(X))
            for start in range(0, len(X), batch_size):
                indices = order[start:start + batch_size]
                self.forward(X[indices], training=True)
                self.backward(y[indices])
                optimizer.step(self.dense_layers)
            train = self.evaluate(X, y)
            self.history["train_loss"].append(train["loss"])
            self.history["train_accuracy"].append(train["accuracy"])
            message = f"Epoch {epoch + 1:02d}/{epochs}: loss={train['loss']:.4f} accuracy={train['accuracy']:.4%}"
            if validation_data is not None:
                val = self.evaluate(*validation_data)
                self.history["val_loss"].append(val["loss"])
                self.history["val_accuracy"].append(val["accuracy"])
                message += f" val_loss={val['loss']:.4f} val_accuracy={val['accuracy']:.4%}"
            self.history["epoch_seconds"].append(time.perf_counter() - started)
            if verbose:
                print(message, flush=True)
        return self.history

    def save(self, path: str | Path) -> None:
        """Save architecture and inference parameters to NPZ (no pickle).

        Optimizer moments, RNG progression and training history are not saved;
        this is an inference checkpoint, not an exact training-resume snapshot.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        config, arrays = [], {}
        for index, layer in enumerate(self.layers):
            item = {"type": type(layer).__name__}
            if isinstance(layer, Dense):
                item.update(input_size=layer.input_size, output_size=layer.output_size,
                            initialization=layer.initialization, l2=layer.l2, dtype=str(layer.weights.dtype))
                arrays[f"W{index}"] = layer.weights
                arrays[f"b{index}"] = layer.biases
            elif isinstance(layer, Dropout):
                item["rate"] = layer.rate
            elif not isinstance(layer, (ReLU, Sigmoid, Softmax)):
                raise ValueError(f"Cannot serialize layer {type(layer).__name__}")
            config.append(item)
        arrays["metadata"] = np.array(json.dumps({"version": 1, "seed": self.seed, "layers": config}))
        with path.open("wb") as handle:
            np.savez_compressed(handle, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "NeuralNetwork":
        """Load only recognized layer types and validate parameter dimensions."""
        constructors = {kind.__name__: kind for kind in (Dense, Dropout, ReLU, Sigmoid, Softmax)}
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"]))
            if metadata.get("version") != 1:
                raise ValueError("Unsupported checkpoint version")
            layers = []
            for index, saved in enumerate(metadata["layers"]):
                config = dict(saved)
                name = config.pop("type")
                if name not in constructors:
                    raise ValueError(f"Unsupported layer type {name!r}")
                layer = constructors[name](**config)
                if isinstance(layer, Dense):
                    for key, parameter in ((f"W{index}", layer.weights), (f"b{index}", layer.biases)):
                        values = archive[key]
                        if values.shape != parameter.shape or not np.all(np.isfinite(values)):
                            raise ValueError("Invalid checkpoint parameter shape or values")
                        parameter[:] = values
                layers.append(layer)
        return cls(layers, seed=metadata["seed"])
