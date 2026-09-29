"""Train a from-scratch MLP on MNIST and save an auditable run."""

import argparse
import time
from pathlib import Path

from neural_network import NeuralNetwork
from neural_network.data import load_mnist
from neural_network.plotting import plot_history, plot_predictions
from neural_network.workflows import ModelConfig, build_model, build_optimizer, run_metadata, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden", nargs="*", type=int, default=[128, 64], help="Hidden widths; omit values for a linear classifier")
    parser.add_argument("--activation", choices=["relu", "sigmoid"], default="relu")
    parser.add_argument("--initialization", choices=["random", "xavier", "he"], default="he")
    parser.add_argument("--optimizer", choices=["sgd", "momentum", "adam"], default="adam")
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-size", type=int, default=5000)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("results/mnist"))
    args = parser.parse_args()
    config = ModelConfig(tuple(args.hidden), args.activation, args.initialization, args.dropout, args.l2,
                         args.optimizer, args.learning_rate, args.momentum, args.beta1, args.beta2, args.epsilon, args.dtype)
    try:
        model, optimizer = build_model(config, args.seed), build_optimizer(config)
        data = load_mnist(args.data_dir, args.validation_size, args.seed, args.train_limit)
        initial = model.evaluate(data.X_train, data.y_train)
        print(f"Initial training loss: {initial['loss']:.6f}; accuracy: {initial['accuracy']:.2%}", flush=True)
        started = time.perf_counter()
        history = model.fit(data.X_train, data.y_train, epochs=args.epochs, batch_size=args.batch_size,
                            optimizer=optimizer, validation_data=(data.X_val, data.y_val))
        duration = time.perf_counter() - started
        model.save(args.output / "model.npz")
        # Evaluate the reloaded artifact, not just the still-live model.
        restored = NeuralNetwork.load(args.output / "model.npz")
        metrics = {"initial_train": initial, "train": restored.evaluate(data.X_train, data.y_train),
                   "validation": restored.evaluate(data.X_val, data.y_val), "test": restored.evaluate(data.X_test, data.y_test)}
        if metrics["train"]["loss"] >= initial["loss"]:
            raise RuntimeError("Training did not reduce loss; inspect the learning rate and configuration")
        metadata = run_metadata(config)
        metadata.update(seed=args.seed, split_seed=args.seed, epochs=args.epochs, batch_size=args.batch_size,
                        validation_size=args.validation_size, train_limit=args.train_limit,
                        samples={"train": len(data.X_train), "validation": len(data.X_val), "test": len(data.X_test)},
                        train_seconds=duration, checkpoint="model.npz", checkpoint_policy="final_epoch",
                        parameter_count=sum(layer.weights.size + layer.biases.size for layer in model.dense_layers))
        save_json(args.output / "run.json", metadata)
        save_json(args.output / "history.json", history)
        save_json(args.output / "metrics.json", metrics)
        plot_history(history, args.output)
        plot_predictions(data.X_test, data.y_test, restored.predict_proba(data.X_test), args.output, args.seed)
        print(f"Test accuracy: {metrics['test']['accuracy']:.2%}; training time: {duration:.1f}s")
        print(f"Saved checkpoint, metrics, and figures to {args.output.resolve()}")
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
