"""Reload an inference checkpoint, evaluate official MNIST test data, and save figures."""

import argparse
import json
from pathlib import Path

from neural_network import NeuralNetwork
from neural_network.data import load_mnist
from neural_network.plotting import plot_predictions
from neural_network.workflows import save_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("results/mnist/model.npz"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("results/evaluation"))
    parser.add_argument("--seed", type=int, default=42, help="Seed used only for figure sampling")
    args = parser.parse_args()
    model = NeuralNetwork.load(args.model)
    data = load_mnist(args.data_dir)
    metrics = model.evaluate(data.X_test, data.y_test)
    save_json(args.output / "metrics.json", metrics)
    plot_predictions(data.X_test, data.y_test, model.predict_proba(data.X_test), args.output, args.seed)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
