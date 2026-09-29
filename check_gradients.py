"""Run exhaustive numerical checks on a tiny network, including its input gradients."""

import argparse
from pathlib import Path

import numpy as np

from neural_network import Dense, NeuralNetwork, ReLU, Sigmoid, Softmax
from neural_network.gradient_check import check_gradients
from neural_network.workflows import save_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/gradient_checks.json"))
    args = parser.parse_args()
    X = np.random.default_rng(0).normal(size=(5, 3))
    reports = {}
    for activation in (ReLU, Sigmoid):
        model = NeuralNetwork([Dense(3, 4, l2=0.07, seed=8), activation(), Dense(4, 3, seed=9), Softmax()])
        reports[activation.__name__] = check_gradients(model, X, np.array([0, 1, 2, 1, 0]))
    save_json(args.output, reports)
    for name, items in reports.items():
        for item in items:
            print(f"{name:7s} {item['tensor']:18s} relative error={item['relative_l2_error']:.2e} passed={item['passed']}")
    if not all(item["passed"] for items in reports.values() for item in items):
        raise SystemExit("Gradient check failed")


if __name__ == "__main__":
    main()
