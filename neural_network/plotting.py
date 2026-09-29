"""Headless Matplotlib figures and transparent CSV prediction artifacts."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

plt.rcParams.update({
    "figure.facecolor": "#f7f9fc", "axes.facecolor": "#ffffff",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelcolor": "#243347", "text.color": "#243347",
    "font.size": 11, "savefig.dpi": 160,
})


def plot_history(history: dict[str, list[float]], output: str | Path) -> None:
    """Plot comparable epoch-end train/validation losses and accuracies."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    for axis, metric, title in zip(axes, ("loss", "accuracy"), ("Cross-entropy + L2", "Classification accuracy")):
        for split, label, color in (("train", "Training", "#2267a7"), ("val", "Validation", "#e07a30")):
            key = f"{split}_{metric}"
            if key in history:
                scale = 100 if metric == "accuracy" else 1
                axis.plot(epochs, np.array(history[key]) * scale, label=label, color=color, linewidth=2.2)
        axis.set(title=title, xlabel="Epoch", ylabel="Accuracy (%)" if metric == "accuracy" else "Mean objective")
        axis.grid(alpha=0.18)
        axis.legend(frameon=False)
    fig.suptitle("MNIST | Learning from handwritten gradients", fontsize=16, fontweight="bold")
    fig.savefig(output / "learning_curves.png")
    plt.close(fig)


def confusion_matrix(y: NDArray, predicted: NDArray, classes: int = 10) -> NDArray:
    """Count actual classes on rows and predicted classes on columns."""
    matrix = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(matrix, (y, predicted), 1)
    return matrix


def plot_predictions(X: NDArray, y: NDArray, probabilities: NDArray, output: str | Path, seed: int = 42) -> None:
    """Save confusion counts and seeded samples of correct/incorrect predictions."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    predicted = probabilities.argmax(axis=1)
    matrix = confusion_matrix(y, predicted)
    np.savetxt(output / "confusion_matrix.csv", matrix, fmt="%d", delimiter=",")
    fig, axis = plt.subplots(figsize=(8.5, 7.5), constrained_layout=True)
    heatmap = axis.imshow(matrix, cmap="Blues")
    for (row, column), value in np.ndenumerate(matrix):
        axis.text(column, row, str(value), ha="center", va="center", fontsize=9,
                  color="white" if value > matrix.max() * 0.5 else "#243347")
    axis.set(xticks=range(10), yticks=range(10), xlabel="Predicted digit", ylabel="Actual digit",
             title=f"MNIST test set | {np.mean(predicted == y):.2%} accuracy")
    fig.colorbar(heatmap, ax=axis, shrink=0.8, label="Number of examples")
    fig.savefig(output / "confusion_matrix.png")
    plt.close(fig)

    with (output / "test_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["test_index", "actual", "predicted", "confidence"])
        for index, (actual, guess, probability) in enumerate(zip(y, predicted, probabilities.max(axis=1))):
            writer.writerow([index, int(actual), int(guess), float(probability)])

    with (output / "per_class_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["digit", "precision", "recall", "f1", "support"])
        for digit in range(10):
            precision = matrix[digit, digit] / max(1, matrix[:, digit].sum())
            recall = matrix[digit, digit] / max(1, matrix[digit].sum())
            f1 = 2 * precision * recall / max(1e-12, precision + recall)
            writer.writerow([digit, float(precision), float(recall), float(f1), int(matrix[digit].sum())])

    rng = np.random.default_rng(seed)
    for correct, filename, title in ((True, "correct_predictions.png", "Correct predictions"),
                                     (False, "incorrect_predictions.png", "Misclassified digits")):
        eligible = np.flatnonzero((predicted == y) == correct)
        selected = rng.choice(eligible, size=min(20, len(eligible)), replace=False)
        fig, axes = plt.subplots(4, 5, figsize=(10, 8), constrained_layout=True)
        for axis in axes.flat:
            axis.axis("off")
        for axis, index in zip(axes.flat, selected):
            axis.imshow(X[index].reshape(28, 28), cmap="gray_r", vmin=0, vmax=1)
            axis.set_title(f"Actual {y[index]} | Pred {predicted[index]}\nConfidence {probabilities[index].max():.1%}",
                           fontsize=10, color="#2267a7" if correct else "#a74136")
        if not len(selected):
            fig.text(0.5, 0.5, "No examples in this category", ha="center")
        fig.suptitle(f"{title} | seeded sample from the test set", fontsize=16, fontweight="bold")
        fig.savefig(output / filename)
        plt.close(fig)


def plot_experiments(summary: list[dict], output: str | Path) -> None:
    """Compare validation performance, with across-seed standard deviations."""
    fig, axis = plt.subplots(figsize=(11, 6), constrained_layout=True)
    positions = np.arange(len(summary))
    means = np.array([item["val_accuracy_mean"] for item in summary]) * 100
    deviations = np.array([item["val_accuracy_std"] for item in summary]) * 100
    axis.barh(positions, means, xerr=deviations, color="#2267a7", capsize=4, height=0.65)
    axis.set(yticks=positions, yticklabels=[item["name"] for item in summary],
             xlabel="Validation accuracy (%) | mean and sample standard deviation",
             title="Controlled MNIST comparisons | equal data and epoch budgets")
    axis.invert_yaxis()
    axis.set_xlim(max(0, means.min() - 5), min(100, means.max() + 3))
    for index, mean in enumerate(means):
        axis.text(mean + deviations[index] + 0.15, index, f"{mean:.2f}%", va="center", fontsize=9)
    axis.grid(axis="x", alpha=0.18)
    fig.savefig(Path(output) / "comparison.png")
    plt.close(fig)
