"""Controlled MNIST comparisons on validation data; never score the test set."""

import argparse
import csv
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from neural_network.data import load_mnist
from neural_network.plotting import plot_experiments
from neural_network.workflows import ModelConfig, build_model, build_optimizer, run_metadata, save_json


def experiment_configs() -> list[tuple[str, ModelConfig]]:
    """Change one setting from the reference except documented optimizer/LR presets."""
    baseline = ModelConfig(dropout=0, l2=0, dtype="float32")
    return [
        ("Adam reference", baseline),
        ("SGD (lr=0.1)", replace(baseline, optimizer="sgd", learning_rate=0.1)),
        ("Momentum (lr=0.01)", replace(baseline, optimizer="momentum", learning_rate=0.01)),
        ("Xavier initialization", replace(baseline, initialization="xavier")),
        ("Random initialization", replace(baseline, initialization="random")),
        ("Adam lr=0.0003", replace(baseline, learning_rate=0.0003)),
        ("Adam lr=0.003", replace(baseline, learning_rate=0.003)),
        ("Dropout=0.2", replace(baseline, dropout=0.2)),
        ("Hidden: 64", replace(baseline, hidden=(64,))),
        ("Hidden: 256,128", replace(baseline, hidden=(256, 128))),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-limit", type=int, default=12000)
    parser.add_argument("--validation-size", type=int, default=5000)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("results/experiments"))
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be distinct")
    data = load_mnist(args.data_dir, args.validation_size, args.split_seed, args.train_limit)
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = run_metadata()
    metadata.update(epochs=args.epochs, batch_size=args.batch_size, seeds=args.seeds, split_seed=args.split_seed,
                    train_samples=len(data.X_train), validation_samples=len(data.X_val),
                    test_evaluated=False, optimizer_note="Optimizer comparisons use specified, different learning-rate presets; no exhaustive tuning.")
    save_json(args.output / "run.json", metadata)
    runs, summary = [], []
    for name, config in experiment_configs():
        group = []
        for seed in args.seeds:
            model = build_model(config, seed)
            started = time.perf_counter()
            history = model.fit(data.X_train, data.y_train, epochs=args.epochs, batch_size=args.batch_size,
                                optimizer=build_optimizer(config), validation_data=(data.X_val, data.y_val), verbose=False)
            row = {"name": name, "seed": seed, "config": asdict(config), "history": history,
                   "train_loss": history["train_loss"][-1], "train_accuracy": history["train_accuracy"][-1],
                   "val_loss": history["val_loss"][-1], "val_accuracy": history["val_accuracy"][-1],
                   "seconds": time.perf_counter() - started}
            runs.append(row)
            group.append(row)
            print(f"{name:24s} seed={seed} validation={row['val_accuracy']:.2%} loss={row['val_loss']:.4f}", flush=True)
            save_json(args.output / "runs.json", runs)
        entry = {"name": name, "config": asdict(config), "seeds": len(group)}
        for metric in ("train_loss", "train_accuracy", "val_loss", "val_accuracy", "seconds"):
            entry[f"{metric}_mean"] = float(np.mean([row[metric] for row in group]))
            entry[f"{metric}_std"] = float(np.std([row[metric] for row in group], ddof=1)) if len(group) > 1 else 0.0
        summary.append(entry)
    save_json(args.output / "summary.json", summary)
    columns = [key for key in summary[0] if key != "config"]
    with (args.output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary)
    lines = ["# Measured experiment results", "",
             f"Each run: {len(data.X_train):,} training / {len(data.X_val):,} validation images, {args.epochs} epochs, batch size {args.batch_size}.", "",
             f"Fixed split seed: {args.split_seed}. Model/batch seeds: {args.seeds}. Test data is never scored.", "",
             "Accuracy values are mean +/- sample standard deviation across seeds (percentage points).", "",
             "| Configuration | Validation accuracy | Validation loss | Mean seconds |", "|---|---:|---:|---:|"]
    for row in summary:
        lines.append(f"| {row['name']} | {row['val_accuracy_mean'] * 100:.2f} +/- {row['val_accuracy_std'] * 100:.2f}% | {row['val_loss_mean']:.4f} | {row['seconds_mean']:.2f} |")
    lines.extend(["", "All configurations use L2=0. The reference uses He initialization, ReLU, hidden widths 128/64, no dropout, and Adam at 0.001.",
                  "Optimizer presets use different stated learning rates; these results do not isolate optimizer choice from learning rate.",
                  "These are short subset experiments, not fully tuned benchmark rankings or confidence intervals.", ""])
    (args.output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    plot_experiments(summary, args.output)


if __name__ == "__main__":
    main()
