# Measured experiment results

Each run: 12,000 training / 5,000 validation images, 5 epochs, batch size 64.

Fixed split seed: 42. Model/batch seeds: [42, 43, 44]. Test data is never scored.

Accuracy values are mean +/- sample standard deviation across seeds (percentage points).

| Configuration | Validation accuracy | Validation loss | Mean seconds |
|---|---:|---:|---:|
| Adam reference | 94.89 +/- 0.33% | 0.1780 | 2.25 |
| SGD (lr=0.1) | 93.49 +/- 0.50% | 0.2171 | 1.95 |
| Momentum (lr=0.01) | 93.69 +/- 0.38% | 0.2138 | 2.21 |
| Xavier initialization | 94.83 +/- 0.11% | 0.1812 | 3.14 |
| Random initialization | 91.35 +/- 0.51% | 0.2985 | 3.69 |
| Adam lr=0.0003 | 93.09 +/- 0.20% | 0.2400 | 3.25 |
| Adam lr=0.003 | 94.96 +/- 0.64% | 0.1911 | 3.84 |
| Dropout=0.2 | 94.59 +/- 0.20% | 0.1817 | 3.42 |
| Hidden: 64 | 93.56 +/- 0.39% | 0.2289 | 1.53 |
| Hidden: 256,128 | 95.63 +/- 0.18% | 0.1569 | 6.37 |

All configurations use L2=0. The reference uses He initialization, ReLU, hidden widths 128/64, no dropout, and Adam at 0.001.
Optimizer presets use different stated learning rates; these results do not isolate optimizer choice from learning rate.
These are short subset experiments, not fully tuned benchmark rankings or confidence intervals.
