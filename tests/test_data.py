import gzip
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_array_equal

from neural_network.data import load_mnist, read_idx


class DataTests(unittest.TestCase):
    def write_idx(self, path, values):
        with gzip.open(path, "wb") as handle:
            handle.write(struct.pack(">I", 2048 + values.ndim))
            handle.write(struct.pack(">" + "I" * values.ndim, *values.shape))
            handle.write(values.astype(np.uint8).tobytes())

    def test_idx_split_normalization_and_reproducibility(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = np.broadcast_to(np.arange(20, dtype=np.uint8)[:, None, None], (20, 28, 28))
            for prefix, values in (("train", images), ("t10k", images[:4])):
                self.write_idx(root / f"{prefix}-images-idx3-ubyte.gz", values)
                self.write_idx(root / f"{prefix}-labels-idx1-ubyte.gz", np.arange(len(values)) % 10)
            assert_array_equal(read_idx(root / "train-images-idx3-ubyte.gz"), images)
            with patch("neural_network.data.download_mnist"):
                data = load_mnist(root, validation_size=5, train_limit=10)
                repeated = load_mnist(root, validation_size=5, train_limit=10)
            self.assertEqual(data.X_train.shape, (10, 784))
            self.assertEqual(data.X_train.dtype, np.float32)
            self.assertEqual(data.X_test.shape, (4, 784))
            self.assertFalse(set(data.X_train[:, 0]) & set(data.X_val[:, 0]))
            assert_array_equal(data.X_train, repeated.X_train)
            self.assertGreaterEqual(data.X_train.min(), 0)
            self.assertLessEqual(data.X_train.max(), 1)

    def test_rejects_truncated_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.gz"
            with gzip.open(path, "wb") as handle:
                handle.write(struct.pack(">II", 2049, 20) + b"abc")
            with self.assertRaises(ValueError):
                read_idx(path)
