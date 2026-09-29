"""Download original MNIST IDX files and preprocess them using only the stdlib/NumPy."""

import gzip
import hashlib
import struct
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
FILES = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}


@dataclass(frozen=True)
class MNIST:
    """Disjoint train/validation partitions and the untouched official test set."""

    X_train: NDArray
    y_train: NDArray
    X_val: NDArray
    y_val: NDArray
    X_test: NDArray
    y_test: NDArray


def _digest(path: Path) -> str:
    # MD5 is the dataset's published identity checksum, not a security primitive.
    return hashlib.md5(path.read_bytes(), usedforsecurity=False).hexdigest()


def download_mnist(directory: str | Path = "data") -> None:
    """Cache checksum-verified gzip files; download to .part before replacing."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for filename, expected in FILES.items():
        destination = directory / filename
        if destination.exists() and _digest(destination) == expected:
            continue
        temporary = destination.with_suffix(destination.suffix + ".part")
        try:
            print(f"Downloading {filename} ...", flush=True)
            with urllib.request.urlopen(BASE_URL + filename, timeout=60) as response, temporary.open("wb") as handle:
                while chunk := response.read(1024 * 1024):
                    handle.write(chunk)
            if _digest(temporary) != expected:
                raise ValueError(f"MNIST checksum mismatch: {filename}")
            temporary.replace(destination)
        except (OSError, urllib.error.URLError, ValueError) as error:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"Could not download MNIST. Check network access or place the original gzip files in {directory.resolve()}") from error


def read_idx(path: str | Path) -> NDArray:
    """Read and validate a gzip-compressed unsigned-byte IDX image/label array."""
    with gzip.open(path, "rb") as handle:
        raw = handle.read()
    if len(raw) < 8:
        raise ValueError("Truncated IDX header")
    magic = struct.unpack(">I", raw[:4])[0]
    if magic not in (2049, 2051):
        raise ValueError(f"Unsupported MNIST IDX magic number: {magic}")
    dimensions = magic & 0xFF
    offset = 4 * (dimensions + 1)
    if len(raw) < offset:
        raise ValueError("Truncated IDX shape")
    shape = struct.unpack(">" + "I" * dimensions, raw[4:offset])
    if any(size == 0 for size in shape) or len(raw) - offset != int(np.prod(shape, dtype=np.int64)):
        raise ValueError("IDX payload size does not match its header")
    if dimensions == 3 and shape[1:] != (28, 28):
        raise ValueError("MNIST images must be 28 by 28 pixels")
    return np.frombuffer(raw, dtype=np.uint8, offset=offset).reshape(shape)


def load_mnist(
    directory: str | Path = "data", validation_size: int = 5000,
    seed: int = 42, train_limit: int | None = None,
) -> MNIST:
    """Flatten images to 784 float32 values in [0,1]; retain integer labels.

    Validation examples are selected from the original training set before any
    optional training subset is taken. The official test partition stays intact.
    """
    download_mnist(directory)
    directory = Path(directory)
    images = read_idx(directory / "train-images-idx3-ubyte.gz")
    labels = read_idx(directory / "train-labels-idx1-ubyte.gz")
    test_images = read_idx(directory / "t10k-images-idx3-ubyte.gz")
    test_labels = read_idx(directory / "t10k-labels-idx1-ubyte.gz")
    if len(images) != len(labels) or len(test_images) != len(test_labels):
        raise ValueError("Image and label counts disagree")
    if np.any(labels > 9) or np.any(test_labels > 9):
        raise ValueError("MNIST labels must be digits in [0, 9]")
    if not isinstance(validation_size, int) or not 0 < validation_size < len(images):
        raise ValueError("validation_size must leave nonempty training and validation sets")
    order = np.random.default_rng(seed).permutation(len(images))
    val_indices, train_indices = order[:validation_size], order[validation_size:]
    if train_limit is not None:
        if not isinstance(train_limit, int) or not 0 < train_limit <= len(train_indices):
            raise ValueError(f"train_limit must be between 1 and {len(train_indices)}")
        train_indices = train_indices[:train_limit]

    def normalize(values: NDArray) -> NDArray:
        return values.reshape(len(values), 784).astype(np.float32) / np.float32(255)

    return MNIST(normalize(images[train_indices]), labels[train_indices].astype(np.int64),
                 normalize(images[val_indices]), labels[val_indices].astype(np.int64),
                 normalize(test_images), test_labels.astype(np.int64))
