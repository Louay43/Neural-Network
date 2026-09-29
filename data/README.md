# MNIST cache

`train.py` automatically downloads the four original gzip-compressed IDX files from the [Google-hosted MNIST mirror](https://storage.googleapis.com/cvdf-datasets/mnist/). The loader uses Python's standard library, validates the known file checksums and IDX headers, and uses NumPy for preprocessing. No machine-learning framework is used to load or train the network.

Cached data is excluded from Git. For offline use, place these original files here:

- `train-images-idx3-ubyte.gz`
- `train-labels-idx1-ubyte.gz`
- `t10k-images-idx3-ubyte.gz`
- `t10k-labels-idx1-ubyte.gz`

The official training set contains 60,000 images. A seeded permutation reserves 5,000 for validation by default; the official 10,000-image test set is kept separate. Images are flattened from 28x28 to 784 pixels and divided by 255. Labels remain integer class indices. No training-set statistics are needed for this scaling.

MNIST was introduced by Yann LeCun, Corinna Cortes, and Christopher J. C. Burges. Dataset ownership remains with its original providers; this project does not redistribute it in source control.
