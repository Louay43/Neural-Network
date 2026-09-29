import unittest

import numpy as np
from numpy.testing import assert_allclose

from neural_network import Dense, Dropout
from neural_network.initializers import initialize


class DenseTests(unittest.TestCase):
    def test_forward_backward_hand_calculated(self):
        layer = Dense(2, 2, l2=0.1)
        layer.weights[:] = [[1, 2], [3, 4]]
        layer.biases[:] = [[0.5, -0.5]]
        X = np.array([[1., 2.], [3., 4.]])
        assert_allclose(layer.forward(X, training=True), [[7.5, 9.5], [15.5, 21.5]])
        assert_allclose(layer.backward(np.array([[1., 2.], [3., 4.]])), [[5, 11], [11, 25]])
        assert_allclose(layer.dW, [[10.1, 14.2], [14.3, 20.4]])
        assert_allclose(layer.db, [[4, 6]])
        self.assertAlmostEqual(layer.regularization_loss(), 1.5)

    def test_invalid_inputs_and_missing_cache(self):
        layer = Dense(2, 3)
        with self.assertRaises(ValueError):
            layer.forward(np.ones((2, 4)))
        with self.assertRaises(RuntimeError):
            layer.backward(np.ones((2, 3)))
        with self.assertRaises(ValueError):
            Dense(2, 3, l2=-1)

    def test_initialization_variance_and_reproducibility(self):
        for name, variance in (("random", 0.0001), ("xavier", 2 / 1500), ("he", 2 / 1000)):
            weights = initialize(1000, 500, name, np.random.default_rng(3))
            self.assertAlmostEqual(weights.var() / variance, 1, delta=0.02)
        assert_allclose(Dense(3, 2, seed=4).weights, Dense(3, 2, seed=4).weights)
        with self.assertRaises(ValueError):
            initialize(2, 2, "unknown")
        with self.assertRaises(ValueError):
            initialize(0, 2)


class DropoutTests(unittest.TestCase):
    def test_mask_scaling_backward_and_evaluation(self):
        layer = Dropout(0.25, seed=7)
        X = np.ones((1000, 100))
        output = layer.forward(X, training=True)
        self.assertAlmostEqual(output.mean(), 1, delta=0.02)
        self.assertTrue(set(np.unique(output)).issubset({0, 1 / 0.75}))
        assert_allclose(layer.backward(np.ones_like(X)), output)
        assert_allclose(layer.forward(X), X)
        with self.assertRaises(RuntimeError):
            layer.backward(X)

    def test_zero_dropout_and_invalid_rates(self):
        layer = Dropout(0)
        assert_allclose(layer.forward(np.ones((3, 2)), True), 1)
        for rate in (-0.1, 1, float("nan")):
            with self.assertRaises(ValueError):
                Dropout(rate)
