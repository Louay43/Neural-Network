import unittest

import numpy as np
from numpy.testing import assert_allclose

from neural_network import Adam, Dense, Momentum, SGD


class OptimizerTests(unittest.TestCase):
    def make_layer(self):
        layer = Dense(1, 2)
        layer.weights[:] = 1
        layer.biases[:] = 0
        layer.dW[:] = [[2, -3]]
        layer.db[:] = [[0.5, -0.5]]
        return layer

    def test_sgd(self):
        layer = self.make_layer()
        SGD(0.1).step([layer])
        assert_allclose(layer.weights, [[0.8, 1.3]])
        assert_allclose(layer.biases, [[-0.05, 0.05]])

    def test_momentum_accumulates(self):
        layer, optimizer = self.make_layer(), Momentum(0.1, 0.9)
        optimizer.step([layer])
        optimizer.step([layer])
        assert_allclose(layer.weights, [[0.42, 1.87]])
        assert_allclose(layer.biases, [[-0.145, 0.145]])

    def test_adam_multistep_and_bias_correction(self):
        layer, optimizer = self.make_layer(), Adam(0.01, beta1=0.8, beta2=0.9)
        expected, m, v = np.ones((1, 2)), np.zeros((1, 2)), np.zeros((1, 2))
        for t, gradient in enumerate(([[2., -3.]], [[-1., 4.]], [[0.5, 0.2]]), 1):
            layer.dW[:] = gradient
            g = np.array(gradient)
            m = 0.8 * m + 0.2 * g
            v = 0.9 * v + 0.1 * g ** 2
            expected -= 0.01 * (m / (1 - 0.8 ** t)) / (np.sqrt(v / (1 - 0.9 ** t)) + 1e-8)
            optimizer.step([layer])
            assert_allclose(layer.weights, expected, atol=1e-14)
            self.assertEqual(optimizer.t, t)
        assert_allclose(layer.biases, [[-0.03, 0.03]], atol=1e-8)

    def test_invalid_settings_and_nonfinite_gradient(self):
        for factory in (lambda: SGD(-1), lambda: Momentum(momentum=1), lambda: Adam(beta1=1)):
            with self.assertRaises(ValueError):
                factory()
        layer = self.make_layer()
        layer.db[0, 0] = np.nan
        with self.assertRaises(ValueError):
            SGD().step([layer])
        assert_allclose(layer.weights, 1)
