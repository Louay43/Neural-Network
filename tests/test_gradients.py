import unittest

import numpy as np
from numpy.testing import assert_allclose

from neural_network import Dense, Dropout, NeuralNetwork, ReLU, Sigmoid, Softmax
from neural_network.gradient_check import check_gradients


class GradientTests(unittest.TestCase):
    def test_complete_chain_including_l2_and_inputs(self):
        for activation in (ReLU, Sigmoid):
            model = NeuralNetwork([Dense(3, 4, l2=0.07, seed=8), activation(), Dense(4, 3, seed=9), Softmax()])
            X = np.random.default_rng(0).normal(size=(5, 3))
            before = [layer.weights.copy() for layer in model.dense_layers]
            report = check_gradients(model, X, np.array([0, 1, 2, 1, 0]))
            for item in report:
                self.assertTrue(item["passed"], item)
                self.assertLess(item["relative_l2_error"], 1e-7)
            for layer, original in zip(model.dense_layers, before):
                assert_allclose(layer.weights, original, rtol=0, atol=0)

    def test_rejects_float32_and_stochastic_dropout(self):
        for layers in ([Dense(2, 2, dtype="float32"), Softmax()], [Dense(2, 2), Dropout(0.2), Softmax()]):
            with self.assertRaises(ValueError):
                check_gradients(NeuralNetwork(layers), np.ones((1, 2)), np.array([0]))

    def test_dropout_derivative_with_fixed_mask(self):
        X = np.random.default_rng(9).normal(size=(2, 3))
        upstream = np.random.default_rng(2).normal(size=(2, 3))
        dropout = Dropout(0.3, seed=7)
        dropout.forward(X, True)
        actual = dropout.backward(upstream)
        numeric = np.zeros_like(X)
        for index in np.ndindex(X.shape):
            plus, minus = X.copy(), X.copy()
            plus[index] += 1e-5
            minus[index] -= 1e-5
            a = np.sum(Dropout(0.3, seed=7).forward(plus, True) * upstream)
            b = np.sum(Dropout(0.3, seed=7).forward(minus, True) * upstream)
            numeric[index] = (a - b) / 2e-5
        assert_allclose(actual, numeric, atol=1e-9)
