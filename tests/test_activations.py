import unittest

import numpy as np
from numpy.testing import assert_allclose

from neural_network import ReLU, Sigmoid, Softmax


class ActivationTests(unittest.TestCase):
    def test_relu(self):
        activation = ReLU()
        assert_allclose(activation.forward(np.array([[-2., 0., 3.]]), True), [[0, 0, 3]])
        assert_allclose(activation.backward(np.ones((1, 3))), [[0, 0, 1]])

    def test_sigmoid_extreme_values(self):
        activation = Sigmoid()
        with np.errstate(over="raise"):
            values = activation.forward(np.array([[-1000., 0., 1000.]]), True)
        assert_allclose(values, [[0, 0.5, 1]])
        assert_allclose(activation.backward(np.ones((1, 3))), [[0, 0.25, 0]])

    def test_softmax_shift_invariance_and_jacobian(self):
        activation = Softmax()
        X = np.array([[1., 2., 3.], [-1000., 0., 1000.]])
        expected = activation.forward(X)
        assert_allclose(activation.forward(X + 10000, True), expected)
        assert_allclose(expected.sum(axis=1), 1)
        upstream = np.array([[0.2, -0.5, 0.7], [2., 3., -1.]])
        explicit = np.stack([(np.diag(p) - np.outer(p, p)) @ g for p, g in zip(expected, upstream)])
        assert_allclose(activation.backward(upstream), explicit, atol=1e-15)
