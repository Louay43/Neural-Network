import unittest

import numpy as np
from numpy.testing import assert_allclose

from neural_network import CategoricalCrossEntropy, Softmax


class LossTests(unittest.TestCase):
    def test_sparse_and_one_hot(self):
        loss = CategoricalCrossEntropy()
        probabilities = np.array([[0.2, 0.8], [0.6, 0.4]])
        y = np.array([1, 0])
        self.assertAlmostEqual(loss.forward(probabilities, y), -np.log([0.8, 0.6]).mean())
        self.assertEqual(loss.forward(probabilities, y), loss.forward(probabilities, np.eye(2)[y]))
        assert_allclose(loss.backward(probabilities, y), [[0, -0.625], [-1 / 1.2, 0]])

    def test_separate_and_fused_derivatives_agree(self):
        logits = np.array([[0.3, -0.5, 1.2], [0.4, 0.1, -0.3]])
        y = np.array([2, 0])
        softmax, loss = Softmax(), CategoricalCrossEntropy()
        p = softmax.forward(logits, True)
        assert_allclose(softmax.backward(loss.backward(p, y)), loss.backward_logits(logits, y), atol=1e-11)
        self.assertAlmostEqual(loss.forward(p, y), loss.forward_logits(logits, y), places=10)

    def test_extreme_logits_retain_loss_and_gradient(self):
        loss = CategoricalCrossEntropy()
        logits = np.array([[1000., -1000.]])
        self.assertEqual(loss.forward_logits(logits, np.array([1])), 2000)
        assert_allclose(loss.backward_logits(logits, np.array([1])), [[1, -1]])

    def test_invalid_labels(self):
        for y in (np.array([-1]), np.array([2]), np.array([0.5]), np.array([[0.2, 0.2]])):
            with self.assertRaises(ValueError):
                CategoricalCrossEntropy().forward(np.array([[0.5, 0.5]]), y)

    def test_integer_logits_preserve_soft_targets(self):
        loss = CategoricalCrossEntropy()
        logits = np.array([[1, 2]])
        y = np.array([[0.25, 0.75]])
        self.assertAlmostEqual(loss.forward_logits(logits, y), loss.forward_logits(logits.astype(float), y))
        assert_allclose(loss.backward_logits(logits, y), loss.backward_logits(logits.astype(float), y))
