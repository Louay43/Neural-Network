import unittest
from dataclasses import replace

import numpy as np
from numpy.testing import assert_array_equal

from neural_network import Dense, NeuralNetwork, Softmax
from neural_network.workflows import ModelConfig, build_model


class WorkflowTests(unittest.TestCase):
    def test_dropout_ablation_preserves_initial_weights(self):
        config = ModelConfig(dropout=0)
        baseline = build_model(config, seed=42)
        dropout = build_model(replace(config, dropout=0.2), seed=42)
        for first, second in zip(baseline.dense_layers, dropout.dense_layers):
            assert_array_equal(first.weights, second.weights)

    def test_one_hot_and_sparse_training_match(self):
        X = np.array([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]])
        y = np.array([0, 1, 1, 0])
        first = NeuralNetwork([Dense(2, 2, seed=4), Softmax()], seed=6)
        second = NeuralNetwork([Dense(2, 2, seed=4), Softmax()], seed=6)
        first.fit(X, y, epochs=3, batch_size=3, verbose=False)
        second.fit(X, np.eye(2)[y], epochs=3, batch_size=3, verbose=False)
        assert_array_equal(first.predict_proba(X), second.predict_proba(X))
