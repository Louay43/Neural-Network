import tempfile
import unittest
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

from neural_network import Adam, Dense, Dropout, NeuralNetwork, ReLU, Softmax


class NetworkTests(unittest.TestCase):
    def model(self, dropout=0.0):
        return NeuralNetwork([Dense(2, 12, seed=1), ReLU(), Dropout(dropout, seed=2), Dense(12, 2, seed=3), Softmax()], seed=4)

    def test_learns_nonlinear_data_and_reproduces_seed(self):
        X = np.random.default_rng(7).normal(size=(120, 2))
        y = (X[:, 0] * X[:, 1] > 0).astype(int)
        first, second = self.model(), self.model()
        before = first.evaluate(X, y)["loss"]
        for model in (first, second):
            model.fit(X, y, epochs=45, batch_size=17, optimizer=Adam(0.02), validation_data=(X, y), verbose=False)
        self.assertLess(first.evaluate(X, y)["loss"], before * 0.2)
        self.assertGreater(first.evaluate(X, y)["accuracy"], 0.95)
        assert_allclose(first.predict_proba(X), second.predict_proba(X), atol=0, rtol=0)

    def test_evaluation_batch_weighting_and_dropout_off(self):
        model = self.model(0.5)
        X = np.random.default_rng(8).normal(size=(11, 2))
        y = np.arange(11) % 2
        a, b = model.evaluate(X, y, 3), model.evaluate(X, y, 11)
        self.assertAlmostEqual(a["loss"], b["loss"])
        self.assertEqual(a["accuracy"], b["accuracy"])
        assert_allclose(model.predict_proba(X), model.predict_proba(X))

    def test_checkpoint_roundtrip(self):
        model = self.model(0.25)
        X = np.array([[1., 2.], [-0.3, 1.]])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            model.save(path)
            restored = NeuralNetwork.load(path)
            assert_allclose(restored.predict_proba(X), model.predict_proba(X), atol=0, rtol=0)
            self.assertEqual(restored.layers[2].rate, 0.25)

    def test_input_validation(self):
        model = self.model()
        with self.assertRaises(ValueError):
            model.fit(np.ones((2, 2)), np.array([0, 1]), batch_size=0)
        with self.assertRaises(ValueError):
            model.predict(np.array([[np.nan, 1.]]))
        with self.assertRaises(ValueError):
            NeuralNetwork([Dense(2, 3), Dense(4, 2), Softmax()])
        with self.assertRaises(RuntimeError):
            model.backward(np.array([0]))
