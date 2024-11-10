import unittest
import numpy as np

from autoop.core.ml.model.regression import MultipleLinearRegression

class TestMultipleLinearRegression(unittest.TestCase):

    def setUp(self):
        # Setting up example data for testing
        self.observations = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        self.ground_truth = np.array([5, 7, 9, 11])
        self.mlr = MultipleLinearRegression()
        self.mlr.fit(self.observations, self.ground_truth)

    def test_init(self):
        # Test that the model initializes correctly
        self.assertEqual(self.mlr.type, "regression")

    def test_fit(self):
        # Test that fitting stores the model parameters correctly
        self.assertIn("parameters", self.mlr.parameters)

    def test_predict(self):
        # Test predict method with multiple observations
        test_observations = np.array([[5, 6], [6, 7]])
        predictions = self.mlr.predict(test_observations)
        self.assertEqual(predictions.shape[0], test_observations.shape[0])

    def test_predict_untrained_model(self):
        # Test predict method on untrained model
        untrained_mlr = MultipleLinearRegression()
        with self.assertRaises(KeyError):  # Expect KeyError when no parameters are set
            untrained_mlr.predict(self.observations)

    def test_prediction_accuracy(self):
        # Test prediction correctness by comparing with manually computed coefficients
        predictions = self.mlr.predict(self.observations)
        expected_predictions = np.array([5, 7, 9, 11])
        self.assertTrue(np.allclose(predictions, expected_predictions))

    def test_invalid_fit(self):
        # Test invalid fit when observations and ground_truth have mismatched sizes
        with self.assertRaises(ValueError):
            self.mlr.fit(self.observations, np.array([5, 7]))  # Incorrect length for ground_truth
