import unittest
import numpy as np

from autoop.core.ml.model.regression import PolynomialRegression

class TestPolynomialRegression(unittest.TestCase):

    def setUp(self):
        # Setting up example data for testing
        self.observations = np.array([[1], [2], [3], [4], [5]])
        self.ground_truth = np.array([1, 4, 9, 16, 25])  # y = x^2 for simplicity
        self.poly_reg = PolynomialRegression(degree=2)
        self.poly_reg.fit(self.observations, self.ground_truth)

    def test_init(self):
        # Test that the model initializes correctly
        self.assertEqual(self.poly_reg.hyperparameters["degree"], 2)
        self.assertEqual(self.poly_reg.type, "regression")
        self.assertIn("degree", self.poly_reg.hyperparameter_descriptions)

    def test_fit(self):
        # Test that fitting stores parameters correctly
        self.assertIsNotNone(self.poly_reg._parameters["coef_"])
        self.assertIsNotNone(self.poly_reg._parameters["intercept_"])

    def test_predict(self):
        # Test that predictions are made correctly
        test_observations = np.array([[6], [7]])
        predictions = self.poly_reg.predict(test_observations)
        self.assertEqual(predictions.shape[0], test_observations.shape[0])
        self.assertTrue(np.all(predictions >= 0))  # Ensure that predictions are valid (positive for this case)

    def test_predict_with_known_input(self):
        # Test that known input gives expected output (since we know y = x^2)
        test_observations = np.array([[6], [7], [8]])
        predictions = self.poly_reg.predict(test_observations)
        expected_predictions = np.array([36, 49, 64])
        np.testing.assert_array_almost_equal(predictions, expected_predictions)

    def test_invalid_degree(self):
        # Test that initializing with an invalid degree raises an error
        with self.assertRaises(ValueError):
            PolynomialRegression(degree=-1)  # Negative degree should be invalid

    def test_predict_untrained_model(self):
        # Test that predictions cannot be made on an untrained model
        poly_reg_untrained = PolynomialRegression(degree=2)
        test_observations = np.array([[6], [7]])
        with self.assertRaises(AttributeError):  # Expecting an AttributeError as fit is not called
            poly_reg_untrained.predict(test_observations)
