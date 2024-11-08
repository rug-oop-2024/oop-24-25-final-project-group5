import unittest
import numpy as np

from autoop.core.ml.model.regression import LassoRegression

class TestLassoRegression(unittest.TestCase):

    def setUp(self):
        # Setting up example data for testing
        self.observations = np.array([[1, 2], [2, 3], [3, 4], [5, 6], [6, 7]])
        self.ground_truth = np.array([1, 2, 3, 4, 5])
        self.lasso = LassoRegression(alpha=0.5, max_iter=500, tol=0.0001, selection='random')
        self.lasso.fit(self.observations, self.ground_truth)

    def test_init(self):
        # Test that the model initializes correctly
        self.assertEqual(self.lasso.hyperparameters["alpha"], 0.5)
        self.assertEqual(self.lasso.hyperparameters["max_iter"], 500)
        self.assertEqual(self.lasso.hyperparameters["selection"], 'random')
        self.assertEqual(self.lasso.type, "regression")
        self.assertIn("alpha", self.lasso.hyperparameter_descriptions)

    def test_fit(self):
        # Test that fitting stores the model parameters correctly
        self.assertTrue("coeff_" in self.lasso.parameters)
        self.assertTrue("intercept_" in self.lasso.parameters)

    def test_predict(self):
        # Test predict method with multiple observations
        test_observations = np.array([[1, 2], [6, 7], [3, 3]])
        predictions = self.lasso.predict(test_observations)
        self.assertEqual(predictions.shape[0], test_observations.shape[0])

    def test_invalid_alpha(self):
        # Test that initializing with invalid alpha (<= -1) raises an error
        with self.assertRaises(ValueError):
            LassoRegression(alpha=-1)

    def test_predict_correctness(self):
        # Test prediction with known ground truth
        test_observations = np.array([[1, 2], [6, 7], [3, 3]])
        self.lasso.fit(self.observations, self.ground_truth)
        predictions = self.lasso.predict(test_observations)
        self.assertTrue(np.allclose(predictions, self.lasso._lasso_model.predict(test_observations)))
