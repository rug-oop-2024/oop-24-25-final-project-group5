import unittest
import numpy as np

from autoop.core.ml.model.classification import RandomForestClassification

class TestRandomForestClassification(unittest.TestCase):

    def setUp(self):
        # Setting up example data for testing
        self.observations = np.array([[1, 2], [2, 3], [3, 4], [5, 6], [6, 7]])
        self.ground_truth = np.array([0, 1, 0, 1, 0])
        self.rf = RandomForestClassification(n_estimators=10, max_depth=3)
        self.rf.fit(self.observations, self.ground_truth)

    def test_init(self):
        # Test that the model initializes correctly
        self.assertEqual(self.rf.hyperparameters["n_estimators"], 10)
        self.assertEqual(self.rf.hyperparameters["max_depth"], 3)
        self.assertEqual(self.rf.type, "classification")
        self.assertIn("n_estimators", self.rf.hyperparameter_descriptions)

    def test_fit(self):
        # Test that fitting stores the model parameters correctly
        self.assertTrue("feature_importances_" in self.rf._parameters)
        self.assertTrue("n_classes_" in self.rf._parameters)

    def test_predict(self):
        # Test predict method with multiple observations
        test_observations = np.array([[1, 2], [6, 7], [3, 3]])
        predictions = self.rf.predict(test_observations)
        self.assertEqual(predictions.shape[0], test_observations.shape[0])

    def test_invalid_n_estimators(self):
        # Test that invalid n_estimators (less than or equal to 0) raises an error
        with self.assertRaises(ValueError):
            RandomForestClassification(n_estimators=0)

    def test_predict_works(self):
        # Test prediction with known ground truth
        test_observations = np.array([[5, 6], [6, 7], [1, 2]])
        self.rf.fit(self.observations, self.ground_truth)
        predictions = self.rf.predict(test_observations)
        self.assertTrue(all(p in self.ground_truth for p in predictions))
