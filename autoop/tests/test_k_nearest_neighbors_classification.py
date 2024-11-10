import unittest
import numpy as np
from collections import Counter

from autoop.core.ml.model.classification import KNearestNeighborsClassification

class TestKNearestNeighborsClassification(unittest.TestCase):

    def setUp(self):
        # Setting up example data for testing
        self.observations = np.array([[1, 2], [2, 3], [3, 4], [5, 6], [6, 7]])
        self.ground_truth = np.array([[0, 1], [1, 0], [0, 1], [1, 0], [0, 1]])
        self.knn = KNearestNeighborsClassification(k=3)
        self.knn.fit(self.observations, self.ground_truth)

    def test_init(self):
        # Test that the model initializes correctly
        self.assertEqual(self.knn.hyperparameters["k"], 3)
        self.assertEqual(self.knn.type, "classification")
        self.assertIn("k", self.knn.hyperparameter_descriptions)

    def test_invalid_k(self):
        # Test that initializing with invalid k raises an error
        with self.assertRaises(ValueError):
            KNearestNeighborsClassification(k=0)

    def test_fit(self):
        # Test that fitting stores observations and ground truth correctly
        self.assertTrue(np.array_equal(self.knn.parameters["observations"], self.observations))
        self.assertTrue(np.array_equal(self.knn.parameters["ground_truth"], self.ground_truth))

    def test_predict_single(self):
        # Test the private method _predict_single
        observation = np.array([4, 5])
        prediction = self.knn._predict_single(observation)
        self.assertIn(prediction, self.ground_truth)  # Ensures it predicts a valid label

    def test_predict(self):
        # Test predict method with multiple observations
        test_observations = np.array([[1, 2], [6, 7], [3, 3]])
        predictions = self.knn.predict(test_observations)
        self.assertEqual(predictions.shape[0], test_observations.shape[0])

    def test_predict_majority_class(self):
        # Test prediction with known majority class
        test_observations = np.array([[5, 6], [6, 7], [1, 2]])
        self.knn.fit(self.observations, self.ground_truth)
        predictions = self.knn.predict(test_observations)
        self.assertTrue(all(p in self.ground_truth for p in predictions))