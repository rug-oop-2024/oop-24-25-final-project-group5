import unittest
import numpy as np

from autoop.core.ml.model.classification import DecisionTreeClassification

class TestDecisionTreeClassification(unittest.TestCase):

    def setUp(self) -> None:
        # Initialize the model with default parameters
        self.model = DecisionTreeClassification()
        # Sample data for testing
        self.observations = np.array([[0, 1], [1, 1], [0, 0], [1, 0]])
        self.ground_truth = np.array([0, 1, 0, 1])
        self.test_data = np.array([[0, 1], [1, 1]])

    def test_init(self):
        # Test that the model initializes correctly with default parameters
        self.assertEqual(self.model.type, "classification")
        self.assertEqual(self.model.hyperparameters["criterion"], "gini")
        self.assertEqual(self.model.hyperparameters["max_depth"], -1)
        self.assertEqual(self.model.hyperparameters["min_samples_split"], 2)

    def test_fit(self):
        # Test the fit method to ensure parameters are set after training
        self.model.fit(self.observations, self.ground_truth)
        self.assertIn("feature_importances_", self.model.parameters)
        self.assertIn("n_node_samples", self.model.parameters)
        self.assertIn("tree_", self.model.parameters)

    def test_predict(self):
        # Train the model first
        self.model.fit(self.observations, self.ground_truth)
        # Test predictions
        predictions = self.model.predict(self.test_data)
        self.assertEqual(predictions.shape, (self.test_data.shape[0],))
        # Check if the predictions contain only expected values
        self.assertTrue(set(predictions).issubset(set(self.ground_truth)))

    def test_hyperparameters(self):
        # Test if hyperparameters are correctly stored in the model
        model = DecisionTreeClassification(criterion="entropy", max_depth=5, min_samples_split=3)
        self.assertEqual(model.hyperparameters["criterion"], "entropy")
        self.assertEqual(model.hyperparameters["max_depth"], 5)
        self.assertEqual(model.hyperparameters["min_samples_split"], 3)