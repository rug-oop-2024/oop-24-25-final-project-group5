from autoop.core.ml.model.model import Model
import numpy as np
from sklearn.ensemble import RandomForestClassifier


class RandomForestClassification(Model):
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2):
        super().__init__()
        # initialize with hyperparameters
        self._rf_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split
        )

        # store hyperparameters
        self.hyperparameters = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split
        }

        self.hyperparameter_descriptions = {
            "n_estimators": "The number of trees in the forest.",
            "max_depth": "The maximum depth of the tree.",
            "min_samples_split": "The minimum number of samples required to split an internal node."
        }

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        # fit random forest model
        self._rf_model.fit(observations, ground_truth)

        # store parameters
        self.parameters["feature_importances_"] = (
            self._rf_model.feature_importances_
        )
        self.parameters["n_classes_"] = self._rf_model.n_classes_
        self.parameters["n_features_"] = self._rf_model.n_features_in_
        self.parameters["n_outputs"] = self._rf_model.n_outputs_

    def predict(self, observations: np.ndarray) -> np.ndarray:
        # make predictions using fitted model
        return self._rf_model.predict(observations)
