from autoop.core.ml.model.model import Model
import numpy as np
from sklearn.ensemble import RandomForestClassifier

class RandomForest(Model):
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2):
        super().__init__()
        #initialize with hyperparameters
        self._rf_model = RandomForestClassifier(
            n_estimators = n_estimators,
            max_depth = max_depth,
            min_samples_split = min_samples_split
        )

        #store hyperparameters
        self._hyperparameters = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split
        }

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        #fit random forest model
        self._rf_model.fit(observations, ground_truth)

        #store parameters
        self._parameters["feature_importances_"] = self._rf_model.feature_importances_
        self._parameters["n_classes_"] = self._rf_model.n_classes_
        self._parameters["n_features_"] = self._rf_model.n_features_in_
        self._parameters["n_outputs"] = self._rf_model.n_outputs_

    def predict(self, observations: np.ndarray) -> np.ndarray:
        #make predictions using fitted model
        return self._rf_model.predict(observations)
    
    #Allow users to read parameters without modifying them
    @property
    def parameters(self):
        return self._parameters