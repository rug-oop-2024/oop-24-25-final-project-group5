from autoop.core.ml.model.model import Model
import numpy as np
from sklearn.ensemble import RandomForestClassifier


class RandomForestClassification(Model):
    """
    Random forest classification model.
    """
    def __init__(self,
                 n_estimators: int = 100,
                 max_depth: int = -1,
                 min_samples_split: int = 2) -> None:
        """Initializes the model and sets the hyperparameters
        based on type of model. Hyperparameters are listed as arguments.

        Arguments:
            n_estimators (int): number of trees in the forest,
                                default is 100.
            max_depth (int): maximum depth of the tree, default is -1
                             and signals no maximum depth.
            min_samples_split (int): minimum number of samples
                                     required to split an internal
                                     node, default is 2.
        """
        if n_estimators <= 0:
            raise ValueError("Model must have a positive amount of "
                             "estimators.")
        if max_depth != -1 and max_depth < 0:
            raise ValueError("Max depth should be bigger than 0 or -1 "
                             "for no limit.")
        if min_samples_split < 2:
            raise ValueError("Min samples split must be at least 2.")

        super().__init__()
        self.type = "classification"
        self.hyperparameters = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split
        }

        self.hyperparameter_descriptions = {
            "n_estimators": "The number of trees in the forest.",
            "max_depth": "The maximum depth of the tree, -1 sets the maximum "
                         "depth to zero.",
            "min_samples_split": "The minimum number of samples required "
                                 "to split an internal node."
        }

        self._rf_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth if max_depth != -1 else None,
            min_samples_split=min_samples_split
        )

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """Method that fits the model based on
        observations and their ground truth and stores
        the model's parameters in a dictionary.

        Arguments:
            observations (np.ndarray): row(s) of a dataset
                                       used for training.
            ground_truth (np.ndarray): value of response for
                                        given observations.
        """
        self._rf_model.fit(observations, ground_truth)

        self._parameters = {
            "feature_importances_": self._rf_model.feature_importances_,
            "n_classes_": self._rf_model.n_classes_,
            "n_features_": self._rf_model.n_features_in_,
            "n_outputs": self._rf_model.n_outputs_
        }

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """Method that returns predictions based on
        a set of observations.

        Arguments:
            observations (np.ndarray): row(s) of a dataset
                                       used for predicting.
        Returns:
            predicted behavior of observation as np.ndarray.
        """
        return self._rf_model.predict(observations)
