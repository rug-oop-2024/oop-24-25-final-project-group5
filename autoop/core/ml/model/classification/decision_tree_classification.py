from autoop.core.ml.model.model import Model
import numpy as np
from sklearn.tree import DecisionTreeClassifier


class DecisionTreeClassification(Model):
    def __init__(self, criterion="gini", max_depth=-1,
                 min_samples_split=2) -> None:
        super().__init__()
        self.type = "classification"
        # initialize with hyperparamters
        self._dt_model = DecisionTreeClassifier(
            criterion=criterion,
            max_depth=max_depth if max_depth != -1 else None,
            min_samples_split=min_samples_split
        )

        self.hyperparameters = {
            "criterion": criterion,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split
        }

        self.hyperparameter_descriptions = {
            "criterion": "The function to measure the quality of a split."
                         "Supported criteria are 'gini' for the Gini impurity"
                         "and 'entropy' for the information gain.",
            "max_depth": "The maximum depth of the tree. If None, then nodes"
                         "are expanded until all leaves are pure or until all"
                         "leaves contain less than min_samples_split samples.",
            "min_samples_split": "The minimum number of samples required to"
                                 "split an internal node."
        }

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        # fit the Decision Tree model
        self._dt_model.fit(observations, ground_truth)
        # store parameters
        self.parameters = {
            "feature_importances_": self._dt_model.feature_importances_,
            "n_node_samples": self._dt_model.n_node_samples,
            "tree_": self._dt_model.tree_
        }

    def predict(self, observations: np.ndarray) -> np.ndarray:
        # make predictions using fitted model
        return self._dt_model.predict(observations)
