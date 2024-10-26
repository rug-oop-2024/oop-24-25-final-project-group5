from autoop.core.ml.model.model import Model
import numpy as np
from sklearn.tree import DecisionTreeClassifier

class DecisionTree(Model):
    def __init__(self, criterion='gini', max_depth=None, min_samples_split=2):
        super().__init__()
        #initialize with hyperparamters
        self._dt_model = DecisionTreeClassifier(
            criterion = criterion,
            max_depth = max_depth,
            min_samples_split = min_samples_split
        )
        #store hyperparameters
        self._hyperparameters = {
            "criterion": criterion,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split
        }

    def fit(self, observations:np.ndarray, ground_truth: np.ndarray) -> None:
        #fit the Decision Tree model
        self._dt_model.fit(observations, ground_truth)
        #store parameters
        self._parameters["feature_importances_"] = self._dt_model.feature_importances_
        self._parameters["n_node_samples"] = self._dt_model.n_node_samples
        self._parameters["tree_"] = self._dt_model.tree_

    def predict(self, observations: np.ndarray) -> np.ndarray:
        #make predictions using fitted model
        return self._dt_model.predict(observations)
    
    #allow users to read the parameters without modifying them
    @property
    def parameters(self):
        return self._parameters  