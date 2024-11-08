from autoop.core.ml.model.model import Model
import numpy as np
from sklearn.tree import DecisionTreeClassifier


class DecisionTreeClassification(Model):
    """
    A wrapper class of the model DecisionTreeClassifier
    from the library sklearn.
    """

    def __init__(self,
                 criterion: str = "gini",
                 max_depth: int = -1,
                 min_samples_split: int = 2) -> None:
        """Initializes the model and sets the hyperparameters
        based on type of model. Hyperparameters are listed as arguments.

        Arguments:
            criterion (str): function to measure quality of split,
                             either 'gini' or 'entropy', default is
                             'gini'.
            max_depth (int): maximum depth of the tree, default is -1
                             and signals no maximum depth.
            min_samples_split (int): minimum number of samples
                                     required to split an internal
                                     node, default is 2.
        """
        if criterion not in ['gini', 'entropy']:
            raise ValueError('Criterion should be either "gini" '
                             'or "entropy".')
        if max_depth != -1 and max_depth < 0:
            raise ValueError("Max depth should be in range [1, inf) "
                             "or -1 for no limit.")
        if min_samples_split < 2:
            raise ValueError("Min samples split should be in range "
                             "[2, inf).")

        super().__init__()
        self.type = "classification"
        self.hyperparameters = {
            "criterion": criterion,
            "max_depth": max_depth if max_depth != -1 else None,
            "min_samples_split": min_samples_split
        }

        self.hyperparameter_descriptions = {
            "criterion": "The function to measure the quality of a split."
                         "Supported criteria are 'gini' for the Gini impurity "
                         "and 'entropy' for the information gain.",
            "max_depth": "The maximum depth of the tree. If None, then nodes"
                         "are expanded until all leaves are pure or until all "
                         "leaves contain less than min_samples_split samples.",
            "min_samples_split": "The minimum number of samples required to "
                                 "split an internal node."
        }
        self._dt_model = DecisionTreeClassifier(
            criterion=criterion,
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
        self._dt_model.fit(observations, ground_truth)
        self.parameters = {
            "feature_importances_": self._dt_model.feature_importances_,
            "n_node_samples": self._dt_model.tree_.n_node_samples,
            "tree_": self._dt_model.tree_
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
        return self._dt_model.predict(observations)
