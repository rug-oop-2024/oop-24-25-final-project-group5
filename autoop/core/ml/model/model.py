from abc import ABC, abstractmethod
from autoop.core.ml.artifact import Artifact
import numpy as np
from copy import deepcopy
import pickle


class Model(ABC):
    """Abstract base class for classification and regression models.
    Implements abstract methods fit and predict, method to_artifact
    and all getter and setter methods.
    """

    _parameters: dict = {}
    _hyperparameters: dict = {}
    _hyperparameter_descriptions: dict = {}
    type: str = ""

    def to_artifact(self, name: str) -> Artifact:
        """Turns the model into an artifact.

        Arguments:
            name (str): name given to the artifact.

        Returns:
            Artifact of the model.
        """
        model_artifact = Artifact(
            name=name,
            data=pickle.dumps(self),
            type=f"model:{self.type}"
        )
        return model_artifact

    @abstractmethod
    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """Abstract fit method to be implemented
        in every subclass of Model.

        Arguments:
            observations (np.ndarray): row(s) of a dataset
                                       used for training.
            ground_truth (np.ndarray): value of response for
                                        given observations.
        """
        pass

    @abstractmethod
    def predict(self, observations: np.ndarray) -> np.ndarray:
        """Abstract predict method to be implemented
        in every subclass of Model.

        Arguments:
            observations (np.ndarray): row(s) of a dataset
                                       used for predicting.
        Returns:
            predicted behavior of observation as np.ndarray.
        """
        pass

    @property
    def parameters(self) -> dict:
        """Getter decorator which returns a deepcopy of self._parameters."""
        return deepcopy(self._parameters)

    @parameters.setter
    def parameters(self, params: dict) -> None:
        """Setter decorator which sets self._parameters to
        the dict passed through as an argument.
        """
        self._parameters = params

    @property
    def hyperparameters(self) -> dict:
        """Getter decorator which returns a
        deepcopy of self._hyperparameters.
        """
        return deepcopy(self._hyperparameters)

    @hyperparameters.setter
    def hyperparameters(self, new_hyperparams: dict) -> None:
        """Setter decorator which sets self._hyperparameters to
        the dict passed through as an argument.
        """
        self._hyperparameters = new_hyperparams

    @property
    def hyperparameter_descriptions(self) -> dict:
        """Getter decorator which returns a
        deepcopy of self._hyperparameter_descriptions.
        """
        return deepcopy(self._hyperparameter_descriptions)

    @hyperparameter_descriptions.setter
    def hyperparameter_descriptions(self, new_hyperparam_desc: dict) -> None:
        """Setter decorator which sets self._hyperparameter_descriptions
        to the dict passed through as an argument.
        """
        self._hyperparameter_descriptions = new_hyperparam_desc
