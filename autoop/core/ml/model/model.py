from abc import ABC, abstractmethod
from autoop.core.ml.artifact import Artifact
import numpy as np
from copy import deepcopy
import pickle


class Model(ABC):
    _parameters: dict = {}
    _hyperparameters: dict = {}
    _hyperparameter_descriptions: dict = {}
    type: str = ""

    def to_artifact(self, name: str) -> Artifact:
        model_artifact = Artifact(
            name=name,
            data=pickle.dumps(self),
            type=f"model:{self.type}"
        )
        return model_artifact

    @abstractmethod
    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        pass

    @abstractmethod
    def predict(self, observations: np.ndarray) -> np.ndarray:
        pass

    @property
    def parameters(self) -> dict:
        return deepcopy(self._parameters)

    @parameters.setter
    def parameters(self, params: dict) -> None:
        self._parameters = params

    @property
    def hyperparameters(self) -> dict:
        return deepcopy(self._hyperparameters)

    @hyperparameters.setter
    def hyperparameters(self, new_hyperparams: dict) -> None:
        self._hyperparameters = new_hyperparams

    @property
    def hyperparameter_descriptions(self) -> dict:
        return deepcopy(self._hyperparameter_descriptions)

    @hyperparameter_descriptions.setter
    def hyperparameter_descriptions(self, new_hyperparam_desc: dict) -> None:
        self._hyperparameter_descriptions = new_hyperparam_desc
