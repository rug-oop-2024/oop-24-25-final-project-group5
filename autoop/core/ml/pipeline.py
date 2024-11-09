import pickle

import numpy as np
from copy import deepcopy

from autoop.core.ml.artifact import Artifact
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature
from autoop.core.ml.metric import Metric
from autoop.core.ml.model import Model
from autoop.functional.preprocessing import preprocess_features


class Pipeline():
    """Pipeline object for modelling."""

    def __init__(self,
                 metrics: list[Metric],
                 dataset: Dataset,
                 model: Model,
                 input_features: list[Feature],
                 target_feature: Feature,
                 split: float = 0.8,
                 ) -> None:
        """Initializes the pipeline object."""
        self._dataset = dataset
        self._model = model
        self._input_features = input_features
        self._target_feature = target_feature
        self._metrics = metrics
        self._artifacts = {}
        self._split = split
        if target_feature.type == "categorical" and \
                model.type != "classification":
            raise ValueError("Model type must be classification for "
                             "categorical target feature")
        if target_feature.type == "numerical" and \
                model.type != "regression":
            raise ValueError("Model type must be regression for "
                             "numerical target feature")

    def __str__(self) -> str:
        """Returns the pipeline as a formatted string."""
        return f"""
Pipeline(
    model={self._model.__class__.__name__},
    input_features={list(map(str, self._input_features))},
    target_feature={str(self._target_feature)},
    split={self._split},
    metrics={list(map(str, self._metrics))},
)
"""

    @property
    def input_features(self) -> list[Feature]:
        """Returns a deepcopy of the list of input features."""
        return deepcopy(self._input_features)

    @property
    def model(self) -> Model:
        """Returns the model used in the pipeline."""
        return self._model

    @property
    def artifacts(self) -> list[Artifact]:
        """Used to get the artifacts generated during the
        pipeline execution to be saved.

        Returns:
            list of artifacts (list[Artifact])
        """
        artifacts = []
        for name, artifact in self._artifacts.items():
            artifact_type = artifact.get("type")
            if artifact_type in ["OneHotEncoder"]:
                data = artifact["encoder"]
                data = pickle.dumps(data)
                artifacts.append(Artifact(name=name, data=data))
            if artifact_type in ["StandardScaler"]:
                data = artifact["scaler"]
                data = pickle.dumps(data)
                artifacts.append(Artifact(name=name, data=data))
        pipeline_data = {
            "input_features": self._input_features,
            "target_feature": self._target_feature,
            "split": self._split,
        }
        artifacts.append(Artifact(name="pipeline_config",
                                  data=pickle.dumps(pipeline_data)))
        artifacts.append(self._model.to_artifact(
            name=f"pipeline_model_{self._model.type}")
        )
        return artifacts

    def _register_artifact(self, name: str, artifact: Artifact) -> None:
        """Registers an artifact in self._artifacts.

        Arguments:
            name (str): artifact's name
            artifact (Artifact): artifact to save in list
        """
        self._artifacts[name] = artifact

    def _preprocess_features(self) -> None:
        """Preprocesses the features into vectors."""
        (target_feature_name, target_data, artifact) = (
            preprocess_features([self._target_feature], self._dataset)[0]
        )
        self._register_artifact(target_feature_name, artifact)
        input_results = (
            preprocess_features(self._input_features, self._dataset)
        )
        for (feature_name, data, artifact) in input_results:
            self._register_artifact(feature_name, artifact)
        # Get the input vectors and output vector,
        # sort by feature name for consistency
        self._output_vector = target_data
        self._input_vectors = (
            [data for (feature_name, data, artifact) in input_results]
        )

    def _split_data(self) -> None:
        """Split the data into training and testing sets
        and stores those split datasets in multiple attributes.
        """
        split = self._split
        self._train_X = [vector[:int(split * len(vector))]
                         for vector in self._input_vectors]
        self._test_X = [vector[int(split * len(vector)):]
                        for vector in self._input_vectors]
        self._train_y = self._output_vector[
            :int(split * len(self._output_vector))
        ]
        self._test_y = self._output_vector[
            int(split * len(self._output_vector)):
        ]

    def _compact_vectors(self, vectors: list[np.array]) -> np.array:
        """Submethod of compacting the vectors using
        np.concatenate.

        Arguments:
            vectors (list[np.array]): vectors to be compacted

        Returns:
            np.array of compacted vectors
        """
        return np.concatenate(vectors, axis=1)

    def _train(self) -> None:
        """Submethod for training the pipeline's model."""
        X = self._compact_vectors(self._train_X)
        Y = self._train_y
        self._model.fit(X, Y)

    def _evaluate(self) -> None:
        """Submethod for evaluating the testing data."""
        X = self._compact_vectors(self._test_X)
        Y = self._test_y
        self._metrics_results = []
        predictions = self._model.predict(X)
        for metric in self._metrics:
            result = metric.evaluate(predictions, Y)
            self._metrics_results.append((metric, result))
        self._predictions = predictions

    def _evaluate_training_set(self) -> None:
        """Submethod for evaluating the testing data."""
        X = self._compact_vectors(self._train_X)
        Y = self._train_y
        self._training_metrics_results = []
        predictions = self._model.predict(X)
        for metric in self._metrics:
            result = metric.evaluate(predictions, Y)
            self._training_metrics_results.append((metric, result))
        self._training_predictions = predictions

    def execute(self) -> dict[str, np.ndarray | list[tuple[Metric, float]]]:
        """Executes the entire pipeline:
            (1.) Preprocesses features.
            (2.) Splits data into training and testing sets.
            (3.) Trains the model based on the training sets.
            (4.) Evaluates the testing and training sets' predictions

        Returns:
            A dict of predictions and metric(s) based on the configurations.
        """
        self._preprocess_features()
        self._split_data()
        self._train()
        self._evaluate()
        self._evaluate_training_set()
        return {
            "metrics": self._metrics_results,
            "predictions": self._predictions,
            "training_metrics": self._training_metrics_results,
            "training_predictions": self._training_predictions
        }
