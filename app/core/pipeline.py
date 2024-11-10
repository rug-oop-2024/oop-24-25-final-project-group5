import pickle
import numpy as np
import streamlit as st

from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature
from autoop.core.ml.pipeline import Pipeline
from autoop.functional.preprocessing import preprocess_features


def load_pipeline(pipelines) -> None:
    """Function to load a pipeline from the registry."""
    pipeline_artifact = st.selectbox(
        "Please choose the pipeline you want to use:",
        options=pipelines,
        format_func=lambda pipeline: f"{pipeline.name} - {pipeline.version}"
    )

    if pipeline_artifact:
        current_pipeline = pickle.loads(pipeline_artifact.read())
        st.session_state.pipeline = current_pipeline


def run_pipeline_prediction(pipeline: Pipeline,
                            dataset: Dataset,
                            features: list[Feature]) -> np.ndarray:
    """Function to run predictions using the selected pipeline.

    Arguments:
        pipeline (Pipeline): the pipeline to use for predictions.
        dataset (Dataset): the dataset to use for predictions.
        features (list[Feature]): feature(s) to use for predictions.

    Returns:
        the predictions as an np.ndarray.
    """
    input_results = preprocess_features(features, dataset)
    input_vectors = (
        [data for (feature_name, data, artifact) in input_results]
    )
    X = np.concatenate(input_vectors, axis=1)
    return pipeline.model.predict(X)
