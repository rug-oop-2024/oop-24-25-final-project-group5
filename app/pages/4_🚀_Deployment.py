import pickle

import numpy as np
import pandas as pd
import streamlit as st

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature
from autoop.core.ml.pipeline import Pipeline
from autoop.functional.feature import detect_feature_types
from autoop.functional.preprocessing import preprocess_features

st.set_page_config(page_title="Deployment", page_icon="🚀")

automl = AutoMLSystem.get_instance()
pipelines = automl.registry.list(type="pipeline")


def write_helper_text(text: str) -> None:
    """Function to write helper text in a specific format."""
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)


st.write("# 🚀 Deployment")
write_helper_text("In this section, you can deploy pre-existing pipelines.")


def load_pipeline() -> None:
    """Function to load a pipeline from the registry."""
    pipeline_artifact = st.selectbox(
        "Please choose the pipeline you want to use:",
        options=pipelines,
        format_func=lambda pipeline: f"{pipeline.name} - {pipeline.version}"
    )

    if pipeline_artifact:
        current_pipeline = pickle.loads(pipeline_artifact.read())
        st.session_state.pipeline = current_pipeline
        print(current_pipeline)
        print(current_pipeline.model.parameters)
        print("artifacts", current_pipeline.artifacts)


def choose_input_features(dataset: Dataset) -> list[Feature]:
    """
    Function to choose input features from a dataset.

    Arguments:
        dataset (Dataset): dataset to choose features from.

    Returns:
        list of chosen feature(s).
    """
    dataset_features = detect_feature_types(dataset)

    feature_names = [feature.name for feature in dataset_features]
    feature_name = st.multiselect(
        "Please choose your input features:",
        options=feature_names,
        format_func=lambda feature_name: f"{feature_name}"
    )

    if not feature_name:
        return None

    st.write(f"You chose the following input features: {feature_name}")

    return [feature for feature in dataset_features
            if feature.name in feature_name]


def run_pipeline_prediction(pipeline: Pipeline,
                            dataset: Dataset,
                            features: list[Feature]) -> np.ndarray:
    """
    Function to run predictions using the selected pipeline.

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


load_pipeline()
summary_tab, load_tab = st.tabs(["Summary", "Load"])
st.session_state.data = None

current_pipeline = st.session_state.pipeline
if current_pipeline:
    with summary_tab:
        st.write("### Pipeline Summary")
        st.write(current_pipeline)

    with load_tab:
        st.write("### Upload .csv for prediction")
        csv_label = "Please provide a .csv to perform predictions."
        csv_data = st.file_uploader(label=csv_label,
                                    type='csv')
        if csv_data is not None:
            st.session_state.data = csv_data

        if st.session_state.data is not None:
            df = pd.read_csv(csv_data)
            # turn df into Dataset for convenience
            dataset = Dataset.from_dataframe(
                name=csv_data.name.split('.')[0],
                asset_path=csv_data.name,
                data=df
            )
            st.dataframe(dataset.read_as_data_frame())
            input_features = choose_input_features(dataset)
            if st.button("Run predictions"):
                predictions = run_pipeline_prediction(
                    current_pipeline, dataset, input_features
                )
                st.write(predictions)
