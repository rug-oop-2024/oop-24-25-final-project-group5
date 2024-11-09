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


def choose_input_features(dataset: Dataset) -> list[Feature]:
    """Function to choose input features from a dataset.

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
        return []

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


if __name__ == "__main__":
    st.session_state.data = None
    st.session_state.pipeline = None
    load_pipeline()
    summary_tab, load_tab = st.tabs(["Summary", "Load"])

    current_pipeline = st.session_state.pipeline
    if current_pipeline:
        with summary_tab:
            st.write("### Pipeline Summary")
            st.write(current_pipeline)

        with load_tab:
            st.write("### Upload csv file for prediction")
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
                if len(input_features) != len(current_pipeline.input_features):
                    st.warning("The length of the selected input features is "
                               "not the same as the length of the pipeline's "
                               "input features. Please select "
                               f"{len(current_pipeline.input_features)} "
                               "input features.")
                if input_features != current_pipeline.input_features:
                    st.warning("The selected input features are not the same. "
                               "Running the predictions with these input "
                               "features will cause the predictions to be off."
                               "\n\n"
                               "Please consider using the same input features "
                               "as those used in the pipeline: "
                               f"{[feature.name for feature
                                   in current_pipeline.input_features]}")
                if st.button("Run predictions"):
                    try:
                        predictions = run_pipeline_prediction(
                            current_pipeline, dataset, input_features
                        )
                        st.write(predictions)
                    except ValueError as e:
                        st.error("Predicting with current input features "
                                 "failed. Please check if you have the "
                                 "same amount of input features selected. "
                                 f"\n\n Error message: {e}")
