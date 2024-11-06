import streamlit as st
import pickle
import pandas as pd
import numpy as np
from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.metric import Metric, METRICS_MAP, METRICS
from autoop.core.ml.model import *
from autoop.core.ml.artifact import Artifact
from autoop.core.ml.feature import Feature
from autoop.functional.feature import detect_feature_types
from autoop.core.ml.pipeline import Pipeline
from autoop.functional.preprocessing import preprocess_features


st.set_page_config(page_title="Deployment", page_icon="🚀")

automl = AutoMLSystem.get_instance()
pipelines = automl.registry.list(type="pipeline")

def write_helper_text(text: str):
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)


st.write("# 🚀 Deployment")
write_helper_text("In this section, you can deploy pre-existing pipelines.")


def load_pipeline():
    # just trying to load pipelines :)
    pipeline_artifact = st.selectbox("Please choose the pipeline you want to use:",
                           options=pipelines,
                           format_func=lambda pipeline: f"{pipeline.name}")
    st.session_state.pipeline = pickle.loads(pipeline_artifact.read())


def choose_input_features(dataset: Dataset) -> list[Feature]:
    dataset_features = detect_feature_types(dataset)

    feature_names = [feature.name for feature in dataset_features]
    feature_name = st.multiselect("Please choose your input features:", options=feature_names,
        format_func=lambda feature_name: f"{feature_name}")

    if not feature_name:
        return None

    st.write(f"You chose the following input features: {feature_name}")

    return [feature for feature in dataset_features if feature.name in feature_name]


def run_pipeline_prediction(pipeline: Pipeline,
                            dataset: Dataset,
                            features: list[Feature]) -> np.ndarray:
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
            st.dataframe(dataset.readAsDataFrame())
            input_features = choose_input_features(dataset)
            if st.button("Run predictions"):
                predictions = run_pipeline_prediction(
                    current_pipeline, dataset, input_features
                )
                st.write(predictions)