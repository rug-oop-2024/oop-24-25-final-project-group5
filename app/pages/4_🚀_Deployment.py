import streamlit as st
import pickle
from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.metric import Metric, METRICS_MAP, METRICS
from autoop.core.ml.model import *
from autoop.core.ml.artifact import Artifact
from autoop.core.ml.feature import Feature
from autoop.functional.feature import detect_feature_types
from autoop.core.ml.pipeline import Pipeline


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

    if pipeline_artifact:
        current_pipeline = pickle.loads(pipeline_artifact.read())
        print(current_pipeline)
        print(current_pipeline.model.parameters)
        # CURRENT ISSUES: SAVED PIPELINE DOES NOT STORE PARAMETERS


load_pipeline()