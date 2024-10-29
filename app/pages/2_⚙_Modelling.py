import streamlit as st
import pandas as pd
import io
from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset

from autoop.core.ml.metric import Metric, METRICS_MAP

from autoop.core.ml.model import *

from autoop.core.ml.feature import Feature
from autoop.functional.feature import detect_feature_types


st.set_page_config(page_title="Modelling", page_icon="⚙")

def write_helper_text(text: str):
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)

st.write("# ⚙ Modelling")
write_helper_text("In this section, you can design a machine learning pipeline to train a model on a dataset.")

automl = AutoMLSystem.get_instance()

datasets = automl.registry.list_of_type(type_class=Dataset, list_type="dataset")

def choose_dataset() -> Dataset:
    dataset = st.selectbox("Please choose your dataset:",
                           options=datasets,
                           format_func=lambda dataset: f"{dataset.name} - {dataset.version}")
    if dataset:
        st.write("You chose the following dataset:")

        # Add a slice button
        if st.button("Slice"):
            st.session_state["dataset_index"] = datasets.index(dataset)
            st.switch_page("pages/3_🔪_Slicing.py")

        st.dataframe(dataset.readAsDataFrame().head(10))
    
    return dataset

def choose_model() -> Model:
    models = get_models()
    model_names = list(models.keys())
    model_name = st.selectbox("Please choose your model:", options=model_names, 
        format_func=lambda model_name: f"{model_name} - {models[model_name]}")

    if not model_name:
        return None


    st.write(f"You chose the following model: {model_name}")
    model_type = get_model(model_name)

    model = model_type()

    # Add hyper parameters
    hyper_params = model.hyperparameters
    hyper_params_desc = model.hyperparameter_descriptions

    if len(hyper_params) == 0:
        st.write("The model does not have any hyperparameters.")
        return model
    
    st.write("Please choose the hyperparameters for the model:")

    hyper_params_selector = st.empty()

    with hyper_params_selector.container():
        for param_name, param_value in hyper_params.items():
            param_type = type(param_value)

            st.write(f"## Parameter: {param_name}")
            st.write(f"Description: {hyper_params_desc[param_name]}")

            if param_type == bool:
                hyper_params[param_name] = st.checkbox(param_name, value=hyper_params[param_name])
            elif param_type == int:
                hyper_params[param_name] = st.number_input(param_name, value=hyper_params[param_name])
            elif param_type == float:
                hyper_params[param_name] = st.number_input(param_name, value=hyper_params[param_name])
            elif param_type == str:
                hyper_params[param_name] = st.text_input(param_name, value=hyper_params[param_name])
            else:
                st.write(f"Unsupported hyperparameter type: {param_type}")

def choose_metrics() -> list[Metric]:
    metrics = METRICS_MAP

    metric_names = list(metrics.keys())
    metric_name = st.multiselect("Please choose your metrics:", options=metric_names,
        format_func=lambda metric_name: f"{metric_name.replace('_', ' ').capitalize()}")

    if not metric_name:
        return None

    st.write(f"You chose the following metrics: {metric_name}")
    return [metrics[metric_name] for metric_name in metric_name]

def choose_input_features(dataset: Dataset) -> list[Feature]:
    dataset_features = detect_feature_types(dataset)

    feature_names = [feature.name for feature in dataset_features]
    feature_name = st.multiselect("Please choose your input features:", options=feature_names,
        format_func=lambda feature_name: f"{feature_name}")
    
    if not feature_name:
        return None

    st.write(f"You chose the following input features: {feature_name}")

    return [feature for feature in dataset_features if feature.name in feature_name]

def choose_target_feature(dataset: Dataset) -> Feature:
    dataset_features = detect_feature_types(dataset)

    feature_names = [feature.name for feature in dataset_features]
    feature_name = st.selectbox("Please choose your target feature:", options=feature_names,
        format_func=lambda feature_name: f"{feature_name}")

    if not feature_name:
        return None

    st.write(f"You chose the following target feature: {feature_name}")

    return [feature for feature in dataset_features if feature.name == feature_name][0]

def choose_data_split() -> float:
    st.write("Please choose the data split ratio: (how much data to use for training)")
    data_split = st.slider("Please choose the data split ratio:", 0.0, 1.0, 0.8, 0.05)
    return data_split

dataset = choose_dataset()
model = choose_model()
metrics = choose_metrics()
input_features = choose_input_features(dataset)
target_feature = choose_target_feature(dataset)
data_split = choose_data_split()

if st.button("Train"):
    pass
