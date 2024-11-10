import streamlit as st

from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature
from autoop.core.ml.metric import Metric, METRICS_MAP, METRICS
from autoop.core.ml.model import get_models, get_model, Model
from autoop.functional.feature import detect_feature_types


def choose_model(feature_type: str) -> Model:
    """
    Function to choose a model and its hyperparameters.

    Arguments:
        feature_type (str): the type of the target feature.
    """
    models = get_models()
    if feature_type == "categorical":
        model_names = [model for model in models
                       if models[model] == "classification"]
    else:
        model_names = [model for model in models
                       if models[model] == "regression"]
    model_name = st.selectbox(
        "Please choose your model:",
        options=model_names,
        format_func=lambda model_name: f"{model_name} - {models[model_name]}"
    )

    if not model_name:
        return None

    st.write(f"You chose the following model: {model_name}")
    model_type = get_model(model_name)

    # Add hyper parameters
    model_instance = model_type()
    hyper_params = model_instance.hyperparameters
    hyper_params_desc = model_instance.hyperparameter_descriptions

    if len(hyper_params) == 0:
        st.write("The model does not have any hyperparameters.")
        return model_instance

    st.write("Please choose the hyperparameters for the model:")
    for param_name, param_value in hyper_params.items():
        param_type = type(param_value)

        st.write(f"## Parameter: {param_name}")
        st.write(f"Description: {hyper_params_desc[param_name]}")

        if param_type == bool:
            hyper_params[param_name] = st.checkbox(
                param_name,
                value=hyper_params[param_name]
            )
        elif param_type == int:
            hyper_params[param_name] = st.number_input(
                param_name,
                value=hyper_params[param_name]
            )
        elif param_type == float:
            hyper_params[param_name] = st.number_input(
                param_name,
                value=hyper_params[param_name]
            )
        elif param_type == str:
            hyper_params[param_name] = st.text_input(
                param_name,
                value=hyper_params[param_name]
            )
        elif param_type is None:
            hyper_params[param_name] = st.text_input(
                param_name,
                value=hyper_params[param_name]
            )
        else:
            st.write(f"Unsupported hyperparameter type: {param_type}")
    try:
        model_instance = model_type(**hyper_params)
    except ValueError as e:
        st.error(f"Error trying to create the model: {e}")
    return model_instance


def choose_metrics(model_type: str) -> list[Metric]:
    """
    Function to choose metrics for the model.

    Arguments:
        model_type (str): the model's type.

    Returns:
        A list of chosen metric(s) and
    """
    st.write("## Metrics")
    metrics = METRICS_MAP
    if model_type == "regression":
        metric_names = METRICS[:3]
    elif model_type == "classification":
        metric_names = METRICS[3:]
    metric_name = st.multiselect(
        "Please choose your metrics:",
        options=metric_names,
        format_func=lambda metric_name:
            f"{metric_name.replace('_', ' ').capitalize()}"
    )

    if not metric_name:
        return []

    st.write(f"You chose the following metrics: {metric_name}")
    return [metrics[metric_name] for metric_name in metric_name]


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
        return []

    st.write(f"You chose the following input features: {feature_name}")

    return [feature for feature in dataset_features
            if feature.name in feature_name]


def choose_target_feature(dataset: Dataset) -> Feature:
    """
    Function to choose the target feature from a dataset.

    Arguments:
        dataset (Dataset): dataset to choose the target feature from.

    Returns:
        chosen target feature.
    """
    dataset_features = detect_feature_types(dataset)

    feature_names = [feature.name for feature in dataset_features]
    feature_name = st.selectbox(
        "Please choose your target feature:",
        options=feature_names,
        format_func=lambda feature_name: f"{feature_name}")

    if not feature_name:
        return None

    st.write(f"You chose the following target feature: {feature_name}")

    return [feature for feature in dataset_features
            if feature.name == feature_name][0]


def choose_data_split() -> float:
    """
    Function to choose the data split.

    Returns:
        float resembling dats split ratio.
    """
    st.write("## Data split")
    st.write("Please choose the data split ratio:")
    data_split = st.slider(
        "Please choose the data split ratio:",
        0.0, 1.0, 0.8, 0.05
    )
    return data_split
