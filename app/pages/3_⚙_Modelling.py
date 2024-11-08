import pickle

import pandas as pd
import streamlit as st

from app.core.system import AutoMLSystem
from autoop.core.ml.artifact import Artifact
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature
from autoop.core.ml.metric import Metric, METRICS_MAP, METRICS
from autoop.core.ml.model import get_models, get_model, Model
from autoop.core.ml.pipeline import Pipeline
from autoop.functional.feature import detect_feature_types

st.set_page_config(page_title="Modelling", page_icon="⚙")


def write_helper_text(text: str) -> None:
    """
    Function to write helper text in a specific format.
    """
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)


st.write("# ⚙ Modelling")
write_helper_text("In this section, you can design a pipeline "
                  "to train a model on a dataset.")

automl = AutoMLSystem.get_instance()

datasets = automl.registry.list_of_type(type_class=Dataset,
                                        list_type="dataset")


def choose_dataset() -> Dataset:
    """
    Function to choose a dataset from the registry.
    Returns: The chosen dataset
    """
    dataset = st.selectbox(
        "Please choose your dataset:",
        options=datasets,
        format_func=lambda dataset: f"{dataset.name} - {dataset.version}"
    )

    if dataset:
        st.write("You chose the following dataset:")

        # Add a slice button
        if st.button("Slice"):
            st.session_state["dataset_index"] = datasets.index(dataset)
            st.switch_page("pages/3_🔪_Slicing.py")

        st.dataframe(dataset.read_as_data_frame().head(10))

    return dataset


def choose_model(feature_type: str) -> Model:
    """
    Function to choose a model and its hyperparameters.

    Args:
        feature_type: The type of the target feature
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
        st.write(f"### Error trying to create the model: {e}")
    return model_instance


def choose_metrics(model_type: str) -> list[Metric]:
    """
    Function to choose metrics for the model.
    Args:
        model_type: The type of the model

    Returns: The chosen metrics
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
        return [], None

    st.write(f"You chose the following metrics: {metric_name}")
    return [metrics[metric_name] for metric_name in metric_name], metric_name


def choose_input_features(dataset: Dataset) -> list[Feature]:
    """
    Function to choose input features from a dataset.
    Args:
        dataset: The dataset to choose features from

    Returns:
        The chosen features
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


def choose_target_feature(dataset: Dataset) -> Feature:
    """
    Function to choose the target feature from a dataset.
    Args:
        dataset: The dataset to choose the target feature from

    Returns: The chosen target feature
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
    Function to choose the data split
    """
    st.write("## Data split")
    st.write("Please choose the data split ratio:")
    data_split = st.slider(
        "Please choose the data split ratio:",
        0.0, 1.0, 0.8, 0.05
    )
    return data_split


if __name__ == "__main__":
    # quick workaround :P
    modelling_pipeline = None
    dataset = choose_dataset()
    input_features = choose_input_features(dataset)
    target_feature = choose_target_feature(dataset)
    if input_features and target_feature:
        model = choose_model(target_feature.type)
        metrics, metric_names = choose_metrics(model.type)
        data_split = choose_data_split()
        if metrics and data_split:
            # create pipeline object
            modelling_pipeline = Pipeline(
                metrics,
                dataset,
                model,
                input_features,
                target_feature,
                data_split
            )

    st.divider()
    st.subheader("Pipeline Overview")
    if modelling_pipeline:
        st.markdown("### Selected dataset:")
        st.markdown(f"**Name**: {dataset.name}")
        st.markdown(f"**Version**: {dataset.version}")
        if st.button(label="View"):
            st.dataframe(dataset.read_as_data_frame())

        st.markdown("### Selected features:")
        input_column, target_column = st.columns(2)
        with input_column:
            st.subheader("Input features:")
            for feature in input_features:
                st.markdown(f"**{feature.name}** (type: {feature.type})")
        with target_column:
            st.subheader("Target feature:")
            st.markdown(f"**{target_feature.name}** "
                        f"(type: {target_feature.type})")

        st.markdown("### Selected model:")
        st.markdown(f"**Name**: {modelling_pipeline.model.__class__.__name__}")
        st.markdown(f"**Type**: {modelling_pipeline.model.type.capitalize()}")
        st.markdown("### Model Hyperparameters:")

        items = modelling_pipeline.model.hyperparameters.items()
        hp_columns = st.columns(len(items))
        for index, (hyper_param, value) in enumerate(items):
            with hp_columns[index]:
                st.metric(f"**{hyper_param}**", value)

        st.markdown("### Model Metrics:")
        metric_columns = st.columns(len(metric_names))
        for index, metric in enumerate(metric_names):
            with metric_columns[index]:
                st.metric(label=f"{metric.replace('_', ' ').capitalize()}",
                          value=None)

    else:
        st.write("Please create the modelling pipeline first.")

    st.divider()
    if modelling_pipeline:
        artifact_name = st.text_input("Enter pipeline name",
                                      max_chars=20,
                                      placeholder="cool_pipeline!")
        artifact_version = st.text_input("Enter pipeline version",
                                         max_chars=10,
                                         placeholder="1.0.0")

        train_button = st.button("Train and Save")
        if train_button:
            results = modelling_pipeline.execute()
            st.success("Succesfully trained and saved the pipeline!"
                       "Please look beneath for the pipeline's results.")

            # Place in session state
            st.session_state["modelling_pipeline_results"] = results
            st.session_state["modelling_pipeline"] = modelling_pipeline

            automl = AutoMLSystem.get_instance()
            pipeline_artifact = Artifact(
                name=artifact_name,
                version=artifact_version,
                data=pickle.dumps(modelling_pipeline),
                asset_path=f"{artifact_name}.txt",
                type="pipeline"
            )
            automl.registry.register(pipeline_artifact)

            # Show results of metrics
            st.write("### Metrics Results")
            print(results)
            metrics_df = pd.DataFrame(results["metrics"])

            # Modify first column with names
            metrics_df.columns = ["Metric", "Value"]

            metrics_df["Metric"] = (metrics_df["Metric"]
                                    .apply(lambda x: type(x).__name__))

            st.write(metrics_df)

            # Show predictions
            st.write("### Predictions")
            st.write(pd.DataFrame(results["predictions"]).transpose())

            # Show training metrics
            st.write("### Training Metrics Results")

            training_metrics_df = pd.DataFrame(results["training_metrics"])

            # Modify first column with names
            training_metrics_df.columns = ["Metric", "Value"]
            training_metrics_df["Metric"] = (
                training_metrics_df["Metric"].apply(lambda x: type(x).__name__)
            )

            st.write(training_metrics_df)

            # Show training predictions
            st.write("### Training Predictions")
            st.write(pd.DataFrame(results["training_predictions"]).transpose())
