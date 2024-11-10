import pickle

import pandas as pd
import streamlit as st

from app.core.system import AutoMLSystem
from app.core.utility import write_helper_text, show_pipeline_summary
from app.core.dataset import choose_dataset
from app.core.modelling import choose_data_split, \
    choose_input_features, choose_metrics, choose_model, choose_target_feature
from autoop.core.ml.artifact import Artifact
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.pipeline import Pipeline

st.set_page_config(page_title="Modelling", page_icon="⚙")

st.write("# ⚙ Modelling")
write_helper_text("In this section, you can design a pipeline "
                  "to train a model on a dataset.")

automl = AutoMLSystem.get_instance()

datasets = automl.registry.list_of_type(type_class=Dataset,
                                        list_type="dataset")


if __name__ == "__main__":
    modelling_pipeline = None
    dataset = choose_dataset(datasets)
    input_features = choose_input_features(dataset)
    target_feature = choose_target_feature(dataset, input_features)
    if input_features and target_feature:
        model = choose_model(target_feature.type)
        metrics = choose_metrics(model.type)
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
        show_pipeline_summary(modelling_pipeline)
    else:
        st.write("Please create the modelling pipeline first.")

    st.divider()
    if modelling_pipeline:
        artifact_name = st.text_input("Enter pipeline name",
                                      max_chars=20,
                                      placeholder="cool_pipeline")
        artifact_version = st.text_input("Enter pipeline version",
                                         max_chars=10,
                                         placeholder="1.0.0")

        train_button = st.button("Train and Save")
        if train_button:
            results = modelling_pipeline.execute()
            st.success("Succesfully trained and saved the pipeline! \n"
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
