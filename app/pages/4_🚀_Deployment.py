import pandas as pd
import streamlit as st

from app.core.system import AutoMLSystem
from app.core.utility import show_pipeline_summary, write_helper_text
from app.core.modelling import choose_input_features
from app.core.pipeline import load_pipeline, run_pipeline_prediction
from autoop.core.ml.dataset import Dataset

st.set_page_config(page_title="Deployment", page_icon="🚀")

automl = AutoMLSystem.get_instance()
pipelines = automl.registry.list(type="pipeline")

st.write("# 🚀 Deployment")
write_helper_text("In this section, you can deploy pre-existing pipelines.")


if __name__ == "__main__":
    st.session_state.data = None
    st.session_state.pipeline = None
    load_pipeline(pipelines)
    summary_tab, load_tab = st.tabs(["Summary", "Load"])

    current_pipeline = st.session_state.pipeline
    if current_pipeline:
        with summary_tab:
            st.write("### Pipeline Summary")
            show_pipeline_summary(current_pipeline)

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
                        st.dataframe(predictions)
                    except ValueError as e:
                        st.error("Predicting with current input features "
                                 "failed. Please check if you have the "
                                 "same amount of input features selected. "
                                 f"\n\n Error message: {e}")
