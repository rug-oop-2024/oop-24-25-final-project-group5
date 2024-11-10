import streamlit as st

from app.core.system import AutoMLSystem
from app.core.dataset import data_slicer
from app.core.utility import write_helper_text
from autoop.core.ml.dataset import Dataset

st.set_page_config(page_title="Slicing", page_icon="🔪")

st.title("🔪 Slicing")
write_helper_text("This page is for slicing the dataset.")

automl = AutoMLSystem.get_instance()

datasets = automl.registry.list_of_type(type_class=Dataset,
                                        list_type="dataset")

if __name__ == "__main__":
    dataset = st.selectbox(
        "Please choose your dataset:",
        options=datasets,
        index=st.session_state.get("dataset_index", 0),
        format_func=lambda dataset: f"{dataset.name} - {dataset.version}"
    )

    if dataset:
        dataset_frame = dataset.read_as_data_frame()

        # For each column, generate a slicer
        slicers = {}
        for column in dataset_frame.columns:
            col_slicer = data_slicer(
                column,
                "categorical" if dataset_frame[column].dtype == object
                else "numerical", dataset_frame)
            slicers[column] = col_slicer

        st.write("## Sliced Dataset")
        sliced_dataset = dataset_frame
        for column, slicer in slicers.items():
            sliced_dataset = slicer.slice(sliced_dataset)

        st.write(sliced_dataset)

        # Save the sliced dataset
        dataset_name = st.text_input(
            "Please enter the name of the sliced dataset:",
            value=f"{dataset.name}"
        )
        dataset_version = st.text_input(
            "Please enter the version of the sliced dataset:",
            value=f"{dataset.version}"
        )

        if st.button("Save sliced dataset"):
            # If name and version are unchanged, warn the user
            if dataset_name == dataset.name:
                st.warning("The name of the sliced dataset is the same "
                           "as the original. Please change the name.")
            else:
                sliced_dataset = Dataset.from_dataframe(
                    sliced_dataset,
                    name=dataset_name,
                    asset_path=f"{dataset_name}.csv",
                    version=dataset_version
                )
                automl.registry.register(sliced_dataset)
                st.success(
                    f"Dataset {dataset_name} - {dataset_version} "
                    "has been saved."
                )
