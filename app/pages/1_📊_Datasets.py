import pandas as pd
import streamlit as st

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset

automl = AutoMLSystem.get_instance()

datasets = automl.registry.list(type="dataset")


def write_helper_text(text: str) -> None:
    """
    Function to write helper text in a specific format.
    Returns:

    """
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)


def upload_dataset(automl: AutoMLSystem) -> None:
    """
    Function to upload a dataset to the database.
    Args:
        automl: The AutoMLSystem instance
    """
    uploaded_data = st.file_uploader("Upload a .csv file", 'csv')
    if uploaded_data is not None:
        df = pd.read_csv(uploaded_data)

        file_name = uploaded_data.name.split('.')[0]
        dataset = Dataset.from_dataframe(
            name=file_name,
            asset_path=uploaded_data.name,
            data=df
        )

        automl = automl.get_instance()
        automl.registry.register(dataset)
        st.success("Your dataset has been uploaded to the database!")


def manage_datasets(automl: AutoMLSystem) -> None:
    """
    Function to manage datasets in the database.
    Args:
        automl: The AutoMLSystem instance
    """
    datasets = automl.registry.list_of_type(
        type_class=Dataset,
        list_type="dataset"
    )
    if datasets:
        st.write("### Datasets in database")
        dataset = st.selectbox(label="Select a dataset to delete:",
                               options=datasets,
                               format_func=lambda x: f"{x.name} - {x.version}")
        if dataset:
            df = dataset.read_as_data_frame()
            st.dataframe(df)

            if st.button("Delete dataset"):
                automl.registry.delete(dataset.id)
                st.success(f"Dataset {dataset.name} has been deleted!")
    else:
        st.write("No datasets uploaded!")


st.set_page_config(page_title="Dataset Management", page_icon="📊")
st.write("# 📊 Dataset Management")
write_helper_text("Manage your datasets by adding, removing and viewing them.")
st.write("## Upload new Datasets")
upload_dataset(automl)

st.write("## Manage existing Datasets")
manage_datasets(automl)
