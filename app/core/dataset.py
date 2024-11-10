from typing import Literal
import streamlit as st
import pandas as pd

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.slicer import Slicer, NumericRangeSlicer, CategoricalSlicer


def upload_dataset(automl: AutoMLSystem) -> None:
    """
    Function to upload a dataset to the database.

    Arguments:
        automl (AutoMLSystem): the auto machine learning instance.
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

    Arguments:
        automl (AutoMLSystem): the auto machine learning instance.
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


def data_slicer(column: str,
                column_type: Literal["categorical", "numerical"],
                dataset_frame: pd.DataFrame) -> Slicer:
    """
    Function that generates the data slicer.

    Arguments:
        column (str): the column to slice.
        column_type (Literal["categorical", "numerical"]):
            the type of the column
        dataset_frame (pd.DataFrame): the dataset frame.

    Returns:
        The slicer object.
    """
    st.write(f"## {column.replace('_', ' ')}")
    if column_type == "categorical":
        slicer = st.multiselect(
            f"Select the range for {column.replace('_', ' ')}",
            dataset_frame[column].unique()
        )

        dataset_frame = dataset_frame[dataset_frame[column].isin(slicer)]
        return CategoricalSlicer(column=column, categories=slicer)
    else:
        slicer = st.slider(
            f"Select the range for {column.replace('_', ' ')}",
            dataset_frame[column].min() - 1,
            dataset_frame[column].max() + 1,
            (dataset_frame[column].min() - 1, dataset_frame[column].max() + 1)
        )
        return NumericRangeSlicer(column=column, min=slicer[0], max=slicer[1])


def choose_dataset(datasets) -> Dataset:
    """
    Function to choose a dataset from the registry.

    Returns:
        the chosen dataset.
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
