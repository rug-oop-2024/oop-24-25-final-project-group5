import streamlit as st
import pandas as pd
import io
from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
from typing import Literal
from autoop.core.ml.slicer import Slicer, NumericRangeSlicer, CategoricalSlicer


st.set_page_config(page_title="Slicing", page_icon="🔪", layout="wide")

st.title("Slicing")

st.write("This page is for slicing the dataset.")

automl = AutoMLSystem.get_instance()

datasets = automl.registry.list_of_type(type_class=Dataset, list_type="dataset")
dataset = st.selectbox("Please choose your dataset:",
                      options=datasets,
                      index=st.session_state.get("dataset_index", 0),
                      format_func=lambda dataset: f"{dataset.name} - {dataset.version}")

def slicer(column: str, column_type: Literal["categorical", "numerical"], dataset_frame: pd.DataFrame) -> Slicer:
    st.write(f"## {column.replace('_', ' ')}")
    if column_type == "categorical":
        slicer = st.multiselect(f"Select the range for {column.replace('_', ' ')}", dataset_frame[column].unique())
        dataset_frame = dataset_frame[dataset_frame[column].isin(slicer)]
        return CategoricalSlicer(column=column, categories=slicer)
    else:
        slicer = st.slider(f"Select the range for {column.replace('_', ' ')}", dataset_frame[column].min()-1, dataset_frame[column].max()+1, (dataset_frame[column].min()-1, dataset_frame[column].max()+1))
        dataset_frame = dataset_frame[(dataset_frame[column] >= slicer[0]) & (dataset_frame[column] <= slicer[1])]
        return NumericRangeSlicer(column=column, min=slicer[0], max=slicer[1])

if dataset:
    dataset_frame = dataset.readAsDataFrame()

    # For each column, generate a slicer
    slicers = {}
    for column in dataset_frame.columns:
        col_slicer = slicer(column, "categorical" if dataset_frame[column].dtype == object else "numerical", dataset_frame)
        slicers[column] = col_slicer

        
    
    st.write("## Sliced Dataset")
    sliced_dataset = dataset_frame
    for column, slicer in slicers.items():
        sliced_dataset = slicer.slice(sliced_dataset)
    
    st.write(sliced_dataset)

    # Save the sliced dataset
    dataset_name = st.text_input("Please enter the name of the sliced dataset:", value=f"{dataset.name}")
    dataset_version = st.text_input("Please enter the version of the sliced dataset:", value=f"{dataset.version}")

    if st.button("Save sliced dataset"):
        # If name and version are unchanged, warn the user
        if dataset_name == dataset.name:
            st.warning("The name of the sliced dataset is the same as the original dataset. Please change the name.")
        else:
            sliced_dataset = Dataset.from_dataframe(sliced_dataset, name=dataset_name, asset_path=f"{dataset_name}.csv", version=dataset_version)
            automl.registry.register(sliced_dataset)
            st.success(f"Dataset {dataset_name} - {dataset_version} has been saved.")