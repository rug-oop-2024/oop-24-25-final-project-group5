import streamlit as st
import pandas as pd

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset

automl = AutoMLSystem.get_instance()

datasets = automl.registry.list(type="dataset")

uploaded_data = st.file_uploader("Choose a file", 'csv')
if uploaded_data is not None:
    df = pd.read_csv(uploaded_data)

    file_name = uploaded_data.name.split('.')[0]
    dataset = Dataset.from_dataframe(
        name=file_name,
        asset_path=uploaded_data.name,
        data=df
    )
    st.write("Your data:")
    st.dataframe(dataset.read())

    automl = automl.get_instance()
    automl.registry.register(dataset)
    st.success("Your file has been uploaded to the dataset!")
