import streamlit as st
import pandas as pd
import io
from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset


st.set_page_config(page_title="Modelling", page_icon="⚙")

def write_helper_text(text: str):
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)

st.write("# ⚙ Modelling")
write_helper_text("In this section, you can design a machine learning pipeline to train a model on a dataset.")

automl = AutoMLSystem.get_instance()

datasets = automl.registry.list_of_type(type_class=Dataset, list_type="dataset")

def choose_dataset():
    dataset = st.selectbox("Please choose your dataset:",
                           options=datasets,
                           format_func=lambda dataset: f"{dataset.name} - {dataset.version}")
    if dataset:
        st.write("You chose the following dataset:")

        # Add a slice button
        if st.button("Slice"):
            st.session_state["dataset_index"] = datasets.index(dataset)
            st.switch_page("pages/3_🔪_Slicing.py")

        st.dataframe(dataset.readAsDataFrame())


choose_dataset()
