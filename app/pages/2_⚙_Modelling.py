import streamlit as st
import pandas as pd
import io
from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset


st.set_page_config(page_title="Modelling", page_icon="📈")

def write_helper_text(text: str):
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)

st.write("# ⚙ Modelling")
write_helper_text("In this section, you can design a machine learning pipeline to train a model on a dataset.")

automl = AutoMLSystem.get_instance()

datasets = automl.registry.list(type="dataset")


# your code here
def choose_dataset():
    dataset = st.selectbox("Please choose your dataset:",
                           options=datasets,
                           format_func=lambda dataset: f"{dataset.name} - {dataset.version}")
    if dataset:
        st.write("You chose the following dataset:")

        bytes = dataset.read()
        csv = bytes.decode()
        df = pd.read_csv(io.StringIO(csv)).head(100)
        st.dataframe(df)


choose_dataset()
