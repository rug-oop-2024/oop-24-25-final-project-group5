import streamlit as st

from app.core.system import AutoMLSystem
from app.core.dataset import upload_dataset, manage_datasets
from app.core.utility import write_helper_text


automl = AutoMLSystem.get_instance()
datasets = automl.registry.list(type="dataset")

st.set_page_config(page_title="Dataset Management", page_icon="📊")
st.write("# 📊 Dataset Management")
write_helper_text("Manage your datasets by adding, "
                  "removing and viewing them.")


if __name__ == "__main__":
    st.write("## Upload new Datasets")
    upload_dataset(automl)

    st.write("## Manage existing Datasets")
    manage_datasets(automl)
