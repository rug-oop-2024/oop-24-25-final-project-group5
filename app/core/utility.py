import streamlit as st

from autoop.core.ml.pipeline import Pipeline


def write_helper_text(text: str) -> None:
    """Function to write helper text in a specific format."""
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)


def show_pipeline_summary(pipeline: Pipeline) -> None:
    """Shows a neat summary of the pipeline.

    Arguments:
        pipeline (Pipeline): pipeline used in the summary.
    """
    st.markdown("### Selected dataset:")
    st.markdown(f"**Name**: {pipeline.dataset.name}")
    st.markdown(f"**Version**: {pipeline.dataset.version}")
    if st.button(label="View"):
        st.dataframe(pipeline.dataset.read_as_data_frame())

    st.markdown("### Selected features:")
    input_column, target_column = st.columns(2)
    with input_column:
        st.subheader("Input features:")
        for feature in pipeline.input_features:
            st.markdown(f"**{feature.name}** (type: {feature.type})")
    with target_column:
        st.subheader("Target feature:")
        st.markdown(f"**{pipeline.target_feature.name}** "
                    f"(type: {pipeline.target_feature.type})")

    st.markdown("### Selected model:")
    st.markdown(f"**Name**: {pipeline.model.__class__.__name__}")
    st.markdown(f"**Type**: {pipeline.model.type.capitalize()}")
    st.markdown("### Model Hyperparameters:")

    items = pipeline.model.hyperparameters.items()
    hp_columns = st.columns(len(items))
    for index, (hyper_param, value) in enumerate(items):
        with hp_columns[index]:
            st.metric(f"**{hyper_param}**", value)

    st.markdown("### Model Metrics:")
    metric_columns = st.columns(len(pipeline.metrics))
    for index, metric in enumerate(pipeline.metrics):
        with metric_columns[index]:
            st.metric(label=f"{metric}",
                      value=None)

    st.markdown("### Data split:")
    st.markdown(f"Data split used during training: {pipeline.split}")
