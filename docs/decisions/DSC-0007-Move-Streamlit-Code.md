# DSC-0003: Moved lots of functions from app/pages to app/core.
# Date: 2024-11-09 / 2024-11-10
# Decision: Most of the functionality behind the Streamlit app was moved to app/core.
# Status: Accepted
# Motivation: Keeping the code in every page would result in messy and long code blocks and would result in some code duplication (e.g. show_pipeline_summary being used mulitple times across pages).
# Reason: This makes the code more readable throughout the project.