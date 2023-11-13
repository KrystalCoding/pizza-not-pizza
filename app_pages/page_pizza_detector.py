import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd

from src.data_management import download_dataframe_as_csv
from src.machine_learning.predictive_analysis import (
    load_model_and_predict,
    resize_input_image,
    plot_predictions_probabilities
)


def page_pizza_detector_body():
    """
    Display the pizza detection page in the Streamlit app.

    This function allows users to upload images for live pizza detection
    and provides predictions and analysis reports for each image.

    Parameters:
    None

    Returns:
    None
    """
    st.success(
        f"* You can download a set of pizza or not-pizza photos "
        "for live prediction. "
        f"You can download the images from "
        "[here](https://www.kaggle.com/datasets/carlosrunner/pizza-not-pizza)."
    )

    st.write("---")

    images_buffer = st.file_uploader(
        'Upload photo samples. You may select more than one.',
        type=['png', 'jpg'], accept_multiple_files=True)

    if images_buffer is not None:
        df_report = pd.DataFrame([])
        for image in images_buffer:

            img_pil = (Image.open(image))
            st.info(f"Photo Sample: **{image.name}**")
            img_array = np.array(img_pil)
            st.image(
                img_pil, caption=f"Image Size: {img_array.shape[1]}px width "
                "x {img_array.shape[0]}px height")

            version = 'v1'
            resized_img = resize_input_image(img=img_pil, version=version)
            pred_proba, pred_class = load_model_and_predict(
                resized_img, version=version)
            plot_predictions_probabilities(pred_proba, pred_class)

            df_report = df_report.append(
                {"Name": image.name, 'Result': pred_class}, ignore_index=True)

        if not df_report.empty:
            st.success("Analysis Report")
            st.table(df_report)
            st.markdown(download_dataframe_as_csv(
                df_report), unsafe_allow_html=True)

    st.subheader("Page Summary")
    st.info(
        "**Business Requirements Addressed:**\n"
        "- Automated Pizza Detection (Addressed through live pizza detection "
        "and analysis reports)\n"
        "- Prediction Reporting (Addressed through live pizza detection and "
        "analysis reports)"
        "\n---\n"
        "**Conclusions:**\n"
        "- The system generates downloadable .csv reports, meeting User "
        "Story 6.\n"
        "- Users can upload images for live pizza detection, fulfilling the "
        "live prediction requirement in User Story 1.\n"
        "- The analysis report, including image names and results, satisfies "
        "the prediction reporting requirement in User Story 2."
    )