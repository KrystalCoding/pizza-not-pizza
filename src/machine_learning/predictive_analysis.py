import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from tensorflow.keras.models import load_model
from PIL import Image
from src.data_management import load_pkl_file


def plot_predictions_probabilities(pred_proba, pred_class):
    """
    Plot prediction probability results as percentages
    """

    # Calculate the percentage likelihood of being pizza and not-pizza
    pizza_percentage = pred_proba * 100
    not_pizza_percentage = 100 - pizza_percentage

    # Create a DataFrame for the probabilities
    prob_per_class = pd.DataFrame({
        'Diagnostic': ['Pizza', 'Not-Pizza'],
        'Percentage': [pizza_percentage, not_pizza_percentage]
    })

    prob_per_class = prob_per_class.round(2)

    fig = px.bar(
        prob_per_class,
        x='Diagnostic',
        y='Percentage',
        range_y=[0, 100],
        width=600,
        height=300,
        template='seaborn',
        labels={'Percentage': 'Prediction'}
    )

    st.plotly_chart(fig)


def resize_input_image(img, version):
    """
    Reshape image to the average image size and convert it to RGB
    """
    image_shape = load_pkl_file(file_path=f"outputs/{version}/image_shape.pkl")
    img_resized = img.resize((image_shape[1], image_shape[0]), Image.ANTIALIAS)

    img_rgb = img_resized.convert('RGB')

    my_image = np.array(img_rgb) / 255

    my_image = np.expand_dims(my_image, axis=0)

    return my_image


def load_model_and_predict(my_image, version):
    """
    Load and perform ML prediction over live images
    """

    model = load_model(f"outputs/{version}/pizza_detector_model.h5")

    pred_proba = model.predict(my_image)[0, 0]

    # Calculate the percentage likelihood of being pizza and not-pizza
    pizza_percentage = pred_proba * 100
    not_pizza_percentage = (1 - pred_proba) * 100

    pred_class = 'Pizza' if pizza_percentage > not_pizza_percentage else 'Not-Pizza'

    st.write(
        f"The predictive analysis indicates the sample cell is "
        f"**{pred_class.lower()}**.\n\n"
        f"- Likelihood of being Pizza: {pizza_percentage:.2f}%\n"
        f"- Likelihood of being Not-Pizza: {not_pizza_percentage:.2f}%")

    return pred_proba, pred_class
