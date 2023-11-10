import streamlit as st
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.image import imread

import itertools
import random


def page_visualizer_body():
    """
    Display the Pizza Visualizer page in Streamlit.

    This function provides a user interface for visualizing pizza
    vs. not-pizza images, differences between
    pizza and not-pizza averages, and creating an image montage.

    Parameters:
    None

    Returns:
    None
    """
    st.write("### Pizza Visualizer")
    st.info(
        "The client is interested in visually differentiating pizza from food "
        "that is not pizza."
    )

    version = 'v1'

    # Check if the image file exists
    if os.path.exists(f"outputs/{version}/avg_var_not_pizza.png"):
        avg_not_pizza = plt.imread(f"outputs/{version}/avg_var_not_pizza.png")
    else:
        st.error("Error: Image file not found.")
        avg_not_pizza = None

    avg_pizza = plt.imread(f"outputs/{version}/avg_var_pizza.png")

    st.warning("While exploring the visual differences, we couldn't find "
               "clear patterns to differentiate pizza from non-pizza. "
               "However, there's a subtle difference in color pigment "
               "between the average images of both labels.")

    # Display images
    st.image(avg_not_pizza, caption='Not Pizza - Average and Variability')
    st.image(avg_pizza, caption='Pizza - Average and Variability')
    st.write("---")

    if st.checkbox("Differences between Pizza and Not-Pizza Averages"):
        diff_between_avgs = plt.imread(f"outputs/{version}/avg_diff.png")

        st.warning(
            "Our visual analysis didn't reveal distinct patterns that we "
            "could intuitively differentiate pizza from non-pizza images."
        )

        st.image(
            diff_between_avgs, caption='Difference between average images')

    if st.checkbox("Create Image Montage"):
        st.info("Click the 'Create Montage' button to generate an image "
                "montage.")
        my_data_dir = 'inputs/carlosrunner/pizza-not-pizza/pizza_not_pizza'
        labels = os.listdir(my_data_dir + '/validation')
        label_to_display = st.selectbox(
            label="Select label", options=labels, index=0)
        if st.button("Create Montage"):
            image_montage(dir_path=my_data_dir + '/validation',
                          label_to_display=label_to_display,
                          nrows=8, ncols=3, figsize=(10, 25))
        st.write("---")


def image_montage(dir_path, label_to_display, nrows, ncols, figsize=(15, 10)):
    """
    Create an image montage of images from the specified directory for a 
    specific label.

    This function creates an image montage of images from the specified 
    directory for a specific label.

    Parameters:
    dir_path (str): The directory path where the images are located.
    label_to_display (str): The label of the images to create the montage for.
    nrows (int): The number of rows in the montage.
    ncols (int): The number of columns in the montage.
    figsize (tuple): The figure size for the montage plot.

    Returns:
    None
    """
    sns.set_style("white")
    labels = os.listdir(dir_path)

    if label_to_display in labels:
        images_list = os.listdir(dir_path+'/' + label_to_display)
        if nrows * ncols < len(images_list):
            img_idx = random.sample(images_list, nrows * ncols)
        else:
            st.warning(
                f"Decrease nrows or ncols to create your montage. "
                "There are {len(images_list)} images in your subset, "
                f"but you requested a montage with {nrows * ncols} spaces."
            )
            return

        list_rows = range(0, nrows)
        list_cols = range(0, ncols)
        plot_idx = list(itertools.product(list_rows, list_cols))

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        for x in range(0, nrows * ncols):
            img = imread(dir_path + '/' + label_to_display + '/' + img_idx[x])
            img_shape = img.shape
            axes[plot_idx[x][0], plot_idx[x][1]].imshow(img)
            axes[plot_idx[x][0], plot_idx[x][1]].set_title(
                f"Width {img_shape[1]}px x Height {img_shape[0]}px"
            )
            axes[plot_idx[x][0], plot_idx[x][1]].set_xticks([])
            axes[plot_idx[x][0], plot_idx[x][1]].set_yticks([])
        plt.tight_layout()

        st.pyplot(fig=fig)
