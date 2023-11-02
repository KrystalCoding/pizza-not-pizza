import streamlit as st
import matplotlib.pyplot as plt


def page_project_hypothesis_body():
    st.write("### Project Hypothesis and Validation")

    st.success(
        "Hypothesis 1: ** Pizza presence can be accurately identified by analyzing the shape and toppings within images.**"
        "\n- To validate this hypothesis, we conducted detailed image analysis, focusing on the circular shape of pizzas and the variety of toppings."
        "\n---\n"
        "Hypothesis 2: ** Exploring different model configurations will lead to improved accuracy in pizza detection.**"
        "\n- To validate this hypothesis, we experimented with various model architectures and hyperparameters, including fine-tuning a pre-trained VGG16 model and assessing performance."
        "\n---\n"
        "Hypothesis 3: ** The project's success relies not only on accurate pizza detection but also on the system's efficiency and response time.**"
        "\n- To validate this hypothesis, we assessed the system's real-time processing capabilities and its ability to handle multiple image classifications efficiently, ensuring it meets the client's operational requirements."
    )
