import streamlit as st
import matplotlib.pyplot as plt


def page_project_hypothesis_body():
    """
    Display project hypotheses and their validation processes.

    This function provides an overview of the project's hypotheses and 
    how they were validated through experiments.

    Parameters:
    None

    Returns:
    None
    """
    st.write("### Project Hypothesis and Validation")

    st.success(
        "Hypothesis 1 -  Pizza Detection Features: ** Pizza presence can be "
        "accurately identified "
        "by analyzing the shape and toppings within images.**\n"
        "- **Validaiton:** Analyzed pizza images for circular shape and "
        "toppings.\n"
        "- **Outcome:** Model successfully captures subtle features for "
        "accurate pizza detection."
        "\n---\n"
        "Hypothesis 2 - Model Configurations: ** Exploring different model "
        "configurations will lead to improved accuracy in pizza detection.**\n"
        "- **Validation:** Explored various model architectures and "
        "hyperparameters, including fine-tuning a pre-trained VGG16 model "
        "and assessing performance.\n"
        "- **Outcome:** VGG16-based model shows promise; ongoing refinements "
        "are needed for high accuracy."
        "\n---\n"
        "Hypothesis 3 - System Efficiency and Response: ** The project's "
        "success relies not only on accurate pizza detection but also on the "
        "system's efficiency and response time.**\n"
        "- **Validation:** Assessed system's real-time processing and "
        "scalability for efficient image classifications.\n"
        "- **Outcome:** System meets operational needs beyond accurate "
        "pizza detection."
    )

    st.info(
        "**User Story 1: Automated Pizza Detection:**\n"
        "- Hypothesis 1 addresses the need for accurate pizza identification.\n"
        "- Hypothesis 2 explores model configurations for improved accuracy.\n"
        "- Hypothesis 3 ensures system efficiency in pizza detection."
        "\n---\n"
        "**User Story 5: Prediction Accuracy:**\n"
        "- Hypothesis 1 demonstrates successful feature extraction for "
        "accurate predictions.\n"
        "- Hypothesis 2 aims at improving accuracy through model "
        "configuration enhancements."
    )
