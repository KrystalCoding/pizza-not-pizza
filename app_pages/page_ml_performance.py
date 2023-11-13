import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.image import imread
from src.machine_learning.evaluate_clf import load_test_evaluation


def page_ml_performance_metrics():
    """
    Display machine learning performance metrics in the Streamlit app.

    This function shows various metrics related to model training,
    including label distribution, accuracy, and loss graphs, and general
    erformance metrics on the test set.

    Parameters:
    None

    Returns:
    None
    """
    version = 'v1'

    st.write("### **Label Distribution Graph:** ")
    st.info(
        "**Description:** The distribution of images across the train, "
        "validation, and test sets. Balanced representations of both pizza "
        "and not-pizza classes ensure a robust model training and evaluation."
    )

    labels_distribution = plt.imread(
        f"outputs/{version}/labels_distribution.png")
    st.image(labels_distribution,
             caption='Labels Distribution on Train, Validation and Test Sets')
    st.write("---")

    st.write("### **Loss and Accuracy Plot:**")
    st.info(
        "**Description:** This plot depicts the training progress over five "
        "epochs. Notable improvements in accuracy and reductions in loss "
        "are observed, indicating the model's ability to learn from the "
        "training data. Validation accuracy consistently increases, "
        "ensuring the model generalizes well to new data.\n"
        "- **Accuracy (0.8579):** This indicates that the model correctly "
        "predicts the class (pizza or not-pizza) about 85.8% of the time "
        "on the test set. Generally, an accuracy above 80% is considered "
        "good, especially for a beginner's project. We are satisfied with this "
        "number, but believe that with further fine tuning, we could raise "
        "the accuracy of predictions.\n"
        "- **Loss (0.7276):** The loss is a measure of how well the model is "
        "performing, with lower values indicating better performance. A "
        "loss of 0.7276 is reasonably low, suggesting that the model is "
        "effective in minimizing errors during training."
    )
    col1, col2 = st.beta_columns(2)
    with col1:
        model_acc = plt.imread(f"outputs/{version}/model_training_acc.png")
        st.image(model_acc, caption='Model Training Accuracy')
    with col2:
        model_loss = plt.imread(f"outputs/{version}/model_training_losses.png")
        st.image(model_loss, caption='Model Training Losses')
    st.write("---")

    st.write("### **Generalised Performance on Test Set:**")
    st.info(
        "**Description:** A summary of the model's performance on the test "
        "set. The model achieved a loss of 0.7276 and an accuracy of 0.8579. "
        "These metrics provide insights into how well the trained model "
        "performs on previously unseen data, validating its effectiveness "
        "in real-world scenarios."
    )
    st.dataframe(pd.DataFrame(load_test_evaluation(
        version), index=['Loss', 'Accuracy']))
    st.write("---")

    st.subheader("Page Summary")
    st.info(
        "**Business Requirements Addressed:**\n"
        "- Automated Pizza Detection (Partially addressed through label "
        "distribution visualization)\n"
        "- Prediction Accuracy (Partially addressed through label distribution "
        "visualization and model training graphs)\n"
        "- Scalability (Partially addressed through label distribution "
        "visualization)\n"
        "- Fast Processing (Partially addressed through model training "
        "graphs)\n"
        "- Continuous Improvement (Addressed through model training history)"
        "\n---\n"
        "**Conclusions:**\n"
        "- Visualization of label distribution and model training accuracy "
        "meets the requirements in User Story 1.\n"
        "- Visualization of model training losses fulfills the need to monitor "
        "model training, as per User Story 2.\n"
        "- Displaying general performance metrics on the test set (Loss and "
        "Accuracy) addresses the requirement in User Story 4."
    )
