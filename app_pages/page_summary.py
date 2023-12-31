import streamlit as st
import matplotlib.pyplot as plt


def page_summary_body():
    """
    Display the summary of the Pizza-Not-Pizza classification project.

    This function provides an overview of the project, including general
    information, dataset content, business requirements,
    and objectives.

    Parameters:
    None

    Returns:
    None
    """
    st.header("Pizza-Not-Pizza Classification Project Summary")

    st.subheader("General Information")
    st.info(
        "The Pizza-Not-Pizza classification project aims to develop an "
        "advanced Machine Learning-based system capable of instantaneously "
        "detecting the presence of pizza in images. This project "
        "assists 'PizzaPal,' an innovative pizzeria, in revolutionizing its "
        "quality assurance process by automating the detection of pizza in "
        "images. "
    )

    st.subheader("Dataset Content")
    st.markdown(
        "The dataset contains 983 featured photos of pizza variations, "
        "as well as 983 photos of food that is not pizza. This "
        "dataset is sourced from [Kaggle](https://www.kaggle.com/"
        "datasets/carlosrunner/pizza-not-pizza)"
    )

    st.subheader("Business Requirements")
    st.success(
        "In today's dynamic food and beverage industry, rapid decision-making "
        "and quality control are paramount. That's why our team embarked on a "
        "mission to assist an innovative pizzeria, 'PizzaPal,' in "
        "revolutionizing its quality assurance process. "
        "The client, PizzaPal, sought our expertise to develop an advanced "
        "Machine Learning-based system capable of instantaneously detecting "
        "the presence of pizza in images. "
    )

    st.markdown(
        "PizzaPal is renowned for its diverse menu of delectable pizzas, each "
        "meticulously crafted to culinary perfection. Ensuring that every "
        "pizza consistently meets its high standards is central to PizzaPal's "
        "brand reputation. "
        "Manual inspection of thousands of pizza images to confirm quality and"
        " adherence to standards is not only time-consuming but also "
        "susceptible to human error. "
    )

    st.markdown(
        "To address this challenge, our solution is designed to automate the "
        "detection process, enabling PizzaPal to expedite quality assessments,"
        " reduce labor costs, and elevate customer satisfaction. "
        "By instantly confirming the presence of pizza in images, PizzaPal "
        "gains a competitive edge in maintaining its culinary excellence. "
    )

    st.markdown(
        "Our system is tailored to PizzaPal's specific needs, enabling "
        "seamless integration into their existing workflow. It provides clear,"
        " accurate, and near-instant results, allowing PizzaPal's quality "
        "control team to focus their expertise on the finer aspects of pizza "
        "perfection. "
    )

    st.markdown(
        "In this school project, we take inspiration from the real-world need "
        "faced by PizzaPal, an imaginary but forward-thinking business "
        "customer. "
        "Our objective is to demonstrate how Machine Learning can empower the "
        "food and beverage industry by automating image classification, "
        "enhancing quality control, and reducing operational inefficiencies. "
    )

    st.success(
        "1. **Automated Pizza Detection**: The system must automate the "
        "process of classifying images as either containing pizza or "
        "not-pizza. This automation should significantly reduce the time and "
        "effort required for manual image inspection.\n"
        "\n---\n"
        "2. **Prediction Accuracy**: The client requires a reliable system "
        "capable of achieving high accuracy (at least 80%) in classifying "
        "pizza and not-pizza images.\n"
        "\n---\n"
        "3. **Scalability**: The system should be scalable to handle a large "
        "number of images, reflecting the client's potential expansion of "
        "operations. It should efficiently process and classify images in "
        "real-time.\n"
        "\n---\n"
        "4. **User-Friendly Interface**: The client expects an easy-to-use "
        "interface for uploading images and receiving classification results. "
        "The system should be intuitive for users with minimal technical "
        "expertise.\n"
        "\n---\n"
        "5. **Prediction Reporting**: The system should provide prediction "
        "reports for each examined image, indicating the classification "
        "(pizza or not-pizza) and the associated confidence level or "
        "probability.\n"
        "\n---\n"
        "6. **Fast Processing**: The client requires a system capable of "
        "processing images quickly and providing near-instantaneous results. "
        "This speed is essential for streamlining decision-making processes.\n"
        "\n---\n"
        "7. **Continuous Improvement**: The system should support continuous "
        "improvement and model retraining to adapt to changes in image data "
        "patterns and to maintain high prediction accuracy."
    )

    # Insert conclusions for Business Requirement 1: Data Visualization
    st.subheader("Data Visualization Conclusions")

    # User Story 1: Interactive Dashboard Navigation
    st.markdown(
        "**User Story 1: Interactive Dashboard Navigation**\n"
        "- **Conclusion:** The implementation of a Streamlit-based interactive "
        "dashboard has successfully addressed the need for easy navigation. "
        "Users can seamlessly explore and comprehend presented data through "
        "an intuitive sidebar."
    )

    # User Story 2: Mean and Standard Deviation Visualization
    st.markdown(
        "**User Story 2: Mean and Standard Deviation Visualization**\n"
        "- **Conclusion:** The creation of 'mean' and 'standard deviation' "
        "images for both pizza and non-pizza categories provides users with "
        "visual cues for differentiating between the two."
    )

    # User Story 3: Visualizing Differences between Averages
    st.markdown(
        "**User Story 3: Visualizing Differences between Averages**\n"
        "- **Conclusion:** The showcased disparity between average pizza and "
        "non-pizza images aids in visual differentiation, contributing to a "
        "better understanding of distinguishing features."
    )

    # User Story 4: Image Montage for Visual Differentiation
    st.markdown(
        "**User Story 4: Image Montage for Visual Differentiation**\n"
        "- **Conclusion:** The development of the image montage feature enhances "
        "visual differentiation, allowing users to explore a collection of pizza "
        "and non-pizza images in a consolidated format."
    )

    # Insert conclusions for Business Requirement 2: Classification
    st.subheader("Classification Conclusions")

    # User Story 5: ML Model for Pizza Detection
    st.markdown(
        "**User Story 5: ML Model for Pizza Detection**\n"
        "- **Conclusion:** The machine learning model has been successfully "
        "trained to predict with an accuracy of 85% or above on the test set. "
        "Users can now upload food images, and the model provides instant "
        "evaluations for pizza detection."
    )

    # Insert conclusions for Business Requirement 3: Report Generation
    st.subheader("Report Generation Conclusions")

    # User Story 6: ML Predictions Report
    st.markdown(
        "**User Story 6: ML Predictions Report**\n"
        "- **Conclusion:** The integration of a feature into the Streamlit "
        "dashboard enables the generation of downloadable .csv reports after "
        "each batch of images is uploaded. This feature provides a comprehensive "
        "overview of prediction results."
        "\n---\n"
    )
    
    st.subheader("Page Summary")
    st.info(
        "**Business Requirements Addressed:**\n"
        "- Automated Pizza Detection (Partially addressed through project "
        "summary and dashboard introduction)\n"
        "- Prediction Accuracy (Partially addressed through project summary "
        "and dashboard introduction)\n"
        "- Scalability\n"
        "- User-Friendly Interface  (Partially addressed through project "
        "summary and dashboard introduction)\n"
        "\n---\n"
        "**Conclusions:**\n"
        "- The interactive dashboard simplifies navigation, ensuring users can "
        "effortlessly explore and comprehend data, meeting the User Story 1 "
        "requirement."
    )


# Main program
if __name__ == "__main__":
    page_summary_body()
