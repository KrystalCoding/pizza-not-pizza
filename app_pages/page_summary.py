import streamlit as st
import matplotlib.pyplot as plt


def page_summary_body():
    st.title("Pizza-Not-Pizza Classification Project Summary")

    st.write("### General Information")
    st.markdown(
        "The Pizza-Not-Pizza classification project aims to develop an advanced Machine Learning-based system capable of instantaneously detecting the presence of pizza in images. This project assists 'PizzaPal,' an innovative pizzeria, in revolutionizing its quality assurance process by automating the detection of pizza in images. "
    )

    st.write("### Dataset Content")
    st.markdown(
        "The dataset contains 983 featured photos of pizza variations, as well as 983 photos of food that is not pizza. This dataset is sourced from [Kaggle](https://www.kaggle.com/datasets/carlosrunner/pizza-not-pizza). "
    )

    st.write("### Business Requirements")
    st.success(
        "In today's dynamic food and beverage industry, rapid decision-making and quality control are paramount. That's why our team embarked on a mission to assist an innovative pizzeria, 'PizzaPal,' in revolutionizing its quality assurance process. " 
        "The client, PizzaPal, sought our expertise to develop an advanced Machine Learning-based system capable of instantaneously detecting the presence of pizza in images. "
    )

    st.markdown(
        "PizzaPal is renowned for its diverse menu of delectable pizzas, each meticulously crafted to culinary perfection. Ensuring that every pizza consistently meets its high standards is central to PizzaPal's brand reputation. " 
        "Manual inspection of thousands of pizza images to confirm quality and adherence to standards is not only time-consuming but also susceptible to human error. "
    )

    st.markdown(
        "To address this challenge, our solution is designed to automate the detection process, enabling PizzaPal to expedite quality assessments, reduce labor costs, and elevate customer satisfaction. " 
        "By instantly confirming the presence of pizza in images, PizzaPal gains a competitive edge in maintaining its culinary excellence. "
    )

    st.markdown(
        "Our system is tailored to PizzaPal's specific needs, enabling seamless integration into their existing workflow. It provides clear, accurate, and near-instant results, allowing PizzaPal's quality control team to focus their expertise on the finer aspects of pizza perfection. "
    )

    st.markdown(
        "In this school project, we take inspiration from the real-world need faced by PizzaPal, an imaginary but forward-thinking business customer. " 
        "Our objective is to demonstrate how Machine Learning can empower the food and beverage industry by automating image classification, enhancing quality control, and reducing operational inefficiencies. "
    )

    st.success(
        "Business Requirements for Pizza vs. Not-Pizza Image Classification System:\n\n"
        "1. Automated Pizza Detection: The system must automate the process of classifying images as either containing pizza or not-pizza. "
        "This automation should significantly reduce the time and effort required for manual image inspection.\n\n"
        "2. Prediction Accuracy: The client requires a reliable system capable of achieving high accuracy in classifying pizza and not-pizza images. "
        "The minimum acceptable accuracy should be defined based on the specific needs of the client.\n\n"
        "3. Scalability: The system should be scalable to handle a large number of images, reflecting the client's potential expansion of operations. "
        "It should efficiently process and classify images in real-time.\n\n"
        "4. User-Friendly Interface: The client expects an easy-to-use interface for uploading images and receiving classification results. "
        "The system should be intuitive for users with minimal technical expertise.\n\n"
        "5. Prediction Reporting: The system should provide prediction reports for each examined image, indicating the classification (pizza or not-pizza) and "
        "the associated confidence level or probability.\n\n"
        "6. Fast Processing: The client requires a system capable of processing images quickly and providing near-instantaneous results. "
        "This speed is essential for streamlining decision-making processes.\n\n"
        "7. Continuous Improvement: The system should support continuous improvement and model retraining to adapt to changes in image data patterns and "
        "to maintain high prediction accuracy."
    )


# Main program
if __name__ == "__main__":
    page_summary_body()

