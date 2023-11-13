![Responsive Design](assets/responsive_design.png)

## Table of Contents
1. [Dataset Content](#dataset-content)
2. [Business Requirements](#business-requirements)
3. [Hypothesis and validation](#hypothesis-and-validation)
4. [Rationale for the model](#the-rationale-for-the-model)
5. [Implementation of the Business Requirements](#the-rationale-to-map-the-business-requirements-to-the-data-visualizations-and-ml-tasks)
6. [ML Business case](#ml-business-case)
7. [Dashboard design](#dashboard-design-streamlit-app-user-interface)
8. [CRISP DM Process](#the-process-of-cross-industry-standard-process-for-data-mining)
9. [Bugs](#bugs)
10. [Deployment](#deployment)
11. [Technologies used](#technologies-used)
12. [Credits](#credits)

### Deployed version at ["Pizza Predictor"](https://pizza-detector-0540d49673e2.herokuapp.com/)

## Dataset Content

The dataset contains 983 featured photos of pizza variations, as well as 983 photos of food that is not pizza. This client is particularly concerned about identifying pizza as it is their flagship product. The dataset is sourced from [Kaggle](https://www.kaggle.com/code/rasikagurav/pizza-or-not-pizza/input).

<details><summary>See Image</summary>
<img src="KAGGLE">
</details>

## Business Requirements

In today's dynamic food and beverage industry, rapid decision-making and quality control are paramount. That's why our team embarked on a mission to assist an innovative pizzeria, "PizzaPal," in revolutionizing its quality assurance process. The client, PizzaPal, sought our expertise to develop an advanced Machine Learning-based system capable of instantaneously detecting the presence of pizza in images.

PizzaPal is renowned for its diverse menu of delectable pizzas, each meticulously crafted to culinary perfection. Ensuring that every pizza consistently meets its high standards is central to PizzaPal's brand reputation. Manual inspection of thousands of pizza images to confirm quality and adherence to standards is not only time-consuming but also susceptible to human error.

To address this challenge, our solution is designed to automate the detection process, enabling PizzaPal to expedite quality assessments, reduce labor costs, and elevate customer satisfaction. By instantly confirming the presence of pizza in images, PizzaPal gains a competitive edge in maintaining its culinary excellence.

Our system is tailored to PizzaPal's specific needs, enabling seamless integration into their existing workflow. It provides clear, accurate, and near-instant results, allowing PizzaPal's quality control team to focus their expertise on the finer aspects of pizza perfection.

In this school project, we take inspiration from the real-world need faced by PizzaPal, an imaginary but forward-thinking business customer. Our objective is to demonstrate how Machine Learning can empower the food and beverage industry by automating image classification, enhancing quality control, and reducing operational inefficiencies.

Join us on this journey as we explore the capabilities of Machine Learning to transform the way businesses, like our visionary client PizzaPal, maintain their commitment to excellence in the world of food and beverage. Discover how cutting-edge technology can optimize operations, improve product quality, and drive success in a competitive industry.

Business Requirements for Pizza vs. Not-Pizza Image Classification System:

1. **Automated Pizza Detection:** The system must automate the process of classifying images as either containing pizza or not-pizza. This automation should significantly reduce the time and effort required for manual image inspection.

2. **Prediction Accuracy:** The client requires a reliable system capable of achieving high accuracy in classifying pizza and not-pizza images. The minimum acceptable accuracy should be defined based on the specific needs of the client.

3. **Scalability:** The system should be scalable to handle a large number of images, reflecting the client's potential expansion of operations. It should efficiently process and classify images in real-time.

4. **User-Friendly Interface:** The client expects an easy-to-use interface for uploading images and receiving classification results. The system should be intuitive for users with minimal technical expertise.

5. **Prediction Reporting:** The system should provide prediction reports for each examined image, indicating the classification (pizza or not-pizza) and the associated confidence level or probability.

6. **Fast Processing:** The client requires a system capable of processing images quickly and providing near-instantaneous results. This speed is essential for streamlining decision-making processes.

7. **Continuous Improvement:** The system should support continuous improvement and model retraining to adapt to changes in image data patterns and to maintain high prediction accuracy.

## Hypothesis and validation

1. **Hypothesis** (Pizza Detection Features): Pizza presence can be accurately identified by analyzing the shape and toppings within images.
   - __Validation__: To validate this hypothesis, we will gather a dataset of pizza images with various shapes and toppings. We will conduct a detailed analysis, including feature extraction and shape recognition. The validation process will involve developing a feature extraction pipeline to capture key pizza characteristics, such as circular shape and the presence of toppings. We will then build an average image study to examine common patterns in pizza images, emphasizing shape and topping distribution.

2. **Hypothesis** (Model Configurations): Exploring different model configurations will lead to improved accuracy in pizza detection.
   - __Validation__: Perform a comprehensive exploration of various model configurations. We will experiment with different network architectures, layers, activation functions, and hyperparameters. For each configuration, we will conduct extensive training and evaluation, keeping the dataset and other factors consistent. We will compare the accuracy and performance metrics of each model to determine which configurations lead to improved pizza detection accuracy.

3. **Hypothesis** (System Efficiency and Response): The Project's Success Relies on Accurate Pizza Detection and Efficient System Response.  
   - __Validation__: Assess system's real-time processing and scalability for efficient image classifications.

### Hypothesis 1
> Pizza presence can be accurately identified by analyzing the shape and toppings within images.

**1. Introduction**

We hypothesize that pizzas exhibit distinctive characteristics that can be leveraged for accurate identification. One of the primary identifiers is the circular, flat shape of pizzas, typically accompanied by a variety of toppings encapsulated within the circular mass. To harness this inherent property in the context of machine learning, we need to preprocess the images to ensure optimal feature extraction and model training.

When working with an image dataset, the crucial step of normalizing the images before training a neural network serves two fundamental purposes:

1. **Consistent Results**: Normalization ensures that the trained neural network produces reliable and consistent predictions when faced with new test images. This consistency is vital for the model's generalization to unseen data.

2. **Facilitating Transfer Learning**: Normalization is integral for transfer learning, a technique where knowledge gained from training on one task is applied to a different but related task. By bringing images to a standardized scale, normalization aids in leveraging pre-existing knowledge from one dataset to enhance the performance on another.

To normalize an image, you need to calculate the **mean** and **standard deviation** of the entire dataset. These values are calculated separately for each color channel (red, green, and blue in the case of RGB images). The calculation involves considering four dimensions of an image: batch size (number of images), number of channels (3 for RGB), height, and width. Since it's impractical to load the entire dataset into memory, the calculation is done on small batches of images, making it a non-trivial task.

<details><summary>See Image</summary>
<img src="MEAN AND STANDARD">
</details>

[Back to top](#table-of-contents)

**2. Observation**

To validate our hypothesis, we observed the following key characteristics:

- Shape Analysis: Pizza images consistently display a circular and flat shape. This distinct feature can serve as a crucial discriminator in identifying pizzas.

- Toppings Variation: The toppings on pizzas vary widely, providing additional cues for detection. These toppings, such as pepperoni, vegetables, or cheese, introduce unique textural and color patterns that can be learned by our model.

**3. Image Analysis**

In addressing business requirement, **Automated Pizza Detection**, we created an accessible and interactive image montage which aids in visually representing distinctive characteristics of pizza images, such as their circular and flat shape, and the variations in toppings. This visual representation contributes to the automated identification of pizzas.

**User Story 1: Easy Navigation and Understanding**: Users, including those without deep technical knowledge, can easily navigate and understand the dataset characteristics through visual aids like image montages, fulfilling the need for intuitive and user-friendly exploration.

**User Story 2: Enhanced Visual Differentiation**: The montage provides a means to enhance visual differentiation by showcasing the diversity in pizza shapes and toppings. This meets the user story requirement for improved visualization features.

**User Story 4: Intuitive Data Representation**: The image montage supports an intuitive representation of data patterns, contributing to meeting the user story requirement for intuitive data representation and exploration.

- Shape Comparison: A montage of pizza images clearly illustrates the uniform circular shape found in pizzas. In contrast, we created a montage of "not-pizza" images, which showcase diverse and irregular shapes. This striking difference serves as a foundation for differentiation.

<details><summary>See Image</summary>
<img src="PIZZA MONTAGE">
</details>
<details><summary>See Image</summary>
<img src="NOT-PIZZA MONTAGE">
</details>

- Toppings Diversity: Analyzing the average and variability in images, we noticed that pizzas tend to exhibit a more centered and circular pattern. In contrast, "not-pizza" images display a wider array of shapes and patterns, emphasizing the uniqueness of pizza toppings.

<details><summary>See Image</summary>
<img src="AVERAGE AND VARIABILITY FOURSOME">
</details>

- Averaging Images: Comparing the average pizza image to the average "not-pizza" image did not reveal any immediate and intuitive difference. This suggests that pizza detection relies on a combination of subtle features, including shape and toppings.

<details><summary>See Image</summary>
<img src="AVERAGES AND DIFFERENCE W BLACK 3RD">
</details>

[Back to top](#table-of-contents)

**3. Conclusion**

Our model demonstrated its capacity to detect these subtle yet distinguishing features, enabling it to make accurate predictions. An effective model goes beyond memorizing training data but generalizes the essential patterns that connect image features to labels. This generalization allows the model to confidently predict pizza presence in future observations, contributing to the automation of pizza detection in our project.

**Sources**:

- [Calculate mean and std of Image Dataset](https://iq.opengenus.org/calculate-mean-and-std-of-image-dataset/)
- [Computing Mean & STD in Image Dataset](https://kozodoi.me/python/deep%20learning/pytorch/tutorial/2021/03/08/image-mean-std.html)

[Back to top](#table-of-contents)

---
### Hypothesis 2
> Exploring different model configurations will lead to improved **Prediction Accuracy**. The experimentation with various model configurations directly contributes to enhancing the accuracy of pizza detection, aligning with the overarching business requirement for accurate predictions.

**User Story 5: Improved Model Accuracy**: The systematic exploration of different model configurations aims at improving accuracy, aligning with the user story requirement for continuous improvement and achieving higher accuracy in pizza detection.

**User Story 6: Systematic Evaluation**: The evaluation of different network architectures, layers, activation functions, and hyperparameters exemplifies a systematic approach to model development, fulfilling the user story requirement for a methodical and informed exploration of configurations.

**1. Introduction**

Understanding the Classification Problem:

In our pizza-not-pizza project, we face a classification problem. We aim to classify images into one of two categories: pizza or not-pizza. This **binary classification** requires us to choose an appropriate activation function for the output layer of our Convolutional Neural Network (CNN).

- **Epoch**: An epoch signifies one complete pass through the training dataset.
- **Loss**: It quantifies how bad the model's prediction is. A lower loss value indicates a better prediction.
- **Accuracy**: Accuracy is the proportion of correct predictions made by the model.

<details><summary>See Image</summary>
<img src="MODEL OUTPUT SHOWING EPOCHS">
</details>

In our learning curve plots, we look for the right fit of the learning algorithm, avoiding both overfitting and underfitting. A good fit is characterized by the following:

- The training loss decreases (or accuracy increases) to a point of stability.
- The validation loss decreases (or accuracy increases) to a point of stability with a small gap compared to the training loss.
- Continued training of a well-fitted model may lead to overfitting. This is why ML models usually have an [early stopping](https://en.wikipedia.org/wiki/Early_stopping) function utilized which interrupts the model's learning phase when it ceasing improving.

<details><summary>See Image</summary>
<img src="EARLY STOPPING CODE">
</details>

**2. Observation**

Our experimentation in the pizza-not-pizza project involved various model configurations and hyperparameter adjustments. We initiated the process with a custom model that featured three convolutional layers, max-pooling, and dense layers. This model was trained with a batch size of 20 for 25 epochs. However, we observed that the custom model did not achieve the desired accuracy, and the loss did not decrease significantly during training. It struggled to capture the intricate features that distinguish pizza from non-pizza images.

<details><summary>See Image</summary>
<img src="OWN MODEL">
</details>

As an alternative, we explored the pre-trained VGG16 model. By fine-tuning the top layers to adapt to our binary classification task of pizza detection, we achieved better results. With a batch size of 35 and training for only 5 epochs, this VGG16-based model displayed improved accuracy. It successfully captured nuanced patterns and features critical for distinguishing between pizza and not-pizza images. Moreover, the loss function showed consistent decreases, indicating better convergence.

<details><summary>See Image</summary>
<img src="VGG16 MODEL 1">
</details>

Encouraged by this initial progress, we further refined our VGG16-based model. We reduced the batch size to 15 and incorporated additional layers, including dense layers, L2 regularization, and dropout layers. We set the batch size to 20. These modifications led to significant improvements in loss. However, we continued to grapple with accuracy, as this model was overfitted and could not make accurate predictions on previously unseen photos.

<details><summary>See Image</summary>
<img src="VGG16 Model 2">
</details>

In order to reduce overfitting, we set the batch size to 16, reduced the two dense layers down to only one simplified one without the l2 parameter, set the patience from 3 to 5, changed the learning rate from 0.001 to 0.0001, and set the epochs to 5.

<details><summary>See Image</summary>
<img src="VGG16 Model 3">
</details>

In summary, our experimentation revealed that the VGG16-based model, with fine-tuned top layers and additional modifications, exhibited potential in distinguishing pizza from not-pizza images. Despite these advancements, achieving high accuracy remained a challenge. Our primary focus in this experiment was to evaluate different model architectures and hyperparameters with the aim of enhancing classification performance for our specific problem.

**3. Conclusion**

In our process of experimentation, we observed that the pre-trained VGG16 model, with fine-tuned top layers, showed promise in distinguishing pizza from not-pizza images. However, achieving high accuracy remained a challenge despite various enhancements to the model. When we achieved an accuracy report of 92+%, it was an exciting moment. It, however, quickly led to a let-down, as that proved to be our 3/4 attempt which was an overfitted model. 

The primary focus of our experiment was to evaluate different model architectures and hyperparameters to improve classification performance for our specific problem. As a result, our conclusions are based on the differences between our custom model and the VGG16-based models. Further refinements and investigations are needed to enhance accuracy and improve the model's performance.

- Loss/Accuracy of our custom model:

<details><summary>See Image</summary>
<img src="OWN MODEL LOSS & ACCURACY PLOTS">
</details>

- Loss/Accuracy of original VGG16 model:

<details><summary>See Image</summary>
<img src="VGG16 1 LOSS & ACCURACY PLOTS">
</details>

- Loss/Accuracy of enhanced VGG16 model:

<details><summary>See Image</summary>
<img src="VGG16 2 LOSS & ACCURACY PLOTS">
</details>

- Final and best model:

<details><summary>See Image</summary>
<img src="VGG16 3 LOSS & ACCURACY PLOTS">
</details>

**Sources**:
- [Backpropagation in Fully Convolutional Networks](https://towardsdatascience.com/backpropagation-in-fully-convolutional-networks-fcns-1a13b75fb56a#:~:text=Backpropagation%20is%20one%20of%20the,respond%20properly%20to%20future%20urges.) by [Giuseppe Pio Cannata](https://cannydatascience.medium.com/)
- [How to use Learning Curves to Diagnose Machine Learning Model Performance](https://machinelearningmastery.com/learning-curves-for-diagnosing-machine-learning-model-performance/) by [Jason Brownlee](https://machinelearningmastery.com/about)
- [Activation Functions: Comparison of Trends in Practice and Research for Deep Learning](https://arxiv.org/pdf/1811.03378.pdf) by *Chigozie Enyinna Nwankpa, Winifred Ijomah, Anthony Gachagan, and Stephen Marshall*

[Back to top](#table-of-contents)

---
### Hypothesis 3 
> The Project's Success Relies on Accurate Pizza Detection and Efficient System Response, addressing business requirement **Continuous Improvement**: Beyond accurate pizza detection, the project's success hinges on system efficiency, response time, and scalability. Validating this hypothesis ensures the system identifies pizzas accurately and meets operational needs.

**User Story 4: Image Montage for Visual Differentiation**: Users seek assurance that the system efficiently handles a larger volume of image classifications for potential business growth.

In the context of Hypothesis 3, system efficiency and response time serve as a form of visual differentiation. The system's quick and accurate distinction between pizza and non-pizza images contributes to enhanced visual differentiation, aligning with the goals of User Story 4.

**1. Introduction**

In our pizza-not-pizza project, accuracy is crucial, but success transcends classification. It includes system efficiency and response time. We hypothesize that success relies on accurate pizza detection, real-time processing, and efficient handling of a large number of image classifications.

**2. Observation**

To validate this hypothesis, we assessed the system's real-time processing capabilities and its efficiency in handling multiple image classifications. This evaluation aims to ensure that the system meets the client's operational requirements. We measured the response time of the system when classifying images, ensuring it can provide near-instantaneous results, which is essential for streamlining decision-making processes. Additionally, we examined the system's scalability to handle a larger volume of images, reflecting the client's potential expansion of operations.

Performance analysis: Moderately Successful
While our code can handle multiple user photo uploads, and Streamlit offers pagination for multiple image uploads, the system seems to crash after aproximately 4 image uploads by any one user during a session. It is functional, but offers the capacity you might expect to get from a free dashboard system.

<details><summary>See Image</summary>
<img src="PAGINATION">
</details>

**3. Conclusion**

The success of our pizza-not-pizza project extends beyond accurate pizza detection, incorporating system efficiency, response time, and scalability. By validating this hypothesis, we ensure the system identifies pizzas accurately while meeting the operational needs of our client.


[Back to top](#table-of-contents)

## The rationale for the model

The VGG16 model is a convolutional neural network with 13 convolutional layers and 3 fully connected layers. It uses a predefined architecture with multiple convolutional and pooling layers, followed by three fully connected layers and an output layer for classification.

<details><summary>See Image</summary>
<img src="VGG16 Model 3 (again)">
</details>

### The goal
zer. Achieving the desired model architecture was a result of systematic trial and error.
It's important to note that while the model we've arrived at may not be the absolute best, it represents the outcome of extensive testing and fine-tuning in alignment with our project goals.

Our primary aim was to develop a robust model that excels in its ability to predict classes from a batch of data while maintaining a high level of generalization. We avoided overfitting, ensuring that the model doesn't merely memorize the training data but learns the underlying patterns that connect features to labels.

Furthermore, we sought to maintain computational efficiency by striking a balance between neural network complexity and the number of trainable parameters. This optimization allowed us to achieve a model that can generalize effectively, maintain high accuracy, and minimize error, all while conserving computational resources.

[Back to top](#table-of-contents)

### Configuring Model Hyperparameters

- **Convolutional Layer Size**: Our pizza detection project utilizes 2D convolutional layers (Conv2D) for processing 2D image data. This choice is optimal, considering that 1D convolutional layers are tailored for 1D data, such as time series.

- **Convolutional Kernel Size**: We employ a 3x3 convolutional filter, effectively processing our 2D image data. This kernel size works well for our images, allowing for zero padding to maintain image size.

- **Number of Neurons**: We select the number of neurons in layers as powers of 2, optimizing computational efficiency. This aligns with the GPU's ability to leverage optimizations related to power-of-two dimensions.

- **Activation Function**: The model uses the `ReLu` (Rectified Linear Unit) activation function for computational efficiency and proven effectiveness in training deep neural networks. Its derivative is either 0 or 1, helping mitigate the vanishing gradient problem.

- **Pooling**: Max pooling is utilized to reduce variance and computational complexity in our pizza detection model. This is apt for identifying pizzas against a relatively darker background by selecting brighter pixels.

- **Output Activation Function**: For **binary classification** of pizza and not-pizza, the model employs the `sigmoid`, ideal for such tasks, producing probabilities in the 0 to 1 range.

- **Dropout**:  The model incorporates a dropout rate of 0.5 to prevent overfitting, especially given the relatively limited number of training samples.

**Source**: 
- [How to choose the size of the convolution filter or Kernel size for CNN?](https://medium.com/analytics-vidhya/how-to-choose-the-size-of-the-convolution-filter-or-kernel-size-for-cnn-86a55a1e2d15) by - [Swarnima Pandey](https://medium.com/@pandeyswarnima)
- [The advantages of ReLu](https://stats.stackexchange.com/questions/126238/what-are-the-advantages-of-relu-over-sigmoid-function-in-deep-neural-networks#:~:text=The%20main%20reason%20why%20ReLu,deep%20network%20with%20sigmoid%20activation.)
- [Maxpooling vs minpooling vs average pooling](https://medium.com/@bdhuma/which-pooling-method-is-better-maxpooling-vs-minpooling-vs-average-pooling-95fb03f45a9#:~:text=Average%20pooling%20method%20smooths%20out,lighter%20pixels%20of%20the%20image.) by 
- [How ReLU and Dropout Layers Work in CNNs](https://www.baeldung.com/cs/ml-relu-dropout-layers)

[Back to top](#table-of-contents)

### Hidden Layers

Hidden layers are vital in our pizza detection project, playing a crucial role in feature extraction and classification based on those features.

When designing hidden layers, two key decisions need attention:

1. **Number of Hidden Layers**: Determining how many hidden layers to include is crucial, avoiding underfitting with too few or overfitting with too many. Our design prioritizes generalization, aiming for effective performance on both training data and new, unobserved images.

2. **Number of Neurons in Each Layer**: Striking a balance is crucial, aiming to have enough neurons to capture intricate features distinguishing pizza images without an excessive number that might lead to overfitting.

In our project, Convolutional Layers handle feature extraction, while Fully Connected Layers make final classifications, each serving its purpose effectively.

[Back to top](#table-of-contents)

- **Convolutional Layers vs. Fully Connected Layers**:
  - **Convolutional Layers**: In our pizza detection model, these layers are specialized for image analysis and feature extraction using convolution. Sharing parameters significantly reduces the parameter count compared to Fully Connected Layers, making them essential for capturing intricate patterns in pizza images.
  - **Fully Connected Layers**: Also known as Dense Layers, these layers are employed in our pizza detection model for the final classification task, distinguishing between pizza and not-pizza images. They perform a linear operation considering every input, making them suitable for our classification goal.

In simplifying the model from two Dense layers with L2 regularization to one simplified Dense layer, we aimed to achieve a balance between model complexity and performance:

1. **Reducing Model Complexity**: Having fewer parameters in the model contributes to a simpler architecture. Simpler models are often preferred, especially when dealing with limited amounts of training data. They are less prone to overfitting, where the model performs well on the training data but struggles with new, unseen data. We switched to this directly after our model proved to be dramatically overfitted.

2. **Avoiding Overfitting**: L2 regularization in Dense layers introduces additional parameters to penalize large weights in the model. While regularization techniques are valuable for preventing overfitting, having a single simplified Dense layer can be effective in achieving regularization without overly complicating the model structure.

3. **Computational Efficiency**: Training a model with fewer parameters typically requires less computational resources and time. This is particularly important in scenarios where computational efficiency is a consideration, such as when deploying the model in resource-constrained environments.

4. **Empirical Validation**: Through experimentation and model evaluation, it was observed that the simplified architecture with one Dense layer provided satisfactory results in terms of accuracy and generalization performance. This empirical validation supported the decision to simplify the model.

In summary, the choice to simplify the model by reducing the number of Dense layers and removing L2 regularization was driven by a desire for a simpler, more computationally efficient architecture that still maintained effective performance on the pizza detection task.

**Source**: 
- [Dense Layer vs convolutional layer](https://datascience.stackexchange.com/questions/85582/dense-layer-vs-convolutional-layer-when-to-use-them-and-how#:~:text=As%20known%2C%20the%20main%20difference,function%20based%20on%20every%20input.)

[Back to top](#table-of-contents)


### Model Compilation

- **Loss**: The loss function is a crucial component that measures the disparity between the predicted and actual output values, reflecting how effectively the neural network models the training data. In our pizza detection project, we employed `binary_crossentropy` as the loss function. This choice aligns with our binary classification task of distinguishing between pizza and not-pizza images.

- **Optimizer**: The optimizer plays a vital role in adjusting neural network attributes, such as weights and learning rates, to expedite convergence while minimizing loss and maximizing accuracy. In our project, we opted for the `adam` optimizer after thorough experimentation and the trial-and-error phase. `Adam` optimization has proven to be effective in various machine learning tasks.

- **Metrics**: The selected metric for assessing model performance is `accuracy`. It quantifies how frequently the model's predictions match the actual labels in our binary classification problem. This metric keeps track of two local variables, total and count, to determine the `accuracy` of the predictions.  

**Source**: 
- [7 tips to choose the best optimizer](https://towardsdatascience.com/7-tips-to-choose-the-best-optimizer-47bb9c1219e) by [Davide Giordano](https://medium.com/@davidegiordano)
- [Impact of Optimizers in Image Classifiers](https://towardsai.net/p/l/impact-of-optimizers-in-image-classifiers)
- [Keras Accuracy Metrics](https://keras.io/api/metrics/accuracy_metrics/#:~:text=metrics.,with%20which%20y_pred%20matches%20y_true%20.)

[Back to top](#table-of-contents)

### Model Training
- **Dropout Rate**: A dropout rate of 0.5 is implemented to prevent overfitting, especially given the relatively limited number of training samples.

- **Early Stopping**: Early stopping is applied to prevent overfitting by monitoring val_loss and halting training after a patience of 5 epochs.

- **Transfer Learning with VGG16**: The pre-trained VGG16 model is loaded, and the top layers are customized for the binary classification task. The layers are frozen to retain pre-trained knowledge.

- **Model Compilation**: The model is compiled with a learning rate adjusted to 0.0001, aiming for improved convergence.

- **ModelCheckpoint**: A checkpoint is defined to save the best model based on val_loss.

- **Training**: The model is trained for 5 epochs with the defined parameters, utilizing the training and validation sets.

[Back to top](#table-of-contents)

## The Rationale for Mapping Business Requirements to Data Visualizations and ML Tasks

Our project revolves around meeting the business requirements of our client, PizzaPal, aiming to enhance its quality assurance process through innovative Machine Learning. The defined business requirements drive the development of a sophisticated system capable of swiftly detecting the presence of pizza in images.

[Back to top](#table-of-contents)

### Business Requirement 1: Data Visualization 
>The initial business requirement focuses on creating data visualizations to intuitively distinguish pizzas from non-pizzas. Each user story in this category corresponds to a specific ML task.

- **User Story 1**: Navigation through an interactive dashboard for visual comprehension.

* Implementation: Streamlit-based dashboard with an intuitive sidebar.

- **User Story 2**: Visualization of "mean" and "standard deviation" images for pizza and non-pizza.

* Implementation: Creation of "mean" and "standard deviation" images.

- **User Story 3**: Visualization of the difference between average pizza and non-pizza images.

* Implementation: Showcasing the disparity between average pizza and non-pizza.

- **User Story 4**: Image montage for visual differentiation.

* Implementation: Development of an image montage feature.

**Please refer to [Hypothesis 1](#hypothesis-1) for more details on why these visualizations are important.**

[Back to top](#table-of-contents)

### Business Requirement 2: Classification
>The second business requirement involves developing a classification system for accurate detection.

- **User Story 5**: Machine Learning model predicting pizza presence with at least 80% accuracy.

* Implementation: Creation of a machine learning model for instant evaluations on food images.

[Back to top](#table-of-contents)

### Business Requirement 3: Report
>The third business requirement centers on generating prediction reports for examined food images.

- **User Story 6**:  ML predictions report generation after each batch of images uploaded.

* Implementation: Integration of a feature for downloadable .csv reports.

[Back to top](#table-of-contents)

## ML Business Case

### PizzaPal's Pizza Detection Revolution
Our mission is to create a state-of-the-art Machine Learning model, achieving an exceptional 80% accuracy on the test set (a goal which we have surpassed). This model not only saves time but ensures precision, consistency, and pure pizza perfection. It empowers users to capture a snapshot of their food creation and receive an instantaneous verdict – pizza or not?

Our dataset, sourced from Kaggle's "Pizza or Not Pizza," comprises 1966 tantalizing food images, forming a treasure trove of culinary artistry. This project isn't just about Machine Learning; it's about enhancing the essence of culinary delight.

<details><summary>See Image</summary>
<img src="SUCCESSFUL PIZZA DETECTION">
</details>

## Dashboard Design (Streamlit App User Interface)

### Page 1: Quick Project Summary
- Quick Project Summary:
    In this section, we provide an overview of the project, its objectives, and the importance of the Pizza-Not-Pizza image classification system. We highlight the business requirements and the dataset used for the project.

<details><summary>See Image</summary>
<img src="STREAMLIT SUMMARY">
</details>

[Back to top](#table-of-contents)

### Pizza Visualizer
This page focuses on visually differentiating pizza images from other types of food. We display the difference between average and variability images for pizza and not-pizza categories. We also present a comparison of average images and offer an image montage for a better visual understanding.

<details><summary>See Image</summary>
<img src="STREAMLIT VISUALIZER">
</details>

- **Average Image Plot**: This plot displays the average image for both pizza and non-pizza categories. It helps the client visually differentiate between the two by showcasing the mean features present in each category. For instance, pizza images tend to show specific characteristics in the mean image, aiding in the identification of pizzas.

- **Standard Deviation Plot**: The standard deviation plot exhibits the variation or noise present in pizza and non-pizza images. Higher variations in the standard deviation indicate diverse toppings or attributes in the images. It visually represents how pizza and non-pizza images differ in terms of features.

<details><summary>See Image</summary>
<img src="STREAMLIT 4x PLOT (again)">
</details>

- **Difference Between Averages**: This plot visually compares the average images of pizza and non-pizza. While there might not be clear-cut patterns to distinguish between the two, the subtle differences in color and shape between the average pizza and non-pizza images are highlighted.

<details><summary>See Image</summary>
<img src="STREAMLIT 3x PLOT (again)">
</details>

- **Image Montage**: The image montage feature creates a collection of images representing both pizza and non-pizza categories. It helps users observe multiple examples of each category, aiding in their ability to differentiate between the two.

<details><summary>See Image</summary>
<img src="IMAGE MONTAGE PIZZA">
</details>
<details><summary>See Image</summary>
<img src="IMAGE MONTAGE NOT-PIZZA">
</details>

[Back to top](#table-of-contents)

### Page 3: Pizza Detection
On this page, users can upload food images to obtain instant predictions about whether they contain pizza or not. We also provide a download link for sample pizza and not-pizza photos.

<details><summary>See Image</summary>
<img src="STREAMLIT DETECTION">
</details>

[Back to top](#table-of-contents)
  
### Page 4: Project Hypothesis and Validation
In this section, we explore our hypothesis about distinguishing pizza and images which contain food that is not pizza, visually. We discuss image montages and various studies conducted during the project.

<details><summary>See Image</summary>
<img src="STREAMLIT HYPOTHESIS">
</details>

- **Prediction Probability Plot**: This plot presents the prediction probabilities as percentages for each class (Pizza and Not-Pizza). It helps users understand the confidence level of the model's predictions. For instance, a higher pizza percentage indicates a stronger likelihood of the image containing pizza.

<details><summary>See Image</summary>
<img src="STREAMLIT PERCENTAGES">
</details>

- **Prediction Result**: The prediction result indicates whether the image is classified as "Pizza" or "Not-Pizza" based on the model's evaluation. The accompanying percentages provide the likelihood of the image belonging to each category.

<details><summary>See Image</summary>
<img src="STREAMLIT PREDICTION">
</details>

[Back to top](#table-of-contents)

### Page 5: ML Performance Metrics
Here, we present metrics related to the project's performance, including the distribution of labels in the training and test sets. We showcase model training history in terms of accuracy and losses and provide general performance metrics on the test set.

<details><summary>See Image</summary>
<img src="STREAMLIT ML PERFORMANCE">
</details>

- **Label Distribution Graph**: This plot illustrates the distribution of labels (Pizza and Not-Pizza) in the train, validation, and test datasets. It shows the frequency of each label in each dataset, helping users understand the dataset's composition.

<details><summary>See Image</summary>
<img src="SUCCESSFUL PIZZA DETECTION">
</details>

- **Loss and Accuracy Plot**: This plot depicts the training progress over five epochs. Notable improvements in accuracy and reductions in loss are observed, indicating the model's ability to learn from the training data. Validation accuracy consistently increases, ensuring the model generalizes well to new data.


<details><summary>See Image</summary>
<img src="STREAMLIT LOSS AND ACCURACY">
</details>

- **Generalized Performance on Test Set**: A summary of the model's performance on the test set. The model achieved a loss of 0.7276 and an accuracy of 0.8579. These metrics provide insights into how well the trained model performs on previously unseen data, validating its effectiveness in real-world scenarios.

<details><summary>See Image</summary>
<img src="STREAMLIT TEST SET RESULTS">
</details>

[pizza-predictor.herokuapp.com](https://pizza-to-be-or-not-to-be.herokuapp.com/)

[Back to top](#table-of-contents)

## The process of Cross-industry standard process for data mining
CRISP-DM, which stands for Cross-Industry Standard Process for Data Mining, is an industry-proven way to guide your data mining efforts.

- As a methodology, it includes descriptions of the typical phases of a project, the tasks involved with each phase, and an explanation of the relationships between these tasks.
- As a process model, CRISP-DM provides an overview of the data mining life cycle.

**Source**: [IBM - crisp overview](https://www.ibm.com/docs/it/spss-modeler/saas?topic=dm-crisp-help-overview)

**This process is documented using the Kanban Board provided by GitHub in this repository project section: [Predict Pizza...or not](https://github.com/KrystalCoding/pizza-not-pizza)**

A kanban board is an agile project management tool designed to help visualize work, limit work-in-progress, and maximize efficiency (or flow). It can help both agile and DevOps teams establish order in their daily work. Kanban boards use cards, columns, and continuous improvement to help technology and service teams commit to the right amount of work, and get it done!

The CRISP-DM process is divided in [sprints](https://www.atlassian.com/agile/scrum/sprints#:~:text=What%20are%20sprints%3F,better%20software%20with%20fewer%20headaches.). Each sprint has Epics based on each CRISP-DM task which were subsequently split into task. Each task can be either in the *To Do*, *In progress*, *Review* status as the workflow proceeds and contains in-depth details.

![Kanban detail](INSERT IMAGE)

[Back to top](#table-of-contents)

## Bugs

### Fixed Bug

While fine-tuning our "Pizza vs. Not Pizza" image classification model, we encountered a critical bug that was hindering the model's performance.

- Description: During the iterative process of adjusting hyperparameters and model architecture, we encountered a persistent bug. The bug manifested as the model's training process stalling with erratic validation accuracy and excessive loss values. Despite our efforts to improve the model, it was clear that something needed to be fixed to make it learn effectively.

- Bug Analysis: After in-depth analysis, we identified several issues contributing to the bug. The batch sizes we used were initially too small and then too large, causing instability during training. Additionally, the model architecture, specifically the number of dense layers, was not optimized for the given task. Lack of dropout layers also led to overfitting. These factors combined to impede the model's training progress.

Fix/Workaround: To address the bug, we implemented several adjustments:

- Adjust Batch Size: We first significantly increased and then mildly decreased the batch size, enhancing the model's stability during training.
- Adjusted Dense Layers: We reconfigured the number and size of dense layers in the model to better suit the complexity of the classification task.
- Added Dropout Layers: To mitigate overfitting, we introduced dropout layers at critical points in the architecture.
- Fine-Tuned Hyperparameters: We carefully fine-tuned other hyperparameters such as learning rate to improve training dynamics.
These fixes collectively resolved the bug and allowed the model to train effectively, ultimately resulting in improved validation accuracy and reduced loss.

This bug fix was a pivotal step in optimizing our "Pizza vs. Not Pizza" image classification model to meet our project's business requirements.
[Back to top](#table-of-contents)

## Unfixed Bug

Images producing false predictions

![pizza/not pizza](INSERT IMAGE)

- ##  
     - __Description__ : The above image, despite looking like pizza/not being pizza was predicted incorrectly. 
     - __Bug__: The problem problem problem. The background being the same colour could be misleading as the model is not able to clearly detect the pizza shape. 
     - __Fix/Workaround__: The model needs further tuning.

     [Back to top](#table-of-contents)

## Deployment
The project is coded and hosted on GitHub and deployed with [Heroku](https://www.heroku.com/). 

### Creating the Heroku app 
The steps needed to deploy this projects are as follows:

1. Create a `requirement.txt` file in GitHub, for Heroku to read, listing the dependencies the program needs in order to run.
2. Set the `runtime.txt` Python version to a Heroku-20 stack currently supported version.
3. `push` the recent changes to GitHub and go to your [Heroku account page](https://id.heroku.com/login) to create and deploy the app running the project. 
3. Chose "CREATE NEW APP", give it a unique name, and select a geographical region. 
4. Add  `heroku/python` buildpack from the _Settings_ tab.
5. From the _Deploy_ tab, chose GitHub as deployment method, connect to GitHub and select the project's repository. 
6. Select the branch you want to deploy, then click Deploy Branch.
7. Click to "Enable Automatic Deploys " or chose to "Deploy Branch" from the _Manual Deploy_ section. 
8. Wait for the logs to run while the dependencies are installed and the app is being built.
9. The mock terminal is then ready and accessible from a link similar to `https://your-projects-name.herokuapp.com/`
10. If the slug size is too large then add large files not required for the app to the `.slugignore` file.

[Back to top](#table-of-contents)
   
### Forking the Repository

- By forking this GitHub Repository you make a copy of the original repository on our GitHub account to view and/or make changes without affecting the original repository. The steps to fork the repository are as follows:
    - Locate the [GitHub Repository](https://github.com/KrystalCoding/pizza-not-pizza) of this project and log into your GitHub account. 
    - Click on the "Fork" button, on the top right of the page, just above the "Settings". 
    - Decide where to fork the repository (your account for instance)
    - You now have a copy of the original repository in your GitHub account.

[Back to top](#table-of-contents)

### Making a local clone

- Cloning a repository pulls down a full copy of all the repository data that GitHub.com has at that point in time, including all versions of every file and folder for the project. The steps to clone a repository are as follows:
    - Locate the [GitHub Repository](https://github.com/KrystalCoding/pizza-not-pizza) of this project and log into your GitHub account. 
    - Click on the "Code" button, on the top right of your page.
    - Chose one of the available options: Clone with HTTPS, Open with Git Hub desktop, Download ZIP. 
    - To clone the repository using HTTPS, under "Clone with HTTPS", copy the link.
    - Open Git Bash. [How to download and install](https://phoenixnap.com/kb/how-to-install-git-windows).
    - Chose the location where you want the repository to be created. 
    - Type:
    ```
    $ git clone https://git.heroku.com/pizza-to-be-or-not-to-be.git
    ```
    - Press Enter, and wait for the repository to be created.
    - Click [Here](https://help.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository#cloning-a-repository-to-github-desktop) for a more detailed explanation. 

__You can find the live link to the site here: [Pizza: To Be or Not To Be]()__

[Back to top](#table-of-contents)

## Technologies used

### Platforms
- [Heroku](https://en.wikipedia.org/wiki/Heroku) To deploy this project
- [Jupiter Notebook](https://jupyter.org/) to edit code for this project
- [Kaggle](https://www.kaggle.com/) to download datasets for this project
- [GitHub](https://github.com/) to store the project code after being pushed from Gitpod.
- [Gitpod](https://www.gitpod.io/) Dashboard was used to write the code and its terminal to 'commit' to GitHub and 'push' to GitHub Pages.
- [Codeanywhere](https://app.codeanywhere.com/) is the crossplatform cloud IDE used to run Jupyter notebooks and host until pushed to GitHub.

### Languages
- [Python](https://www.python.org/)
- [Markdown](https://en.wikipedia.org/wiki/Markdown)
  
### Main Data Analysis and Machine Learning Libraries
<pre>
- tensorflow-cpu 2.6.0  used for creating the model
- numpy 1.19.2          used for converting to array 
- scikit-learn 0.24.2   used for evaluating the model
- streamlit 0.85.0      used for creating the dashboard
- pandas 1.1.2          used for creating/saving as dataframe
- matplotlib 3.3.1      used for plotting the set's distribution
- keras 2.6.0           used for setting model's hyperparamters
- plotly 5.12.0         used for plotting the model's learning curve 
- seaborn 0.11.0        used for plotting the model's confusion matrix
</pre>

[Back to top](#table-of-contents)

## Credits

### Content
- The pizza-not-pizza dataset was linked from [Kaggle](https://www.kaggle.com/code/rasikagurav/pizza-or-not-pizza), created by [Rasika Gurav](https://www.kaggle.com/rasikagurav)

- The [CRISP DM](https://www.datascience-pm.com/crisp-dm-2/) steps adopted from [Introduction to CRISP-DM](https://www.ibm.com/docs/en/spss-modeler/saas?topic=guide-introduction-crisp-dm) articles from IBM.

### Media
- The banner image is from [](), the lettering colour is []()

### Code

-  The template used for this project belongs to CodeInstitute - [GitHub](https://github.com/Code-Institute-Submissions) and [here is their website](https://codeinstitute.net/global/).
- App pages for the Streamlit dashboard, data collection and data visualization jupiter notebooks are from [Code Institute Walthrough Project](https://github.com/Code-Institute-Solutions/WalkthroughProject01) and where used as a backbone for this project.

### Formatting

- Some of the graphs and  Dashboard format were inspired by this [GitHub repository](https://github.com/cla-cif/Cherry-Powdery-Mildew-Detectorh) by fellow Code Institute student.
- GitHub README isnpired formatting by [ocassidydev](ttps://github.com/ocassidydev/mushroom-safety).

### Acknowledgements

Thanks to [Code Institute](https://codeinstitute.net/global/) and my one-off session mentor Mo Shami. 

### Deployed version at [cherry-powdery-mildew-detector.herokuapp.com]()

[Back to top](#table-of-contents)
