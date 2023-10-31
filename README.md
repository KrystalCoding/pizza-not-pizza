![Responsive Design](INSERT IMAGE)

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

### Deployed version at [pizza-predictor.herokuapp.com](https://pizza-to-be-or-not-to-be.herokuapp.com/)

## Dataset Content

The dataset contains 983 featured photos of pizza variations, as well as 983 photos of food that is not pizza. This client is particularly concerned about identifying pizza as it is their flagship product. The dataset is sourced from [Kaggle](https://www.kaggle.com/code/rasikagurav/pizza-or-not-pizza/input).

## Business Requirements

In today's dynamic food and beverage industry, rapid decision-making and quality control are paramount. That's why our team embarked on a mission to assist an innovative pizzeria, "PizzaPal," in revolutionizing its quality assurance process. The client, PizzaPal, sought our expertise to develop an advanced Machine Learning-based system capable of instantaneously detecting the presence of pizza in images.

PizzaPal is renowned for its diverse menu of delectable pizzas, each meticulously crafted to culinary perfection. Ensuring that every pizza consistently meets its high standards is central to PizzaPal's brand reputation. Manual inspection of thousands of pizza images to confirm quality and adherence to standards is not only time-consuming but also susceptible to human error.

To address this challenge, our solution is designed to automate the detection process, enabling PizzaPal to expedite quality assessments, reduce labor costs, and elevate customer satisfaction. By instantly confirming the presence of pizza in images, PizzaPal gains a competitive edge in maintaining its culinary excellence.

Our system is tailored to PizzaPal's specific needs, enabling seamless integration into their existing workflow. It provides clear, accurate, and near-instant results, allowing PizzaPal's quality control team to focus their expertise on the finer aspects of pizza perfection.

In this school project, we take inspiration from the real-world need faced by PizzaPal, an imaginary but forward-thinking business customer. Our objective is to demonstrate how Machine Learning can empower the food and beverage industry by automating image classification, enhancing quality control, and reducing operational inefficiencies.

Join us on this journey as we explore the capabilities of Machine Learning to transform the way businesses, like our visionary client PizzaPal, maintain their commitment to excellence in the world of food and beverage. Discover how cutting-edge technology can optimize operations, improve product quality, and drive success in a competitive industry.

Business Requirements for Pizza vs. Not-Pizza Image Classification System:

1. Automated Pizza Detection: The system must automate the process of classifying images as either containing pizza or not-pizza. This automation should significantly reduce the time and effort required for manual image inspection.

2. Prediction Accuracy: The client requires a reliable system capable of achieving high accuracy in classifying pizza and not-pizza images. The minimum acceptable accuracy should be defined based on the specific needs of the client.

3. Scalability: The system should be scalable to handle a large number of images, reflecting the client's potential expansion of operations. It should efficiently process and classify images in real-time.

4. User-Friendly Interface: The client expects an easy-to-use interface for uploading images and receiving classification results. The system should be intuitive for users with minimal technical expertise.

5. Prediction Reporting: The system should provide prediction reports for each examined image, indicating the classification (pizza or not-pizza) and the associated confidence level or probability.

6. Fast Processing: The client requires a system capable of processing images quickly and providing near-instantaneous results. This speed is essential for streamlining decision-making processes.

7. Continuous Improvement: The system should support continuous improvement and model retraining to adapt to changes in image data patterns and to maintain high prediction accuracy.

## Hypothesis and validation

1. **Hypothesis**: Pizza presence can be accurately identified by analyzing the shape and toppings within images.
   - Validation Plan: To validate this hypothesis, we will gather a dataset of pizza images with various shapes and toppings. We will conduct a detailed analysis, including feature extraction and shape recognition. The validation process will involve developing a feature extraction pipeline to capture key pizza characteristics, such as circular shape and the presence of toppings. We will then build an average image study to examine common patterns in pizza images, emphasizing shape and topping distribution.

2. **Hypothesis**: Exploring different model configurations will lead to improved accuracy in pizza detection.
   - To validate this hypothesis, we will perform a comprehensive exploration of various model configurations. We will experiment with different network architectures, layers, activation functions, and hyperparameters. For each configuration, we will conduct extensive training and evaluation, keeping the dataset and other factors consistent. We will compare the accuracy and performance metrics of each model to determine which configurations lead to improved pizza detection accuracy.

3. **Hypothesis**: Converting `RGB` images to `grayscale` improves image classification performance.  
   - __How to validate__: Understand how colours are represented in tensors. Train and compare identical models changing only the image color.

### Hypothesis 1
> Pizza presence can be accurately identified by analyzing the shape and toppings within images.

**1. Introduction**

We hypothesize that pizzas exhibit distinctive characteristics that can be leveraged for accurate identification. One of the primary identifiers is the circular, flat shape of pizzas, typically accompanied by a variety of toppings encapsulated within the circular mass. To harness this inherent property in the context of machine learning, we need to preprocess the images to ensure optimal feature extraction and model training.

When we are dealing with an Image dataset, it's important to normalize the images in the dataset before training a Neural Network on it. This is required because of the following two core reasons:
- It helps the trained Neural Network give consistent results for new test images.
- Helps in Transfer Learning
To normalize an image, one will need the mean and standard deviation of the entire dataset.

To calculate the **mean** and **standard deviation**, the mathematical formula takes into consideration four dimensions of an image (B, C, H, W) where:
- B is batch size that is number of images
- C is the number of channels in the image which will be 3 for RGB images.
- H is the height of each image
- W is the width of each image
Mean and std is calculated separately for each channel. The challenge is that we cannot load the entire dataset into memory to calculate these paramters. We can load a small set of images (batches) one by one and this can make the computation of mean and std non-trivial.

[Back to top](#table-of-contents)

**2. Observation**

To validate our hypothesis, we observed the following key characteristics:

- Shape Analysis: Pizza images consistently display a circular and flat shape. This distinct feature can serve as a crucial discriminator in identifying pizzas.

- Toppings Variation: The toppings on pizzas vary widely, providing additional cues for detection. These toppings, such as pepperoni, vegetables, or cheese, introduce unique textural and color patterns that can be learned by our model.

**3. Image Analysis**

To gain deeper insights, we performed image analysis:

![montage-pizza](INSERT IMAGE)
![montage-not-pizza](INSERT IMAGE)

- Shape Comparison: A montage of pizza images clearly illustrates the uniform circular shape found in pizzas. In contrast, we created a montage of "not-pizza" images, which showcase diverse and irregular shapes. This striking difference in shape serves as a foundation for differentiation.

- Toppings Diversity: Analyzing the average and variability in images, we noticed that pizzas tend to exhibit a more centered and circular pattern. In contrast, "not-pizza" images display a wider array of shapes and patterns, emphasizing the uniqueness of pizza toppings.

![average variability between samples](INSERT IMAGE)

- Averaging Images: Comparing the average pizza image to the average "not-pizza" image did not reveal any immediate and intuitive difference. This suggests that pizza detection relies on a combination of subtle features, including shape and toppings.

![average variability between samples](INSERT IMAGE)

[Back to top](#table-of-contents)

**3. Conclusion**

Our model demonstrated its capacity to detect these subtle yet distinguishing features, enabling it to make accurate predictions. An effective model goes beyond memorizing training data but generalizes the essential patterns that connect image features to labels. This generalization allows the model to confidently predict pizza presence in future observations, contributing to the automation of pizza detection in our project.

**Sources**:

- [Calculate mean and std of Image Dataset](https://iq.opengenus.org/calculate-mean-and-std-of-image-dataset/)
- [Computing Mean & STD in Image Dataset](https://kozodoi.me/python/deep%20learning/pytorch/tutorial/2021/03/08/image-mean-std.html)

[Back to top](#table-of-contents)

---
### Hypothesis 2
> Exploring different model configurations will lead to improved accuracy in pizza detection.

**1. Introduction**

Understanding the Classification Problem:

In our pizza-not-pizza project, we face a classification problem. We aim to classify images into one of two categories: pizza or not-pizza. This **binary classification** requires us to choose an appropriate activation function for the output layer of our Convolutional Neural Network (CNN).

- **Epoch**: An epoch signifies one complete pass through the training dataset.
- **Loss**: It quantifies how bad the model's prediction is. A lower loss value indicates a better prediction.
- **Accuracy**: Accuracy is the proportion of correct predictions made by the model.

In our learning curve plots, we look for the right fit of the learning algorithm, avoiding both overfitting and underfitting. A good fit is characterized by the following:

- The training loss decreases (or accuracy increases) to a point of stability.
- The validation loss decreases (or accuracy increases) to a point of stability with a small gap compared to the training loss.
- Continued training of a well-fitted model may lead to overfitting. This is why ML models usually have an [early stopping](https://en.wikipedia.org/wiki/Early_stopping) function utilized which interrupts the model's learning phase when it ceasing improving.

**2. Observation**

Our experimentation in the pizza-not-pizza project involved various model configurations and hyperparameter adjustments. We initiated the process with a custom model that featured three convolutional layers, max-pooling, and dense layers. This model was trained with a batch size of 20 for 25 epochs. However, we observed that the custom model did not achieve the desired accuracy, and the loss did not decrease significantly during training. It struggled to capture the intricate features that distinguish pizza from non-pizza images.

As an alternative, we explored the pre-trained VGG16 model. By fine-tuning the top layers to adapt to our binary classification task of pizza detection, we achieved better results. With a batch size of 35 and training for only 5 epochs, this VGG16-based model displayed improved accuracy. It successfully captured nuanced patterns and features critical for distinguishing between pizza and not-pizza images. Moreover, the loss function showed consistent decreases, indicating better convergence.

Encouraged by this initial progress, we further refined our VGG16-based model. We reduced the batch size to 15 and incorporated additional layers, including dense layers, L2 regularization, and dropout layers. These modifications led to significant improvements in loss. However, we continued to grapple with accuracy.

In summary, our experimentation revealed that the VGG16-based model, with fine-tuned top layers and additional modifications, exhibited potential in distinguishing pizza from not-pizza images. Despite these advancements, achieving high accuracy remained a challenge. Our primary focus in this experiment was to evaluate different model architectures and hyperparameters with the aim of enhancing classification performance for our specific problem.

- Loss/Accuracy of our custom model:

   ![custom_model](INSERT IMAGE)

- Loss/Accuracy of original VGG16 model:

   ![vgg16_initial_model](INSERT IMAGE)

- Loss/Accuracy of enhanced VGG16 model:

   ![vgg16_enhanced_model](INSERT IMAGE)

**3. Conclusion**

In our pizza-not-pizza project, we observed that the pre-trained VGG16 model, with fine-tuned top layers, showed promise in distinguishing pizza from not-pizza images. However, achieving high accuracy remained a challenge despite various enhancements to the model. The primary focus of our experiment was to evaluate different model architectures and hyperparameters to improve classification performance for our specific problem. As a result, our conclusions are based on the differences between our custom model and the VGG16-based models. Further refinements and investigations are needed to enhance accuracy and improve the model's performance.

- Loss/Accuracy of our custom model:

   ![custom_model](INSERT IMAGE)

- Loss/Accuracy of original VGG16 model:

   ![vgg16_initial_model](INSERT IMAGE)

- Loss/Accuracy of enhanced VGG16 model:

   ![vgg16_enhanced_model](INSERT IMAGE)

**Sources**:
- [Backpropagation in Fully Convolutional Networks](https://towardsdatascience.com/backpropagation-in-fully-convolutional-networks-fcns-1a13b75fb56a#:~:text=Backpropagation%20is%20one%20of%20the,respond%20properly%20to%20future%20urges.) by [Giuseppe Pio Cannata](https://cannydatascience.medium.com/)
- [How to use Learning Curves to Diagnose Machine Learning Model Performance](https://machinelearningmastery.com/learning-curves-for-diagnosing-machine-learning-model-performance/) by [Jason Brownlee](https://machinelearningmastery.com/about)
- [Activation Functions: Comparison of Trends in Practice and Research for Deep Learning](https://arxiv.org/pdf/1811.03378.pdf) by *Chigozie Enyinna Nwankpa, Winifred Ijomah, Anthony Gachagan, and Stephen Marshall*

[Back to top](#table-of-contents)

---
### Hypothesis 3 
> The Addition of Dropout Layers Improves Pizza Detection Model Performance

**1. Introduction**

Dropout layers are a popular regularization technique used in deep learning to prevent overfitting. When dropout is applied, it randomly sets a fraction of input units to 0 at each update during training, which helps to prevent complex co-adaptations on training data and, in turn, enhances model generalization.

For our pizza detection project, we hypothesize that incorporating dropout layers into the model architecture will improve its overall performance. We aim to explore whether the addition of dropout layers helps to reduce overfitting and enhance the model's ability to accurately classify pizza and non-pizza images.

**2. Observation**

To validate this hypothesis, we experimented with two versions of the model: one with dropout layers and one without. Both models shared the same architecture and hyperparameters.

Upon comparing the results, we observed that the model with dropout layers exhibited improved performance. It displayed lower signs of overfitting and better generalization on the test dataset. The dropout layers introduced an element of randomness during training, preventing the model from relying too heavily on specific features and patterns, and thereby enhancing its ability to make accurate predictions.

Performance comparison between models with and without dropout layers:
Model with dropout layers: ![model_with_dropout](INSERT IMAGE)
Model without dropout layers: ![model_without_dropout](INSERT IMAGE)

**3. Conclusion**

Our hypothesis that the addition of dropout layers improves pizza detection model performance has been validated through our experiments. The inclusion of dropout layers contributed to reducing overfitting and enhancing the model's ability to generalize better on unseen data. This finding underscores the importance of regularization techniques, such as dropout, in improving the accuracy and reliability of pizza detection models. Further exploration of dropout hyperparameters and variations in dropout strategies can lead to even more significant performance improvements.

[Back to top](#table-of-contents)

## The rationale for the model

The VGG16 model is a convolutional neural network with 13 convolutional layers and 3 fully connected layers. It uses a predefined architecture with multiple convolutional and pooling layers, followed by three fully connected layers and an output layer for classification.

### The goal

The process of developing this model involved carefully configuring hyperparameters, determining the optimal number of hidden layers and nodes, and selecting an appropriate optimizer. Achieving the desired model architecture was a result of systematic trial and error.
It's important to note that while the model we've arrived at may not be the absolute best, it represents the outcome of extensive testing and fine-tuning in alignment with our project goals.

Our primary aim was to develop a robust model that excels in its ability to predict classes from a batch of data while maintaining a high level of generalization. We avoided overfitting, ensuring that the model doesn't merely memorize the training data but learns the underlying patterns that connect features to labels.

Furthermore, we sought to maintain computational efficiency by striking a balance between neural network complexity and the number of trainable parameters. This optimization allowed us to achieve a model that can generalize effectively, maintain high accuracy, and minimize error, all while conserving computational resources.

[Back to top](#table-of-contents)

### Configuring Model Hyperparameters

- **Convolutional Layer Size**: n our pizza detection project, we use 2D convolutional layers (Conv2D) because our dataset consists of 2D images. 1D convolutional layers are not suitable for our project, as they are designed for 1D data like time series.

- **Convolutional Kernel Size**: We choose a 3x3 convolutional filter because it effectively processes our 2D image data. This kernel size works well for our images and allows for zero padding, maintaining the size of our images.

- **Number of Neurons**: We select the number of neurons in our layers as powers of 2 to optimize computational efficiency. This choice aligns with the GPU's ability to leverage optimizations related to power-of-two dimensions.

- **Activation Function**: The `ReLu`(Rectified Linear Unit) activation function is used in our model. `ReLu` is preferred because it is computationally efficient and empirically proven to work well in training deep neural networks. Its derivative is either 0 or 1, helping mitigate the vanishing gradient problem.

- **Pooling**: We use max pooling to reduce variance and computational complexity in our pizza detection model. This choice is appropriate for our project, as it selects the brighter pixels in the image. Max pooling works well when we aim to identify pizza against a relatively darker background.

- **Output Activation Function**: For **binary classification** of pizza and not-pizza, our model employs the `sigmoid` activation function. This choice is ideal for binary classification tasks, where we aim to predict one of two classes, pizza or not-pizza. The `sigmoid` function outputs probabilities within the range of 0 to 1, making it well-suited for this purpose.

- **Dropout**:  The model incorporates a dropout rate of 0.5. Dropout is essential in our project to prevent overfitting, particularly given the relatively limited number of training samples.

**Source**: 
- [How to choose the size of the convolution filter or Kernel size for CNN?](https://medium.com/analytics-vidhya/how-to-choose-the-size-of-the-convolution-filter-or-kernel-size-for-cnn-86a55a1e2d15) by - [Swarnima Pandey](https://medium.com/@pandeyswarnima)
- [The advantages of ReLu](https://stats.stackexchange.com/questions/126238/what-are-the-advantages-of-relu-over-sigmoid-function-in-deep-neural-networks#:~:text=The%20main%20reason%20why%20ReLu,deep%20network%20with%20sigmoid%20activation.)
- [Maxpooling vs minpooling vs average pooling](https://medium.com/@bdhuma/which-pooling-method-is-better-maxpooling-vs-minpooling-vs-average-pooling-95fb03f45a9#:~:text=Average%20pooling%20method%20smooths%20out,lighter%20pixels%20of%20the%20image.) by - [Madhushree Basavarajaiah](https://medium.com/@bdhuma)
- [How ReLU and Dropout Layers Work in CNNs](https://www.baeldung.com/cs/ml-relu-dropout-layers)

[Back to top](#table-of-contents)

### Hidden Layers

Hidden layers are crucial components of neural networks, responsible for feature extraction and classification based on those features. In our pizza detection project, these layers play a vital role in learning and distinguishing the essential characteristics that define a pizza image.

When it comes to designing the hidden layers, two key decisions need to be made:

1. Number of Hidden Layers: Determining how many hidden layers to include in the neural network is essential. Too few hidden layers might lead to underfitting, where the network can't capture the complex patterns present in the dataset. However, using too many hidden layers can introduce overfitting issues, causing the model to perform well on the training data but poorly on new, unseen images.

2. Number of Neurons in Each Layer: The choice of the number of neurons in each hidden layer is a critical factor in the network's performance. For our pizza detection task, we should aim to strike a balance. We want to have enough neurons to learn the intricate features that distinguish pizza images while avoiding an excessive number of neurons that might lead to overfitting.

In our project, the network's design should prioritize the ability to generalize, ensuring that the model performs well on both the training data and new, unobserved images. This can be achieved by keeping the number of nodes in the hidden layers as low as possible, while still effectively capturing the unique attributes of pizza and not-pizza images.

[Back to top](#table-of-contents)

- **Convolutional Layers vs. Fully Connected Layers**:
  - **Convolutional Layers**: In our pizza detection model, Convolutional Layers serve as the backbone for feature extraction. These layers are specifically designed for analyzing images and extracting relevant features. They achieve this by using a technique known as convolution, which allows them to share parameters, significantly reducing the number of parameters compared to Fully Connected Layers. Convolutional Layers are essential for capturing intricate patterns and details in pizza images.
  - **Fully Connected Layers**: While Convolutional Layers are ideal for feature extraction, Fully Connected Layers, also known as Dense Layers, are primarily used for making final classifications in certain neural network architectures. In our pizza detection model, we've structured these layers to perform the ultimate classification task, distinguishing between pizza and not-pizza images. These layers utilize a linear operation that considers every input, making them suitable for our classification goal.

**Source**: 
- [Dense Layer vs convolutional layer](https://datascience.stackexchange.com/questions/85582/dense-layer-vs-convolutional-layer-when-to-use-them-and-how#:~:text=As%20known%2C%20the%20main%20difference,function%20based%20on%20every%20input.)

[Back to top](#table-of-contents)


### Model Compilation

- **Loss**: The loss function is a crucial component that measures the disparity between the predicted and actual output values, reflecting how effectively the neural network models the training data. In our pizza detection project, we employed `binary_crossentropy` as the loss function. This choice aligns with our binary classification task of distinguishing between pizza and not-pizza images. (See [Hypothesis 2](#Hypothesis-2) for more details.)

- **Optimizer**: The optimizer plays a vital role in adjusting neural network attributes, such as weights and learning rates, to expedite convergence while minimizing loss and maximizing accuracy. In our project, we opted for the `adam` optimizer after thorough experimentation and the trial-and-error phase. `Adam` optimization has proven to be effective in various machine learning tasks.

- **Metrics**: The selected metric for assessing model performance is `accuracy`. It quantifies how frequently the model's predictions match the actual labels in our binary classification problem. This metric keeps track of two local variables, total and count, to determine the `accuracy` of the predictions.  

**Source**: 
- [7 tips to choose the best optimizer](https://towardsdatascience.com/7-tips-to-choose-the-best-optimizer-47bb9c1219e) by [Davide Giordano](https://medium.com/@davidegiordano)
- [Impact of Optimizers in Image Classifiers](https://towardsai.net/p/l/impact-of-optimizers-in-image-classifiers)
- [Keras Accuracy Metrics](https://keras.io/api/metrics/accuracy_metrics/#:~:text=metrics.,with%20which%20y_pred%20matches%20y_true%20.)

[Back to top](#table-of-contents)

## The Rationale for Mapping Business Requirements to Data Visualizations and ML Tasks

Our project is driven by a set of defined business requirements aimed at assisting an innovative pizzeria, "PizzaPal," in enhancing its quality assurance process through Machine Learning. The client, PizzaPal, required an advanced system capable of swiftly detecting the presence of pizza in images. This system's development is rooted in several key business requirements.

[Back to top](#table-of-contents)

### Business Requirement 1: Data Visualization 
>The first business requirement involves creating data visualizations to distinguish pizzas from non-pizzas. To address this requirement, we derived specific user stories, and each user story corresponds to a machine learning task. All these tasks have been manually tested and validated for functionality. They are as follows

- **User Story 1**: As a client, I want to easily navigate through an interactive dashboard to visualize and comprehend the presented data.
    - This user story led to the development of a Streamlit-based dashboard with an intuitive navigation sidebar.

- **User Story 2**: As a client, I want to view and compare the "mean" and "standard deviation" images for pizza and non-pizza, helping me visually differentiate the two.
    - This resulted in the creation of the "mean" and "standard deviation" images for both pizza and non-pizza.

- **User Story 3**: As a client, I want to visualize the difference between an average pizza and non-pizza, facilitating visual differentiation.
    - This was addressed by showcasing the disparity between an average pizza and non-pizza.

- **User Story 4**: As a client, I want to view an image montage representing pizzas and non-pizzas, aiding visual differentiation.
    - This led to the development of an image montage feature for both pizzas and non-pizzas.

**Please refer to [Hypothesis 1](#hypothesis-1)for more details on why these visualizations are important.**

[Back to top](#table-of-contents)

### Business Requirement 2: Classification
>The second business requirement revolves around developing a classification system that can accurately determine whether a given image contains pizza or not. To fulfill this requirement, we derived the following user story:

- **User Story 5**: As a client, I want a Machine Learning model to predict with an accuracy of at least 80% whether a given image contains pizza or another food group.
    - This user story led to the creation of a machine learning model that is capable of achieving the specified prediction accuracy. Users can upload food images to the dashboard, and the model provides instant evaluations.

[Back to top](#table-of-contents)

### Business Requirement 3: Report
>The third business requirement involves generating prediction reports for examined food images. This requirement aligns with the following user story:

- **User Story 6**: As a client, I want to obtain a report from the Machine Learning predictions on new images.
    - As a result, we incorporated a feature into the Streamlit dashboard to produce downloadable .csv reports after each batch of images is uploaded, offering a comprehensive overview of the prediction results.

[Back to top](#table-of-contents)

## ML Business Case

### Pizza Classifier
Our primary Machine Learning objective is to develop a model capable of distinguishing between images containing pizza and those that do not. This classification problem is categorized as supervised learning, involving a binary classification model. The success metrics for our model include achieving an accuracy of 85% or higher on the test set.

In practical terms, this model will enable users to take a picture of a food item, and upon uploading it to the application, the model will provide an instant prediction regarding whether the image contains pizza or not. This approach significantly accelerates the assessment process, replacing manual inspections.

The model's training data is drawn from the "Pizza or Not Pizza" dataset available on Kaggle, consisting of 1966 food images. By leveraging Machine Learning, we aim to offer a faster and more reliable pizza detection system that aligns with PizzaPal's quality control objectives, reduces operational inefficiencies, and enhances customer satisfaction.
![pizza_detector](INSERT IMAGE)

## Dashboard Design (Streamlit App User Interface)

Our project dashboard consists of several pages that provide insights into the Pizza-Not-Pizza image classification system. Let's explore each page:

### Page 1: Quick Project Summary
- Quick Project Summary:
    In this section, we provide an overview of the project, its objectives, and the importance of the Pizza-Not-Pizza image classification system. We highlight the business requirements and the dataset used for the project.

![Page1](INSERT IMAGE)

[Back to top](#table-of-contents)

### Pizza Visualizer
This page focuses on visually differentiating pizza images from other types of food. We display the difference between average and variability images for pizza and not-pizza categories. We also present a comparison of average images and offer an image montage for a better visual understanding.

![Page2](INSERT IMAGE)

[Back to top](#table-of-contents)

### Page 3: Pizza Detection
On this page, users can upload food images to obtain instant predictions about whether they contain pizza or not. We also provide a download link for sample pizza and not-pizza photos.

![Page3](INSERT IMAGE)

[Back to top](#table-of-contents)
  
### Page 4: Project Hypothesis and Validation
In this section, we explore our hypothesis about distinguishing parasitized and uninfected cells visually. We discuss image montages and various studies conducted during the project.

![Page4](INSERT IMAGE)

[Back to top](#table-of-contents)

### Page 5: ML Performance Metrics
Here, we present metrics related to the project's performance, including the distribution of labels in the training and test sets. We showcase model training history in terms of accuracy and losses and provide general performance metrics on the test set.

Our dashboard offers a comprehensive view of the Pizza-Not-Pizza image classification system and its applications in the food and beverage industry. Feel free to explore each page to gain deeper insights into this innovative project. [pizza-predictor.herokuapp.com](https://pizza-to-be-or-not-to-be.herokuapp.com/)

![Page5](INSERT IMAGE)

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
