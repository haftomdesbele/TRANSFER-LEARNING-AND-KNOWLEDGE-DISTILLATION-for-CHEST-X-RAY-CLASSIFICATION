
**Transfer Learning and Knowledge Distillation for Chest X-ray Classification**
---

**Overview**
---
This project uses two powerful and efficient deep learning approaches for chest X-ray classification. A combination of Transfer Learning and Knowledge Distillation to create a high-performing and lightweight model. This makes the approach particularly suitable for deployment in low-resource healthcare settings, where computational efficiency and reliability are crucial.


The project implements  NLP tasks  such as tokenization, lemmatization, stop word removal, the Multinomial Naive Bayes algorithm on the Bag of Words model, and model evaluation metrics.

**Introduction**
---
Deep Learning techniques like CNN come with 2 problems: 

Data Scarcity: Training a deep learning model from scratch on limited labeled medical data can lead to poor performance and overfitting.

Deployment Constraints: Large, high-performance models are often too slow or resource-intensive for deployment on edge devices or in clinical settings with limited computational power.

Our solution addresses these challenges by employing a two-stage approach:

Transfer Learning: We fine-tune a pre-trained, large-scale model (the "teacher") to achieve high accuracy on the classification task. This leverages the extensive knowledge the model has already learned from a massive, general-purpose dataset.

Knowledge Distillation: We then use the knowledge from this large teacher model to train a smaller, more efficient "student" model. The student model learns to mimic the teacher's nuanced predictions, resulting in a lightweight model that retains a high degree of performance, making it suitable for deployment.

**Features**
---
Transfer Learning: Fine-tuning of state-of-the-art architecture  (ResNet-50) on a chest X-ray dataset.

Knowledge Distillation: Implementation of a distillation pipeline to compress a large teacher model into a compact student model.

Comprehensive Evaluation: Detailed comparison of the performance of the baseline, teacher, and student models using relevant medical diagnosis metrics (e.g., AUC-ROC, F1-Score, Precision, and Recall).

Baseline Model: A simple, custom-built Convolutional Neural Network (CNN) is included to serve as a benchmark for comparison.


**Dataset**
---
This project utilizes the "Chest X-Ray Images (Pneumonia)" dataset, which is available on Kaggle. It contains images labeled as either Normal or Pneumonia.

**Dependencies**
---
*This project uses the following libraries and tools:*

**Python 3.12.2**: The core programming language used for implementing the classifier and handling data.

**NumPy**: Essential for numerical operations, especially in vectorizing text data.

**Scikit-learn**: implementing the Multinomial Naive Bayes classifier, and .

**torch**: A core deep learning framework used for building and training neural networks.

**torch.nn**: The neural networks module for creating network layers and functions.

**torch.optim**: The optimization module for implementing various optimization algorithms.

**torch.optim.lr_scheduler**: Used to adjust the learning rate during training.

**torchvision**: A package that provides access to popular datasets, model architectures, and image transformations for computer vision.

**matplotlib.pyplot**: A plotting library used for creating visualizations, such as the confusion matrix and accuracy/loss plots.

**sklearn.metrics**: A module from scikit-learn for calculating metrics like the confusion matrix, ROC curve, and AUC score.Evaluating the models through metrics like accuracy,f1-score, precision, and recall.

**os**: A standard Python library for interacting with the operating system, used for file path operations.

**time**: A standard Python library for time-related functions, used to track training time.

**google.colab**: A utility used to mount Google Drive for accessing data and saving models.








