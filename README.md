
**TRANSFER-LEARNING-AND-KNOWLEDGE-DISTILLATION-for-CHEST-X-RAY-CLASSIFICATION**


*Project Overview*


Features
Transfer Learning: Fine-tuning of state-of-the-art architecture  (ResNet-50) on a chest X-ray dataset.

Knowledge Distillation: Implementation of a distillation pipeline to compress a large teacher model into a compact student model.

Comprehensive Evaluation: Detailed comparison of the performance of the baseline, teacher, and student models using relevant medical diagnosis metrics (e.g., AUC-ROC, F1-Score, Precision, and Recall).

Baseline Model: A simple, custom-built Convolutional Neural Network (CNN) is included to serve as a benchmark for comparison.

Dataset
This project utilizes the "Chest X-Ray Images (Pneumonia)" dataset, which is available on Kaggle. It contains images labeled as either Normal or Pneumonia. The dataset exhibits a class imbalance, which is a realistic challenge in medical datasets.


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



**Methodology**
---
***Dataset Preparation:***

Conducting a thorough examination of the dataset to understand the distribution of spam and non-spam emails.
Understanding the dataset and checking for null data if there exits then,
The dataset is split into training and testing sets using a 75-25 ratio to ensure a balanced evaluation of the model.
Stop words like "is," "am," "are," and "in" are removed as they do not contribute significantly to the classification task.
From the 3000 columns in the dataset, removing stop words reduced the features to 2867 columns.
For the final classifier, 228 columns with the highest frequency in the dataset were used to improve the model's performance.

***Training:***

The Multinomial Naive Bayes classifier is trained using the word frequency from the training set.
The model learns the probability distributions of words in spam and non-spam emails, enabling it to make predictions on unseen data.

**_Evaluation:_**

After training, the classifier is tested on the remaining 25% of the dataset, and its performance is evaluated using: metrics such as f1-score, accuracy, precision, and recall.

_This shows the result:_

               precision    recall  f1-score  

           0       0.95      0.90      0.92      
           1       0.76      0.88      0.82 
     accuracy                           0.89

**Dependencies**
---
*This project uses the following libraries and tools:*

**Python 3.12.2**: The core programming language used for implementing the classifier and handling data.

**NumPy**: Essential for numerical operations, especially in vectorizing text data.

**Pandas**: Used for loading and manipulating the dataset.

**Scikit-learn**: implementing the Multinomial Naive Bayes classifier, and evaluating the model through metrics like accuracy,f1-score, precision, and recall.

**NLTK**: For natural language processing tasks such as tokenization and stop word removal.

**Email Spam Collection Dataset**: The dataset contains 5172 emails, of which 1527 are spam and the remaining are non-spam.




