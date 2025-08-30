
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
This project utilizes the "Chest X-Ray Images (Pneumonia)" dataset, which is available on Kaggle. It contains images labeled as either Normal or Pneumonia. The dataset exhibits a class imbalance, which is a realistic challenge in medical datasets.

**Dependencies**
---
*This project uses the following libraries and tools:*

**Python 3.12.2**: The core programming language used for implementing the classifier and handling data.

**NumPy**: Essential for numerical operations, especially in vectorizing text data.

**Pandas**: Used for loading and manipulating the dataset.

**Scikit-learn**: implementing the Multinomial Naive Bayes classifier, and evaluating the model through metrics like accuracy,f1-score, precision, and recall.

**NLTK**: For natural language processing tasks such as tokenization and stop word removal.

**Email Spam Collection Dataset**: The dataset contains 5172 emails, of which 1527 are spam and the remaining are non-spam.






