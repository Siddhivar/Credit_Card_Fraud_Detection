# Credit Card Fraud Detection

This project implements machine learning techniques to detect fraudulent credit card transactions using the **Credit Card Fraud Detection** dataset. The dataset is highly imbalanced, and the project demonstrates the application of various techniques to handle class imbalance, model training, and evaluation using metrics such as accuracy, precision, recall, F1-score, and confusion matrix.

## Table of Contents
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Usage](#usage)
- [Requirements](#requirements)
- [License](#license)

## Project Overview

This project focuses on building a machine learning model to detect fraud in credit card transactions. We used a **Random Forest Classifier** and **Logistic Regression** as the models for fraud detection. The dataset is highly imbalanced, so techniques like **SMOTE** (Synthetic Minority Over-sampling Technique) and **Tomek Links** were used to balance the classes.

### Key Features:
- Data preprocessing steps to handle missing values and imbalance in the data.
- Use of **SMOTE** to oversample the minority class.
- **Logistic Regression** and **Random Forest Classifier** models for fraud detection.
- Model evaluation using **accuracy**, **precision**, **recall**, **F1-score**, and **confusion matrix**.

## Installation

To run this project, make sure to have the necessary dependencies installed. You can install them by running the following command:

```bash
pip install -r requirements.txt
```
## Data Preprocessing
The dataset is preprocessed with the following steps:

#### Handling missing values: Checked for missing values and handled them.
#### Class imbalance handling: SMOTE was used for oversampling the minority class.
#### Feature scaling: StandardScaler was used to scale the features to bring them to the same range.

## Model Training
The following models were used:

1.Logistic Regression: Used for binary classification of fraudulent vs. legitimate transactions.
2.Random Forest Classifier: An ensemble model used with 100 estimators and a maximum depth of 10 to classify the transactions.

## Logistic Regression
```bash
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(max_iter=500)
log_reg.fit(X_train, Y_train)
```
## Random Forest
```bash
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1)
rf_model.fit(X_train, Y_train)
```
## Model Evaluation
Each model is evaluated based on the following metrics:

Accuracy: The overall proportion of correct predictions.
Precision: The proportion of true positive predictions out of all positive predictions.
Recall: The proportion of true positive predictions out of all actual positives.
F1-score: The harmonic mean of precision and recall.
Confusion Matrix: A table that visualizes the performance of the model in terms of true positives, false positives, true negatives, and false negatives.

## Usage
1.Clone the repository:
```bash
git clone https://github.com/your-username/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```
2.Install all dependencies:
```bash
pip install -r requirements.txt
```
3.Run the model:
```bash
python credit_card_fraud_detection.py
```
This will load the dataset, preprocess the data, train the models, and output the evaluation metrics and confusion matrix.

## Requirements
Python 3.6 or above
scikit-learn
imblearn
pandas
numpy
matplotlib
seaborn (optional for visualizations)
Install the required libraries using the following command:

```bash
pip install scikit-learn imblearn pandas numpy matplotlib seaborn
```
