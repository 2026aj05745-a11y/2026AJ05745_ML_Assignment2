
# BITS Pilani M.Tech AIML - Machine Learning Assignment 2

## 1. Problem Statement

The objective of this assignment is to implement and compare five machine learning classification algorithms using a common dataset.

The project also includes an interactive Streamlit application that allows users to upload test data, select a classification model and view its evaluation metrics.

## 2. Dataset Description

**Dataset:** Wisconsin Diagnostic Breast Cancer Dataset

**Source:** UCI Machine Learning Repository

**Dataset URL:** https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

- Total instances: 569
- Input features: 30
- Target classes: 2
- Classification type: Binary
- Missing values: 0

Target labels:
- 0: Malignant
- 1: Benign

The dataset was divided into 80% training data and 20% testing data using stratified sampling.

## 3. GitHub Repository Link

Add the GitHub repository URL after creating the repository.

## 4. Models Used and Evaluation Metrics

Five classification algorithms were implemented:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors
4. Gaussian Naive Bayes
5. Support Vector Machine

### Model Comparison

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9211 | 0.9163 | 0.9565 | 0.9167 | 0.9362 | 0.8341 |
| KNN | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| Naive Bayes | 0.9298 | 0.9868 | 0.9444 | 0.9444 | 0.9444 | 0.8492 |
| Support Vector Machine | 0.9825 | 0.9950 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |

Note: Precision, Recall and F1 use class 1 (Benign) as the positive class.

### Observations

**Logistic Regression:** Achieved 98.25% accuracy and 99.54% AUC. It correctly classified 112 out of 114 test samples.

**Decision Tree:** Achieved 92.11% accuracy and correctly classified 105 samples.

**KNN:** Achieved 95.61% accuracy and correctly classified 109 samples.

**Naive Bayes:** Achieved 92.98% accuracy and 98.68% AUC. It correctly classified 106 samples.

**Support Vector Machine:** Achieved 98.25% accuracy and correctly classified 112 samples. Its performance was very close to Logistic Regression.

**Overall Winner:** Logistic Regression performed best based on the highest AUC and joint top performance on the other main metrics. SVM was a very close second.

## 5. Streamlit Application

The application provides:

- CSV test dataset upload.
- Classification model selection.
- Accuracy, AUC, Precision, Recall, F1 and MCC.
- Confusion matrix.
- Classification report.
- Comparison of all five models.
- Downloadable comparison results.

The application loads saved models and evaluates uploaded test data without retraining.

**Live Streamlit App:** Add the deployed application URL here.

## 6. Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Streamlit
- GitHub

## 7. Conclusion

In this assignment, five classification algorithms were implemented using the Wisconsin Diagnostic Breast Cancer dataset.

Their performance was evaluated using six classification metrics. Logistic Regression achieved the best overall results, while Support Vector Machine showed nearly identical performance.

An interactive Streamlit application was developed to make model evaluation easier through a web interface.
