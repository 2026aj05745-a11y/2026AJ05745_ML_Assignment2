# Machine Learning Assignment 2

## 1. Problem Statement

The objective of this project is to implement and compare five machine learning classification algorithms using a common dataset. The models are trained and evaluated to identify whether a breast tumour is malignant or benign.

The project includes model development, performance evaluation, saving trained models, and developing an interactive web application using Streamlit.

The five classification algorithms implemented are:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

The performance of each model is evaluated using Accuracy, AUC Score, Precision, Recall, F1 Score and Matthews Correlation Coefficient (MCC).

## 2. Dataset Description

**Dataset Name:** Wisconsin Diagnostic Breast Cancer Dataset

**Dataset Source:** UCI Machine Learning Repository  
https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

The Wisconsin Diagnostic Breast Cancer dataset is used for this binary classification problem. It contains numerical measurements calculated from digitized images of breast mass samples.

The dataset contains 569 instances and 30 numerical input features. The target variable represents two classes: Malignant and Benign.

### Dataset Information

| Description | Details |
|---|---|
| Dataset name | Wisconsin Diagnostic Breast Cancer |
| Total instances | 569 |
| Total input features | 30 |
| Target variable | Diagnosis |
| Classification type | Binary classification |
| Missing values | 0 |
| Malignant samples | 212 |
| Benign samples | 357 |
| Training samples | 455 |
| Testing samples | 114 |

### Target Classes

- 0 – Malignant
- 1 – Benign

The dataset was divided into training and testing sets. The training set contains 455 samples, while the testing set contains 114 samples.

## 3. GitHub Repository Link

**GitHub Repository:**  
https://github.com/2026aj05745-a11y/2026AJ05745_ML_Assignment2

The repository contains the source code, trained classification models, test dataset, requirements file and project documentation.

### Project Structure

```text
2026AJ05745_ML_Assignment2/
│
├── app.py
├── 2026AJ05745.ipynb
├── README.md
├── requirements.txt
├── test_data.csv
│
├── model/
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   └── support_vector_machine.joblib
│
└── results/
    └── metrics.csv
```

## 4. Models Used and Evaluation Metrics

All five classification models were implemented using the same dataset and evaluated on the testing data.

The following six evaluation metrics were calculated for each model:

- **Accuracy:** Measures the overall percentage of correct predictions.
- **AUC Score:** Measures the ability of the model to distinguish between the two classes.
- **Precision:** Measures how many predicted positive samples are actually positive.
- **Recall:** Measures how many actual positive samples are correctly identified.
- **F1 Score:** Represents the harmonic mean of precision and recall.
- **MCC:** Measures the quality of binary classification predictions, considering all four confusion matrix categories.

### 4.1 Model Performance Comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9211 | 0.9163 | 0.9565 | 0.9167 | 0.9362 | 0.8341 |
| KNN | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| Naive Bayes | 0.9298 | 0.9868 | 0.9444 | 0.9444 | 0.9444 | 0.8492 |
| Support Vector Machine | 0.9825 | 0.9950 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |

### 4.2 Observations on Model Performance

| ML Model Name | Observation about Model Performance |
|---|---|
| Logistic Regression | Logistic Regression achieved 98.25% accuracy and the highest AUC score of 0.9954. It provided excellent classification performance with very few incorrect predictions. |
| Decision Tree | Decision Tree achieved 92.11% accuracy. It performed reasonably well but had comparatively lower AUC and MCC scores than the other models. |
| KNN | KNN achieved 95.61% accuracy and 0.9788 AUC. It provided good classification results and performed better than Decision Tree and Naive Bayes in terms of accuracy. |
| Naive Bayes | Gaussian Naive Bayes achieved 92.98% accuracy and an AUC score of 0.9868. It showed good class separation, although its overall accuracy was lower than Logistic Regression, KNN and SVM. |
| Support Vector Machine | SVM achieved 98.25% accuracy and an AUC score of 0.9950. It performed very well and produced results close to Logistic Regression. |

### 4.3 Overall Winner

Based on the evaluation results, **Logistic Regression is the overall winner** for this dataset.

It achieved the highest accuracy jointly with Support Vector Machine at 98.25%, and obtained the highest AUC score of 0.9954. It also achieved an MCC score of 0.9623.

Support Vector Machine also demonstrated excellent performance, with results very close to Logistic Regression.

## 5. Streamlit Web Application

An interactive web application was developed using Streamlit to demonstrate the trained classification models.

**Live Streamlit Application:**  
https://2026aj05745mlassignment2-uu7t7fun4govctqppa6tx9.streamlit.app

### Features of the Application

1. **Dataset Upload:** Users can upload a labelled test dataset in CSV format.
2. **Model Selection:** Users can select any one of the five trained classification models.
3. **Evaluation Metrics:** The application displays Accuracy, AUC, Precision, Recall, F1 Score and MCC.
4. **Confusion Matrix:** Displays the confusion matrix for the selected model.
5. **Classification Report:** Displays precision, recall, F1 score and support for each class.
6. **Model Comparison:** Displays the performance comparison of all five classification models.

The application loads the previously saved trained models and evaluates them using the uploaded test data. It does not retrain the models on the uploaded test dataset.

## 6. Requirements and Dependencies

The project uses the following Python libraries:

- Streamlit
- Scikit-learn
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Joblib

The dependencies are listed in the `requirements.txt` file.

### Running the Application Locally

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

## 7. Conclusion

In this assignment, five machine learning classification algorithms were implemented and evaluated using the Wisconsin Diagnostic Breast Cancer dataset.

The results show that Logistic Regression and Support Vector Machine provided the best overall classification performance. Logistic Regression achieved the highest AUC score, while both models achieved the same accuracy.

An interactive Streamlit application was also developed and deployed to demonstrate the models using uploaded test data.

This project provided practical experience in machine learning model implementation, evaluation, model saving, GitHub repository management and web application deployment.
