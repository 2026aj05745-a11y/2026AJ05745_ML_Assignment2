
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# Page configuration
st.set_page_config(
    page_title="Breast Cancer ML Model Evaluation",
    page_icon="📊",
    layout="wide"
)

# Application heading
st.title("Machine Learning Classification Dashboard")

st.markdown("""
This application compares five machine learning classification
models using the Wisconsin Diagnostic Breast Cancer dataset.

Upload a labelled test CSV, select a model, and explore its
performance through evaluation metrics and visualizations.
""")

st.divider()

# Saved model locations
MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "K-Nearest Neighbors": "model/knn.joblib",
    "Gaussian Naive Bayes": "model/naive_bayes.joblib",
    "Support Vector Machine": "model/support_vector_machine.joblib"
}

# Expected feature columns
@st.cache_data
def load_feature_names():
    reference = pd.read_csv("test_data.csv")
    return [column for column in reference.columns if column != "target"]

feature_names = load_feature_names()

# Sidebar
st.sidebar.header("Evaluation Settings")

uploaded_file = st.sidebar.file_uploader(
    "Upload test dataset",
    type=["csv"]
)

selected_model = st.sidebar.selectbox(
    "Select classification model",
    list(MODEL_FILES.keys())
)

st.sidebar.caption(
    "The uploaded CSV must contain 30 input features "
    "and a target column."
)

# Evaluation function
def evaluate_model(model_name, X, y):

    model_path = MODEL_FILES[model_name]

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    model = joblib.load(model_path)

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    scores = {
        "Accuracy": accuracy_score(y, predictions),
        "AUC": roc_auc_score(y, probabilities),
        "Precision": precision_score(
            y, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y, predictions, zero_division=0
        ),
        "F1 Score": f1_score(
            y, predictions, zero_division=0
        ),
        "MCC": matthews_corrcoef(y, predictions)
    }

    return model, predictions, scores


if uploaded_file is None:

    st.info("Upload your test_data.csv file using the sidebar.")

    st.subheader("Available Classification Models")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Models", "5")
    col2.metric("Input Features", "30")
    col3.metric("Dataset Type", "Binary")

    st.markdown("### Models included")

    for name in MODEL_FILES:
        st.write("•", name)

else:

    try:
        data = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Dataset")

        st.write("Dataset shape:", data.shape)

        st.dataframe(data.head(), use_container_width=True)

        # Validate required columns
        if (
            len(data.columns) != len(feature_names) + 1
            or set(data.columns) != set(feature_names + ["target"])
        ):
            st.error(
                "Incorrect CSV structure. Please upload a file "
                "containing all 30 expected features and target."
            )

        elif data.isnull().any().any():
            st.error(
                "The uploaded dataset contains missing values."
            )

        elif not set(data["target"].unique()).issubset({0, 1}):
            st.error(
                "Target values must be 0 or 1."
            )

        elif data["target"].nunique() != 2:
            st.error(
                "The test dataset must contain both target classes "
                "to calculate AUC."
            )

        else:
            X = data[feature_names]
            y = data["target"].astype(int)

            # Evaluate selected model
            model, predictions, scores = evaluate_model(
                selected_model, X, y
            )

            st.divider()

            st.subheader(
                f"Performance: {selected_model}"
            )

            # Display metrics
            metric_names = [
                "Accuracy", "AUC", "Precision",
                "Recall", "F1 Score", "MCC"
            ]

            columns = st.columns(3)

            for index, metric in enumerate(metric_names):
                columns[index % 3].metric(
                    metric,
                    f"{scores[metric]:.4f}"
                )

            # Confusion matrix
            st.divider()
            st.subheader("Confusion Matrix")

            cm = confusion_matrix(
                y,
                predictions,
                labels=[0, 1]
            )

            fig, ax = plt.subplots(figsize=(7, 5))

            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["Malignant", "Benign"],
                yticklabels=["Malignant", "Benign"],
                ax=ax
            )

            ax.set_xlabel("Predicted Class")
            ax.set_ylabel("Actual Class")
            ax.set_title(selected_model)

            st.pyplot(fig)
            plt.close(fig)

            # Classification report
            st.divider()
            st.subheader("Classification Report")

            report = classification_report(
                y,
                predictions,
                labels=[0, 1],
                target_names=["Malignant", "Benign"],
                output_dict=True,
                zero_division=0
            )

            report_df = pd.DataFrame(report).transpose()

            st.dataframe(
                report_df.round(4),
                use_container_width=True
            )

            # Compare all models
            st.divider()
            st.subheader("Comparison of All Five Models")

            comparison = []

            for name in MODEL_FILES:

                _, _, model_scores = evaluate_model(
                    name, X, y
                )

                comparison.append({
                    "Model": name,
                    **model_scores
                })

            comparison_df = pd.DataFrame(comparison)

            st.dataframe(
                comparison_df.round(4),
                use_container_width=True
            )

            # Download comparison results
            csv_data = comparison_df.to_csv(index=False)

            st.download_button(
                label="Download Model Comparison CSV",
                data=csv_data,
                file_name="model_comparison.csv",
                mime="text/csv"
            )

    except Exception as error:
        st.error(f"Unable to evaluate the uploaded data: {error}")

st.divider()

st.caption(
    "M.Tech AIML | Machine Learning Assignment 2 | BITS Pilani"
)
