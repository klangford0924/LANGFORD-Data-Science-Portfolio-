import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    confusion_matrix, mean_squared_error, r2_score
)

# Streamlit page config
st.set_page_config(page_title="Interactive ML App", layout="wide")
st.title("🧠 Interactive Machine Learning Explorer")

# Sidebar - Data source selection
st.sidebar.header("📂 Dataset Options")
dataset_choice = st.sidebar.radio("Choose data source:", [
    "Upload CSV",
    "Sample: Iris (KNN)",
    "Sample: Diabetes (Linear Regression)",
    "Sample: Breast Cancer (Logistic Regression)"
])

# Load dataset
data = None  # define upfront to avoid undefined errors

if dataset_choice == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
    else:
        st.warning("Please upload a dataset or choose a sample dataset.")
        st.stop()

elif dataset_choice == "Sample: Iris (KNN)":
    iris = load_iris(as_frame=True)
    data = iris.frame
    st.info("""
    🌸 **Iris Dataset**  
    A classic dataset used for flower classification.  
    It contains 150 samples of iris flowers across 3 species (*setosa, versicolor, virginica*),  
    with 4 features: petal and sepal length and width.
    """)

elif dataset_choice == "Sample: Diabetes (Linear Regression)":
    diabetes = load_diabetes(as_frame=True)
    data = diabetes.frame
    st.info("""
    🩺 **Diabetes Dataset**  
    This dataset tracks the progression of diabetes in patients over time.  
    It includes 442 samples with features like age, BMI, blood pressure, and more.  
    The target is a continuous score representing disease progression one year later.
    """)

elif dataset_choice == "Sample: Breast Cancer (Logistic Regression)":
    cancer = load_breast_cancer(as_frame=True)
    data = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    data['target'] = cancer.target
    st.info("""
    🎗️ **Breast Cancer Wisconsin Dataset**  
    This dataset contains 569 samples of tumors, with 30 numeric features extracted from images.  
    The goal is to classify tumors as **malignant (0)** or **benign (1)**.  
    Ideal for binary classification with models like **Logistic Regression**.
    """)

# Data preview
if data is not None:
    with st.expander("🔍 Data Preview"):
        st.write(data.head())
else:
    st.error("No dataset found. Please check your selection.")
    st.stop()

# Sidebar - Feature/target selection
st.sidebar.header("🔧 Model Settings")
target_column = st.sidebar.selectbox("Select target column", data.columns)
feature_columns = st.sidebar.multiselect(
    "Select feature columns",
    [col for col in data.columns if col != target_column],
    default=[col for col in data.columns if col != target_column][:2]
)

if not feature_columns:
    st.warning("Please select at least one feature column.")
    st.stop()

# Model type
model_type = st.sidebar.selectbox("Choose Model Type", [
    "K-Nearest Neighbors (KNN)",
    "Logistic Regression",
    "Linear Regression"
])

# Model guidance
with st.sidebar.expander("🧠 Not sure which model to choose?"):
    st.markdown("""
    📝 If your target column has *categories/labels*, go with **KNN or Logistic Regression**.  
    If it's *numerical/continuous*, use **Linear Regression**!
    """)

# Hyperparameters
if model_type == "K-Nearest Neighbors (KNN)":
    k = st.sidebar.slider("Number of neighbors (k)", 1, 20, 5)
elif model_type == "Logistic Regression":
    max_iter = st.sidebar.slider("Max Iterations", 100, 1000, 300)

# Data preparation
X = data[feature_columns].copy()
y = data[target_column].copy()

le = LabelEncoder()

# Encode categorical features
for col in X.columns:
    if X[col].dtype == "object":
        X[col] = le.fit_transform(X[col])

# Encode target if classification
if model_type in ["K-Nearest Neighbors (KNN)", "Logistic Regression"] and y.dtype == "object":
    y = le.fit_transform(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model initialization
if model_type == "K-Nearest Neighbors (KNN)":
    model = KNeighborsClassifier(n_neighbors=k)
elif model_type == "Logistic Regression":
    model = LogisticRegression(max_iter=max_iter)
elif model_type == "Linear Regression":
    model = LinearRegression()

# Train model
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Results tabs
tabs = st.tabs(["📊 Metrics", "📈 Visualizations", "📄 Raw Predictions"])

# Tab 1: Metrics
with tabs[0]:
    st.subheader("📊 Model Performance")
    if model_type in ["K-Nearest Neighbors (KNN)", "Logistic Regression"]:
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        st.metric("Accuracy", f"{acc:.2f}")
        st.metric("Precision", f"{prec:.2f}")
        st.metric("Recall", f"{rec:.2f}")
    else:
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        st.metric("R² Score", f"{r2:.2f}")
        st.metric("Mean Squared Error", f"{mse:.2f}")

# Tab 2: Visualizations
with tabs[1]:
    st.subheader("📈 Visual Output")
    fig, ax = plt.subplots()
    if model_type in ["K-Nearest Neighbors (KNN)", "Logistic Regression"]:
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap="Greens", ax=ax)
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    else:
        ax.scatter(y_test, y_pred)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title("Actual vs. Predicted")
    st.pyplot(fig)

# Tab 3: Raw predictions
with tabs[2]:
    st.subheader("📄 Raw Predictions")
    results_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
    st.write(results_df.reset_index(drop=True))
