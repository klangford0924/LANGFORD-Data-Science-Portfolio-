# 🧠 Interactive Machine Learning Explorer

Welcome to the **Interactive Machine Learning Explorer** — a beginner-friendly [Streamlit](https://streamlit.io) web app that allows users to interactively explore supervised machine learning models using built-in datasets or their own CSV files.
---


## 🚀 Project Overview

This app was created as part of my Introduction to Data Science course to demonstrate an understanding of model building, user interaction, and deployment with Streamlit. It empowers users—regardless of coding experience—to:
- Upload and explore datasets
- Choose and apply supervised learning models
- Tune hyperparameters
- View visual evaluation metrics interactively

The project highlights key data science skills: model implementation, interactive web development, and user-centered design.



---

## 🛠️ Instructions

### 📍 Option 1: Try It Online  
Just click the link below — no install needed!  
📎 [Live Demo App](https://langford-datascience-machinelearningproject.streamlit.app/)

### 💻 Option 2: Run Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/interactive-ml-explorer.git
   cd interactive-ml-explorer
   ```
   
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   
4. Launch the app:
   ```bash
   streamlit run app.py
   ```
---


## ⚙️ App Features

- Upload your own CSV file or select a built-in dataset:
  - 🌸 Iris
  - 🩺 Diabetes
  - 🎗️ Breast Cancer

- Choose from three supervised models:
  - K-Nearest Neighbors (KNN)
  - Logistic Regression
  - Linear Regression

- Feature selection: Choose input and target variables
- Tune key hyperparameters:
  - `k` (KNN)
  - `max_iter` (Logistic Regression)
- View output:
  - Accuracy, confusion matrix, MSE, classification reports
  - Interactive visualizations (scatter plots, decision boundaries)

---

## 📚 References
- Datasets from scikit-learn
- Powered by Streamlit


