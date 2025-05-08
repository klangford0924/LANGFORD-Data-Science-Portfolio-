# 🔍 Machine Learning Unsupervised App

Welcome to the **Machine Learning Unsupervised App**, a Streamlit web app that allows users to explore unsupervised machine learning techniques on real or custom datasets. This app is designed for students, data enthusiasts, or practitioners who want to visualize clustering behavior and learn how preprocessing and hyperparameters influence results.

---

## 🚀 Project Overview

This app enables users to:
- Upload their own datasets or use built-in samples (Iris dataset or synthetic blobs).
- Preprocess data with scaling and missing value strategies.
- Apply unsupervised ML models: **KMeans**, **Hierarchical Clustering**, and **PCA**.
- Visualize PCA projections, dendrograms, elbow plots, and silhouette scores.

It is designed as an educational and interactive tool for understanding unsupervised learning concepts in a hands-on way.

---

## 🛠️ Instructions

### 📦 Run Locally
1. **Clone the repository**  
```bash
git clone https://github.com/klangford0924/LANGFORD-Data-Science-Portfolio-.git
cd LANGFORD-Data-Science-Portfolio-/MLUnsupervisedApp
```

2. Install required libraries
```bash
pip install -r requirements.txt
Run the app
```

3. Run the app
```bash
streamlit run DataScienceFinalProject.Langford.py
```

Or... skip the coding and launch the deployed version here: 
🔗 [Interactive Clustering Visualizer (Streamlit App)](https://mlunsuperivisedlearning-langford.streamlit.app/)



## ⚙️ Features
**Dataset flexibility**: Upload your own CSV or choose from sample datasets.

**Preprocessing controls**: Choose between dropping/filling missing values, apply scaling (Standard/MinMax/None).

**Model selection**:

**KMeans Clustering**:
- Select number of clusters.
- View elbow plot and silhouette score.

**Hierarchical Clustering**:
- Choose linkage method (ward, complete, average, single).
- Visualize dendrogram.

**PCA**:
- Select number of components.
- Visualize PCA projection.

**Evaluation Tools**: Dynamic charts and scoring metrics help guide model selection and evaluation.

## 📚 References
- [Guide to Principal Component Analysis – Turing.com](https://www.turing.com/kb/guide-to-principal-component-analysis)


### 📸 App Preview



