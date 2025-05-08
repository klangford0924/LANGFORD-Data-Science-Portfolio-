import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import load_iris, make_blobs

# Set app title
st.title("Interactive Clustering Visualizer")

# Sidebar: User selects data source
st.sidebar.markdown("### 📁 Dataset Source")
data_source = st.sidebar.radio("Select data source", ["Upload CSV", "Sample: Iris Dataset", "Sample: Synthetic Blobs"])

# Load dataset based on selection
if data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        st.info(" 📁 Upload a CSV file to begin.")
        st.stop()
elif data_source == "Sample: Iris Dataset":
    iris = load_iris(as_frame=True)
    df = iris.frame
else:
    X, y = make_blobs(n_samples=150, centers=3, n_features=4, random_state=42)
    df = pd.DataFrame(X, columns=[f"Feature_{i+1}" for i in range(X.shape[1])])

# Display first few rows of dataset
st.write("## 🔭 Preview of Dataset", df.head())

# --- Preprocessing Options ---
st.sidebar.markdown("### 📌 Preprocessing Options")
use_only_numeric = st.sidebar.checkbox("Use numeric columns only", value=True)
missing_strategy = st.sidebar.selectbox("Missing Value Strategy", ["Drop rows", "Fill with mean", "Fill with 0"])
scaling_method = st.sidebar.selectbox("Scaling Method", ["StandardScaler", "MinMaxScaler", "None"])

# Apply preprocessing steps
df_proc = df.copy()

# Keep only numeric data if selected
if use_only_numeric:
    df_proc = df_proc.select_dtypes(include=np.number)

# Handle missing values
if missing_strategy == "Drop rows":
    df_proc = df_proc.dropna()
elif missing_strategy == "Fill with mean":
    df_proc = df_proc.fillna(df_proc.mean(numeric_only=True))
elif missing_strategy == "Fill with 0":
    df_proc = df_proc.fillna(0)

# Apply scaling method
if scaling_method == "StandardScaler":
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_proc)
elif scaling_method == "MinMaxScaler":
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(df_proc)
else:
    X_scaled = df_proc.values

# --- ML Technique Selection ---
st.sidebar.markdown("### 📌 Choose ML Technique")
ml_method = st.sidebar.selectbox("Machine Learning Technique", ["KMeans Clustering", "Hierarchical Clustering", "Principle Componet Analyisis Only"])

# Execute ML workflow based on user selection
if ml_method == "KMeans Clustering":
    st.sidebar.markdown("### KMeans Parameters")
    k = st.sidebar.slider("Number of clusters (k)", 2, 10, 3)
    model = KMeans(n_clusters=k, random_state=42)
    cluster_labels = model.fit_predict(X_scaled)
    silhouette = silhouette_score(X_scaled, cluster_labels)

    # Elbow plot to help choose optimal k
    with st.expander("Elbow Plot (KMeans Inertia)", expanded=True):
        inertia = []
        for k_ in range(1, 11):
            km = KMeans(n_clusters=k_, random_state=42)
            km.fit(X_scaled)
            inertia.append(km.inertia_)

        fig, ax = plt.subplots()
        ax.plot(range(1, 11), inertia, 'bo-')
        ax.set_xlabel('Number of Clusters (k)')
        ax.set_ylabel('Inertia')
        ax.set_title('Elbow Method')
        st.pyplot(fig)

    # Silhouette plot to assess clustering quality
    with st.expander("Silhouette Score", expanded=True):
        scores = []
        for k_ in range(2, 11):
            km = KMeans(n_clusters=k_, random_state=42)
            labels = km.fit_predict(X_scaled)
            scores.append(silhouette_score(X_scaled, labels))

        fig, ax = plt.subplots()
        ax.plot(range(2, 11), scores, 'ro-')
        ax.set_xlabel('Number of Clusters')
        ax.set_ylabel('Silhouette Score')
        ax.set_title('Silhouette Scores for KMeans')
        st.pyplot(fig)
        st.write(f"**Selected k = {k}: Silhouette Score = {scores[k-2]:.3f}**")

elif ml_method == "Hierarchical Clustering":
    st.sidebar.markdown("### 📌Hierarchical Parameters")
    linkage_method = st.sidebar.selectbox("Linkage Method", ["ward", "complete", "average", "single"])
    num_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 3)
    model = AgglomerativeClustering(n_clusters=num_clusters, linkage=linkage_method)
    cluster_labels = model.fit_predict(X_scaled)
    silhouette = silhouette_score(X_scaled, cluster_labels)

    # Display dendrogram
    with st.expander(" Dendrogram", expanded=True):
        Z = linkage(X_scaled, method=linkage_method)
        fig, ax = plt.subplots(figsize=(10, 5))
        dendrogram(Z, ax=ax)
        ax.set_title(f'Dendrogram ({linkage_method} linkage)')
        st.pyplot(fig)

else:
    cluster_labels = None
    silhouette = None

# PCA Dimensionality Reduction
st.sidebar.markdown("### 📌 PCA Components")
n_components = st.sidebar.slider("PCA Components", 2, min(len(df_proc.columns), 5), 2)
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)

# Visualize PCA result
with st.expander("PCA Scatter Plot", expanded=True):
    fig, ax = plt.subplots()
    if cluster_labels is not None:
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='Set1')
        ax.legend(*scatter.legend_elements(), title="Cluster")
    else:
        ax.scatter(X_pca[:, 0], X_pca[:, 1])
    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    ax.set_title('PCA Projection')
    st.pyplot(fig)
    st.write(f"**Explained Variance:** {pca.explained_variance_ratio_[:n_components]}")

# Sidebar: Show clustering summary metrics
if silhouette is not None:
    st.sidebar.markdown("### 📌 Cluster Quality Summary")
    st.sidebar.write(f"Silhouette Score: {silhouette:.3f}")
    st.sidebar.write(f"Explained Variance: {sum(pca.explained_variance_ratio_[:n_components]):.2%}")
