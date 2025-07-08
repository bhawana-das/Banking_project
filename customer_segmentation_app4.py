# File: customer_segmentation_app4.py

# --- Import necessary libraries ---
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

def run():
    st.title("🧑‍🤝‍🧑 Customer Segmentation - Use Case 4")
    st.write("Cluster customers based on behavior using K-Means.")

    # --- Upload Section ---
    uploaded_file = st.file_uploader("📂 Upload your customer dataset (CSV)", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        st.subheader("🔍 Data Preview")
        st.dataframe(df.head())

        # --- Feature Selection ---
        st.subheader("🔧 Select Features for Clustering")
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        selected_features = st.multiselect("Select at least two numeric features:", numeric_cols, default=numeric_cols)

        if len(selected_features) < 2:
            st.warning("⚠️ Please select at least two features to continue.")
            st.stop()

        data = df[selected_features].dropna()
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)

        # --- KMeans Clustering ---
        k = 3
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(scaled_data)
        df['Cluster'] = cluster_labels

        # --- Activity Level Labeling ---
        cluster_sums = kmeans.cluster_centers_.sum(axis=1)
        sorted_indices = np.argsort(cluster_sums)
        activity_labels = ['Low Activity', 'Medium Activity', 'High Activity']
        label_map = {sorted_indices[i]: activity_labels[i] for i in range(3)}
        df['Activity_Level'] = df['Cluster'].map(label_map)

        # --- Cluster Count ---
        st.subheader("📊 Cluster Distribution")
        st.bar_chart(df['Activity_Level'].value_counts())

        # --- PCA for 2D Visualization ---
        st.subheader("📈 PCA Cluster Visualization")
        pca = PCA(n_components=2)
        pca_components = pca.fit_transform(scaled_data)
        df['PCA1'] = pca_components[:, 0]
        df['PCA2'] = pca_components[:, 1]

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Activity_Level', palette='Set2', s=80, ax=ax)
        ax.set_title("Customer Clusters (PCA View)")
        st.pyplot(fig)

        # --- Download Segmented Data ---
        st.subheader("📄 Segmented Dataset")
        st.dataframe(df)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download CSV", csv, file_name="segmented_customers.csv", mime="text/csv")

    else:
        st.info("📁 Please upload a dataset to begin.")
