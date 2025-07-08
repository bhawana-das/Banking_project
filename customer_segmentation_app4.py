# --- Import necessary libraries ---
import streamlit as st  # Web application framework
import pandas as pd     # For data manipulation
import numpy as np      # For numerical operations
from sklearn.cluster import KMeans  # For clustering
from sklearn.preprocessing import StandardScaler  # For feature scaling
from sklearn.decomposition import PCA  # For dimensionality reduction
import matplotlib.pyplot as plt  # For plotting
import seaborn as sns  # For enhanced visualizations

def run():
    st.title("Customer segmentation Prediction - Use Case 4")
    st.write("Welcome to Use Case 4 App!")
# --- Configure the Streamlit app ---
st.set_page_config(page_title="🧑‍🤝‍🧑 Customer Segmentation", layout="wide")

# --- App Title and Description ---
st.title("🧑‍🤝‍🧑 Customer Segmentation Using K-Means Clustering")
st.markdown("Cluster customers into Low, Medium, and High activity groups using transaction behavior.")

# --- Upload CSV File ---
uploaded_file = st.file_uploader("📂 Upload your customer dataset (CSV)", type=["csv"])

# --- If file is uploaded ---
if uploaded_file is not None:
    # --- Load CSV into a DataFrame ---
    df = pd.read_csv(uploaded_file)
    
    # --- Show preview of the data ---
    st.subheader("🔍 Data Preview")
    st.dataframe(df.head())

    # --- Let user select numerical features for clustering ---
    st.subheader("🧮 Select Features for Clustering")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()  # Identify numeric columns
    selected_features = st.multiselect("Choose at least 2 numerical features:", numeric_cols, default=numeric_cols)

    # --- Stop if fewer than two features selected ---
    if len(selected_features) < 2:
        st.warning("⚠️ Please select at least two features to proceed.")
        st.stop()

    # --- Prepare selected data and drop rows with missing values ---
    data = df[selected_features].dropna()

    # --- Standardize the selected features ---
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    # --- Apply KMeans clustering with 3 clusters ---
    k = 3
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(scaled_data)
    df['Cluster'] = cluster_labels  # Store cluster labels in DataFrame

    # --- Map cluster indexes to activity levels based on cluster centers ---
    cluster_sums = kmeans.cluster_centers_.sum(axis=1)  # Sum of each cluster center’s features
    sorted_indices = np.argsort(cluster_sums)  # Sort cluster indices by activity level
    activity_labels = ['Low Activity', 'Medium Activity', 'High Activity']  # Custom labels
    label_map = {sorted_indices[i]: activity_labels[i] for i in range(3)}  # Map index to label
    df['Activity_Level'] = df['Cluster'].map(label_map)  # Apply label mapping

    # --- Show number of customers in each activity level ---
    st.subheader("📊 Cluster Counts")
    st.bar_chart(df['Activity_Level'].value_counts())  # Bar chart of cluster sizes

    # --- Reduce dimensions to 2D using PCA for visualization ---
    st.subheader("📈 Cluster Visualization (PCA 2D)")
    pca = PCA(n_components=2)
    components = pca.fit_transform(scaled_data)
    df['PCA1'] = components[:, 0]
    df['PCA2'] = components[:, 1]

    # --- Create scatter plot of clusters using PCA components ---
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Activity_Level', palette='Set2', s=80, ax=ax)
    ax.set_title("Customer Segmentation (PCA Projection)", fontsize=14)
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.grid(True)
    st.pyplot(fig)  # Display plot in Streamlit

    # --- Show full segmented dataset ---
    st.subheader("📄 Segmented Data Table")
    st.dataframe(df)

    # --- Provide download button for segmented data as CSV ---
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Segmented Data as CSV", csv, file_name="segmented_customers.csv", mime='text/csv')

else:
    # --- If no file is uploaded, show info message ---
    st.info("💡 Please upload a CSV file with numeric features like transaction count, amount, etc.")

st.success("App Ready")

