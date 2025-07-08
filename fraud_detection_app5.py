# --- Import necessary libraries ---
import streamlit as st  # For building interactive web apps
import pandas as pd  # For data manipulation
import numpy as np  # For numerical operations
from sklearn.ensemble import IsolationForest, RandomForestClassifier  # ML models
from sklearn.preprocessing import StandardScaler  # Feature scaling
from sklearn.metrics import classification_report, accuracy_score  # Model evaluation
import matplotlib.pyplot as plt  # Plotting
import seaborn as sns  # Statistical plots

# --- Streamlit app configuration ---
st.set_page_config(page_title="🚨 Fraud / Outlier Detection", layout="wide")  # Set app title and layout
st.title("🚨 Fraud / Outlier Detection using ML Models")  # Main title
st.markdown("Detect suspicious banking transactions using Isolation Forest (unsupervised) and Random Forest (supervised, if labels exist).")  # Description

# --- File upload section ---
uploaded_file = st.file_uploader("📂 Upload transaction dataset (CSV)", type=["csv"])  # Upload CSV file

# --- Check if file is uploaded ---
if uploaded_file:
    df = pd.read_csv(uploaded_file)  # Read CSV into DataFrame
    st.subheader("🔍 Data Preview")  # Subheading for data preview
    st.dataframe(df.head())  # Show first few rows of data

    # --- Feature selection for analysis ---
    st.subheader("🧮 Select Features for Analysis")  # Subheading for feature selection
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()  # Get numerical columns
    selected_features = st.multiselect("Select numerical features:", numeric_cols, default=numeric_cols)  # Allow user to select features

    # --- Validation: at least two features required ---
    if len(selected_features) < 2:
        st.warning("⚠️ Please select at least two features.")  # Warning if not enough features selected
        st.stop()  # Stop execution if validation fails

    # --- Data preparation ---
    X = df[selected_features].dropna()  # Remove rows with missing values in selected features
    scaler = StandardScaler()  # Initialize standard scaler
    X_scaled = scaler.fit_transform(X)  # Scale selected features

    # --- Unsupervised model: Isolation Forest for anomaly detection ---
    st.subheader("🌲 Isolation Forest (Unsupervised Anomaly Detection)")  # Subheading for Isolation Forest
    contamination = st.slider("Expected % of anomalies", 0.01, 0.10, 0.02, step=0.01)  # Slider to choose contamination rate
    iso = IsolationForest(contamination=contamination, random_state=42)  # Initialize Isolation Forest
    df['anomaly_score'] = iso.fit_predict(X_scaled)  # Fit model and get predictions (-1 = anomaly)
    df['anomaly_flag'] = (df['anomaly_score'] == -1).astype(int)  # Convert to binary flag (1 = anomaly)

    # --- Show anomaly count ---
    st.write(f"🔎 Detected Anomalies: {df['anomaly_flag'].sum()} out of {len(df)} rows")  # Display number of anomalies

    # --- Plot anomaly distribution ---
    fig, ax = plt.subplots()  # Create matplotlib figure
    sns.countplot(data=df, x='anomaly_flag', ax=ax)  # Bar plot of anomaly counts
    ax.set_title("Anomaly Distribution (Isolation Forest)")  # Set plot title
    ax.set_xlabel("Anomaly (1 = Fraud/Outlier)")  # X-axis label
    st.pyplot(fig)  # Show plot in Streamlit

    # --- Optional: Supervised model (Random Forest) if 'is_fraud' label exists ---
    if 'is_fraud' in df.columns:
        st.subheader("🧠 Random Forest (Supervised - Labeled Fraud)")  # Subheading for supervised learning
        clf = RandomForestClassifier(n_estimators=100, random_state=42)  # Initialize Random Forest
        clf.fit(X_scaled, df['is_fraud'])  # Train model on scaled features and labels
        df['predicted_fraud'] = clf.predict(X_scaled)  # Predict fraud labels

        # --- Show accuracy and classification report ---
        acc = accuracy_score(df['is_fraud'], df['predicted_fraud'])  # Calculate accuracy
        st.success(f"✅ Accuracy: {acc*100:.2f}%")  # Display accuracy
        st.text("Classification Report:")  # Display label
        st.text(classification_report(df['is_fraud'], df['predicted_fraud']))  # Show detailed report

        # --- Plot predicted fraud counts ---
        fig2, ax2 = plt.subplots()  # Create another figure
        sns.countplot(data=df, x='predicted_fraud', ax=ax2)  # Plot predicted fraud distribution
        ax2.set_title("Predicted Fraud Distribution (Random Forest)")  # Title
        ax2.set_xlabel("Predicted Fraud (1 = Fraud)")  # Label
        st.pyplot(fig2)  # Display plot

    else:
        st.info("📌 No 'is_fraud' label found — skipping supervised model.")  # Inform user if label missing

    # --- Download button for results ---
    csv = df.to_csv(index=False).encode('utf-8')  # Convert DataFrame to CSV bytes
    st.download_button("⬇️ Download Result with Anomaly Labels", csv, file_name="fraud_detection_results.csv", mime='text/csv')  # Download button

else:
    st.info("📁 Please upload a CSV file with transaction data to begin.")  # Prompt user to upload file
