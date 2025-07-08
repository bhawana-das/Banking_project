# File: fraud_detection_app5.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

def run():
    st.title("🚨 Fraud / Outlier Detection - Use Case 5")
    st.markdown("Detect suspicious transactions using machine learning models.")

    # File upload
    uploaded_file = st.file_uploader("📂 Upload your transaction dataset (CSV)", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.subheader("📄 Data Preview")
        st.dataframe(df.head())

        # Feature selection
        st.subheader("🧮 Select Numeric Features for Detection")
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        selected_features = st.multiselect("Select at least two features", numeric_cols, default=numeric_cols)

        if len(selected_features) < 2:
            st.warning("⚠️ Please select at least two numeric features to continue.")
            st.stop()

        X = df[selected_features].dropna()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # --- Isolation Forest ---
        st.subheader("🌲 Isolation Forest (Unsupervised)")
        contamination = st.slider("Expected Anomaly Ratio", 0.01, 0.10, 0.02, step=0.01)
        iso_model = IsolationForest(contamination=contamination, random_state=42)
        df['anomaly_score'] = iso_model.fit_predict(X_scaled)
        df['anomaly_flag'] = (df['anomaly_score'] == -1).astype(int)

        st.write(f"🔎 Total Anomalies Detected: {df['anomaly_flag'].sum()} out of {len(df)}")

        # Anomaly plot
        fig, ax = plt.subplots()
        sns.countplot(data=df, x='anomaly_flag', ax=ax)
        ax.set_title("Anomaly Flag Distribution")
        ax.set_xlabel("Anomaly Flag (1 = Outlier)")
        st.pyplot(fig)

        # --- Random Forest if labels exist ---
        if 'is_fraud' in df.columns:
            st.subheader("🧠 Random Forest (Supervised)")
            rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_model.fit(X_scaled, df['is_fraud'])
            df['predicted_fraud'] = rf_model.predict(X_scaled)

            acc = accuracy_score(df['is_fraud'], df['predicted_fraud'])
            st.success(f"✅ Accuracy: {acc*100:.2f}%")

            st.text("Classification Report:")
            st.text(classification_report(df['is_fraud'], df['predicted_fraud']))

            # Plot prediction
            fig2, ax2 = plt.subplots()
            sns.countplot(data=df, x='predicted_fraud', ax=ax2)
            ax2.set_title("Predicted Fraud Distribution")
            st.pyplot(fig2)
        else:
            st.info("ℹ️ 'is_fraud' label not found. Supervised model skipped.")

        # Download output
        st.subheader("⬇️ Download Results")
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV with Anomaly Flags", csv, file_name="fraud_detection_output.csv", mime='text/csv')

    else:
        st.info("📁 Please upload a CSV file to get started.")
