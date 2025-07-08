# File: transaction_channel_app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns

def run():
    st.title("🔍 Transaction Channel Prediction - Use Case 2")
    st.write("Predict whether a transaction was made via ATM, QR, Mobile, or Online based on behavior.")

    # --- Upload CSV ---
    uploaded_file = st.file_uploader("📂 Upload CSV with Transaction Data", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File successfully uploaded!")
        st.subheader("🔍 Preview:")
        st.dataframe(df.head())

        required_cols = ['transaction_amount', 'account_balance', 'hour_of_day', 'channel']
        if not all(col in df.columns for col in required_cols):
            st.error(f"❌ Required columns missing: {required_cols}")
            return

        # Encode labels
        channel_map = {label: idx for idx, label in enumerate(df['channel'].unique())}
        df['channel_encoded'] = df['channel'].map(channel_map)

        # Features and Target
        X = df[['transaction_amount', 'account_balance', 'hour_of_day']]
        y = df['channel_encoded']

        # Scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train-Test Split
        x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # --- Select Model ---
        st.subheader("⚙️ Select Classification Model")
        model_name = st.selectbox("Choose Model", ["Decision Tree", "K-Nearest Neighbors"])

        if model_name == "Decision Tree":
            model = DecisionTreeClassifier()
        else:
            model = KNeighborsClassifier(n_neighbors=5)

        model.fit(x_train, y_train)
        accuracy = model.score(x_test, y_test)
        st.info(f"✅ Model Accuracy: {accuracy:.2%}")

        # --- Predict from User Input ---
        st.subheader("📲 Predict Transaction Channel")

        txn_amt = st.slider("Transaction Amount (NPR)", 100, 100000, 5000)
        acc_bal = st.slider("Account Balance (NPR)", 100, 100000, 10000)
        hour = st.slider("Hour of Day (0–23)", 0, 23, 12)

        input_df = pd.DataFrame([[txn_amt, acc_bal, hour]], columns=X.columns)
        input_scaled = scaler.transform(input_df)
        pred = model.predict(input_scaled)[0]
        predicted_channel = list(channel_map.keys())[list(channel_map.values()).index(pred)]

        st.success(f"📲 Predicted Channel: {predicted_channel}")

        # --- Visualization ---
        st.subheader("📊 Channel Distribution")
        fig, ax = plt.subplots()
        sns.countplot(data=df, x='channel', ax=ax)
        ax.set_title("Transaction Count by Channel")
        st.pyplot(fig)

    else:
        st.info("📁 Please upload a CSV file to proceed.")
