# --- Import necessary libraries ---
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def run():
    st.title("💰 High-Spending Customer Prediction - Use Case 1")
    st.write("Classify whether a customer belongs to the top 10% based on their transaction behavior.")

    # --- Step 1: Upload CSV Data from User ---
    uploaded_file = st.file_uploader("📂 Upload your transaction CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("✅ CSV data loaded successfully!")
        st.write("Preview of data:")
        st.dataframe(df.head())

        required_cols = ['transaction_amount', 'balance_after_transaction']
        if not all(col in df.columns for col in required_cols):
            st.error(f"❌ CSV must contain columns: {required_cols}")
            return

        if 'is_top_spender' not in df.columns:
            threshold = np.percentile(df['transaction_amount'], 90)
            df['is_top_spender'] = (df['transaction_amount'] > threshold).astype(int)

        # --- Step 2: Data Preprocessing ---
        X = df[['transaction_amount', 'balance_after_transaction']]
        y = df['is_top_spender']
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        model_rf = RandomForestClassifier(random_state=42)
        model_lr = LogisticRegression()
        model_lr.fit(x_train, y_train)
        model_rf.fit(x_train, y_train)

        # --- Step 3: Input Features ---
        st.subheader("🧾 Input Features")
        transaction_amount = st.slider("Transaction Amount (NPR)", 1000, 100000, 25000, step=1000)
        balance_after_txn = st.slider("Balance After Transaction (NPR)", 100, 10000, 3000, step=100)

        input_data = pd.DataFrame([[transaction_amount, balance_after_txn]],
                                  columns=['transaction_amount', 'balance_after_transaction'])
        input_scaled = scaler.transform(input_data)

        # --- Step 4: Prediction ---
        prediction_lr = model_lr.predict(input_scaled)[0]
        pred_prob_lr = model_lr.predict_proba(input_scaled)[0][1]

        st.subheader("📊 Prediction Result")
        if prediction_lr == 1:
            st.success(f"✅ Likely a High-Spender! (Confidence: {pred_prob_lr:.2%})")
        else:
            st.info(f"ℹ️ Likely not a High-Spender (Confidence: {1 - pred_prob_lr:.2%})")

        # --- Step 5: Visual Insight ---
        st.subheader("📈 Transaction Scatter Plot")
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x="transaction_amount", y="balance_after_transaction",
                        hue="is_top_spender", palette={0: "blue", 1: "red"}, alpha=0.6, ax=ax)
        ax.scatter(transaction_amount, balance_after_txn, c='black', s=200, marker='X', label='Your Input')
        ax.set_title("Customer Transaction Visualization")
        ax.legend(title='Top Spender')
        st.pyplot(fig)

        # --- Step 6: Heatmap ---
        st.subheader("🗺️ Heatmap of Top-Spender Probability")
        df['amount_bin'] = pd.cut(df['transaction_amount'], bins=10)
        df['balance_bin'] = pd.cut(df['balance_after_transaction'], bins=10)
        heatmap_data = df.groupby(['amount_bin', 'balance_bin'])['is_top_spender'].mean().unstack().fillna(0)
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.heatmap(heatmap_data, cmap='YlOrRd', annot=True, fmt=".2f", linewidths=0.5, ax=ax2)
        ax2.set_xlabel("Balance After Transaction (Binned)")
        ax2.set_ylabel("Transaction Amount (Binned)")
        ax2.set_title("🗺️ Heatmap of Top-Spender Probability per Bin")
        st.pyplot(fig2)

        # --- Final Footer ---
        st.success("✅ Use Case 1 ready")
    else:
        st.info("📁 Please upload a CSV file to proceed.")
