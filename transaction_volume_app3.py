# File: transaction_volume_app3.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def run():
    st.title("📈 Transaction Volume Forecasting - Use Case 3")
    st.write("Predict future transaction volume based on past transaction trends using Linear Regression.")

    # --- Step 1: Upload CSV File ---
    uploaded_file = st.file_uploader("📂 Upload CSV with 'transaction_date' and 'transaction_amount'", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # --- Step 2: Data Preprocessing ---
        df['transaction_date'] = pd.to_datetime(df['transaction_date'])
        daily_df = df.groupby(df['transaction_date'].dt.date)['transaction_amount'].sum().reset_index()
        daily_df.rename(columns={'transaction_date': 'date', 'transaction_amount': 'total_amount'}, inplace=True)
        daily_df['date_ordinal'] = pd.to_datetime(daily_df['date']).map(pd.Timestamp.toordinal)

        st.subheader("📅 Daily Aggregated Transactions")
        st.dataframe(daily_df.head())

        X = daily_df[['date_ordinal']]
        y = daily_df['total_amount']

        # --- Step 3: Train/Test Split ---
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        # --- Step 4: Train Model ---
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # --- Step 5: Plot Actual vs Predicted ---
        plt.figure(figsize=(10, 6))
        plt.plot(daily_df['date'][X_train.index], y_train, label='Train Actual')
        plt.plot(daily_df['date'][X_test.index], y_test, label='Test Actual')
        plt.plot(daily_df['date'][X_test.index], y_pred, label='Test Predicted', linestyle='--')
        plt.xlabel("Date")
        plt.ylabel("Transaction Amount")
        plt.title("Transaction Volume Forecasting")
        plt.legend()
        plt.grid(True)
        st.pyplot(plt)

        # --- Step 6: Forecast Future 7 Days ---
        st.subheader("🔮 Forecast Next 7 Days")
        last_ordinal = daily_df['date_ordinal'].max()
        future_ordinals = np.array([last_ordinal + i for i in range(1, 8)]).reshape(-1, 1)
        future_preds = model.predict(future_ordinals)
        future_dates = pd.to_datetime([pd.Timestamp.fromordinal(int(x)) for x in future_ordinals.flatten()])

        forecast_df = pd.DataFrame({'date': future_dates, 'forecasted_amount': future_preds})
        st.dataframe(forecast_df)

        # --- Step 7: Plot Forecast ---
        plt.figure(figsize=(10, 5))
        plt.plot(daily_df['date'], daily_df['total_amount'], label='Historical')
        plt.plot(forecast_df['date'], forecast_df['forecasted_amount'], label='Forecast', marker='o')
        plt.xlabel("Date")
        plt.ylabel("Transaction Amount")
        plt.title("📉 Historical and Forecasted Transactions")
        plt.legend()
        plt.grid(True)
        st.pyplot(plt)

    else:
        st.info("📁 Please upload a CSV file to start forecasting.")
