import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

st.title("Transaction Volume Forecasting Using Date Ordinal")   # Display app title

# File uploader widget to upload CSV with required columns
uploaded_file = st.file_uploader("Upload CSV with 'transaction_date' and 'transaction_amount' columns", type=["csv"])

# Check if a file is uploaded
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)    # Read CSV into DataFrame
    
    # Convert 'transaction_date' column to datetime format
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    
    # Group by date (ignoring time) and sum 'transaction_amount' per day
    daily_df = df.groupby(df['transaction_date'].dt.date).agg({'transaction_amount':'sum'}).reset_index()
    
    # Rename columns for clarity
    daily_df.rename(columns={'transaction_date':'date', 'transaction_amount':'total_amount'}, inplace=True)
    
    # Create numeric feature from date using ordinal (number of days since year 1)
    daily_df['date_ordinal'] = pd.to_datetime(daily_df['date']).map(pd.Timestamp.toordinal)
    
    st.write("Aggregated daily transaction amounts:")   # Show aggregated daily amounts table
    st.dataframe(daily_df.head())                        # Display first few rows
    
    # Prepare input (X) and output (y) for regression model
    X = daily_df[['date_ordinal']]
    y = daily_df['total_amount']
    
    # Split data into training and testing sets without shuffling (time series)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Initialize and train linear regression model on training data
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predict transaction amounts for test dates
    y_pred = model.predict(X_test)
    
    # Plot actual vs predicted transaction amounts
    plt.figure(figsize=(10,6))
    plt.plot(daily_df['date'][X_train.index], y_train, label='Train Actual')      # Training actual values
    plt.plot(daily_df['date'][X_test.index], y_test, label='Test Actual')         # Testing actual values
    plt.plot(daily_df['date'][X_test.index], y_pred, label='Test Predicted', linestyle='--')  # Predicted values
    plt.xlabel('Date')
    plt.ylabel('Total Transaction Amount')
    plt.title('Transaction Volume Forecasting')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)                  # Display plot in Streamlit app
    
    # Section to forecast next 7 days after last date in dataset
    st.subheader("Forecast Next 7 Days")
    
    # Get ordinal value of last date in historical data
    last_ordinal = daily_df['date_ordinal'].max()
    
    # Create ordinal values for next 7 days
    future_ordinals = np.array([last_ordinal + i for i in range(1, 8)]).reshape(-1,1)
    
    # Predict transaction amounts for future dates
    future_preds = model.predict(future_ordinals)
    
    # Convert ordinal numbers back to datetime
    future_dates = pd.to_datetime([pd.Timestamp.fromordinal(int(x)) for x in future_ordinals.flatten()])
    
    # Create DataFrame for forecasted results
    forecast_df = pd.DataFrame({'date': future_dates, 'forecasted_amount': future_preds})
    
    st.dataframe(forecast_df)     # Display forecast table
    
    # Plot historical data and forecast
    plt.figure(figsize=(10,5))
    plt.plot(daily_df['date'], daily_df['total_amount'], label='Historical')     # Historical transaction amounts
    plt.plot(forecast_df['date'], forecast_df['forecasted_amount'], label='Forecast', marker='o')  # Forecasted amounts
    plt.xlabel('Date')
    plt.ylabel('Transaction Amount')
    plt.title('Historical and Forecasted Transaction Volumes')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)                # Display plot in app

else:
    st.info("Please upload a CSV file to start forecasting.")   # Show info if no file uploaded
    
st.success("App Ready")