# File: transaction_channel_app.py

# Import the Streamlit library to create the web app
import streamlit as st

# Import pandas for handling tabular data
import pandas as pd

# Import numpy for numerical operations
import numpy as np

# Import function to split dataset into training and testing sets
from sklearn.model_selection import train_test_split

# Import standard scaler to normalize features
from sklearn.preprocessing import StandardScaler

# Import Decision Tree classifier from sklearn
from sklearn.tree import DecisionTreeClassifier

# Import K-Nearest Neighbors classifier from sklearn
from sklearn.neighbors import KNeighborsClassifier

# Import matplotlib for plotting graphs
import matplotlib.pyplot as plt

# Import seaborn for advanced visualizations
import seaborn as sns

def run():
    st.title("Transaction channel Prediction - Use Case 2")
    st.write("Welcome to Use Case 2 App!")


# Set the title and icon of the Streamlit web app
st.set_page_config(page_title="Transaction Channel Predictor", page_icon="🔍")

# Set the main title of the web app
st.title("🔍 Transaction Channel Prediction")

# Display a description under the title
st.write("Classify whether a transaction was made via ATM, QR, Mobile, or Online based on behavioral features.")


# Section for uploading a CSV file
st.subheader("1. Upload CSV with Transaction Data")

# Streamlit widget to allow file upload; only allows CSV files
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])


# Check if a file is uploaded
if uploaded_file is not None:
    # Read the uploaded CSV file into a pandas DataFrame
    df = pd.read_csv(uploaded_file)

    # Show success message after successful upload
    st.success("✅ File successfully uploaded!")

    # Show the first 5 rows of the uploaded data
    st.write("Preview:")
    st.dataframe(df.head())

    # Define the columns required to proceed with training
    required_cols = ['transaction_amount', 'account_balance', 'hour_of_day', 'channel']

    # Check if all required columns are present in the file
    if not all(col in df.columns for col in required_cols):
        # Stop the app if columns are missing
        st.stop()

    # Create a dictionary to map channel labels to numeric values
    channel_map = {label: idx for idx, label in enumerate(df['channel'].unique())}

    # Add a new column to the DataFrame for encoded channel values
    df['channel_encoded'] = df['channel'].map(channel_map)

    # Define feature columns (independent variables)
    X = df[['transaction_amount', 'account_balance', 'hour_of_day']]

    # Define the target column (dependent variable)
    y = df['channel_encoded']

    # Create a StandardScaler object to normalize the features
    scaler = StandardScaler()

    # Apply the scaler to the feature data
    X_scaled = scaler.fit_transform(X)

    # Split the dataset into training and testing sets (80% train, 20% test)
    x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


    # Section to choose the classification algorithm
    st.subheader("2. Choose Classification Algorithm")

    # Dropdown menu to select the machine learning model
    algorithm = st.selectbox("Select Model", ["Decision Tree", "K-Nearest Neighbors"])


    # Based on user selection, create the corresponding model
    if algorithm == "Decision Tree":
        model = DecisionTreeClassifier()
    else:
        model = KNeighborsClassifier(n_neighbors=5)

    # Train the selected model using training data
    model.fit(x_train, y_train)

    # Calculate accuracy on test data
    acc = model.score(x_test, y_test)

    # Display model accuracy
    st.info(f"🔎 Model trained with accuracy: {acc:.2%}")


    # Section to take user input and predict transaction channel
    st.subheader("3. Predict Transaction Channel")

    # Slider to input transaction amount
    txn_amt = st.slider("Transaction Amount (NPR)", 100, 100000, 5000)

    # Slider to input account balance
    acc_bal = st.slider("Account Balance (NPR)", 100, 100000, 10000)

    # Slider to input hour of transaction
    hour = st.slider("Hour of Transaction (0–23)", 0, 23, 13)

    # Create a DataFrame using user inputs
    input_data = pd.DataFrame([[txn_amt, acc_bal, hour]], columns=X.columns)

    # Scale the user input using the same scaler
    input_scaled = scaler.transform(input_data)

    # Predict the channel using the trained model
    pred = model.predict(input_scaled)[0]

    # Decode the numeric prediction back to original channel label
    predicted_channel = list(channel_map.keys())[list(channel_map.values()).index(pred)]

    # Show the predicted channel to the user
    st.success(f"📲 Predicted Channel: {predicted_channel}")


    # Section to visualize transaction channel distribution
    st.subheader("4. Channel Distribution")

    # Create a plot for the number of transactions per channel
    fig, ax = plt.subplots()
    sns.countplot(data=df, x='channel', ax=ax)

    # Set plot title
    ax.set_title("Transaction Count by Channel")

    # Display the plot in the Streamlit app
    st.pyplot(fig)

# If no file is uploaded yet, display a message
else:
    st.info("📁 Upload a CSV file to begin prediction.")


# Show a small footer note
st.caption("Created using Streamlit and Scikit-learn")

# Display a success message at the bottom
st.success("App ready")
