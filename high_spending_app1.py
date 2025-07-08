# --- Import necessary libraries ---
import streamlit as st  # Web app framework
import pandas as pd     # For handling tabular data
import numpy as np      # For numerical operations
import seaborn as sns   # For advanced visualizations
from sklearn.linear_model import LogisticRegression  # Logistic Regression model
from sklearn.ensemble import RandomForestClassifier  # Random Forest model
from sklearn.model_selection import train_test_split  # For splitting dataset into training and testing
from sklearn.preprocessing import StandardScaler  # For feature scaling
import matplotlib.pyplot as plt  # For plotting
import pickle  # For saving/loading models (not used in this code)

# --- Configure the Streamlit app ---
st.set_page_config(page_title="Banking Transaction Analysis", layout="wide")  # Set page title and layout to wide

# --- App Title and Description ---
st.title("📚 Bank Transaction Analysis & Prediction App")  # Main title
st.set_page_config(page_title="Top Spender Prediction", page_icon="💰")  # Override page title and icon
st.title("💰 High-Spending Customer Prediction")  # Updated title
st.write("Classify whether a customer belongs to the top 10% based on their transaction behavior.")  # App description

# --- Step 1: Load CSV Data from User ---
st.subheader("1. Upload Real Transaction Data (CSV)")  # Subsection title
uploaded_file = st.file_uploader("Upload your transaction CSV file", type=["csv"])  # File upload widget

# --- If CSV is uploaded ---
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)  # Read CSV into a DataFrame
    st.success("✅ CSV data loaded successfully!")  # Show success message
    st.write("Preview of data:")  # Show preview message
    st.dataframe(df.head())  # Display first few rows of the data

    # --- Ensure required columns exist ---
    required_cols = ['transaction_amount', 'balance_after_transaction']  # Required columns for analysis
    if not all(col in df.columns for col in required_cols):  # Check if all required columns are present
        st.error(f"❌ CSV must contain columns: {required_cols}")  # Show error if columns missing
        st.stop()  # Stop execution

    # --- Automatically create target column if not present ---
    if 'is_top_spender' not in df.columns:  # If target column is missing
        threshold = np.percentile(df['transaction_amount'], 90)  # Calculate 90th percentile
        df['is_top_spender'] = (df['transaction_amount'] > threshold).astype(int)  # Create binary target column
else:
    st.info("📂 Please upload a CSV file to continue.")  # Message when no file is uploaded
    st.stop()  # Stop execution

# --- Step 2: Data Preprocessing ---
X = df[['transaction_amount', 'balance_after_transaction']]  # Select input features
y = df['is_top_spender']  # Target variable
scaler = StandardScaler()  # Initialize standard scaler
X_scaled = scaler.fit_transform(X)  # Apply scaling to features

# --- Step 3: Train Classification Models ---
x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)  # Split dataset
model_rf = RandomForestClassifier(random_state=42)  # Initialize Random Forest model
model_lr = LogisticRegression()  # Initialize Logistic Regression model
model_lr.fit(x_train, y_train)  # Train Logistic Regression
model_rf.fit(x_train, y_train)  # Train Random Forest

# --- Step 4: Accept User Input for Prediction ---
st.subheader("2. Input Features")  # Subsection title
transaction_amount = st.slider("Transaction Amount (in NPR)", 1000, 100000, 25000, step=1000)  # Slider for transaction
balance_after_txn = st.slider("Balance After Transaction (in NPR)", 100, 10000, 3000, step=100)  # Slider for balance

# Create a DataFrame from user input
input_data = pd.DataFrame([[transaction_amount, balance_after_txn]],
                          columns=['transaction_amount', 'balance_after_transaction'])
input_scaled = scaler.transform(input_data)  # Scale input data

# --- Step 5: Scatterplot for Visual Insight ---
st.subheader("4. Visual Insight (Enhanced Plot)")  # Subsection title
fig, ax = plt.subplots(figsize=(8, 6))  # Create a matplotlib figure
sns.scatterplot(  # Plot scatter of existing data
    data=df,
    x="transaction_amount",
    y="balance_after_transaction",
    hue="is_top_spender",  # Color based on label
    palette={0: "blue", 1: "red"},  # Blue for non-top spenders, red for top
    alpha=0.6,  # Transparency
    ax=ax
)
# Highlight the user input point with a black 'X'
ax.scatter(transaction_amount, balance_after_txn, c='black', s=200, marker='X', label='Your Input')
ax.set_title("Customer Transaction Visualization")  # Set plot title
ax.legend(title='Top Spender', loc='upper right')  # Show legend
st.pyplot(fig)  # Display plot in Streamlit

# --- Step 6: Perform Prediction using Trained Models ---
prediction = model_rf.predict(input_scaled)[0]  # Predict with Random Forest
pred_prob = model_rf.predict_proba(input_scaled)[0][1]  # Probability from Random Forest
prediction_lr = model_lr.predict(input_data)[0]  # Predict with Logistic Regression
pred_prob_lr = model_lr.predict_proba(input_data)[0][1]  # Probability from Logistic Regression

st.subheader("3. Prediction Result")  # Subsection title
if prediction_lr == 1:  # If predicted as top spender
    st.success(f"✅ Prediction: This customer is likely a High-Spender! (Confidence: {pred_prob_lr:.2%})")  # Show result

# --- Step 7: Heatmap Visualization of Top Spender Density ---
st.subheader("4. Heatmap Insight: Top Spender Density")  # Subsection title

# Create bins for amount and balance
df['amount_bin'] = pd.cut(df['transaction_amount'], bins=10)  # Bin transaction amount
df['balance_bin'] = pd.cut(df['balance_after_transaction'], bins=10)  # Bin balance after transaction

# Calculate the average top spender ratio per bin
heatmap_data = df.groupby(['amount_bin', 'balance_bin'])['is_top_spender'].mean().unstack().fillna(0)

# Convert bin labels to string for axis formatting
heatmap_data.index = heatmap_data.index.astype(str)
heatmap_data.columns = heatmap_data.columns.astype(str)

# Plot the heatmap using seaborn
fig, ax = plt.subplots(figsize=(10, 6))  # Create a larger figure
sns.heatmap(heatmap_data, cmap='YlOrRd', annot=True, fmt=".2f", linewidths=0.5, ax=ax)  # Create heatmap
ax.set_xlabel("Balance After Transaction (Binned)")  # X-axis label
ax.set_ylabel("Transaction Amount (Binned)")  # Y-axis label
ax.set_title("🗺️ Heatmap of Top-Spender Probability per Bin")  # Title

st.pyplot(fig)  # Show plot in Streamlit

# --- Final Footer ---
st.caption("Built with ❤️ using Streamlit and Scikit-learn")  # Footer message
st.success("✅ App Ready")  # Final success message
