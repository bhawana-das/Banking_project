import streamlit as st
from streamlit_option_menu import option_menu

# Import each use case app as a module
import high_spending_app1
import transaction_channel_app2
import transaction_volume_app3
import customer_segmentation_app4
import fraud_detection_app5

st.set_page_config(page_title="Banking Project", layout="wide")

def main():
    st.title("🏦 Bank Transaction Analysis Project")
    st.write("Welcome! Please select a use case from the sidebar.")

    with st.sidebar:
        selected = option_menu(
            menu_title="Main Menu",
            options=[
                "Use Case 1 - High Spending Prediction",
                "Use Case 2 - Transaction Channel Prediction",
                "Use Case 3 - Transaction Volume Forecasting",
                "Use Case 4 - Customer Segmentation",
                "Use Case 5 - Fraud Detection"
            ],
            default_index=0,
        )

    if selected == "Use Case 1 - High Spending Prediction":
        high_spending_app1.run()

    elif selected == "Use Case 2 - Transaction Channel Prediction":
        transaction_channel_app2.run()

    elif selected == "Use Case 3 - Transaction Volume Forecasting":
        transaction_volume_app3.run()

    elif selected == "Use Case 4 - Customer Segmentation":
        customer_segmentation_app4.run()

    elif selected == "Use Case 5 - Fraud Detection":
        fraud_detection_app5.run()

if __name__ == "__main__":
    main()
