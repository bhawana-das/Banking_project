import streamlit as st
from streamlit_option_menu import option_menu
import high_spending_app1, transaction_channel_app2, transaction_volume_app3, customer_segmentation_app4, fraud_detection_app5

st.set_page_config(page_title="Banking Project", layout="wide")

def run():
    st.title("Banking Project")
    st.write("Welcome to Bank transaction analysis app!")

# Navigation
with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=["Use Case 1", "Use Case 2", "Use Case 3", "Use Case 4", "Use Case 5"],
        default_index=0,
    )

# Render selected page
if selected == "Use Case 1":
    high_spending_app1.run()
elif selected == "Use Case 2":
    transaction_channel_app2.run()
elif selected == "Use Case 3":
    transaction_volume_app3.run()
elif selected == "Use Case 4":
    customer_segmentation_app4.run()
elif selected == "Use Case 5":
    fraud_detection_app5.run()
