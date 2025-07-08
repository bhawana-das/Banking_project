import streamlit as st
from streamlit_option_menu import option_menu

import high_spending_app1
import transaction_channel_app2
import transaction_volume_app3
import customer_segmentation_app4
import fraud_detection_app5

st.set_page_config(page_title="Banking Project", layout="wide")

def main():
    st.title("Banking Project Dashboard")
    
    # Sidebar मा menu बनाउने
    with st.sidebar:
        selected = option_menu(
            menu_title="Main Menu",
            options=["Use Case 1", "Use Case 2", "Use Case 3", "Use Case 4", "Use Case 5"],
            default_index=0,
        )
    
    # user को चयन अनुसार सम्बन्धित app को run function call गर्ने
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

if __name__ == "__main__":
    main()
