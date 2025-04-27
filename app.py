import streamlit as st
import requests

# URL for the Flask API
API_URL = "http://127.0.0.1:5000/predict"

# Streamlit UI
st.title("Retail Store Demand Prediction")

# Input fields for manual data input
price = st.number_input("Price", min_value=0, value=100)
discount = st.number_input("Discount (%)", min_value=0, value=10)
demand_forecast = st.number_input("Demand Forecast", min_value=0, value=100)
competitor_pricing = st.number_input("Competitor Pricing", min_value=0, value=90)
inventory_level = st.number_input("Inventory Level", min_value=0, value=500)
units_ordered = st.number_input("Units Ordered", min_value=0, value=50)

# Button to trigger prediction
if st.button("Predict"):
    # Prepare the data to send to the API
    data = {
        "Price": price,
        "Discount": discount,
        "Units Ordered": units_ordered,
        "Demand Forecast": demand_forecast,
        "Competitor Pricing": competitor_pricing,
        "Inventory Level": inventory_level
    }

    try:
        # Send the request to the Flask API
        response = requests.post(API_URL, json=data)
        
        # Display the results
        if response.status_code == 200:
            predictions = response.json()
            st.write("Predicted Demand Class:")
            for model, pred in predictions.items():
                st.write(f"{model}: {pred}")
        else:
            st.error("Error in prediction!")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
