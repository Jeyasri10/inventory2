import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('model.pkl')

# Title
st.title('Retail Store Demand Prediction')

# Create input fields manually
price = st.number_input('Price', min_value=0.0, format="%.2f")
discount = st.number_input('Discount (%)', min_value=0.0, max_value=100.0, format="%.2f")
demand_forecast = st.number_input('Demand Forecast', min_value=0.0, format="%.2f")
competitor_pricing = st.number_input('Competitor Pricing', min_value=0.0, format="%.2f")
holiday_promotion = st.selectbox('Holiday/Promotion', [0, 1])  # 0: No, 1: Yes
year = st.number_input('Year', min_value=2000, max_value=2100, step=1)
month = st.number_input('Month', min_value=1, max_value=12, step=1)
day = st.number_input('Day', min_value=1, max_value=31, step=1)

# Derived features
discounted_price = price * (1 - discount / 100)
price_difference = price - competitor_pricing
stock_to_order_ratio = 1  # Set dummy value (you can adjust as needed)
forecast_accuracy = 0.1   # Set dummy value (adjust as needed)

# When button is clicked
if st.button('Predict Demand Class'):
    input_data = np.array([
        price, discount, demand_forecast, competitor_pricing,
        discounted_price, price_difference, stock_to_order_ratio,
        forecast_accuracy, holiday_promotion, year, month, day
    ]).reshape(1, -1)

    # Predict
    prediction = model.predict(input_data)[0]

    # Show prediction
    if prediction == 0:
        st.success('Prediction: Low Demand')
    elif prediction == 1:
        st.success('Prediction: Medium Demand')
    else:
        st.success('Prediction: High Demand')
