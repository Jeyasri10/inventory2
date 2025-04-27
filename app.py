import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Example of loading a pre-trained model
# Assuming the model is saved as 'model.pkl'
# model = joblib.load('model.pkl')

# Create the user input interface
st.title("Manual Model Input Test")

# Collect user input using streamlit widgets
price = st.number_input("Enter Price", min_value=0.0, step=0.1)
discount = st.number_input("Enter Discount Percentage", min_value=0.0, step=0.1)
demand_forecast = st.number_input("Enter Demand Forecast", min_value=0.0, step=0.1)
competitor_pricing = st.number_input("Enter Competitor Pricing", min_value=0.0, step=0.1)
inventory_level = st.number_input("Enter Inventory Level", min_value=0.0, step=1)
units_ordered = st.number_input("Enter Units Ordered", min_value=0, step=1)

# When the user clicks "Predict"
if st.button("Predict"):
    # Prepare the input data for the model
    input_data = pd.DataFrame({
        'Price': [price],
        'Discount': [discount],
        'Demand Forecast': [demand_forecast],
        'Competitor Pricing': [competitor_pricing],
        'Inventory Level': [inventory_level],
        'Units Ordered': [units_ordered]
    })

    # Scale input data if necessary (if the model expects scaled data)
    scaler = StandardScaler()
    input_data_scaled = scaler.fit_transform(input_data)

    # Predict with your pre-trained model
    # prediction = model.predict(input_data_scaled)
    prediction = [10]  # Placeholder prediction
    
    st.write(f"Prediction: {prediction[0]}")

    # Optional: Display Training and Validation Error (e.g., from previous model evaluation)
    train_error = 7.45  # Example MAE
    val_error = 7.39  # Example MAE
    st.write(f"Training Error (MAE): {train_error}")
    st.write(f"Validation Error (MAE): {val_error}")
