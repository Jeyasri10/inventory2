# Import necessary libraries
import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error as mae
from sklearn.model_selection import train_test_split

# Function to load the model and data (replace with actual trained model)
def load_model():
    # Load your dataset (adjust the path)
    df = pd.read_csv('retail_store_inventory.csv')  # Example path
    
    # Feature engineering
    df['Discounted Price'] = df['Price'] * (1 - df['Discount'] / 100)
    df['Price Difference'] = df['Price'] - df['Competitor Pricing']
    df['Stock to Order Ratio'] = df['Inventory Level'] / (df['Units Ordered'] + 1)
    df['Forecast Accuracy'] = abs(df['Demand Forecast'] - df['Units Sold']) / (df['Units Sold'] + 1)
    
    # Define feature columns for prediction
    feature_cols = ['Price', 'Discount', 'Demand Forecast', 'Competitor Pricing',
                    'Discounted Price', 'Price Difference', 'Stock to Order Ratio', 'Forecast Accuracy']
    
    X = df[feature_cols]
    y = df['Units Sold']
    
    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.05, random_state=22)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Train models (use your actual trained model here)
    model = LinearRegression()  # Replace with your actual model
    model.fit(X_train_scaled, y_train)
    
    return model, scaler, X_train, X_val, y_train, y_val

# Load models and data
model, scaler, X_train, X_val, y_train, y_val = load_model()

# Streamlit app interface
st.title("Retail Sales Prediction")

# Input fields for manual prediction
price = st.number_input("Enter Price", min_value=0, value=100, step=1)
discount = st.number_input("Enter Discount (%)", min_value=0, value=10, step=1)
demand_forecast = st.number_input("Enter Demand Forecast", min_value=0, value=100, step=1)
competitor_price = st.number_input("Enter Competitor Price", min_value=0, value=90, step=1)

# Prepare features for prediction (same as in your model)
input_data = {
    'Price': price,
    'Discount': discount,
    'Demand Forecast': demand_forecast,
    'Competitor Pricing': competitor_price,
}

# Additional feature engineering
input_df = pd.DataFrame([input_data])
input_df['Discounted Price'] = input_df['Price'] * (1 - input_df['Discount'] / 100)
input_df['Price Difference'] = input_df['Price'] - input_df['Competitor Pricing']
input_df['Stock to Order Ratio'] = input_df['Price'] / (input_df['Demand Forecast'] + 1)  # Example
input_df['Forecast Accuracy'] = abs(input_df['Demand Forecast'] - input_df['Price']) / (input_df['Price'] + 1)  # Example

# Predict using the trained model
scaled_input = scaler.transform(input_df[['Price', 'Discount', 'Demand Forecast', 'Competitor Pricing', 
                                          'Discounted Price', 'Price Difference', 'Stock to Order Ratio', 'Forecast Accuracy']])
predicted_units = model.predict(scaled_input)

# Display prediction result
st.write(f"Predicted Units Sold: {predicted_units[0]:.2f}")

# Optionally, display the training and validation error
train_preds = model.predict(scaler.transform(X_train))
val_preds = model.predict(scaler.transform(X_val))

train_error = mae(y_train, train_preds)
val_error = mae(y_val, val_preds)

st.write(f"Training Error (MAE): {train_error:.4f}")
st.write(f"Validation Error (MAE): {val_error:.4f}")
