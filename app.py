import streamlit as st
 import pandas as pd
 from sklearn.linear_model import LinearRegression
 from sklearn.linear_model import LinearRegression, Lasso, Ridge
 from sklearn.model_selection import train_test_split
 from sklearn.preprocessing import StandardScaler
 from sklearn.metrics import mean_absolute_error as mae
 
 # Example of loading a pre-trained model
 # Assuming the model is saved as 'model.pkl'
 # model = joblib.load('model.pkl')
 # Load your dataset
 @st.cache
 def load_data():
     df = pd.read_csv('retail_store_inventory.csv')  # Make sure the CSV file is in the correct location
     return df
 
 # Create the user input interface
 st.title("Manual Model Input Test")
 # Step 1: Load Data
 df = load_data()
 
 # Collect user input using streamlit widgets
 price = st.number_input("Enter Price", min_value=0.0, step=0.1)
 discount = st.number_input("Enter Discount Percentage", min_value=0.0, step=0.1)
 demand_forecast = st.number_input("Enter Demand Forecast", min_value=0.0, step=0.1)
 competitor_pricing = st.number_input("Enter Competitor Pricing", min_value=0.0, step=0.1)
 inventory_level = st.number_input("Enter Inventory Level", min_value=0.0, step=1)
 units_ordered = st.number_input("Enter Units Ordered", min_value=0, step=1)
 # Step 2: Feature Engineering and Preprocessing
 df['Date'] = pd.to_datetime(df['Date'])
 df['Year'] = df['Date'].dt.year
 df['Month'] = df['Date'].dt.month
 df['Day'] = df['Date'].dt.day
 
 # When the user clicks "Predict"
 if st.button("Predict"):
     # Prepare the input data for the model
 df['Discounted Price'] = df['Price'] * (1 - df['Discount'] / 100)
 df['Price Difference'] = df['Price'] - df['Competitor Pricing']
 df['Stock to Order Ratio'] = df['Inventory Level'] / (df['Units Ordered'] + 1)
 df['Forecast Accuracy'] = abs(df['Demand Forecast'] - df['Units Sold']) / (df['Units Sold'] + 1)
 
 def classify_units(units):
     if units <= 50:
         return 0  # Low
     elif units <= 150:
         return 1  # Medium
     else:
         return 2  # High
 
 df['Demand Class'] = df['Units Sold'].apply(classify_units)
 
 feature_cols = [
     'Price', 'Discount', 'Demand Forecast', 'Competitor Pricing',
     'Discounted Price', 'Price Difference', 'Stock to Order Ratio',
     'Forecast Accuracy', 'Holiday/Promotion', 'Year', 'Month', 'Day'
 ]
 
 # Step 3: Prepare Features and Target
 X = df[feature_cols]
 y = df['Demand Class']
 
 # Split the data into training and validation sets
 X_train, X_val, Y_train, Y_val = train_test_split(X, y, test_size=0.2, random_state=22)
 
 # Scale numerical features
 numerical_cols = ['Price', 'Discount', 'Demand Forecast', 'Competitor Pricing', 'Discounted Price', 
                   'Stock to Order Ratio', 'Inventory Level', 'Units Ordered']
 scaler = StandardScaler()
 X_train_scaled = scaler.fit_transform(X_train[numerical_cols])
 X_val_scaled = scaler.transform(X_val[numerical_cols])
 
 # Step 4: Train Models
 models = [
     ("Linear Regression", LinearRegression()),
     ("Lasso Regression", Lasso(alpha=0.1)),
     ("Ridge Regression", Ridge(alpha=1.0))
 ]
 
 trained_models = {}
 
 for name, model in models:
     model.fit(X_train_scaled, Y_train)
     trained_models[name] = model
 
 # Step 5: Create a function for prediction
 def predict_demand(price, discount, demand_forecast, competitor_pricing, inventory_level, units_ordered):
     input_data = pd.DataFrame({
         'Price': [price],
         'Discount': [discount],
         'Demand Forecast': [demand_forecast],
         'Competitor Pricing': [competitor_pricing],
         'Inventory Level': [inventory_level],
         'Units Ordered': [units_ordered]
         'Discounted Price': [price * (1 - discount / 100)],
         'Price Difference': [price - competitor_pricing],
         'Stock to Order Ratio': [inventory_level / (units_ordered + 1)],
         'Forecast Accuracy': [abs(demand_forecast - units_ordered) / (units_ordered + 1)],
         'Holiday/Promotion': [0],  # Add default value if applicable
         'Year': [2025],  # Add default value if applicable
         'Month': [5],  # Add default value if applicable
         'Day': [15],  # Add default value if applicable
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
     # Scale the input data using the same scaler used for training data
     input_scaled = scaler.transform(input_data[numerical_cols])
 
     # Predict using the trained models
     predictions = {}
     for name, model in trained_models.items():
         prediction = model.predict(input_scaled)
         predictions[name] = prediction[0]
 
     return predictions
 
 # Step 6: Build Streamlit Interface
 st.title("Retail Store Demand Prediction")
 
 # Create input fields for manual data input
 price = st.number_input("Price", min_value=0, value=100)
 discount = st.number_input("Discount (%)", min_value=0, value=10)
 demand_forecast = st.number_input("Demand Forecast", min_value=0, value=100)
 competitor_pricing = st.number_input("Competitor Pricing", min_value=0, value=90)
 inventory_level = st.number_input("Inventory Level", min_value=0, value=500)
 units_ordered = st.number_input("Units Ordered", min_value=0, value=50)
 
 # Button to trigger prediction
 if st.button("Predict"):
     predictions = predict_demand(price, discount, demand_forecast, competitor_pricing, inventory_level, units_ordered)
 
     st.write(f"Predicted Demand Class for Linear Regression: {predictions['Linear Regression']}")
     st.write(f"Predicted Demand Class for Lasso Regression: {predictions['Lasso Regression']}")
     st.write(f"Predicted Demand Class for Ridge Regression: {predictions['Ridge Regression']}")
 
     # Display training and validation errors
     st.subheader("Model Performance")
 
     for name, model in trained_models.items():
         # Calculate training and validation errors
         train_preds = model.predict(X_train_scaled)
         train_error = mae(Y_train, train_preds)
 
         val_preds = model.predict(X_val_scaled)
         val_error = mae(Y_val, val_preds)
 
         st.write(f'{name}:')
         st.write(f"  Training Error (MAE): {train_error:.4f}")
         st.write(f"  Validation Error (MAE): {val_error:.4f}")
