from flask import Flask, request, jsonify
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as mae

app = Flask(__name__)

# Load the dataset
df = pd.read_csv('retail_store_inventory.csv')  # Make sure the CSV file is in the correct location

# Preprocess data (same as you did for your app.py)
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day

df['Discounted Price'] = df['Price'] * (1 - df['Discount'] / 100)
df['Price Difference'] = df['Price'] - df['Competitor Pricing']
df['Stock to Order Ratio'] = df['Inventory Level'] / (df['Units Ordered'] + 1)
df['Forecast Accuracy'] = abs(df['Demand Forecast'] - df['Units Sold']) / (df['Units Sold'] + 1)

feature_cols = [
    'Price', 'Discount', 'Demand Forecast', 'Competitor Pricing',
    'Discounted Price', 'Price Difference', 'Stock to Order Ratio',
    'Forecast Accuracy', 'Holiday/Promotion', 'Year', 'Month', 'Day'
]

# Prepare Features and Target
X = df[feature_cols]
y = df['Demand Class']

# Split data into training and validation sets
X_train, X_val, Y_train, Y_val = train_test_split(X, y, test_size=0.2, random_state=22)

# Scale numerical features
numerical_cols = ['Price', 'Discount', 'Demand Forecast', 'Competitor Pricing', 'Discounted Price', 
                  'Stock to Order Ratio', 'Inventory Level', 'Units Ordered']
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train[numerical_cols])
X_val_scaled = scaler.transform(X_val[numerical_cols])

# Train the model
model = LinearRegression()
model.fit(X_train_scaled, Y_train)

# Make predictions on training and validation data
train_preds = model.predict(X_train_scaled)
val_preds = model.predict(X_val_scaled)

# Calculate training and validation errors (MAE)
train_error = mae(Y_train, train_preds)
val_error = mae(Y_val, val_preds)

# API route for prediction
@app.route('/', methods=['GET'])
def home():
    return "Welcome to the Inventory Prediction App!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input data from the request
        data = request.json
        price = data['Price']
        discount = data['Discount']
        units_ordered = data['Units Ordered']
        demand_forecast = data['Demand Forecast']
        competitor_pricing = data['Competitor Pricing']
        
        # Prepare the input data for prediction
        input_data = pd.DataFrame({
            'Price': [price],
            'Discount': [discount],
            'Units Ordered': [units_ordered],
            'Demand Forecast': [demand_forecast],
            'Competitor Pricing': [competitor_pricing],
            'Discounted Price': [price * (1 - discount / 100)],
            'Price Difference': [price - competitor_pricing],
            'Stock to Order Ratio': [units_ordered / (units_ordered + 1)],
            'Forecast Accuracy': [abs(demand_forecast - units_ordered) / (units_ordered + 1)],
            'Holiday/Promotion': [0],  # Assume no promotions for now
            'Year': [2025],
            'Month': [5],
            'Day': [15],
        })

        # Scale the input data using the same scaler
        input_scaled = scaler.transform(input_data[numerical_cols])

        # Make prediction
        prediction = model.predict(input_scaled)

        # Return prediction and errors
        return jsonify({
            'prediction': prediction.tolist(),
            'training_error': train_error,
            'validation_error': val_error
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
