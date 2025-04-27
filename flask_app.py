from flask import Flask, request, jsonify
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the trained models and scaler globally
scaler = StandardScaler()

# Dummy trained models (Replace these with your actual trained models)
trained_models = {
    "Linear Regression": LinearRegression(),
    "Lasso Regression": Lasso(alpha=0.1),
    "Ridge Regression": Ridge(alpha=1.0)
}

# Step 1: Sample trained models (You should load the actual models here)
for model_name, model in trained_models.items():
    model.fit([[1, 1, 1, 1, 1, 1]], [0])  # Placeholder fit, replace with actual model fitting

@app.route('/')
def home():
    return "Welcome to the Inventory Prediction App!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input data from the request body (expecting JSON)
        data = request.json
        
        # Extract input values from the request JSON
        price = data['Price']
        discount = data['Discount']
        demand_forecast = data['Demand Forecast']
        competitor_pricing = data['Competitor Pricing']
        inventory_level = data['Inventory Level']
        units_ordered = data['Units Ordered']

        # Preprocess the data (here we do feature engineering based on your app)
        input_data = pd.DataFrame({
            'Price': [price],
            'Discount': [discount],
            'Demand Forecast': [demand_forecast],
            'Competitor Pricing': [competitor_pricing],
            'Discounted Price': [price * (1 - discount / 100)],
            'Price Difference': [price - competitor_pricing],
            'Stock to Order Ratio': [inventory_level / (units_ordered + 1)],
            'Forecast Accuracy': [abs(demand_forecast - units_ordered) / (units_ordered + 1)],
            'Holiday/Promotion': [0],  # Default value
            'Year': [2025],  # Default value
            'Month': [5],  # Default value
            'Day': [15]  # Default value
        })

        # Scale the input data using the same scaler used for training data
        input_scaled = scaler.fit_transform(input_data)

        # Make predictions using all models (Linear, Lasso, Ridge)
        predictions = {}
        for model_name, model in trained_models.items():
            prediction = model.predict(input_scaled)
            predictions[model_name] = prediction[0]  # Only take the first prediction

        return jsonify(predictions)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
