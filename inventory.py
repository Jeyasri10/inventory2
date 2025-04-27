import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error as mae

# Step 1: Load the DataFrame (Ensure the path to your CSV is correct)
try:
    df = pd.read_csv('retail_store_inventory.csv')  # Replace with your actual dataset path
    print("DataFrame loaded successfully.")
except FileNotFoundError:
    print("Error: The file was not found. Please check the file path.")
    exit()

# Check the first few rows to ensure that data is loaded correctly
print(df.head())

# Step 2: Check if 'Units Sold' column is present
if 'Units Sold' not in df.columns:
    raise ValueError("'Units Sold' column is missing from the dataset.")

# Step 3: Feature Engineering - Creating 'Demand Class'
def classify_units(units):
    if units <= 50:
        return 0  # Low
    elif units <= 150:
        return 1  # Medium
    else:
        return 2  # High

df['Demand Class'] = df['Units Sold'].apply(classify_units)

# Ensure 'Demand Class' is created correctly
print(f"Demand Class column created: {df['Demand Class'].head()}")

# Step 4: Prepare Features and Target
feature_cols = ['Price', 'Discount', 'Demand Forecast', 'Competitor Pricing',
                'Discounted Price', 'Price Difference', 'Stock to Order Ratio',
                'Forecast Accuracy', 'Holiday/Promotion', 'Year', 'Month', 'Day']

# Check if the feature columns are in the DataFrame
missing_cols = [col for col in feature_cols if col not in df.columns]
if missing_cols:
    raise ValueError(f"Missing columns in the DataFrame: {missing_cols}")

# Prepare features (X) and target (y)
X = df[feature_cols]
y = df['Demand Class']

# Step 5: Train-Test Split
X_train, X_val, Y_train, Y_val = train_test_split(X, y, test_size=0.2, random_state=22)

# Step 6: Scale numerical features
numerical_cols = ['Price', 'Discount', 'Demand Forecast', 'Competitor Pricing', 'Discounted Price', 
                  'Stock to Order Ratio', 'Inventory Level', 'Units Ordered']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train[numerical_cols])
X_val_scaled = scaler.transform(X_val[numerical_cols])

# Step 7: Train Models (Linear Regression, Lasso Regression, Ridge Regression)
models = {
    "Linear Regression": LinearRegression(),
    "Lasso Regression": Lasso(alpha=0.1),
    "Ridge Regression": Ridge(alpha=1.0)
}

trained_models = {}

for model_name, model in models.items():
    model.fit(X_train_scaled, Y_train)
    trained_models[model_name] = model

# Step 8: Evaluate Models on Training and Validation Data
for model_name, model in trained_models.items():
    # Training predictions and error
    train_preds = model.predict(X_train_scaled)
    train_error = mae(Y_train, train_preds)

    # Validation predictions and error
    val_preds = model.predict(X_val_scaled)
    val_error = mae(Y_val, val_preds)

    # Print training and validation errors
    print(f"{model_name}:")
    print(f"  Training Error (MAE): {train_error:.4f}")
    print(f"  Validation Error (MAE): {val_error:.4f}")

# Optional: Display predictions on validation data for inspection
print("\nPredictions on Validation Data (for first 5 samples):")
for model_name, model in trained_models.items():
    val_preds = model.predict(X_val_scaled)
    print(f"\n{model_name} predictions on validation data:")
    print(val_preds[:5])  # Print the first few predictions
