import pandas as pd
import matplotlib.pyplot as plt
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

# Step 2: Check the columns in the dataset
print(f"Columns in the DataFrame: {df.columns}")

# Step 3: Define the list of columns you want to check
feature_cols = ['Price', 'Discount', 'Demand Forecast', 'Competitor Pricing',
                'Discounted Price', 'Price Difference', 'Stock to Order Ratio', 
                'Forecast Accuracy', 'Holiday/Promotion', 'Year', 'Month', 'Day']

# Step 4: Check if the columns are missing in the DataFrame
missing_cols = [col for col in feature_cols if col not in df.columns]
if missing_cols:
    print(f"Missing columns: {missing_cols}")
    raise ValueError(f"Missing columns in the DataFrame: {missing_cols}")
else:
    print("All specified columns are present in the DataFrame.")

# Step 5: Data Preprocessing and Feature Engineering
# Convert Date column and extract time-based features
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day

# Feature engineering
df['Discounted Price'] = df['Price'] * (1 - df['Discount'] / 100)
df['Price Difference'] = df['Price'] - df['Competitor Pricing']
df['Stock to Order Ratio'] = df['Inventory Level'] / (df['Units Ordered'] + 1)
df['Forecast Accuracy'] = abs(df['Demand Forecast'] - df['Units Sold']) / (df['Units Sold'] + 1)

# Classify the units sold for Demand Class
def classify_units(units):
    if units <= 50:
        return 0  # Low
    elif units <= 150:
        return 1  # Medium
    else:
        return 2  # High

df['Demand Class'] = df['Units Sold'].apply(classify_units)

# Prepare features and target
X = df[feature_cols]
y = df['Demand Class']

# Step 6: Split the data into training and validation sets
X_train, X_val, Y_train, Y_val = train_test_split(X, y, test_size=0.2, random_state=22)

# Step 7: Scale numerical features
numerical_cols = ['Price', 'Discount', 'Demand Forecast', 'Competitor Pricing', 
                  'Discounted Price', 'Stock to Order Ratio', 'Inventory Level', 
                  'Units Ordered']
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train[numerical_cols])
X_val_scaled = scaler.transform(X_val[numerical_cols])

# Step 8: Define models for training
models = [
    ("Linear Regression", LinearRegression()),
    ("Lasso Regression", Lasso(alpha=0.1)),
    ("Ridge Regression", Ridge(alpha=1.0))
]

# Train and evaluate each model
trained_models = {}
for name, model in models:
    print(f'Training {name}...')
    model.fit(X_train_scaled, Y_train)
    trained_models[name] = model

    # Training Error (MAE)
    train_preds = model.predict(X_train_scaled)
    train_error = mae(Y_train, train_preds)
    print(f'Training Error (MAE) for {name}: {train_error:.4f}')

    # Validation Error (MAE)
    val_preds = model.predict(X_val_scaled)
    val_error = mae(Y_val, val_preds)
    print(f'Validation Error (MAE) for {name}: {val_error:.4f}')
    print()
