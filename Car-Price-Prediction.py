import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

# 1. DATA LOADING

print("--- Loading Your Data: car_data.csv ---")
df = pd.read_csv('car_data.csv')
print("First 5 rows of your dataset:")
print(df.head())


# 2. DATA PREPROCESSING AND FEATURE ENGINEERING

print("\n--- Preprocessing Data and Engineering Features ---")

# Calculate car age
current_year = 2025
df['car_age'] = current_year - df['year']

# Drop columns that are no longer needed
df.drop(['name', 'year'], axis=1, inplace=True)

# Map 'owner' column to numerical values
owner_mapping = {
    'First Owner': 0,
    'Second Owner': 1,
    'Third Owner': 2,
    'Fourth & Above Owner': 3,
    'Test Drive Car': 4
}
df['owner'] = df['owner'].replace(owner_mapping)

# One-hot encode categorical columns
df = pd.get_dummies(df, columns=['fuel', 'seller_type', 'transmission'], drop_first=True)

print("\nDataset after all preprocessing:")
print(df.head())


# 3. MODEL TRAINING

print("\n--- Splitting Data and Training Model ---")
X = df.drop('selling_price', axis=1)
y = df['selling_price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
print("\nModel training complete.")


# 4. MODEL EVALUATION

print("\n--- Evaluating Model Performance ---")
predictions = rf_model.predict(X_test)
r2 = metrics.r2_score(y_test, predictions)
mae = metrics.mean_absolute_error(y_test, predictions)
rmse = np.sqrt(metrics.mean_squared_error(y_test, predictions))

print(f"R-squared (R²): {r2:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")


# 5. MAKING A PREDICTION ON NEW DATA

print("\n--- Making a Prediction on New Data ---")
print("Model expects columns in this order:")
print(list(X.columns))

# Example: create a new car data sample matching the columns in X
new_car_data = np.zeros((1, len(X.columns)))
# Set values for the first three columns
new_car_data[0, X.columns.get_loc('km_driven')] = 70000
new_car_data[0, X.columns.get_loc('owner')] = 0
new_car_data[0, X.columns.get_loc('car_age')] = 18
# Set one-hot encoded columns as needed
if 'fuel_Petrol' in X.columns:
    new_car_data[0, X.columns.get_loc('fuel_Petrol')] = 1
if 'seller_type_Individual' in X.columns:
    new_car_data[0, X.columns.get_loc('seller_type_Individual')] = 1
if 'transmission_Manual' in X.columns:
    new_car_data[0, X.columns.get_loc('transmission_Manual')] = 1

new_car_df = pd.DataFrame(data=new_car_data, columns=X.columns)
predicted_price = rf_model.predict(new_car_df)
print(f"\nPredicted Selling Price for the new car: ₹{predicted_price[0]:,.2f}")