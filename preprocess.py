import pandas as pd
import numpy as np

# 1. Load the large dataset
print("Loading the Telco Churn dataset...")
df = pd.read_csv('churn_data.csv')

# 2. Inspect the dataset size and columns
print(f"Dataset successfully loaded! Total Rows: {df.shape[0]}, Total Columns: {df.shape[1]}")

# 3. Clean Missing Values hidden as blank spaces
# In this specific dataset, the 'TotalCharges' column has empty spaces " " for new customers.
# We force those spaces into 'NaN' (Not a Number) values, then fill them with 0.
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(0)

print("\nMissing values successfully handled!")

# 4. View a snapshot of what we have
print("\n--- First 3 Rows of the Dataset ---")
print(df[['customerID', 'tenure', 'MonthlyCharges', 'TotalCharges', 'Churn']].head(3))


# --- PHASE 2: FEATURE ENGINEERING ---

#removing cutomer id column
df = df.drop('customerID', axis=1)

df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
df = pd.get_dummies(df, drop_first=True)

print("\n--- Phase 2 Complete: Data is now pure numbers! ---")
print(f"Old column count was 21. New column count: {df.shape[1]}")