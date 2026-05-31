import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

print("Step 1: Loading and preparing data...")
df = pd.read_csv('churn_data.csv')

# (Re-running our cleaning steps from last time)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
df = df.drop('customerID', axis=1)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
df = pd.get_dummies(df, drop_first=True)

print("Step 2: Splitting data into Train and Test sets...")
X = df.drop('Churn', axis=1)
y = df['Churn']

# Splitting 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training shapes: {X_train.shape}, Testing shapes: {X_test.shape}")

print("\nStep 3: Initializing and Training Logistic Regression...")
# Create the model structure
model = LogisticRegression(max_iter=1000) 

# Train the model (this is where the math happens!)
model.fit(X_train, y_train) 

# Score the model on the unseen test exam
accuracy = model.score(X_test, y_test)
print(f"\nSuccess! Baseline Model Accuracy on Unseen Data: {accuracy * 100:.2f}%")

from sklearn.metrics import confusion_matrix, classification_report

print("\n--- Phase 4: Model Evaluation ---")

# Ask the model to actually predict the test data, instead of just scoring it
y_predictions = model.predict(X_test)

# 1. The Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_predictions))

# 2. The Classification Report (Precision and Recall)
print("\nClassification Report:")
print(classification_report(y_test, y_predictions))