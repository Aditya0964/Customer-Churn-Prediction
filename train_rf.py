import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

print("Step 1: Preparing data...")
df = pd.read_csv('churn_data.csv')
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
df = df.drop('customerID', axis=1)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
df = pd.get_dummies(df, drop_first=True)

X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nStep 2: Training Random Forest Classifier...")
# n_estimators=100 means building a forest of 100 independent decision trees.
# random_state=42 keeps randomness predictable so we get the same results each run.
# We added class_weight='balanced' to force the AI to pay attention to churners
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_train, y_train)

print("\nStep 3: Evaluating Random Forest...")
y_predictions = rf_model.predict(X_test)

print("\nRandom Forest Confusion Matrix:")
print(confusion_matrix(y_test, y_predictions))

print("\nRandom Forest Classification Report:")
print(classification_report(y_test, y_predictions))