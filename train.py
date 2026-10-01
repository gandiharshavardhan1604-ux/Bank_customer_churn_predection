import pandas as pd

# Load the dataset
data = pd.read_csv("data/Churn_Modelling.csv")

# Display first 5 rows
print("First 5 rows:")
print(data.head())

# Display dataset shape
print("\nDataset Shape:")
print(data.shape)

# Display dataset information
print("\nDataset Information:")
data.info()

# Remove unnecessary columns
data = data.drop(["RowNumber", "CustomerId", "Surname"], axis=1)

print("\nColumns after removing unnecessary columns:")
print(data.columns)

# Separate features and target
X = data.drop("Exited", axis=1)
y = data["Exited"]

print("\nFeatures (X):")
print(X.head())

print("\nTarget (y):")
print(y.head())

# Convert categorical columns into numerical values
X = pd.get_dummies(X, columns=["Geography", "Gender"], drop_first=True)

print("\nEncoded Features:")
print(X.head())
from sklearn.model_selection import train_test_split

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Scale the features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create the model
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train_scaled, y_train)

print("\nModel training completed!")
# Make predictions on test data
y_pred = model.predict(X_test_scaled)

print("\nPredictions:")
print(y_pred[:10],y_test[:10])
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

from sklearn.metrics import classification_report

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

import joblib

# Save the model and scaler
model_data = {
    "model": model,
    "scaler": scaler
}

joblib.dump(model_data, "model.pkl")

print("\nModel saved successfully as model.pkl")