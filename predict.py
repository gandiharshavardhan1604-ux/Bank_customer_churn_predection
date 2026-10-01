import pandas as pd
import joblib

# Load the trained model and scaler
model_data = joblib.load("model.pkl")

model = model_data["model"]
scaler = model_data["scaler"]

# Get customer details
credit_score = float(input("Enter Credit Score: "))
age = float(input("Enter Age: "))
tenure = float(input("Enter Tenure: "))
balance = float(input("Enter Balance: "))
num_products = float(input("Enter Number of Products: "))
has_card = float(input("Has Credit Card? (1=Yes, 0=No): "))
active_member = float(input("Is Active Member? (1=Yes, 0=No): "))
salary = float(input("Enter Estimated Salary: "))

geography = input("Enter Geography (France/Spain/Germany): ")
gender = input("Enter Gender (Male/Female): ")

# Convert categorical values
geography_germany = 1 if geography == "Germany" else 0
geography_spain = 1 if geography == "Spain" else 0
gender_male = 1 if gender == "Male" else 0

# Create customer data
customer = pd.DataFrame([[
    credit_score,
    age,
    tenure,
    balance,
    num_products,
    has_card,
    active_member,
    salary,
    geography_germany,
    geography_spain,
    gender_male
]], columns=[
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
    "Geography_Germany",
    "Geography_Spain",
    "Gender_Male"
])

# Scale the customer data
customer_scaled = scaler.transform(customer)

# Make prediction
prediction = model.predict(customer_scaled)

# Display result
if prediction[0] == 1:
    print("\nCustomer may churn.")
else:
    print("\nCustomer is likely to stay.")