import pandas as pd
import numpy as np

# Generate a sample telecom customer churn dataset
np.random.seed(42)

# Create random data for customer features
n = 1000  # Number of customers

# Customer IDs
customer_id = np.arange(1, n+1)

# Features
gender = np.random.choice(['Male', 'Female'], n)
age = np.random.randint(18, 70, n)
tenure = np.random.randint(1, 72, n)  # Months with the telecom company
service_type = np.random.choice(['Mobile', 'Broadband', 'Both'], n)
monthly_spend = np.random.randint(30, 150, n)  # Monthly spend in USD
is_international_plan = np.random.choice([0, 1], n)
has_phone_service = np.random.choice([0, 1], n)
is_online_security = np.random.choice([0, 1], n)
is_tech_support = np.random.choice([0, 1], n)
is_churn = np.random.choice([0, 1], n)  # Target variable: 0 = not churned, 1 = churned

# Create DataFrame
df = pd.DataFrame({
    'CustomerID': customer_id,
    'Gender': gender,
    'Age': age,
    'Tenure': tenure,
    'ServiceType': service_type,
    'MonthlySpend': monthly_spend,
    'InternationalPlan': is_international_plan,
    'PhoneService': has_phone_service,
    'OnlineSecurity': is_online_security,
    'TechSupport': is_tech_support,
    'Churn': is_churn
})

# Save the DataFrame to a CSV file
df.to_csv('telecom_data1.csv', index=False)

print("telecom_data.csv has been created.")
