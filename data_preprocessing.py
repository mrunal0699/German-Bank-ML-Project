import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

# Read the dataset
data = pd.read_csv("C:\\Users\\Mrunal Jadhav\\OneDrive\\Desktop\\Final Project German Bank\\data\\credit.csv")

# Display basic information about the dataset
print(data.info())

# Check for missing values
print(data.isnull().sum())

# Perform data cleaning and preprocessing (e.g., encoding categorical variables, handling missing values)
# ...
# Encode categorical variables
categorical_cols = ['checking_balance', 'credit_history', 'purpose', 'savings_balance', 'employment_duration', 'other_credit', 'housing', 'job', 'phone']
encoder = LabelEncoder()
for col in categorical_cols:
    data[col] = encoder.fit_transform(data[col])

# Save the cleaned data to a new CSV file within the 'data' directory
if not os.path.exists('data'):
    os.makedirs('data')
# Save the cleaned data to a new CSV file
data.to_csv('data/cleaned_data.csv', index=False)

input("Press Enter to exit...")
