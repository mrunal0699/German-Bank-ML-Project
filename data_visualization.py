import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned data
cleaned_data = pd.read_csv('data/cleaned_data.csv')

# Basic data exploration
print(cleaned_data.head())  # Display the first few rows of the dataset
print(cleaned_data.info())  # Display information about the dataset 

# Data visualization
# Create a histogram of ages
plt.figure(figsize=(8, 6))
sns.histplot(cleaned_data['age'], bins=20, kde=True)
plt.title('Distribution of Ages')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Convert the 'job' column to categorical type
cleaned_data['job'] = cleaned_data['job'].astype('category')

# Create a bar plot of job types
plt.figure(figsize=(10, 6))
sns.countplot(data=cleaned_data, x='job')
plt.title('Distribution of Job Types')
plt.xlabel('Job Type')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

#Create a scatter plot of age vs. loan amount
plt.figure(figsize=(8, 6))
sns.scatterplot(data=cleaned_data, x='age', y='amount', hue='default')
plt.title('Age vs. Loan Amount')
plt.xlabel('Age')
plt.ylabel('Loan Amount')
plt.legend()
plt.show()


# Visualize distribution of default status
plt.figure(figsize=(6, 4))
sns.countplot(x='default', data=cleaned_data)
plt.title('Distribution of Default Status')
plt.show()


# Visualize correlations between features
plt.figure(figsize=(10, 8))
sns.heatmap(cleaned_data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

input("Press Enter to exit...")
