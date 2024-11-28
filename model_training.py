import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression  
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  

# Load cleaned data
cleaned_data = pd.read_csv('data/cleaned_data.csv')

# Encode categorical variables
label_encoder = LabelEncoder()
for col in cleaned_data.select_dtypes(include=['object']).columns:
    cleaned_data[col] = label_encoder.fit_transform(cleaned_data[col])

# Split data into features (X) and target (y)
X = cleaned_data.drop('default', axis=1)
y = cleaned_data['default']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate models
models = {
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC()
}

results = {}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)
    results[model_name] = {'accuracy': accuracy, 'classification_report': classification_rep}

# Generate figures or tables
plt.figure(figsize=(10, 6))
sns.barplot(x=list(results.keys()), y=[result['accuracy'] for result in results.values()])
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.tight_layout()

for model_name, result in results.items():
    print(f"Model: {model_name}")
    print(f"Accuracy: {result['accuracy']:.2f}")
    print("Classification Report:")
    print(result['classification_report'])
    print()

# Save the figure
plt.savefig('accuracy_comparison.png')

print("Training and evaluation completed.")

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train a logistic regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Predictions on the test set
y_pred = model.predict(X_test_scaled)

# Evaluate the model
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Generate a bar plot showing the distribution of default status based on job type
plt.figure(figsize=(10, 6))
sns.countplot(x='job', hue='default', data=cleaned_data)
plt.title('Distribution of Default Status by Job Type')
plt.xticks(rotation=45)
plt.show()
