import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Load the dataset

df = pd.read_csv('telecom_data1.csv')

# Display basic info

print(df.info())


# Checking for missing values

print(df.isnull().sum())

# Check for Duplicates

print(f"Duplicicated Rows: {df.duplicated().sum()}")

# Display the first few rows to understand the data structure

print(f"First Few rows of info: \n{df.head()}")

# Describe the data to check for outliers ,data distributions

print(f" Data Descriptions: \n{df.describe()}")


# Visualize the distribution of numerical features

df.hist(bins=30, figsize=(15, 10))
plt.show()

# Visualizing categorical features

categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    plt.figure(figsize=(8,6))
    sns.countplot(data=df, x=col)
    plt.title(f'Distribution of {col}')
    plt.show()


# Filter the DataFrame for numeric columns only
numeric_cols = df.select_dtypes(include=['number'])

# Correlation Matrix to check the relationship between numerical features
plt.figure(figsize=(12, 8))
sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

# Check feature importances using a simple tree-based model ( e.g, RandomForest )
from sklearn.ensemble import RandomForestClassifier

# Assuming 'Churn' is the Target variable and the rest are features
X = df.drop('Churn', axis= 1 )
y = df[ 'Churn']

# Encode categorical variables if any
X = pd.get_dummies(X)

# Fit a Random forest model to check feature importances
model =  RandomForestClassifier()
model.fit(X, y)

#Plot Feature importances
feature_importances =  model.feature_importances_
indices =  np.argsort(feature_importances)[::-1]
plt.figure(figsize=(12,6))
plt.barh(X.columns[indices],feature_importances[indices])
plt.title('Feature Importances')
plt.show()


# /// 3)Data Processing and reprocessing  and splitting ///

# Handle missing values (simple imputation for now )
X.fillna(X.mean(), inplace=True )

# Encoding categorical variables using one-hot encoding

X =  pd.get_dummies(X)

# Split the data into training  and test sets (80/20 split)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2 , random_state= 42 )


# Display the size of the datasets

print(f'Training data shape : {X_train.shape} ')
print(f'Test data shape: {X_test.shape} ')



#  //////////////  4) Model building  //////////

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Initialize classifiers
models = {
    'Random Forest': RandomForestClassifier(),
    'Logistic Regression': LogisticRegression(),
    'Support Vector Machine': SVC()
}

# Train each model and evaluate performance
results = {}
for model_name, model in models.items():
    print(f"\nTraining {model_name}...")

    # Train the model
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Save the results
    results[model_name] = {'Accuracy': accuracy, 'F1 Score': f1}

    print(f"{model_name} Accuracy: {accuracy:.4f}")
    print(f"{model_name} F1 Score: {f1:.4f}")

    # Display confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"{model_name} Confusion Matrix:")
    print(cm)

# Display all results in a DataFrame
results_df = pd.DataFrame(results).T
print("\nModel Comparison:")
print(results_df)



# Complete









