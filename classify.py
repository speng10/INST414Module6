import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("earthquake_data.csv")

# Handle missing values
data.fillna({
    'cdi': data['cdi'].mean(),
    'mmi': data['mmi'].mean(),
    'alert': 'green',  # Assuming least severe if missing
    'dmin': data['dmin'].median(),
    'gap': data['gap'].median(),
    'continent': 'Unknown',
    'country': 'Unknown',
    'magType': 'Unknown',  # Handle missing magType
}, inplace=True)

# Drop unnecessary columns if they exist
data.drop(columns=['title', 'date_time', 'net', 'location'], inplace=True, errors='ignore')

# Encode categorical features (excluding 'alert' since it's our target)
data = pd.get_dummies(data, columns=['continent', 'country', 'magType'], drop_first=True)

# Convert 'alert' to a categorical column and encode it as the target
data['alert'] = data['alert'].astype('category')
data['alert_encoded'] = data['alert'].cat.codes  # Convert alert to numeric codes

# Define features (X) and target (y)
X = data.drop(columns=['alert', 'alert_encoded'])  # Only predictors in X, no alert
y = data['alert_encoded']  # Target variable (alert)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42, max_depth=5)  # Limit depth for interpretability
clf.fit(X_train, y_train)

# Test the model
y_pred = clf.predict(X_test)

# Define the correct target_names based on unique classes
target_names = data['alert'].cat.categories  # Use alert categories for labels

# Print the classification report with the correct labels
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))

# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Plot the tree with the correct class names
plt.figure(figsize=(20, 10))
plot_tree(clf, feature_names=X.columns, class_names=target_names, filled=True)
plt.title("Decision Tree Visualization")
plt.show()
