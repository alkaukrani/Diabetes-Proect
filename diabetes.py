# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Import necessary modules for modeling and evaluation
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# from sklearn.model_selection import learning_curve

# # Load data
# data = pd.read_csv('/Users/alkadeviukrani/Downloads/diabetes.csv')

# # Initial data inspection
# print("First 5 rows of data:")
# print(data.head())
# print("\nData Information:")
# print(data.info())
# print("\nSummary Statistics:")
# print(data.describe())
# print("\nMissing Values Count:")
# print(data.isnull().sum())



# # Define the function to assign diabetes types based on age, glucose, BMI, etc.
# def assign_diabetes_type(row):
#     if row['Age'] < 30 and row['Glucose'] > 140:  # Adjust thresholds if needed
#         return 'Type 1'
#     elif row['Age'] >= 30 and (row['BMI'] >= 25 or row['BloodPressure'] >= 80):
#         return 'Type 2'
#     else:
#         return 'Unknown'

# # Apply function to a sample to test it independently
# sample_data = data.head(10).copy()  # Copy the first 10 rows for testing
# sample_data['Diabetes_Type'] = sample_data.apply(assign_diabetes_type, axis=1)

# # Print the sample to verify the Diabetes_Type column is created
# print("\nSample data with assigned Diabetes_Type column:")
# print(sample_data[['Age', 'Glucose', 'BMI', 'BloodPressure', 'Diabetes_Type', 'Outcome_Type']])





# # Plot distributions for key features
# sns.histplot(data['Glucose'], kde=True)
# plt.title('Distribution of Glucose Levels')
# plt.show()

# sns.histplot(data['Pregnancies'], kde=True)
# plt.title('Distribution of Pregnancies')
# plt.show()

# # Define a function to assign diabetes types based on age, glucose, BMI, etc.


# # Define features (X) and the new target (y)
# X = data.drop(['Outcome', 'Diabetes_Type', 'Outcome_Type'], axis=1)
# y = data['Outcome_Type']

# # Split the dataset into training and testing sets (60% train, 40% test)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# # **UPDATED**: Use Random Forest for multi-class classification
# model = RandomForestClassifier(random_state=42)
# model.fit(X_train, y_train)

# # Make predictions on the test set
# y_prediction = model.predict(X_test)

# # Evaluate the model's accuracy
# accuracy = accuracy_score(y_test, y_prediction)
# print(f'\nAccuracy: {accuracy * 100:.2f}%')
# print("\nClassification Report:")
# print(classification_report(y_test, y_prediction))



# # Plot the learning curve
# train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5)
# plt.plot(train_sizes, train_scores.mean(axis=1), label='Training Accuracy')
# plt.plot(train_sizes, test_scores.mean(axis=1), label='Test Accuracy')
# plt.xlabel('Training Size')
# plt.ylabel('Accuracy')
# plt.title('Learning Curve')
# plt.legend()
# plt.show()




import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Import necessary modules for modeling and evaluation
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import learning_curve

# Load data
data = pd.read_csv('/Users/alkadeviukrani/Downloads/diabetes.csv')

# Initial data inspection
print("First 5 rows of data:")
print(data.head())
print("\nData Information:")
print(data.info())
print("\nSummary Statistics:")
print(data.describe())
print("\nMissing Values Count:")
print(data.isnull().sum())

# Define the function to assign diabetes types based on age, glucose, BMI, etc.
def assign_diabetes_type(row):
    if row['Age'] < 30 and row['Glucose'] > 140:  # Adjust thresholds if needed
        return 'Type 1'
    elif row['Age'] >= 30 and (row['BMI'] >= 25 or row['BloodPressure'] >= 80):
        return 'Type 2'
    else:
        return 'Unknown'

# Apply function to create Diabetes_Type column
data['Diabetes_Type'] = data.apply(assign_diabetes_type, axis=1)

# Apply function to a sample to test it independently
sample_data = data.head(10).copy() # Copy the first 10 rows for testing
sample_data['Diabetes_Type'] = sample_data.apply(assign_diabetes_type, axis=1)

# Print the sample to verify the Diabetes_Type column is created
print("\nSample data with assigned Diabetes_Type column:")
print(sample_data[['Age', 'Glucose', 'BMI', 'BloodPressure', 'Diabetes_Type']])

# Plot distributions for key features
sns.histplot(data['Glucose'], kde=True)
plt.title('Distribution of Glucose Levels')
plt.show()

sns.histplot(data['Pregnancies'], kde=True)
plt.title('Distribution of Pregnancies')
plt.show()

# Define features (X) and the new target (y)
# Only drop 'Outcome' and 'Diabetes_Type' as 'Outcome_Type' doesn't exist
X = data.drop(['Outcome', 'Diabetes_Type'], axis=1)
y = data['Diabetes_Type']

# Split the dataset into training and testing sets (60% train, 40% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.90, random_state=42)

# Use Random Forest for multi-class classification
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_prediction = model.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_prediction)
print(f'\nAccuracy: {accuracy * 100:.2f}%')
print("\nClassification Report:")
print(classification_report(y_test, y_prediction))

# Display the confusion matrix
cm = confusion_matrix(y_test, y_prediction)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Unknown', 'Type 1', 'Type 2'], yticklabels=['Unknown', 'Type 1', 'Type 2'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Plot the learning curve
train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5)
plt.plot(train_sizes, train_scores.mean(axis=1), label='Training Accuracy')
plt.plot(train_sizes, test_scores.mean(axis=1), label='Test Accuracy')
plt.xlabel('Training Size')
plt.ylabel('Accuracy')
plt.title('Learning Curve')
plt.legend()
plt.show()
