# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 16:17:54 2023
problem_statements
1) Delivery_time -> Predict delivery time using sorting time
"""

#delivery_time_csv (file)
# Import necessary librarie
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
# Load the dataset
file_path  = ('F:\delivery_time.csv')  # Update with your actual file path
data = pd.read_csv(file_path)
# Display the first few rows of the dataset
print(data.head())
# Visualize the data if applicable
# For example, if you have columns 'Sorting Time' and 'Delivery Time'
plt.scatter(data['Sorting Time'], data['Delivery Time'])
plt.xlabel('Sorting Time')
plt.ylabel('Delivery Time')
plt.title('Sorting Time vs Delivery Time')
plt.show()
# Prepare the data for modeling
X = data[['Sorting Time']]
y = data['Delivery Time']
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)
# Make predictions on the test set
y_pred = model.predict(X_test)
# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
# Plot the regression line
plt.scatter(X_test, y_test, label='Actual')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted')
plt.xlabel('Sorting Time')
plt.ylabel('Delivery Time') 
plt.title('Linear Regression Model')
plt.legend()
plt.show()





"2) Salary_hike -> Build a prediction model for Salary_hike"

"Salary_data.csv(file)."

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
# Load the dataset
file_path = r'F:\Salary_data.csv'
data = pd.read_csv(file_path)
# Display the first few rows of the dataset
print(data.head())
# EDA and Data Visualization
plt.scatter(data['YearsExperience'], data['Salary'])
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Scatter Plot of Years of Experience vs. Salary')
plt.show()
# Split the data into training and testing sets
X = data[['YearsExperience']]
y = data['Salary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train a simple linear regression model
model = LinearRegression()
model.fit(X_train, y_train)
# Make predictions on the test set
y_pred = model.predict(X_test)
# Evaluate the model
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')
# Plot the regression line
plt.scatter(X_test, y_test, label='Actual')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Simple Linear Regression Model')
plt.legend()
plt.show()



































