#%% Template for preprocessing
# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%% Importing the dataset
dataset = pd.read_csv('/Users/roeeyn/Desktop/Machine Learning A-Z/Part 2 - Regression/Section 4 - Simple Linear Regression/Salary_Data.csv')
X = dataset.iloc[:, :-1].values  # Indepedent variable
y = dataset.iloc[:, 1].values  # Dependent variable

#%% Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

#%% Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#%% Linear Regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#%% Predicting the test set result
y_pred = regressor.predict(X_test)

#%% Visualizing the training set results
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()

#%% Visualizing the training set results
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.plot(X_test, regressor.predict(X_test), color='green')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()
