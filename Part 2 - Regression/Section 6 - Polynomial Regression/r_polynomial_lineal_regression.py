# Data Preprocessing Template

#%% Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%% Importing the dataset
dataset = pd.read_csv('/Users/roeeyn/Desktop/ML A-Z/Part 2 - Regression/Section 6 - Polynomial Regression/Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#%% Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

#%% Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

#%% Visualizing Linear
plt.scatter(X, y, color="red")
plt.plot(X, lin_reg.predict(X), color="blue")
plt.title("Truth of Bluff (Linear Regression)")
plt.xlabel("Position Level")
plt.ylabel('Salary')
plt.show()

#%% Visualizing Polynomial
plt.scatter(X, y, color="red")
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color="blue")
plt.title("Truth of Bluff (Polynomial Regression)")
plt.xlabel("Position Level")
plt.ylabel('Salary')
plt.show()

#%% Fitting values
print(lin_reg.predict([[6.5]]))
print(lin_reg_2.predict(poly_reg.fit_transform([[6.5]])))
