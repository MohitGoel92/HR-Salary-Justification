# Polynomial Regression

# Importing the libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing the dataset

ds = pd.read_csv('Position_Salaries.csv')
X = ds.iloc[:,1:-1].values
y = ds.iloc[:,-1].values

# Visualising the dataset
# As the dataset appears to be following a parabolic path, a linear regression will be a poor fit.

plt.scatter(X, y, c='red')
plt.title('HR Salaries Dataset')
plt.xlabel('Seniority Level')
plt.ylabel('Salary £s')
plt.show()

# There is no missing data

# There is no categorical data to encode

# As the dataset is small, we will not be splitting the dataset into the training set and testing set as we wish 
# to have maximum accuracy, therefore taking the whole data into consideration. As this is salary negotiation, we
# must be confident in our model as the negotiation can fall apart if we go ahead with inaccurate presumptions.

# We do not require feature scaling as the linear_model library takes care of this for us. We are only adding 
# polynomial terms to the multiple linear regression equation, therefore the linear regression library.

# Fitting the polynomial regressor to our dataset

from sklearn.preprocessing import PolynomialFeatures
pf = PolynomialFeatures(degree = 4)

X_poly = pf.fit_transform(X)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()

lr.fit(X_poly, y)

# Visualising the Polynomial Linear Regression model

plt.scatter(X, y, c='red')
plt.plot(X, lr.predict(pf.fit_transform(X)), c='blue')
plt.title('HR Salaries Dataset')
plt.xlabel('Seniority Level')
plt.ylabel('Salary £s')
plt.show()

# Visualising the Polynomial Regression Model in higher resolution (smoother curve)

X_grid = np.arange(min(X), max(X), step=0.01)
X_grid = X_grid.reshape((len(X_grid)), 1)

plt.scatter(X, y, c='red')
plt.plot(X_grid, lr.predict(pf.fit_transform(X_grid)), c='blue')
plt.title('HR Salaries Dataset')
plt.xlabel('Seniority Level')
plt.ylabel('Salary £s')
plt.show()

# Predicting the position level 6.5

y_pred = lr.predict(pf.transform([[6.5]]))

