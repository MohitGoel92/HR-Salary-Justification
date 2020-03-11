# Decision Tree Regression

# Importing the libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing the dataset

ds = pd.read_csv('Position_Salaries.csv')
X = ds.iloc[:,1:-1].values
y = ds.iloc[:,-1].values

# Visualising the dataset

plt.scatter(X, y, c='red')
plt.title('HR Salaries Dataset')
plt.xlabel('Seniority Level')
plt.ylabel('Salary £s')
plt.show()

# There is no missing data

# There is no categorical data to encode

# As the dataset is small, we will not be splitting the dataset into the training set and testing set as we wish 
# to have maximum accuracy, therefore taking the whole data into consideration. As this is a salary negotiation, we
# must be confident in our model as the negotiation can fall apart if we go ahead with inaccurate presumptions.

# We will not be using feature scaling as we wish to have interpretability of the data when visualised

# Fitting the Decision Tree Regressor to the dataset

from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor()
dtr.fit(X,y)

# Predicting the position level 6.5 salary

y_pred = dtr.predict([[6.5]])

# Visualising the Decision Tree Regression model

plt.scatter(X, y, c='red')
plt.plot(X, dtr.predict(X), c='blue')
plt.title('HR Salaries Dataset')
plt.xlabel('Seniority Level')
plt.ylabel('Salary £s')
plt.show()

# As the Regression Tree is not a continuous regression model, we must visualise the model using a higher resolution

X_grid = np.arange(min(X), max(X), step = 0.01)
X_grid = X_grid.reshape((len(X_grid)),1)
plt.scatter(X, y, c='red')
plt.plot(X_grid, dtr.predict(X_grid), c='blue')
plt.title('HR Salaries Dataset')
plt.xlabel('Seniority Level')
plt.ylabel('Salary £s')
plt.show()