# Support Vector Machine - Regression

# Importing the libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

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
# to have maximum accuracy, therefore taking the whole data into consideration. As this is salary discussion, we
# be confident in our model as the negotiation can fall apart if we go ahead with inaccurate presumptions.

# Feature Scaling

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
XX = sc_X.fit_transform(X)
yy = sc_y.fit_transform(y.reshape(-1,1))

# Fitting the SVR to the dataset

from sklearn.svm import SVR
svr = SVR(kernel = 'rbf')
svr.fit(XX, yy)

# Predicting the position level 6.5

y_pred = sc_y.inverse_transform(svr.predict(sc_X.transform(np.array([[6.5]]))))

# Visualising the SVR 

plt.scatter(XX, yy, c='red')
plt.plot(XX, svr.predict(XX), c='blue')
plt.title('HR Salaries Dataset')
plt.xlabel('Seniority Level')
plt.ylabel('Salary £s')
plt.show()

# Visualising the SVR in higher resolution (smoother curve)

X_grid = np.arange(min(XX), max(XX), step = 0.01)
X_grid = X_grid.reshape((len(X_grid)), 1)

plt.scatter(XX, yy, c='red')
plt.plot(X_grid, svr.predict(X_grid), c='blue')
plt.title('HR Salaries Dataset')
plt.xlabel('Seniority Level')
plt.ylabel('Salary £s')
plt.show()