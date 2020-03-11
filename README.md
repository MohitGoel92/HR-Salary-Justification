# HR Salary Justification

**Task:**

We are working within the HR department in a company and are about to hire a new employee. We are about to negotiate the salary. The new hire has 20+ years of experience and says he/she used to earn a salary of £160,000, therefore being the desired minimum salary for their new role. The HR team has reached out to the new hire's previous employer to verify, but they only sent back a table of salaries that the company uses to band their employees. They also stated that he/she was a regional manager for two years and it takes 4 years to go from regional manager to partner. Lets predict if his/her claim is truth or bluff. Our task therefore being to predict the salary of someone who is at level 6.5 (mid way between regional manager and partner).

We will be completing this task using Polynomial, Support Vector Machine, Decision Tree and Random Forest Regression.

## Polynomial Linear Regression

Polynomial regression is also referred to as Polynomial Linear Regression as "Linear" refers to the coefficients of the (X)^i terms. We do not require feature scaling as we are only adding polynomial terms to the multiple linear regression function, therefore using the linear regression library (sklearn.linear_model).

**Pros of Polynomial Regression:**

- Works on any size of dataset 
- Works very well on non-linear problems.

**Cons of Polynomial Regression:**

- Need to choose the right polynomial degree for a good bias/variance trade off.

 The dataset received from the new hire's company has been plotted on the scatter graph below. 

<img src = 'Screen1.png' width='700'>

The diagram below shows the Polynomial Regressor being applied to the dataset, along with a higher resolution plot (step size 0.01) for a smoother curve. 

<img src = 'Screen2.png' width='700'>

The predicted salary for someone at level 6.5 has been predicted to be just under £159,000. Although this is just over £1,000 under the new hire's claimed previous salary, we accept the new hire's claim and state that the new hire was telling us the truth.

## Support Vector Machine - SVR

Support vector machines (SVM) support linear and non-linear regression that we refer to as SVR. SVR performs linear regression in a higher dimensional space. The vector we get when we evaluate the test point for all points in the training set is the representation of the test point in the higher dimensional space. Once we have that vector we use it to perform a linear regression. The work of the SVM is to approximate the function we used to generate the training set F(X) = y

The vectors closest to the test points are referred to as the support vectors. We can evaluate our function anywhere so any vectors could be closest to our test evaluation location. Please see the diagram below for clarification.

<img src = 'Screen6.png' width='700'>

**Note:**

SVR has a different regressional goal compared to linear regression. In linear regression, we are trying to minimise the error between the prediction and data. In SVR however, our goal is to make sure that errors do not exceed our threshold.

**Pros of SVR:**

- Easily adaptable
- Works very well on non-linear problems 
- Not biased by outliers.

**Cons of SVR:**

- Compulsory to apply feature scaling
- Not as well known 
- Can be difficult to understand.

In this analysis, we are choosing the Gaussian RBF Kernel (Radial Basis Function) and regularisation as due to the training sets with noise, the reguliser will prevent wild fluctuations between data points by smoothing out the prior. Any points away from the SVR curve will be treated as an outlier (as illustrated by the salary of the CEO in our dataset).

The diagram below shows the SVR being applied to the dataset, along with a higher resolution plot (step size 0.01) for a smoother curve. 

<img src = 'Screen5.png' width='700'>

The predicted salary for someone at level 6.5 has been predicted to be just over £170,000, being in support of the new hire's claim. We therefore state that the new hire was telling us the truth.

## Decision Tree Regression

**CART:** Classification and Regression Trees

The diagram below illustrates an example of how the Decision Tree Regression works. The small segments are called "terminal leaves". 

<img src = 'Screen7.png' width='700'>

We take the average of each terminal leaf and the value obtained will be the value predicted for any point that lands in that terminal leaf.

<img src = 'Screen8.png' width='700'>

The tree diagram below shows how each new point will be assigned a new predicted value. 

<img src = 'Screen9.png' width='700'>

**Pros of Decision Tree Regression:**

- Interpretability
- Feature scaling not being required
- Works on both linear/non-linear problems

**Cons of Decision Tree Regression:**

- Poor results on too small datasets
- Overfitting can easily occur

The diagram below shows the Decision Tree Regressor model being applied to the dataset.

<img src = 'Screen11.png' width='700'>

As the Decision Tree Regressor is not a continuous regression model, we must visualise the model using a higher resolution. The diagram below shows the Decision Tree Regressor being applied to the dataset, along with a higher resolution plot (step size 0.01).

<img src = 'Screen12.png' width='700'>

The expected salary for someone at level 6.5 has been predicted to be £150,000. This is £10,000 under the new hire's claimed previous salary. Therefore, if we base our decision solely on the Decision Tree Regression model, we do not accept the new hire's claim and state that the new hire was bluffing. However, as the dataset is small and Decision Tree Regression yield poor results on small datasets, the validity of the result/decision is questionable.

## Random-Forest-Regression

The Random Forest model takes a team of Decision Trees, therefore, multiple trees coming together to make a forest. The ultimate prediction of the Random Forest itself is the average of the different predictions of all the different trees in the forest.

**Ensemble Learning:** When you take the same algorithm multiple times and make it more powerful than the original.

- **Step 1:** Pick at random K data points from the training set
- **Step 2:** Build the Decision Tree associated to these K data points
- **Step 3:** Choose the number of N trees you want to build and repeat the above two steps
- **Step 4:** For a new data point, make each one of your N trees predict the value of y for the data point in question and assign the new data point the average across all the predicted y-values. It is common practice to select a large number of trees, for instance, N=300 (300 trees).

As we're getting a high number of predictions for the y-value and then taking the average (predicting based on a forest of trees and not one tree), this improves the accuracy of our predictions as we're taking the average of many predictions. If we just take one tree, it may just be an "okay" or "not great" tree but as we're taking the average, it's less likely to arrive at a bad prediction. Ensemble algorithms are more stable due to any changes in the data that may impact a tree will unlikely impact a forest of trees.

**Pros of Random Forest Regression:**

- Powerful and accurate
- Good performance on linear and non-linear problems

**Cons of Random Forest Regression:**

- No interpretability
- Overfitting can easily occur
- Need to choose the number of trees

The diagram below shows the Random Forest Regressor model being applied to the dataset.

<img src = 'Screen14.png' width='700'>

As the Random Forest Regressor is not a continuous regression model, we must visualise the model using a higher resolution. The diagram below shows the Random Forest Regressor being applied to the dataset, along with a higher resolution plot (step size 0.01).

**Note:** 
As the number of trees N (n_estimators) increases, the number of steps do not increase as the averages converge to a similar number.

<img src = 'Screen15.png' width='700'>

The expected salary for someone at level 6.5 has been predicted to be £160,333. This is in exact agreement with the new hire's claimed previous salary. We therefore state that the new hire was tell us the truth.
