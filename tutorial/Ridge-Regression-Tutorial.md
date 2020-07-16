## Preparing the dataset
For this tutorial, we're going to work with the California housing dataset.
The task of the dataset is to predict the median price of housing from different
economic and geographic features. Since the dataset comes prepackaged with sklearn, it's
easy to load.
```python
import numpy as np
from sklearn.datasets import fetch_california_housing
X, y = fetch_california_housing(return_X_y=True)

# Remove some extreme outliers
occ_max = np.percentile(X[:,5], 99)
X[:, 5] = [min(x, occ_max) for x in X[:, 5]]
```
We'll split the data into random training and validation subsets.
```python
from sklearn.model_selection import train_test_split
N = 50
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=N, random_state=1)
```
## Fit a ridge regression model
Ridge regression builds a linear model with regressors chosen to minimize

<p align="center">
  <img src="https://raw.githubusercontent.com/rnburn/peak-engines/master/images/ridge-regression-equation.png" height="50" width="354">
</p>

With the right regularization matrix *Γ*, ridge regression can prevent overfitting and improve 
performance on out-of-sample predictions. How do we choose *Γ*? Most commonly, *Γ* is restricted to
the form

<p align="center">
  <img src="https://raw.githubusercontent.com/rnburn/peak-engines/master/images/gamma.png" height="17" width="77">
</p>

and *α* is chosen to minimize the mean squared error on a cross-validation of the training data. **sklearn**'s
[RidgeCV](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html), for example,
will take a list of regularization parameters to try and will select the one that performs the best on a
generalized cross-validation of the data.

With **peak-engine**'s RidgeRegressionModel, there's no need to supply regularization parameters to try. 
By treating cross-validation as an optimization problem and computing derivatives, it can quickly find
the best regularization parameter for either a leave-one-out or generalized cross-validation.

```python
from peak_engines import RidgeRegressionModel
model_rr1 = RidgeRegressionModel()
model_rr1.fit(X_train, y_train)
print(model_rr1.regularization_)
```
prints:
```
[0.11534536 0.11534536 0.11534536 0.11534536 0.11534536 0.11534536
 0.11534536 0.11534536]
```

Futhermore, RidgeRegressionModel isn't limited to a single regularizer. We can fit models with separate
regularizers for each regressor.
```python
model_rrk = RidgeRegressionModel(grouping_mode='none')
model_rrk.fit(X_train, y_train)
print(model_rrk.regularization_)
```
prints:
```
[ 2.30923985e-01 -8.67658544e+02  6.82272953e+02  1.24145187e+00
  3.75926298e+03  8.97075279e-01  9.67867510e-02  8.47172098e-02]
```

## Validate the models
Let's compare the performance of these models on the validation dataset to least squares and a
dummy model.
```python
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error
model_ls = LinearRegression()
model_ls.fit(X_train, y_train)
model_dummy = DummyRegressor()
model_dummy.fit(X_train, y_train)
print("LS   ", mean_squared_error(y_test, model_ls.predict(X_test))**.5)
print("DUMMY", mean_squared_error(y_test, model_dummy.predict(X_test))**.5)
print("RR1  ", mean_squared_error(y_test, model_rr1.predict(X_test))**.5)
print("RRk  ", mean_squared_error(y_test, model_rrk.predict(X_test))**.5)
```
prints:
```
LS    1.007685362000467
DUMMY 1.160695414363397
RR1   1.1299420329197944
RRk   0.7832389321263064
```
Of course, these results are sensitive to the random split. For a more complete analysis, take a
look at this [notebook](https://github.com/rnburn/peak-engines/blob/master/example/ridge_regression/california_housing.ipynb).
