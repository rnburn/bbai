# peak-engines 
![](https://github.com/rnburn/peak-engines/workflows/CI/badge.svg) [![PyPI version](https://img.shields.io/pypi/v/peak-engines.svg)](https://badge.fury.io/py/peak-engines) [![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/) [![API Reference](http://img.shields.io/badge/api-reference-blue.svg)](https://github.com/rnburn/peak-engines/blob/master/doc/Reference.pdf)

**peak-engines** is machine learning framework that focuses on applying advanced optimization algorithms
to build better models.

## Installation

**peak-engines** supports both OSX and Linux on x86-64.

### Python

```
pip install peak-engines
```

### R

Linux:
```
R --slave -e "install.packages( \
 'https://github.com/rnburn/peak-engines/releases/download/v0.2.7/PeakEngines-linux_0.2.7.tar.gz')"
```

OSX:
```
R --slave -e "install.packages( \
 'https://github.com/rnburn/peak-engines/releases/download/v0.2.7/PeakEngines-osx_0.2.7.tar.gz')"
```

### Fit Logistic Regression Hyperparameters
The leave-one-out cross-validation of logistic regression can be efficiently approximated. At a 
high level, this is how it works: For given hyperparameters `C`, we
1. Find the parameters `b` that optimize logistic regression for the given `C`.
2. For each data index `i`, we compute the hessian `H_{-i}` and gradient `g_{-i}` of the 
log-likelihood with the ith data entry removed. (We can reuse the hessian computed from (1) to do this
with a minimal amount of work.)
3. We apply the [Matrix Inversion Lemma](https://en.wikipedia.org/wiki/Woodbury_matrix_identity) to
efficiently compute the inverse `H_{-i}^{-1}`.
4. We use `H_{-i}^{-1}` and `g_{-i}` to take a single step of [Newton's method](https://en.wikipedia.org/wiki/Newton%27s_method_in_optimization) to approximate the logistic regression coefficients with the ith entry removed `b_{-i}`.
5. And finally, we used the `b_{-i}`'s to approximate the out-of-sample predictions and estimate the leave-one-out cross-validation.

See the paper 
[A scalable estimate of the out-of-sample prediction error via approximate leave-one-out](https://arxiv.org/abs/1801.10243) 
by Kamiar Rad and Arian Maleki for more details.

We can, furthermore, differentiate the Approximate Leave-One-Out metric with respect to the hyperparameters and quickly climb to the best performing `C`. Here's how to do it with **peak-engines**:

Load an example dataset
```python
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
X, y = load_breast_cancer(return_X_y=True)
X = StandardScaler().fit_transform(X)
```
Find the best performing `C`
```python
model = peak_engines.LogisticRegressionModel()
model.fit(X, y)
print('C =', model.C_[0])
```
prints
```
C = 0.66474879
```
If we compute the LOOCV by brute force and compare to the ALOOCV, we can see how accurate the approximation is

![alt text](https://raw.githubusercontent.com/rnburn/peak-engines/master/images/logistic-regression-aloocv.png "Aloocv")

## Fit Ridge Regression Hyperparameters
By expressing cross-validation as an optimization objective and computing derivatives, 
**peak-engines** is able to efficiently find regularization parameters that lead to the best
score on a leave-one-out or generalized cross-validation. It, futhermore, scales to handle 
multiple regularizers. Here's an example of how it works
```python
import numpy as np
from sklearn.datasets import load_boston
X, y = load_boston(return_X_y=True)
from peak_engines import RidgeRegressionModel
model = RidgeRegressionModel(normalize=True)
# Fit will automatically find the best alpha that minimizes the Leave-one-out Cross-validation.
# when you call fit. There's no need to provide a search space because peak_engines optimizes the
# LOOCV directly. It the computes derivatives of the LOOCV with respect to the hyperparameters and
# is able to quickly zero in on the best alpha.
model.fit(X, y)
print('alpha =', model.alpha_)
```
prints
```
alpha = 0.009274259071634289
```

## Fit Warped Linear Regression
Let *X* and *y* denote the feature matrix and target vector of a regression dataset. Under the
assumption of normally distributed errors, Ordinary Least Squares (OLS) finds the linear model
that maximizes the likelihood of the dataset

<p align="center">
  <img src="https://raw.githubusercontent.com/rnburn/peak-engines/master/images/ols-likelihood.png" height="66" width="381">
</p>

What happens when errors aren't normally distributed? Well, the model will be misspecified and 
there's no reason to think its likelihood predictions will be accurate. This is where 
Warped Linear Regression can help. It introduces an extra step to OLS where it transforms the 
target vector using a malleable, monotonic function *f* parameterized by *ψ* and adjusts the parameters to
maximize the likelihood of the transformed dataset

<p align="center">
  <img src="https://raw.githubusercontent.com/rnburn/peak-engines/master/images/ols-likelihood-transformed.png" height="66" width="542">
</p>

By introducing the additional transformation step, Warped Linear Regression is more general-purpose
than OLS while still retaining the strong structure and interpretability.  Here's how you use it

Load an example dataset
```python
from sklearn.datasets import load_boston
X, y = load_boston(return_X_y=True)
```

Fit a warped linear regression model
```python
import peak_engines
model = peak_engines.WarpedLinearRegressionModel()
model.fit(X_train, y_train)
```
Visualize the warping function
```python
import numpy as np
import matplotlib.pyplot as plt
y_range = np.arange(np.min(y), np.max(y), 0.01)
z = model.warper_.compute_latent(y_range)
plt.plot(y_range, z)
plt.xlabel('Median Housing Value in $1000s')
plt.ylabel('Latent Variable')
plt.scatter(y, model.warper_.compute_latent(y))
```
![alt text](https://raw.githubusercontent.com/rnburn/peak-engines/master/images/getting_started_warp.png "Warping Function")

## Tutorials

* [Ridge Regression](https://github.com/rnburn/peak-engines/blob/master/tutorial/Ridge-Regression-Tutorial.md)
* [Warped Linear Regression](https://towardsdatascience.com/how-to-build-a-warped-linear-regression-model-3e778e30a201)

## Articles

* [Optimizing Approximate Leave-one-out Cross-validation to Tune Hyperparameters](https://arxiv.org/abs/2011.10218)
* [How to Do Ridge Regression Better](https://towardsdatascience.com/how-to-do-ridge-regression-better-34ecb6ee3b12)
* [What Form of Cross-Validation Should You Use?](https://medium.com/p/what-form-of-cross-validation-should-you-use-76aaecc45c75?source=email-f55ad0a8217--writer.postDistributed&sk=a63ac2a04e49a12e7aa4c12a75b18502)
* [What to Do When Your Model Has a Non-Normal Error Distribution](https://medium.com/p/what-to-do-when-your-model-has-a-non-normal-error-distribution-f7c3862e475f?source=email-f55ad0a8217--writer.postDistributed&sk=f3d494b5f5a8b593f404e7af19a2fb37)

## Examples

* [example/ridge_regression/california_housing.ipynb](https://github.com/rnburn/peak-engines/blob/master/example/ridge_regression/california_housing.ipynb):
  Measure the performance of different ridge regression models for predicting housing values.
* [example/ridge_regression/pollution.ipynb](https://github.com/rnburn/peak-engines/blob/master/example/ridge_regression/pollution.ipynb):
  Build ridge regression models to predict mortality rates from air quality.
* [example/warped_linear_regression/boston_housing.ipynb](https://github.com/rnburn/peak-engines/blob/master/example/warped_linear_regression/boston_housing.ipynb):
  Build a warped linear regression model to predict housing values.
* [example/warped_linear_regression/abalone.ipynb](https://github.com/rnburn/peak-engines/blob/master/example/abalone.ipynb): 
  Predict the age of sea snails using warped linear regression.

## Documentation
See [doc/Reference.pdf](https://github.com/rnburn/peak-engines/blob/master/doc/Reference.pdf)
