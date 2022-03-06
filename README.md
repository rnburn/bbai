## bbai
![](https://github.com/rnburn/peak-engines/workflows/CI/badge.svg) [![PyPI version](https://img.shields.io/pypi/v/bbai.svg)](https://badge.fury.io/py/bbai) [![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/) [![API Reference](http://img.shields.io/badge/api-reference-blue.svg)](https://buildingblock.ai/bbai.glm)

Many models, even basic ones, have hyperparameters. 

Hyperparameters are used, among other things, to prevent overfitting and can have a big impact on a model's performance.

Typically, hyperparameters are treated separately from other model parameters and are set using non-deterministic processes. For example, a hyperparameter specifying regularization strength might be set by estimating out-of-sample performance with a randomized cross-validation and searching through the space of parameters with a brute-force or black-box strategy.

But such methods will never produce the *best value* and tweaking is frequently deployed to achieve better performance.

By comparison, this project provides models that use *deterministic*, *exact* algorithms to set hyperparameters. By using exact processes, you can remove tweaking from the model fitting process, and instead focus on model specification.

## Installation

**bbai** supports both Linux and OSX on x86-64.

```
pip install bbai
```

## Usage

### Ridge Regression
Fit a ridge regression model with the regularization parameter *exactly* set so as to minimize mean squared error on a leave-one-out cross-validation of the training data set
```python
# load example data set
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
X, y = load_boston(return_X_y=True)
X = StandardScaler().fit_transform(X)

# fit model
from bbai.glm import RidgeRegression
model = RidgeRegression()
model.fit(X, y)
```

### Logistic Regression
Fit a logistic regression model with the regularization parameter *exactly* set so as to maximize likelihood on an approximate leave-one-out cross-validation of the training data set
```python
# load example data set
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
X, y = load_breast_cancer(return_X_y=True)
X = StandardScaler().fit_transform(X)

# fit model
from bbai.glm import LogisticRegression
model = LogisticRegression()
model.fit(X, y)
```

### Bayesian Ridge Regression
Fit a Bayesian ridge regression model where the hyperparameter controlling the regularization strength is integrated over.
```python
# load example data set
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
X, y = load_boston(return_X_y=True)
X = StandardScaler().fit_transform(X)

# fit model
from bbai.glm import BayesianRidgeRegression
model = BayesianRidgeRegression()
model.fit(X, y)
```

### Logistic Regression MAP with Jeffreys Prior
Fit a logistic regression MAP model with Jeffreys prior.
```python
# load example data set
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
X, y = load_breast_cancer(return_X_y=True)
X = StandardScaler().fit_transform(X)

# fit model
from bbai.glm import LogisticRegressionMAP
model = LogisticRegressionMAP()
model.fit(X, y)
```

## How it works
* [Optimizing Approximate Leave-one-out Cross-validation to Tune Hyperparameters](https://arxiv.org/abs/2011.10218)
* [An Algorithm for Bayesian Ridge Regression with Full Hyperparameter Integration](https://buildingblock.ai/bayesian-ridge-regression)

## Examples

* [01-digits](https://buildingblock.ai/multinomial-logistic-regression-example): Fit a multinomial logistic regression model to predict digits.
* [02-iris](example/02-iris.py): Fit a multinomial logistic regression model to the Iris data set.
* [03-bayesian](example/03-bayesian.py): Fit a Bayesian ridge regression model with hyperparameter integration.
* [04-curve-fitting](example/04-curve-fitting.ipynb): Fit a Bayesian ridge regression model with hyperparameter integration.
* [05-jeffreys1](example/05-jeffreys1.ipynb): Fit an MAP logistic regression model with Jeffreys prior. Plot out the prior and posterior distributions.

## Documentation

* [buildingblock.ai](https://buildingblock.ai/)
* [Getting Started](https://buildingblock.ai/get-started)
* [Logistic Regression Guide](https://buildingblock.ai/logistic-regression-guide)
* [Reference](https://buildingblock.ai/bbai.glm)
