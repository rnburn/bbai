## bbai
![](https://github.com/rnburn/peak-engines/workflows/CI/badge.svg) [![PyPI version](https://img.shields.io/pypi/v/bbai.svg)](https://badge.fury.io/py/bbai) [![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/) [![API Reference](http://img.shields.io/badge/api-reference-blue.svg)](https://buildingblock.ai/bbai.glm)

Deterministic, exact algorithms for objective Bayesian inference and hyperparameter optimization.

## Installation

**bbai** supports both Linux and OSX on x86-64.

```
pip install bbai
```

## Usage

### Fit Lasso Regression to Optimize LOOCV
Efficiently fit a lasso regression model with λ set so as to optimize performance on 
a leave-one-out cross-validation of the data set.

```python
from sklearn.datasets import load_diabetes
from bbai.glm import Lasso
X, y = load_diabetes(return_X_y=True)
model = Lasso().fit(X, y) 
            # Fit lasso regression using hyperparameters that maximize performance on
            # leave-one-out cross-validation.
print(model.lambda_) # print out the hyperparameter that maximizes
                     # leave-one-out cross validation performance
  # prints: 22.179
print(model.intercept_, model.coef_) # print out the coefficients that maximizes
                                     # leave-one-out cross validation performance
  # prints: 152.13 [0, -193.9, 521.8, 295.15, -99.28, 0, -222.67, 0, 511.95, 52.85]
```
See [24-lasso-diabetes](example/24-lasso-diabetes.ipynb) for the full example.

### Fully Bayesian Single-variable Logistic Regression with Reference Prior
https://www.objectivebayesian.com/p/election-2024

Build a fully Bayesian logistic regression model with a single unknown weight using Jeffreys
prior (or reference prior, which are the same for only a single parameter).

```python
from bbai.glm import BayesianLogisticRegression1

x = [-5, 2, 8, 1]
y = [0, 1, 0, 1]
model = BayesianLogisticRegression1()

# Fit a posterior distribution for w with the logistic
# regression reference prior
model.fit(x, y)

# Print the posterior probability that w < 0.123
print(model.cdf(0.123))
```


### Hypothesis testing using Expected Encompassing Intrinsic Bayes Factors (EEIBF)
https://www.objectivebayesian.com/p/hypothesis-testing

The EEIBF method is described in the paper *Default Bayes Factors for Nonnested Hypothesis Testing*
by James Berger and Julia Mortera ([postscript](http://www2.stat.duke.edu/~berger/papers/mortera.ps)).

The python code below shows how to test these three hypotheses for the mean of normally distributed 
data with unknown variance.
```
H_equal: mean = 0
H_left: mean < 0
H_right: mean > 0
```
```python
from bbai.stat import NormalMeanHypothesis
import numpy as np

np.random.seed(0)
data = np.random.normal(0.123, 1.5, size=9)
probs = NormalMeanHypothesis().test(data)
print(probs.equal) # posterior probability for H_equal 0.235
print(probs.left) # posterior probability for H_left 0.0512
print(probs.right) # posterior probability for H_right 0.713
```
See [example/19-hypothesis-first-t.ipynb](example/19-hypothesis-first-t.ipynb) for an example and
[example/18-hypothesis-eeibf-validation.ipynb](example/18-hypothesis-eeibf-validation.ipynb) for a
step-by-step validation of the method against the paper.

### Objective Bayesian inference for comparing binomial proportions
https://www.objectivebayesian.com/p/binomial-comparison

Fit a posterior distribution with a reference prior to compare binomial proportions:
```python
from bbai.model import DeltaBinomialModel

# Some example data
a1, b1, a2, b2 = 5, 3, 2, 7

# Fit a posterior distribution with likelihood function
#     L(theta, x) = (theta + x)^a1 * (1 - theta - x)^b1 * x^a2 (1-x)^b2
# where theta represents the difference of the two binomial distribution probabilities
model = DeltaBinomialModel(prior='reference')
model.fit(a1, b1, a2, b2)

# Print the probability that theta < 0.123
print(model.cdf(0.123))
     # Prints 0.10907436812863071
```

### Efficient approximation of multivariable functions using adaptive sparse grids at Chebyshev nodes.
```python
from bbai.numeric import SparseGridInterpolator
import numpy as np

# A test function
def f(x, y, z):
    t1 = 0.68 * np.abs(x - 0.3)
    t2 = 1.25 * np.abs(y - 0.15)
    t3 = 1.86 * np.abs(z - 0.09)
    return np.exp(-t1 - t2 - t3)

# Fit a sparse grid to approximate f
ranges = [(-2, 5), (1, 3), (-2, 2)]
interp = SparseGridInterpolator(tolerance=1.0e-4, ranges=ranges)
interp.fit(f)
print('num_pts =', interp.points.shape[1])
    # prints 10851

# Test the accuracy at a random point of the domain
print(interp.evaluate(1.84, 2.43, 0.41), f(1.84, 2.43, 0.41))
#    prints 0.011190847391188667 0.011193746554063376

# Integrate the approximation over the range
print(interp.integral)
#    prints 0.6847335267327939
```
### Objective Bayesian Inference for Gaussian Process Models
Construct prediction distributions for Gaussian process models using full integration over the
parameter space with a noninformative, reference prior.
```python
import numpy as np
from bbai.gp import BayesianGaussianProcessRegression, RbfCovarianceFunction

# Make an example data set
def make_location_matrix(N):
    res = np.zeros((N, 1))
    step = 1.0 / (N - 1)
    for i in range(N):
        res[i, 0] = i * step
    return res
def make_covariance_matrix(S, sigma2, theta, eta):
    N = len(S)
    res = np.zeros((N, N))
    for i in range(N):
        si = S[i]
        for j in range(N):
            sj = S[j]
            d = np.linalg.norm(si - sj)
            res[i, j] = np.exp(-0.5*(d/theta)**2)
        res[i, i] += eta
    return sigma2 * res
def make_target_vector(K):
    return np.random.multivariate_normal(np.zeros(K.shape[0]), K)
np.random.seed(0)
N = 20
sigma2 = 25
theta = 0.01
eta = 0.1
params = (sigma2, theta, eta)
S = make_location_matrix(N)
K = make_covariance_matrix(S, sigma2, theta, eta)
y = make_target_vector(K)

# Fit a Gaussian process model to the data
model = BayesianGaussianProcessRegression(kernel=RbfCovarianceFunction())
model.fit(S, y)

# Construct the prediction distribution for x=0.1
preds, pred_pdfs = model.predict([[0.1]], with_pdf=True)
high, low = pred_pdfs.ppf(0.75), pred_pdfs.ppf(0.25)

# Print the mean and %25-%75 credible set of the prediction distribution
print(preds[0], '(%f to %f)' % (low, high))
```

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
* [Deterministic Objective Bayesian Inference for Spatial Models](https://buildingblock.ai/bayesian-gaussian-process.pdf)
* [Optimizing Approximate Leave-one-out Cross-validation to Tune Hyperparameters](https://arxiv.org/abs/2011.10218)
* [An Algorithm for Bayesian Ridge Regression with Full Hyperparameter Integration](https://buildingblock.ai/bayesian-ridge-regression)
* [How to Fit Logistic Regression with a Noninformative Prior](https://buildingblock.ai/logistic-regression-jeffreys)

## Examples

* [01-digits](https://buildingblock.ai/multinomial-logistic-regression-example): Fit a multinomial logistic regression model to predict digits.
* [02-iris](example/02-iris.py): Fit a multinomial logistic regression model to the Iris data set.
* [03-bayesian](example/03-bayesian.py): Fit a Bayesian ridge regression model with hyperparameter integration.
* [04-curve-fitting](example/04-curve-fitting.ipynb): Fit a Bayesian ridge regression model with hyperparameter integration.
* [05-jeffreys1](example/05-jeffreys1.ipynb): Fit a logistic regression MAP model with Jeffreys prior and a single regressor.
* [06-jeffreys2](example/06-jeffreys2.ipynb): Fit a logistic regression MAP model with Jeffreys prior and two regressors.
* [07-jeffreys-breast-cancer](example/07-jeffreys-breast-cancer.py): Fit a logistic regression MAP model with Jeffreys prior to the breast cancer data set.
* [08-soil-cn](example/08-soil-cn.ipynb): Fit a Bayesian Gaussian process with a non-informative prior to a data set of soil carbon-to-nitrogen samples.
* [11-meuse-zinc](example/11-meuse-zinc.ipynb): Fit a Bayesian Gaussian process with a non-informative prior to a data set of zinc concetrations taken in a flood plain of the Meuse river.
* [13-sparse-grid](example/13-sparse-grid.ipynb): Build adaptive sparse grids for interpolation and integration.

## Documentation

* [buildingblock.ai](https://buildingblock.ai/)
* [Getting Started](https://buildingblock.ai/get-started)
* [Logistic Regression Guide](https://buildingblock.ai/logistic-regression-guide)
* [Reference](https://buildingblock.ai/bbai.glm)
