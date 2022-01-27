from .._computation._computation_handle import get_computation_handle

import numpy as np

class BayesianRidgeRegression(object):
    """Implements an algorithm for regularized bayesian linear regression fit. It uses a
    hyperparameter to specify regularization strength and performs full hyperparameter regularization
    with an approximately noninformative prior.

    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether a constant column should be added to the feature matrix.

    Examples
    --------
    >>> from sklearn.datasets import load_boston
    >>> from bbai import RidgeRegression
    >>> X, y = load_boston(return_X_y=True)
    >>> model = BayesianRidgeRegression().fit(X, y) 
    >>> print(model.weight_mean_vector_) # print out expected weight vector
    >>> print(model.noise_variance_mean_) # print out expected noise variance
    >>> print(model.weight_covariance_matrix_) # print out weight covariance matrix
    """
    def __init__(
            self,
            fit_intercept=True):
        self.params_ = {}
        self._handle = get_computation_handle()
        self.set_params(
                fit_intercept=fit_intercept,
        )

    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return self.params_

    def set_params(self, **parameters):
        """Set parameters for this estimator."""
        for parameter, value in parameters.items():
            self.params_[parameter] = value

    def fit(self, X, y):
        """Fit the model to the training data."""
        self.intercept_ = 0.0
        if self.params_['fit_intercept']:
            self.intercept_ = np.mean(y)
        y -= self.intercept_
        response = self._handle.fit_bayesian_glm(
                X = X,
                y = y,
        )
        self.weight_mean_vector_ = response.weight_mean_vector
        self.weight_covariance_matrix_ = response.weight_covariance_matrix
        self.noise_variance_mean_ = response.noise_variance_mean

    def predict(self, X):
        """Predict target values."""
        return np.dot(X, self.weight_mean_vector_) + self.intercept_
