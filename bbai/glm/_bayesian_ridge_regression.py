from .._computation._bridge import glm

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
        response = glm.fit_bayesian_glm(dict(feature_matrix=X, target_vector=y))
        self.weight_mean_vector_ = response['weight_mean_vector']
        self.coef_ = self.weight_mean_vector_
        self.weight_covariance_matrix_ = response['weight_covariance_matrix']
        self.noise_variance_mean_ = response['noise_variance_mean']

    def predict(self, X, return_std=False):
        """Predict target values."""
        y_pred = np.dot(X, self.weight_mean_vector_) + self.intercept_
        if not return_std:
            return y_pred
        y_stddev = np.zeros(len(y_pred))
        for i, xi in enumerate(X):
            y_stddev[i] = self.noise_variance_mean_
            y_stddev[i] += np.dot(xi, np.dot(self.weight_covariance_matrix_, xi))
            y_stddev[i] = np.sqrt(y_stddev[i])
        return y_pred, y_stddev
