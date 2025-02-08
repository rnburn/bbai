from .._computation._bridge import glm
from ._loss_link import LossLink

import numpy as np

class RidgeRegression(object):
    """Implements regularized regression with regularizers fit so
    as to maximize performance on leave-one-out cross-validation.

    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether a constant column should be added to the feature matrix.

    normalize : bool, default=False
        Whether to center and rescale the feature matrix columns.

    alpha : float, default=None
        Regularization strength. If None, bbai will choose alpha so as to maximize performance
        on leave-one-out cross-validation.

    tolerance : float, default=0.0001
        The tolerance for the optimizer to use when deciding to stop the objective. With a lower
        value, the optimizer will be more stringent when deciding whether to stop searching.

    Examples
    --------
    >>> from sklearn.datasets import load_boston
    >>> from bbai import RidgeRegression
    >>> X, y = load_boston(return_X_y=True)
    >>> model = RidgeRegression().fit(X, y) 
                    # Defaults to use the regularizer l2 and finds the
                    # hyperparameters that maximize performance on
                    # leave-one-out cross-validation.
    >>> print(model.alpha_) # print out the hyperparameters
    """
    def __init__(
            self,
            fit_intercept=True,
            normalize=False,
            alpha=None,
            tolerance=0.0001):
        self.params_ = {}
        self.set_params(
                fit_intercept=fit_intercept,
                normalize=normalize,
                alpha=alpha,
                tolerance=tolerance
        )
        if tolerance <= 0:
            raise RuntimeError("invalid tolerance")

    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return self.params_

    def set_params(self, **parameters):
        """Set parameters for this estimator."""
        for parameter, value in parameters.items():
            self.params_[parameter] = value

    def fit(self, X, y):
        """Fit the model to the training data."""
        self._loss_link = 'l2'
        alpha = self.params_['alpha']
        hyperparameters = np.array([])
        if alpha:
            hyperparameters = np.array([np.sqrt(alpha)])
        response = glm.fit_glm(dict(
            loss_link = LossLink.l2,
            normalize = self.params_['normalize'],
            fit_intercept = self.params_['fit_intercept'],
            feature_matrix = X,
            target_vector = y,
            hyperparameter_vector = hyperparameters,
        ))
        self.coef_ = response['weight_matrix'][:, 0]
        self.intercept_ = response['intercept_vector'][0]
        self.alpha_ = response['hyperparameter_vector'][0] ** 2
        return self

    def predict(self, X):
        """Predict target values."""
        return np.dot(X, self.coef_) + self.intercept_
