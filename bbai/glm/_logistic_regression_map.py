from .._computation._bridge import glm

import numpy as np
import scipy.linalg

class LogisticRegressionMAP(object):
    """Implements logistic regression with weights chosen so as to maximize a 
    posterior distribution (Maximum a Posteriori Probability or MAP) using Jeffrey's prior.

    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether constant columns should be added to the feature matrix.

    tolerance : float, default=0.0001
        The tolerance for the optimizer to use when deciding to stop the objective. With a lower
        value, the optimizer will be more stringent when deciding whether to stop searching.

    Examples
    --------
    >>> from sklearn.datasets import load_breast_cancer
    >>> from bbai.glm import LogisticRegressionMAP
    >>> X, y = load_breast_cancer(return_X_y=True)
    >>> model = LogisticRegressionMAP(normalize=True).fit(X, y)
    >>> print(model.coef_) # prints out the MAP weights
    >>> print(model.laplace_covariance_matrix_)
            # prints out the covariance matrix for the Laplace approximation
            # at the MAP estimate -- can be used to obtain error estimates of the 
            # weights.
    """
    def __init__(
            self,
            fit_intercept=True,
            tolerance=0.0001):
        self.params_ = {}
        self.set_params(
                fit_intercept=fit_intercept,
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
        y = np.array(y, dtype=np.float64)
        for yi in y:
            if yi != 0 and yi != 1:
                raise RuntimeError('y must be 0 or 1')
        n, p = X.shape
        fit_intercept = self.params_['fit_intercept']
        X_p = X
        if fit_intercept:
            X_p = np.hstack((X, np.ones((n, 1))))
        if X_p.shape[1] > n:
            raise RuntimeError('only supports n <= p')
        response = glm.fit_map(dict(
                feature_matrix = X_p,
                target_vector = y,
        ))
        w = response['weight_vector']
        intercept = 0.0
        if fit_intercept:
            intercept = w[-1]
            w = w[:-1]
        self.coef_ = np.array([w])
        self.intercept_ = [intercept]
        self.hessian_ = response['hessian']
        self.hessian_inverse_ = None


    def predict(self, X):
        """Predict class labels for the given feature matrix."""
        predict_proba = self.predict_proba(X)
        result = np.zeros(len(X))
        for i, predi in enumerate(predict_proba):
            result[i] = self.classes_[np.argmax(predi)]
        return result

    def predict_log_proba(self, X):
        """Predict class log probabilities for the given feature matrix."""
        return np.log(self.predict_proba(X))

    def predict_proba(self, X):
        """Predict class probabilities for the given feature matrix."""
        u = np.dot(X, self.coef_[0, :]) + self.intercept_[0]
        t = 1 / (1 + np.exp(-u))
        result = np.zeros((len(t), 2), dtype=np.double)
        result[:, 0] = 1 - t
        result[:, 1] = t
        return result

    @property
    def laplace_covariance_matrix_(self):
        """Return the covariance matrix of the Laplace approximation about the MAP estimate."""
        if self.hessian_inverse_ is not None:
            return self.hessian_inverse_
        L, _ = scipy.linalg.lapack.dpotrf(self.hessian_)
        K, _ = scipy.linalg.lapack.dpotri(L)
        self.hessian_inverse_ = K + np.triu(K, k=1).T
        return self.hessian_inverse_
        
