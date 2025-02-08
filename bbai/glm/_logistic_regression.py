from .._computation._bridge import glm
from ._loss_link import LossLink

import numpy as np

def _get_hyperparameters(penalty, C, alpha, beta):
    if C is None and alpha is None:
        return np.array([])
    if penalty != 'elasticnet':
        if alpha:
            return np.array([np.sqrt(alpha)])
        return np.array([1 / np.sqrt(2 * C)])
    return np.array([np.sqrt(alpha), np.sqrt(beta)])

class LogisticRegression(object):
    """Implements logistic regression with regularizers fit so
    as to maximize performance on approximate leave-one-out cross-validation.

    See https://arxiv.org/abs/2011.10218 for background on the approach.

    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether constant columns should be added to the feature matrix.

    normalize : bool, default=False
        Whether to center and rescale the feature matrix columns.

    penalty : {'l2'}, default='l2'
        Regularization function to use

        - 'l2' will use the function sum_i alpha \|w_i\|^2

    C: float, default=None
        Inverse of regularization strength. If None, bbai will choose C so as to maximize
        performance on approximate leave-one-out cross-validation.

    tolerance : float, default=0.0001
        The tolerance for the optimizer to use when deciding to stop the objective. With a lower
        value, the optimizer will be more stringent when deciding whether to stop searching.

    Examples
    --------
    >>> from sklearn.datasets import load_breast_cancer
    >>> from bbai.glm import LogisticRegression
    >>> X, y = load_breast_cancer(return_X_y=True)
    >>> model = LogisticRegression(normalize=True).fit(X, y)
                    # Defaults to use the regularizer l2 and finds the
                    # hyperparameters that maximize performance on
                    # approximate leave-one-out cross-validation.
    >>> print(model.C_) # print out the hyperparameters
    """
    def __init__(
            self,
            fit_intercept=True,
            normalize=False,
            penalty='l2',
            active_classes = 'auto',
            C=None,
            alpha=None, beta=None,
            weights0=None,
            tolerance=0.0001):
        self.params_ = {}
        self.set_params(
                fit_intercept=fit_intercept,
                normalize=normalize,
                penalty=penalty,
                active_classes=active_classes,
                C=C,
                alpha=alpha,
                beta=beta,
                weights0=weights0,
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
        self.classes_ = list(sorted(set(y)))
        self._loss_link = 'multinomial_logistic'
        loss_link = LossLink.multinomial_logistic
        if self.params_['active_classes'] == 'm1' or len(self.classes_) == 2:
            self._loss_link = 'multinomial_logistic_m1'
            loss_link = LossLink.multinomial_logistic_m1
        self._num_active_classes = len(self.classes_)
        if self._loss_link == 'multinomial_logistic_m1':
            self._num_active_classes -= 1
        C = self.params_['C']
        alpha = self.params_['alpha']
        beta = self.params_['beta']
        hyperparameters = _get_hyperparameters('l2', C, alpha, beta)
        response = glm.fit_glm(dict(
            loss_link = loss_link,
            normalize = self.params_['normalize'],
            fit_intercept = self.params_['fit_intercept'],
            feature_matrix = X,
            target_vector = y,
            hyperparameter_vector = hyperparameters,
        ))
        self.aloocv_ = response['aloocv']
        self.aloocvs_ = response['aloocv_vector']
        self.coef_ = response['weight_matrix'].T
        self.intercept_ = response['intercept_vector']
        if self._num_active_classes == 1:
            # we invert so as to match the conventional way of representing weights
            self.coef_ = -self.coef_
            self.intercept_ = -self.intercept_
        self.alpha_ = response['hyperparameter_vector'][0] ** 2
        self.C_ = 0.5 / self.alpha_
        return self


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
        if len(X.shape) != 2 or X.shape[1] != self.coef_.shape[1]:
            raise RuntimeError("invalid X")
        if self._num_active_classes == 1:
            return self._predict_proba_binomial(X)
        return self._predict_proba_multinomial(X)

    def _predict_proba_binomial(self, X):
        u = np.dot(X, self.coef_[0, :]) + self.intercept_[0]
        t = 1 / (1 + np.exp(-u))
        result = np.zeros((len(t), 2), dtype=np.double)
        result[:, 0] = 1 - t
        result[:, 1] = t
        return result

    def _predict_proba_multinomial(self, X):
        U = np.dot(X, self.coef_.T) + self.intercept_
        if self._num_active_classes < len(self.classes_):
            U = np.hstack((U, np.zeros((U.shape[0], 1))))
        for i, ui in enumerate(U):
            exp_ui = np.exp(ui)
            U[i, :] = exp_ui / np.sum(exp_ui)
        return U
