from .._computation._bridge import glm

import numpy as np

class Lasso:
    """Implements Lasso regression with the regularization parameter fit so
    as to maximize performance on leave-one-out cross-validation.

    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether a constant column should be added to the feature matrix.

    fit_beta_path : bool, default=False
        If lambda is specified, fit the matrix that determines the regressors, beta, for 
        lambda in the range [specified_lambda, oo)

    lambda : float, default=None
        Regularization strength. If None, bbai will choose lambda so as to maximize performance
        on leave-one-out cross-validation.
         
        If X represents the design matrix and y the target vector, then regressors, beta, will 
        determined so that 
              X^T (y - X beta) = lambda gamma
        where 
              gamma_j = sign(beta_j), if beta_j != 0
              gamma_j in [-1, 1], otherwise
        If fit_intercept is true, then the intercept will correspond to an implicit column of
        ones in X with no regularization.

    Examples
    --------
    >>> from sklearn.datasets import load_diabetes
    >>> from bbai.glm import Lasso
    >>> X, y = load_diabetes(return_X_y=True)
    >>> model = Lasso().fit(X, y) 
                    # Fit lasso regression using hyperparameters that maximize performance on
                    # leave-one-out cross-validation.
    >>> print(model.lambda_) # print out the hyperparameter that maximizes
                             # leave-one-out cross validation performance
          # prints: 22.179
    >>> print(model.intercept_, model.coef_) # print out the coefficients that maximizes
                                             # leave-one-out cross validation performance
          # prints: 152.13 [0, -193.9, 521.8, 295.15, -99.28, 0, -222.67, 0, 511.95, 52.85]
    """
    def __init__(self, lambda_=None, fit_intercept=True, fit_beta_path=False):
        self.params_ = {}
        self.set_params(
                lambda_ = lambda_,
                fit_intercept = fit_intercept,
                fit_beta_path = fit_beta_path
        )
        self.coef_ = None

    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return self.params_

    def set_params(self, **parameters):
        """Set parameters for this estimator."""
        for parameter, value in parameters.items():
            self.params_[parameter] = value

    def fit(self, X, y):
        """Fit the model to the training data."""
        assert X.shape[0] == y.shape[0] and X.shape[1] > 0
        lambda_ = self.params_['lambda_']
        fit_intercept = self.params_['fit_intercept']
        fit_beta_path = self.params_['fit_beta_path']
        if lambda_ is None:
            res = glm.loo_lasso_fit(X, y, fit_intercept)
            self.cost_fn_ = res['cost_function']
            self.lambda_ = res['lambda_opt']
            self.loo_mse_ = res['cost_opt'] / len(y)
            beta = res['beta_opt']
        else:
            res = glm.lasso_fit(X, y, lambda_, fit_intercept, fit_beta_path)
            beta = res['beta']
            self.lambda_ = lambda_
            self.beta_path_ = res['beta_path']

        self.beta_ = beta
        if fit_intercept: 
            self.intercept_ = beta[0]
            self.coef_ = beta[1:]
        else:
            self.coef_ = beta
            self.intercept_ = 0
        return self

    def predict(self, Xp):
        """Predict target values."""
        fit_intercept = self.params_['fit_intercept']
        return self.intercept_ + np.dot(Xp, self.coef_)

    def evaluate_lo_cost(self, lda):
        """Evaluate the cost at an arbitrary value of lambda.

        The cost function is produced as an artifact of optimization when lambda=None
        """
        # Note: the evaluation function isn't very efficient. It's used mainly for testing.
        # 
        # If you want more efficient evaluation, you should extract the segments of polynomial and
        # use them directly.
        N = self.cost_fn_.shape[1]
        if lda >= self.cost_fn_[0, N-1]:
            return self.cost_fn_[1, N-1]
        for j in range(N-1):
            if lda >= self.cost_fn_[0, j+1]:
                continue
            a0, a1, a2 = self.cost_fn_[1:, j]
            return a0 + a1 * lda + a2 * lda**2
