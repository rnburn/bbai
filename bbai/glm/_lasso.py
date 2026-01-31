from .._computation._bridge import glm
from .._computation._bridge import mdlls
import numpy as np

class _LoErrors:
    def __init__(self, ptr):
        self.ptr_ = ptr

    def segments(self, i):
        return mdlls.lo_errors_segments(self.ptr_, i)

    def evaluate_t(self, t):
        return mdlls.lo_errors_evaluate_t(self.ptr_, t)

    def evaluate_lambda(self, lambda_):
        return mdlls.lo_errors_evaluate_lambda(self.ptr_, lambda_)

class _LoSquaredError:
    def __init__(self, ptr, path, domain):
        self.ptr_ = ptr
        self.path_ = path
        self.domain_ = domain
    
    def evaluate_lambda(self, lambda_):
        if self.domain_ == 't':
            t = self.path_.lambda_to_t(lambda_)
            return self.evaluate_t(t)
        return mdlls.lo_squared_error_evaluate(self.ptr_, -lambda_)

    def evaluate_t(self, t):
        if self.domain_ == 'lambda':
            lda = self.path_.t_to_lambda(t)
            return self.evaluate_lambda(lda)
        return mdlls.lo_squared_error_evaluate(self.ptr_, t)

    def solve_t(self, val):
        res = mdlls.lo_squared_error_solve(self.ptr_, val)
        if self.domain_ == 't':
            return res
        return [self.path_.lambda_to_t(-l) for l in res]

    @property
    def segments_(self):
        return mdlls.lo_squared_error_segments(self.ptr_)

    @property
    def minimums_(self):
        res = mdlls.lo_squared_error_minimums(self.ptr_)
        if self.domain_ == 'lambda':
            res[0, :] = -res[0, :]
        return res

    def plot_points(self, xmin=-np.inf, xmax=np.inf):
        if self.domain_ == 't':
            return self._plot_points(xmin, xmax)
        lambdax, valx = self._plot_points(-xmax, -xmin)
        lambdax = reversed(lambdax)
        valx = reversed(valx)
        lambdax_p = []
        valx_p = []
        for lx, vx in zip(lambdax, valx):
            lambdax_p.append(np.flip(-lx))
            valx_p.append(np.flip(vx))
        return lambdax_p, valx_p

    def _plot_points(self, xmin=-np.inf, xmax=np.inf):
        xs = []
        ys = []
        entries = self.segments_.T
        n = len(entries)
        for i in range(n):
            x = entries[i][0]
            if x > xmax or i == n-1:
                break;
            low = max(x, xmin)
            high, c, b, a = entries[i+1]
            if high < xmin:
                continue
            high = min(high, xmax)
            pts = np.linspace(low, high, num=4)
            def f(t):
                return c + b * t + a * t * t
            vals = f(pts)
            xs.append(pts)
            ys.append(vals)
        return xs, ys

class _LoSolutionPath:
    def __init__(self, ptr):
        self.ptr_ = ptr

    def lambda_to_t(self, lambda_):
        return mdlls.solution_path_t_from_lambda(self.ptr_, lambda_)

    def t_to_lambda(self, t):
        return mdlls.solution_path_lambda_from_t(self.ptr_, t)

    def beta_from_t(self, t):
        return mdlls.solution_path_beta_from_t(self.ptr_, t)

    @property
    def segments_(self):
        return mdlls.solution_path_segments(self.ptr_)


class Lasso:
    """Implements Lasso regression with the regularization parameter fit so
    as to maximize performance on leave-one-out cross-validation.

    See https://arxiv.org/abs/2508.14368

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
        be chosen to minimize

            1/2 || y - X beta ||_2^2 + lambda \sum_j | beta_j |

        If fit_intercept is true, then the intercept will correspond to an implicit column of
        ones in X with no regularization.

    t : float, default=None
        Max l1-norm of regressors.

        If X represents the design matrix and y the target vector, then regressors, beta, will 
        be chosen so to minimize

            || y - X beta ||_2^2 with \sum_j | beta_j | <= t

        If fit_intercept is true, then the intercept will correspond to an implicit column of
        ones in X with no restriction.

    loo_errors : bool, default=False
        Compute the leave-one-out error function.

    early_exit_threshold : float, default=np.inf
        As the segments of the leave-one-out cross-validation are filled in, exit early if

            ((leave-one-out error at last segment endpoint) - (best leave-one-out error)) / (best leave-one-out error)
               > (early_exit_threshold).
        
        For data sets where fitting the full leave-one-out cost is too expensive; early_exit_threshold 
        allows the algorithm to abort early if it finds a suitable local optimum.

    loo_mode : str, default='lambda'
        If loo_mode=='lambda', select lambda to minimize leave-one-out cross validation with the
        parameter lambda fixed. If loo_mode=='t', select t to minimize leave-one-out
        cross-validation error with the parameter t fixed.

    fit_beta_path : bool, default=False
        Compute the solution path as a function of t or lambda.

    Examples
    --------
    >>> from sklearn.datasets import load_diabetes
    >>> from bbai.glm import Lasso
    >>> X, y = load_diabetes(return_X_y=True)
    >>> model = Lasso(loo_mode='lambda').fit(X, y) 
                    # Fit lasso regression using hyperparameters that maximize performance on
                    # leave-one-out cross-validation.
    >>> print(model.lambda_) # print out the hyperparameter that maximizes
                             # leave-one-out cross validation performance
          # prints: 22.179
    >>> print(model.intercept_, model.coef_) # print out the coefficients that maximizes
                                             # leave-one-out cross validation performance
          # prints: 152.13 [0, -193.9, 521.8, 295.15, -99.28, 0, -222.67, 0, 511.95, 52.85]
    """
    def __init__(self, 
                 lambda_=None, 
                 t=None, 
                 fit_intercept=True, 
                 loo_errors=False,
                 early_exit_threshold=np.inf,
                 loo_mode='t',
                 fit_beta_path=False, 
                 graph_file="",
    ):
        self.params_ = {}
        self.set_params(
                lambda_ = lambda_,
                t = t,
                fit_intercept = fit_intercept,
                loo_errors = loo_errors,
                early_exit_threshold = early_exit_threshold,
                loo_mode = loo_mode,
                fit_beta_path = fit_beta_path,
                graph_file = graph_file
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
        n = len(y)

        t = self.params_['t']
        lambda_ = self.params_['lambda_']
        if t is not None or lambda_ is not None:
            return self._lars(X, y)

        mode = self.params_['loo_mode']
        fit_intercept = self.params_['fit_intercept']
        loo_errors = self.params_['loo_errors']
        use_t = 0
        if mode == 't':
            use_t = 1

        res = mdlls.lo_lars(X, y, use_t, fit_intercept, 
                            loo_errors,
                            self.params_['early_exit_threshold'],
                            self.params_['graph_file'])
        self.t_ = res['t_opt']
        self.lambda_ = res['lambda_opt']
        self.loo_mse_ = res['squared_error_opt'] / n
        self.beta_path_ = _LoSolutionPath(res['solution_path'])
        self.loo_squared_error_ = _LoSquaredError(res['lo_squared_error'], self.beta_path_, mode)

        self.t_max_ = mdlls.solution_path_t_max(res['solution_path'])
        self.s_ = self.t_ / self.t_max_

        self.coef_ = res['beta_opt']
        self.beta_ = self.coef_
        self.intercept_ = 0
        if fit_intercept:
            self.intercept_ = self.coef_[0]
            self.coef_ = self.coef_[1:]
        if loo_errors:
            self.loo_errors_ = _LoErrors(res['lo_errors'])

        return self

    def _lars(self, X, y):
        t = self.params_['t']
        lambda_ = self.params_['lambda_']
        fit_beta_path = self.params_['fit_beta_path']
        fit_intercept = self.params_['fit_intercept']

        threshold = lambda_
        use_t = 0
        if t is not None:
            threshold = t
            use_t = True
        res = mdlls.lars(X, y, threshold, use_t, fit_intercept, fit_beta_path)
        self.t_ = res['t']
        self.lambda_ = res['lambda_']
        self.intercept_ = 0
        self.beta_ = res['beta']
        self.coef_ = res['beta']
        if fit_intercept:
            self.intercept_ = self.coef_[0]
            self.coef_ = self.coef_[1:]
        if fit_beta_path:
            self.beta_path_ = _LoSolutionPath(res['solution_path'])

        return self

    def predict(self, Xp):
        """Predict target values."""
        return self.intercept_ + np.dot(Xp, self.coef_)

class LassoAlo:
    """Similar to Lasso but uses approximate leave-one-out cross-validation instead
    of leave-one-out cross-validation (ALO).

    If your data set is too large for Lasso, then ALO can be nearly as good as 
    leave-one-out cross-validation while being significantly less expensive to compute.
    """

    def __init__(self, fit_intercept=True,
                 early_exit_threshold=np.inf
                 ):
        self.params_ = {}
        self.set_params(
                fit_intercept = fit_intercept,
                early_exit_threshold = early_exit_threshold,
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
        n = len(y)

        fit_intercept = self.params_['fit_intercept']
        early_exit_threshold = self.params_['early_exit_threshold']

        res = mdlls.alo_lars(X, y, fit_intercept, early_exit_threshold)

        self.beta_path_ = _LoSolutionPath(res['solution_path'])
        self.loo_squared_error_ = _LoSquaredError(res['lo_squared_error'], self.beta_path_, 'lambda')
        self.lambda_ = res['lambda_opt']
        self.t_ = res['t_opt']
        self.t_max_ = mdlls.solution_path_t_max(res['solution_path'])
        self.loo_mse_ = res['squared_error_opt'] / n
        self.s_ = self.t_ / self.t_max_

        self.coef_ = res['beta_opt']
        self.beta_ = self.coef_
        self.intercept_ = 0
        if fit_intercept:
            self.intercept_ = self.coef_[0]
            self.coef_ = self.coef_[1:]
        return self

    def predict(self, Xp):
        """Predict target values."""
        return self.intercept_ + np.dot(Xp, self.coef_)
