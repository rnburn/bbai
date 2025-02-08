from .._computation._bridge import gp
from ._covariance_function import RbfCovarianceFunction
from ._bayesian_regression_pdfs import BayesianRegressionPDFs
from ._marginal import Marginal, LogMarginal
from ._marginal_regressor import MarginalRegressor
from ._marginal_sigma2_signal import MarginalSigma2Signal
from ._problem_validation import validate_problem

import scipy.linalg
from collections import namedtuple
import numpy as np
import scipy.special


class BayesianGaussianProcessRegression:
    """Fit a Bayesian Gaussian process regression model with nugget effects and a non-informative 
    reference prior.

    For background on the priors used see

    * Ren, Sun, and He. Objective Bayesian analysis for a spatial model with nugget effects
    https://www.sciencedirect.com/science/article/abs/pii/S037837581200081X
    * Berger, Oliveira, and Sanso. Objective Bayesian Analysis of Spatially Correlated Data
    https://www.tandfonline.com/doi/abs/10.1198/016214501753382282?journalCode=uasa20

    Parameters
    ----------
    kernel :
        Specify the covariance of target values given their associated spatial points.

    length0 : float, default=2.0
        The initial length0 hyperparameter value to start optimization from

    noise_ratio0 : float, default=0.05
        The initial noise_ratio0 hyperparameter value to start optimization from

    Examples
    --------
    >>> from bbai.gp import BayesianGaussianProcessRegression, RbfCovarianceFunction
    >>> import numpy as np
    >>> 
    >>> # Make an interpolation data set
    >>> def make_dataset():
    >>>     def target_generator(rng, X):
    >>>         target = -2.0 + 0.5*X**2 + np.sin(3 * X)
    >>>         target += rng.normal(0, 0.3, size=target.shape)
    >>>         return target.squeeze()
    >>>     rng = np.random.RandomState(0)
    >>>     N = 20
    >>>     Z = rng.uniform(0, 5, size=N).reshape(-1, 1)
    >>>     y = target_generator(rng, Z)
    >>>     X = np.hstack((Z**2, np.ones((N, 1))))
    >>>     return Z, X, y
    >>> 
    >>> # Fit a Bayesian Gaussian process model
    >>> Z, X, y = make_dataset()
    >>> model = BayesianGaussianProcessRegression(kernel=RbfCovarianceFunction())
    >>> model.fit(Z, y, design_matrix = X)
    >>> 
    >>> # Print out the median of the hyperparameters
    >>> print('length median =', model.marginal_length_.ppf(0.5))
    >>> print('noise_ratio median =', model.marginal_noise_ratio_.ppf(0.5))
    >>> print('sigma2_signal median =', model.marginal_sigma2_signal_.ppf(0.5))
    """
    def __init__(self,
            kernel=RbfCovarianceFunction(),
            length0=2.0,
            noise_ratio0=0.05,
            tolerance=1.0e-4):
        self.params_ = {}
        self.set_params(
                kernel=kernel,
                length0=length0,
                noise_ratio0=noise_ratio0,
                tolerance=tolerance,
        )

    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return self.params_

    def set_params(self, **parameters):
        """Set parameters for this estimator."""
        for parameter, value in parameters.items():
            self.params_[parameter] = value

    def fit(self, sample_matrix, y, design_matrix=None):
        """Fit the model to the training data."""
        sample_matrix = np.array(sample_matrix, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        if not design_matrix is None:
            design_matrix = np.array(design_matrix, dtype=np.float64)
        else:
            design_matrix = np.zeros((0, 0))
        num_regressors = design_matrix.shape[1]
        validate_problem(sample_matrix, design_matrix, y)
        length0 = self.params_['length0']
        noise_ratio0 = self.params_['noise_ratio0']
        kernel = self.params_['kernel']
        response = gp.fit_regression_bayesian(dict(
                tolerance = self.params_['tolerance'],
                covariance_function_id = kernel.id_,
                covariance_function_parameter_vector = kernel.params_,
                sample_matrix = sample_matrix,
                design_matrix = design_matrix,
                target_vector = y,
                hyperparameter0_vector = np.log(np.array([length0, noise_ratio0])),
        ))
        self.train_sample_matrix_ = sample_matrix
        self.predictor_ = response['predictor']
        self.hyperparameter_matrix_ = response['hyperparameter_matrix']
        self.weight_vector_ = response['weight_vector']

        m = response['marginal_log_length']
        self.marginal_log_length_ = Marginal(
                response['log_length'],
                m['a'], m['b'],
                response['marginal_point_vector'],
                response['marginal_integral_point_vector'],
                m['value_vector'], m['integral_vector'],
        )
        self.marginal_length_ = LogMarginal(self.marginal_log_length_)
        self.marginal_regressors_ = []
        for j in range(num_regressors):
            self.marginal_regressors_.append(
                    MarginalRegressor(
                        response['weight_vector'],
                        response['beta_hat_matrix'],
                        response['axi_diagonal_matrix'],
                        response['s2_vector'],
                        len(y) - num_regressors,
                        j
                    )
            )

        m = response['marginal_log_noise_ratio']
        self.marginal_log_noise_ratio_ = Marginal(
                response['log_noise_ratio'],
                m['a'], m['b'],
                response['marginal_point_vector'],
                response['marginal_integral_point_vector'],
                m['value_vector'], m['integral_vector'],
        )
        self.marginal_noise_ratio_ = LogMarginal(self.marginal_log_noise_ratio_)

        self.marginal_sigma2_signal_ = MarginalSigma2Signal(
                response['weight_vector'],
                response['s2_vector'],
                len(y) - num_regressors,
        )

        self.length_mode_ = np.exp(response['log_length'])
        self.noise_ratio_mode_ = np.exp(response['log_noise_ratio'])
        return self

    def predict(self, sample_matrix, design_matrix=None, with_pdf=False):
        """Predict target values."""
        sample_matrix = np.array(sample_matrix, dtype=np.float64)
        if design_matrix is not None:
            design_matrix = np.array(design_matrix, dtype=np.float64)
        else:
            design_matrix = np.zeros((len(sample_matrix), 0))
        kernel = self.params_['kernel']
        response = gp.predict_regression_bayesian(dict(
                covariance_function_id = kernel.id_,
                covariance_function_parameter_vector = kernel.params_,
                train_sample_matrix = self.train_sample_matrix_,
                sample_matrix = sample_matrix,
                design_matrix = design_matrix,
                predictor = self.predictor_,
                compute_pdf = with_pdf,
        ))
        if not with_pdf:
            return response['prediction_mean_vector']
        pdfs = []
        num_train = len(self.train_sample_matrix_)
        num_regressors = design_matrix.shape[1]
        pdfs = BayesianRegressionPDFs(
                response['pdf_matrix'],
                num_train - num_regressors,
        )
        return response['prediction_mean_vector'], pdfs
