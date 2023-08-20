from .._computation._bridge import gp
from ._regression_map_pdf import RegressionMAPPDF
from ._covariance_function import RbfCovarianceFunction

import numpy as np

class GaussianProcessRegressionMAP(object):
    """Fit a Gaussian process regression model with nugget effects so as to maximize a 
    posterior distribution (Maximum a Posteriori Probability or MAP) using a non-informative
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

    """
    def __init__(
            self,
            use_log_parameters=False,
            kernel=RbfCovarianceFunction(),
            length0=2.0,
            noise_ratio0=0.05):
        self.params_ = {}
        self.set_params(
                kernel=kernel,
                length0=length0,
                noise_ratio0=noise_ratio0,
                use_log_parameters=use_log_parameters,
        )
        self.length_ = None
        self.noise_ratio_ = None
        self.train_sample_matrix_ = None
        self.intermediate1_prediction_vector_ = None
        self.beta_vector_ = None
        self.packed_gl_matrix_ = None

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
        length0 = self.params_['length0']
        noise_ratio0 = self.params_['noise_ratio0']
        kernel = self.params_['kernel']
        response = gp.fit_regression_map(dict(
                covariance_function_id = kernel.id_,
                covariance_function_parameter_vector = kernel.params_,
                sample_matrix = sample_matrix,
                design_matrix = design_matrix,
                target_vector = y,
                hyperparameter0_vector = np.array([length0, noise_ratio0]),
                use_log_parameters = self.params_['use_log_parameters'],
        ))
        self.train_sample_matrix_ = sample_matrix
        self.length_ = response['length']
        self.noise_ratio_ = response['noise_ratio']
        self.prediction_b_value_ = response['prediction_b_value']
        self.hessian_ = response['hessian']
        self.intermediate1_prediction_vector_ = response['intermediate1_prediction_vector']
        self.intermediate2_prediction_matrix_ = response['intermediate2_prediction_matrix']
        self.beta_vector_ = response['beta_vector']
        self.packed_gl_matrix_ = response['packed_gl_matrix']

    def predict(self, sample_matrix, design_matrix=None, with_pdf=False):
        """Predict target values."""
        sample_matrix = np.array(sample_matrix, dtype=np.float64)
        if not design_matrix is None:
            design_matrix = np.array(design_matrix, dtype=np.float64)
        else:
            design_matrix = np.zeros((0, 0))
        kernel = self.params_['kernel']
        response = gp.predict_regression_map(dict(
                prediction_b_value = self.prediction_b_value_,
                covariance_function_id = kernel.id_,
                covariance_function_parameter_vector = kernel.params_,
                train_sample_matrix = self.train_sample_matrix_,
                sample_matrix = sample_matrix,
                design_matrix = design_matrix,
                hyperparameter_vector = np.array([self.length_, self.noise_ratio_]),
                intermediate1_prediction_vector = self.intermediate1_prediction_vector_,
                intermediate2_prediction_matrix = self.intermediate2_prediction_matrix_,
                beta_vector = self.beta_vector_,
                packed_gl_matrix = 
                    self.packed_gl_matrix_ if with_pdf else np.zeros(0),
                compute_pdf = with_pdf,
        ))
        if not with_pdf:
            return response['prediction_mean_vector']
        pdf = RegressionMAPPDF(
                response['prediction_packed_r22l_matrix'],
                self.prediction_b_value_,
                response['log_pdf_normalizer'],
                response['prediction_mean_vector'],
                self.train_sample_matrix_.shape[0] - design_matrix.shape[1],
        )
        return response['prediction_mean_vector'], pdf
