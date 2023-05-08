from collections import namedtuple

ErrorResponse = namedtuple('ErrorResponse', ['message'])

fit_glm_fields = [
    'hyperparameters',
    'aloocv',
    'aloocvs',
    'weights',
    'intercepts',
    'raw_weights',
]
FitGlmResponse = namedtuple('FitGlmResponse', fit_glm_fields)
FitGlmResponse.__new__.__defaults__ = (None,) * len(fit_glm_fields)

fit_glm_map_fields = [
        'weights',
        'hessian',
]
FitGlmMAPResponse = namedtuple('FitGlmMAPResponse', fit_glm_map_fields)
FitGlmMAPResponse.__new__.__defaults__ = (None,) * len(fit_glm_map_fields)

fit_bayesian_glm_fields = [
    'weight_mean_vector',
    'weight_covariance_matrix',
    'noise_variance_mean',
]
FitBayesianGlmResponse = namedtuple('FitBayesianGlmResponse', fit_bayesian_glm_fields)
FitBayesianGlmResponse.__new__.__defaults__ = (None,) * len(fit_bayesian_glm_fields)

fit_sparse_glm_fields = [
    'hyperparameters',
    'aloocv',
    'aloocvs',
    'weights',
    'intercepts',
    'raw_weights',
]
FitSparseGlmResponse = namedtuple('FitSparseGlmResponse', fit_sparse_glm_fields)
FitSparseGlmResponse.__new__.__defaults__ = (None,) * len(fit_sparse_glm_fields)

fit_gp_regression_map_fields = [
        'length',
        'noise_ratio',
        'prediction_b_value',
        'hessian',
        'intermediate1_prediction_vector',
        'intermediate2_prediction_matrix',
        'beta_vector',
        'packed_gl_matrix',
]
FitGpRegressionMapResponse = namedtuple('FitGpRegressionMapResponse', fit_gp_regression_map_fields)
FitGpRegressionMapResponse.__new__.__defaults__ = (None,) * len(fit_gp_regression_map_fields)

predict_gp_regression_map_fields = [
        'log_pdf_normalizer',
        'prediction_mean_vector',
        'prediction_packed_r22l_matrix',
]
PredictGpRegressionMapResponse = namedtuple('PredictGpRegressionMapResponse', predict_gp_regression_map_fields)
PredictGpRegressionMapResponse.__new__.__defaults__ = (None,) * len(predict_gp_regression_map_fields)

fit_bayesian_gp_regression_fields = [
        'log_length',
        'log_noise_ratio',
        'weight_vector',
        's2_vector',
        'axi_diagonals',
        'beta_hat_matrix',
        'predictor',
        'hyperparameter_matrix',
        'marginal_point_vector',
        'marginal_integral_point_vector',
        'marginal_log_length',
        'marginal_log_noise_ratio',
]
FitBayesianGpRegressionResponse = namedtuple('FitBayesianGpRegressionResponse', fit_bayesian_gp_regression_fields)
FitBayesianGpRegressionResponse.__new__.__defaults__ = (None,) * len(fit_bayesian_gp_regression_fields)

predict_bayesian_gp_regression_fields = [
        'prediction_mean_vector',
        'pdf_matrix',
]
PredictBayesianGpRegressionResponse = namedtuple(
        'PredictBayesianGpRegressionResponse', predict_bayesian_gp_regression_fields)
PredictBayesianGpRegressionResponse.__new__.__defaults__ = (None,) * len(predict_bayesian_gp_regression_fields)

bayesian_gp_pred_pdf_fields = [
        'res_vector',
]
BayesianGpPredPdfResponse = namedtuple(
        'BayesianGpPredPdfResponse', bayesian_gp_pred_pdf_fields)
BayesianGpPredPdfResponse.__new__.__defaults__ = (None,) * len(bayesian_gp_pred_pdf_fields)
