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
