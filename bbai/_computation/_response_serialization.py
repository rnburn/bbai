from . import _header
from . import _socket_utility
from . import _protocol
from . import _response
from ._raw_weights import RawWeights
from ._reader import Reader
import struct
import numpy as np

def read_marginal(reader):
    a = reader.read_double()
    b = reader.read_double()
    value_vector = reader.read_vector()
    integral_vector = reader.read_vector()
    return a, b, value_vector, integral_vector

def read_error_response(reader):
    message = reader.read_string()
    result = _response.ErrorResponse(message)
    return result
    
def read_fit_glm_response(reader):
    hyperparameters = reader.read_vector()
    aloocv = reader.read_double()
    aloocvs = reader.read_vector()
    weight_matrix = reader.read_matrix()
    intercepts = reader.read_vector()
    result = _response.FitGlmResponse(
        hyperparameters = hyperparameters, 
        aloocv = aloocv,
        aloocvs = aloocvs,
        weights = weight_matrix.T,
        intercepts = intercepts,
    )
    return result

def read_fit_glm_map_response(reader):
    weights = reader.read_vector()
    hessian = reader.read_symmetric_matrix()
    result = _response.FitGlmMAPResponse(
        weights = weights,
        hessian = hessian,
    )
    return result

def read_fit_bayesian_glm_response(reader):
    weight_mean_vector = reader.read_vector()
    weight_covariance_matrix = reader.read_symmetric_matrix()
    noise_variance_mean = reader.read_double()
    result = _response.FitBayesianGlmResponse(
            weight_mean_vector = weight_mean_vector,
            weight_covariance_matrix = weight_covariance_matrix,
            noise_variance_mean = noise_variance_mean,
    )
    return result

def read_fit_sparse_glm_response(reader):
    hyperparameters = reader.read_vector()
    aloocv = reader.read_double()
    aloocvs = reader.read_vector()
    weight_matrix = reader.read_sparse_matrix()
    intercepts = reader.read_vector()
    raw_weights = RawWeights.read(reader)
    result = _response.FitSparseGlmResponse(
        hyperparameters = hyperparameters, 
        aloocv = aloocv,
        aloocvs = aloocvs,
        weights = weight_matrix.toarray(),
        intercepts = intercepts,
        raw_weights = raw_weights,
    )
    return result

def read_fit_gp_regression_map_response(reader):
    length = reader.read_double()
    noise_ratio = reader.read_double()
    prediction_b_value = reader.read_double()
    hessian = reader.read_symmetric_matrix()
    intermediate1_prediction_vector = reader.read_vector()
    intermediate2_prediction_matrix = reader.read_matrix()
    beta_vector = reader.read_vector()
    packed_gl_matrix = reader.read_vector()
    return _response.FitGpRegressionMapResponse(
            length = length,
            noise_ratio = noise_ratio,
            prediction_b_value = prediction_b_value,
            hessian = hessian,
            intermediate1_prediction_vector = intermediate1_prediction_vector,
            intermediate2_prediction_matrix = intermediate2_prediction_matrix,
            beta_vector = beta_vector,
            packed_gl_matrix = packed_gl_matrix,
    )

def read_predict_gp_regression_map_response(reader):
    log_pdf_normalizer = reader.read_double()
    prediction_mean_vector = reader.read_vector()
    prediction_packed_r22l_matrix = reader.read_vector()
    return _response.PredictGpRegressionMapResponse(
            log_pdf_normalizer = log_pdf_normalizer,
            prediction_mean_vector = prediction_mean_vector,
            prediction_packed_r22l_matrix = prediction_packed_r22l_matrix,
    )

def read_fit_bayesian_gp_regression_response(reader):
    log_length = reader.read_double()
    log_noise_ratio = reader.read_double()
    weight_vector = reader.read_vector()
    s2_vector = reader.read_vector()
    axi_diagonals = reader.read_matrix()
    beta_hat_matrix = reader.read_matrix()
    predictor = reader.read_blob()
    hyperparameter_matrix = reader.read_matrix()
    marginal_point_vector = reader.read_vector()
    marginal_integral_point_vector = reader.read_vector()
    marginal_log_length = read_marginal(reader)
    marginal_log_noise_ratio = read_marginal(reader)
    return _response.FitBayesianGpRegressionResponse(
            log_length = log_length,
            log_noise_ratio = log_noise_ratio,
            weight_vector = weight_vector,
            s2_vector = s2_vector,
            axi_diagonals = axi_diagonals,
            beta_hat_matrix = beta_hat_matrix,
            predictor = predictor,
            hyperparameter_matrix = hyperparameter_matrix,
            marginal_point_vector = marginal_point_vector,
            marginal_integral_point_vector = marginal_integral_point_vector,
            marginal_log_length = marginal_log_length,
            marginal_log_noise_ratio = marginal_log_noise_ratio,
    )

def read_predict_bayesian_gp_regression_response(reader):
    prediction_mean_vector = reader.read_vector()
    pdf_matrix = reader.read_matrix()
    return _response.PredictBayesianGpRegressionResponse(
            prediction_mean_vector = prediction_mean_vector,
            pdf_matrix = pdf_matrix,
    )

def read_bayesian_gp_pred_pdf_response(reader):
    res_vector = reader.read_vector()
    return _response.BayesianGpPredPdfResponse(
            res_vector = res_vector,
    )

def read_response(sock):
    buf = _socket_utility.read(sock, _header.header_size)
    header = _header.Header.read(Reader(buf))
    assert header.version == _protocol.protocol_version
    buf = _socket_utility.read(sock, header.size)
    reader = Reader(buf)
    tp = reader.read_uint8()
    if tp == 0:
        result = read_error_response(reader)
    elif tp == 1:
        result = read_fit_glm_response(reader)
    elif tp == 2:
        result = read_fit_sparse_glm_response(reader)
    elif tp == 3:
        result = read_fit_bayesian_glm_response(reader)
    elif tp == 4:
        result = read_fit_glm_map_response(reader)
    elif tp == 5:
        result = read_fit_gp_regression_map_response(reader)
    elif tp == 6:
        result = read_predict_gp_regression_map_response(reader)
    elif tp == 7:
        result = read_fit_bayesian_gp_regression_response(reader)
    elif tp == 8:
        result = read_predict_bayesian_gp_regression_response(reader)
    elif tp == 9:
        result = read_bayesian_gp_pred_pdf_response(reader)
    else:
        assert False, "unknown response type: %d" % tp
    return result
