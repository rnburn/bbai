from . import _header
from . import _socket_utility
from . import _protocol
from . import _response
from ._raw_weights import RawWeights
from ._reader import Reader
import struct
import numpy as np

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
    else:
        assert False, "unknown response type"
    return result
