from ._writer import Writer
from ._raw_weights import RawWeights

import numpy as np

def encode_loss_link(loss_link):
    if loss_link == "l2":
        return 0
    if loss_link == "multinomial_logistic":
        return 1
    if loss_link == "multinomial_logistic_m1":
        return 2
    assert False, "unknown loss link %s" % loss_link

def encode_regularizer(regularizer):
    if regularizer == "l2":
        return 0
    if regularizer  == 'l1':
        return 1
    if regularizer == 'elasticnet':
        return 2
    assert False, "unknown regularizer %s" % regularizer

def make_fit_glm_request(
        loss_link, regularizer,
        normalize, fit_intercept,
        X, y, hyperparameters, weights0=None):
    writer = Writer()
    writer.write_uint8(0) # request_type
    writer.write_uint8(encode_loss_link(loss_link))
    writer.write_uint8(encode_regularizer(regularizer))
    writer.write_uint8(normalize)
    writer.write_uint8(fit_intercept)
    writer.write_matrix(X)
    writer.write_vector(y)
    writer.write_vector(hyperparameters)

    # weights0
    if weights0 is None:
        weights0 = RawWeights([])
    weights0.write(writer)

    writer.prepend_header()
    return writer.tobytes()

def make_fit_glm_map_request(X, y):
    writer = Writer()
    writer.write_uint8(2) # request_type
    writer.write_matrix(X)
    writer.write_vector(y)

    writer.prepend_header()
    return writer.tobytes()

def make_fit_bayesian_glm_request(
        X, y):
    writer = Writer()
    writer.write_uint8(1) # request_type
    writer.write_matrix(X)
    writer.write_vector(y)

    writer.prepend_header()
    return writer.tobytes()
