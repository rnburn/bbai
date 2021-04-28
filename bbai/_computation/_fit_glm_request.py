import struct

from ._header import header_size
from ._serialization_utility import \
        to_array_bytes, \
        make_vector_format, \
        make_matrix_format

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
    assert False, "unknown regularizer %s" % regularizer


def make_format(X, y, hyperparameters):
    return "".join([
        "=",
        # header
        "H",  # version
        "Q",  # request size
        # request
        "B",  # request type
        "B",  # loss link
        "B",  # regularizer
        "B",  # normalize
        "B",  # fit_intercept
        make_matrix_format(X),  # feature_matrix
        make_vector_format(y),  # target_vector
        make_vector_format(hyperparameters),  # hyperparameters
    ])

def make_fit_glm_request(
        loss_link,
        regularizer,
        normalize,
        fit_intercept,
        X, y, hyperparameters):
    f = make_format(X, y, hyperparameters)
    request_size = struct.calcsize(f) - header_size
    num_data, num_features = X.shape
    num_hyperparameters = len(hyperparameters)
    pack_args = [
        # header
        1,  # version
        request_size,  # request_size
        # request
        0,  # request type
        encode_loss_link(loss_link),  # loss_link
        encode_regularizer(regularizer),  # regularizer
        int(normalize),  # normalize
        int(fit_intercept),  # fit_intercept
        # feature_matrix
        num_data,
        num_features,
        to_array_bytes(X),
        # target_vector
        num_data,
        to_array_bytes(y),
        # hyperparameter_vector
        num_hyperparameters,
        to_array_bytes(hyperparameters),
    ]
    return struct.pack(f, *pack_args)
