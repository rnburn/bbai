from ._writer import Writer
from ._raw_weights import RawWeights

import numpy as np

def encode_covariance_function(covariance_function):
    if covariance_function == "rbf":
        return 0
    elif covariance_function == 'power1':
        return 1
    assert False, "unknown covariance_function %s" % loss_link

def make_fit_gp_regression_map_request(
        covariance_function,
        sample_matrix, y, design_matrix,
        hyperparameter0_vector,
        use_log_parameters):
    writer = Writer()
    writer.write_uint8(3) # request_type

    writer.write_uint8(encode_covariance_function(covariance_function.name_))
    writer.write_vector(np.array(covariance_function.params_, dtype=np.float64))

    writer.write_matrix(sample_matrix)
    writer.write_matrix(design_matrix)
    writer.write_vector(y)
    writer.write_vector(hyperparameter0_vector)
    writer.write_uint8(use_log_parameters)

    writer.prepend_header()
    return writer.tobytes()

def make_fit_bayesian_gp_regression_request(
        tolerance,
        covariance_function,
        sample_matrix, y, design_matrix,
        hyperparameter0_vector):
    writer = Writer()
    writer.write_uint8(5) # request_type

    writer.write_double(tolerance)
    writer.write_uint8(encode_covariance_function(covariance_function.name_))
    writer.write_vector(np.array(covariance_function.params_, dtype=np.float64))

    writer.write_matrix(sample_matrix)
    writer.write_matrix(design_matrix)
    writer.write_vector(y)
    writer.write_vector(hyperparameter0_vector)

    writer.prepend_header()
    return writer.tobytes()

def make_predict_gp_regression_map_request(
        prediction_b_value,
        covariance_function,
        train_sample_matrix,
        sample_matrix, design_matrix,
        hyperparameter_vector,
        intermediate1_prediction_vector,
        intermediate2_prediction_matrix,
        beta_vector,
        packed_gl_matrix,
        with_pdf):
    writer = Writer()
    writer.write_uint8(4) # request_type
    writer.write_double(prediction_b_value)

    writer.write_uint8(encode_covariance_function(covariance_function.name_))
    writer.write_vector(np.array(covariance_function.params_, dtype=np.float64))

    writer.write_vector(hyperparameter_vector)
    writer.write_matrix(train_sample_matrix)

    writer.write_matrix(sample_matrix)
    writer.write_matrix(design_matrix)

    writer.write_vector(intermediate1_prediction_vector)
    writer.write_matrix(intermediate2_prediction_matrix)
    writer.write_vector(beta_vector)
    writer.write_vector(packed_gl_matrix)

    if with_pdf:
        writer.write_uint8(1)
    else:
        writer.write_uint8(0)

    writer.prepend_header()
    return writer.tobytes()

def make_predict_bayesian_gp_regression_request(
        covariance_function,
        train_sample_matrix,
        sample_matrix, design_matrix,
        predictor,
        with_pdf):
    writer = Writer()
    writer.write_uint8(6) # request_type

    writer.write_uint8(encode_covariance_function(covariance_function.name_))
    writer.write_vector(np.array(covariance_function.params_, dtype=np.float64))

    writer.write_matrix(train_sample_matrix)
    writer.write_matrix(sample_matrix)
    writer.write_matrix(design_matrix)

    writer.write_blob(predictor)

    if with_pdf:
        writer.write_uint8(1)
    else:
        writer.write_uint8(0)

    writer.prepend_header()
    return writer.tobytes()

def make_bayesian_gp_pred_pdf_request(op, df, pdf_matrix, z):
    writer = Writer()
    writer.write_uint8(7) # request_type
    if op == 'pdf':
        writer.write_uint8(0)
    elif op == 'cdf':
        writer.write_uint8(1)
    elif op == 'ppf':
        writer.write_uint8(2)
    else:
        assert False, "unknown op"
    writer.write_uint64(df)
    writer.write_matrix(pdf_matrix)
    writer.write_double(z)

    writer.prepend_header()
    return writer.tobytes()
