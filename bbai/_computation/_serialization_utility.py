import numpy as np

def to_array_bytes(array):
    if array.dtype != np.float64:
        array = np.array(array, dtype=np.float64)
    return array.tobytes(order='F')

def make_vector_format(x):
    n = x.shape[0]
    return "Q%ds" % (n * 8)

def make_matrix_format(X):
    m, n = X.shape
    return "QQ%ds" % (m * n * 8)
