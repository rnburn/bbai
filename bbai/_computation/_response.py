from . import _header
from . import _socket_utility
from . import _protocol
import struct
import numpy as np

def read_vector(buf):
    n = struct.unpack("Q", bytes(buf[:8]))
    n = n[0]
    buf = buf[8:]
    size = n * 8
    v = np.frombuffer(buf[:size], dtype=np.float64)
    return v, buf[size:]

def read_matrix(buf):
    m, n = struct.unpack("QQ", bytes(buf[:16]))
    buf = buf[16:]
    size = m*n*8
    v = np.frombuffer(buf[:size], dtype=np.float64)
    mat = v.reshape((m, n), order='F')
    return mat, buf[size:]

def read_response_type(buf):
    tp = struct.unpack("B", buf[:1])
    return tp[0], buf[1:]
    
def read_fit_glm_response(buf):
    hyperparameters, buf = read_vector(buf)
    weight_matrix, buf = read_matrix(buf)
    biases, buf = read_vector(buf)
    result = hyperparameters, weight_matrix, biases
    return result, buf

def read_response(sock):
    buf = _socket_utility.read(sock, _header.header_size)
    version, size = struct.unpack(_header.header_format, bytes(buf))
    assert version == _protocol.protocol_version
    buf = _socket_utility.read(sock, size)
    tp, buf = read_response_type(buf)
    assert tp == 1
    result, buf = read_fit_glm_response(buf)
    assert len(buf) == 0
    return result
