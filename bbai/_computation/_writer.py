import struct
import numpy as np
from ._header import header_size
from ._protocol import protocol_version

class Writer(object):
    def __init__(self):
        self.format_ = '='
        self.args_ = []

    def tobytes(self):
        return struct.pack(self.format_, *self.args_)

    def write_uint8(self, c):
        self.format_ += 'B'
        self.args_.append(int(c))

    def write_uint16(self, s):
        self.format_ += 'H'
        self.args_.append(int(s))

    def write_uint64(self, n):
        self.format_ += 'Q'
        self.args_.append(int(n))

    def prepend_header(self):
        request_size = struct.calcsize(self.format_)
        self.format_ = '=HQ' + self.format_[1:]
        self.args_ = [protocol_version, request_size] + self.args_

    def write_bytes(self, array):
        self.format_ += '%ds' % (array.dtype.itemsize * array.size)
        self.args_.append(array.tobytes(order='F'))


    def write_vector(self, v):
        self.write_uint64(v.size)
        self.write_bytes(v)

    def write_matrix(self, m):
        assert len(m.shape) == 2
        self.write_uint64(m.shape[0])
        self.write_uint64(m.shape[1])
        self.write_bytes(m)
