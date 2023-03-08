import struct
import scipy
import scipy.sparse
import numpy as np

class Reader(object):
    def __init__(self, buf):
        self.buf_ = buf

    def read_uint8(self):
        tp = struct.unpack("B", self.buf_[:1])
        self.buf_ = self.buf_[1:]
        return tp[0]

    def read_uint16(self):
        tp = struct.unpack("H", self.buf_[:2])
        self.buf_ = self.buf_[2:]
        return tp[0]

    def read_uint64(self):
        tp = struct.unpack("Q", bytes(self.buf_[:8]))
        self.buf_ = self.buf_[8:]
        return tp[0]

    def read_int64s(self, n):
        size = n*8
        result = np.frombuffer(self.buf_[:size], dtype=np.int64)
        self.buf_ = self.buf_[size:]
        return result

    def read_double(self):
        x = struct.unpack("d", bytes(self.buf_[:8]))
        self.buf_ = self.buf_[8:]
        return x[0]

    def read_blob(self):
        n = self.read_uint64()
        result = self.buf_[:n]
        self.buf_ = self.buf_[n:]
        return result

    def read_doubles(self, n):
        size = n*8
        result = np.frombuffer(self.buf_[:size], dtype=np.float64)
        self.buf_ = self.buf_[size:]
        return result

    def read_string(self):
        n = self.read_uint64()
        s = self.buf_[:n].tobytes().decode('utf-8')
        self.buf_ = self.buf_[n:]
        return s

    def read_int64_vector(self):
        n = self.read_uint64()
        return self.read_int64s(n)

    def read_vector(self):
        n = self.read_uint64()
        return self.read_doubles(n)

    def read_matrix(self):
        m = self.read_uint64()
        n = self.read_uint64()
        result = self.read_doubles(m*n)
        return result.reshape((m, n), order='F')

    def read_symmetric_matrix(self):
        res = self.read_matrix()
        n = res.shape[0]
        for i in range(n):
            for j in range(i+1, n):
                res[i, j] = res[j, i]
        return res

    def read_sparse_matrix(self):
        m = self.read_uint64()
        n = self.read_uint64()
        num_nonzero = self.read_uint64()

        values = self.read_doubles(num_nonzero)
        column_indexes = self.read_int64s(num_nonzero)
        row_start_indexes = self.read_int64s(m + 1)
        
        return scipy.sparse.csr_matrix(
            (values, column_indexes, row_start_indexes), 
            shape=(m, n)
        )
