import numpy as np

class RawWeights(object):
    def __init__(self, weights=[], active_indexes=None):
        self.weights = np.array(weights, dtype=np.float64)
        self.is_sparse = active_indexes is not None
        self.active_indexes = active_indexes

    @staticmethod
    def read(reader):
        tp = reader.read_uint8()
        if tp == 0:
            weights = reader.read_vector()
            return RawWeights(weights)
        elif tp == 1:
            weights = reader.read_vector()
            active_indexes = reader.read_int64_vector()
            return RawWeights(weights, active_indexes)
        assert False, "unexpected type"

    def write(self, writer):
        if not self.is_sparse:
            writer.write_uint8(0)
            writer.write_vector(self.weights)
        else:
            writer.write_uint8(1)
            writer.write_vector(self.weights)
            writer.write_vector(np.array(self.active_indexes, dtype=np.int64))
