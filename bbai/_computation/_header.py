header_size = 2 + 8

header_format = "".join([
    "=",
    "H",  # version
    "Q",  # size
])

class Header(object):
    def __init__(self, version, size):
        self.version = version
        self.size = size

    @staticmethod
    def read(reader):
        version = reader.read_uint16()
        size = reader.read_uint64()
        return Header(version, size)
