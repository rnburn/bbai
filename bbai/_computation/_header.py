header_size = 2 + 8

header_format = "".join([
    "=",
    "H",  # version
    "Q",  # size
])
