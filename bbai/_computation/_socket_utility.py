def read(sock, n):
    buf = memoryview(bytearray(n))
    count = 0
    while count < n:
        count += sock.recv_into(buf[count:])
    return buf
