from ._glm_request import make_fit_glm_request, \
        make_fit_glm_map_request, \
        make_fit_bayesian_glm_request

from ._gp_request import make_fit_gp_regression_map_request, \
        make_fit_bayesian_gp_regression_request, \
        make_predict_gp_regression_map_request, \
        make_predict_bayesian_gp_regression_request, \
        make_bayesian_gp_pred_pdf_request

from ._response_serialization import read_response
from . import _response

import tempfile
import subprocess
import socket
import time
import pathlib

def _get_socket_address():
    name = next(tempfile._get_candidate_names())
    return tempfile._get_default_tempdir() + "/" + name

def _establish_sidecar_connection(address):
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    start_time = time.time()
    timeout = 5
    elapse = 0
    duration = 0.01
    while True:
        time.sleep(duration)
        duration *= 2
        try:
            sock.connect(address)
            return sock
        except socket.error:
            elapse = time.time() - start_time
            if elapse > timeout:
                raise

def _spawn_sidecar():
    srcdir = pathlib.Path(__file__).parent.absolute()
    exe = str(srcdir / "../bbai")
    socket_address = _get_socket_address()
    command = [
            exe,
            "--unix_domain_socket",
            socket_address,
            "--poll_parent",
    ]
    subprocess.Popen(command)
    return _establish_sidecar_connection(socket_address)

class ComputationHandle(object):
    def __init__(self):
        self._sock = _spawn_sidecar()

    def fit_glm(self, **kwargs):
        request = make_fit_glm_request(**kwargs)
        self._sock.sendall(request)
        response = read_response(self._sock)
        if type(response) == _response.ErrorResponse:
            raise RuntimeError(response.message)
        return response

    def fit_glm_map(self, **kwargs):
        request = make_fit_glm_map_request(**kwargs)
        self._sock.sendall(request)
        response = read_response(self._sock)
        if type(response) == _response.ErrorResponse:
            raise RuntimeError(response.message)
        return response

    def fit_bayesian_glm(self, **kwargs):
        request = make_fit_bayesian_glm_request(**kwargs)
        self._sock.sendall(request)
        response = read_response(self._sock)
        if type(response) == _response.ErrorResponse:
            raise RuntimeError(response.message)
        return response

    def fit_gp_regression_map(self, **kwargs):
        request = make_fit_gp_regression_map_request(**kwargs)
        self._sock.sendall(request)
        response = read_response(self._sock)
        if type(response) == _response.ErrorResponse:
            raise RuntimeError(response.message)
        return response

    def predict_gp_regression_map(self, **kwargs):
        request = make_predict_gp_regression_map_request(**kwargs)
        self._sock.sendall(request)
        response = read_response(self._sock)
        if type(response) == _response.ErrorResponse:
            raise RuntimeError(response.message)
        return response

    def fit_bayesian_gp_regression(self, **kwargs):
        request = make_fit_bayesian_gp_regression_request(**kwargs)
        self._sock.sendall(request)
        response = read_response(self._sock)
        if type(response) == _response.ErrorResponse:
            raise RuntimeError(response.message)
        return response

    def predict_bayesian_gp_regression(self, **kwargs):
        request = make_predict_bayesian_gp_regression_request(**kwargs)
        self._sock.sendall(request)
        response = read_response(self._sock)
        if type(response) == _response.ErrorResponse:
            raise RuntimeError(response.message)
        return response

    def bayesian_gp_pred_pdf(self, **kwargs):
        request = make_bayesian_gp_pred_pdf_request(**kwargs)
        self._sock.sendall(request)
        response = read_response(self._sock)
        if type(response) == _response.ErrorResponse:
            raise RuntimeError(response.message)
        return response

_handle = None

def get_computation_handle():
    global _handle
    if _handle is None:
        _handle = ComputationHandle()
    return _handle
