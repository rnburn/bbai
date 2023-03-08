import numpy as np

def validate_problem(Z, X, y):
    if len(Z.shape) != 2:
        raise RuntimeError("Z must be a matrix")
    if len(X.shape) != 2:
        raise RuntimeError("X must be a matrix")
    if len(y.shape) != 1:
        raise RuntimeError("y must be a vector")
    if Z.shape[0] != len(y):
        raise RuntimeError("dimensions of Z and y do not match")
    if not np.isfinite(Z).all() or not np.isfinite(X).all() or not np.isfinite(y).all():
        raise RuntimeError("all values must be finite")
    if X.shape[1] == 0:
        return
    if np.linalg.matrix_rank(X) != X.shape[1]:
        raise RuntimeError("X must be of full rank")
