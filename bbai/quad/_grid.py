from collections import namedtuple
import numpy as np
import scipy.special

EvaluationPoint = namedtuple(
        'EvaluationPoint',
        [
            'x',
            'weight',
            'multiplier',
            'val',
        ],
)

EvaluationGrid = namedtuple(
        'EvaluationGrid',
        [
            'k',
            'eigenvalues',
            'eigenvectors',
            'x0',
            'x1_points',
            'x2_points',
            'points',
        ],
)

def make_evaluation_grid2d(f, x0, hessian, k):
    eigenvalues, eigenvectors = np.linalg.eig(hessian)
    res = []
    roots, weights = scipy.special.roots_hermite(k)
    t1 = np.sqrt(eigenvalues[0] / 2)
    t2 = np.sqrt(eigenvalues[1] / 2)
    points = []
    for i in range(k):
        xi = roots[i] / t1
        wi = weights[i] / t1
        for j in range(k):
            yj = roots[j] / t2
            wj = weights[j] / t2
            m = np.exp(-roots[i]**2 - roots[j]**2)
            pt = x0 + xi * eigenvectors[:, 0] + yj * eigenvectors[:, 1]
            val = f(pt)
            point = EvaluationPoint(
                    x = np.array([xi, yj]),
                    weight = wi * wj,
                    multiplier = m,
                    val = val,
            )
            points.append(point)
    return EvaluationGrid(
            k = k,
            eigenvalues = eigenvalues,
            eigenvectors = eigenvectors,
            x0 = x0,
            x1_points = roots / t1,
            x2_points = roots / t2,
            points = points,
    )

def make_evaluation_grid2dX(f, x0, hessian, k):
    eigenvalues, eigenvectors = np.linalg.eig(hessian)
    res = []
    roots = np.polynomial.chebyshev.chebpts1(k)
    t1 = np.sqrt(eigenvalues[0] / 2)
    t2 = np.sqrt(eigenvalues[1] / 2)
    points = []
    for i in range(k):
        xi = roots[i] / t1
        for j in range(k):
            yj = roots[j] / t2
            m = np.exp(-roots[i]**2 - roots[j]**2)
            pt = x0 + xi * eigenvectors[:, 0] + yj * eigenvectors[:, 1]
            val = f(pt)
            point = EvaluationPoint(
                    x = np.array([xi, yj]),
                    weight = None,
                    multiplier = m,
                    val = val,
            )
            points.append(point)
    return EvaluationGrid(
            k = k,
            eigenvalues = eigenvalues,
            eigenvectors = eigenvectors,
            x0 = x0,
            x1_points = roots / t1,
            x2_points = roots / t2,
            points = points,
    )
