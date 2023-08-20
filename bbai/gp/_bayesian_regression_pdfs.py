from .._computation._bridge import gp

import numpy as np
from scipy.optimize import root_scalar
from scipy.special import hyp2f1, loggamma
import scipy.stats

class PdfOperation:
    pdf = 0
    cdf = 1
    ppf = 2

class BayesianRegressionPDFs(object):
    def __init__(self, pdf_matrix, df, is_scalar=False):
        self.pdf_matrix_ = pdf_matrix
        self.df_ = df
        self._is_scalar = is_scalar

    def __call__(self, y):
        return self.pdf(y)

    def pdf(self, y):
        res = gp.pdf_operation(dict(
                op = PdfOperation.pdf,
                degrees_of_freedom = self.df_,
                pdf_matrix = self.pdf_matrix_,
                z = y,
        ))
        if self._is_scalar:
            return res[0]
        return res

    def cdf(self, y):
        res = gp.pdf_operation(dict(
                op = PdfOperation.cdf,
                degrees_of_freedom = self.df_,
                pdf_matrix = self.pdf_matrix_,
                z = y,
        ))
        if self._is_scalar:
            return res[0]
        return res

    def ppf(self, p):
        res = gp.pdf_operation(dict(
                op = PdfOperation.ppf,
                degrees_of_freedom = self.df_,
                pdf_matrix = self.pdf_matrix_,
                z = p,
        ))
        if self._is_scalar:
            return res[0]
        return res

    def __getitem__(self, key):
        if isinstance(key, slice):
            pdf_matrix_p = self.pdf_matrix_[:, key]
            return BayesianRegressionPDFs(pdf_matrix_p, self.df_)
        pdf_matrix_p = self.pdf_matrix_[:, key:key+1]
        return BayesianRegressionPDFs(pdf_matrix_p, self.df_, True)
