from .._computation._computation_handle import get_computation_handle

import numpy as np
from scipy.optimize import root_scalar
from scipy.special import hyp2f1, loggamma
import scipy.stats

class BayesianRegressionPDFs(object):
    def __init__(self, pdf_matrix, df, is_scalar=False):
        self.pdf_matrix_ = pdf_matrix
        self.df_ = df
        self._handle = get_computation_handle()
        self._is_scalar = is_scalar

    def __call__(self, y):
        return self.pdf(y)

    def pdf(self, y):
        res = self._handle.bayesian_gp_pred_pdf(
                op = 'pdf',
                df = self.df_,
                pdf_matrix = self.pdf_matrix_,
                z = y,
        )
        if self._is_scalar:
            return res.res_vector[0]
        return res.res_vector

    def cdf(self, y):
        res = self._handle.bayesian_gp_pred_pdf(
                op = 'cdf',
                df = self.df_,
                pdf_matrix = self.pdf_matrix_,
                z = y,
        )
        if self._is_scalar:
            return res.res_vector[0]
        return res.res_vector

    def ppf(self, p):
        res = self._handle.bayesian_gp_pred_pdf(
                op = 'ppf',
                df = self.df_,
                pdf_matrix = self.pdf_matrix_,
                z = p,
        )
        if self._is_scalar:
            return res.res_vector[0]
        return res.res_vector

    def __getitem__(self, key):
        if isinstance(key, slice):
            pdf_matrix_p = self.pdf_matrix_[:, key]
            return BayesianRegressionPDFs(pdf_matrix_p, self.df_)
        pdf_matrix_p = self.pdf_matrix_[:, key:key+1]
        return BayesianRegressionPDFs(pdf_matrix_p, self.df_, True)
