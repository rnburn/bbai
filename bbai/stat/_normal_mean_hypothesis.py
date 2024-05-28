import numpy as np
import scipy
from collections import namedtuple

from bbai._computation._bridge import stat

class _NormalMeanHypothesisResult3:
    def __init__(self, z, n, sigma=None):
        t = np.sqrt(n - 1) * z

        dist = scipy.stats.t(n - 1)

        pdf_t = dist.pdf(t)

        # self.left_to_star_N corresponds to the value B_{20}^N in [1]
        self.left_to_star_N = dist.cdf(-t)

        # self.right_to_star_N corresponds to the value B_{30}^N in [1]
        self.right_to_star_N = 1 - self.left_to_star_N

        # self.equal_to_star_N corresponds to the value B_{10}^N in [1]
        if sigma is not None:
          self.equal_to_star_N = np.sqrt(n-1) * pdf_t / sigma
        else:
          self.equal_to_star_N = None

        
        root2_z = np.sqrt(2) * z

        g1 = stat.normal_eeibf_g1(root2_z)

        # self.correction_left corresponds to the value
        #   E_{\hat{\theta}}^{H_0}[B_{20}^N(X(l))]
        # in [1] sec 2.4.2
        self.correction_left = 0.5 - g1 / np.pi

        # self.correction_right corresponds to the value
        #   E_{\hat{\theta}}^{H_0}[B_{30}^N(X(l))]
        # in [1] sec 2.4.2
        self.correction_right = 0.5 + g1 / np.pi

        # self.factor_left corresponds to the value
        #   B_{20}^{EEI}
        # in [1] sec 2.4.2
        self.factor_left = self.left_to_star_N / self.correction_left

        # self.factor_right corresponds to the value
        #   B_{30}^{EEI}
        # in [1] sec 2.4.2
        self.factor_right = self.right_to_star_N / self.correction_right

        g2 = stat.normal_eeibf_g2(root2_z)

        # self.correction_equal corresponds to the value
        #   E_{\hat{\theta}}^{H_0}[B_{10}^N(X(l))]
        # in [1] sec 2.4.2
        correction_equal = np.sqrt(2) * g2 / np.pi
        if sigma is not None:
          self.correction_equal = correction_equal / sigma
        else:
          self.correction_equal = None

        # self.factor_equal corresponds to the value
        #   B_{10}^{EEI}
        # in [1] sec 2.4.2
        self.factor_equal = pdf_t * np.sqrt(n - 1) / correction_equal

        # self.left, self.equal, self.right are the posterior probabilities
        # of the three hypotheses
        total = self.factor_left + self.factor_equal + self.factor_right
        self.left = self.factor_left / total
        self.equal = self.factor_equal / total
        self.right = self.factor_right / total

class _NormalMeanHypothesisResult2:
    def __init__(self, z, n):
        res3 = _NormalMeanHypothesisResult3(z, n)

        self.left_to_star_N = res3.left_to_star_N
        self.right_to_star_N = res3.right_to_star_N

        
        self.correction_left = res3.correction_left
        self.correction_right = res3.correction_right

        self.factor_left = res3.factor_left
        self.factor_right = res3.factor_right


        total = self.factor_left + self.factor_right
        self.left = self.factor_left / total
        self.right = self.factor_right / total

class NormalMeanHypothesis:
    """Implements normal mean hypothesis testing with unknown variance using the
    encompassing expected intrinsic Bayes factor (EEIBF) method described in the paper

     1: Default Bayes Factors for Nonnested Hypothesis Testing 
           by J. O. Berger and  J. Mortera
           url: https://www.jstor.org/stable/2670175
           postscript: http://www2.stat.duke.edu/~berger/papers/mortera.ps

    The class provides both two tailed hypothesis testing (i.e. H_1: mu < 0, H_2: mu >= 0) and
    hypothesis testing with equality (H_1: mu == 0, H_2: mu < 0, H_3: mu > 0).

    The implementation uses an accurate and efficient deterministic algorithm to compute the
    correction factors 

         E_{\hat{\theta}}^{H_0}[B_{i0}^N (X(l))]
    
    (see [1] section 2.4.2)

    Parameters
    ----------
    mu0 : double, default=0
        The split point for the hypotheses

    with_equal: bool, default=True
        Whether to include an equals hypothesis

    Examples
    --------
    >>> from bbai.stat import NormalMeanHypothesis
    >>> y = np.random.normal(size=9)
    >>> res = NormalMeanHypothesis().test(y)
    >>> print(res.left) # posterior probability for the hypothesis mu < 0
    >>> print(res.equal) # posterior probability for the hypothesis mu == 0
    >>> print(res.right) # posterior probability for the hypothesis mu > 0
    """
    def __init__(self, mu0 = 0, with_equal=True):
        self.mu0_ = mu0
        self.with_equal_ = with_equal

    def test(self, data):
        n = len(data)
        mean = np.mean(data)
        std = np.std(data)
        z = (mean - self.mu0_) / std
        if self.with_equal_:
          return _NormalMeanHypothesisResult3(z, n, std)
        else:
          return _NormalMeanHypothesisResult2(z, n)

    def test_t(self, t, n):
        z = t / np.sqrt(n - 1)
        if self.with_equal_:
          return _NormalMeanHypothesisResult3(z, n)
        else:
          return _NormalMeanHypothesisResult2(z, n)
