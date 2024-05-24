import pytest
import numpy as np
from bbai.stat import NormalMeanHypothesis

def test_normal_mean_hypothesis():
    hy = NormalMeanHypothesis()
    data = np.array([1, 2, 3])

    t1 = hy.test(data)
    assert t1.left + t1.equal + t1.right == 1.0
    assert t1.left < t1.right
    assert t1.equal < t1.right

    t2 = hy.test(-data)
    assert t2.left == pytest.approx(t1.right)
    assert t2.equal == pytest.approx(t1.equal)

    t3 = hy.test(list(data) + [4])
    assert t3.right > t1.right

    t4 = NormalMeanHypothesis(mu0=1.23).test(data + 1.23)
    assert t4.left == pytest.approx(t1.left)
    assert t4.equal == pytest.approx(t1.equal)
    assert t4.right == pytest.approx(t1.right)

def test_normal_mean_two_tailed_hypothesis():
    h1 = NormalMeanHypothesis()
    h1p = NormalMeanHypothesis(with_equal=False)
    data = np.array([1, 2, 3])

    t1 = h1.test(data)
    t1p = h1p.test(data)
    assert t1.factor_left == t1p.factor_left
    assert t1.factor_right == t1p.factor_right
    assert t1p.left == pytest.approx(1 - t1p.right)
    assert t1p.left == pytest.approx(t1.factor_left / (t1.factor_left + t1.factor_right))

if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
