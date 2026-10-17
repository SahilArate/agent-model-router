import math

import pytest

from app.registry.beta import beta_cdf, beta_ppf


def test_uniform_quantile_is_identity():
    for q in (0.05, 0.5, 0.95):
        assert beta_ppf(q, 1, 1) == pytest.approx(q, abs=1e-9)


def test_closed_form_cases():
    # Beta(2,1) has CDF x^2, Beta(1,2) has CDF 1-(1-x)^2, Beta(n,1) has CDF x^n.
    assert beta_ppf(0.25, 2, 1) == pytest.approx(math.sqrt(0.25), abs=1e-9)
    assert beta_ppf(0.25, 1, 2) == pytest.approx(1 - math.sqrt(0.75), abs=1e-9)
    assert beta_ppf(0.05, 31, 1) == pytest.approx(0.05 ** (1 / 31), abs=1e-9)


def test_symmetric_median():
    assert beta_ppf(0.5, 5, 5) == pytest.approx(0.5, abs=1e-9)


def test_known_values_from_scipy():
    assert beta_ppf(0.05, 9, 3) == pytest.approx(0.52991, abs=1e-4)
    assert beta_ppf(0.05, 181, 21) == pytest.approx(0.85870, abs=1e-4)


def test_cdf_and_ppf_are_inverses():
    for a, b in ((0.5, 0.5), (3, 7), (40, 4)):
        x = beta_ppf(0.3, a, b)
        assert beta_cdf(x, a, b) == pytest.approx(0.3, abs=1e-9)


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        beta_cdf(0.5, 0, 1)
    with pytest.raises(ValueError):
        beta_ppf(1.5, 1, 1)