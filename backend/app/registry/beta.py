"""Beta-distribution maths without a SciPy dependency.

Only what routing needs: the CDF (regularised incomplete beta) and its inverse.
"""

import math

_TINY = 1e-300
_EPS = 3e-14
_MAX_ITER = 300


def _continued_fraction(x: float, a: float, b: float) -> float:
    """Lentz's continued fraction for the incomplete beta function."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > _TINY else _TINY)
    h = d

    for m in range(1, _MAX_ITER + 1):
        m2 = 2 * m

        step = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + step * d
        d = 1.0 / (d if abs(d) > _TINY else _TINY)
        c = 1.0 + step / c
        c = c if abs(c) > _TINY else _TINY
        h *= d * c

        step = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + step * d
        d = 1.0 / (d if abs(d) > _TINY else _TINY)
        c = 1.0 + step / c
        c = c if abs(c) > _TINY else _TINY
        delta = d * c
        h *= delta

        if abs(delta - 1.0) < _EPS:
            break

    return h


def beta_cdf(x: float, a: float, b: float) -> float:
    """P(X <= x) for X ~ Beta(a, b)."""
    if a <= 0 or b <= 0:
        raise ValueError("a and b must be positive")
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0

    log_front = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    front = math.exp(log_front)

    if x < (a + 1.0) / (a + b + 2.0):
        return front * _continued_fraction(x, a, b) / a
    return 1.0 - front * _continued_fraction(1.0 - x, b, a) / b


def beta_ppf(q: float, a: float, b: float) -> float:
    """The q-th quantile of Beta(a, b), found by bisection on the CDF."""
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be between 0 and 1")

    low, high = 0.0, 1.0
    for _ in range(100):
        mid = (low + high) / 2.0
        if beta_cdf(mid, a, b) < q:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0