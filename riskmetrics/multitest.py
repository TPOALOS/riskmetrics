"""Sharpe-ratio inference and multiple-testing corrections.

Follows Lo (2002), Mertens (2002) and Bailey & Lopez de Prado (2012, 2014). All Sharpe ratios in this
module are PER PERIOD unless a function says otherwise; use `annualise` / `deannualise` to convert.
The formulas assume returns are roughly independent through time; autocorrelated returns make the
estimates look better than they are.
"""
import math
from statistics import NormalDist

N = NormalDist()
EULER_GAMMA = 0.5772156649015329


def annualise(sr, periods_per_year):
    return sr * math.sqrt(periods_per_year)


def deannualise(sr, periods_per_year):
    return sr / math.sqrt(periods_per_year)


def sharpe_variance_factor(sr, skew, kurt):
    """1 - skew*SR + (kurt - 1)/4 * SR^2 : the non-normality-adjusted variance term (kurt is raw, normal = 3)."""
    return 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr


def sharpe_std_error(sr, n, skew=0.0, kurt=3.0):
    """Standard error of a per-period Sharpe estimate from n observations."""
    f = sharpe_variance_factor(sr, skew, kurt)
    return math.sqrt(f / (n - 1)) if f > 0 and n > 1 else float("nan")


def psr(sr, n, benchmark=0.0, skew=0.0, kurt=3.0):
    """Probabilistic Sharpe Ratio: chance that the true Sharpe exceeds `benchmark` (per period)."""
    f = sharpe_variance_factor(sr, skew, kurt)
    if f <= 0 or n < 2:
        return float("nan")
    return N.cdf((sr - benchmark) * math.sqrt(n - 1) / math.sqrt(f))


def expected_max_sharpe(trials, sharpe_std):
    """Expected maximum Sharpe among `trials` unskilled strategies whose Sharpe estimates have std `sharpe_std`."""
    if trials < 1:
        raise ValueError("trials must be at least 1")
    if trials == 1:
        return 0.0
    g = EULER_GAMMA
    return sharpe_std * ((1 - g) * N.inv_cdf(1 - 1.0 / trials) + g * N.inv_cdf(1 - 1.0 / (trials * math.e)))


def dsr(sr, n, trials, sharpe_std=None, skew=0.0, kurt=3.0):
    """Deflated Sharpe Ratio: PSR against the best Sharpe you would expect from luck after `trials` attempts.

    sharpe_std defaults to the null standard error 1/sqrt(n - 1) when the spread of the trials is unknown.
    """
    if sharpe_std is None:
        sharpe_std = 1.0 / math.sqrt(n - 1)
    return psr(sr, n, expected_max_sharpe(trials, sharpe_std), skew, kurt)


def min_track_record(sr, benchmark=0.0, confidence=0.95, skew=0.0, kurt=3.0):
    """Observations needed for the Sharpe estimate to beat `benchmark` with the given confidence (inf if it never can)."""
    if sr <= benchmark:
        return float("inf")
    f = sharpe_variance_factor(sr, skew, kurt)
    return 1.0 + f * (N.inv_cdf(confidence) / (sr - benchmark)) ** 2


def p_value(sr, n, skew=0.0, kurt=3.0):
    """One-sided p-value for H0: true Sharpe <= 0."""
    p = psr(sr, n, 0.0, skew, kurt)
    return float("nan") if p != p else 1.0 - p


def holm(pvalues):
    """Holm-Bonferroni adjusted p-values, returned in the original order."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adj, running = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adj[i] = running
    return adj


def bonferroni(pvalues):
    m = len(pvalues)
    return [min(1.0, m * p) for p in pvalues]


def haircut_sharpe(sr, n, trials):
    """Sharpe after a Bonferroni haircut for `trials` attempts (per period). Returns (haircut_sr, adjusted_p)."""
    p = p_value(sr, n)
    adj = min(1.0, trials * p)
    if adj >= 1.0:
        return 0.0, adj
    return N.inv_cdf(1 - adj) / math.sqrt(n - 1), adj
