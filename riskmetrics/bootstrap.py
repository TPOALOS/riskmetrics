"""Stationary bootstrap (Politis & Romano, 1994) confidence intervals for performance statistics."""
import math
import random

from . import stats


def stationary_resample(r, rng, mean_block):
    """One resample of the same length, built from random-length blocks that wrap around the sample."""
    n = len(r)
    p = 1.0 / mean_block
    out, i = [], rng.randrange(n)
    for _ in range(n):
        out.append(r[i])
        i = rng.randrange(n) if rng.random() < p else (i + 1) % n
    return out


def default_block(n):
    return max(1.0, round(n ** (1.0 / 3.0), 1))


def confidence_intervals(r, periods_per_year, rf_annual=0.0, resamples=2000, level=0.95, seed=None, mean_block=None):
    """Percentile intervals for Sharpe, CAGR and max drawdown. Reproducible when `seed` is given."""
    if len(r) < 10:
        raise stats.StatsError("need at least 10 observations for a bootstrap")
    if resamples < 100:
        raise stats.StatsError("use at least 100 resamples")
    rng = random.Random(seed)
    block = mean_block or default_block(len(r))
    sharpes, cagrs, dds = [], [], []
    for _ in range(resamples):
        s = stationary_resample(r, rng, block)
        sh = stats.sharpe(s, periods_per_year, rf_annual)
        if sh is not None:
            sharpes.append(sh)
        cagrs.append(stats.cagr(s, periods_per_year))
        dds.append(stats.max_drawdown(s)["depth"])
    lo_q, hi_q = (1 - level) / 2, 1 - (1 - level) / 2
    ci = lambda xs: (stats.quantile(sorted(xs), lo_q), stats.quantile(sorted(xs), hi_q)) if xs else (None, None)
    return {"level": level, "resamples": resamples, "mean_block": block, "seed": seed,
            "sharpe": ci(sharpes), "cagr": ci(cagrs), "max_drawdown": ci(dds)}
