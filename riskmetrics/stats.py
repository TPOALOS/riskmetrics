"""Performance and risk statistics from a list of simple per-period returns. Pure standard library."""
import math
import statistics


class StatsError(ValueError):
    pass


def mean(x):
    return sum(x) / len(x)


def stdev(x):
    """Sample standard deviation (n - 1)."""
    if len(x) < 2:
        raise StatsError("need at least 2 observations")
    return statistics.stdev(x)


def skewness(x):
    """Population skewness (third standardised moment)."""
    m, n = mean(x), len(x)
    s2 = sum((v - m) ** 2 for v in x) / n
    return 0.0 if s2 == 0 else sum((v - m) ** 3 for v in x) / n / s2 ** 1.5


def kurtosis(x):
    """Population *raw* kurtosis (a normal distribution has 3)."""
    m, n = mean(x), len(x)
    s2 = sum((v - m) ** 2 for v in x) / n
    return 3.0 if s2 == 0 else sum((v - m) ** 4 for v in x) / n / s2 ** 2


def quantile(sorted_x, q):
    """Linear-interpolation quantile (the usual 'type 7') of an already sorted list."""
    if not sorted_x:
        raise StatsError("empty data")
    pos = (len(sorted_x) - 1) * q
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    return sorted_x[lo] + (sorted_x[hi] - sorted_x[lo]) * (pos - lo)


def equity_curve(r):
    """Wealth path starting at 1.0 (length n + 1)."""
    w, out = 1.0, [1.0]
    for v in r:
        w *= 1.0 + v
        out.append(w)
    return out


def total_return(r):
    return equity_curve(r)[-1] - 1.0


def cagr(r, periods_per_year):
    wealth = equity_curve(r)[-1]
    if wealth <= 0:
        return -1.0
    return wealth ** (periods_per_year / len(r)) - 1.0


def sharpe_per_period(r, rf_per_period=0.0):
    ex = [v - rf_per_period for v in r]
    sd = stdev(ex)
    return None if sd == 0 else mean(ex) / sd


def sharpe(r, periods_per_year, rf_annual=0.0):
    """Annualised Sharpe ratio; None when the returns have no variation."""
    sp = sharpe_per_period(r, rf_annual / periods_per_year)
    return None if sp is None else sp * math.sqrt(periods_per_year)


def sortino(r, periods_per_year, mar_annual=0.0):
    mar = mar_annual / periods_per_year
    dd = math.sqrt(sum(min(v - mar, 0.0) ** 2 for v in r) / len(r))
    return None if dd == 0 else (mean(r) - mar) / dd * math.sqrt(periods_per_year)


def drawdown_episodes(r):
    """List of dicts {peak, trough, recovery, depth}; positions index the equity curve (0 = start). Sorted worst first."""
    eq = equity_curve(r)
    episodes, peak_pos, peak = [], 0, eq[0]
    trough_pos, trough = 0, eq[0]
    in_dd = False
    for pos in range(1, len(eq)):
        if eq[pos] >= peak:
            if in_dd:
                episodes.append({"peak": peak_pos, "trough": trough_pos, "recovery": pos, "depth": trough / peak - 1.0})
                in_dd = False
            peak_pos, peak = pos, eq[pos]
            trough_pos, trough = pos, eq[pos]
        else:
            in_dd = True
            if eq[pos] < trough:
                trough_pos, trough = pos, eq[pos]
    if in_dd:
        episodes.append({"peak": peak_pos, "trough": trough_pos, "recovery": None, "depth": trough / peak - 1.0})
    return sorted(episodes, key=lambda e: e["depth"])


def max_drawdown(r):
    eps = drawdown_episodes(r)
    return eps[0] if eps else {"peak": 0, "trough": 0, "recovery": 0, "depth": 0.0}


def var_cvar(r, alpha=0.95):
    """Historical VaR and expected shortfall as positive loss fractions of one period."""
    s = sorted(r)
    cut = quantile(s, 1.0 - alpha)
    tail = [v for v in s if v <= cut]
    return max(0.0, -cut), max(0.0, -mean(tail))


def metrics(r, periods_per_year, rf_annual=0.0):
    if len(r) < 2:
        raise StatsError("need at least 2 observations")
    dd = max_drawdown(r)
    var95, cvar95 = var_cvar(r, 0.95)
    g = cagr(r, periods_per_year)
    return {
        "observations": len(r), "years": len(r) / periods_per_year, "total_return": total_return(r), "cagr": g,
        "volatility": stdev(r) * math.sqrt(periods_per_year), "sharpe": sharpe(r, periods_per_year, rf_annual),
        "sortino": sortino(r, periods_per_year, rf_annual), "max_drawdown": dd["depth"],
        "calmar": None if dd["depth"] == 0 else g / abs(dd["depth"]),
        "drawdown_periods": (dd["trough"] - dd["peak"]), "recovered": dd["recovery"] is not None,
        "var95": var95, "cvar95": cvar95, "skewness": skewness(r), "kurtosis": kurtosis(r),
        "best_period": max(r), "worst_period": min(r), "hit_rate": sum(1 for v in r if v > 0) / len(r),
        "profit_factor": None if not any(v < 0 for v in r) else sum(v for v in r if v > 0) / -sum(v for v in r if v < 0),
    }
