"""Text and JSON rendering."""
import json
import math

from . import DISCLAIMER, __version__


def pct(v, digits=1):
    return "n/a" if v is None or v != v else "%.*f%%" % (digits, v * 100)


def num(v, digits=2):
    return "n/a" if v is None or v != v else ("inf" if v == float("inf") else "%.*f" % (digits, v))


def label_at(labels, pos):
    """Label of an equity-curve position (0 = before the first return)."""
    if pos is None:
        return "not recovered"
    if pos == 0 or not labels:
        return "start" if pos == 0 else "#%d" % pos
    return labels[pos - 1]


def render_summary(results, warnings):
    names = list(results)
    out = ["riskmetrics %s" % __version__]
    out += ["warning: " + w for w in warnings]
    rows = [("Observations", lambda m: "%d" % m["observations"]), ("Years", lambda m: num(m["years"])),
            ("Total return", lambda m: pct(m["total_return"])), ("CAGR", lambda m: pct(m["cagr"])), ("Volatility (ann.)", lambda m: pct(m["volatility"])),
            ("Sharpe (ann.)", lambda m: num(m["sharpe"])), ("  95% interval (analytic)", lambda m: "%s .. %s" % (num(m["sharpe_ci"][0]), num(m["sharpe_ci"][1])) if m["sharpe_ci"] else "n/a"),
            ("  P(true Sharpe > 0)", lambda m: pct(m["psr0"], 0)), ("Sortino (ann.)", lambda m: num(m["sortino"])),
            ("Max drawdown", lambda m: pct(m["max_drawdown"])), ("Calmar", lambda m: num(m["calmar"])),
            ("VaR 95% (1 period)", lambda m: pct(m["var95"], 2)), ("CVaR 95% (1 period)", lambda m: pct(m["cvar95"], 2)),
            ("Skewness", lambda m: num(m["skewness"])), ("Kurtosis (normal = 3)", lambda m: num(m["kurtosis"])),
            ("Best / worst period", lambda m: "%s / %s" % (pct(m["best_period"], 2), pct(m["worst_period"], 2))),
            ("Hit rate", lambda m: pct(m["hit_rate"], 0)), ("Profit factor", lambda m: num(m["profit_factor"]))]
    boot = any("bootstrap" in m for m in results.values())
    if boot:
        rows += [("Bootstrap Sharpe", lambda m: "%s .. %s" % tuple(num(x) for x in m["bootstrap"]["sharpe"]) if "bootstrap" in m else "n/a"),
                 ("Bootstrap CAGR", lambda m: "%s .. %s" % tuple(pct(x) for x in m["bootstrap"]["cagr"]) if "bootstrap" in m else "n/a"),
                 ("Bootstrap max drawdown", lambda m: "%s .. %s" % tuple(pct(x) for x in m["bootstrap"]["max_drawdown"]) if "bootstrap" in m else "n/a")]
    width = max(len(r[0]) for r in rows) + 2
    colw = max(18, max(len(n) for n in names) + 2)
    out.append("")
    out.append(" " * width + "".join(n.ljust(colw) for n in names))
    for label, fn in rows:
        out.append(label.ljust(width) + "".join(fn(results[n]).ljust(colw) for n in names))
    first = results[names[0]]
    if boot and "bootstrap" in first:
        b = first["bootstrap"]
        out.append("")
        out.append("Bootstrap: %d resamples, mean block %.1f, %d%% percentile intervals%s" % (
            b["resamples"], b["mean_block"], round(b["level"] * 100), ", seed %s" % b["seed"] if b["seed"] is not None else ""))
    out += ["", DISCLAIMER]
    return "\n".join(out)


def render_drawdowns(name, episodes, labels, top):
    out = ["Worst drawdowns for %s" % name, "%-3s %9s  %-20s %-20s %-20s %s" % ("#", "depth", "peak", "trough", "recovery", "periods down")]
    for i, e in enumerate(episodes[:top], 1):
        end = e["recovery"] if e["recovery"] is not None else None
        out.append("%-3d %9s  %-20s %-20s %-20s %s" % (i, pct(e["depth"]), label_at(labels, e["peak"]), label_at(labels, e["trough"]),
                                                      label_at(labels, e["recovery"]), (end - e["peak"]) if end is not None else "ongoing"))
    if not episodes:
        out.append("  no drawdowns")
    return "\n".join(out + ["", DISCLAIMER])


def to_json(obj):
    def clean(v):
        if isinstance(v, float) and (v != v or math.isinf(v)):
            return None
        if isinstance(v, dict):
            return {k: clean(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [clean(x) for x in v]
        return v
    return json.dumps(clean(obj), indent=2)
