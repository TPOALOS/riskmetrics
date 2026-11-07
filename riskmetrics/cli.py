"""Command line interface."""
import argparse
import math
import statistics
import sys

from . import DISCLAIMER, __version__, bootstrap, data, multitest as mt, report, stats


def add_series_args(p):
    p.add_argument("file", help="CSV file with one or more return (or price) columns")
    p.add_argument("--col", action="append", default=[], metavar="NAME", help="column name or 1-based index (repeatable; default: all numeric columns)")
    p.add_argument("--prices", action="store_true", help="the columns are prices, not returns")
    p.add_argument("--date-column", help="name of the date/label column (auto-detected otherwise)")
    add_freq_args(p)


def add_freq_args(p):
    p.add_argument("--freq", choices=sorted(data.FREQUENCIES), help="observation frequency preset (daily = 252 periods/year)")
    p.add_argument("--periods-per-year", type=float, help="periods per year (overrides --freq); e.g. 2190 for 4-hour bars on a 365-day market")
    p.add_argument("--rf", type=float, default=0.0, help="annual risk-free rate as a fraction (default 0)")


def build_parser():
    ap = argparse.ArgumentParser(prog="riskmetrics", description="Performance statistics with honest uncertainty and multiple-testing corrections.")
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("summary", help="full statistics for each column")
    add_series_args(p)
    p.add_argument("--bootstrap", type=int, default=0, metavar="N", help="add stationary-bootstrap intervals with N resamples")
    p.add_argument("--seed", type=int, help="random seed for the bootstrap")
    p.add_argument("--format", choices=("text", "json"), default="text")
    p = sub.add_parser("drawdowns", help="worst drawdown episodes of one column")
    add_series_args(p)
    p.add_argument("--top", type=int, default=5)
    p = sub.add_parser("compare", help="rank columns and correct their Sharpe ratios for having tried several")
    add_series_args(p)
    p.add_argument("--trials", type=int, help="total number of variants you tried (default: number of columns)")
    p.add_argument("--format", choices=("text", "json"), default="text")
    p = sub.add_parser("deflate", help="Deflated Sharpe Ratio from numbers you already have")
    p.add_argument("--sharpe", type=float, required=True, help="annualised Sharpe ratio of the best variant")
    p.add_argument("--n-obs", type=int, required=True, help="number of return observations behind it")
    p.add_argument("--trials", type=int, required=True, help="how many variants you tried")
    add_freq_args(p)
    p.add_argument("--trial-sharpe-std", type=float, help="standard deviation of the annualised Sharpe ratios across your trials")
    p.add_argument("--trial-sharpes", metavar="FILE", help="file with the annualised Sharpe of every trial (one number per line); estimates the spread")
    p.add_argument("--skew", type=float, default=0.0)
    p.add_argument("--kurt", type=float, default=3.0, help="raw kurtosis (normal = 3)")
    p.add_argument("--format", choices=("text", "json"), default="text")
    return ap


def periods(args):
    if args.periods_per_year:
        if args.periods_per_year <= 0:
            raise ValueError("--periods-per-year must be positive")
        return args.periods_per_year
    return data.FREQUENCIES[args.freq or "daily"]


def series_metrics(r, ppy, rf):
    m = stats.metrics(r, ppy, rf)
    sr_p = stats.sharpe_per_period(r, rf / ppy)
    if sr_p is None:
        m["sharpe_ci"], m["psr0"] = None, None
    else:
        se = mt.sharpe_std_error(sr_p, len(r), m["skewness"], m["kurtosis"])
        m["sharpe_ci"] = (mt.annualise(sr_p - 1.96 * se, ppy), mt.annualise(sr_p + 1.96 * se, ppy)) if se == se else None
        m["psr0"] = mt.psr(sr_p, len(r), 0.0, m["skewness"], m["kurtosis"])
    return m


def load(args):
    ds = data.load_csv(args.file, args.col, args.prices, args.date_column)
    return ds


def cmd_summary(args, out):
    ds, ppy = load(args), periods(args)
    results = {}
    for name, r in ds.series.items():
        results[name] = series_metrics(r, ppy, args.rf)
        if args.bootstrap:
            results[name]["bootstrap"] = bootstrap.confidence_intervals(r, ppy, args.rf, args.bootstrap, seed=args.seed)
    if args.format == "json":
        out.write(report.to_json({"tool": "riskmetrics " + __version__, "periods_per_year": ppy, "risk_free": args.rf, "warnings": ds.warnings,
                                  "series": results, "disclaimer": DISCLAIMER}) + "\n")
    else:
        out.write(report.render_summary(results, ds.warnings) + "\n")


def cmd_drawdowns(args, out):
    ds = load(args)
    for name, r in ds.series.items():
        out.write(report.render_drawdowns(name, stats.drawdown_episodes(r), ds.labels[name], args.top) + "\n\n")


def cmd_compare(args, out):
    ds, ppy = load(args), periods(args)
    names = list(ds.series)
    if len(names) < 2:
        raise ValueError("compare needs at least 2 columns (use --col to choose them)")
    trials = args.trials or len(names)
    if trials < len(names):
        raise ValueError("--trials cannot be smaller than the number of columns")
    rows = []
    for n in names:
        r = ds.series[n]
        m = stats.metrics(r, ppy, args.rf)
        sp = stats.sharpe_per_period(r, args.rf / ppy)
        rows.append({"name": n, "n": len(r), "sp": sp, "sharpe": m["sharpe"], "skew": m["skewness"], "kurt": m["kurtosis"]})
    pvals = [mt.p_value(r["sp"], r["n"], r["skew"], r["kurt"]) if r["sp"] is not None else 1.0 for r in rows]
    holm_p = mt.holm(pvals + [1.0] * (trials - len(pvals)))[:len(pvals)]    # untested trials are counted as p = 1 (worst case)
    sps = [r["sp"] for r in rows if r["sp"] is not None]
    spread = statistics.stdev(sps) if len(sps) >= 2 else None
    for r, p, h in zip(rows, pvals, holm_p):
        r.update(p=p, holm=h, psr0=mt.psr(r["sp"], r["n"], 0.0, r["skew"], r["kurt"]) if r["sp"] is not None else None,
                 dsr=mt.dsr(r["sp"], r["n"], trials, spread, r["skew"], r["kurt"]) if r["sp"] is not None else None)
        r["survives"] = bool(r["dsr"] is not None and r["dsr"] > 0.95 and h < 0.05)
    rows.sort(key=lambda r: -(r["sharpe"] if r["sharpe"] is not None else -1e9))
    if args.format == "json":
        out.write(report.to_json({"tool": "riskmetrics " + __version__, "periods_per_year": ppy, "trials": trials, "sharpe_spread_per_period": spread,
                                  "ranking": [{k: r[k] for k in ("name", "n", "sharpe", "p", "holm", "psr0", "dsr", "survives")} for r in rows],
                                  "disclaimer": DISCLAIMER}) + "\n")
        return
    out.write("Trials counted: %d%s\n\n" % (trials, " (columns)" if not args.trials else ""))
    out.write("%-22s %6s %9s %10s %10s %8s %8s  %s\n" % ("series", "obs", "Sharpe", "p (raw)", "p (Holm)", "PSR(0)", "DSR", "survives?"))
    for r in rows:
        out.write("%-22s %6d %9s %10s %10s %8s %8s  %s\n" % (r["name"][:22], r["n"], report.num(r["sharpe"]), "%.4f" % r["p"], "%.4f" % r["holm"],
                                                           report.pct(r["psr0"], 0), report.pct(r["dsr"], 0), "yes" if r["survives"] else "no"))
    out.write("\n'survives' = Holm-adjusted p < 0.05 and DSR > 95%%. Both account for %d trials; DSR uses the spread of the Sharpe ratios across the columns.\n" % trials)
    out.write("If you tried more variants than these columns, pass --trials; every variant counts, including the ones you discarded.\n\n" + DISCLAIMER + "\n")


def cmd_deflate(args, out):
    ppy = periods(args)
    if args.n_obs < 3 or args.trials < 1:
        raise ValueError("need --n-obs >= 3 and --trials >= 1")
    sp = mt.deannualise(args.sharpe, ppy)
    std = None
    if args.trial_sharpes:
        with open(args.trial_sharpes, encoding="utf-8") as fh:
            vals = [float(t.strip().split(",")[-1]) for t in fh if t.strip() and t.strip()[0] in "-+.0123456789"]
        if len(vals) < 2:
            raise ValueError("--trial-sharpes needs at least 2 numbers")
        std = statistics.stdev([mt.deannualise(v, ppy) for v in vals])
        if args.trials < len(vals):
            raise ValueError("--trials (%d) is smaller than the %d Sharpe ratios in the file" % (args.trials, len(vals)))
    elif args.trial_sharpe_std is not None:
        std = mt.deannualise(args.trial_sharpe_std, ppy)
    used_std = std if std is not None else 1.0 / math.sqrt(args.n_obs - 1)
    sr0 = mt.expected_max_sharpe(args.trials, used_std)
    hair, adj_p = mt.haircut_sharpe(sp, args.n_obs, args.trials)
    res = {"sharpe_annual": args.sharpe, "observations": args.n_obs, "trials": args.trials, "periods_per_year": ppy,
           "trial_sharpe_std_assumed": std is None, "luck_benchmark_sharpe_annual": mt.annualise(sr0, ppy),
           "psr_vs_zero": mt.psr(sp, args.n_obs, 0.0, args.skew, args.kurt), "dsr": mt.dsr(sp, args.n_obs, args.trials, std, args.skew, args.kurt),
           "bonferroni_p": adj_p, "haircut_sharpe_annual": mt.annualise(hair, ppy),
           "min_track_record_years": mt.min_track_record(sp, 0.0, 0.95, args.skew, args.kurt) / ppy}
    if args.format == "json":
        res["disclaimer"] = DISCLAIMER
        out.write(report.to_json(res) + "\n")
        return
    out.write("Sharpe %.2f (annualised) from %d observations, best of %d trials\n\n" % (args.sharpe, args.n_obs, args.trials))
    out.write("  P(true Sharpe > 0)            %s\n" % report.pct(res["psr_vs_zero"], 1))
    out.write("  Sharpe expected from luck     %s   (best of %d unskilled trials%s)\n" % (
        report.num(res["luck_benchmark_sharpe_annual"]), args.trials, ", spread assumed" if std is None else ""))
    out.write("  Deflated Sharpe Ratio (DSR)   %s   (>95%% = unlikely to be luck)\n" % report.pct(res["dsr"], 1))
    out.write("  Bonferroni-adjusted p-value   %.4f\n" % adj_p)
    out.write("  Haircut Sharpe                %s   (%s of the original)\n" % (report.num(res["haircut_sharpe_annual"]), report.pct(res["haircut_sharpe_annual"] / args.sharpe if args.sharpe else None, 0)))
    out.write("  Track record needed           %s years for 95%% confidence that Sharpe > 0\n\n" % report.num(res["min_track_record_years"]))
    if std is None:
        out.write("Note: the spread of your trials' Sharpe ratios was not given, so the null spread 1/sqrt(n-1) was assumed. "
                  "Use --trial-sharpes FILE or --trial-sharpe-std for a better estimate.\n\n")
    out.write(DISCLAIMER + "\n")


def main(argv=None, out=None):
    out = out or sys.stdout
    try:
        args = build_parser().parse_args(argv)
        {"summary": cmd_summary, "drawdowns": cmd_drawdowns, "compare": cmd_compare, "deflate": cmd_deflate}[args.cmd](args, out)
        return 0
    except (data.DataError, stats.StatsError, ValueError, OSError) as exc:
        sys.stderr.write("riskmetrics: error: %s\n" % exc)
        return 2
