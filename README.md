# riskmetrics

[![tests](https://github.com/TPOALOS/riskmetrics/actions/workflows/tests.yml/badge.svg)](https://github.com/TPOALOS/riskmetrics/actions/workflows/tests.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![python](https://img.shields.io/badge/python-3.9%2B-blue)
![dependencies](https://img.shields.io/badge/dependencies-none-brightgreen)

**Performance statistics that admit how little they know.** Give it a CSV of returns (or prices) and it reports the usual numbers
(CAGR, volatility, Sharpe, Sortino, drawdowns, VaR/CVaR), puts **confidence intervals** around them, and, most importantly, corrects the
Sharpe ratio for **how many variants you tried** before picking the best one.

Best-of-100 backtests are almost always lucky. `riskmetrics deflate` shows how lucky:

```bash
python3 -m riskmetrics deflate --sharpe 1.5 --n-obs 750 --trials 100
```

```
Sharpe 1.50 (annualised) from 750 observations, best of 100 trials

  P(true Sharpe > 0)            99.5%
  Sharpe expected from luck     1.47   (best of 100 unskilled trials, spread assumed)
  Deflated Sharpe Ratio (DSR)   52.2%   (>95% = unlikely to be luck)
  Bonferroni-adjusted p-value   0.4936
  Haircut Sharpe                0.01   (1% of the original)
  Track record needed           1.21 years for 95% confidence that Sharpe > 0
```

A Sharpe of 1.5 over three years looks excellent on its own (99.5% sure it is positive). After admitting it was the best of 100 tries, it is
about what luck alone would produce.

Pure Python standard library: no pandas, no numpy, no install step.

## Commands

```bash
python3 -m riskmetrics summary returns.csv                      # all numeric columns
python3 -m riskmetrics summary prices.csv --prices --col close --freq daily
python3 -m riskmetrics summary r.csv --bootstrap 2000 --seed 1  # adds bootstrap intervals
python3 -m riskmetrics drawdowns returns.csv --col strategy --top 5
python3 -m riskmetrics compare variants.csv --trials 40         # rank columns, correct for multiple testing
python3 -m riskmetrics deflate --sharpe 1.5 --n-obs 750 --trials 100 --trial-sharpes my_trial_sharpes.txt
```

All commands accept `--format json`, `--freq daily|daily365|weekly|monthly|quarterly|yearly` or an explicit `--periods-per-year N`
(for example `2190` for 4-hour bars on a market that never closes), and `--rf` for an annual risk-free rate. **Nothing is assumed about your market:**
you choose the annualisation.

### `summary`

```
                           steady
Observations               756
Years                      3.00
CAGR                       11.5%
Volatility (ann.)          11.0%
Sharpe (ann.)              1.04
  95% interval (analytic)  -0.09 .. 2.18
  P(true Sharpe > 0)       96%
Sortino (ann.)             1.56
Max drawdown               -6.9%
VaR 95% (1 period)         1.10%
CVaR 95% (1 period)        1.42%
Bootstrap Sharpe           -0.01 .. 2.24
Bootstrap max drawdown     -21.6% .. -6.7%
```

(Synthetic data from [`examples/make_sample.py`](examples/make_sample.py).) Notice the interval: three years of data cannot tell a Sharpe of 1.0 from 0.
The bootstrap interval for **max drawdown** is a useful reality check too, because the drawdown you saw is only one draw from what could have happened.

### `compare`

Several columns = several variants. `compare` ranks them and applies the corrections:

```
Trials counted: 20

series                    obs    Sharpe    p (raw)   p (Holm)   PSR(0)      DSR  survives?
steady                    756      1.04     0.0357     0.7143      96%      76%  no
volatile                  756      0.83     0.0754     1.0000      92%      63%  no
coin_flip                 756      0.39     0.2513     1.0000      75%      33%  no
```

Pass `--trials` with the **total number of variants you tried**, including the ones you threw away: only counting the survivors is exactly the
selection bias this is meant to fix. "Survives" means Holm-adjusted p < 0.05 and DSR > 95%.

### `deflate`

Use it when you only have numbers: the best Sharpe, the number of observations and the number of trials. If you kept the Sharpe of every trial, pass them with
`--trial-sharpes FILE` (one per line) so the spread of your trials is measured instead of assumed.

## What is inside

| Statistic | Notes |
|-----------|-------|
| Sharpe, Sortino, Calmar, CAGR, volatility | per-period returns, sample standard deviation; Sortino uses downside deviation around the risk-free rate |
| Max drawdown and the worst episodes | peak, trough and recovery; verified against a brute-force reference in the tests |
| Historical VaR / CVaR | linear-interpolated quantile, reported as a positive loss per period |
| Analytic Sharpe interval, PSR | adjusts for skewness and kurtosis (Lo 2002; Mertens 2002; Bailey and Lopez de Prado 2012) |
| Minimum track record length | how many observations before the Sharpe is distinguishable from a benchmark |
| Deflated Sharpe Ratio | corrects for selection among many trials and for non-normality (Bailey and Lopez de Prado 2014) |
| Holm and Bonferroni | multiple-testing adjusted p-values (Holm 1979), plus a Bonferroni "haircut" Sharpe |
| Stationary bootstrap | resamples blocks of random length to keep some serial dependence (Politis and Romano 1994) |

The citations are from memory; check them against the papers if you rely on them.

## Honest limitations

- **Assumptions matter.** The Sharpe-ratio inference assumes returns are roughly independent over time. Autocorrelated returns (smoothed or illiquid assets, overlapping positions) make
  every estimate look better than it is. The bootstrap keeps some dependence but is not a cure.
- **DSR needs a believable trial count and spread.** If you do not know how many variants you tried, the answer is only as good as your guess. When the spread of the trials is not given,
  `deflate` assumes the null spread `1/sqrt(n-1)`, which is stated in the output.
- **Statistics describe the sample.** They say nothing about regime changes, costs you did not model, survivorship in your data, or look-ahead bias. (A companion linter, `backtestlint`, hunts for the last one in code.)
- Few observations give wide intervals. That is the honest answer, not a bug.
- This is research tooling. **Nothing here is investment advice, a forecast or a recommendation**, and every output says so.

## Input format

A CSV with an optional date column and one or more numeric columns. Percent signs (`1.5%`), empty cells and `nan` are understood; text columns are skipped with a warning;
`--prices` converts prices to returns. A return of -100% or worse is rejected (it usually means the column holds prices).

## Tests

```bash
python3 -m unittest -v        # 83 tests
```

The tests compare against independent references where they exist (the `statistics` module, a brute-force drawdown search, Monte-Carlo simulation of the expected maximum Sharpe,
the Holm worked example) and check the properties the methods must have (DSR falls with more trials and rises with more data; Holm never beats Bonferroni; the minimum track record
length reproduces its own confidence level).

## License

MIT (c) TPOALOS
