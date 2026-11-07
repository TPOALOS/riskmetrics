# Changelog

## 1.0.0

First release.

- `summary`: CAGR, volatility, Sharpe with analytic and stationary-bootstrap intervals, Sortino, Calmar, max drawdown, VaR/CVaR, skewness and kurtosis, hit rate, profit factor.
- `drawdowns`: the worst drawdown episodes with peak, trough and recovery dates.
- `compare`: ranks several return series and corrects their Sharpe ratios for the number of variants tried (Holm-adjusted p-values, Probabilistic and Deflated Sharpe Ratio).
- `deflate`: Deflated Sharpe Ratio, Bonferroni haircut and minimum track record length from numbers you already have.
- CSV input with returns or prices, percent signs, missing values, date columns and frequency presets.
