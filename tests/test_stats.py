import math
import random
import statistics
import unittest

from riskmetrics import stats

SMALL = [0.01, -0.02, 0.03, 0.0, -0.01, 0.02, 0.01, -0.005, 0.015, 0.0]


def brute_max_drawdown(r):
    """Independent reference: worst peak-to-trough fall over all (i, j) pairs of the wealth path."""
    w, path = 1.0, [1.0]
    for v in r:
        w *= 1 + v
        path.append(w)
    worst = 0.0
    for i in range(len(path)):
        for j in range(i, len(path)):
            worst = min(worst, path[j] / path[i] - 1)
    return worst


class BasicTests(unittest.TestCase):
    def test_mean_and_stdev_match_the_statistics_module(self):
        self.assertAlmostEqual(stats.mean(SMALL), statistics.mean(SMALL))
        self.assertAlmostEqual(stats.stdev(SMALL), statistics.stdev(SMALL))
        with self.assertRaises(stats.StatsError):
            stats.stdev([0.1])

    def test_moments(self):
        sym = [-1, 1, -1, 1, -1, 1]
        self.assertAlmostEqual(stats.skewness(sym), 0.0)
        self.assertAlmostEqual(stats.kurtosis(sym), 1.0)                 # two-point distribution: raw kurtosis is exactly 1
        self.assertGreater(stats.skewness([0, 0, 0, 0, 10]), 1)
        self.assertLess(stats.skewness([0, 0, 0, 0, -10]), -1)
        self.assertEqual(stats.skewness([2, 2, 2]), 0.0)
        self.assertEqual(stats.kurtosis([2, 2, 2]), 3.0)
        rng = random.Random(3)
        big = [rng.gauss(0, 1) for _ in range(200000)]
        self.assertAlmostEqual(stats.kurtosis(big), 3.0, delta=0.08)
        self.assertAlmostEqual(stats.skewness(big), 0.0, delta=0.03)

    def test_quantile(self):
        xs = sorted(range(1, 11))
        self.assertEqual(stats.quantile(xs, 0.5), 5.5)
        self.assertEqual(stats.quantile(xs, 0.0), 1)
        self.assertEqual(stats.quantile(xs, 1.0), 10)
        self.assertAlmostEqual(stats.quantile(xs, 0.25), 3.25)
        with self.assertRaises(stats.StatsError):
            stats.quantile([], 0.5)

    def test_equity_and_returns(self):
        self.assertEqual(stats.equity_curve([0.1, 0.1]), [1.0, 1.1, 1.1 * 1.1])
        self.assertAlmostEqual(stats.total_return([0.1, 0.1]), 0.21)
        self.assertAlmostEqual(stats.cagr([0.1, 0.1], 2), 0.21)                 # two periods = one year
        self.assertAlmostEqual(stats.cagr([0.1, 0.1], 1), 1.21 ** 0.5 - 1)
        self.assertEqual(stats.cagr([-1.0 + 1e-12, -0.5], 1) > -1.0, True)

    def test_total_loss_cagr(self):
        self.assertEqual(stats.cagr([0.1, -1.0], 2), -1.0)


class RatioTests(unittest.TestCase):
    def test_sharpe_definition(self):
        sr = stats.sharpe(SMALL, 252)
        self.assertAlmostEqual(sr, statistics.mean(SMALL) / statistics.stdev(SMALL) * math.sqrt(252))

    def test_sharpe_with_risk_free_rate(self):
        rf = 0.05
        ex = [v - rf / 252 for v in SMALL]
        self.assertAlmostEqual(stats.sharpe(SMALL, 252, rf), statistics.mean(ex) / statistics.stdev(ex) * math.sqrt(252))

    def test_sharpe_scale_invariance_and_annualisation(self):
        double = [2 * v for v in SMALL]
        self.assertAlmostEqual(stats.sharpe(double, 252), stats.sharpe(SMALL, 252))
        self.assertAlmostEqual(stats.sharpe(SMALL, 252) / stats.sharpe(SMALL, 63), 2.0)       # sqrt(252/63)

    def test_no_variation_has_no_sharpe(self):
        self.assertIsNone(stats.sharpe([0.01] * 10, 252))
        self.assertIsNone(stats.sortino([0.01] * 10, 252))

    def test_sortino_only_penalises_losses(self):
        gains_only_more_volatile = [0.0, 0.1, 0.0, 0.2, -0.01, 0.0]
        dd = math.sqrt(sum(min(v, 0) ** 2 for v in gains_only_more_volatile) / 6)
        expected = statistics.mean(gains_only_more_volatile) / dd * math.sqrt(12)
        self.assertAlmostEqual(stats.sortino(gains_only_more_volatile, 12), expected)

    def test_var_and_cvar(self):
        r = [-0.05, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
        var, cvar = stats.var_cvar(r, 0.9)
        self.assertAlmostEqual(var, 0.032)                            # q(0.1) = -0.05 + 0.9*(0.02) = -0.032
        self.assertAlmostEqual(cvar, 0.05)                            # mean of returns at or below -0.032
        self.assertGreaterEqual(cvar, var)
        self.assertEqual(stats.var_cvar([0.01, 0.02, 0.03], 0.95), (0.0, 0.0))     # no losses: no VaR


class DrawdownTests(unittest.TestCase):
    def test_matches_brute_force_on_random_paths(self):
        rng = random.Random(5)
        for _ in range(60):
            r = [rng.gauss(0.0005, 0.02) for _ in range(rng.randint(2, 60))]
            self.assertAlmostEqual(stats.max_drawdown(r)["depth"], brute_max_drawdown(r), places=12)

    def test_known_episode_and_recovery(self):
        r = [0.10, -0.20, -0.10, 0.30, 0.50]
        eps = stats.drawdown_episodes(r)
        worst = eps[0]
        self.assertAlmostEqual(worst["depth"], (1.1 * 0.8 * 0.9) / 1.1 - 1)
        self.assertEqual((worst["peak"], worst["trough"]), (1, 3))
        self.assertEqual(worst["recovery"], 5)                         # wealth 1.1*0.8*0.9*1.3*1.5 = 1.5444 > 1.1 at the last period

    def test_unrecovered_drawdown_and_none(self):
        eps = stats.drawdown_episodes([0.1, -0.1, -0.1])
        self.assertIsNone(eps[0]["recovery"])
        self.assertEqual(stats.drawdown_episodes([0.01, 0.02, 0.03]), [])
        self.assertEqual(stats.max_drawdown([0.01, 0.02])["depth"], 0.0)

    def test_multiple_episodes_sorted_by_depth(self):
        r = [0.1, -0.05, 0.1, -0.2, 0.3]
        depths = [e["depth"] for e in stats.drawdown_episodes(r)]
        self.assertEqual(depths, sorted(depths))
        self.assertEqual(len(depths), 2)

    def test_drawdown_while_starting_down(self):
        eps = stats.drawdown_episodes([-0.1, -0.1, 0.5])
        self.assertEqual((eps[0]["peak"], eps[0]["trough"]), (0, 2))
        self.assertAlmostEqual(eps[0]["depth"], 0.81 - 1)


class MetricsTests(unittest.TestCase):
    def test_metrics_bundle_is_consistent(self):
        m = stats.metrics(SMALL, 252)
        self.assertEqual(m["observations"], 10)
        self.assertAlmostEqual(m["years"], 10 / 252)
        self.assertAlmostEqual(m["calmar"], m["cagr"] / abs(m["max_drawdown"]))
        self.assertAlmostEqual(m["hit_rate"], 0.5)
        self.assertAlmostEqual(m["profit_factor"], sum(v for v in SMALL if v > 0) / -sum(v for v in SMALL if v < 0))
        self.assertEqual((m["best_period"], m["worst_period"]), (0.03, -0.02))
        self.assertTrue(m["recovered"])

    def test_no_losses_means_undefined_ratios(self):
        m = stats.metrics([0.01, 0.02, 0.03], 12)
        self.assertIsNone(m["profit_factor"])
        self.assertIsNone(m["calmar"])
        self.assertEqual(m["max_drawdown"], 0.0)

    def test_too_few_observations(self):
        with self.assertRaises(stats.StatsError):
            stats.metrics([0.01], 252)


if __name__ == "__main__":
    unittest.main()
