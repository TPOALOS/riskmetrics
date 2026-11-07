import math
import random
import unittest
from statistics import NormalDist

from riskmetrics import bootstrap, multitest as mt, stats

N = NormalDist()


class SharpeInferenceTests(unittest.TestCase):
    def test_annualise_roundtrip(self):
        self.assertAlmostEqual(mt.deannualise(mt.annualise(0.1, 252), 252), 0.1)
        self.assertAlmostEqual(mt.annualise(0.1, 4), 0.2)

    def test_normal_returns_variance_factor(self):
        self.assertAlmostEqual(mt.sharpe_variance_factor(0.1, 0.0, 3.0), 1 + 0.5 * 0.01)       # the classic 1 + SR^2/2
        self.assertAlmostEqual(mt.sharpe_std_error(0.1, 101), math.sqrt((1 + 0.005) / 100))
        self.assertTrue(math.isnan(mt.sharpe_std_error(0.1, 1)))

    def test_fat_tails_and_negative_skew_widen_the_error(self):
        base = mt.sharpe_std_error(0.2, 500)
        self.assertGreater(mt.sharpe_std_error(0.2, 500, skew=-1.0, kurt=8.0), base)
        self.assertLess(mt.sharpe_std_error(0.2, 500, skew=1.0, kurt=3.0), base)

    def test_psr_matches_the_formula(self):
        sr, n, bench, skew, kurt = 0.12, 250, 0.05, -0.4, 5.0
        denom = math.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr * sr)
        self.assertAlmostEqual(mt.psr(sr, n, bench, skew, kurt), N.cdf((sr - bench) * math.sqrt(n - 1) / denom))

    def test_psr_behaves(self):
        self.assertAlmostEqual(mt.psr(0.0, 100), 0.5)                          # Sharpe equal to the benchmark: coin flip
        self.assertGreater(mt.psr(0.1, 400), mt.psr(0.1, 100))                 # more data, more confidence
        self.assertLess(mt.psr(-0.1, 100), 0.5)
        self.assertGreater(mt.psr(0.1, 100, benchmark=0.0), mt.psr(0.1, 100, benchmark=0.05))
        self.assertTrue(math.isnan(mt.psr(0.1, 1)))
        self.assertTrue(math.isnan(mt.psr(1.0, 100, 0.0, 5.0, 3.0)))          # impossible moment combination

    def test_p_value_is_one_minus_psr(self):
        self.assertAlmostEqual(mt.p_value(0.1, 200), 1 - mt.psr(0.1, 200))
        self.assertAlmostEqual(mt.p_value(0.0, 200), 0.5)

    def test_minimum_track_record(self):
        n_star = mt.min_track_record(0.1, 0.0, 0.95)
        self.assertAlmostEqual(n_star, 1 + (1 + 0.005) * (N.inv_cdf(0.95) / 0.1) ** 2)
        self.assertGreater(mt.min_track_record(0.05), mt.min_track_record(0.1))   # weaker edge needs a longer record
        self.assertEqual(mt.min_track_record(0.0), float("inf"))
        self.assertEqual(mt.min_track_record(-0.1, 0.0), float("inf"))
        # at n* observations PSR equals the requested confidence
        self.assertAlmostEqual(mt.psr(0.1, n_star), 0.95, places=6)


class MultipleTestingTests(unittest.TestCase):
    def test_expected_max_sharpe(self):
        self.assertEqual(mt.expected_max_sharpe(1, 1.0), 0.0)
        self.assertAlmostEqual(mt.expected_max_sharpe(100, 1.0), 2.53, delta=0.05)           # best of 100 standard normals is about 2.5
        self.assertAlmostEqual(mt.expected_max_sharpe(100, 0.5), 0.5 * mt.expected_max_sharpe(100, 1.0))
        values = [mt.expected_max_sharpe(n, 1.0) for n in (2, 5, 10, 50, 500, 5000)]
        self.assertEqual(values, sorted(values))
        with self.assertRaises(ValueError):
            mt.expected_max_sharpe(0, 1.0)

    def test_expected_max_matches_simulation(self):
        rng = random.Random(11)
        sim = sum(max(rng.gauss(0, 1) for _ in range(50)) for _ in range(4000)) / 4000
        self.assertAlmostEqual(mt.expected_max_sharpe(50, 1.0), sim, delta=0.1)

    def test_dsr_with_one_trial_is_the_psr_against_zero(self):
        self.assertAlmostEqual(mt.dsr(0.12, 300, 1), mt.psr(0.12, 300, 0.0))
        self.assertAlmostEqual(mt.dsr(0.12, 300, 1, skew=-0.5, kurt=6.0), mt.psr(0.12, 300, 0.0, -0.5, 6.0))

    def test_dsr_falls_with_trials_and_rises_with_data(self):
        by_trials = [mt.dsr(0.15, 500, n) for n in (1, 5, 20, 100, 1000)]
        self.assertEqual(by_trials, sorted(by_trials, reverse=True))
        by_obs = [mt.dsr(0.15, n, 50) for n in (100, 300, 1000, 5000)]
        self.assertEqual(by_obs, sorted(by_obs))
        self.assertTrue(all(0.0 <= v <= 1.0 for v in by_trials + by_obs))

    def test_dsr_uses_the_given_spread(self):
        wide = mt.dsr(0.15, 500, 50, sharpe_std=0.2)
        narrow = mt.dsr(0.15, 500, 50, sharpe_std=0.02)
        self.assertLess(wide, narrow)

    def test_the_classic_lucky_backtest_gets_deflated(self):
        # 1 year of daily data, annual Sharpe 2.0, picked as the best of 200 random tries
        sp = mt.deannualise(2.0, 252)
        self.assertGreater(mt.psr(sp, 252), 0.97)             # looks great on its own...
        self.assertLess(mt.dsr(sp, 252, 200), 0.6)            # ...but it is what luck produces

    def test_holm_known_example(self):
        self.assertEqual([round(x, 10) for x in mt.holm([0.01, 0.04, 0.03])], [0.03, 0.06, 0.06])
        self.assertEqual(mt.holm([0.5]), [0.5])
        self.assertEqual(mt.holm([]), [])

    def test_holm_properties(self):
        rng = random.Random(2)
        for _ in range(100):
            p = [rng.random() for _ in range(rng.randint(1, 10))]
            h = mt.holm(p)
            self.assertTrue(all(a >= b - 1e-12 for a, b in zip(h, p)))                 # never smaller than the raw p
            self.assertTrue(all(0 <= v <= 1 for v in h))
            order = sorted(range(len(p)), key=lambda i: p[i])
            adj_sorted = [h[i] for i in order]
            self.assertEqual(adj_sorted, sorted(adj_sorted))                           # monotone in the raw ordering
            b = mt.bonferroni(p)
            self.assertTrue(all(x <= y + 1e-12 for x, y in zip(h, b)))                 # Holm is never worse than Bonferroni

    def test_bonferroni(self):
        self.assertEqual(mt.bonferroni([0.01, 0.6]), [0.02, 1.0])

    def test_haircut_sharpe(self):
        sp = mt.deannualise(1.5, 252)
        none_lost, p1 = mt.haircut_sharpe(sp, 750, 1)
        self.assertAlmostEqual(none_lost, N.inv_cdf(1 - p1) / math.sqrt(749))
        h10, p10 = mt.haircut_sharpe(sp, 750, 10)
        h100, p100 = mt.haircut_sharpe(sp, 750, 100)
        self.assertLess(h10, none_lost)
        self.assertLessEqual(h100, h10)
        self.assertLess(p1, p10)
        self.assertEqual(mt.haircut_sharpe(sp, 750, 100000)[0], 0.0)                   # wiped out entirely
        self.assertLessEqual(none_lost, sp * 1.0001)


class BootstrapTests(unittest.TestCase):
    def sample(self, n=400, drift=0.001, seed=4):
        rng = random.Random(seed)
        return [rng.gauss(drift, 0.01) for _ in range(n)]

    def test_resample_has_same_length_and_only_original_values(self):
        r = self.sample(50)
        rs = bootstrap.stationary_resample(r, random.Random(1), 5.0)
        self.assertEqual(len(rs), 50)
        self.assertTrue(set(rs) <= set(r))

    def test_block_length_one_is_the_ordinary_bootstrap(self):
        r = list(range(100))
        rs = bootstrap.stationary_resample(r, random.Random(1), 1.0)
        consecutive = sum(1 for a, b in zip(rs, rs[1:]) if b == (a + 1) % 100)
        self.assertLess(consecutive, 10)                               # almost never consecutive by chance
        long_blocks = bootstrap.stationary_resample(r, random.Random(1), 50.0)
        consecutive_long = sum(1 for a, b in zip(long_blocks, long_blocks[1:]) if b == (a + 1) % 100)
        self.assertGreater(consecutive_long, 80)                       # long blocks keep the original order

    def test_reproducible_with_seed_and_different_without(self):
        r = self.sample()
        a = bootstrap.confidence_intervals(r, 252, resamples=200, seed=9)
        b = bootstrap.confidence_intervals(r, 252, resamples=200, seed=9)
        c = bootstrap.confidence_intervals(r, 252, resamples=200, seed=10)
        self.assertEqual(a, b)
        self.assertNotEqual(a["sharpe"], c["sharpe"])

    def test_intervals_bracket_the_estimate_and_are_ordered(self):
        r = self.sample(600)
        ci = bootstrap.confidence_intervals(r, 252, resamples=400, seed=1)
        point = stats.sharpe(r, 252)
        lo, hi = ci["sharpe"]
        self.assertLess(lo, point)
        self.assertGreater(hi, point)
        for key in ("sharpe", "cagr", "max_drawdown"):
            self.assertLessEqual(ci[key][0], ci[key][1])
        self.assertLess(ci["max_drawdown"][1], 0.0)

    def test_more_data_gives_narrower_intervals(self):
        small = bootstrap.confidence_intervals(self.sample(150), 252, resamples=300, seed=1)["sharpe"]
        large = bootstrap.confidence_intervals(self.sample(1500), 252, resamples=300, seed=1)["sharpe"]
        self.assertLess(large[1] - large[0], small[1] - small[0])

    def test_coverage_is_roughly_nominal(self):
        # true Sharpe of N(mu, sigma) per period is mu/sigma; check the 90% interval covers it often enough
        mu, sigma, covered, trials = 0.0005, 0.01, 0, 40
        truth = mu / sigma * math.sqrt(252)
        for s in range(trials):
            rng = random.Random(100 + s)
            r = [rng.gauss(mu, sigma) for _ in range(500)]
            lo, hi = bootstrap.confidence_intervals(r, 252, resamples=300, level=0.90, seed=s)["sharpe"]
            covered += lo <= truth <= hi
        self.assertGreaterEqual(covered / trials, 0.7)

    def test_input_validation(self):
        with self.assertRaises(stats.StatsError):
            bootstrap.confidence_intervals([0.01] * 5, 252)
        with self.assertRaises(stats.StatsError):
            bootstrap.confidence_intervals([0.01, -0.01] * 20, 252, resamples=10)

    def test_flat_series_has_no_sharpe_interval(self):
        ci = bootstrap.confidence_intervals([0.001] * 30, 252, resamples=100, seed=1)
        self.assertEqual(ci["sharpe"], (None, None))

    def test_default_block_length(self):
        self.assertEqual(bootstrap.default_block(1000), 10.0)
        self.assertEqual(bootstrap.default_block(1), 1.0)


if __name__ == "__main__":
    unittest.main()
