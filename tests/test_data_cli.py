import io
import json
import os
import random
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from riskmetrics import DISCLAIMER, cli, data

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Base(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name

    def write(self, name, text):
        path = os.path.join(self.dir, name)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        return path

    def synth(self, name="r.csv", n=400, cols=("alpha", "beta", "noise"), drifts=(0.0008, 0.0002, 0.0), seed=3):
        rng = random.Random(seed)
        rows = ["date," + ",".join(cols)]
        for i in range(n):
            rows.append("2021-%04d,%s" % (i, ",".join("%.5f" % rng.gauss(d, 0.01) for d in drifts)))
        return self.write(name, "\n".join(rows) + "\n")

    def run_cli(self, *argv):
        out = io.StringIO()
        with mock.patch("sys.stderr", io.StringIO()) as err:
            code = cli.main(list(argv), out=out)
        return code, out.getvalue(), err.getvalue()


class ParseTests(unittest.TestCase):
    def test_parse_number(self):
        self.assertEqual(data.parse_number("0.5"), 0.5)
        self.assertEqual(data.parse_number(" -1.5% "), -0.015)
        self.assertEqual(data.parse_number("1e-3"), 0.001)
        for missing in ("", "nan", "NaN", "NULL", "n/a", "-", "--", "None", "inf"):
            self.assertIsNone(data.parse_number(missing), missing)
        with self.assertRaises(ValueError):
            data.parse_number("abc")
        with self.assertRaises(ValueError):
            data.parse_number("1,5")


class LoadTests(Base):
    def test_header_dates_and_columns(self):
        ds = data.load_csv(self.write("a.csv", "date,a,b\n2020-01-01,0.01,0.02\n2020-01-02,-0.01,0.0\n2020-01-03,0.02,0.01\n"))
        self.assertEqual(list(ds.series), ["a", "b"])
        self.assertEqual(ds.series["a"], [0.01, -0.01, 0.02])
        self.assertEqual(ds.labels["a"], ["2020-01-01", "2020-01-02", "2020-01-03"])

    def test_no_header_and_semicolons_and_percent(self):
        ds = data.load_csv(self.write("c.csv", "a;b\n1%;2%\n-1%;0%\n2%;1%\n"))
        self.assertEqual(ds.series["a"], [0.01, -0.01, 0.02])
        ds = data.load_csv(self.write("d.csv", "0.01\n-0.02\n0.03\n"))
        self.assertEqual(list(ds.series), ["col1"])

    def test_tab_separated_and_bom(self):
        ds = data.load_csv(self.write("t.csv", "﻿a\tb\n0.01\t0.02\n0.02\t-0.01\n"))
        self.assertEqual(list(ds.series), ["a", "b"])

    def test_column_selection(self):
        path = self.write("e.csv", "date,a,b,c\nx1,0.01,0.02,0.03\nx2,0.02,0.01,0.0\nx3,0.0,0.0,0.01\n")
        self.assertEqual(list(data.load_csv(path, ["B"]).series), ["b"])
        self.assertEqual(list(data.load_csv(path, ["2", "c"]).series), ["a", "c"])
        with self.assertRaises(data.DataError):
            data.load_csv(path, ["zzz"])
        with self.assertRaises(data.DataError):
            data.load_csv(path, ["9"])

    def test_missing_values_are_dropped_with_a_warning(self):
        ds = data.load_csv(self.write("m.csv", "date,a\nd1,0.01\nd2,\nd3,nan\nd4,0.02\nd5,0.03\n"))
        self.assertEqual(ds.series["a"], [0.01, 0.02, 0.03])
        self.assertEqual(ds.labels["a"], ["d1", "d4", "d5"])
        self.assertTrue(any("ignored 2" in w for w in ds.warnings))

    def test_non_numeric_columns_are_skipped(self):
        ds = data.load_csv(self.write("n.csv", "date,note,a\nd1,hello,0.01\nd2,world,0.02\nd3,x,0.0\n"))
        self.assertEqual(list(ds.series), ["a"])
        self.assertTrue(any("note" in w for w in ds.warnings))

    def test_prices_become_returns(self):
        ds = data.load_csv(self.write("p.csv", "date,close\nd1,100\nd2,110\nd3,99\n"), prices=True)
        self.assertAlmostEqual(ds.series["close"][0], 0.10)
        self.assertAlmostEqual(ds.series["close"][1], -0.10)
        self.assertEqual(ds.labels["close"], ["d2", "d3"])

    def test_prices_must_be_positive(self):
        with self.assertRaises(data.DataError):
            data.load_csv(self.write("p0.csv", "c\n100\n0\n50\n"), prices=True)

    def test_prices_given_as_returns_are_caught(self):
        with self.assertRaises(data.DataError):
            data.load_csv(self.write("big.csv", "r\n0.1\n-1.5\n0.2\n"))
        ds = data.load_csv(self.write("pr.csv", "close\n100\n101\n102\n"))
        self.assertTrue(any("--prices" in w for w in ds.warnings))

    def test_errors(self):
        for name, text in (("empty.csv", ""), ("one.csv", "a\n0.01\n"), ("alltext.csv", "a,b\nx,y\nz,w\n"), ("single.csv", "a\n0.01\nnan\nnan\n")):
            with self.assertRaises(data.DataError, msg=name):
                data.load_csv(self.write(name, text))
        with self.assertRaises(data.DataError):
            data.load_csv(os.path.join(self.dir, "missing.csv"))
        with self.assertRaises(data.DataError):
            data.load_csv(self.write("dc.csv", "date,a\nd1,0.01\nd2,0.02\n"), date_column="nope")

    def test_explicit_date_column_and_ragged_rows(self):
        ds = data.load_csv(self.write("r.csv", "when,a\nd1,0.01\nd2\nd3,0.02\nd4,0.03\n"), date_column="when")
        self.assertEqual(ds.series["a"], [0.01, 0.02, 0.03])


class SummaryCommandTests(Base):
    def test_text_summary_with_bootstrap(self):
        code, out, _ = self.run_cli("summary", self.synth(), "--col", "alpha", "--bootstrap", "200", "--seed", "1")
        self.assertEqual(code, 0)
        for part in ("Sharpe (ann.)", "Max drawdown", "CVaR 95%", "Bootstrap Sharpe", "seed 1", DISCLAIMER):
            self.assertIn(part, out)

    def test_json_summary(self):
        code, out, _ = self.run_cli("summary", self.synth(), "--format", "json", "--freq", "weekly", "--bootstrap", "150", "--seed", "2")
        d = json.loads(out)
        self.assertEqual(d["periods_per_year"], 52)
        self.assertEqual(set(d["series"]), {"alpha", "beta", "noise"})
        s = d["series"]["alpha"]
        self.assertEqual(s["observations"], 400)
        self.assertIn("bootstrap", s)
        self.assertLessEqual(s["sharpe_ci"][0], s["sharpe"])
        self.assertGreaterEqual(s["sharpe_ci"][1], s["sharpe"])
        self.assertIn("disclaimer", d)

    def test_periods_per_year_option_scales_the_sharpe(self):
        path = self.synth()
        sharpe = lambda *a: json.loads(self.run_cli("summary", path, "--col", "alpha", "--format", "json", *a)[1])["series"]["alpha"]["sharpe"]
        self.assertAlmostEqual(sharpe("--periods-per-year", "2520") / sharpe("--periods-per-year", "252"), 10 ** 0.5, places=6)

    def test_prices_and_risk_free_rate(self):
        path = self.write("px.csv", "date,px\n" + "\n".join("d%d,%.4f" % (i, 100 * 1.0005 ** i * (1 + 0.01 * ((i * 7) % 5 - 2))) for i in range(300)) + "\n")
        code, out, _ = self.run_cli("summary", path, "--prices", "--format", "json")
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)["series"]["px"]["observations"], 299)
        a = json.loads(self.run_cli("summary", path, "--prices", "--format", "json")[1])["series"]["px"]["sharpe"]
        b = json.loads(self.run_cli("summary", path, "--prices", "--format", "json", "--rf", "0.05")[1])["series"]["px"]["sharpe"]
        self.assertLess(b, a)

    def test_flat_series_does_not_crash(self):
        code, out, _ = self.run_cli("summary", self.write("flat.csv", "r\n0.001\n0.001\n0.001\n0.001\n"))
        self.assertEqual(code, 0)
        self.assertIn("n/a", out)
        self.assertEqual(json.loads(self.run_cli("summary", self.write("flat2.csv", "r\n0.001\n0.001\n0.001\n"), "--format", "json")[1])["series"]["r"]["sharpe"], None)

    def test_errors_exit_2(self):
        for argv in (("summary", "/nonexistent.csv"), ("summary", self.write("bad.csv", "a\n0.1\n")), ("summary", self.synth(), "--periods-per-year", "-5"),
                     ("summary", self.synth(), "--bootstrap", "20"), ("summary", self.synth(), "--col", "nope")):
            code, _, err = self.run_cli(*argv)
            self.assertEqual(code, 2, argv)
            self.assertIn("riskmetrics: error", err)


class DrawdownCommandTests(Base):
    def test_drawdown_table(self):
        path = self.write("dd.csv", "date,r\nd1,0.1\nd2,-0.2\nd3,-0.1\nd4,0.3\nd5,0.5\nd6,-0.05\n")
        code, out, _ = self.run_cli("drawdowns", path, "--top", "3")
        self.assertEqual(code, 0)
        self.assertIn("d1", out)
        self.assertIn("d3", out)
        self.assertIn("d5", out)
        self.assertIn("-28.0%", out)                    # 1.1 -> 0.8 * 0.9 * 1.1 = 0.792: down 28% from the d1 peak
        self.assertIn(DISCLAIMER, out)

    def test_no_drawdowns(self):
        code, out, _ = self.run_cli("drawdowns", self.write("up.csv", "r\n0.01\n0.02\n0.03\n"))
        self.assertIn("no drawdowns", out)


class CompareCommandTests(Base):
    def test_ranking_and_columns(self):
        code, out, _ = self.run_cli("compare", self.synth(), "--trials", "50")
        self.assertEqual(code, 0)
        lines = [l for l in out.splitlines() if l.startswith(("alpha", "beta", "noise"))]
        self.assertEqual([l.split()[0] for l in lines][0], "alpha")                   # highest Sharpe first
        self.assertIn("Both account for 50 trials", out)
        self.assertIn(DISCLAIMER, out)

    def test_json_and_survivor_logic(self):
        strong = self.synth("s.csv", n=3000, cols=("edge", "dud"), drifts=(0.004, 0.0), seed=8)
        d = json.loads(self.run_cli("compare", strong, "--format", "json")[1])
        ranking = {r["name"]: r for r in d["ranking"]}
        self.assertTrue(ranking["edge"]["survives"])
        self.assertFalse(ranking["dud"]["survives"])
        self.assertEqual(d["trials"], 2)
        for r in d["ranking"]:
            self.assertGreaterEqual(r["holm"], r["p"])

    def test_more_declared_trials_make_survival_harder(self):
        path = self.synth("t.csv", n=700, cols=("a", "b"), drifts=(0.0012, 0.0), seed=5)
        dsr = lambda *a: {r["name"]: r["dsr"] for r in json.loads(self.run_cli("compare", path, "--format", "json", *a)[1])["ranking"]}["a"]
        holm = lambda *a: {r["name"]: r["holm"] for r in json.loads(self.run_cli("compare", path, "--format", "json", *a)[1])["ranking"]}["a"]
        self.assertLess(dsr("--trials", "500"), dsr("--trials", "2"))
        self.assertGreater(holm("--trials", "500"), holm("--trials", "2"))

    def test_input_checks(self):
        self.assertEqual(self.run_cli("compare", self.synth(cols=("only",), drifts=(0.001,)))[0], 2)
        self.assertEqual(self.run_cli("compare", self.synth(), "--trials", "2")[0], 2)


class DeflateCommandTests(Base):
    def test_text_and_json(self):
        code, out, _ = self.run_cli("deflate", "--sharpe", "1.5", "--n-obs", "750", "--trials", "100")
        self.assertEqual(code, 0)
        for part in ("Deflated Sharpe Ratio", "Haircut Sharpe", "Track record needed", "spread assumed", DISCLAIMER):
            self.assertIn(part, out)
        d = json.loads(self.run_cli("deflate", "--sharpe", "1.5", "--n-obs", "750", "--trials", "100", "--format", "json")[1])
        self.assertTrue(d["trial_sharpe_std_assumed"])
        self.assertLess(d["dsr"], d["psr_vs_zero"])
        self.assertLess(d["haircut_sharpe_annual"], 1.5)

    def test_more_trials_lower_the_dsr(self):
        dsr = lambda n: json.loads(self.run_cli("deflate", "--sharpe", "1.5", "--n-obs", "750", "--trials", str(n), "--format", "json")[1])["dsr"]
        self.assertGreater(dsr(1), dsr(10))
        self.assertGreater(dsr(10), dsr(1000))

    def test_trial_sharpes_file_sets_the_spread(self):
        path = self.write("trials.txt", "# annualised Sharpe ratios of every variant\n0.2\n0.5\n-0.1\n0.9\n0.3\n1.1\n")
        code, out, _ = self.run_cli("deflate", "--sharpe", "1.1", "--n-obs", "500", "--trials", "6", "--trial-sharpes", path, "--format", "json")
        d = json.loads(out)
        self.assertEqual(code, 0)
        self.assertFalse(d["trial_sharpe_std_assumed"])
        code, out, _ = self.run_cli("deflate", "--sharpe", "1.1", "--n-obs", "500", "--trials", "3", "--trial-sharpes", path)
        self.assertEqual(code, 2)                                                    # fewer trials than numbers in the file

    def test_explicit_spread_and_frequency(self):
        a = json.loads(self.run_cli("deflate", "--sharpe", "1.0", "--n-obs", "500", "--trials", "20", "--trial-sharpe-std", "0.2", "--format", "json")[1])
        b = json.loads(self.run_cli("deflate", "--sharpe", "1.0", "--n-obs", "500", "--trials", "20", "--trial-sharpe-std", "0.8", "--format", "json")[1])
        self.assertGreater(a["dsr"], b["dsr"])
        w = json.loads(self.run_cli("deflate", "--sharpe", "1.0", "--n-obs", "100", "--trials", "5", "--freq", "weekly", "--format", "json")[1])
        self.assertEqual(w["periods_per_year"], 52)

    def test_bad_inputs(self):
        for argv in (("--sharpe", "1", "--n-obs", "2", "--trials", "5"), ("--sharpe", "1", "--n-obs", "100", "--trials", "0")):
            self.assertEqual(self.run_cli("deflate", *argv)[0], 2)
        empty = self.write("e.txt", "# only comments\n")
        self.assertEqual(self.run_cli("deflate", "--sharpe", "1", "--n-obs", "100", "--trials", "5", "--trial-sharpes", empty)[0], 2)


class EntryPointTests(unittest.TestCase):
    def test_module_entry_point(self):
        r = subprocess.run([sys.executable, "-m", "riskmetrics", "--version"], cwd=ROOT, capture_output=True, text=True)
        self.assertIn("1.0.0", r.stdout + r.stderr)

    def test_every_output_carries_the_disclaimer(self):
        self.assertIn("not investment advice", DISCLAIMER)


if __name__ == "__main__":
    unittest.main()
