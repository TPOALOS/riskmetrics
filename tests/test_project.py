import os
import re
import unittest

import riskmetrics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def read(*parts):
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as fh:
        return fh.read()

class ProjectConsistencyTests(unittest.TestCase):
    def test_version_matches_pyproject_and_changelog(self):
        version = re.search(r'^version = "([^"]+)"', read("pyproject.toml"), re.M).group(1)
        self.assertEqual(version, riskmetrics.__version__)
        self.assertIn("## " + version, read("CHANGELOG.md"))

    def test_console_script_target_exists(self):
        target = re.search(r'^riskmetrics = "([^"]+)"', read("pyproject.toml"), re.M).group(1)
        module, func = target.split(":")
        self.assertTrue(callable(getattr(__import__(module, fromlist=[func]), func)))

    def test_workflow_is_pinned_and_least_privilege(self):
        wf = read(".github", "workflows", "tests.yml")
        self.assertIn("permissions:\n  contents: read", wf)
        for ref in re.findall(r"uses: (\S+)", wf):
            self.assertRegex(ref, r"@[0-9a-f]{40}$")
        self.assertNotIn("pull_request_target", wf)

    def test_no_dependencies_are_declared(self):
        self.assertNotRegex(read("pyproject.toml"), r"(?m)^dependencies\s*=")

    def test_examples_contain_no_private_paths(self):
        for name in os.listdir(os.path.join(ROOT, "examples")):
            text = read("examples", name)
            self.assertFalse(os.path.expanduser("~") in text or "/Users/" in text, name)

    def test_sample_data_is_synthetic_and_reproducible(self):
        import subprocess, sys, tempfile, shutil
        before = read("examples", "sample_returns.csv")
        self.assertTrue(before.startswith("date,steady,volatile,coin_flip"))
        self.assertEqual(len(before.splitlines()), 757)
        with tempfile.TemporaryDirectory() as tmp:
            script = os.path.join(tmp, "make_sample.py")
            shutil.copy(os.path.join(ROOT, "examples", "make_sample.py"), script)
            subprocess.run([sys.executable, script], check=True, capture_output=True)
            with open(os.path.join(tmp, "sample_returns.csv")) as fh:
                self.assertEqual(fh.read(), before)

    def test_readme_cites_the_methods_it_implements(self):
        readme = read("README.md")
        for name in ("Lo", "Mertens", "Bailey", "Politis", "Holm"):
            self.assertIn(name, readme)

if __name__ == "__main__":
    unittest.main()
