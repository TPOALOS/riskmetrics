"""Write examples/sample_returns.csv: three SYNTHETIC daily return series (seeded random numbers, not real data).

    python3 examples/make_sample.py
"""
import os
import random

rng = random.Random(2024)
rows = ["date,steady,volatile,coin_flip"]
for i in range(756):                                   # three years of business days
    rows.append("day%04d,%.5f,%.5f,%.5f" % (i, rng.gauss(0.0006, 0.007), rng.gauss(0.0008, 0.018), rng.gauss(0.0, 0.01)))
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_returns.csv")
with open(path, "w") as fh:
    fh.write("\n".join(rows) + "\n")
print("wrote", path)
