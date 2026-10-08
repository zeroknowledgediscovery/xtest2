#!/usr/bin/env python3
"""Private reference oracle: estimates differential probe null direction."""
import json
import os
from pathlib import Path
import numpy as np

BASE = Path(os.environ.get("WORKSPACE_DIR", "/workspace"))
DATA = BASE / "data"
OUT = BASE / "output"

def bits(path):
    return np.frombuffer(path.read_bytes().strip(), dtype=np.uint8) - 48

def conditional_table(sequence, depth=7):
    n = 1 << depth
    ns = len(sequence)
    contexts = np.zeros(ns - depth, dtype=np.int32)
    for offset in range(depth):
        contexts = 2 * contexts + sequence[offset:ns-depth+offset]
    ones = np.bincount(contexts, weights=sequence[depth:], minlength=n)
    totals = np.bincount(contexts, minlength=n)
    return (ones + 0.5) / (totals + 1.0)

def kl(p, q):
    return np.mean(p*np.log(p/q) + (1-p)*np.log((1-p)/(1-q)))

def main():
    ex = json.loads((DATA / "experiment.json").read_text())
    n, k, eps = ex["n_modules"], ex["n_probes"], ex["excitation_strength"]
    probes = [
        conditional_table(bits(DATA / f"probe_{j+1:02d}.txt"))
        for j in range(k)
    ]
    J = np.empty((k, n))
    for i in range(n):
        plus = conditional_table(bits(DATA / f"module_{i+1:02d}_plus.txt"))
        minus = conditional_table(bits(DATA / f"module_{i+1:02d}_minus.txt"))
        for j, q in enumerate(probes):
            J[j, i] = (kl(plus,q) - kl(minus,q)) / (2*eps)
    _, singular, vt = np.linalg.svd(J, full_matrices=False)
    c = np.where(vt[-1] >= 0, 1, -1)
    if c[0] < 0:
        c = -c
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "solution.json").write_text(
        json.dumps({"polarities": c.tolist()}, indent=2) + "\n")
    (OUT / "analysis.md").write_text(
        "# Differential stochastic probe analysis\n\n"
        "Estimated conditional probabilities from independent module and reference streams, "
        "computed differential sequence-likelihood divergences for positive and negative "
        "calibration excitations, and inferred the minimum right-singular direction. "
        "Normalized binary signs so the first sign is positive.\n\n"
        f"Smallest two singular values: {singular[-2]:.6g}, {singular[-1]:.6g}.\n"
    )
    print("Signs:", c.tolist())
    print("Two smallest singular values:", singular[-2:])

if __name__ == "__main__":
    main()
