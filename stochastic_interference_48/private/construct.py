#!/usr/bin/env python3
"""Construct the 48-module planted stochastic-nullspace experiment."""
import json
from pathlib import Path
import numpy as np
from scipy.linalg import hadamard
try:
    from numba import njit
except ImportError:
    njit = lambda f: f

@njit
def _iterate_markov(probs, uniforms, initial, mask):
    out = np.empty(len(uniforms), dtype=np.uint8)
    ctx = initial
    for t in range(len(uniforms)):
        bit = int(uniforms[t] < probs[ctx])
        out[t] = bit
        ctx = ((ctx << 1) & mask) | bit
    return out

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "task" / "environment" / "data"
DATA.mkdir(parents=True, exist_ok=True)
SEED = 20261008
N_MODULES = 48
DEPTH = 7
AMPLITUDE = 1.2
EXCITATION = 0.65
N_SYMBOLS = 130000
N_PROBES = 64
N_PROBE_SYMBOLS = 130000

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def generate_bits(probs, n, rng, initial=0):
    depth = int(np.log2(len(probs)))
    uniforms = rng.random(n)
    return _iterate_markov(probs, uniforms, initial, (1 << depth) - 1)

def write_stream(path, bits):
    path.write_bytes((bits + 48).tobytes() + b"\n")

def build():
    rng = np.random.default_rng(SEED)
    signs = rng.choice([-1, 1], size=N_MODULES).astype(int)
    signs[0] = 1
    nctx = 1 << DEPTH
    feature = np.zeros((nctx, N_MODULES), dtype=int)
    for w in range(nctx):
        cw = w ^ (1 << (DEPTH - 1))
        if w > cw:
            continue
        weighted = np.array([1] * (N_MODULES // 2) +
                            [-1] * (N_MODULES // 2))
        rng.shuffle(weighted)
        feature[w] = weighted * signs
        feature[cw] = -feature[w]
    assert np.all(feature @ signs == 0)
    assert np.linalg.matrix_rank(feature) == N_MODULES - 1
    logits = AMPLITUDE * feature

    for i in range(N_MODULES):
        for name, sign in [("plus", 1), ("minus", -1)]:
            brng = np.random.default_rng(SEED + 1000 + 25 * i + (sign < 0))
            b = generate_bits(sigmoid(sign * EXCITATION * logits[:, i]),
                              N_SYMBOLS, brng, initial=i % nctx)
            write_stream(DATA / f"module_{i+1:02d}_{name}.txt", b)

    probe_logits = []
    h = hadamard(nctx // 2)
    for j in range(N_PROBES):
        p = np.concatenate([h[:, j], -h[:, j]]).astype(float) * 0.9
        probe_logits.append(p.tolist())
        b = generate_bits(sigmoid(p), N_PROBE_SYMBOLS,
                          np.random.default_rng(SEED + 5000 + j),
                          initial=j % nctx)
        write_stream(DATA / f"probe_{j+1:02d}.txt", b)

    cal_controls = []
    for k in range(6):
        controls = np.zeros(N_MODULES)
        chosen = rng.choice(N_MODULES, size=(2 if k < 3 else 3),
                            replace=False)
        controls[chosen] = EXCITATION * rng.choice([-1, 1], size=len(chosen))
        cal_controls.append(controls.tolist())
        b = generate_bits(sigmoid(logits @ controls), N_SYMBOLS,
                          np.random.default_rng(SEED + 8000 + k),
                          initial=k % nctx)
        write_stream(DATA / f"calibration_{k+1:02d}.txt", b)

    (DATA / "calibration.json").write_text(
        json.dumps({"controls": cal_controls}, indent=2) + "\n")
    (DATA / "experiment.json").write_text(json.dumps({
        "n_modules": N_MODULES,
        "excitation_strength": EXCITATION,
        "streams_per_module": ["plus", "minus"],
        "n_probes": N_PROBES,
        "n_calibrations": 6,
        "alphabet": [0, 1],
        "purpose": "Recover normalized signs of the maximum-entropy unit-strength joint experiment."
    }, indent=2) + "\n")
    (ROOT / "private" / "truth.json").write_text(json.dumps({
        "signs": signs.tolist(),
        "depth": DEPTH,
        "amplitude": AMPLITUDE,
        "feature_matrix": feature.tolist(),
        "probe_logits": probe_logits
    }, indent=2) + "\n")
    print("Generated", N_MODULES, "modules;", N_PROBES,
          "probes; rank", np.linalg.matrix_rank(logits))

if __name__ == "__main__":
    build()
