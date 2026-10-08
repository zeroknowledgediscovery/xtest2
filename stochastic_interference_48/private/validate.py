#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
t=json.loads((ROOT/"private"/"truth.json").read_text())
L=t["amplitude"]*np.asarray(t["feature_matrix"])
s=np.asarray(t["signs"],dtype=int)
assert len(s)==48 and s[0]==1
assert np.linalg.matrix_rank(L)==47
assert np.allclose(L@s,0)
assert np.array_equal(L[:64],-L[64:])
D=ROOT/"task"/"environment"/"data"
assert len(list(D.glob("module_*_plus.txt")))==48
assert len(list(D.glob("module_*_minus.txt")))==48
assert len(list(D.glob("probe_*.txt")))==64
assert len(list(D.glob("calibration_*.txt")))==6
h=lambda p: -(p*np.log2(p)+(1-p)*np.log2(1-p))
p0=1/(1+np.exp(-L@s))
flip=s.copy(); flip[1]*=-1
p1=1/(1+np.exp(-L@flip))
print("generator_rank",np.linalg.matrix_rank(L))
print("exact_null_residual",np.max(np.abs(L@s)))
print("planted_entropy_bits",np.mean(h(p0)))
print("single_wrong_sign_entropy_bits",np.mean(h(p1)))
print("source_files",len(list(D.iterdir())))
