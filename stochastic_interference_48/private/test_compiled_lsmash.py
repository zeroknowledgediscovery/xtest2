#!/usr/bin/env python3
"""Independently test the compiled lsmash Python/C++ package, not a KL surrogate."""
import json, os, time
from pathlib import Path
import numpy as np
import lsmash as ls

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"task"/"environment"/"data"
TRUTH=np.asarray(json.loads((ROOT/"private"/"truth.json").read_text())["signs"],dtype=int)
names=[f"module_{i:02d}_{pol}.txt" for i in range(1,49) for pol in ("plus","minus")]
names += [f"probe_{j:02d}.txt" for j in range(1,65)]
n=int(os.environ.get("LSMASH_N", "130000"))
t0=time.monotonic()
seqs=[[b-48 for b in (DATA/name).read_bytes().strip()[:n]] for name in names]
print("Loaded",len(seqs),"streams at",n,"symbols each",flush=True)
opt=ls.LsmashOptions()
opt.data_type="symbolic"
opt.sae=False
D=ls.from_sequences(seqs,opt)
print("Compiled LSmash distances computed",D.shape,"elapsed",round(time.monotonic()-t0,2),flush=True)
assert D.shape==(160,160) and np.isfinite(D).all()
J=(D[0:96:2,96:160]-D[1:96:2,96:160]).T
_,singular,V=np.linalg.svd(J,full_matrices=False)
est=np.where(V[-1]>=0,1,-1)
if est[0]<0:est=-est
result={"algorithm":"compiled_lsmash","n":n,
        "correct":int(np.sum(est==TRUTH)),
        "exact":bool(np.array_equal(est,TRUTH)),
        "smallest_singular_values":singular[-3:].tolist(),
        "distance_min":float(D.min()),
        "distance_max":float(D.max()),
        "elapsed_seconds":round(time.monotonic()-t0,2)}
print("RESULT",json.dumps(result),flush=True)
(ROOT/"compiled_lsmash_result.json").write_text(json.dumps(result,indent=2)+"\n")
