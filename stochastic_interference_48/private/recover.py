#!/usr/bin/env python3
"""Differential-probe nullspace recovery. Not the published compiled LSmash metric."""
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"task"/"environment"/"data"

def bits(path,part):
    a=np.frombuffer(path.read_bytes().strip(),dtype=np.uint8)-48
    return a[:len(a)//2] if part=="first" else a[len(a)//2:] if part=="second" else a

def cond(a,depth):
    n=len(a)-depth
    w=np.zeros(n,dtype=np.int32)
    for i in range(depth): w=2*w+a[i:i+n]
    count=np.bincount(w,minlength=1<<depth)
    one=np.bincount(w,weights=a[depth:],minlength=1<<depth)
    return (one+0.5)/(count+1)

def divergence(p,q):
    return np.mean(p*np.log(p/q)+(1-p)*np.log((1-p)/(1-q)))

def recover(depth=7,part="full"):
    ex=json.loads((DATA/"experiment.json").read_text())
    probes=[cond(bits(DATA/f"probe_{j:02d}.txt",part),depth) for j in range(1,ex["n_probes"]+1)]
    J=np.zeros((len(probes),ex["n_modules"]))
    for i in range(1,ex["n_modules"]+1):
        p=cond(bits(DATA/f"module_{i:02d}_plus.txt",part),depth)
        m=cond(bits(DATA/f"module_{i:02d}_minus.txt",part),depth)
        for j,q in enumerate(probes):
            J[j,i-1]=(divergence(p,q)-divergence(m,q))/(2*ex["excitation_strength"])
    _,s,V=np.linalg.svd(J,full_matrices=False)
    out=np.where(V[-1]>=0,1,-1)
    if out[0]<0: out=-out
    truth=json.loads((ROOT/"private"/"truth.json").read_text())["signs"]
    print(json.dumps({"depth":depth,"part":part,"correct":out.tolist()==truth,
                      "signs":out.tolist(),"smallest_singular_values":s[-2:].tolist()}))
    assert out.tolist()==truth
if __name__=="__main__":
    recover(int(sys.argv[1]) if len(sys.argv)>1 else 7,
            sys.argv[2] if len(sys.argv)>2 else "full")
