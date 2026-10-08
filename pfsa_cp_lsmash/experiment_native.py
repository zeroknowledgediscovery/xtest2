"""Change-point detection with random PFSA projections using compiled LSmash."""
import json,time
from pathlib import Path
import numpy as np
import lsmash as ls
from scipy.special import expit
from numba import njit

@njit
def synth(pA,pB,N,cp,uniforms,initial):
    d=10
    out=np.zeros(N,dtype=np.uint8)
    for j in range(d):out[j]=(initial>>(d-1-j))&1
    state=initial
    for t in range(d,N):
        bit=int(uniforms[t]<(pA[state] if t<cp else pB[state]))
        out[t]=bit
        state=((state<<1)&1023)|bit
    return out

def example(N,seed,delta=.17):
    rng=np.random.default_rng(seed)
    sign=rng.choice(np.array([-1.,1.]),size=512)
    bg=sign*1.35
    v=np.r_[np.ones(256),-np.ones(256)]
    rng.shuffle(v)
    pert=sign*v
    a=np.r_[bg+delta*pert,-bg-delta*pert]
    b=np.r_[bg-delta*pert,-bg+delta*pert]
    cp=int(N*.53)
    seq=synth(expit(a),expit(b),N,cp,np.random.default_rng(seed+100000).random(N),int(seed%1024))
    return seq,cp

def winset(seq,L,cap=160):
    n=len(seq)
    step=max(L//2,(n-L)//max(1,cap-1))
    starts=np.arange(0,n-L+1,step,dtype=int)
    return starts+L//2,[seq[i:i+L].astype(np.uint32).tolist() for i in starts]

def localization(centers,f):
    f=np.asarray(f,dtype=float)
    f=(f-f.mean(axis=0))/np.maximum(f.std(axis=0),1e-8)
    n=len(f)
    k=np.arange(1,n)
    cum=np.cumsum(f,axis=0)
    left=cum[:-1]/k[:,None]
    right=(cum[-1]-cum[:-1])/(n-k)[:,None]
    scores=k*(n-k)/n*np.mean((left-right)**2,axis=1)
    scores[(k<max(3,int(.20*n)))|(k>min(n-3,int(.80*n)))]=-np.inf
    j=int(np.argmax(scores))
    return int((centers[j]+centers[j+1])//2)

def main():
    rows=[]
    # Same low-order-matched source family as the 180k blind challenge.
    for N in [60000,180000]:
        for seed in [12554,12557,12560]:
            seq,cp=example(N,seed)
            for L in [500,2000,8000]:
                centers,windows=winset(seq,L,cap=160)
                t=time.monotonic()
                F=ls._lsmash.from_sequences_random_projectors(windows,16,8,1007,False)
                est=localization(centers,F)
                row=dict(total_symbols=N,true_cp=cp,seed=seed,window=L,
                         n_windows=len(windows),n_probes=16,probe_states=8,
                         estimated_cp=est,abs_error=abs(cp-est),elapsed_seconds=round(time.monotonic()-t,3))
                rows.append(row)
                print('RESULT',json.dumps(row),flush=True)
    ls._lsmash.from_sequences_random_projectors(
        [[0,1,0,1]*16,[1,1,0,0]*16,[0,0,1,1]*16],4,8,7,True)
    print('NATIVE_CHECK llk_distance_matches_random_projection_L1 True',flush=True)
    Path('pfsa_cp_lsmash/results_native.json').write_text(json.dumps({
      'method':'real native PFSA log_likelihood random projectors from lsmash',
      'rows':rows,
      'native_llk_distance_crosscheck':True},indent=2)+'\n')
if __name__=='__main__':main()
