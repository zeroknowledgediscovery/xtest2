"""Parameter sweep with more expressive real native LSmash PFSA projectors."""
import json,time
from pathlib import Path
import numpy as np
import lsmash as ls
from experiment_native import example,winset,localization

def main():
    rows=[]
    for N in [60000,180000]:
        for seed in [12554,12557,12560]:
            seq,cp=example(N,seed)
            for window in [500,2000,8000]:
                center,windows=winset(seq,window,cap=160)
                t=time.monotonic()
                feats=ls._lsmash.from_sequences_random_projectors(windows,32,32,1007,False)
                found=localization(center,feats)
                row=dict(total_symbols=N,true_cp=cp,seed=seed,
                         window=window,n_windows=len(windows),n_probes=32,probe_states=32,
                         estimated_cp=found,abs_error=abs(found-cp),
                         elapsed_seconds=round(time.monotonic()-t,2))
                rows.append(row)
                print("RESULT",json.dumps(row),flush=True)
    Path("pfsa_cp_lsmash/results_native_32state.json").write_text(
        json.dumps({"rows":rows,"native_pfsa_likelihood":True},indent=2)+"\n")
if __name__=="__main__":main()
