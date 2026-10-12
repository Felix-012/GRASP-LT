import csv, collections, statistics
import os, sys
CSV = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("GRASP_CLASSIFIER_CSV", os.path.join(os.environ.get("GRASP_OUT", "outputs"), "paper", "tables", "classifier_runs.csv"))
rows=list(csv.DictReader(open(CSV)))
# Per-class F1 columns only. Do not count `macro_f1` as a fourteenth class.
LABELS=[c[:-3] for c in rows[0] if c.endswith("_f1") and c != "macro_f1"]
data=collections.defaultdict(list)
for r in rows:
    try: data[r["model"]].append({l:float(r[l+"_f1"]) for l in LABELS})
    except: pass
def nz(m): 
    return {l:(statistics.mean(c[l] for c in data[m])>1e-9) for l in LABELS}
def mcnemar(a,b):  # classes a-recovers-not-b (b_disc) vs b-not-a (c_disc)
    A,B=nz(a),nz(b); bd=sum(1 for l in LABELS if A[l] and not B[l]); cd=sum(1 for l in LABELS if B[l] and not A[l])
    # Exact binomial test on discordant classes.
    from math import comb
    n=bd+cd; k=min(bd,cd)
    one_sided=sum(comb(n,i) for i in range(0,k+1))/2**n if n>0 else 1.0
    two_sided=min(1.0, 2.0*one_sided)
    return bd,cd,one_sided,two_sided
for base,disp in [("real","Real"),("mocle","MoCLE"),
                  ("moe_grasp_highdim","MoE-GRASP-HighDim"),("moe_grasp_lowdim","MoE-GRASP-LowDim")]:
    bd,cd,p1,p2=mcnemar("grasp_highdim",base)
    print(f"GRASP-HighDim vs {disp:18s}: recovers {bd} classes it lacks, lacks {cd} it recovers | exact directional p={p1:.4f}, two-sided p={p2:.4f}")
