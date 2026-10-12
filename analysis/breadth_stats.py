import csv, collections, statistics
import numpy as np
from scipy.stats import wilcoxon

import os, sys
CSV = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("GRASP_CLASSIFIER_CSV", os.path.join(os.environ.get("GRASP_OUT", "outputs"), "paper", "tables", "classifier_runs.csv"))
rows=list(csv.DictReader(open(CSV)))
# Per-class F1 columns only. `macro_f1` also ends in `_f1` but is not a class.
LABELS=[c[:-3] for c in rows[0] if c.endswith("_f1") and c != "macro_f1"]
def key(r): return (r["group"], r["seed"])  # (gen-seed, classifier-seed)

# per (model) -> per (cell) -> per-class f1 dict + macro
data=collections.defaultdict(dict)
for r in rows:
    try: m=r["model"]; data[m][key(r)]={l:float(r[l+"_f1"]) for l in LABELS}|{"macro":float(r["macro_f1"])}
    except: pass

def perclass_mean(m):
    cells=list(data[m].values())
    return {l: statistics.mean(c[l] for c in cells) for l in LABELS}

print("=== breadth: nonzero per-class F1 (mean over runs) ===")
NAMES={"real":"Real","grasp_highdim":"GRASP-HighDim","grasp_lowdim":"GRASP-LowDim",
       "grasp_768":"GRASP (768d)","moe_grasp_lowdim":"MoE-GRASP-LowDim",
       "moe_grasp_highdim":"MoE-GRASP-HighDim","mocle":"MoCLE"}
for m,disp in NAMES.items():
    if m not in data: continue
    pc=perclass_mean(m); nz=sum(1 for v in pc.values() if v>1e-9)
    print(f"  {disp:20s} nonzero {nz:2d}/{len(LABELS)}  macro {statistics.mean(c['macro'] for c in data[m].values()):.4f}")

print("\n=== GRASP-HighDim vs Real: per-class (mean over runs) ===")
g=perclass_mean("grasp_highdim"); r=perclass_mean("real")
gt=lt=eq=0; recovered=[]
for l in LABELS:
    d=g[l]-r[l]
    if abs(d)<1e-9: eq+=1
    elif d>0: gt+=1
    else: lt+=1
    if r[l]<1e-9 and g[l]>1e-9: recovered.append((l,round(g[l],3)))
print(f"  classes GRASP>real: {gt} | real>GRASP: {lt} | tie: {eq}  (of {len(LABELS)})")
print(f"  tail classes GRASP recovers (real=0, GRASP>0): {recovered}")
gv=np.array([g[l] for l in LABELS]); rv=np.array([r[l] for l in LABELS])
nz=(gv-rv)!=0
try:
    w=wilcoxon(gv[nz], rv[nz], alternative="greater")
    print(f"  per-class one-sided Wilcoxon (GRASP>real), n_nonzero_pairs={nz.sum()}: p={w.pvalue:.4f}")
except Exception as e: print("  wilcoxon:",e)

print("\n=== directional consistency vs baselines (aligned gen x cls cells) ===")
ref="grasp_highdim"
for m,disp in NAMES.items():
    if m in (ref,"real") or m not in data: continue
    common=sorted(set(data[ref])&set(data[m]))
    wins=sum(1 for k in common if data[ref][k]["macro"]>data[m][k]["macro"])
    md=statistics.mean(data[ref][k]["macro"]-data[m][k]["macro"] for k in common)
    print(f"  GRASP-HighDim > {disp:20s} in {wins}/{len(common)} cells | mean diff {md:+.4f}")
