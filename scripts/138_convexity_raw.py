import numpy as np
from scipy.stats import spearmanr
import importlib.util, sys

spec = importlib.util.spec_from_file_location("e", "scripts/110_hierarchical_birthdeath_v2_fixed.py")
E = importlib.util.module_from_spec(spec); sys.modules["e"] = E; spec.loader.exec_module(E)

DATA = "data/processed/full_data_graphs_withdel"

recs_t = E.load_month(DATA, "2022-05")   # cloud at t
recs_f = E.load_month(DATA, "2022-11")   # cloud at t+6

# point cloud at t: list of (frozenset, freq)
total_t = sum(c for _, c in recs_t)
cloud_t = [(s, c / total_t) for s, c in recs_t]
present = {s for s, _ in recs_t}

total_f = sum(c for _, c in recs_f)

rows = []
for s_new, c in recs_f:
    w = c / total_f
    if w < 1e-4 or s_new in present:
        continue

    # for each observed constellation at t, how many of s_new's mutations does it cover?
    best_cover = max(len(s_new & s_t) for s_t, _ in cloud_t)
    residual   = len(s_new) - best_cover   # mutations no single constellation covers

    # nearest neighbour distance
    nn_dist = min(len(s_new.symmetric_difference(s_t)) for s_t, _ in cloud_t)

    rows.append((len(s_new), residual, nn_dist, np.log(w)))

A = np.array(rows)
size, resid, nndist, logw = A.T

print(f"novel above threshold: {len(A)}")
print(f"residual==0 (within one constellation): {(resid==0).mean():.1%}")
print(f"residual>0  (cross constellation):      {(resid>0).mean():.1%}")

print("\nsize-stratified gap (cross minus within, in nn_dist):")
for sz in np.unique(size):
    m = size == sz
    wi = nndist[m & (resid == 0)]
    cr = nndist[m & (resid > 0)]
    if len(wi) < 3 or len(cr) < 3: continue
    print(f"  size {int(sz):2d}  within {wi.mean():.2f}  cross {cr.mean():.2f}  "
          f"gap {cr.mean()-wi.mean():+.2f}")

# overall, ignoring size confound
wi = nndist[resid == 0]
cr = nndist[resid > 0]
print(f"\noverall (size-confounded, indicative only):")
print(f"  within  n={len(wi)}  mean nn_dist {wi.mean():.2f}")
print(f"  cross   n={len(cr)}  mean nn_dist {cr.mean():.2f}")
print(f"  gap {cr.mean()-wi.mean():+.2f}")

# coarse size split
for label, m in [("small (<=5)", size<=5), ("large (>5)", size>5)]:
    wi = nndist[m & (resid==0)]
    cr = nndist[m & (resid>0)]
    if len(wi)<2 or len(cr)<2: continue
    print(f"  {label}  within {wi.mean():.2f}  cross {cr.mean():.2f}  gap {cr.mean()-wi.mean():+.2f}")