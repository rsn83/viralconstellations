#!/usr/bin/env python
"""
137_lineage_components.py -- run the mechanism tests on components whose
identity is not in question.

THE PROBLEM THIS ADDRESSES
--------------------------
Every mechanism test so far has been null: growth is not predictable from
composition, appearance is not predictable from background, the mixture
weights have almost no headroom. Two readings fit that, and the data as used
cannot separate them.

  the biology  -- lineage growth really is not a function of lineage contents.
  the measure  -- it is, but the components being compared across months are
                  refitted each time, so "component seven" in March and
                  "component seven" in April are not the same object. A
                  regression on their weight change is then comparing
                  different things, and any real effect is destroyed before
                  it can be seen.

Pango lineages settle it. They are assigned per sequence by an external tool,
so a lineage in March IS the same lineage in April, by construction. Substitute
them for fitted components and the second reading is eliminated. If the tests
turn positive, the nulls were an artefact of clustering. If they stay null with
exact identity supplied, the nulls are about the virus.

WHAT IS BUILT
-------------
One pass over the metadata gives, per month and per lineage: how many sequences
it had, and how often each spike mutation appeared within it. That is pi_L(t)
and theta_L directly, with no fitting anywhere -- so no initialisation, no
local optima, no correspondence problem.

Substitutions are read from the amino-acid substitution field, keeping the
spike ones; deletions from the deletion field, matched to the vocabulary by
their nucleotide range. Both are looked up in the same vocabulary the rest of
the project uses, so the mutation indices mean what they mean everywhere else.
"""
import argparse, csv, re, subprocess, sys
from collections import defaultdict
import numpy as np

EPS = 1e-12
SUB = re.compile(r"^S:([A-Z*])(\d+)([A-Z*])$")


def load_vocab(path):
    """(aa_pos, residue) -> node index, plus a deletion-string index."""
    sub, dele, names, V = {}, {}, {}, 0
    with open(path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            i = int(row["node_idx"]); V = max(V, i + 1)
            pos = str(row["aa_pos"]).strip()
            res = row["residue"].strip()
            names[i] = f"{pos}{res}"
            if res.startswith("del"):
                dele[res[3:]] = i          # keyed by the nucleotide range
            else:
                sub[(pos, res)] = i
    return sub, dele, names, V


def stream(path):
    p = subprocess.Popen(["zstdcat", path], stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, text=True, bufsize=1 << 20)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--min-seqs", type=int, default=50,
                    help="ignore a lineage-month with fewer sequences than "
                         "this; a weight change between two tiny counts is "
                         "noise")
    ap.add_argument("--host", default="Human")
    ap.add_argument("--out", default="",
                    help="optional .npz to write pi, theta and labels to")
    ap.add_argument("--lams", default="1,10,100,1000,10000")
    ap.add_argument("--max-rows", type=int, default=0,
                    help="stop after this many rows, for a quick check")
    args = ap.parse_args()

    sub, dele, names, V = load_vocab(args.vocab)
    print(f"vocabulary {V:,} nodes: {len(sub):,} substitutions, "
          f"{len(dele):,} deletions")

    p = stream(args.metadata)
    hdr = p.stdout.readline().rstrip("\n").split("\t")
    col = {c: i for i, c in enumerate(hdr)}
    need = ["date", "pango_lineage", "Nextclade_pango", "deletions",
            "aaSubstitutions", "host"]
    for c in need:
        if c not in col:
            sys.exit(f"metadata is missing column {c}")

    cnt = defaultdict(float)                        # (month, lineage) -> seqs
    mut = defaultdict(lambda: defaultdict(float))   # (month, lineage) -> node
    n = kept = 0
    drop = defaultdict(int)                         # why rows were discarded
    for line in p.stdout:
        n += 1
        if args.max_rows and n > args.max_rows:
            break
        f = line.rstrip("\n").split("\t")
        if len(f) <= col["aaSubstitutions"]:
            drop["short row"] += 1; continue
        if args.host and f[col["host"]].strip().lower() != args.host.lower():
            drop[f"host != {args.host}"] += 1; continue
        d = f[col["date"]]
        if len(d) < 7 or "?" in d[:7]:              # need a real year-month
            drop["no year-month"] += 1; continue
        lin = f[col["Nextclade_pango"]].strip()
        if not lin or lin == "?":
            lin = f[col["pango_lineage"]].strip()
        if not lin or lin == "?":
            drop["no lineage"] += 1; continue
        key = (d[:7], lin)
        cnt[key] += 1.0
        m = mut[key]
        for tok in f[col["aaSubstitutions"]].split(","):
            g = SUB.match(tok)
            if g is None:
                continue
            j = sub.get((g.group(2), g.group(3)))
            if j is not None:
                m[j] += 1.0
        dl = f[col["deletions"]]
        if dl:
            for tok in dl.split(","):
                j = dele.get(tok.strip())
                if j is not None:
                    m[j] += 1.0
        kept += 1
        if n % 1000000 == 0:
            print(f"  {n:,} rows read, {kept:,} usable, "
                  f"{len(cnt):,} lineage-months", flush=True)
    p.stdout.close(); p.wait()
    print(f"\n{n:,} rows read, {kept:,} usable, {len(cnt):,} lineage-months")
    if drop:
        print("  discarded:")
        for k, v in sorted(drop.items(), key=lambda kv: -kv[1]):
            print(f"    {v:>12,}  {k}")
    if not cnt:
        sys.exit("nothing usable -- check the host value and the date column")

    keep = [k for k in cnt if cnt[k] >= args.min_seqs]
    if not keep:
        top = sorted(cnt.values(), reverse=True)[:5]
        sys.exit(f"no lineage-month reaches --min-seqs {args.min_seqs}; "
                 f"the largest are {[int(x) for x in top]}. Lower it.")
    months = sorted({k[0] for k in keep})
    lins = sorted({k[1] for k in keep})
    print(f"{len(months)} months {months[0]}..{months[-1]}, "
          f"{len(lins):,} lineages with a month of at least "
          f"{args.min_seqs} sequences")

    mi = {m: i for i, m in enumerate(months)}
    li = {l: i for i, l in enumerate(lins)}
    Npt = np.zeros((len(months), len(lins)))
    TH = np.zeros((len(months), len(lins), V), dtype=np.float32)
    for (mo, ln) in keep:
        i, j = mi[mo], li[ln]
        c = cnt[(mo, ln)]
        Npt[i, j] = c
        for node, v in mut[(mo, ln)].items():
            TH[i, j, node] = v / c
    Pi = Npt / np.maximum(Npt.sum(1, keepdims=True), EPS)

    if args.out:
        np.savez_compressed(args.out, Pi=Pi, N=Npt, theta=TH,
                            months=np.array(months), lineages=np.array(lins))
        print(f"wrote {args.out}")

    # ---- the fitness regression, now with identity fixed by construction
    X, y, w, tt = [], [], [], []
    for t in range(len(months) - 1):
        for j in range(len(lins)):
            a, b = Pi[t, j], Pi[t + 1, j]
            if Npt[t, j] < args.min_seqs or Npt[t + 1, j] < args.min_seqs:
                continue
            X.append(TH[t, j]); y.append(np.log(b + EPS) - np.log(a + EPS))
            w.append(a); tt.append(t)
    if len(X) < 50:
        sys.exit("too few lineage-month transitions; lower --min-seqs")
    X = np.stack(X).astype(float); y = np.array(y)
    w = np.array(w); tt = np.array(tt)
    yr = np.array([int(months[t][:4]) for t in tt])
    print(f"\n{len(X):,} lineage-month transitions across "
          f"{len(set(yr.tolist()))} years")

    def ridge(Xa, ya, wa, lam):
        sw = np.sqrt(wa)[:, None]
        mx = (wa[:, None] * Xa).sum(0) / wa.sum()
        my = float((wa * ya).sum() / wa.sum())
        Xc = (Xa - mx[None, :]) * sw; yc = (ya - my) * sw.ravel()
        b = np.linalg.solve(Xc.T @ Xc + lam * np.eye(Xa.shape[1]), Xc.T @ yc)
        return b, my, mx

    def r2(ya, pa, wa):
        my = float((wa * ya).sum() / wa.sum())
        return 1 - float((wa * (ya - pa) ** 2).sum()) / \
            max(float((wa * (ya - my) ** 2).sum()), EPS)

    yrs = sorted(set(yr.tolist()))
    print(f"\n  LEAVE-ONE-YEAR-OUT: is lineage growth a function of lineage "
          f"contents?\n")
    print(f"  {'lambda':>9}" + "".join(f"{y_:>9}" for y_ in yrs)
          + f"{'mean':>9}")
    best = None
    for lam in (float(x) for x in args.lams.split(",")):
        outs = []
        for y_ in yrs:
            tr = yr != y_; te = ~tr
            if te.sum() < 10 or tr.sum() < 50:
                outs.append(np.nan); continue
            b, my, mx = ridge(X[tr], y[tr], w[tr], lam)
            outs.append(r2(y[te], my + (X[te] - mx[None, :]) @ b, w[te]))
        m = float(np.nanmean(outs))
        print(f"  {lam:>9.0f}" + "".join(f"{o:>9.3f}" for o in outs)
              + f"{m:>9.3f}")
        if best is None or m > best[0]:
            best = (m, lam)

    m, lam = best
    rng = np.random.default_rng(0)
    ctrl = []
    for _ in range(20):
        yp = rng.permutation(y)
        outs = []
        for y_ in yrs:
            tr = yr != y_; te = ~tr
            if te.sum() < 10 or tr.sum() < 50: continue
            b, my, mx = ridge(X[tr], yp[tr], w[tr], lam)
            outs.append(r2(yp[te], my + (X[te] - mx[None, :]) @ b, w[te]))
        if outs: ctrl.append(float(np.mean(outs)))
    print(f"\n  best mean {m:+.3f} at lambda {lam:.0f}   "
          f"shuffled control: mean {np.mean(ctrl):+.3f}, max {np.max(ctrl):+.3f}")

    b, my, mx = ridge(X, y, w, lam)
    o = np.argsort(-b)
    print(f"\n  mutations with the largest fitted effect on lineage growth")
    print(f"    {'rank':>5}{'mutation':>16}{'f':>10}      "
          f"{'rank':>5}{'mutation':>16}{'f':>10}")
    for r in range(15):
        i, j = int(o[r]), int(o[-(r + 1)])
        print(f"    {r+1:>5}{names.get(i, str(i)):>16}{b[i]:>10.4f}      "
              f"{r+1:>5}{names.get(j, str(j)):>16}{b[j]:>10.4f}")
    print("""
  Identity here is external and exact: a lineage in one month is the same
  lineage in the next, with no matching step and no fitting. So this removes
  the one explanation the fitted-component version could not rule out.

  clearly above the shuffled control
      -> lineage growth IS a function of lineage contents, and the earlier
         nulls were caused by components that did not persist across months.
         The problem then is correspondence, not biology.
  at or below the shuffled control
      -> growth is not predictable from composition even with perfect
         identity. The nulls mean what they appeared to mean, and no amount
         of better clustering will change that.
""")


if __name__ == "__main__":
    sys.exit(main())
