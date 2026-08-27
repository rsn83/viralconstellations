#!/usr/bin/env python
"""
121_novelty_anatomy.py -- what KIND of novelty is in the later months?

THE QUESTION
------------
At six months out, only about one in five sequences carries a set that was seen
in training, and the model scores those unseen sets worse than a table that
assigns one flat constant to everything it does not recognise. Two very
different things could be going on, and they need different fixes:

  new combination   every mutation in the set was seen in training, but this
                    particular combination was not. The model CAN reach these:
                    given a block, mutations are independent, so any subset of
                    the vocabulary gets a real probability. If it scores them
                    badly, that is a calibration problem -- blocks so sharp
                    that a set one mutation off is crushed by a (1-theta) term
                    near zero. Smoothing theta is the fix.

  new mutation      the set contains at least one mutation with no training
                    occurrences. The model CANNOT reach these at all: an unseen
                    mutation sits at the clip floor in every block, forever, so
                    no amount of smoothing helps. Only making theta a function
                    of the mutation -- position, residue change, domain,
                    exposure -- gives it a value by resemblance.

WHAT IS REPORTED
----------------
Per test month, sequences are split three ways -- seen set, unseen set built
only from known mutations, unseen set containing at least one novel mutation --
with the share of sequences in each and how far each novel set sits from its
nearest training set. Distance matters because a set one mutation away from
something common is a very different object from one twenty mutations away.

WHAT IT DECIDES
---------------
If almost all the novelty is new COMBINATIONS, the deficit is calibration and
the Beta prior sweep is the right line of attack.
If a large share involves new MUTATIONS, no smoothing will reach it and the
vocabulary itself is the boundary.
"""
import argparse, importlib.util, sys
import numpy as np

EPS = 1e-12


def load_engine(path):
    spec = importlib.util.spec_from_file_location("engine", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def nearest_distance(target, train_sets, cap=40000):
    """Smallest symmetric difference to any training set.

    Exhaustive over the training sets, capped: with tens of thousands of them
    and many test sets this is the expensive part, and the distribution is
    what matters rather than a per-sequence exact value."""
    best = None
    for s in train_sets[:cap]:
        d = len(target ^ s)
        if best is None or d < best:
            best = d
            if best == 0:
                break
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine",
                    default="scripts/110_hierarchical_birthdeath_v2.py")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--horizons", default="1,2,3,6")
    ap.add_argument("--min-count", type=int, default=3)
    ap.add_argument("--distance-sample", type=int, default=400,
                    help="novel sets sampled per month for the distance "
                         "distribution; the full set is used for the shares")
    args = ap.parse_args()

    E = load_engine(args.engine)
    names, V = E.load_names(args.vocab)
    tr = E.months_in_range(args.train)
    recs = [E.load_month(args.data_dir, ym) for ym in tr]
    if any(r is None for r in recs):
        sys.exit("missing training months")

    # what the model was actually built from: records surviving the min-count
    # filter, since anything below it was never available to be learned
    filt = [[(s_, c) for s_, c in r if c >= args.min_count] for r in recs]
    train_sets = list({s_ for r in filt for s_, _ in r})
    known = set()
    for r in filt:
        for s_, _ in r:
            known |= set(s_)
    print(f"train {tr[0]}..{tr[-1]}   min-count {args.min_count}")
    print(f"distinct training sets {len(train_sets):,}   "
          f"mutations ever seen {len(known):,} of {V:,} in the vocabulary\n")
    train_lookup = set(train_sets)
    rng = np.random.default_rng(0)

    print(f"  {'month':<9}{'h':>3}{'seen set':>11}{'new combo':>11}"
          f"{'new mutation':>14}{'median dist':>13}{'p90 dist':>10}")
    for h in (int(x) for x in args.horizons.split(",")):
        ym = E.ym_add(tr[-1], h)
        rec = E.load_month(args.data_dir, ym)
        if rec is None:
            print(f"  {ym:<9}{h:>3}  missing"); continue
        tot = 0.0; seen = 0.0; combo = 0.0; newmut = 0.0
        novel = []
        for s_, c in rec:
            c = float(c); tot += c
            if s_ in train_lookup:
                seen += c
            elif set(s_) <= known:
                combo += c; novel.append(s_)
            else:
                newmut += c; novel.append(s_)
        if not tot:
            continue
        if novel:
            pick = rng.choice(len(novel),
                              size=min(args.distance_sample, len(novel)),
                              replace=False)
            ds = [nearest_distance(set(novel[i]), [set(x) for x in train_sets])
                  for i in pick]
            med = float(np.median(ds)); p90 = float(np.percentile(ds, 90))
        else:
            med = p90 = 0.0
        print(f"  {ym:<9}{h:>3}{100*seen/tot:>10.1f}%{100*combo/tot:>10.1f}%"
              f"{100*newmut/tot:>13.1f}%{med:>13.0f}{p90:>10.0f}")

    print("""
  new combo    every mutation was seen in training, this arrangement was not.
               Reachable by the model in principle; if scored badly that is
               calibration, and smoothing theta is the lever.
  new mutation at least one mutation never seen in training. Unreachable at
               any setting of theta, because an unobserved mutation has no
               data behind it. Only a theta built from properties of the
               mutation reaches these.
  dist         symmetric difference to the nearest training set. A median of 1
               or 2 means the novelty is a small edit away from something
               familiar and a well-calibrated model should score it well. A
               median in the tens means the later months are a different
               population, and neither smoothing nor features will rescue it --
               the blocks themselves are missing.
""")


if __name__ == "__main__":
    sys.exit(main())
