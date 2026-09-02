#!/usr/bin/env python3
"""
177 -- WHY DO 171, 175 AND 176 REPORT DIFFERENT COVERAGE?

Three scripts, same data, same months, same MIN_COUNT, three answers to
"what fraction of new constellations are reachable from the population":

    171   0.511 - 0.541      radius-1
    175   0.653              radius-1
    176   0.376              radius-1     (0.401 with radius-2 added)

At most one of these is the number to quote. Every recall figure in the
project divides by one of them, so the denominator has to be settled before
any of those figures means anything.

The definitions differ in three places. This script computes coverage under
each variant on identical months and reports where the drop happens.

    A  distance-to-any-background     is min_{S in P} |S xor S'| <= 1 ?
                                      P = top-N backgrounds by mass
                                      (this is 175's definition)

    B  subset-and-size                exists S in P with S subset of S'
                                      and |S'| = |S| + 1 ?
                                      differs from A when S' is SMALLER than
                                      a background, or differs by a swap
                                      rather than a pure addition

    C  in-enumerated-candidate-list   is the exact (S, D) pair present in the
                                      built candidate list?
                                      (this is 176's definition; also 171's)
                                      differs from B when D is outside the
                                      vocabulary used to build candidates

Vocabulary matters: 171/176 build candidates over mutations seen in the
TRAINING window only. A new constellation containing a mutation that never
appeared in training cannot be produced, whatever the distance is.

The script also reports, for constellations that fail C but pass A, WHY they
failed -- which of the three filters removed them.

USAGE
    python scripts/177_coverage_reconcile.py \
        --events data/processed/events_v3.tsv \
        --ladder scripts/171_ladder.py \
        --train-window 12 --test-end 2025-02

GIT
    git add scripts/177_coverage_reconcile.py
    git commit -m "177: reconcile conflicting coverage definitions"
    git push
"""

import argparse
import importlib.util
from collections import defaultdict

import numpy as np


def load_ladder(path):
    spec = importlib.util.spec_from_file_location("ladder171", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="data/processed/events_v3.tsv")
    ap.add_argument("--ladder", default="scripts/171_ladder.py")
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--max-bg", type=int, default=300,
                    help="175 used 200, 171/176 used 300; swept below")
    ap.add_argument("--train-window", type=int, default=12)
    ap.add_argument("--test-end", default="2025-02")
    args = ap.parse_args()

    L = load_ladder(args.ladder)
    print("loading ...")
    monthly = L.load_events(args.events)
    pops = {m: L.population(monthly[m]) for m in sorted(monthly)}
    pops = {m: p for m, p in pops.items() if p}
    months = sorted(pops)

    tr_end = L.TRAIN_END[:7]
    all_train = [m for m in months if m <= tr_end]
    train_months = all_train[-args.train_window:] if args.train_window > 0 \
        else all_train
    test_months = [m for m in months if tr_end < m <= args.test_end]

    # two candidate vocabularies: full training history vs the window
    vocab_full = {i for m in all_train for S in pops[m] for i in S}
    vocab_win = {i for m in train_months for S in pops[m] for i in S}
    print(f"  train window {len(train_months)}m | test {len(test_months)}m")
    print(f"  vocab: all training {len(vocab_full):,} | "
          f"window only {len(vocab_win):,}")

    seen_ever = set()
    for m in all_train:
        seen_ever |= set(pops[m])

    tot = defaultdict(int)
    n_new_tot = 0
    size_new, size_bg = [], []

    for m in test_months:
        t_ix = months.index(m)
        nxt = t_ix + args.horizon
        if nxt >= len(months):
            break
        pT, pN = pops[m], pops[months[nxt]]
        new = L.new_constellations(pT, pN, seen_ever)
        seen_ever |= set(pT)
        if not new:
            continue

        bgs = [S for S, _ in
               sorted(pT.items(), key=lambda kv: -kv[1])[:args.max_bg]]
        if not bgs:
            continue
        by_size = defaultdict(list)
        for S in bgs:
            by_size[len(S)].append(S)

        n_new_tot += len(new)
        size_new += [len(S) for S in new]
        size_bg += [len(S) for S in bgs]

        for Sn in new:
            # A: hamming distance <= 1 to any background
            dA = min(len(S ^ Sn) for S in bgs)
            if dA <= 1:
                tot["A_d1"] += 1
            if dA <= 2:
                tot["A_d2"] += 1

            # B: strict superset by exactly one element
            parentsB = [S for S in by_size.get(len(Sn) - 1, ()) if S < Sn]
            if parentsB:
                tot["B_add1"] += 1

            # C: B, and the added mutation is in the candidate vocabulary
            addsC_full = [next(iter(Sn - S)) for S in parentsB]
            if any(D in vocab_full for D in addsC_full):
                tot["C_vocab_full"] += 1
            if any(D in vocab_win for D in addsC_full):
                tot["C_vocab_win"] += 1

            # diagnosis for sets that pass A but fail B
            if dA <= 1 and not parentsB:
                # distance 1 without being a superset means either the
                # nearest background is LARGER (S' is a deletion of it) or
                # sizes are equal (a swap)
                near = min(bgs, key=lambda S: len(S ^ Sn))
                if len(Sn) < len(near):
                    tot["why_smaller_than_bg"] += 1
                elif len(Sn) == len(near):
                    tot["why_equal_size_swap"] += 1
                else:
                    tot["why_other"] += 1

    if n_new_tot == 0:
        print("  NO NEW CONSTELLATIONS FOUND")
        return

    def f(k):
        return tot[k] / n_new_tot

    print(f"\n  new constellations, all months   {n_new_tot:,}")
    print(f"  median |new set| {int(np.median(size_new))} | "
          f"median |background| {int(np.median(size_bg))}")
    print("\n  COVERAGE UNDER EACH DEFINITION")
    print(f"    A  hamming <= 1 to any background      {f('A_d1'):.3f}"
          "   <- 175's definition")
    print(f"    A  hamming <= 2                        {f('A_d2'):.3f}")
    print(f"    B  strict superset, exactly +1         {f('B_add1'):.3f}")
    print(f"    C  B and addition in full-train vocab  "
          f"{f('C_vocab_full'):.3f}")
    print(f"    C  B and addition in window vocab      "
          f"{f('C_vocab_win'):.3f}   <- 176's definition")

    print("\n  WHERE THE DROP HAPPENS")
    print(f"    A -> B  lost {f('A_d1') - f('B_add1'):+.3f}")
    print(f"      new set smaller than nearest background "
          f"{f('why_smaller_than_bg'):.3f}")
    print(f"      equal size (substitution, not addition) "
          f"{f('why_equal_size_swap'):.3f}")
    print(f"      other                                  {f('why_other'):.3f}")
    print(f"    B -> C  lost {f('B_add1') - f('C_vocab_win'):+.3f} "
          "(addition absent from candidate vocabulary)")

    print("\n  A counts sets reachable by ADDITION, DELETION or SUBSTITUTION.")
    print("  B and C count ADDITION only. If the A->B gap is large, the")
    print("  radius-1 figure quoted from 175 is not a ceiling for an")
    print("  addition-only generator, and 0.653 should not be used as one.")


if __name__ == "__main__":
    main()
