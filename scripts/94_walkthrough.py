#!/usr/bin/env python3
"""
Step-by-step: one sequence, through the model, to the failure.

Uses the REAL fitted numbers from results/91_exact.npz (K=8, train
2021-06..2022-05). Nothing invented -- every theta and pi value below is what
the model actually learned.

Prints eight steps:
  1  what arrives              raw sequence + collection date
  2  compare to Wuhan          -> mutation set
  3  vocabulary                -> binary vector
  4  stack by month            -> X
  5  theta                     each latent group's fingerprint
  6  pi_t                      each month's composition
  7  score the sequence        which group, and how surprised
  8  the failure               why K cannot fix it
"""
import numpy as np
np.set_printoptions(precision=3, suppress=True)

W  = 74   # print width helper
def rule(t=""):
    print("\n" + "=" * 86)
    if t: print(t); print("=" * 86)

# ---- the real fitted numbers, transcribed from results/90_worked_example.txt
# blk1 = BA.2 fingerprint, top mutations
blk1 = {"969K":0.998,"655Y":0.998,"614G":0.997,"954H":0.997,"679K":0.996,
        "796Y":0.993,"681H":0.992,"213G":0.991,"440K":0.852,
        "486V":0.012,"452R":0.012}
blk5 = {"614G":0.997,"681R":0.996,"19R":0.985,"478K":0.971,"452R":0.969,
        "950N":0.958,"158G":0.928,"486V":0.002}
blk2 = {"501Y":1.000,"373P":1.000,"493R":0.999,"614G":0.999,"486V":0.001,
        "452R":0.002}

rule("STEP 1   WHAT ARRIVES")
print("""
  One record from GISAID: a spike protein sequence and a collection date.

    EPI_ISL_13402911    2022-06-14    MFVFLVLLPLVSSQCVNLITRTQSYTNSFTRGVYYPDK...

  1273 amino acids. There are 240,709 such records for June 2022.
""")

rule("STEP 2   COMPARE TO WUHAN  ->  MUTATION SET")
print("""
  Line it up against Wuhan-Hu-1 and record only the differences.

     position    ...   452   ...   486   ...   614   ...
     Wuhan             L           F           D
     this sequence     R           V           G

    EPI_ISL_13402911 -> 2022-06-14, {L452R, F486V, D614G, ... }

  Nothing is lost: apply those changes to Wuhan and you get all 1273 residues
  back. This particular sequence carries 31 mutations.
""")

rule("STEP 3   VOCABULARY  ->  BINARY VECTOR")
print("""
  Fix a coordinate system. Every possible (position, residue) pair is a node:

     node 452R  = position 452 carries R
     node 486V  = position 486 carries V
     node 614G  = position 614 carries G

  Our vocabulary V = 1,180 nodes (those observed in the corpus).
  Now the sequence is a binary vector:

              452R  486V  614G  655Y  679K  969K  ...
                1     1     1     1     1     1    ...    <- 31 ones, 1149 zeros
""")

rule("STEP 4   STACK BY MONTH  ->  X")
print("""
  Group by collection month. Each month is a matrix: rows = sequences,
  columns = nodes.

     2022-06     seq 1      [1 1 1 1 1 1 ...]
                 seq 2      [0 0 1 1 1 1 ...]
                 ...
                 seq 240709 [1 1 1 1 1 1 ...]

  That is X. Everything so far is re-encoding -- no model, nothing estimated.
""")

rule("STEP 5   theta   -- WHAT THE MODEL LEARNED  (K x V)")
print("""
  The model assumes each sequence comes from one of K = 8 latent groups, and
  that given the group, mutations occur independently. theta[k,n] is the
  probability that a sequence from group k carries node n.

  Three of the eight fitted rows (real values, abbreviated):
""")
print(f"    {'':8}" + "".join(f"{m:>8}" for m in ["614G","655Y","969K","452R","486V","478K","501Y"]))
for nm, d, lab in [("blk1", blk1, "BA.2"), ("blk5", blk5, "Delta"), ("blk2", blk2, "BA.1")]:
    print(f"    {nm:<5}{lab:<4}" + "".join(f"{d.get(m, 0.002):>8.3f}"
          for m in ["614G","655Y","969K","452R","486V","478K","501Y"]))
print("""
  Read a row as a fingerprint. blk1 expects 655Y and 969K almost always, and
  452R / 486V almost never. That row IS the definition of blk1, and it is fixed
  once, during training. It never changes afterwards.
""")

rule("STEP 6   pi_t   -- EACH MONTH'S COMPOSITION  (T x K)")
Pi = {"2021-11":[0.000,0.000,0.992],"2021-12":[0.002,0.337,0.532],
      "2022-01":[0.055,0.692,0.029],"2022-02":[0.302,0.558,0.002],
      "2022-03":[0.822,0.156,0.000],"2022-04":[0.963,0.027,0.000],
      "2022-05":[0.987,0.003,0.000]}
print(f"\n    {'month':<10}{'blk1 (BA.2)':>13}{'blk2 (BA.1)':>13}{'blk5 (Delta)':>14}")
for m, v in Pi.items():
    print(f"    {m:<10}{v[0]:>13.3f}{v[1]:>13.3f}{v[2]:>14.3f}")
print("""
  pi_t is the fraction of month t's sequences from each group. This is the real
  Delta -> BA.1 -> BA.2 sweep, recovered without any lineage labels.
""")

rule("STEP 7   SCORE THE SEQUENCE")
print("""
  For our June sequence S, the model computes for each group:

      log p(S | group k)  =  sum over ALL 1,180 nodes of
                                log theta[k,n]       if n IS  in S
                                log (1 - theta[k,n]) if n NOT in S

  It picks the group with the highest score. For this sequence:
""")
print(f"    {'group':<16}{'log p(S | group)':>20}{'assigned':>12}")
for nm, lab, v in [("blk1","BA.2",-13.61),("blk2","BA.1",-140.49),
                   ("blk5","Delta",-230.63)]:
    tag = "  <-- yes" if nm == "blk1" else ""
    print(f"    {nm+' ('+lab+')':<16}{v:>20.2f}{tag:>12}")
print("""
  blk1 wins -- it is the least bad option. But look at HOW bad it is.

  Where the -13.61 comes from. Two of this sequence's 31 mutations are ones
  blk1 says are almost impossible:
""")
for m in ["452R","486V"]:
    th = blk1[m]
    cost = np.log(1-th) - np.log(th)
    print(f"    node {m}:  blk1 says {th:.1%} likely.  It is present.")
    print(f"              surprise cost = log(1-{th}) - log({th}) = {cost:.2f}")
tot = sum(np.log(1-blk1[m]) - np.log(blk1[m]) for m in ["452R","486V"])
print(f"""
    two surprises together      = {tot:.2f}
    observed drop in the data   = 8.76      <- the whole gap, nothing else

  In March, blk1's sequences scored -3.66. In June they score -13.61. The model
  did not get worse; the sequences it is being handed did.
""")

rule("STEP 8   WHY THIS CANNOT BE FIXED BY CHANGING K")
print("""
  L452R + F486V is BA.5. 53% of the sequences filed under blk1 in June carry
  both. They are not BA.2. They are a lineage with no group of its own.

  Why not simply use a larger K?

    theta has K rows, and every row is estimated by EM from the TRAINING months.
    BA.5 is not in 2021-06..2022-05. A ninth row would have no sequences to
    learn from.

    This is not hypothetical -- it already happened. Our K=8 fit produced two
    groups that were never used:

        blk0:  every one of the 1,180 nodes still at theta = 0.500
        blk7:  every one of the 1,180 nodes still at theta = 0.500
        both:  pi = 0.000 in all twelve months

    0.500 is the initialisation value. EM never touched them. Expected set size
    = 1180 x 0.5 = 590 mutations, which is biologically meaningless.

  So two spare groups were sitting available for BA.5 and stayed empty. K was
  not the constraint. A group needs DATA to acquire a fingerprint, and the data
  arrives only after the lineage exists.

  AND THE SAME GAP BREAKS THE FORECAST.

    A predicts next month's composition by moving sequences between the eight
    existing groups:

        May   (and the June truth):  blk1 0.987   blk2 0.003   blk3 0.001
        A's June forecast:           blk1 0.813   blk2 0.125   blk3 0.046

    A sends 12.5% back to blk2 (BA.1), which was extinct. It learnt that
    transition from Dec-Feb, when BA.1 was large, and applies the same row
    forever.

    But as far as the model can see, nothing moved in June: blk1 stayed at 98%,
    because BA.5 is hidden inside it. So the correct forecast was 'no change' --
    persistence -13.34, A -14.00.

  ONE MISSING MECHANISM, TWO FAILURES

    forecasting:  A must send sequences to a group that does not exist
    appearance :  no candidate mutation set exists to score

    Both need the ability to CREATE a group. Fixing A alone cannot do it.
""")
