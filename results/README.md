116, 117: chain-vs-slope on frozen blocks. The slope rows REFIT beta/gamma
per series and so discard hierarchical shrinkage on gamma. The chain's
apparent win is against a weakened slope. Superseded by 118_chain2 /
118_slope, which compare inside the real fitting loop.

118: chain integrated into fitting. Chain loses to slope at every horizon.
120: Beta prior sweep. Beta(5,5) is the only thing that has moved h=6 (+3.19).
121: at h=6, 75.6% new combination, 3.8% new mutation, median edit distance 3.
122: responsibility entropy 0.064 of 3.871 -- collapse is real, but a
     background component buys almost nothing.

124/125: copying (Li-Stephens) scored on a fitted model. s=0 reproduces the
     mixture exactly. Best cell lam=0, s=0.005: h=6 all -28.848 vs -30.489.
     Tree-distance switching (lam>0) LOSES -- useful switches go to blocks the
     tree calls distant.
126: refitting WITH copying. Inconclusive, not refuted: gradient step size was
     ad hoc and births were disabled, so K stayed at 48 and blocks could not
     reorganise.
128/129: per-mutation fitness f. Null. Pooled over 5 windows, leave-one-window-
     out R2 -0.039 vs shuffled control -0.045. f learns which lineage was
     winning, not why; it does not transfer across sweeps.
130: background effect on mutation appearance. Null, and properly powered
     (782-1504 strata per window). Mean gain over exposure alone -0.004.
131/134/135: low-rank emission. Beats full rank on the TRAINING objective at
     every rank (-3.47 at rank 4 vs -3.76 reference), which means 110's M-step
     is under-converged. But 8-13 nats WORSE held-out everywhere. Confounded:
     the low-rank fit has no hierarchical prior and no birth-death, so this
     compares regularisation regimes, not ranks.
132: eps* = (novel fraction)(mean edits)/V. Predicted Beta a* 4.70 (BQ.1) and
     ~3.1 (Omicron); measured optimum 5 on both.
133: Omicron-window Beta sweep. Optimum a=5, -78.0 at h=3 vs -145.3 unsmoothed.
     Still 48 nats behind flat-15, so sharpness is most of the failure at K=48
     but not all of it.
