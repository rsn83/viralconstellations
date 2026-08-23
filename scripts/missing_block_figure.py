import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

INK, MUTED = "#1a1a1a", "#8a8a8a"
LIVE, DEAD, BAD, NEW = "#3d5a80", "#dcdcdc", "#c1414a", "#e07a35"

months  = ["21-12","22-01","22-02","22-03","22-04","22-05","22-06"]
fit_b1  = [-4.72,-4.28,-3.89,-3.66,-3.83,-7.06,-13.61]

fig = plt.figure(figsize=(14, 10.0))
gs  = fig.add_gridspec(2, 2, hspace=.80, wspace=.30,
                       left=.075, right=.955, top=.795, bottom=.175)

fig.text(.075,.965,"The model cannot create a new latent group",
         fontsize=19,fontweight="bold",color=INK,va="top")
fig.text(.075,.912,"One missing mechanism produces both the forecasting failure and the appearance failure.",
         fontsize=11.5,color=MUTED,va="top")
fig.text(.075,.878,"blk = latent group: a set of sequences sharing a mutation-set fingerprint. Groups here correspond to lineages.",
         fontsize=9.6,color=MUTED,va="top",style="italic")
for _x,_c,_t in [(.075,LIVE,"the model: groups, fingerprints, and the do-nothing forecast"),
                 (.470,BAD, "A's forecast"),
                 (.610,NEW, "BA.5 \u2014 the lineage with no group")]:
    fig.patches.append(plt.Rectangle((_x,.838),.011,.016,transform=fig.transFigure,
                                     facecolor=_c,edgecolor="none",zorder=5))
    fig.text(_x+.017,.846,_t,fontsize=9.2,color=INK,va="center")

# ---------------- (a) ----------------
ax = fig.add_subplot(gs[0,0]); ax.set_xlim(0,10); ax.set_ylim(-0.6,7); ax.axis("off")
ax.set_title("(a)   A closed world of 8 latent groups, fixed before June",
             fontsize=12.5,fontweight="bold",color=INK,loc="left",pad=14)
for (x,y),lb in zip([(1.5,5.4),(3.7,5.4),(5.9,5.4),(8.1,5.4),(2.6,3.2),(4.8,3.2)],
                    ["blk5\nDelta","blk2\nBA.1","blk3\nBA.1b","blk1\nBA.2","blk4\nAlpha","blk6"]):
    ax.add_patch(Circle((x,y),.78,facecolor=LIVE,edgecolor="none"))
    ax.text(x,y,lb,ha="center",va="center",fontsize=7.6,color="white",
            fontweight="bold",linespacing=1.3)
for x,lb in [(7.0,"blk0"),(9.0,"blk7")]:
    ax.add_patch(Circle((x,3.2),.78,facecolor=DEAD,edgecolor=MUTED,
                        linestyle=(0,(2,2)),lw=1.1))
    ax.text(x,3.2,lb+"\nunused",ha="center",va="center",fontsize=7.6,
            color=MUTED,linespacing=1.3)
ax.add_patch(Circle((8.1,1.15),.78,facecolor="none",edgecolor=NEW,
                    linestyle=(0,(3,2)),lw=2.2))
ax.text(8.1,1.15,"BA.5",ha="center",va="center",fontsize=8.6,color=NEW,fontweight="bold")
ax.annotate("",xy=(8.1,2.0),xytext=(8.1,4.55),
            arrowprops=dict(arrowstyle="-|>",color=NEW,lw=2.0,linestyle=(0,(3,2))))
ax.text(6.9,1.15,"no group\nexists",fontsize=8.8,color=NEW,fontweight="bold",
        ha="right",va="center",linespacing=1.3)
ax.text(0,-.16,"blk0 and blk7 got no sequences in any month, so EM never updated them: every\n"
                "mutation still sits at its starting value of 0.50. Two spare groups were available\n"
                "for BA.5 and stayed empty \u2014 a group needs training data to learn a fingerprint,\n"
                "and BA.5 was not in the training months. So raising K would not have helped.",
        fontsize=8.8,color=MUTED,transform=ax.transAxes,linespacing=1.55)

# ---------------- (b) ----------------
ax = fig.add_subplot(gs[0,1])
ax.set_title("(b)   Sequences put in blk1 stop matching blk1",
             fontsize=12.5,fontweight="bold",color=INK,loc="left",pad=14)
x = np.arange(len(months))
ax.axhspan(-5.0,-3.3,color=LIVE,alpha=.10,zorder=0)
ax.plot(x,fit_b1,"-o",color=LIVE,lw=2.2,ms=6,zorder=3)
ax.plot(x[-1],fit_b1[-1],"o",color=NEW,ms=12,zorder=4)
ax.annotate("",xy=(5.55,-13.61),xytext=(5.55,-4.85),
            arrowprops=dict(arrowstyle="<|-|>",color=NEW,lw=1.7))
ax.text(5.38,-8.4,"8.76 worse\n= exactly what\n2 unexpected\nmutations cost",fontsize=9.6,color=NEW,fontweight="bold",
        ha="right",va="center",linespacing=1.35,
        bbox=dict(boxstyle="round,pad=0.25",fc="white",ec="none"))
ax.text(.03,.90,"clean BA.2",transform=ax.transAxes,fontsize=9.2,
        color=LIVE,fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(months,fontsize=9)
ax.set_ylim(-15.5,-2.4); ax.set_ylabel("how well blk1 explains its own sequences\n(log-likelihood; higher is better)",
              fontsize=9.4,color=INK,linespacing=1.5)
ax.tick_params(labelsize=9,colors=MUTED)
for s in ("top","right"): ax.spines[s].set_visible(False)
for s in ("left","bottom"): ax.spines[s].set_color(MUTED)
ax.text(0,-.44,"The y-axis is how well blk1's fingerprint explains the sequences it was given.\n"
                "A mutation blk1 calls unlikely, but which is present, costs 4.4 points.\n"
                "Two of them cost 8.8. The observed drop is 8.76 \u2014 exactly the two mutations\n"
                "in panel (d), and nothing else. (5.6 sd below training, so not noise.)",
        transform=ax.transAxes,fontsize=8.8,color=MUTED,linespacing=1.55)

# ---------------- (c) ----------------
ax = fig.add_subplot(gs[1,0])
ax.set_title("(c)   A moves sequences that never moved",
             fontsize=12.5,fontweight="bold",color=INK,loc="left",pad=14)
xs=np.arange(4); w=.36
ax.bar(xs-w/2,[.987,.003,.001,.009],w,color=LIVE,label="May's composition = what June actually was",zorder=3)
ax.bar(xs+w/2,[.813,.125,.046,.016],w,color=BAD, label="A's June forecast",zorder=3)
ax.set_xticks(xs); ax.set_xticklabels(["blk1\n(BA.2)","blk2\n(BA.1)","blk3","other"],fontsize=9)
ax.set_ylim(0,1.18); ax.set_ylabel("fraction of that month's sequences",fontsize=10,color=INK)
ax.tick_params(labelsize=9,colors=MUTED)
ax.legend(fontsize=8.8,frameon=False,loc="upper center",bbox_to_anchor=(.60,1.04))
for s in ("top","right"): ax.spines[s].set_visible(False)
for s in ("left","bottom"): ax.spines[s].set_color(MUTED)
ax.annotate("12.5% of sequences sent back\nto BA.1, already extinct",
            xy=(1.19,.128),xytext=(1.62,.50),fontsize=9.2,color=BAD,
            fontweight="bold",linespacing=1.35,
            arrowprops=dict(arrowstyle="-|>",color=BAD,lw=1.5))
ax.text(0,-.44,"As far as the model can see nothing moved, so copying May is right.\n"
                "Scoring all 240,709 June sequences \u2014 how well each forecast explains them\n"
                "(higher is better):     copy May  \u221213.34        via A  \u221214.00",
        transform=ax.transAxes,fontsize=9,color=MUTED,linespacing=1.55)

# ---------------- (d) ----------------
ax = fig.add_subplot(gs[1,1])
ax.set_title("(d)   June's blk1 sequences do not match blk1's fingerprint",
             fontsize=12.5,fontweight="bold",color=INK,loc="left",pad=14)
mut=["486V","452R","704L","452Q","3G","76I"]
th =[.012,.012,.081,.072,.003,.001]; obs=[.538,.529,.260,.246,.061,.044]
y=np.arange(len(mut))[::-1]
ax.barh(y,obs,.55,color=NEW,zorder=3,label="observed in June's blk1 sequences")
ax.barh(y,th ,.55,color=LIVE,zorder=4,label="blk1 fingerprint (learned in training)")
ax.set_yticks(y); ax.set_yticklabels(mut,fontsize=9.5,fontweight="bold")
ax.set_xlim(0,.80); ax.set_xlabel("frequency",fontsize=10,color=INK)
ax.tick_params(labelsize=9,colors=MUTED)
ax.legend(fontsize=9.2,frameon=False,loc="lower right")
for s in ("top","right"): ax.spines[s].set_visible(False)
for s in ("left","bottom"): ax.spines[s].set_color(MUTED)
ax.add_patch(plt.Rectangle((0,y[1]-.40),.575,1.80,fill=False,edgecolor=NEW,
                           lw=1.9,linestyle=(0,(3,2)),zorder=6))
ax.text(.60,y[1]+.50,"L452R + F486V = BA.5\n\nblk1 calls each 1% likely.\nEach is present in 53%.\nThat is the 8.76 in (b).",
        fontsize=9.2,color=NEW,fontweight="bold",va="center",linespacing=1.45)
ax.text(0,-.32,"The fingerprint is fixed and cannot change; what changed is which sequences\nget filed under blk1. That same pair is the mutation set the model cannot propose.",
        transform=ax.transAxes,fontsize=9,color=MUTED,linespacing=1.5)

fig.text(.075,.040,"no way to create a latent group   →   A misallocates sequences (forecasting)   +   no candidate mutation set exists to score (appearance)",
         fontsize=11,color=INK,fontweight="bold")
fig.savefig("/home/claude/missing_block.png",dpi=190,facecolor="white")
print("saved")
