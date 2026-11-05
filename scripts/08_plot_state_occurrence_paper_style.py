#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Paper-style dynamic state occurrence trajectories
=================================================
模仿参考文献 Figure 2 的视觉形式：

- 2 dynamic states × 3 groups (HC/M0/M1)
- 每个被试：浅灰色 rolling state-occurrence trajectory
- 组平均：蓝色粗线
- 深灰色panel背景
- 标题显示该组该state的平均 fractional occupancy (%)

说明：
每个subject原始state sequence为0/1指标。
为了得到类似原文的平滑个体轨迹，对binary sequence做30-window
centered rolling mean，仅用于可视化，不参与统计检验。

Window50, step=1 TR, TR=1 s.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 1. Paths
# ============================================================
ROOT = Path(
    os.environ.get(
        "WM_DYNAMIC_ROOT",
        Path(__file__).resolve().parent.parent / "data",
    )
)
BASE = ROOT/"StateSpecific_27Edge_Median_Window50"
SEQ_FILE = BASE/"State_sequences/All_subject_state_sequences.csv"
FIG_DIR = BASE/"Figures"
FIG_DIR.mkdir(parents=True,exist_ok=True)

OUT_PNG = FIG_DIR/"Figure_Window50_StateOccurrence_PaperStyle.png"
OUT_PDF = FIG_DIR/"Figure_Window50_StateOccurrence_PaperStyle.pdf"
OUT_CSV = FIG_DIR/"Figure_Window50_StateOccurrence_PaperStyle_groupmeans.csv"

ROLLING_WINDOWS = 30
GROUPS = ["HC","M0","M1"]
STATES = [1,2]

if not SEQ_FILE.exists():
    raise FileNotFoundError(f"Missing:\n{SEQ_FILE}")

# ============================================================
# 2. Load state sequences
# ============================================================
df = pd.read_csv(SEQ_FILE)

required = ["Window","Start_sec","State","Subject","Group"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise RuntimeError(f"Missing columns: {missing}")

df["Window"] = pd.to_numeric(df["Window"],errors="coerce")
df["Start_sec"] = pd.to_numeric(df["Start_sec"],errors="coerce")
df["State"] = pd.to_numeric(df["State"],errors="coerce")

print("="*70)
print("STATE OCCURRENCE PAPER-STYLE FIGURE")
print("="*70)

print("\nSubjects:")
print(df.groupby("Group")["Subject"].nunique())

# QC: 每个subject应该541 windows
nw = df.groupby(["Group","Subject"]).size()
if not (nw==541).all():
    print("\nWARNING: Some subjects do not have 541 windows:")
    print(nw[nw!=541])

# ============================================================
# 3. Build subject-specific rolling trajectories
# ============================================================
trajectory_rows = []

for group in GROUPS:
    dg = df[df["Group"]==group].copy()

    for subject,dsub in dg.groupby("Subject"):
        dsub = dsub.sort_values("Window").copy()

        for state in STATES:
            indicator = (dsub["State"].to_numpy()==state).astype(float)

            rolling = (
                pd.Series(indicator)
                .rolling(
                    window=ROLLING_WINDOWS,
                    center=True,
                    min_periods=1
                )
                .mean()
                .to_numpy()
            )

            for w,t,y in zip(
                dsub["Window"],
                dsub["Start_sec"],
                rolling
            ):
                trajectory_rows.append({
                    "Group":group,
                    "Subject":subject,
                    "State":state,
                    "Window":w,
                    "Start_sec":t,
                    "RollingOccurrence":y
                })

traj = pd.DataFrame(trajectory_rows)

# ============================================================
# 4. Group mean trajectories
# ============================================================
group_mean = (
    traj.groupby(
        ["Group","State","Window","Start_sec"],
        as_index=False
    )["RollingOccurrence"]
    .agg(["mean","std","count"])
    .reset_index()
)

group_mean["sem"] = group_mean["std"]/np.sqrt(group_mean["count"])
group_mean["ci95_low"] = group_mean["mean"]-1.96*group_mean["sem"]
group_mean["ci95_high"] = group_mean["mean"]+1.96*group_mean["sem"]

group_mean.to_csv(OUT_CSV,index=False)

# ============================================================
# 5. Overall fractional occupancy for panel titles
# ============================================================
fo = (
    df.assign(
        State1=(df["State"]==1).astype(float),
        State2=(df["State"]==2).astype(float)
    )
    .groupby(["Group","Subject"])[["State1","State2"]]
    .mean()
    .reset_index()
)

fo_summary = (
    fo.groupby("Group")[["State1","State2"]]
    .mean()*100
)

print("\nMean fractional occupancy (%):")
print(fo_summary.round(3))

# Hard QC against established results
expected = {
    ("HC","State1"):88.5397,
    ("HC","State2"):11.4603,
    ("M0","State1"):68.8672,
    ("M0","State2"):31.1328,
    ("M1","State1"):72.3396,
    ("M1","State2"):27.6604
}

for (g,s),exp in expected.items():
    got = float(fo_summary.loc[g,s])
    if abs(got-exp)>0.25:
        raise RuntimeError(
            f"FO QC failed: {g} {s} = {got:.3f}%, "
            f"expected approximately {exp:.3f}%"
        )

print("\nFO QC: PASS")

# ============================================================
# 6. Plot style: mimic reference Figure 2
# ============================================================
plt.rcParams.update({
    "font.family":"Arial",
    "font.size":9,
    "axes.titlesize":10,
    "axes.labelsize":9,
    "xtick.labelsize":8,
    "ytick.labelsize":8
})

fig,axes = plt.subplots(
    2,3,
    figsize=(12.5,7.0),
    sharex=True,
    sharey=True
)

panel_face = "#3b3b3b"
individual_color = "#dddddd"
mean_color = "#5060ff"

for r,state in enumerate(STATES):
    for c,group in enumerate(GROUPS):

        ax = axes[r,c]
        ax.set_facecolor(panel_face)

        # -----------------------------------------
        # Individual subject trajectories
        # -----------------------------------------
        ds = traj[
            (traj["Group"]==group)&
            (traj["State"]==state)
        ]

        for subject,dsub in ds.groupby("Subject"):
            dsub = dsub.sort_values("Start_sec")

            ax.plot(
                dsub["Start_sec"],
                dsub["RollingOccurrence"]*100,
                color=individual_color,
                alpha=0.38,
                linewidth=0.55,
                zorder=1
            )

        # -----------------------------------------
        # Group mean trajectory
        # -----------------------------------------
        gm = group_mean[
            (group_mean["Group"]==group)&
            (group_mean["State"]==state)
        ].sort_values("Start_sec")

        ax.plot(
            gm["Start_sec"],
            gm["mean"]*100,
            color=mean_color,
            linewidth=2.0,
            zorder=3
        )

        # State percentage in title
        pct = fo_summary.loc[group,f"State{state}"]

        ax.set_title(
            f"State {state} — {group} ({pct:.1f}%)",
            fontweight="bold",
            pad=5
        )

        ax.set_xlim(
            df["Start_sec"].min(),
            df["Start_sec"].max()
        )

        ax.set_ylim(0,100)

        ax.grid(False)

        # White-looking plot internals similar to source figure
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_color("#555555")

        # State labels on left
        if c==0:
            ax.set_ylabel(
                "Rolling state occurrence (%)"
            )

        if r==1:
            ax.set_xlabel(
                "Time (s)"
            )

# ============================================================
# 7. Figure annotations
# ============================================================
fig.text(
    0.008,0.73,
    "Low-connectivity state",
    rotation=90,
    va="center",
    ha="center",
    fontsize=10,
    fontweight="bold"
)

fig.text(
    0.008,0.29,
    "High-connectivity state",
    rotation=90,
    va="center",
    ha="center",
    fontsize=10,
    fontweight="bold"
)

fig.suptitle(
    "State occurrences over time",
    fontsize=14,
    fontweight="bold",
    y=0.995
)

fig.text(
    0.5,0.012,
    "Thin lines: individual participants; blue line: group mean. "
    "30-window rolling mean shown for visualization.",
    ha="center",
    va="bottom",
    fontsize=8
)

plt.tight_layout(
    rect=[0.035,0.045,1,0.96],
    h_pad=1.8,
    w_pad=1.2
)

fig.savefig(
    OUT_PNG,
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

fig.savefig(
    OUT_PDF,
    bbox_inches="tight",
    facecolor="white"
)

plt.close(fig)

# ============================================================
# 8. Also create a compact 2-panel group-comparison version
# ============================================================
OUT2_PNG = FIG_DIR/"Figure_Window50_StateOccurrence_GroupComparison.png"
OUT2_PDF = FIG_DIR/"Figure_Window50_StateOccurrence_GroupComparison.pdf"

group_colors = {
    "HC":"#222222",
    "M0":"#e41a1c",
    "M1":"#377eb8"
}

fig,axes = plt.subplots(
    1,2,
    figsize=(10.5,4.2),
    sharex=True,
    sharey=True
)

for ax,state in zip(axes,[1,2]):
    ax.set_facecolor(panel_face)

    for group in GROUPS:
        gm = group_mean[
            (group_mean["Group"]==group)&
            (group_mean["State"]==state)
        ].sort_values("Start_sec")

        ax.plot(
            gm["Start_sec"],
            gm["mean"]*100,
            linewidth=2.0,
            color=group_colors[group],
            label=group
        )

    ax.set_title(
        f"State {state} "
        f"({'Low' if state==1 else 'High'} connectivity)",
        fontweight="bold"
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylim(0,100)
    ax.grid(False)

axes[0].set_ylabel("State occurrence (%)")
axes[0].legend(frameon=False,loc="best")

fig.suptitle(
    "State occurrences over time",
    fontsize=14,
    fontweight="bold"
)

plt.tight_layout(rect=[0,0,1,0.94])

fig.savefig(
    OUT2_PNG,
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    OUT2_PDF,
    bbox_inches="tight"
)

plt.close(fig)

print("\n"+"="*70)
print("FINISHED")
print("="*70)
print("Paper-style figure:")
print(OUT_PNG)
print("\nCompact comparison figure:")
print(OUT2_PNG)
print("\nGroup-mean trajectory data:")
print(OUT_CSV)
