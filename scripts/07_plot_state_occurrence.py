import os,glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===============================
# PATH
# ===============================

BASE = os.environ.get(
    "WM_DYNAMIC_ROOT",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
    ),
)

CENTROID_FILE=os.path.join(
    BASE,
    "StateSpecific_27Edge_Median_Window50",
    "Window50_K2_centroids_used_State1Low_State2High.csv"
)

OUT_BASE=os.path.join(
    BASE,
    "StateSpecific_27Edge_Median_Window50"
)

SEQ_DIR=os.path.join(
    OUT_BASE,
    "State_sequences"
)

FIG_DIR=os.path.join(
    OUT_BASE,
    "Figures"
)

os.makedirs(SEQ_DIR,exist_ok=True)
os.makedirs(FIG_DIR,exist_ok=True)


# ===============================
# dynamic files
# ===============================

GROUP_FILES={

"HC":
os.path.join(
BASE,
"HC_window_050_step_1",
"dynamic_edge_matrices",
"*HC_dynamic27.csv"
),

"M0":
os.path.join(
BASE,
"window_050_step_1",
"dynamic_edge_matrices",
"*M0_dynamic27.csv"
),

"M1":
os.path.join(
BASE,
"window_050_step_1",
"dynamic_edge_matrices",
"*M1_dynamic27.csv"
)

}


# ===============================
# Load centroid
# ===============================

print("="*60)
print("LOAD CENTROID")
print("="*60)

cent=pd.read_csv(CENTROID_FILE)

cent.columns=cent.columns.astype(str).str.strip()

print(cent.head())


# 第一列为 State1_Low / State2_High
state1=cent.iloc[0,1:].values.astype(float)
state2=cent.iloc[1,1:].values.astype(float)

assert len(state1)==27
assert len(state2)==27

print("State1 mean =",np.mean(state1))
print("State2 mean =",np.mean(state2))


# ===============================
# assign state
# ===============================

def assign_state(file):

    df=pd.read_csv(file)

    # columns:
    # Window Start_TR Start_sec + 27 edges
    X=df.iloc[:,3:].values.astype(float)

    labels=[]

    for x in X:

        d1=np.linalg.norm(x-state1)
        d2=np.linalg.norm(x-state2)

        if d1<d2:
            labels.append(1)
        else:
            labels.append(2)

    out=pd.DataFrame({

        "Window":df["Window"],
        "Start_sec":df["Start_sec"],
        "State":labels

    })

    return out



# ===============================
# Generate sequences
# ===============================

print("\nGENERATE STATE SEQUENCES")

all_seq=[]


for group,pattern in GROUP_FILES.items():

    files=sorted(glob.glob(pattern))

    print(group,"N =",len(files))

    for f in files:

        name=os.path.basename(f)

        if group=="HC":
            sid=name.replace("_HC_dynamic27.csv","")
        else:
            sid=name.replace("_dynamic27.csv","")

        seq=assign_state(f)

        seq["Subject"]=sid
        seq["Group"]=group


        save=os.path.join(
            SEQ_DIR,
            f"{sid}_{group}_state_sequence.csv"
        )

        seq.to_csv(
            save,
            index=False
        )

        all_seq.append(seq)



seq_all=pd.concat(
    all_seq,
    ignore_index=True
)


seq_all.to_csv(
    os.path.join(
        SEQ_DIR,
        "All_subject_state_sequences.csv"
    ),
    index=False
)


print("\nSequence QC")
print(
seq_all.groupby("Group")
.Subject.nunique()
)



# ===============================
# occurrence calculation
# ===============================

print("\nCALCULATE OCCURRENCE")


occ=[]


for group in ["HC","M0","M1"]:

    tmp=seq_all[
        seq_all.Group==group
    ]

    n=tmp.Subject.nunique()


    for w in sorted(tmp.Window.unique()):

        tw=tmp[tmp.Window==w]

        for state in [1,2]:

            p=np.sum(
                tw.State==state
            )/n

            sec=tw.Start_sec.mean()

            occ.append(
                [
                    group,
                    w,
                    sec,
                    state,
                    p
                ]
            )


occ=pd.DataFrame(
occ,
columns=[
"Group",
"Window",
"Start_sec",
"State",
"Occurrence"
]
)


occ.to_csv(
os.path.join(
FIG_DIR,
"Figure2_StateOccurrence_data.csv"
),
index=False
)



# ===============================
# smoothing
# ===============================

def smooth(y,n=30):

    return pd.Series(y).rolling(
        n,
        center=True,
        min_periods=1
    ).mean().values



# ===============================
# Plot Figure 2
# ===============================

print("\nDRAW FIGURE")


fig,axs=plt.subplots(
2,
1,
figsize=(10,8),
sharex=True
)


colors={
"HC":"black",
"M0":"red",
"M1":"blue"
}


for state,ax in zip([1,2],axs):

    for group in ["HC","M0","M1"]:

        tmp=occ[
            (occ.Group==group)&
            (occ.State==state)
        ]

        ax.plot(
            tmp.Start_sec,
            smooth(tmp.Occurrence.values),
            linewidth=2.5,
            label=group,
            color=colors[group]
        )


    ax.set_ylim(0,1)

    ax.set_ylabel(
        "Occurrence probability",
        fontsize=12
    )

    ax.set_title(
        f"State {state}",
        fontsize=14,
        fontweight="bold"
    )

    ax.grid(
        alpha=0.3
    )


axs[1].set_xlabel(
"Time (s)",
fontsize=12
)


axs[0].legend(
frameon=False,
ncol=3,
fontsize=11
)


plt.tight_layout()


outfile=os.path.join(
FIG_DIR,
"Figure_Window50_StateOccurrence_Trajectory.png"
)


plt.savefig(
outfile,
dpi=300,
bbox_inches="tight"
)


plt.close()


print("\nDONE")
print(outfile)
