#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Window50 K=2 state-specific 27-edge median connectivity profiles
================================================================
文献方法：
For each participant and each dynamic state, the state-specific
connectivity profile is represented by the element-wise median
connectivity across all windows assigned to that state.

数据：
- Patients: 42 subjects × M0/M1 = 84 scans
- HC: 48 scans
- Window length = 50 TR
- Step = 1 TR
- 541 windows/scan
- 27 fixed MetS-related NBS edges
- Fisher-z connectivity
- Fixed patient-derived K=2 centroids
- Euclidean nearest-centroid assignment
- State1 = low-connectivity state
- State2 = high-connectivity state
"""

import os
from pathlib import Path
from itertools import combinations
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================
# 1. Paths
# ============================================================
ROOT = Path(
    os.environ.get(
        "WM_DYNAMIC_ROOT",
        Path(__file__).resolve().parent.parent / "data",
    )
)
PATIENT_DIR = ROOT/"window_050_step_1/dynamic_edge_matrices"
HC_DIR = ROOT/"HC_window_050_step_1/dynamic_edge_matrices"
CENTROID_FILE = ROOT/"Window_Sensitivity_K2_10/window_050/centroids/centroids_K02.csv"
OUT_DIR = ROOT/"StateSpecific_27Edge_Median_Window50"
FIG_DIR = OUT_DIR/"Figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

for p in [PATIENT_DIR,HC_DIR,CENTROID_FILE]:
    if not p.exists():
        raise FileNotFoundError(f"Missing: {p}")

# ============================================================
# 2. Fixed 27 NBS edges
# ============================================================
EDGES = [
    (1,37),(1,38),(2,7),(2,15),(4,37),(7,8),(7,17),
    (15,37),(17,37),(17,39),(21,35),(34,37),(35,37),
    (35,38),(36,37),(37,38),(37,39),(37,40),(37,43),
    (37,44),(37,45),(37,46),(37,47),(37,48),(38,40),
    (38,46),(38,48)
]
EDGE_NAMES = [f"ROI{a}-ROI{b}" for a,b in EDGES]
EXPECTED_WINDOWS = 541

# ============================================================
# 3. Read centroids EXACTLY by edge names
# ============================================================
def load_centroids(path):
    df = pd.read_csv(path)

    required = ["State","Centroid_Mean_Strength"] + EDGE_NAMES
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Centroid file missing columns: {missing}")

    if set(df["State"].astype(int)) != {1,2}:
        raise RuntimeError("Centroid file must contain State 1 and State 2.")

    df["State"] = df["State"].astype(int)
    df = df.sort_values("State")

    c1 = df.loc[df["State"]==1,EDGE_NAMES].iloc[0].to_numpy(float)
    c2 = df.loc[df["State"]==2,EDGE_NAMES].iloc[0].to_numpy(float)
    centroids = np.vstack([c1,c2])

    reported1 = float(df.loc[df["State"]==1,"Centroid_Mean_Strength"].iloc[0])
    reported2 = float(df.loc[df["State"]==2,"Centroid_Mean_Strength"].iloc[0])
    computed1 = float(np.mean(c1))
    computed2 = float(np.mean(c2))

    print("\n============================================================")
    print("FIXED K=2 CENTROIDS")
    print("============================================================")
    print(f"State 1 reported mean = {reported1:.6f}")
    print(f"State 1 computed mean = {computed1:.6f}")
    print(f"State 2 reported mean = {reported2:.6f}")
    print(f"State 2 computed mean = {computed2:.6f}")

    # Hard QC：必须与既有最终结果一致
    if not np.isclose(reported1,0.15712878474113576,atol=1e-6):
        raise RuntimeError("State1 centroid mean does not match final Window50 result.")
    if not np.isclose(reported2,0.5746509225445815,atol=1e-6):
        raise RuntimeError("State2 centroid mean does not match final Window50 result.")
    if not np.isclose(computed1,reported1,atol=1e-6):
        raise RuntimeError("State1 27-edge mean does not match reported centroid mean.")
    if not np.isclose(computed2,reported2,atol=1e-6):
        raise RuntimeError("State2 27-edge mean does not match reported centroid mean.")

    print("Centroid QC: PASS")
    return centroids

CENTROIDS = load_centroids(CENTROID_FILE)

# ============================================================
# 4. Read dynamic27 CSV by exact edge columns
# ============================================================
def load_dynamic_csv(path):
    df = pd.read_csv(path)

    missing = [c for c in EDGE_NAMES if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing edges {missing}")

    x = df[EDGE_NAMES].apply(pd.to_numeric,errors="coerce").to_numpy(float)

    if x.shape != (EXPECTED_WINDOWS,27):
        raise ValueError(f"{path.name}: shape={x.shape}, expected (541,27)")

    return x

# ============================================================
# 5. File naming
# ============================================================
PAT_RE = re.compile(r"^(sub-\d+)_ses-(M[01])_dynamic27\.csv$",re.I)
HC_RE = re.compile(r"^(sub-H\d+)_HC_dynamic27\.csv$",re.I)

def patient_info(path):
    m = PAT_RE.match(path.name)
    if not m:
        return None
    return m.group(1).lower(),m.group(2).upper()

def hc_info(path):
    m = HC_RE.match(path.name)
    if not m:
        return None
    return m.group(1).upper()

patient_files = sorted([p for p in PATIENT_DIR.glob("*.csv") if patient_info(p)])
hc_files = sorted([p for p in HC_DIR.glob("*.csv") if hc_info(p)])

print("\n============================================================")
print("INPUT FILES")
print("============================================================")
print(f"Patient files = {len(patient_files)}")
print(f"HC files      = {len(hc_files)}")

if len(patient_files)!=84:
    raise RuntimeError(f"Expected 84 patient scans, found {len(patient_files)}")
if len(hc_files)!=48:
    raise RuntimeError(f"Expected 48 HC scans, found {len(hc_files)}")

# ============================================================
# 6. Fixed-state assignment
# ============================================================
def assign_states(x):
    valid = np.isfinite(x).all(axis=1)
    labels = np.zeros(x.shape[0],dtype=int)

    xv = x[valid,:]
    d1 = np.sum((xv-CENTROIDS[0])**2,axis=1)
    d2 = np.sum((xv-CENTROIDS[1])**2,axis=1)
    labels[valid] = np.where(d1<=d2,1,2)

    return labels,valid

# ============================================================
# 7. Literature-style element-wise median profile
# ============================================================
def process_scan(path,subject,group,session,condition):
    x = load_dynamic_csv(path)
    labels,valid = assign_states(x)

    qc = {
        "Subject":subject,
        "Group":group,
        "Session":session,
        "Condition":condition,
        "SourceFile":str(path),
        "N_Windows":len(x),
        "N_ValidWindows":int(valid.sum()),
        "N_InvalidWindows":int((~valid).sum())
    }

    rows = []

    for state in [1,2]:
        idx = labels==state
        nwin = int(idx.sum())

        qc[f"State{state}_NWindows"] = nwin
        qc[f"State{state}_FO"] = nwin/valid.sum()

        # If a subject never visits a state, retain NA
        if nwin==0:
            med = np.full(27,np.nan)
        else:
            med = np.nanmedian(x[idx,:],axis=0)

        for e,((n1,n2),edge) in enumerate(zip(EDGES,EDGE_NAMES),1):
            rows.append({
                "Subject":subject,
                "Group":group,
                "Session":session,
                "Condition":condition,
                "State":state,
                "Edge":e,
                "Node1":n1,
                "Node2":n2,
                "EdgeName":edge,
                "N_StateWindows":nwin,
                "Median_FisherZ":med[e-1]
            })

    return rows,qc

# ============================================================
# 8. Process all scans
# ============================================================
all_rows,all_qc,failed = [],[],[]

print("\n============================================================")
print("PROCESS PATIENTS")
print("============================================================")

for i,p in enumerate(patient_files,1):
    subject,session = patient_info(p)
    try:
        rows,qc = process_scan(p,subject,"MetS",session,session)
        all_rows.extend(rows)
        all_qc.append(qc)
        print(f"[{i:02d}/84] {subject} {session} | "
              f"S1={qc['State1_NWindows']:3d} S2={qc['State2_NWindows']:3d}")
    except Exception as e:
        failed.append({"File":str(p),"Reason":str(e)})
        print("[FAILED]",p.name,e)

print("\n============================================================")
print("PROCESS HC")
print("============================================================")

for i,p in enumerate(hc_files,1):
    subject = hc_info(p)
    try:
        rows,qc = process_scan(p,subject,"HC","HC","HC")
        all_rows.extend(rows)
        all_qc.append(qc)
        print(f"[{i:02d}/48] {subject} HC | "
              f"S1={qc['State1_NWindows']:3d} S2={qc['State2_NWindows']:3d}")
    except Exception as e:
        failed.append({"File":str(p),"Reason":str(e)})
        print("[FAILED]",p.name,e)

long_df = pd.DataFrame(all_rows)
qc_df = pd.DataFrame(all_qc)
failed_df = pd.DataFrame(failed)

if len(qc_df)!=132:
    raise RuntimeError(f"Expected 132 scans, successfully processed {len(qc_df)}")

# ============================================================
# 9. HARD QC against previous final Window50 results
# ============================================================
print("\n============================================================")
print("STATE ASSIGNMENT QC")
print("============================================================")

expected_occurrence = {"HC":36,"M0":42,"M1":38}
expected_fo2 = {"HC":0.114603,"M0":0.311328,"M1":0.276604}

qc_pass = True

for condition in ["HC","M0","M1"]:
    d = qc_df[qc_df["Condition"]==condition]

    n1 = int((d["State1_NWindows"]>0).sum())
    n2 = int((d["State2_NWindows"]>0).sum())
    fo1 = d["State1_FO"].mean()
    fo2 = d["State2_FO"].mean()

    print(f"{condition}:")
    print(f" State1 occurrence = {n1}/{len(d)}")
    print(f" State2 occurrence = {n2}/{len(d)}")
    print(f" Mean FO State1    = {fo1:.6f}")
    print(f" Mean FO State2    = {fo2:.6f}")

    if n2 != expected_occurrence[condition]:
        qc_pass = False
        print(f" **FAIL occurrence: expected {expected_occurrence[condition]}")

    if abs(fo2-expected_fo2[condition])>0.002:
        qc_pass = False
        print(f" **FAIL FO: expected approximately {expected_fo2[condition]:.6f}")

if not qc_pass:
    raise RuntimeError(
        "\nState assignment does not reproduce previous final Window50 results.\n"
        "No final tables/figures will be interpreted until this is resolved."
    )

print("\nState-assignment QC: PASS")

# ============================================================
# 10. Wide-format profile table
# ============================================================
wide_df = long_df.pivot(
    index=["Subject","Group","Session","Condition"],
    columns=["State","EdgeName"],
    values="Median_FisherZ"
)
wide_df.columns = [f"State{s}_{edge}" for s,edge in wide_df.columns]
wide_df = wide_df.reset_index()

# ============================================================
# 11. Group descriptive statistics
# ============================================================
def q25(x): return x.quantile(.25)
def q75(x): return x.quantile(.75)

desc_df = (
    long_df.groupby(
        ["Condition","State","Edge","Node1","Node2","EdgeName"]
    )["Median_FisherZ"]
    .agg(N="count",Mean="mean",SD="std",
         Median="median",Q25=q25,Q75=q75)
    .reset_index()
)

# ============================================================
# 12. Patient paired M1-M0 change
# ============================================================
patient_long = long_df[long_df["Group"]=="MetS"].copy()

delta_df = patient_long.pivot(
    index=["Subject","State","Edge","Node1","Node2","EdgeName"],
    columns="Session",
    values="Median_FisherZ"
).reset_index()

delta_df["Delta_M1_minus_M0"] = delta_df["M1"]-delta_df["M0"]

delta_desc = (
    delta_df.groupby(
        ["State","Edge","Node1","Node2","EdgeName"]
    )["Delta_M1_minus_M0"]
    .agg(
        N_Pairs="count",
        Mean_Delta="mean",
        SD_Delta="std",
        Median_Delta="median",
        Q25=q25,
        Q75=q75
    )
    .reset_index()
)

# ============================================================
# 13. Group profile pattern correlations
# ============================================================
corr_rows = []

for state in [1,2]:
    d = desc_df[desc_df["State"]==state]
    profiles = {}

    for condition in ["HC","M0","M1"]:
        profiles[condition] = (
            d[d["Condition"]==condition]
            .set_index("EdgeName")
            .reindex(EDGE_NAMES)["Mean"]
            .to_numpy(float)
        )

    profiles["Centroid"] = CENTROIDS[state-1]

    for a,b in combinations(["HC","M0","M1","Centroid"],2):
        x,y = profiles[a],profiles[b]
        ok = np.isfinite(x)&np.isfinite(y)
        r = np.corrcoef(x[ok],y[ok])[0,1]
        corr_rows.append({
            "State":state,
            "Profile1":a,
            "Profile2":b,
            "N_Edges":int(ok.sum()),
            "Pattern_r":r
        })

corr_df = pd.DataFrame(corr_rows)

# ============================================================
# 14. Save tables
# ============================================================
LONG_FILE = OUT_DIR/"Window50_K2_StateSpecific_27edge_median_profiles_long.csv"
WIDE_FILE = OUT_DIR/"Window50_K2_StateSpecific_27edge_median_profiles_wide.csv"
QC_FILE = OUT_DIR/"Window50_K2_StateSpecific_scan_QC.csv"
DESC_FILE = OUT_DIR/"Window50_K2_StateSpecific_27edge_group_descriptives.csv"
DELTA_FILE = OUT_DIR/"Window50_K2_StateSpecific_27edge_M0_M1_paired_delta.csv"
DELTA_DESC_FILE = OUT_DIR/"Window50_K2_StateSpecific_27edge_M0_M1_delta_descriptives.csv"
CORR_FILE = OUT_DIR/"Window50_K2_StateSpecific_group_profile_correlations.csv"
CENTROID_OUT = OUT_DIR/"Window50_K2_centroids_used_State1Low_State2High.csv"
FAILED_FILE = OUT_DIR/"Window50_K2_StateSpecific_failed_files.csv"

long_df.to_csv(LONG_FILE,index=False)
wide_df.to_csv(WIDE_FILE,index=False)
qc_df.to_csv(QC_FILE,index=False)
desc_df.to_csv(DESC_FILE,index=False)
delta_df.to_csv(DELTA_FILE,index=False)
delta_desc.to_csv(DELTA_DESC_FILE,index=False)
corr_df.to_csv(CORR_FILE,index=False)
failed_df.to_csv(FAILED_FILE,index=False)

pd.DataFrame(
    CENTROIDS,
    index=["State1_Low","State2_High"],
    columns=EDGE_NAMES
).to_csv(CENTROID_OUT)

# ============================================================
# 15. Heatmap function
# ============================================================
def heatmap(mat,row_labels,col_labels,title,cbar_label,stem,nmat=None,delta=False):
    mat = np.asarray(mat,float)
    fig,ax = plt.subplots(figsize=(7,10))

    if delta:
        vmax = np.nanmax(np.abs(mat))
        if not np.isfinite(vmax) or vmax==0:
            vmax = 1
        im = ax.imshow(mat,aspect="auto",cmap="coolwarm",vmin=-vmax,vmax=vmax)
    else:
        im = ax.imshow(mat,aspect="auto",cmap="viridis")

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels,fontsize=10)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels,fontsize=8)
    ax.set_ylabel("MetS-related NBS edge")
    ax.set_title(title,fontsize=13,pad=12)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if not np.isfinite(mat[i,j]):
                txt = "NA"
            elif nmat is None:
                txt = f"{mat[i,j]:.3f}"
            else:
                txt = f"{mat[i,j]:.3f}\n(n={int(nmat[i,j])})"
            ax.text(j,i,txt,ha="center",va="center",fontsize=6.4)

    cb = fig.colorbar(im,ax=ax,fraction=.04,pad=.03)
    cb.set_label(cbar_label,fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR/f"{stem}.pdf",bbox_inches="tight")
    fig.savefig(FIG_DIR/f"{stem}.png",dpi=300,bbox_inches="tight")
    plt.close(fig)

# ============================================================
# 16. HC/M0/M1 profile heatmaps
# ============================================================
for state,state_label in [
    (1,"State 1: low-connectivity state"),
    (2,"State 2: high-connectivity state")
]:
    d = desc_df[desc_df["State"]==state]

    means = (
        d.pivot(index="EdgeName",columns="Condition",values="Mean")
        .reindex(index=EDGE_NAMES,columns=["HC","M0","M1"])
    )
    ns = (
        d.pivot(index="EdgeName",columns="Condition",values="N")
        .reindex(index=EDGE_NAMES,columns=["HC","M0","M1"])
    )

    heatmap(
        means.to_numpy(),
        EDGE_NAMES,
        ["HC","M0","M1"],
        f"{state_label}\nState-specific element-wise median connectivity",
        "Mean subject-specific median Fisher-z",
        f"Figure_Window50_State{state}_HC_M0_M1_27edge_profile",
        nmat=ns.to_numpy()
    )

# ============================================================
# 17. M1-M0 delta heatmap
# ============================================================
dm = (
    delta_desc.pivot(index="EdgeName",columns="State",values="Mean_Delta")
    .reindex(index=EDGE_NAMES,columns=[1,2])
)
dn = (
    delta_desc.pivot(index="EdgeName",columns="State",values="N_Pairs")
    .reindex(index=EDGE_NAMES,columns=[1,2])
)

heatmap(
    dm.to_numpy(),
    EDGE_NAMES,
    ["State 1\nLow","State 2\nHigh"],
    "M1 - M0 change in state-specific edge connectivity",
    "Mean paired Δ median Fisher-z",
    "Figure_Window50_StateSpecific_27edge_M1_minus_M0_delta",
    nmat=dn.to_numpy(),
    delta=True
)

# ============================================================
# 18. Print pattern correlations
# ============================================================
print("\n============================================================")
print("GROUP PROFILE PATTERN CORRELATIONS")
print("============================================================")
print(corr_df.to_string(index=False,float_format=lambda x:f"{x:.4f}"))

# ============================================================
# 19. Finish
# ============================================================
print("\n============================================================")
print("FINISHED")
print("============================================================")
print("Output:",OUT_DIR)

print("\nMain tables:")
for p in [
    LONG_FILE,QC_FILE,DESC_FILE,
    DELTA_FILE,DELTA_DESC_FILE,CORR_FILE
]:
    print(" ",p.name)

print("\nFigures:")
for p in sorted(FIG_DIR.glob("*.png")):
    print(" ",p.name)

print("\nIMPORTANT:")
print("Heatmaps are descriptive, not significance maps.")
print("Next analysis: State-specific multivariate overall test")
print("followed by edge-wise follow-up and FDR correction.")
