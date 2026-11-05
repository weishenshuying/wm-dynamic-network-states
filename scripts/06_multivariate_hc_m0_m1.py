#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Window50 state-specific HC-M0-M1 27-edge characterization
=========================================================
Reference-paper-inspired hierarchy:
state-specific multivariate test -> edge-wise follow-up -> FDR

Each state is analysed independently.

Planned multivariate contrasts:
1. M0 - HC : independent
2. M1 - HC : independent
3. M1 - M0 : paired

Overall multivariate tests:
- M0-HC / M1-HC:
  two-sample Hotelling T2 + 10,000 label permutations
- M1-M0:
  paired Hotelling T2 + 10,000 subject-level sign flips

Multiplicity:
- Holm across the 3 multivariate contrasts within each state = primary
- Holm across all 6 state x contrast tests = sensitivity
- Edge follow-up: BH-FDR across 27 edges within each contrast
- Additional 81-edge and 162-edge BH-FDR saved as sensitivity

Contrast directions:
M0-HC = M0 minus HC
M1-HC = M1 minus HC
M1-M0 = M1 minus M0

IMPORTANT:
The 27 edges were originally selected by static M0-vs-HC NBS.
Therefore HC comparisons here are secondary characterization,
not independent validation.

State 2 connectivity analyses are conditional on visiting State 2.
"""

import os
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================
# 1. Paths / settings
# ============================================================
ROOT = Path(
    os.environ.get(
        "WM_DYNAMIC_ROOT",
        Path(__file__).resolve().parent.parent / "data",
    )
)
PROFILE_DIR = ROOT/"StateSpecific_27Edge_Median_Window50"
INPUT_FILE = PROFILE_DIR/"Window50_K2_StateSpecific_27edge_median_profiles_long.csv"
OUT_DIR = PROFILE_DIR/"Statistics_HC_M0_M1"
FIG_DIR = OUT_DIR/"Figures"
OUT_DIR.mkdir(parents=True,exist_ok=True)
FIG_DIR.mkdir(parents=True,exist_ok=True)

N_PERM = 10000
SEED = 2026

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Missing input:\n{INPUT_FILE}")

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
P = len(EDGE_NAMES)
CONTRAST_ORDER = ["M0-HC","M1-HC","M1-M0"]

# ============================================================
# 3. Multiple-comparison correction
# ============================================================
def bh_fdr(pvals):
    x = np.asarray(pvals,float)
    out = np.full(len(x),np.nan)
    ok = np.isfinite(x)
    p = x[ok]
    if len(p)==0:
        return out
    order = np.argsort(p)
    ps = p[order]
    m = len(ps)
    adj = ps*m/np.arange(1,m+1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj,0,1)
    temp = np.empty(m)
    temp[order] = adj
    out[ok] = temp
    return out

def holm(pvals):
    x = np.asarray(pvals,float)
    out = np.full(len(x),np.nan)
    ok = np.isfinite(x)
    p = x[ok]
    if len(p)==0:
        return out
    order = np.argsort(p)
    ps = p[order]
    m = len(ps)
    adj = np.array([(m-i)*ps[i] for i in range(m)])
    adj = np.maximum.accumulate(adj)
    adj = np.clip(adj,0,1)
    temp = np.empty(m)
    temp[order] = adj
    out[ok] = temp
    return out

# ============================================================
# 4. Load subject x 27 profile matrix
# ============================================================
df = pd.read_csv(INPUT_FILE)

required = [
    "Subject","Group","Session","Condition",
    "State","EdgeName","Median_FisherZ"
]
missing = [c for c in required if c not in df.columns]
if missing:
    raise RuntimeError(f"Missing columns: {missing}")

def condition_matrix(state,condition):
    d = df[(df["State"]==state)&(df["Condition"]==condition)].copy()
    w = d.pivot(index="Subject",columns="EdgeName",values="Median_FisherZ")
    w = w.reindex(columns=EDGE_NAMES)
    complete = np.isfinite(w.to_numpy(float)).all(axis=1)
    w = w.loc[complete]
    return w.index.to_numpy(),w.to_numpy(float)

def paired_m0_m1(state):
    d = df[
        (df["State"]==state)&
        (df["Condition"].isin(["M0","M1"]))
    ].copy()

    w = d.pivot(
        index="Subject",
        columns=["Condition","EdgeName"],
        values="Median_FisherZ"
    )

    for cond in ["M0","M1"]:
        for e in EDGE_NAMES:
            if (cond,e) not in w.columns:
                w[(cond,e)] = np.nan

    m0 = w[[("M0",e) for e in EDGE_NAMES]].to_numpy(float)
    m1 = w[[("M1",e) for e in EDGE_NAMES]].to_numpy(float)
    ok = np.isfinite(m0).all(axis=1)&np.isfinite(m1).all(axis=1)

    return w.index.to_numpy()[ok],m0[ok],m1[ok]

# ============================================================
# 5. Two-sample Hotelling T2
# ============================================================
def hotelling_two_sample(x,y):
    """
    Difference direction = mean(x) - mean(y)
    Classical pooled-covariance two-sample Hotelling T2.
    """
    x = np.asarray(x,float)
    y = np.asarray(y,float)
    n1,p = x.shape
    n2,p2 = y.shape
    assert p==p2

    mx = x.mean(axis=0)
    my = y.mean(axis=0)
    s1 = np.cov(x,rowvar=False,ddof=1)
    s2 = np.cov(y,rowvar=False,ddof=1)

    sp = ((n1-1)*s1+(n2-1)*s2)/(n1+n2-2)
    rank = np.linalg.matrix_rank(sp)
    cond = np.linalg.cond(sp)

    try:
        sinv = np.linalg.inv(sp)
        inv_method = "inverse"
    except np.linalg.LinAlgError:
        sinv = np.linalg.pinv(sp)
        inv_method = "pseudoinverse"

    diff = mx-my
    t2 = float((n1*n2/(n1+n2))*diff@sinv@diff)

    df1 = p
    df2 = n1+n2-p-1
    f = float(
        ((n1+n2-p-1)/((n1+n2-2)*p))*t2
    )
    pval = float(stats.f.sf(f,df1,df2))

    return {
        "N1":n1,"N2":n2,"P":p,
        "T2":t2,"F":f,"df1":df1,"df2":df2,
        "P_parametric":pval,
        "Covariance_Rank":rank,
        "Covariance_Condition_Number":cond,
        "Inverse_Method":inv_method
    }

# ============================================================
# 6. Independent label-permutation Hotelling
# ============================================================
def permutation_two_sample(x,y,n_perm,seed):
    x = np.asarray(x,float)
    y = np.asarray(y,float)
    n1 = len(x)
    pooled = np.vstack([x,y])
    rng = np.random.default_rng(seed)

    obs = hotelling_two_sample(x,y)["T2"]
    exceed = 0

    for _ in range(n_perm):
        ix = rng.permutation(len(pooled))
        xp = pooled[ix[:n1]]
        yp = pooled[ix[n1:]]

        tperm = hotelling_two_sample(xp,yp)["T2"]
        if tperm >= obs-1e-12:
            exceed += 1

    pperm = (exceed+1)/(n_perm+1)
    return obs,pperm,exceed

# ============================================================
# 7. Paired Hotelling T2 + sign flips
# ============================================================
def hotelling_paired(delta):
    delta = np.asarray(delta,float)
    n,p = delta.shape
    if n<=p:
        raise ValueError(f"Need n>p; n={n}, p={p}")

    md = delta.mean(axis=0)
    s = np.cov(delta,rowvar=False,ddof=1)
    rank = np.linalg.matrix_rank(s)
    cond = np.linalg.cond(s)

    try:
        sinv = np.linalg.inv(s)
        inv_method = "inverse"
    except np.linalg.LinAlgError:
        sinv = np.linalg.pinv(s)
        inv_method = "pseudoinverse"

    t2 = float(n*md@sinv@md)
    f = float((n-p)/(p*(n-1))*t2)
    df1 = p
    df2 = n-p
    pval = float(stats.f.sf(f,df1,df2))

    return {
        "N1":n,"N2":n,"P":p,
        "T2":t2,"F":f,"df1":df1,"df2":df2,
        "P_parametric":pval,
        "Covariance_Rank":rank,
        "Covariance_Condition_Number":cond,
        "Inverse_Method":inv_method
    }

def permutation_paired(delta,n_perm,seed):
    delta = np.asarray(delta,float)
    rng = np.random.default_rng(seed)

    obs = hotelling_paired(delta)["T2"]
    exceed = 0

    for _ in range(n_perm):
        signs = rng.choice([-1.,1.],size=len(delta))
        dp = delta*signs[:,None]
        tperm = hotelling_paired(dp)["T2"]
        if tperm >= obs-1e-12:
            exceed += 1

    pperm = (exceed+1)/(n_perm+1)
    return obs,pperm,exceed

# ============================================================
# 8. Effect-size helpers
# ============================================================
def hedges_g(x,y):
    """Positive means x > y."""
    x = np.asarray(x,float)
    y = np.asarray(y,float)
    n1,n2 = len(x),len(y)
    s1 = np.std(x,ddof=1)
    s2 = np.std(y,ddof=1)

    sp = np.sqrt(
        ((n1-1)*s1**2+(n2-1)*s2**2)/(n1+n2-2)
    )
    if sp==0:
        return np.nan

    d = (np.mean(x)-np.mean(y))/sp
    df = n1+n2-2
    j = 1-3/(4*df-1)
    return j*d

def cohens_dz(d):
    d = np.asarray(d,float)
    sd = np.std(d,ddof=1)
    return np.mean(d)/sd if sd>0 else np.nan

# ============================================================
# 9. Edge-wise independent test
# ============================================================
def independent_edge_tests(x,y,state,contrast):
    rows = []

    for j,e in enumerate(EDGE_NAMES):
        a = x[:,j]
        b = y[:,j]

        n1,n2 = len(a),len(b)
        m1,m2 = np.mean(a),np.mean(b)
        sd1,sd2 = np.std(a,ddof=1),np.std(b,ddof=1)
        diff = m1-m2

        # Welch t-test
        t,p = stats.ttest_ind(a,b,equal_var=False)

        se = np.sqrt(sd1**2/n1+sd2**2/n2)
        num = (sd1**2/n1+sd2**2/n2)**2
        den = (
            (sd1**2/n1)**2/(n1-1)+
            (sd2**2/n2)**2/(n2-1)
        )
        welch_df = num/den
        tcrit = stats.t.ppf(.975,welch_df)
        lo = diff-tcrit*se
        hi = diff+tcrit*se

        u,up = stats.mannwhitneyu(
            a,b,alternative="two-sided",method="auto"
        )

        rows.append({
            "State":state,
            "Contrast":contrast,
            "Edge":e,
            "N_Group1":n1,
            "N_Group2":n2,
            "Group1_Mean":m1,
            "Group1_SD":sd1,
            "Group2_Mean":m2,
            "Group2_SD":sd2,
            "Mean_Difference":diff,
            "CI95_Low":lo,
            "CI95_High":hi,
            "Effect_Size":hedges_g(a,b),
            "Effect_Size_Type":"Hedges_g",
            "Test_Statistic":t,
            "Univariate_P":p,
            "Robust_Test_Statistic":u,
            "Robust_P":up
        })

    out = pd.DataFrame(rows)
    out["Edge_q_BH_within_contrast"] = bh_fdr(out["Univariate_P"])
    out["Robust_q_BH_within_contrast"] = bh_fdr(out["Robust_P"])
    return out

# ============================================================
# 10. Edge-wise paired test
# ============================================================
def paired_edge_tests(m0,m1,state):
    rows = []

    for j,e in enumerate(EDGE_NAMES):
        a = m0[:,j]
        b = m1[:,j]
        d = b-a
        n = len(d)

        md = np.mean(d)
        sd = np.std(d,ddof=1)
        se = sd/np.sqrt(n)
        tcrit = stats.t.ppf(.975,n-1)

        t,p = stats.ttest_rel(b,a)

        if np.allclose(d,0):
            w,wp = 0.,1.
        else:
            w,wp = stats.wilcoxon(
                b,a,
                zero_method="wilcox",
                alternative="two-sided",
                method="auto"
            )

        rows.append({
            "State":state,
            "Contrast":"M1-M0",
            "Edge":e,
            "N_Group1":n,
            "N_Group2":n,
            "Group1_Mean":np.mean(b),
            "Group1_SD":np.std(b,ddof=1),
            "Group2_Mean":np.mean(a),
            "Group2_SD":np.std(a,ddof=1),
            "Mean_Difference":md,
            "CI95_Low":md-tcrit*se,
            "CI95_High":md+tcrit*se,
            "Effect_Size":cohens_dz(d),
            "Effect_Size_Type":"Cohens_dz",
            "Test_Statistic":t,
            "Univariate_P":p,
            "Robust_Test_Statistic":w,
            "Robust_P":wp
        })

    out = pd.DataFrame(rows)
    out["Edge_q_BH_within_contrast"] = bh_fdr(out["Univariate_P"])
    out["Robust_q_BH_within_contrast"] = bh_fdr(out["Robust_P"])
    return out

# ============================================================
# 11. Run the 6 state x contrast multivariate tests
# ============================================================
overall_rows = []
edge_tables = []
subject_rows = []

print("\n============================================================")
print("STATE-SPECIFIC HC-M0-M1 MULTIVARIATE CHARACTERIZATION")
print("============================================================")

for state in [1,2]:

    hc_ids,hc = condition_matrix(state,"HC")
    m0_ids,m0 = condition_matrix(state,"M0")
    m1_ids,m1 = condition_matrix(state,"M1")
    pair_ids,m0_pair,m1_pair = paired_m0_m1(state)

    print(f"\n================ STATE {state} ================")
    print(f"HC visitors = {len(hc_ids)}")
    print(f"M0 visitors = {len(m0_ids)}")
    print(f"M1 visitors = {len(m1_ids)}")
    print(f"M0-M1 complete pairs = {len(pair_ids)}")

    for condition,ids in [
        ("HC",hc_ids),("M0",m0_ids),("M1",m1_ids)
    ]:
        for s in ids:
            subject_rows.append({
                "State":state,
                "Analysis":"GroupProfile",
                "Condition":condition,
                "Subject":s
            })

    # ---------- M0 - HC ----------
    print("\nM0 - HC")
    h = hotelling_two_sample(m0,hc)
    obs,pperm,exc = permutation_two_sample(
        m0,hc,N_PERM,SEED+state*100+1
    )

    print(
        f"T2={h['T2']:.5f} | F({h['df1']},{h['df2']})={h['F']:.5f} | "
        f"parametric p={h['P_parametric']:.8g} | perm p={pperm:.8g}"
    )

    overall_rows.append({
        "State":state,"Contrast":"M0-HC",
        "Design":"Independent",
        "N1":len(m0),"N2":len(hc),
        "Group1":"M0","Group2":"HC",
        "Hotelling_T2":h["T2"],"F":h["F"],
        "df1":h["df1"],"df2":h["df2"],
        "Parametric_P":h["P_parametric"],
        "Permutation_P":pperm,
        "Permutation_N":N_PERM,
        "Permutation_Exceedances":exc,
        "Covariance_Rank":h["Covariance_Rank"],
        "Covariance_Condition_Number":h["Covariance_Condition_Number"],
        "Inverse_Method":h["Inverse_Method"]
    })

    edge_tables.append(
        independent_edge_tests(m0,hc,state,"M0-HC")
    )

    # ---------- M1 - HC ----------
    print("M1 - HC")
    h = hotelling_two_sample(m1,hc)
    obs,pperm,exc = permutation_two_sample(
        m1,hc,N_PERM,SEED+state*100+2
    )

    print(
        f"T2={h['T2']:.5f} | F({h['df1']},{h['df2']})={h['F']:.5f} | "
        f"parametric p={h['P_parametric']:.8g} | perm p={pperm:.8g}"
    )

    overall_rows.append({
        "State":state,"Contrast":"M1-HC",
        "Design":"Independent",
        "N1":len(m1),"N2":len(hc),
        "Group1":"M1","Group2":"HC",
        "Hotelling_T2":h["T2"],"F":h["F"],
        "df1":h["df1"],"df2":h["df2"],
        "Parametric_P":h["P_parametric"],
        "Permutation_P":pperm,
        "Permutation_N":N_PERM,
        "Permutation_Exceedances":exc,
        "Covariance_Rank":h["Covariance_Rank"],
        "Covariance_Condition_Number":h["Covariance_Condition_Number"],
        "Inverse_Method":h["Inverse_Method"]
    })

    edge_tables.append(
        independent_edge_tests(m1,hc,state,"M1-HC")
    )

    # ---------- M1 - M0 ----------
    print("M1 - M0")
    delta = m1_pair-m0_pair
    h = hotelling_paired(delta)
    obs,pperm,exc = permutation_paired(
        delta,N_PERM,SEED+state*100+3
    )

    print(
        f"T2={h['T2']:.5f} | F({h['df1']},{h['df2']})={h['F']:.5f} | "
        f"parametric p={h['P_parametric']:.8g} | perm p={pperm:.8g}"
    )

    overall_rows.append({
        "State":state,"Contrast":"M1-M0",
        "Design":"Paired",
        "N1":len(pair_ids),"N2":len(pair_ids),
        "Group1":"M1","Group2":"M0",
        "Hotelling_T2":h["T2"],"F":h["F"],
        "df1":h["df1"],"df2":h["df2"],
        "Parametric_P":h["P_parametric"],
        "Permutation_P":pperm,
        "Permutation_N":N_PERM,
        "Permutation_Exceedances":exc,
        "Covariance_Rank":h["Covariance_Rank"],
        "Covariance_Condition_Number":h["Covariance_Condition_Number"],
        "Inverse_Method":h["Inverse_Method"]
    })

    edge_tables.append(
        paired_edge_tests(m0_pair,m1_pair,state)
    )

    for s in pair_ids:
        subject_rows.append({
            "State":state,
            "Analysis":"M1-M0_Paired",
            "Condition":"M0_and_M1",
            "Subject":s
        })

overall_df = pd.DataFrame(overall_rows)
edges_df = pd.concat(edge_tables,ignore_index=True)
subjects_df = pd.DataFrame(subject_rows)

# ============================================================
# 12. Multivariate multiplicity
# ============================================================
overall_df["Parametric_P_Holm_within_state"] = np.nan
overall_df["Permutation_P_Holm_within_state"] = np.nan

for state in [1,2]:
    ix = overall_df["State"]==state
    overall_df.loc[ix,"Parametric_P_Holm_within_state"] = holm(
        overall_df.loc[ix,"Parametric_P"]
    )
    overall_df.loc[ix,"Permutation_P_Holm_within_state"] = holm(
        overall_df.loc[ix,"Permutation_P"]
    )

overall_df["Parametric_P_Holm_all6"] = holm(
    overall_df["Parametric_P"]
)
overall_df["Permutation_P_Holm_all6"] = holm(
    overall_df["Permutation_P"]
)

# ============================================================
# 13. Edge multiplicity sensitivity
# ============================================================
edges_df["Edge_q_BH_all81_within_state"] = np.nan
edges_df["Robust_q_BH_all81_within_state"] = np.nan

for state in [1,2]:
    ix = edges_df["State"]==state
    edges_df.loc[ix,"Edge_q_BH_all81_within_state"] = bh_fdr(
        edges_df.loc[ix,"Univariate_P"]
    )
    edges_df.loc[ix,"Robust_q_BH_all81_within_state"] = bh_fdr(
        edges_df.loc[ix,"Robust_P"]
    )

edges_df["Edge_q_BH_all162"] = bh_fdr(
    edges_df["Univariate_P"]
)
edges_df["Robust_q_BH_all162"] = bh_fdr(
    edges_df["Robust_P"]
)

# Add multivariate gatekeeper to every edge
gate = overall_df.set_index(
    ["State","Contrast"]
)["Permutation_P_Holm_within_state"].to_dict()

edges_df["Multivariate_Perm_Holm"] = [
    gate[(s,c)]
    for s,c in zip(edges_df["State"],edges_df["Contrast"])
]

edges_df["Multivariate_Gate_Passed"] = (
    edges_df["Multivariate_Perm_Holm"]<.05
)

edges_df["Primary_Edge_Significant"] = (
    edges_df["Multivariate_Gate_Passed"] &
    (edges_df["Edge_q_BH_within_contrast"]<.05)
)

# ============================================================
# 14. Descriptive group profiles for plotting
# ============================================================
profile_rows = []

for state in [1,2]:
    for condition in ["HC","M0","M1"]:
        ids,x = condition_matrix(state,condition)

        for j,e in enumerate(EDGE_NAMES):
            profile_rows.append({
                "State":state,
                "Condition":condition,
                "Edge":e,
                "N":len(x),
                "Mean":np.mean(x[:,j]),
                "SD":np.std(x[:,j],ddof=1),
                "Median":np.median(x[:,j])
            })

profile_df = pd.DataFrame(profile_rows)

# ============================================================
# 15. Save tables
# ============================================================
OVERALL_FILE = OUT_DIR/"Window50_StateSpecific_HC_M0_M1_multivariate_contrasts.csv"
EDGE_FILE = OUT_DIR/"Window50_StateSpecific_HC_M0_M1_27edge_followup.csv"
PROFILE_FILE = OUT_DIR/"Window50_StateSpecific_HC_M0_M1_group_profiles.csv"
SUBJECT_FILE = OUT_DIR/"Window50_StateSpecific_HC_M0_M1_included_subjects.csv"
PRIMARY_FILE = OUT_DIR/"Window50_StateSpecific_HC_M0_M1_primary_significant_edges.csv"
NOMINAL_FILE = OUT_DIR/"Window50_StateSpecific_HC_M0_M1_nominal_edges.csv"
FDR_FILE = OUT_DIR/"Window50_StateSpecific_HC_M0_M1_withinContrast_FDR_edges.csv"

overall_df.to_csv(OVERALL_FILE,index=False)
edges_df.to_csv(EDGE_FILE,index=False)
profile_df.to_csv(PROFILE_FILE,index=False)
subjects_df.to_csv(SUBJECT_FILE,index=False)

edges_df[
    edges_df["Primary_Edge_Significant"]
].to_csv(PRIMARY_FILE,index=False)

edges_df[
    edges_df["Univariate_P"]<.05
].to_csv(NOMINAL_FILE,index=False)

edges_df[
    edges_df["Edge_q_BH_within_contrast"]<.05
].to_csv(FDR_FILE,index=False)

for state in [1,2]:
    overall_df[
        overall_df["State"]==state
    ].to_csv(
        OUT_DIR/f"Window50_State{state}_multivariate_contrasts.csv",
        index=False
    )

    edges_df[
        edges_df["State"]==state
    ].to_csv(
        OUT_DIR/f"Window50_State{state}_27edge_followup.csv",
        index=False
    )

# ============================================================
# 16. Plot 1: HC/M0/M1 group-profile heatmaps
# ============================================================
def plot_group_heatmap(state):
    d = profile_df[profile_df["State"]==state]

    mat = (
        d.pivot(index="Edge",columns="Condition",values="Mean")
        .reindex(index=EDGE_NAMES,columns=["HC","M0","M1"])
    )
    nmat = (
        d.pivot(index="Edge",columns="Condition",values="N")
        .reindex(index=EDGE_NAMES,columns=["HC","M0","M1"])
    )

    a = mat.to_numpy(float)
    ns = nmat.to_numpy(float)

    fig,ax = plt.subplots(figsize=(7.2,10.2))
    im = ax.imshow(a,aspect="auto",cmap="viridis")

    ax.set_xticks(range(3))
    ax.set_xticklabels(["HC","M0","M1"])
    ax.set_yticks(range(27))
    ax.set_yticklabels(EDGE_NAMES,fontsize=8)
    ax.set_ylabel("MetS-related NBS edge")
    ax.set_title(
        f"State {state}: state-specific median connectivity profiles\n"
        "Group mean Fisher-z"
    )

    for i in range(27):
        for j in range(3):
            ax.text(
                j,i,
                f"{a[i,j]:.3f}\n(n={int(ns[i,j])})",
                ha="center",va="center",fontsize=6.2
            )

    cb = fig.colorbar(im,ax=ax,fraction=.04,pad=.03)
    cb.set_label("Mean subject-specific median Fisher-z")
    fig.tight_layout()

    fig.savefig(
        FIG_DIR/f"Figure_State{state}_HC_M0_M1_group_profiles_heatmap.pdf",
        bbox_inches="tight"
    )
    fig.savefig(
        FIG_DIR/f"Figure_State{state}_HC_M0_M1_group_profiles_heatmap.png",
        dpi=300,bbox_inches="tight"
    )
    plt.close(fig)

# ============================================================
# 17. Plot 2: contrast mean-difference heatmaps
# ============================================================
def plot_contrast_heatmap(state):
    d = edges_df[edges_df["State"]==state].copy()

    mat = (
        d.pivot(index="Edge",columns="Contrast",values="Mean_Difference")
        .reindex(index=EDGE_NAMES,columns=CONTRAST_ORDER)
    )

    sig = (
        d.pivot(index="Edge",columns="Contrast",values="Primary_Edge_Significant")
        .reindex(index=EDGE_NAMES,columns=CONTRAST_ORDER)
    )

    a = mat.to_numpy(float)
    smat = sig.fillna(False).to_numpy(bool)

    vmax = np.nanmax(np.abs(a))
    if not np.isfinite(vmax) or vmax==0:
        vmax = 1

    fig,ax = plt.subplots(figsize=(8,10.2))
    im = ax.imshow(
        a,aspect="auto",
        cmap="coolwarm",
        vmin=-vmax,vmax=vmax
    )

    ax.set_xticks(range(3))
    ax.set_xticklabels(
        ["M0 - HC","M1 - HC","M1 - M0"]
    )
    ax.set_yticks(range(27))
    ax.set_yticklabels(EDGE_NAMES,fontsize=8)
    ax.set_ylabel("MetS-related NBS edge")
    ax.set_title(
        f"State {state}: state-specific edge contrasts\n"
        "Raw mean Fisher-z differences"
    )

    for i in range(27):
        for j in range(3):
            star = "*" if smat[i,j] else ""
            ax.text(
                j,i,
                f"{a[i,j]:+.3f}{star}",
                ha="center",va="center",fontsize=7
            )

    cb = fig.colorbar(im,ax=ax,fraction=.04,pad=.03)
    cb.set_label("Mean Fisher-z difference")
    fig.text(
        .5,.01,
        "* multivariate Holm-corrected permutation gate passed "
        "and edge BH-FDR q < 0.05",
        ha="center",fontsize=8
    )
    fig.tight_layout(rect=[0,.03,1,1])

    fig.savefig(
        FIG_DIR/f"Figure_State{state}_HC_M0_M1_contrast_heatmap.pdf",
        bbox_inches="tight"
    )
    fig.savefig(
        FIG_DIR/f"Figure_State{state}_HC_M0_M1_contrast_heatmap.png",
        dpi=300,bbox_inches="tight"
    )
    plt.close(fig)

# ============================================================
# 18. Plot 3: forest plot for every contrast
# ============================================================
def plot_forest(state,contrast):
    d = edges_df[
        (edges_df["State"]==state)&
        (edges_df["Contrast"]==contrast)
    ].copy()

    d = d.sort_values("Mean_Difference")
    y = np.arange(len(d))

    x = d["Mean_Difference"].to_numpy(float)
    lo = d["CI95_Low"].to_numpy(float)
    hi = d["CI95_High"].to_numpy(float)
    primary = d["Primary_Edge_Significant"].to_numpy(bool)

    labels = []
    for e,sig in zip(d["Edge"],primary):
        labels.append(f"{e} *" if sig else e)

    gate_p = float(d["Multivariate_Perm_Holm"].iloc[0])

    fig,ax = plt.subplots(figsize=(8.3,10.5))

    ax.errorbar(
        x,y,
        xerr=np.vstack([x-lo,hi-x]),
        fmt="o",
        capsize=2.5,
        linewidth=1
    )

    ax.axvline(0,linestyle="--",linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels,fontsize=8)
    ax.set_xlabel(f"Mean Fisher-z difference ({contrast})")
    ax.set_ylabel("")
    ax.set_title(
        f"State {state}: {contrast}\n"
        f"27-edge effects with 95% CI | "
        f"multivariate permutation Holm P = {gate_p:.4g}"
    )

    fig.text(
        .5,.01,
        "* primary significant edge: multivariate gate + "
        "within-contrast BH-FDR q < 0.05",
        ha="center",fontsize=8
    )

    fig.tight_layout(rect=[0,.03,1,1])

    safe = contrast.replace("-","_minus_")

    fig.savefig(
        FIG_DIR/f"Figure_State{state}_{safe}_27edge_forest.pdf",
        bbox_inches="tight"
    )
    fig.savefig(
        FIG_DIR/f"Figure_State{state}_{safe}_27edge_forest.png",
        dpi=300,bbox_inches="tight"
    )
    plt.close(fig)

# Generate all figures
for state in [1,2]:
    plot_group_heatmap(state)
    plot_contrast_heatmap(state)

    for contrast in CONTRAST_ORDER:
        plot_forest(state,contrast)

# ============================================================
# 19. Console summary
# ============================================================
print("\n============================================================")
print("MULTIVARIATE CONTRAST RESULTS")
print("============================================================")

show_cols = [
    "State","Contrast","Design","N1","N2",
    "Hotelling_T2","F","df1","df2",
    "Parametric_P",
    "Permutation_P",
    "Permutation_P_Holm_within_state",
    "Permutation_P_Holm_all6",
    "Covariance_Condition_Number"
]

print(
    overall_df[show_cols].to_string(
        index=False,
        float_format=lambda x:f"{x:.6g}"
    )
)

print("\n============================================================")
print("EDGE-WISE SUMMARY")
print("============================================================")

for state in [1,2]:
    for contrast in CONTRAST_ORDER:

        d = edges_df[
            (edges_df["State"]==state)&
            (edges_df["Contrast"]==contrast)
        ].copy()

        gate = d["Multivariate_Perm_Holm"].iloc[0]

        print(f"\nState {state} | {contrast}")
        print(f"Multivariate permutation Holm P = {gate:.6g}")
        print(
            f"Nominal edge P<.05 = "
            f"{(d['Univariate_P']<.05).sum()}/27"
        )
        print(
            f"Within-contrast BH-FDR q<.05 = "
            f"{(d['Edge_q_BH_within_contrast']<.05).sum()}/27"
        )
        print(
            f"Primary gatekeeper+FDR significant = "
            f"{d['Primary_Edge_Significant'].sum()}/27"
        )

        top = d.sort_values("Univariate_P").head(8)

        print(
            top[
                [
                    "Edge",
                    "N_Group1","N_Group2",
                    "Group1_Mean","Group2_Mean",
                    "Mean_Difference",
                    "Effect_Size",
                    "Univariate_P",
                    "Edge_q_BH_within_contrast",
                    "Robust_P"
                ]
            ].to_string(
                index=False,
                float_format=lambda x:f"{x:.6g}"
            )
        )

print("\n============================================================")
print("FILES SAVED")
print("============================================================")
print("Output:")
print(OUT_DIR)

print("\nMain tables:")
for p in [
    OVERALL_FILE,EDGE_FILE,PROFILE_FILE,
    PRIMARY_FILE,NOMINAL_FILE,FDR_FILE
]:
    print(" ",p.name)

print("\nFigures:")
for p in sorted(FIG_DIR.glob("*.png")):
    print(" ",p.name)

print("\nINTERPRETATION NOTES")
print("1. HC comparisons are secondary characterization because")
print("   the 27 edges were selected from static M0-vs-HC NBS.")
print("2. State 2 comparisons are conditional on State 2 visitation.")
print("3. Do not call M1 'normalized' merely because M1-HC is nonsignificant.")
print("4. Primary edge inference requires:")
print("   multivariate permutation Holm P<0.05 AND edge BH-FDR q<0.05.")
