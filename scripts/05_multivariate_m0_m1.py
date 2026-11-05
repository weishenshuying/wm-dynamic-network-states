#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Window50 state-specific 27-edge longitudinal analysis: M0 vs M1
================================================================
Primary logic inspired by the reference paper:
State-specific multivariate screening -> edge-wise follow-up -> FDR

Adaptation for the present longitudinal design:
- State 1 and State 2 analysed independently
- Subject-specific state profile = element-wise median connectivity
- M0 vs M1 is paired
- Overall 27-dimensional effect:
    1) Paired Hotelling T^2
    2) 10,000 subject-level sign-flip permutations
- Edge-wise follow-up:
    paired t-test
    Wilcoxon signed-rank
    Cohen's dz
    mean paired delta + 95% CI
    BH-FDR within each state
- Combined 54-edge BH-FDR is also saved as sensitivity analysis

Delta definition:
    Delta = M1 - M0

IMPORTANT:
State 2 analysis includes only subjects who visited State 2
at BOTH M0 and M1.
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
OUT_DIR = PROFILE_DIR/"Statistics_M0_M1"
FIG_DIR = OUT_DIR/"Figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

N_PERM = 10000
SEED = 2026

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Missing input file:\n{INPUT_FILE}")

# ============================================================
# 2. Fixed 27 edges
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

# ============================================================
# 3. BH-FDR
# ============================================================
def bh_fdr(pvals):
    pvals = np.asarray(pvals,dtype=float)
    q = np.full(len(pvals),np.nan)
    ok = np.isfinite(pvals)
    p = pvals[ok]
    if len(p)==0:
        return q

    order = np.argsort(p)
    ranked = p[order]
    m = len(ranked)

    adj = ranked*m/np.arange(1,m+1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj,0,1)

    tmp = np.empty(m)
    tmp[order] = adj
    q[ok] = tmp
    return q

# ============================================================
# 4. Paired Hotelling T2
# ============================================================
def hotelling_paired(delta):
    """
    delta: n x p matrix, each row = M1-M0 for one subject.
    Tests multivariate mean(delta)=0.

    T2 = n * mean' S^-1 mean
    F = (n-p)/(p*(n-1))*T2
    df1=p, df2=n-p
    """
    delta = np.asarray(delta,float)
    n,p = delta.shape

    if n <= p:
        raise ValueError(f"Hotelling T2 requires n>p; n={n}, p={p}")

    mean_delta = delta.mean(axis=0)
    S = np.cov(delta,rowvar=False,ddof=1)

    rank = np.linalg.matrix_rank(S)
    cond = np.linalg.cond(S)

    # inverse if possible; pseudoinverse only as numerical fallback
    try:
        Sinv = np.linalg.inv(S)
        inverse_method = "inverse"
    except np.linalg.LinAlgError:
        Sinv = np.linalg.pinv(S)
        inverse_method = "pseudoinverse"

    T2 = float(n * mean_delta @ Sinv @ mean_delta)
    F = float((n-p)/(p*(n-1))*T2)
    df1 = p
    df2 = n-p
    p_parametric = float(stats.f.sf(F,df1,df2))

    return {
        "N":n,"P":p,"T2":T2,"F":F,
        "df1":df1,"df2":df2,
        "P_parametric":p_parametric,
        "Covariance_Rank":rank,
        "Covariance_Condition_Number":cond,
        "Inverse_Method":inverse_method
    }

# ============================================================
# 5. Sign-flip permutation of paired Hotelling T2
# ============================================================
def permutation_hotelling(delta,n_perm=10000,seed=2026):
    """
    Subject-level sign flip:
    each participant's entire 27-dimensional delta vector receives +1/-1.
    This preserves covariance structure across edges within each subject.
    """
    delta = np.asarray(delta,float)
    rng = np.random.default_rng(seed)

    obs = hotelling_paired(delta)["T2"]
    exceed = 0

    n = delta.shape[0]

    for _ in range(n_perm):
        signs = rng.choice([-1.0,1.0],size=n)
        dp = delta*signs[:,None]

        mean_delta = dp.mean(axis=0)
        S = np.cov(dp,rowvar=False,ddof=1)

        try:
            Sinv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            Sinv = np.linalg.pinv(S)

        T2_perm = float(n * mean_delta @ Sinv @ mean_delta)

        if T2_perm >= obs-1e-12:
            exceed += 1

    # +1 correction prevents zero permutation p-values
    p_perm = (exceed+1)/(n_perm+1)
    return obs,p_perm,exceed

# ============================================================
# 6. Paired Cohen's dz
# ============================================================
def cohens_dz(delta):
    delta = np.asarray(delta,float)
    delta = delta[np.isfinite(delta)]

    if len(delta)<2:
        return np.nan

    sd = np.std(delta,ddof=1)
    if sd==0:
        return np.nan

    return np.mean(delta)/sd

# ============================================================
# 7. Edge-wise paired analysis
# ============================================================
def edgewise_tests(m0,m1,edge_names,state):
    rows = []

    for j,edge in enumerate(edge_names):
        x0 = m0[:,j]
        x1 = m1[:,j]
        ok = np.isfinite(x0)&np.isfinite(x1)

        a = x0[ok]
        b = x1[ok]
        d = b-a
        n = len(d)

        mean0 = np.mean(a)
        sd0 = np.std(a,ddof=1)
        mean1 = np.mean(b)
        sd1 = np.std(b,ddof=1)

        mean_delta = np.mean(d)
        sd_delta = np.std(d,ddof=1)
        se_delta = sd_delta/np.sqrt(n)

        tcrit = stats.t.ppf(.975,n-1)
        ci_low = mean_delta-tcrit*se_delta
        ci_high = mean_delta+tcrit*se_delta

        tstat,tp = stats.ttest_rel(b,a,nan_policy="omit")

        try:
            if np.allclose(d,0):
                wstat,wp = 0.0,1.0
            else:
                wstat,wp = stats.wilcoxon(
                    b,a,
                    zero_method="wilcox",
                    alternative="two-sided",
                    method="auto"
                )
        except Exception:
            wstat,wp = np.nan,np.nan

        dz = cohens_dz(d)

        rows.append({
            "State":state,
            "Edge":edge,
            "N_Pairs":n,
            "M0_Mean":mean0,
            "M0_SD":sd0,
            "M1_Mean":mean1,
            "M1_SD":sd1,
            "Delta_M1_minus_M0":mean_delta,
            "Delta_SD":sd_delta,
            "Delta_CI95_Low":ci_low,
            "Delta_CI95_High":ci_high,
            "Cohens_dz":dz,
            "Paired_t":tstat,
            "Paired_t_p":tp,
            "Wilcoxon_W":wstat,
            "Wilcoxon_p":wp,
            "Direction":"M1 > M0" if mean_delta>0 else
                        "M1 < M0" if mean_delta<0 else "No change"
        })

    out = pd.DataFrame(rows)

    out["Paired_t_q_BH_within_state"] = bh_fdr(out["Paired_t_p"])
    out["Wilcoxon_q_BH_within_state"] = bh_fdr(out["Wilcoxon_p"])

    return out

# ============================================================
# 8. Build state-specific paired matrices
# ============================================================
df = pd.read_csv(INPUT_FILE)

required = [
    "Subject","Group","Session","Condition","State",
    "EdgeName","Median_FisherZ"
]
missing = [c for c in required if c not in df.columns]
if missing:
    raise RuntimeError(f"Input missing columns: {missing}")

df = df[df["Group"]=="MetS"].copy()

def state_pair_matrices(state):
    ds = df[df["State"]==state].copy()

    wide = ds.pivot(
        index="Subject",
        columns=["Session","EdgeName"],
        values="Median_FisherZ"
    )

    # Ensure all expected columns exist
    m0_cols = [("M0",e) for e in EDGE_NAMES]
    m1_cols = [("M1",e) for e in EDGE_NAMES]

    for c in m0_cols+m1_cols:
        if c not in wide.columns:
            wide[c] = np.nan

    m0 = wide[m0_cols].to_numpy(float)
    m1 = wide[m1_cols].to_numpy(float)

    complete = np.isfinite(m0).all(axis=1) & np.isfinite(m1).all(axis=1)

    subjects = wide.index.to_numpy()[complete]
    m0 = m0[complete,:]
    m1 = m1[complete,:]

    return subjects,m0,m1

# ============================================================
# 9. Run State 1 and State 2
# ============================================================
overall_rows = []
edge_results = []
paired_subject_rows = []

print("\n============================================================")
print("STATE-SPECIFIC MULTIVARIATE M0-M1 ANALYSIS")
print("============================================================")

for state in [1,2]:
    subjects,m0,m1 = state_pair_matrices(state)
    delta = m1-m0

    print(f"\n---------------- STATE {state} ----------------")
    print(f"Complete M0-M1 pairs = {len(subjects)}")
    print(f"Dimension             = {delta.shape[1]} edges")

    # Save exact included subjects
    for s in subjects:
        paired_subject_rows.append({
            "State":state,
            "Subject":s
        })

    # Parametric Hotelling
    h = hotelling_paired(delta)

    print("\nPaired Hotelling T2:")
    print(f" T2 = {h['T2']:.6f}")
    print(f" F({h['df1']},{h['df2']}) = {h['F']:.6f}")
    print(f" parametric p = {h['P_parametric']:.8g}")
    print(f" covariance rank = {h['Covariance_Rank']}/{P}")
    print(f" condition number = {h['Covariance_Condition_Number']:.4g}")

    # Permutation
    print(f"\nRunning {N_PERM:,} sign-flip permutations ...")
    obs,pperm,exceed = permutation_hotelling(
        delta,n_perm=N_PERM,seed=SEED+state
    )

    print(f" permutation p = {pperm:.8g}")
    print(f" exceedances = {exceed}/{N_PERM}")

    overall_rows.append({
        "State":state,
        "N_Pairs":h["N"],
        "N_Edges":h["P"],
        "Hotelling_T2":h["T2"],
        "F":h["F"],
        "df1":h["df1"],
        "df2":h["df2"],
        "Parametric_P":h["P_parametric"],
        "Permutation_N":N_PERM,
        "Permutation_P":pperm,
        "Permutation_Exceedances":exceed,
        "Covariance_Rank":h["Covariance_Rank"],
        "Covariance_Condition_Number":h["Covariance_Condition_Number"],
        "Inverse_Method":h["Inverse_Method"],
        "Gatekeeper_Parametric_Plt005":h["P_parametric"]<.05,
        "Gatekeeper_Permutation_Plt005":pperm<.05
    })

    # Edge-wise follow-up is calculated for transparency;
    # interpretation should respect the multivariate gatekeeper.
    er = edgewise_tests(m0,m1,EDGE_NAMES,state)
    edge_results.append(er)

overall_df = pd.DataFrame(overall_rows)
edges_df = pd.concat(edge_results,ignore_index=True)
subjects_df = pd.DataFrame(paired_subject_rows)

# ============================================================
# 10. 54-test FDR sensitivity analysis
# ============================================================
edges_df["Paired_t_q_BH_all54"] = bh_fdr(edges_df["Paired_t_p"])
edges_df["Wilcoxon_q_BH_all54"] = bh_fdr(edges_df["Wilcoxon_p"])

# Add gatekeeper information to every edge
gate_map_param = dict(zip(overall_df["State"],
                          overall_df["Gatekeeper_Parametric_Plt005"]))
gate_map_perm = dict(zip(overall_df["State"],
                         overall_df["Gatekeeper_Permutation_Plt005"]))

edges_df["Overall_Parametric_Gate_Passed"] = edges_df["State"].map(gate_map_param)
edges_df["Overall_Permutation_Gate_Passed"] = edges_df["State"].map(gate_map_perm)

# Primary inferential flag:
# overall permutation p<.05 AND within-state paired-t BH q<.05
edges_df["Primary_Followup_Significant"] = (
    edges_df["Overall_Permutation_Gate_Passed"] &
    (edges_df["Paired_t_q_BH_within_state"]<.05)
)

# ============================================================
# 11. Save tables
# ============================================================
OVERALL_FILE = OUT_DIR/"Window50_StateSpecific_M0_M1_multivariate_overall.csv"
EDGE_FILE = OUT_DIR/"Window50_StateSpecific_M0_M1_27edge_followup.csv"
SUBJECT_FILE = OUT_DIR/"Window50_StateSpecific_M0_M1_included_subjects.csv"

overall_df.to_csv(OVERALL_FILE,index=False)
edges_df.to_csv(EDGE_FILE,index=False)
subjects_df.to_csv(SUBJECT_FILE,index=False)

# Separate state tables
for state in [1,2]:
    edges_df[edges_df["State"]==state].to_csv(
        OUT_DIR/f"Window50_State{state}_M0_M1_27edge_followup.csv",
        index=False
    )

# Significant / nominal subsets
nominal_df = edges_df[edges_df["Paired_t_p"]<.05].copy()
fdr_df = edges_df[edges_df["Paired_t_q_BH_within_state"]<.05].copy()
primary_df = edges_df[edges_df["Primary_Followup_Significant"]].copy()

nominal_df.to_csv(
    OUT_DIR/"Window50_StateSpecific_M0_M1_nominal_Plt005.csv",
    index=False
)
fdr_df.to_csv(
    OUT_DIR/"Window50_StateSpecific_M0_M1_withinState_FDR_q_lt005.csv",
    index=False
)
primary_df.to_csv(
    OUT_DIR/"Window50_StateSpecific_M0_M1_primary_gatekeeper_plus_FDR.csv",
    index=False
)

# ============================================================
# 12. Forest plots of paired edge changes
# ============================================================
def forest_plot(d,state):
    d = d.copy().sort_values("Delta_M1_minus_M0")
    y = np.arange(len(d))

    fig,ax = plt.subplots(figsize=(8.0,10.5))

    x = d["Delta_M1_minus_M0"].to_numpy()
    lo = d["Delta_CI95_Low"].to_numpy()
    hi = d["Delta_CI95_High"].to_numpy()

    ax.errorbar(
        x,y,
        xerr=np.vstack([x-lo,hi-x]),
        fmt="o",
        capsize=2.5,
        linewidth=1
    )

    ax.axvline(0,linestyle="--",linewidth=1)

    ax.set_yticks(y)
    ax.set_yticklabels(d["Edge"],fontsize=8)
    ax.set_xlabel("Paired change in median Fisher-z (M1 - M0)")
    ax.set_ylabel("")
    ax.set_title(
        f"State {state}: M1 - M0 state-specific edge changes\n"
        f"Mean paired change with 95% CI"
    )

    # Mark within-state FDR significance
    for yi,(_,row) in enumerate(d.iterrows()):
        if row["Paired_t_q_BH_within_state"]<.05:
            side = hi[yi] if x[yi]>=0 else lo[yi]
            ax.text(
                side,yi,"  *",
                va="center",
                ha="left" if x[yi]>=0 else "right",
                fontsize=11
            )

    fig.tight_layout()

    fig.savefig(
        FIG_DIR/f"Figure_Window50_State{state}_M0_M1_27edge_delta_forest.pdf",
        bbox_inches="tight"
    )
    fig.savefig(
        FIG_DIR/f"Figure_Window50_State{state}_M0_M1_27edge_delta_forest.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)

for state in [1,2]:
    forest_plot(edges_df[edges_df["State"]==state],state)

# ============================================================
# 13. Console summary
# ============================================================
print("\n============================================================")
print("MULTIVARIATE OVERALL RESULTS")
print("============================================================")
print(
    overall_df[
        [
            "State","N_Pairs","Hotelling_T2","F","df1","df2",
            "Parametric_P","Permutation_P",
            "Covariance_Rank","Covariance_Condition_Number"
        ]
    ].to_string(index=False,float_format=lambda x:f"{x:.6g}")
)

print("\n============================================================")
print("EDGE-WISE SUMMARY")
print("============================================================")

for state in [1,2]:
    d = edges_df[edges_df["State"]==state].copy()

    print(f"\nState {state}:")
    print(f" nominal paired-t p<.05 = {(d['Paired_t_p']<.05).sum()}/27")
    print(f" within-state BH-FDR q<.05 = "
          f"{(d['Paired_t_q_BH_within_state']<.05).sum()}/27")
    print(f" all-54 BH-FDR q<.05 = "
          f"{(d['Paired_t_q_BH_all54']<.05).sum()}/27")

    top = d.sort_values("Paired_t_p").head(10)

    print("\nTop edges by paired-t p:")
    print(
        top[
            [
                "Edge","N_Pairs","M0_Mean","M1_Mean",
                "Delta_M1_minus_M0","Cohens_dz",
                "Paired_t_p","Paired_t_q_BH_within_state",
                "Wilcoxon_p"
            ]
        ].to_string(index=False,float_format=lambda x:f"{x:.6g}")
    )

print("\n============================================================")
print("FILES SAVED")
print("============================================================")
print("Output folder:")
print(OUT_DIR)

print("\nMain files:")
print(OVERALL_FILE.name)
print(EDGE_FILE.name)
print(SUBJECT_FILE.name)

print("\nFigures:")
for p in sorted(FIG_DIR.glob("*.png")):
    print(p.name)

print("\nInterpretation rule:")
print("Primary edge-wise inference should be interpreted only if")
print("the corresponding state-level multivariate gatekeeper passes.")
print("Primary robust gatekeeper here = sign-flip permutation P < 0.05.")
