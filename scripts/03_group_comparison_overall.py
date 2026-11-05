import os
import sys
import numpy as np
import pandas as pd

from scipy import stats

try:
    import statsmodels.formula.api as smf
except ImportError:
    raise RuntimeError(
        "statsmodels 未安装。请先运行：python3 -m pip install statsmodels"
    )


# ==============================================================================
# 1. Paths
# ==============================================================================

ROOT = os.environ.get(
    "WM_DYNAMIC_ROOT",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
    ),
)

HC_FILE = os.path.join(
    ROOT,
    "HC_window_050_step_1",
    "Projection_to_patient_K2",
    "HC_window50_projected_temporal_metrics.csv"
)

PATIENT_FILE = os.path.join(
    ROOT,
    "Primary_Window50_StateAnalysis",
    "Window50_subject_dynamic_metrics.csv"
)

OUTPUT_DIR = os.path.join(
    ROOT,
    "HC_M0_M1_Window50_Comparison"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==============================================================================
# 2. Metrics
# ==============================================================================

METRICS = [
    "DynamicMean",
    "Variability_SD",
    "P90_P10",
    "FO_State2",
    "MDT_State1_windows",
    "MDT_State2_windows",
    "Switching_Rate"
]

METRIC_LABELS = {
    "DynamicMean":
        "Dynamic network mean strength",

    "Variability_SD":
        "Dynamic network variability (SD)",

    "P90_P10":
        "Dynamic range (P90-P10)",

    "FO_State2":
        "High-state fractional occupancy",

    "MDT_State1_windows":
        "Low-state mean dwell time",

    "MDT_State2_windows":
        "High-state mean dwell time",

    "Switching_Rate":
        "State switching rate"
}


# ==============================================================================
# 3. Holm correction
# ==============================================================================

def holm_adjust(pvals):

    pvals = np.asarray(pvals, dtype=float)

    result = np.full(
        len(pvals),
        np.nan
    )

    valid = np.where(
        np.isfinite(pvals)
    )[0]

    if len(valid) == 0:
        return result

    p = pvals[valid]

    order = np.argsort(p)

    sorted_p = p[order]

    m = len(sorted_p)

    adjusted_sorted = np.empty(m)

    running_max = 0.0

    for i, pv in enumerate(sorted_p):

        adj = (m - i) * pv

        running_max = max(
            running_max,
            adj
        )

        adjusted_sorted[i] = min(
            running_max,
            1.0
        )

    adjusted = np.empty(m)

    adjusted[order] = adjusted_sorted

    result[valid] = adjusted

    return result


# ==============================================================================
# 4. Read files
# ==============================================================================

print("=" * 110)
print("HC vs M0 vs M1 DYNAMIC ANALYSIS")
print("WINDOW = 50 TR | STEP = 1 TR | FIXED PATIENT K=2 STATES")
print("=" * 110)

print("\nHC file:")
print(HC_FILE)

print("\nPatient file:")
print(PATIENT_FILE)


if not os.path.exists(HC_FILE):

    raise FileNotFoundError(
        f"HC file not found:\n{HC_FILE}"
    )


if not os.path.exists(PATIENT_FILE):

    raise FileNotFoundError(
        f"Patient file not found:\n{PATIENT_FILE}"
    )


hc = pd.read_csv(
    HC_FILE
)

pat = pd.read_csv(
    PATIENT_FILE
)


# ==============================================================================
# 5. QC
# ==============================================================================

print("\nHC shape:", hc.shape)
print("Patient shape:", pat.shape)

if len(hc) != 48:

    raise RuntimeError(
        f"Expected 48 HC, found {len(hc)}"
    )


if "Session" not in pat.columns:

    raise RuntimeError(
        "Patient file has no Session column."
    )


print("\nPatient sessions:")
print(
    pat["Session"]
    .value_counts()
    .sort_index()
)


for metric in METRICS:

    if metric not in hc.columns:
        raise RuntimeError(
            f"HC missing metric: {metric}"
        )

    if metric not in pat.columns:
        raise RuntimeError(
            f"Patient missing metric: {metric}"
        )


# ==============================================================================
# 6. Build unified long-format table
# ==============================================================================

hc_long = hc[
    ["Subject"] + METRICS + ["Visit_State2"]
].copy()

hc_long["Condition"] = "HC"

# unique model subject ID
hc_long["ModelSubject"] = (
    "HC_" + hc_long["Subject"].astype(str)
)


pat_long = pat[
    ["Subject", "Session"] + METRICS + ["Visit_State2"]
].copy()

pat_long["Condition"] = pat_long["Session"]

pat_long["ModelSubject"] = (
    "PAT_" + pat_long["Subject"].astype(str)
)


long_df = pd.concat(
    [
        hc_long.drop(
            columns=[],
            errors="ignore"
        ),
        pat_long
    ],
    ignore_index=True,
    sort=False
)


# Keep only HC/M0/M1
long_df = long_df[
    long_df["Condition"].isin(
        ["HC", "M0", "M1"]
    )
].copy()

long_df["Condition"] = pd.Categorical(
    long_df["Condition"],
    categories=["HC", "M0", "M1"],
    ordered=True
)


combined_file = os.path.join(
    OUTPUT_DIR,
    "Window50_HC_M0_M1_combined_subject_metrics.csv"
)

long_df.to_csv(
    combined_file,
    index=False
)


# ==============================================================================
# 7. Descriptive statistics
# ==============================================================================

desc_rows = []

for metric in METRICS:

    for condition in ["HC", "M0", "M1"]:

        x = long_df.loc[
            long_df["Condition"] == condition,
            metric
        ].dropna().astype(float)

        desc_rows.append({
            "Metric":
                metric,

            "Metric_Label":
                METRIC_LABELS[metric],

            "Condition":
                condition,

            "N":
                len(x),

            "Mean":
                x.mean(),

            "SD":
                x.std(ddof=1),

            "Median":
                x.median(),

            "Q1":
                x.quantile(0.25),

            "Q3":
                x.quantile(0.75),

            "Min":
                x.min(),

            "Max":
                x.max()
        })


desc_df = pd.DataFrame(
    desc_rows
)


desc_file = os.path.join(
    OUTPUT_DIR,
    "Window50_HC_M0_M1_descriptive_statistics.csv"
)

desc_df.to_csv(
    desc_file,
    index=False
)


# ==============================================================================
# 8. LMM fitting function
# ==============================================================================

def fit_lmm(data, metric):

    d = data[
        [
            "ModelSubject",
            "Condition",
            metric
        ]
    ].dropna().copy()

    d = d.rename(
        columns={
            metric: "Y"
        }
    )

    formula = (
        "Y ~ C(Condition, Treatment(reference='HC'))"
    )

    model = smf.mixedlm(
        formula,
        data=d,
        groups=d["ModelSubject"]
    )

    # try several optimizers
    last_error = None

    for method in [
        "lbfgs",
        "bfgs",
        "cg",
        "powell"
    ]:

        try:

            result = model.fit(
                reml=False,
                method=method,
                maxiter=2000,
                disp=False
            )

            if result.converged:
                return result, d, method

        except Exception as e:

            last_error = e

    # return final attempt if possible
    try:

        result = model.fit(
            reml=False,
            method="powell",
            maxiter=5000,
            disp=False
        )

        return result, d, "powell"

    except Exception:

        raise RuntimeError(
            f"LMM failed for {metric}: {last_error}"
        )


# ==============================================================================
# 9. Main LMM analyses
# ==============================================================================

omnibus_rows = []
pairwise_rows = []


for metric in METRICS:

    print("\n" + "=" * 100)
    print(METRIC_LABELS[metric])
    print("=" * 100)

    result, d, optimizer = fit_lmm(
        long_df,
        metric
    )

    pnames = list(
        result.params.index
    )

    m0_name = (
        "C(Condition, Treatment(reference='HC'))[T.M0]"
    )

    m1_name = (
        "C(Condition, Treatment(reference='HC'))[T.M1]"
    )

    if (
        m0_name not in pnames
        or
        m1_name not in pnames
    ):

        raise RuntimeError(
            f"Cannot find condition coefficients for {metric}\n"
            f"Parameters: {pnames}"
        )


    # --------------------------------------------------------------------------
    # Omnibus Wald test:
    # H0: beta_M0 = beta_M1 = 0
    # --------------------------------------------------------------------------

    beta = np.array([
        result.params[m0_name],
        result.params[m1_name]
    ])

    cov_all = result.cov_params()

    cov2 = cov_all.loc[
        [m0_name, m1_name],
        [m0_name, m1_name]
    ].to_numpy()

    try:

        chi2 = float(
            beta.T
            @
            np.linalg.inv(cov2)
            @
            beta
        )

        omnibus_p = stats.chi2.sf(
            chi2,
            df=2
        )

    except np.linalg.LinAlgError:

        chi2 = np.nan
        omnibus_p = np.nan


    omnibus_rows.append({
        "Metric":
            metric,

        "Metric_Label":
            METRIC_LABELS[metric],

        "N_Observations":
            len(d),

        "N_Subjects":
            d["ModelSubject"].nunique(),

        "Optimizer":
            optimizer,

        "Converged":
            bool(result.converged),

        "Wald_Chi2":
            chi2,

        "df":
            2,

        "Omnibus_p":
            omnibus_p,

        "Random_Intercept_Var":
            float(result.cov_re.iloc[0, 0]),

        "Residual_Var":
            float(result.scale)
    })


    # --------------------------------------------------------------------------
    # Pairwise LMM contrasts
    # --------------------------------------------------------------------------

    b0 = float(
        result.params[m0_name]
    )

    b1 = float(
        result.params[m1_name]
    )

    v00 = float(
        cov_all.loc[
            m0_name,
            m0_name
        ]
    )

    v11 = float(
        cov_all.loc[
            m1_name,
            m1_name
        ]
    )

    v01 = float(
        cov_all.loc[
            m0_name,
            m1_name
        ]
    )


    contrasts = [
        (
            "M0 - HC",
            b0,
            v00
        ),

        (
            "M1 - HC",
            b1,
            v11
        ),

        (
            "M1 - M0",
            b1 - b0,
            v11 + v00 - 2 * v01
        )
    ]


    metric_pair_rows = []

    for contrast_name, estimate, variance in contrasts:

        se = (
            np.sqrt(variance)
            if variance >= 0
            else np.nan
        )

        z = (
            estimate / se
            if (
                np.isfinite(se)
                and se > 0
            )
            else np.nan
        )

        p = (
            2
            *
            stats.norm.sf(
                abs(z)
            )
            if np.isfinite(z)
            else np.nan
        )

        ci_low = (
            estimate
            -
            1.96 * se
            if np.isfinite(se)
            else np.nan
        )

        ci_high = (
            estimate
            +
            1.96 * se
            if np.isfinite(se)
            else np.nan
        )

        metric_pair_rows.append({
            "Metric":
                metric,

            "Metric_Label":
                METRIC_LABELS[metric],

            "Contrast":
                contrast_name,

            "Estimate":
                estimate,

            "SE":
                se,

            "Z":
                z,

            "P_raw":
                p,

            "CI95_Low":
                ci_low,

            "CI95_High":
                ci_high
        })


    temp_pair = pd.DataFrame(
        metric_pair_rows
    )

    temp_pair["P_Holm_within_metric"] = holm_adjust(
        temp_pair["P_raw"].to_numpy()
    )

    pairwise_rows.extend(
        temp_pair.to_dict(
            orient="records"
        )
    )


# ==============================================================================
# 10. Save LMM results
# ==============================================================================

omnibus_df = pd.DataFrame(
    omnibus_rows
)

# Additional Holm correction across the 7 omnibus tests
omnibus_df["Omnibus_p_Holm_7metrics"] = holm_adjust(
    omnibus_df["Omnibus_p"].to_numpy()
)


pairwise_df = pd.DataFrame(
    pairwise_rows
)


omnibus_file = os.path.join(
    OUTPUT_DIR,
    "Window50_LMM_omnibus_HC_M0_M1.csv"
)

pairwise_file = os.path.join(
    OUTPUT_DIR,
    "Window50_LMM_pairwise_HC_M0_M1.csv"
)

omnibus_df.to_csv(
    omnibus_file,
    index=False
)

pairwise_df.to_csv(
    pairwise_file,
    index=False
)


# ==============================================================================
# 11. Non-parametric sensitivity analyses
# ==============================================================================

nonparam_rows = []

for metric in METRICS:

    hc_x = long_df.loc[
        long_df["Condition"] == "HC",
        ["ModelSubject", metric]
    ].dropna()

    m0_x = long_df.loc[
        long_df["Condition"] == "M0",
        ["ModelSubject", metric]
    ].dropna()

    m1_x = long_df.loc[
        long_df["Condition"] == "M1",
        ["ModelSubject", metric]
    ].dropna()


    # HC vs M0
    U0, p0 = stats.mannwhitneyu(
        hc_x[metric],
        m0_x[metric],
        alternative="two-sided"
    )


    # HC vs M1
    U1, p1 = stats.mannwhitneyu(
        hc_x[metric],
        m1_x[metric],
        alternative="two-sided"
    )


    # paired M0 vs M1
    patient_pair = (
        pat_long[
            [
                "ModelSubject",
                "Session",
                metric
            ]
        ]
        .pivot(
            index="ModelSubject",
            columns="Session",
            values=metric
        )
        .dropna(
            subset=["M0", "M1"]
        )
    )

    try:

        W, pp = stats.wilcoxon(
            patient_pair["M1"],
            patient_pair["M0"],
            zero_method="wilcox",
            alternative="two-sided"
        )

    except ValueError:

        W = np.nan
        pp = np.nan


    temp = pd.DataFrame([
        {
            "Metric":
                metric,

            "Contrast":
                "M0 vs HC",

            "Test":
                "Mann-Whitney U",

            "N1":
                len(m0_x),

            "N2":
                len(hc_x),

            "Statistic":
                U0,

            "P_raw":
                p0
        },

        {
            "Metric":
                metric,

            "Contrast":
                "M1 vs HC",

            "Test":
                "Mann-Whitney U",

            "N1":
                len(m1_x),

            "N2":
                len(hc_x),

            "Statistic":
                U1,

            "P_raw":
                p1
        },

        {
            "Metric":
                metric,

            "Contrast":
                "M1 vs M0",

            "Test":
                "Wilcoxon signed-rank",

            "N1":
                len(patient_pair),

            "N2":
                len(patient_pair),

            "Statistic":
                W,

            "P_raw":
                pp
        }
    ])

    temp[
        "P_Holm_within_metric"
    ] = holm_adjust(
        temp["P_raw"].to_numpy()
    )

    nonparam_rows.extend(
        temp.to_dict(
            orient="records"
        )
    )


nonparam_df = pd.DataFrame(
    nonparam_rows
)


nonparam_file = os.path.join(
    OUTPUT_DIR,
    "Window50_nonparametric_sensitivity_HC_M0_M1.csv"
)

nonparam_df.to_csv(
    nonparam_file,
    index=False
)


# ==============================================================================
# 12. State 2 visit prevalence
# ==============================================================================

visit_desc_rows = []

for condition in ["HC", "M0", "M1"]:

    x = long_df.loc[
        long_df["Condition"] == condition,
        "Visit_State2"
    ].dropna().astype(int)

    visit_desc_rows.append({
        "Condition":
            condition,

        "N_Total":
            len(x),

        "Visited_State2_N":
            int(x.sum()),

        "Not_Visited_State2_N":
            int((1 - x).sum()),

        "Visited_State2_Percent":
            100 * x.mean()
    })


visit_desc_df = pd.DataFrame(
    visit_desc_rows
)


# HC vs M0 Fisher
hc_visit = long_df.loc[
    long_df["Condition"] == "HC",
    "Visit_State2"
].dropna().astype(int)

m0_visit = long_df.loc[
    long_df["Condition"] == "M0",
    "Visit_State2"
].dropna().astype(int)

m1_visit = long_df.loc[
    long_df["Condition"] == "M1",
    "Visit_State2"
].dropna().astype(int)


table_hc_m0 = np.array([
    [
        int(hc_visit.sum()),
        int((1 - hc_visit).sum())
    ],
    [
        int(m0_visit.sum()),
        int((1 - m0_visit).sum())
    ]
])

table_hc_m1 = np.array([
    [
        int(hc_visit.sum()),
        int((1 - hc_visit).sum())
    ],
    [
        int(m1_visit.sum()),
        int((1 - m1_visit).sum())
    ]
])


odds0, fisher_p0 = stats.fisher_exact(
    table_hc_m0,
    alternative="two-sided"
)

odds1, fisher_p1 = stats.fisher_exact(
    table_hc_m1,
    alternative="two-sided"
)


# M0 vs M1 exact McNemar through discordant pairs
visit_pair = (
    pat_long[
        [
            "ModelSubject",
            "Session",
            "Visit_State2"
        ]
    ]
    .pivot(
        index="ModelSubject",
        columns="Session",
        values="Visit_State2"
    )
    .dropna(
        subset=["M0", "M1"]
    )
    .astype(int)
)

b = int(
    np.sum(
        (visit_pair["M0"] == 1)
        &
        (visit_pair["M1"] == 0)
    )
)

c = int(
    np.sum(
        (visit_pair["M0"] == 0)
        &
        (visit_pair["M1"] == 1)
    )
)

if b + c > 0:

    mcnemar_p = stats.binomtest(
        min(b, c),
        n=b + c,
        p=0.5,
        alternative="two-sided"
    ).pvalue

else:

    mcnemar_p = 1.0


visit_tests_df = pd.DataFrame([
    {
        "Contrast":
            "M0 vs HC",

        "Test":
            "Fisher exact",

        "Statistic":
            odds0,

        "P_raw":
            fisher_p0
    },

    {
        "Contrast":
            "M1 vs HC",

        "Test":
            "Fisher exact",

        "Statistic":
            odds1,

        "P_raw":
            fisher_p1
    },

    {
        "Contrast":
            "M1 vs M0",

        "Test":
            "Exact McNemar",

        "Statistic":
            np.nan,

        "Discordant_M0_yes_M1_no":
            b,

        "Discordant_M0_no_M1_yes":
            c,

        "P_raw":
            mcnemar_p
    }
])


visit_tests_df[
    "P_Holm_3contrasts"
] = holm_adjust(
    visit_tests_df["P_raw"].to_numpy()
)


visit_desc_file = os.path.join(
    OUTPUT_DIR,
    "Window50_State2_visit_prevalence_HC_M0_M1.csv"
)

visit_test_file = os.path.join(
    OUTPUT_DIR,
    "Window50_State2_visit_tests_HC_M0_M1.csv"
)

visit_desc_df.to_csv(
    visit_desc_file,
    index=False
)

visit_tests_df.to_csv(
    visit_test_file,
    index=False
)


# ==============================================================================
# 13. Print results
# ==============================================================================

print()
print("=" * 120)
print("DESCRIPTIVE STATISTICS")
print("=" * 120)

for metric in METRICS:

    print()
    print(METRIC_LABELS[metric])

    print(
        desc_df.loc[
            desc_df["Metric"] == metric,
            [
                "Condition",
                "N",
                "Mean",
                "SD",
                "Median"
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )


print()
print("=" * 120)
print("LMM OMNIBUS TESTS")
print("=" * 120)

print(
    omnibus_df[
        [
            "Metric",
            "N_Observations",
            "N_Subjects",
            "Wald_Chi2",
            "df",
            "Omnibus_p",
            "Omnibus_p_Holm_7metrics",
            "Converged"
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.6g}"
    )
)


print()
print("=" * 120)
print("LMM PAIRWISE CONTRASTS")
print("Estimate direction: M0-HC, M1-HC, M1-M0")
print("=" * 120)

print(
    pairwise_df[
        [
            "Metric",
            "Contrast",
            "Estimate",
            "SE",
            "Z",
            "P_raw",
            "P_Holm_within_metric",
            "CI95_Low",
            "CI95_High"
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.6g}"
    )
)


print()
print("=" * 120)
print("STATE 2 VISIT PREVALENCE")
print("=" * 120)

print(
    visit_desc_df.to_string(
        index=False
    )
)

print()
print(
    visit_tests_df.to_string(
        index=False
    )
)


print()
print("=" * 120)
print("FILES SAVED")
print("=" * 120)

print(combined_file)
print(desc_file)
print(omnibus_file)
print(pairwise_file)
print(nonparam_file)
print(visit_desc_file)
print(visit_test_file)

print()
print("FINISHED")
