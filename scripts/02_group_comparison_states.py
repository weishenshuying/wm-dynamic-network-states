import os
import numpy as np
import pandas as pd

from scipy import stats
import statsmodels.formula.api as smf


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
    "Window50_M0_M1_Delta_all_dynamic_metrics.csv"
)

OUTPUT_DIR = os.path.join(
    ROOT,
    "HC_M0_M1_Window50_StateComparison"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ==============================================================================
# 2. Main state metrics
# ==============================================================================

METRICS = [
    "FO_State2",
    "MDT_State1_windows",
    "MDT_State2_windows",
    "Switching_Rate",
    "N_Visits_State2",
    "State1_Strength",
    "State2_Strength",
    "State2_minus_State1_Strength"
]

LABELS = {
    "FO_State2":
        "High-state fractional occupancy",

    "MDT_State1_windows":
        "Low-state mean dwell time",

    "MDT_State2_windows":
        "High-state mean dwell time",

    "Switching_Rate":
        "State switching rate",

    "N_Visits_State2":
        "High-state number of visits",

    "State1_Strength":
        "Low-state network strength",

    "State2_Strength":
        "High-state network strength",

    "State2_minus_State1_Strength":
        "State strength contrast"
}


# ==============================================================================
# 3. Holm correction
# ==============================================================================

def holm_adjust(pvals):

    pvals = np.asarray(
        pvals,
        dtype=float
    )

    out = np.full(
        len(pvals),
        np.nan
    )

    valid = np.where(
        np.isfinite(pvals)
    )[0]

    if len(valid) == 0:
        return out

    p = pvals[valid]

    order = np.argsort(p)

    sorted_p = p[order]

    m = len(sorted_p)

    adjusted_sorted = np.empty(m)

    previous = 0.0

    for i, pv in enumerate(sorted_p):

        adj = (m - i) * pv

        previous = max(
            previous,
            adj
        )

        adjusted_sorted[i] = min(
            previous,
            1.0
        )

    adjusted = np.empty(m)

    adjusted[order] = adjusted_sorted

    out[valid] = adjusted

    return out


# ==============================================================================
# 4. Read data
# ==============================================================================

hc = pd.read_csv(
    HC_FILE
)

pw = pd.read_csv(
    PATIENT_FILE
)

print("=" * 110)
print("HC vs M0 vs M1 STATE DYNAMIC ANALYSIS")
print("50 TR | STEP=1 | FIXED PATIENT-DERIVED K=2 STATES")
print("=" * 110)

print("\nHC:")
print(HC_FILE)
print("Shape:", hc.shape)

print("\nPatients:")
print(PATIENT_FILE)
print("Shape:", pw.shape)


if len(hc) != 48:
    raise RuntimeError(
        f"Expected 48 HC, found {len(hc)}"
    )

if len(pw) != 42:
    raise RuntimeError(
        f"Expected 42 patients, found {len(pw)}"
    )


# ==============================================================================
# 5. Check required variables
# ==============================================================================

for metric in METRICS:

    if metric not in hc.columns:
        raise RuntimeError(
            f"HC missing: {metric}"
        )

    for session in ["M0", "M1"]:

        col = f"{session}_{metric}"

        if col not in pw.columns:
            raise RuntimeError(
                f"Patient file missing: {col}"
            )


# ==============================================================================
# 6. Convert patient wide table -> long table
# ==============================================================================

patient_rows = []

for _, row in pw.iterrows():

    subject = row["Subject"]

    for session in ["M0", "M1"]:

        temp = {
            "Subject":
                subject,

            "ModelSubject":
                "PAT_" + str(subject),

            "Condition":
                session
        }

        for metric in METRICS:

            temp[metric] = row[
                f"{session}_{metric}"
            ]

        # State2 visit:
        # MDT is undefined when State2 never occurs.
        temp["Visit_State2"] = int(
            pd.notna(
                row[
                    f"{session}_MDT_State2_windows"
                ]
            )
        )

        patient_rows.append(
            temp
        )


pat = pd.DataFrame(
    patient_rows
)


# ==============================================================================
# 7. HC long table
# ==============================================================================

hc_long = hc[
    ["Subject"] + METRICS
].copy()

hc_long["ModelSubject"] = (
    "HC_"
    +
    hc_long["Subject"].astype(str)
)

hc_long["Condition"] = "HC"

hc_long["Visit_State2"] = (
    hc_long[
        "MDT_State2_windows"
    ].notna().astype(int)
)


# ==============================================================================
# 8. Combined long-format dataset
# ==============================================================================

long_df = pd.concat(
    [
        hc_long[
            [
                "Subject",
                "ModelSubject",
                "Condition",
                "Visit_State2"
            ]
            +
            METRICS
        ],

        pat[
            [
                "Subject",
                "ModelSubject",
                "Condition",
                "Visit_State2"
            ]
            +
            METRICS
        ]
    ],
    ignore_index=True
)


long_df["Condition"] = pd.Categorical(
    long_df["Condition"],
    categories=[
        "HC",
        "M0",
        "M1"
    ],
    ordered=True
)


combined_file = os.path.join(
    OUTPUT_DIR,
    "Window50_HC_M0_M1_state_metrics_long.csv"
)

long_df.to_csv(
    combined_file,
    index=False
)


# ==============================================================================
# 9. Descriptive statistics
# ==============================================================================

desc_rows = []

for metric in METRICS:

    for group in [
        "HC",
        "M0",
        "M1"
    ]:

        x = long_df.loc[
            long_df["Condition"] == group,
            metric
        ].dropna().astype(float)

        desc_rows.append({
            "Metric":
                metric,

            "Metric_Label":
                LABELS[metric],

            "Condition":
                group,

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


desc = pd.DataFrame(
    desc_rows
)

desc_file = os.path.join(
    OUTPUT_DIR,
    "Window50_HC_M0_M1_state_descriptives.csv"
)

desc.to_csv(
    desc_file,
    index=False
)


# ==============================================================================
# 10. LMM function
# ==============================================================================

def fit_lmm(metric):

    d = long_df[
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

    model = smf.mixedlm(
        "Y ~ C(Condition, Treatment(reference='HC'))",
        data=d,
        groups=d["ModelSubject"]
    )

    result = None
    used_method = None

    for method in [
        "lbfgs",
        "bfgs",
        "cg",
        "powell"
    ]:

        try:

            candidate = model.fit(
                reml=False,
                method=method,
                maxiter=3000,
                disp=False
            )

            result = candidate
            used_method = method

            if candidate.converged:
                break

        except Exception:
            continue

    if result is None:
        raise RuntimeError(
            f"LMM failed: {metric}"
        )

    return result, d, used_method


# ==============================================================================
# 11. LMM omnibus + pairwise contrasts
# ==============================================================================

omnibus_rows = []
pair_rows = []


for metric in METRICS:

    print("\n" + "=" * 100)
    print(LABELS[metric])
    print("=" * 100)

    result, d, method = fit_lmm(
        metric
    )

    m0_name = (
        "C(Condition, Treatment(reference='HC'))[T.M0]"
    )

    m1_name = (
        "C(Condition, Treatment(reference='HC'))[T.M1]"
    )

    if (
        m0_name not in result.params.index
        or
        m1_name not in result.params.index
    ):
        raise RuntimeError(
            f"Missing group coefficients: {metric}"
        )


    # --------------------------------------------------------------------------
    # Omnibus Wald chi-square
    # --------------------------------------------------------------------------

    beta = np.array([
        result.params[m0_name],
        result.params[m1_name]
    ])

    cov = result.cov_params()

    cov2 = cov.loc[
        [m0_name, m1_name],
        [m0_name, m1_name]
    ].to_numpy()

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

    omnibus_rows.append({
        "Metric":
            metric,

        "Metric_Label":
            LABELS[metric],

        "N_Observations":
            len(d),

        "N_Subjects":
            d["ModelSubject"].nunique(),

        "Optimizer":
            method,

        "Converged":
            result.converged,

        "Wald_Chi2":
            chi2,

        "df":
            2,

        "Omnibus_p":
            omnibus_p
    })


    # --------------------------------------------------------------------------
    # Pairwise contrasts
    # --------------------------------------------------------------------------

    b0 = float(
        result.params[m0_name]
    )

    b1 = float(
        result.params[m1_name]
    )

    v00 = float(
        cov.loc[
            m0_name,
            m0_name
        ]
    )

    v11 = float(
        cov.loc[
            m1_name,
            m1_name
        ]
    )

    v01 = float(
        cov.loc[
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
            v11 + v00 - 2*v01
        )
    ]


    temp_rows = []

    for name, est, var in contrasts:

        se = np.sqrt(
            max(
                var,
                0
            )
        )

        z = (
            est / se
            if se > 0
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

        temp_rows.append({
            "Metric":
                metric,

            "Metric_Label":
                LABELS[metric],

            "Contrast":
                name,

            "Estimate":
                est,

            "SE":
                se,

            "Z":
                z,

            "P_raw":
                p,

            "CI95_Low":
                est - 1.96*se,

            "CI95_High":
                est + 1.96*se
        })


    temp = pd.DataFrame(
        temp_rows
    )

    temp[
        "P_Holm_3contrasts"
    ] = holm_adjust(
        temp["P_raw"].to_numpy()
    )

    pair_rows.extend(
        temp.to_dict(
            orient="records"
        )
    )


omnibus = pd.DataFrame(
    omnibus_rows
)

omnibus[
    "P_Holm_8metrics"
] = holm_adjust(
    omnibus[
        "Omnibus_p"
    ].to_numpy()
)


pairwise = pd.DataFrame(
    pair_rows
)


omnibus_file = os.path.join(
    OUTPUT_DIR,
    "Window50_LMM_state_omnibus.csv"
)

pair_file = os.path.join(
    OUTPUT_DIR,
    "Window50_LMM_state_pairwise.csv"
)

omnibus.to_csv(
    omnibus_file,
    index=False
)

pairwise.to_csv(
    pair_file,
    index=False
)


# ==============================================================================
# 12. State2 visit prevalence
# ==============================================================================

visit_rows = []

for group in [
    "HC",
    "M0",
    "M1"
]:

    x = long_df.loc[
        long_df["Condition"] == group,
        "Visit_State2"
    ].astype(int)

    visit_rows.append({
        "Condition":
            group,

        "N":
            len(x),

        "Visited_State2":
            int(x.sum()),

        "Not_Visited_State2":
            int((1-x).sum()),

        "Percent_Visited":
            100*x.mean()
    })


visit_desc = pd.DataFrame(
    visit_rows
)


# HC vs M0 / HC vs M1
def fisher_visit(group):

    hc_x = long_df.loc[
        long_df["Condition"] == "HC",
        "Visit_State2"
    ].astype(int)

    g_x = long_df.loc[
        long_df["Condition"] == group,
        "Visit_State2"
    ].astype(int)

    tab = np.array([
        [
            hc_x.sum(),
            (1-hc_x).sum()
        ],
        [
            g_x.sum(),
            (1-g_x).sum()
        ]
    ])

    odds, p = stats.fisher_exact(
        tab
    )

    return odds, p


odds_m0, p_m0 = fisher_visit(
    "M0"
)

odds_m1, p_m1 = fisher_visit(
    "M1"
)


# M0 vs M1 McNemar
visit_pair = pat.pivot(
    index="ModelSubject",
    columns="Condition",
    values="Visit_State2"
).dropna()

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

    p_mc = stats.binomtest(
        min(b, c),
        n=b+c,
        p=0.5
    ).pvalue

else:

    p_mc = 1.0


visit_tests = pd.DataFrame([
    {
        "Contrast":
            "M0 vs HC",

        "Test":
            "Fisher exact",

        "Statistic":
            odds_m0,

        "P_raw":
            p_m0
    },

    {
        "Contrast":
            "M1 vs HC",

        "Test":
            "Fisher exact",

        "Statistic":
            odds_m1,

        "P_raw":
            p_m1
    },

    {
        "Contrast":
            "M1 vs M0",

        "Test":
            "Exact McNemar",

        "Statistic":
            np.nan,

        "P_raw":
            p_mc,

        "M0_yes_M1_no":
            b,

        "M0_no_M1_yes":
            c
    }
])


visit_tests[
    "P_Holm_3contrasts"
] = holm_adjust(
    visit_tests[
        "P_raw"
    ].to_numpy()
)


visit_desc_file = os.path.join(
    OUTPUT_DIR,
    "Window50_State2_visit_prevalence.csv"
)

visit_test_file = os.path.join(
    OUTPUT_DIR,
    "Window50_State2_visit_tests.csv"
)

visit_desc.to_csv(
    visit_desc_file,
    index=False
)

visit_tests.to_csv(
    visit_test_file,
    index=False
)


# ==============================================================================
# 13. Print
# ==============================================================================

print("\n" + "=" * 120)
print("DESCRIPTIVE STATISTICS")
print("=" * 120)

for metric in METRICS:

    print("\n" + LABELS[metric])

    print(
        desc.loc[
            desc["Metric"] == metric,
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


print("\n" + "=" * 120)
print("LMM OMNIBUS")
print("=" * 120)

print(
    omnibus[
        [
            "Metric",
            "N_Observations",
            "N_Subjects",
            "Wald_Chi2",
            "df",
            "Omnibus_p",
            "P_Holm_8metrics",
            "Converged"
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.6g}"
    )
)


print("\n" + "=" * 120)
print("LMM PAIRWISE")
print("=" * 120)

print(
    pairwise[
        [
            "Metric",
            "Contrast",
            "Estimate",
            "SE",
            "Z",
            "P_raw",
            "P_Holm_3contrasts",
            "CI95_Low",
            "CI95_High"
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.6g}"
    )
)


print("\n" + "=" * 120)
print("STATE 2 VISIT PREVALENCE")
print("=" * 120)

print(
    visit_desc.to_string(
        index=False
    )
)

print()

print(
    visit_tests.to_string(
        index=False
    )
)


print("\n" + "=" * 120)
print("FILES SAVED")
print("=" * 120)

print(combined_file)
print(desc_file)
print(omnibus_file)
print(pair_file)
print(visit_desc_file)
print(visit_test_file)

print("\nFINISHED")
