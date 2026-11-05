import os
import glob
import re
import numpy as np
import pandas as pd


# ==============================================================================
# HC projection to fixed patient-derived K=2 states
#
# Main settings:
# Window = 50 TR
# Step   = 1 TR
# K      = 2
#
# IMPORTANT:
# HC data are NOT re-clustered here.
# Every HC window is assigned to the fixed K=2 centroids derived from
# the patient M0+M1 analysis.
# ==============================================================================


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

HC_MATRIX_DIR = os.path.join(
    ROOT,
    "HC_window_050_step_1",
    "dynamic_edge_matrices"
)

PATIENT_K_ROOT = os.path.join(
    ROOT,
    "Window_Sensitivity_K2_10",
    "window_050"
)

OUTPUT_DIR = os.path.join(
    ROOT,
    "HC_window_050_step_1",
    "Projection_to_patient_K2"
)

STATESEQ_DIR = os.path.join(
    OUTPUT_DIR,
    "state_sequences"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

os.makedirs(
    STATESEQ_DIR,
    exist_ok=True
)


# ==============================================================================
# 2. Automatically find patient-derived K=2 centroid
# ==============================================================================

centroid_candidates = glob.glob(
    os.path.join(
        PATIENT_K_ROOT,
        "**",
        "centroids_K02.csv"
    ),
    recursive=True
)

if len(centroid_candidates) == 0:

    raise RuntimeError(
        "Cannot find patient centroids_K02.csv under:\n"
        + PATIENT_K_ROOT
    )

if len(centroid_candidates) > 1:

    print(
        "Multiple centroid files found:"
    )

    for x in centroid_candidates:
        print(x)

    raise RuntimeError(
        "Multiple centroid files found. "
        "Please verify which one is the final patient centroid."
    )

CENTROID_FILE = centroid_candidates[0]


# ==============================================================================
# 3. Fixed 27 NBS disease-related edges
# ==============================================================================

EDGES = [
    (1, 37),
    (1, 38),
    (2, 7),
    (2, 15),
    (4, 37),
    (7, 8),
    (7, 17),
    (15, 37),
    (17, 37),
    (17, 39),
    (21, 35),
    (34, 37),
    (35, 37),
    (35, 38),
    (36, 37),
    (37, 38),
    (37, 39),
    (37, 40),
    (37, 43),
    (37, 44),
    (37, 45),
    (37, 46),
    (37, 47),
    (37, 48),
    (38, 40),
    (38, 46),
    (38, 48)
]

EDGE_NAMES = [
    f"ROI{i}-ROI{j}"
    for i, j in EDGES
]


# ==============================================================================
# 4. Utility function
# ==============================================================================

def get_run_lengths(labels, target_state):

    runs = []
    current = 0

    for state in labels:

        if state == target_state:

            current += 1

        else:

            if current > 0:
                runs.append(current)
                current = 0

    if current > 0:
        runs.append(current)

    return np.asarray(
        runs,
        dtype=float
    )


# ==============================================================================
# 5. Load fixed patient K=2 centroids
# ==============================================================================

print("=" * 100)
print("HC PROJECTION TO FIXED PATIENT-DERIVED K=2 STATES")
print("WINDOW = 50 TR | STEP = 1 TR")
print("=" * 100)

print()
print("Patient centroid file:")
print(CENTROID_FILE)

centroid_df = pd.read_csv(
    CENTROID_FILE
)

missing_edges = [
    e
    for e in EDGE_NAMES
    if e not in centroid_df.columns
]

if missing_edges:

    raise RuntimeError(
        f"Patient centroid file missing edges: "
        f"{missing_edges}"
    )

centroids = centroid_df[
    EDGE_NAMES
].to_numpy(
    dtype=float
)

if centroids.shape != (2, 27):

    raise RuntimeError(
        f"Unexpected centroid shape: "
        f"{centroids.shape}"
    )


# ==============================================================================
# 6. Guarantee state ordering
#
# State 1 = low-connectivity state
# State 2 = high-connectivity state
# ==============================================================================

centroid_mean = centroids.mean(
    axis=1
)

if centroid_mean[0] > centroid_mean[1]:

    centroids = centroids[
        [1, 0],
        :
    ]

    centroid_mean = centroids.mean(
        axis=1
    )

print()

print(
    "State 1 centroid mean strength = "
    f"{centroid_mean[0]:.6f}"
)

print(
    "State 2 centroid mean strength = "
    f"{centroid_mean[1]:.6f}"
)

print(
    "State 1 = low-connectivity state"
)

print(
    "State 2 = high-connectivity state"
)


# ==============================================================================
# 7. Find HC dynamic matrices
# ==============================================================================

files = sorted(
    glob.glob(
        os.path.join(
            HC_MATRIX_DIR,
            "sub-H*_HC_dynamic27.csv"
        )
    )
)

print()
print(
    "HC subjects found =",
    len(files)
)

if len(files) != 48:

    raise RuntimeError(
        f"Expected 48 HC subjects, "
        f"found {len(files)}"
    )


# ==============================================================================
# 8. Project each HC subject to fixed patient states
# ==============================================================================

metric_rows = []

for n, filepath in enumerate(
    files,
    start=1
):

    basename = os.path.basename(
        filepath
    )

    m = re.search(
        r"(sub-H\d+)",
        basename
    )

    if m is None:

        raise RuntimeError(
            f"Cannot parse HC subject ID: "
            f"{basename}"
        )

    subject = m.group(1)

    df = pd.read_csv(
        filepath
    )


    # --------------------------------------------------------------------------
    # QC
    # --------------------------------------------------------------------------

    if len(df) != 541:

        raise RuntimeError(
            f"{subject}: expected 541 windows, "
            f"found {len(df)}"
        )

    missing = [
        e
        for e in EDGE_NAMES
        if e not in df.columns
    ]

    if missing:

        raise RuntimeError(
            f"{subject}: missing edges "
            f"{missing}"
        )

    X = df[
        EDGE_NAMES
    ].to_numpy(
        dtype=float
    )

    if (
        np.isnan(X).any()
        or
        np.isinf(X).any()
    ):

        raise RuntimeError(
            f"{subject}: NaN/Inf in dynamic matrix"
        )


    # --------------------------------------------------------------------------
    # Nearest-centroid state assignment
    # --------------------------------------------------------------------------

    dist1 = np.linalg.norm(
        X - centroids[0],
        axis=1
    )

    dist2 = np.linalg.norm(
        X - centroids[1],
        axis=1
    )

    labels = np.where(
        dist1 <= dist2,
        1,
        2
    )


    # --------------------------------------------------------------------------
    # State counts / FO
    # --------------------------------------------------------------------------

    n_windows = len(labels)

    n_state1 = int(
        np.sum(labels == 1)
    )

    n_state2 = int(
        np.sum(labels == 2)
    )

    fo1 = (
        n_state1
        /
        n_windows
    )

    fo2 = (
        n_state2
        /
        n_windows
    )


    # --------------------------------------------------------------------------
    # Runs / MDT
    # --------------------------------------------------------------------------

    runs1 = get_run_lengths(
        labels,
        1
    )

    runs2 = get_run_lengths(
        labels,
        2
    )

    mdt1 = (
        float(
            np.mean(runs1)
        )
        if len(runs1) > 0
        else np.nan
    )

    mdt2 = (
        float(
            np.mean(runs2)
        )
        if len(runs2) > 0
        else np.nan
    )


    # --------------------------------------------------------------------------
    # Visit counts
    # --------------------------------------------------------------------------

    visits1 = len(
        runs1
    )

    visits2 = len(
        runs2
    )


    # --------------------------------------------------------------------------
    # Median run
    # --------------------------------------------------------------------------

    median1 = (
        float(
            np.median(runs1)
        )
        if len(runs1) > 0
        else np.nan
    )

    median2 = (
        float(
            np.median(runs2)
        )
        if len(runs2) > 0
        else np.nan
    )


    # --------------------------------------------------------------------------
    # Maximum run
    # --------------------------------------------------------------------------

    max1 = (
        float(
            np.max(runs1)
        )
        if len(runs1) > 0
        else np.nan
    )

    max2 = (
        float(
            np.max(runs2)
        )
        if len(runs2) > 0
        else np.nan
    )


    # --------------------------------------------------------------------------
    # State switching
    # --------------------------------------------------------------------------

    switch_count = int(
        np.sum(
            labels[1:]
            !=
            labels[:-1]
        )
    )

    switching_rate = (
        switch_count
        /
        (n_windows - 1)
    )


    # --------------------------------------------------------------------------
    # Network strength
    # --------------------------------------------------------------------------

    network_strength = np.mean(
        X,
        axis=1
    )

    dynamic_mean = float(
        np.mean(
            network_strength
        )
    )

    dynamic_sd = float(
        np.std(
            network_strength,
            ddof=1
        )
    )

    p90_p10 = float(
        np.percentile(
            network_strength,
            90
        )
        -
        np.percentile(
            network_strength,
            10
        )
    )


    # --------------------------------------------------------------------------
    # State-specific strength
    # --------------------------------------------------------------------------

    state1_strength = (
        float(
            np.mean(
                network_strength[
                    labels == 1
                ]
            )
        )
        if n_state1 > 0
        else np.nan
    )

    state2_strength = (
        float(
            np.mean(
                network_strength[
                    labels == 2
                ]
            )
        )
        if n_state2 > 0
        else np.nan
    )

    state_contrast = (
        state2_strength
        -
        state1_strength
        if (
            np.isfinite(
                state1_strength
            )
            and
            np.isfinite(
                state2_strength
            )
        )
        else np.nan
    )


    # --------------------------------------------------------------------------
    # Save subject-level metrics
    # --------------------------------------------------------------------------

    metric_rows.append(
        {
            "Subject":
                subject,

            "Group":
                "HC",

            "Window_TR":
                50,

            "Step_TR":
                1,

            "N_Windows":
                n_windows,

            "N_State1":
                n_state1,

            "N_State2":
                n_state2,

            "FO_State1":
                fo1,

            "FO_State2":
                fo2,

            "Visit_State1":
                int(
                    n_state1 > 0
                ),

            "Visit_State2":
                int(
                    n_state2 > 0
                ),

            "N_Visits_State1":
                visits1,

            "N_Visits_State2":
                visits2,

            "MDT_State1_windows":
                mdt1,

            "MDT_State2_windows":
                mdt2,

            "MedianRun_State1":
                median1,

            "MedianRun_State2":
                median2,

            "MaxRun_State1":
                max1,

            "MaxRun_State2":
                max2,

            "Switch_Count":
                switch_count,

            "Switching_Rate":
                switching_rate,

            "DynamicMean":
                dynamic_mean,

            "Variability_SD":
                dynamic_sd,

            "P90_P10":
                p90_p10,

            "State1_Strength":
                state1_strength,

            "State2_Strength":
                state2_strength,

            "State2_minus_State1_Strength":
                state_contrast
        }
    )


    # --------------------------------------------------------------------------
    # Save full state sequence
    # --------------------------------------------------------------------------

    seq = pd.DataFrame(
        {
            "Subject":
                subject,

            "Group":
                "HC",

            "Window":
                df["Window"],

            "Start_TR":
                df["Start_TR"],

            "Start_sec":
                df["Start_sec"],

            "State":
                labels,

            "Network_Strength":
                network_strength,

            "Distance_State1":
                dist1,

            "Distance_State2":
                dist2
        }
    )

    seq_file = os.path.join(
        STATESEQ_DIR,
        f"{subject}_HC_window50_state_sequence.csv"
    )

    seq.to_csv(
        seq_file,
        index=False
    )

    print(
        f"[{n:02d}/48] "
        f"{subject} | "
        f"FO1={fo1:.3f} | "
        f"FO2={fo2:.3f} | "
        f"MDT1={mdt1:.2f} | "
        f"MDT2="
        f"{mdt2 if np.isfinite(mdt2) else 'NA'} | "
        f"Switch={switching_rate:.4f}"
    )


# ==============================================================================
# 9. Save subject-level metrics
# ==============================================================================

metrics = pd.DataFrame(
    metric_rows
)

metrics_file = os.path.join(
    OUTPUT_DIR,
    "HC_window50_projected_temporal_metrics.csv"
)

metrics.to_csv(
    metrics_file,
    index=False
)


# ==============================================================================
# 10. HC descriptive summary
# ==============================================================================

summary_vars = [
    "DynamicMean",
    "Variability_SD",
    "P90_P10",

    "FO_State1",
    "FO_State2",

    "MDT_State1_windows",
    "MDT_State2_windows",

    "N_Visits_State1",
    "N_Visits_State2",

    "Switching_Rate",

    "State1_Strength",
    "State2_Strength",
    "State2_minus_State1_Strength"
]

summary_rows = []

for variable in summary_vars:

    x = metrics[
        variable
    ].dropna()

    summary_rows.append(
        {
            "Metric":
                variable,

            "N":
                len(x),

            "Mean":
                x.mean(),

            "SD":
                x.std(
                    ddof=1
                ),

            "Median":
                x.median(),

            "Min":
                x.min(),

            "Max":
                x.max()
        }
    )

summary = pd.DataFrame(
    summary_rows
)

summary_file = os.path.join(
    OUTPUT_DIR,
    "HC_window50_group_summary.csv"
)

summary.to_csv(
    summary_file,
    index=False
)


# ==============================================================================
# 11. State visit prevalence
# ==============================================================================

visit_summary = pd.DataFrame(
    [
        {
            "Group":
                "HC",

            "N_Total":
                len(metrics),

            "State1_Visited_N":
                int(
                    metrics[
                        "Visit_State1"
                    ].sum()
                ),

            "State1_Visited_Percent":
                100
                *
                metrics[
                    "Visit_State1"
                ].mean(),

            "State2_Visited_N":
                int(
                    metrics[
                        "Visit_State2"
                    ].sum()
                ),

            "State2_Visited_Percent":
                100
                *
                metrics[
                    "Visit_State2"
                ].mean()
        }
    ]
)

visit_file = os.path.join(
    OUTPUT_DIR,
    "HC_window50_state_visit_prevalence.csv"
)

visit_summary.to_csv(
    visit_file,
    index=False
)


# ==============================================================================
# 12. Save analysis metadata for reproducibility
# ==============================================================================

metadata = pd.DataFrame(
    [
        {
            "Analysis":
                "HC projection to fixed patient K2",

            "Window_TR":
                50,

            "Step_TR":
                1,

            "K":
                2,

            "N_Edges":
                27,

            "N_HC":
                len(metrics),

            "Centroid_File":
                CENTROID_FILE,

            "State1_Centroid_Mean":
                centroid_mean[0],

            "State2_Centroid_Mean":
                centroid_mean[1],

            "Assignment":
                "Nearest Euclidean centroid",

            "Feature_Standardization":
                "None",

            "State1_Definition":
                "Low-connectivity",

            "State2_Definition":
                "High-connectivity"
        }
    ]
)

metadata_file = os.path.join(
    OUTPUT_DIR,
    "HC_window50_projection_metadata.csv"
)

metadata.to_csv(
    metadata_file,
    index=False
)


# ==============================================================================
# 13. Final output
# ==============================================================================

print()
print("=" * 110)
print("HC PROJECTION FINISHED")
print("=" * 110)

print()
print("Patient fixed centroids:")

print(
    f"State1 mean = "
    f"{centroid_mean[0]:.6f}"
)

print(
    f"State2 mean = "
    f"{centroid_mean[1]:.6f}"
)

print()
print("HC visit prevalence:")

print(
    visit_summary.to_string(
        index=False
    )
)

print()
print("HC group summary:")

print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

print()
print("Saved subject metrics:")
print(metrics_file)

print()
print("Saved state sequences:")
print(STATESEQ_DIR)

print()
print("Saved group summary:")
print(summary_file)

print()
print("Saved visit prevalence:")
print(visit_file)

print()
print("Saved metadata:")
print(metadata_file)

print()
print("FINISHED")
