# Code Map

How the individual scripts fit together, what each reads and writes, and where the
look-alike files outside `scripts/` belong.

---

## 1. Pipeline at a glance

```
fisher-z edge time series (per subject)
        │
        │  ① sliding window: 50 TR / step 1 TR  →  541 windows
        ▼
data/window_050_step_1/dynamic_edge_matrices/      84 patient scans (42 × M0/M1)
data/HC_window_050_step_1/dynamic_edge_matrices/   48 HC scans
        │
        │  ② K-means (K=2) on pooled patient windows  →  2 centroids
        ▼
data/Window_Sensitivity_K2_10/window_050/centroids/centroids_K02.csv
        │
        │  ③ project every HC window onto the patient centroids (no re-clustering)
        ▼
scripts/01_compute_dynamic_metrics.py
   → HC_window_050_step_1/Projection_to_patient_K2/
        HC_window50_projected_temporal_metrics.csv
        state_sequences/<subject>_HC_window50_state_sequence.csv
        │
        │  ④ per-state, per-subject median connectivity over the 27 edges
        ▼
scripts/04_state_specific_edge_profiles.py
   → StateSpecific_27Edge_Median_Window50/
        Window50_K2_centroids_used_State1Low_State2High.csv
        Window50_K2_StateSpecific_27edge_median_profiles_long.csv
        │
        │  ⑤ group descriptives + LMM + paired tests
        ▼
scripts/02_group_comparison_states.py
   → HC_M0_M1_Window50_StateComparison/
        Window50_HC_M0_M1_state_metrics_long.csv      per-subject long table
        Window50_HC_M0_M1_state_descriptives.csv
        Window50_HC_M0_M1_summary_table.csv           paper table (base version)
        Window50_LMM_state_omnibus.csv
        Window50_LMM_state_pairwise.csv
        │
        │  ⑥ state-specific multivariate testing (27 edges)
        ▼
scripts/05_multivariate_m0_m1.py          M1 vs M0, paired Hotelling T²
scripts/06_multivariate_hc_m0_m1.py       HC / M0 / M1, three contrasts
        │
        │  ⑦ visualisation
        ▼
scripts/07_plot_state_occurrence.py
scripts/08_plot_state_occurrence_paper_style.py
```

Steps ① and ② are **not included** in this repository — see README §8.

---

## 2. Script responsibilities

| Script | Role | Key outputs |
|---|---|---|
| `01_compute_dynamic_metrics.py` | **Core.** Assigns every HC window to the nearest patient centroid, then computes FO, MDT, median/max run length, visit counts, switching rate and state strengths | `HC_window50_projected_temporal_metrics.csv`, state sequences |
| `02_group_comparison_states.py` | **Core.** Three-group descriptives, LMM omnibus/post-hoc, paired contrasts; writes the summary table | `..._state_metrics_long.csv`, `..._summary_table.csv`, `Window50_M0_M1_Delta_all_dynamic_metrics.csv` |
| `03_group_comparison_overall.py` | Sibling of `02_` using whole-scan dynamic metrics instead of state-specific ones | `Window50_HC_M0_M1_descriptive_statistics.csv` |
| `04_state_specific_edge_profiles.py` | For each subject × state, takes the element-wise median across windows assigned to that state | `..._median_profiles_long.csv`, centroid-usage file |
| `05_multivariate_m0_m1.py` | Paired Hotelling T² + 10,000 sign-flip permutations; edge follow-up with paired t, Wilcoxon, Cohen's dz, BH-FDR | `Window50_StateSpecific_M0_M1_multivariate_overall.csv` |
| `06_multivariate_hc_m0_m1.py` | Same hierarchy extended to three groups (M0−HC and M1−HC independent, M1−M0 paired) | `Window50_StateSpecific_HC_M0_M1_multivariate_contrasts.csv` |
| `07_plot_state_occurrence.py` | Extracts state sequences and plots state-occurrence trajectories | `Figure_Window50_StateOccurrence_Trajectory.png` |
| `08_plot_state_occurrence_paper_style.py` | Paper-style Figure 2 (30-window centred rolling mean, visualisation only) | `Figure_Window50_StateOccurrence_PaperStyle.png` |
| `09_matlab_metrics_reference.m` | MATLAB/DBC implementation of the same metric set, for cross-checking | `Disease27Edge_K2_AllSubjects.csv` |

---

## 3. Files that look related but are not part of this pipeline

These live in the original project tree, outside `scripts/`, and cover adjacent
analyses. They are listed here so they are not mistaken for duplicates:

| File | Actual purpose |
|---|---|
| `linshi.m` | Converts `Disease27Edge (271×27)` into DBC-compatible 21×21 matrix cells — format conversion only |
| `combineKmeansstates.m` | Merges K-means state labels with disease-network dynamic FC (HC group) |
| `Extract_DiseaseNetwork_27Edges.m` | Extracts the 27 disease edges from whole-brain matrices |
| `untitled4.m` | Edge-wise validation of the 167 NBS-significant edges |
| `untitled5.m` | Longitudinal H/M0/M1 validation of those 167 edges |
| `untitled6.m` | Per-subject mean Fisher-z over the 167-edge NBS subnetwork |
| `generate_delta_27edges*.py` | Static delta-edge matrices (no sliding window) |

---

## 4. Output naming conventions

| Prefix / suffix | Meaning |
|---|---|
| `Window50_` | 50 TR sliding window |
| `_State1` / `_State2` | Low-connectivity / high-connectivity state |
| `_windows` on MDT fields | Dwell time expressed in windows (TR = 1 s, so windows ≈ seconds) |
| `_long.csv` | One row per subject × condition (tidy format) |
| `_wide.csv` | One row per subject, one column per metric |
| `M0` / `M1` | Pre- / post-intervention |
| `HC` / `H` | Healthy controls |

---

## 5. Subject matching

Patients are matched across M0 and M1 by the `Subject` field (`sub-001` … ).
HC subjects use a separate namespace (`sub-H001` … ) to avoid collisions.
Group membership is carried in a `Condition` column with values `HC`, `M0`, `M1`.
