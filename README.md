# Dynamic White-Matter Network States in Metabolic Syndrome

Code for analysing **dynamic functional connectivity states of the white-matter network**
in patients with metabolic syndrome (MetS) before (M0) and after (M1) a short-term
time-restricted eating (eTRE) intervention, compared with healthy controls (HC).

Patients' M0 and M1 scans come from the same 42 participants (paired design);
48 HC scans serve as the reference group.

---

## 1. 方法概要

| 项目 | 设置 |
|---|---|
| 网络 | 白质 21 个 ROI 构成的 27 条疾病相关边（NBS 结果固定为 mask） |
| 连接值 | Fisher-z 转换后的边强度 |
| 滑窗 | 窗宽 50 TR，步长 1 TR，TR = 1 s → 每条扫描 541 个窗口 |
| 聚类 | K-means，K = 2（由患者 M0+M1 全部窗口得出质心） |
| 状态命名 | State 1 = 低连接状态，State 2 = 高连接状态 |
| HC 处理 | **不重新聚类**，每个 HC 窗口按欧氏最近质心分配到患者质心 |
| 分组比较 | 线性混合模型（LMM）+ 配对 t 检验 + Cohen's dz + BH-FDR |

> 关键设计：健康对照**不参与聚类**，只做投影。这样三组的状态定义完全一致，
> 避免"各组各聚各的"导致的不可比问题。

---

## 2. 动态指标定义

设某被试的状态序列为 `s_1 … s_T`（`T = 541`），目标状态为 `k`。

| 指标 | 定义 | 说明 |
|---|---|---|
| **Fractional occupancy (FO)** | `FO_k = #{t : s_t = k} / T` | 该状态占全部窗口的比例 |
| **Mean dwell time (MDT)** | 所有属于状态 `k` 的连续游程长度的算术平均 | 平均一次"停留"多久（单位：窗口数） |
| **Median run length** | 所有属于状态 `k` 的连续游程长度的中位数 | 对长游程更稳健 |
| **Switching rate** | `NT / (T − 1)`，`NT` = 状态切换总次数 | 每分钟切换频次（TR = 1 s） |
| **State-specific strength** | 该状态所有窗口上 27 条边的逐边中位数，再取平均 | 状态特异的网络强度 |

游程（run）定义：状态序列中连续相同标签的最长片段。

---

## 3. 目录结构

```
.
├── README.md
├── requirements.txt
├── reproduce_table.py            # 复现论文动态指标表
├── scripts/
│   ├── 01_compute_dynamic_metrics.py        ★ 核心：指标计算
│   ├── 02_group_comparison_states.py        ★ 核心：三组比较 + 产表
│   ├── 03_group_comparison_overall.py       整体动态指标版本
│   ├── 04_state_specific_edge_profiles.py   状态特异 27 边中位数轮廓
│   ├── 05_multivariate_m0_m1.py             M0 vs M1 多元检验
│   ├── 06_multivariate_hc_m0_m1.py          HC/M0/M1 三组多元检验
│   ├── 07_plot_state_occurrence.py          状态出现率轨迹图
│   ├── 08_plot_state_occurrence_paper_style.py  论文风格 Figure 2
│   └── 09_matlab_metrics_reference.m        MATLAB 参考实现（同一套指标）
└── docs/
    └── code_map.md               # 完整代码地图与结果溯源
```

---

## 4. 数据准备

所有脚本从环境变量 `WM_DYNAMIC_ROOT` 指向的目录读取数据，未设置时默认读 `./data/`。

```bash
export WM_DYNAMIC_ROOT=/path/to/your/analysis_root
```

`WM_DYNAMIC_ROOT` 下需要存在以下文件（相对路径）：

| 文件 | 生成步骤 | 内容 |
|---|---|---|
| `window_050_step_1/dynamic_edge_matrices/*.csv` | 上游（本仓库未包含） | 患者每个窗口的 27 边向量 |
| `HC_window_050_step_1/dynamic_edge_matrices/*.csv` | 上游（本仓库未包含） | HC 每个窗口的 27 边向量 |
| `Window_Sensitivity_K2_10/window_050/centroids/centroids_K02.csv` | 上游（本仓库未包含） | 患者 K=2 状态质心 |
| `Window50_M0_M1_Delta_all_dynamic_metrics.csv` | `01_` 的输出格式 | 患者逐被试动态指标 |

单个动态边矩阵的格式（`sub-001_ses-M0_dynamic27.csv`）：

```
window,edge_1,edge_2,...,edge_27
1,0.213,...
2,0.198,...
```

---

## 5. 运行顺序

```bash
python3 scripts/01_compute_dynamic_metrics.py        # → HC 逐被试指标 + 状态序列
python3 scripts/04_state_specific_edge_profiles.py   # → 状态特异 27 边中位数轮廓
python3 scripts/02_group_comparison_states.py        # → 三组描述统计 + LMM + 表
python3 scripts/05_multivariate_m0_m1.py             # → M0 vs M1 多元检验
python3 scripts/06_multivariate_hc_m0_m1.py          # → 三组多元检验
python3 scripts/07_plot_state_occurrence.py          # → 状态序列 + 轨迹图
python3 scripts/08_plot_state_occurrence_paper_style.py
```

`03_` 是 `02_` 的姊妹版本，用整体动态指标（不分状态），可作为敏感性分析。

---

## 6. 复现论文表格

```bash
python3 reproduce_table.py
```

输出 `Table_DynamicStates_M0_M1_H.csv`，并在终端打印 Markdown 表格。
该脚本只读取已生成的结果文件，**仅依赖 Python 标准库**（无需 numpy/scipy）。

| Metric | M0 | M1 | H | Paired t | P | Cohen's dz |
|---|---|---|---|---|---|---|
| Fractional occupancy | 0.311 ± 0.243 | 0.277 ± 0.260 | 0.115 ± 0.126 | −1.036 | 0.306 | −0.160 |
| State 1 MDT | 87.449 ± 62.543 | 137.035 ± 148.312 | 234.070 ± 196.748 | 2.818 | 0.007 | 0.435 |
| State 2 MDT | 34.077 ± 24.057 | 39.409 ± 48.088 | 20.795 ± 14.509 | 0.750 | 0.458 | 0.122 |
| State 1 median run length | 64.917 ± 63.633 | 119.024 ± 156.240 | 221.083 ± 207.044 | 2.682 | 0.011 | 0.414 |
| State 2 median run length | 26.513 ± 17.390 | 32.224 ± 47.671 | 18.083 ± 14.758 | 0.715 | 0.479 | 0.116 |
| Switching rate | 0.0175 ± 0.0082 | 0.0154 ± 0.0105 | 0.0104 ± 0.0089 | −1.364 | 0.180 | −0.211 |

**注意口径**：M0/M1 的均值±SD 使用**配对子样本**。State 2 的 MDT 和 median run length
在 4 名被试的 M1 中不存在（从未进入 State 2），因此 n = 38 而非 42。

---

## 7. 环境依赖

```bash
pip install -r requirements.txt
```

Python ≥ 3.9；`reproduce_table.py` 无需任何第三方库。

---

## 8. 已知限制

- **滑窗构建**与**患者 K-means 聚类**两步的代码不在本仓库中，仅提供输入/输出格式说明。
  因此从原始 NIfTI 到动态边矩阵的过程目前无法端到端复现。
- `scripts/09_matlab_metrics_reference.m` 中的 `<PROJECT_ROOT>` 需替换为本地路径后才能运行。
- 聚类使用固定随机种子无法完全保证跨平台一致，K-means 结果可能存在极小差异。

---

## 9. 数据与伦理

本仓库**不包含任何原始影像、被试人口学或临床数据**，仅包含分析代码。
受试者以 `sub-001` 形式匿名编号，不含可识别信息。

---

## 10. 参考

方法学参考：Schlemm E, et al. *Equalization of brain state occupancy accompanies cognitive
resilience in cerebral small vessel disease.* Biological Psychiatry, 2022.
