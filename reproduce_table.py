#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
复现论文中的动态网络状态指标表
================================
生成内容 = 你提供的这张表：
    Metric / M0 / M1 / H / Paired t (M1-M0) / P / Adjusted P / Cohen's dz

数据来源（全部为已存在的结果文件，本脚本只读取、不改写）：
  [1] HC_M0_M1_Window50_StateComparison/Window50_HC_M0_M1_state_metrics_long.csv
      -> 逐受试者 FO_State2 / MDT_State1 / MDT_State2 / Switching_Rate
  [2] Window50_M0_M1_Delta_all_dynamic_metrics.csv
      -> 逐受试者 MedianRun_State1 / MedianRun_State2（患者 M0/M1）
  [3] HC_window_050_step_1/Projection_to_patient_K2/HC_window50_projected_temporal_metrics.csv
      -> 健康对照(H)的 MedianRun_State1 / MedianRun_State2

统计口径（已严格对齐你的表）：
  * 均值±SD：M0/M1 采用「配对子样本」口径（即有 M1 数据的同一批受试者）。
             例：State 2 MDT 的 M0 用 n=38 而非 n=42（34.077 vs 32.498）。
             H 组用该指标全部有效被试。
  * Paired t：配对 t 检验，差值方向 = M1 - M0。
  * P value：双侧。
  * Adjusted P：6 个指标 BH-FDR。
  * Cohen's dz：配对差值均值 / 配对差值 SD。

依赖：仅 Python 标准库（无需 numpy/scipy）。若已装 scipy 会自动使用 scipy。

用法：
    python3 reproduce_table.py
    # 或指定数据目录
    WM_DYNAMIC_ROOT=/path/to/data python3 reproduce_table.py
输出：
    Table_DynamicStates_M0_M1_H.csv
    并在终端打印 Markdown 表格
"""

import csv
import math
import os

# ============================================================
# 0. 路径（数据目录由环境变量 WM_DYNAMIC_ROOT 指定，默认 ./data）
# ============================================================
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("WM_DYNAMIC_ROOT", os.path.join(HERE, "data"))

F_LONG = os.path.join(ROOT, "HC_M0_M1_Window50_StateComparison",
                      "Window50_HC_M0_M1_state_metrics_long.csv")
F_DELTA = os.path.join(ROOT, "Window50_M0_M1_Delta_all_dynamic_metrics.csv")
F_HC = os.path.join(ROOT, "HC_window_050_step_1", "Projection_to_patient_K2",
                    "HC_window50_projected_temporal_metrics.csv")
OUT_CSV = os.path.join(HERE, "Table_DynamicStates_M0_M1_H.csv")


# ============================================================
# 1. 统计工具（纯标准库实现）
# ============================================================
def _betacf(a, b, x, max_iter=200, eps=3e-16):
    """连分式展开，用于正则化不完全 Beta 函数。"""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            break
    return h


def _betai(a, b, x):
    """正则化不完全 Beta 函数 I_x(a,b)。"""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_sf_two_sided(t, df):
    """双侧 t 检验 p 值。"""
    try:
        from scipy import stats
        return float(2.0 * stats.t.sf(abs(t), df))
    except Exception:
        pass
    if df <= 0:
        return float("nan")
    x = df / (df + t * t)
    return _betai(df / 2.0, 0.5, x)


def mean(xs):
    return sum(xs) / len(xs)


def sd(xs):
    n = len(xs)
    if n < 2:
        return float("nan")
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


def paired_t(a, b):
    """a=M0, b=M1，差值 d = b - a（即 M1-M0）。"""
    d = [y - x for x, y in zip(a, b)]
    n = len(d)
    md, sdd = mean(d), sd(d)
    t = md / (sdd / math.sqrt(n))
    p = t_sf_two_sided(t, n - 1)
    dz = md / sdd
    return {"n": n, "t": t, "p": p, "dz": dz,
            "m0": mean(a), "sd0": sd(a), "m1": mean(b), "sd1": sd(b)}


def bh_fdr(pvals):
    """Benjamini-Hochberg FDR，返回与输入同序的调整 p。"""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    prev = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        val = min(prev, pvals[i] * m / rank)
        adj[i] = min(val, 1.0)
        prev = adj[i]
    return adj


# ============================================================
# 2. 读取数据
# ============================================================
def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def col_by_subject(rows, key, cond=None):
    out = {}
    for r in rows:
        if cond is not None and r.get("Condition") != cond:
            continue
        v = (r.get(key) or "").strip()
        if v in ("", "NA", "nan", "NaN"):
            continue
        try:
            out[r["Subject"]] = float(v)
        except ValueError:
            continue
    return out


long_rows = read_csv(F_LONG)
delta_rows = read_csv(F_DELTA)
hc_rows = read_csv(F_HC)

M0 = {k: col_by_subject(long_rows, k, "M0") for k in
      ("FO_State2", "MDT_State1_windows", "MDT_State2_windows", "Switching_Rate")}
M1 = {k: col_by_subject(long_rows, k, "M1") for k in M0}

# 中位数游程长度只存在于 Delta 表（患者）
for src_key, dst in (("MedianRun_State1", "M0_MedianRun_State1"),):
    pass
delta_map = {r["Subject"]: r for r in delta_rows}


def delta_col(key):
    out = {}
    for s, r in delta_map.items():
        v = (r.get(key) or "").strip()
        if v in ("", "NA", "nan", "NaN"):
            continue
        try:
            out[s] = float(v)
        except ValueError:
            pass
    return out


M0["MedianRun_State1"] = delta_col("M0_MedianRun_State1")
M1["MedianRun_State1"] = delta_col("M1_MedianRun_State1")
M0["MedianRun_State2"] = delta_col("M0_MedianRun_State2")
M1["MedianRun_State2"] = delta_col("M1_MedianRun_State2")

HC = {
    "FO_State2": col_by_subject(hc_rows, "FO_State2"),
    "MDT_State1_windows": col_by_subject(hc_rows, "MDT_State1_windows"),
    "MDT_State2_windows": col_by_subject(hc_rows, "MDT_State2_windows"),
    "MedianRun_State1": col_by_subject(hc_rows, "MedianRun_State1"),
    "MedianRun_State2": col_by_subject(hc_rows, "MedianRun_State2"),
    "Switching_Rate": col_by_subject(hc_rows, "Switching_Rate"),
}

# 表中行的顺序与标签
ROWS = [
    ("Fractional occupancy",       "FO_State2",           "%.3f"),
    ("State 1 MDT",                "MDT_State1_windows",  "%.3f"),
    ("State 2 MDT",                "MDT_State2_windows",  "%.3f"),
    ("State 1 median run length",  "MedianRun_State1",    "%.3f"),
    ("State 2 median run length",  "MedianRun_State2",    "%.3f"),
    ("Switching rate",             "Switching_Rate",      "%.4f"),
]


# ============================================================
# 3. 计算
# ============================================================
results = []
for label, key, fmt in ROWS:
    m0, m1 = M0[key], M1[key]
    common = sorted(set(m0) & set(m1))
    a = [m0[s] for s in common]
    b = [m1[s] for s in common]
    st = paired_t(a, b)

    # H 组：该指标全部有效被试
    h_vals = list(HC[key].values())

    results.append({
        "label": label, "key": key, "fmt": fmt,
        "M0": (st["m0"], st["sd0"], len(a)),
        "M1": (st["m1"], st["sd1"], len(b)),
        "H": (mean(h_vals), sd(h_vals), len(h_vals)),
        "t": st["t"], "p": st["p"], "dz": st["dz"], "n": st["n"],
    })

adj_p = bh_fdr([r["p"] for r in results])

for r, ap in zip(results, adj_p):
    r["p_adj"] = ap


# ============================================================
# 4. 输出
# ============================================================
def fmt_pm(triple, fmt):
    m, s, n = triple
    return f"{fmt % m} ± {fmt % s}"


def fmt_p(p):
    if p < 0.001:
        return "< 0.001"
    return f"{p:.3f}"


with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["Metric", "M0", "M1", "H",
                "Paired t (M1-M0)", "P value", "Adjusted P (BH)", "Cohen's dz",
                "n_paired"])
    for r in results:
        w.writerow([
            r["label"],
            fmt_pm(r["M0"], r["fmt"]),
            fmt_pm(r["M1"], r["fmt"]),
            fmt_pm(r["H"], r["fmt"]),
            f"{r['t']:.3f}",
            fmt_p(r["p"]),
            fmt_p(r["p_adj"]),
            f"{r['dz']:.3f}",
            str(r["n"]),
        ])

print("\n动态网络状态指标表（复现结果）\n")
hdr = ["Metric", "M0", "M1", "H", "Paired t", "P", "Adj.P", "dz", "n"]
print("| " + " | ".join(hdr) + " |")
print("|" + "|".join(["---"] * len(hdr)) + "|")
for r, ap in zip(results, adj_p):
    cells = [
        r["label"],
        fmt_pm(r["M0"], r["fmt"]),
        fmt_pm(r["M1"], r["fmt"]),
        fmt_pm(r["H"], r["fmt"]),
        f"{r['t']:.3f}",
        fmt_p(r["p"]),
        fmt_p(ap),
        f"{r['dz']:.3f}",
        str(r["n"]),
    ]
    print("| " + " | ".join(cells) + " |")

print(f"\n已保存：{OUT_CSV}")
print("\n说明：Adjusted P 为 6 个指标整体 BH-FDR 的结果。")
print("      若论文表中 Adjusted P 采用其他校正族，请据此调整。")
