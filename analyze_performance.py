"""
analyze_performance.py
CSC580 Assignment - Performance Analyst deliverable
=====================================================
Reads results.csv (produced by benchmark_runner.py) and produces:

  1. metrics_summary.csv     - Speedup / Efficiency / Overhead per task & size
  2. exp5_scaling_table.csv  - Node-count scaling table + Amdahl estimate
  3. Five required charts (Section 7.3 of the brief), saved as PNG:
       chart1_bar_seq_vs_mpi_per_task.png
       chart2_line_speedup_vs_datasize.png
       chart3_bar_mpi_task_breakdown.png
       chart4_efficiency_per_task.png
       chart5_amdahl_overlay.png
  4. analysis_summary.md     - plain-English write-up you can paste into
                                Section 8 (Results and Discussion) of the report

Formulas used (exactly as specified in Section 7.1 of the brief):
    Speedup    S = T_seq / T_par
    Efficiency E = S / P * 100%
    Overhead   T_overhead = P * T_par - T_seq
    Amdahl Sm  = 1 / (f + (1-f)/P)     [f = serial fraction, estimated from
                                          measured speedup at P=4, then used
                                          to draw the theoretical curve]

Run with:  python analyze_performance.py
Requires:  pandas, matplotlib   ->   pip install pandas matplotlib
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_CSV = "results.csv"
OUT_DIR = Path("performance_report")

CORE_TASKS = ["BasicStats", "Histogram", "Sorting", "Correlation", "MovingAverage", "OutlierDetection"]
TASK_DISPLAY = {
    "BasicStats": "Basic Stats",
    "Histogram": "Histogram",
    "Sorting": "Sorting",
    "Correlation": "Correlation",
    "MovingAverage": "Moving Avg",
    "OutlierDetection": "Outliers",
}
SIZE_ORDER = ["small", "medium", "large"]


def load_data():
    df = pd.read_csv(RESULTS_CSV)
    df = df[df["task"].isin(CORE_TASKS)].copy()

    # Clean up duplicates by averaging the time if the SAME (mode, size,
    # N, nodes, task) configuration was measured more than once -- this
    # includes cases where two different experiment loops both happen to
    # exercise the same configuration (e.g. EXP1-2-3 and EXP5 both run
    # mpi/medium/P=4). "experiment" is intentionally excluded from the
    # group key: it is just a label for which loop produced a row, not a
    # distinguishing property of the run itself. Including it here used to
    # let the same run be double-counted under two experiment tags, which
    # corrupted totals_by_run() and crashed chart1's reindex() on
    # duplicate task labels.
    group_cols = ["mode", "size_label", "N", "nodes", "task"]
    df = df.groupby(group_cols, as_index=False)["time_ms"].mean()

    return df

def totals_by_run(df):
    """Sum the 6 core task times -> total compute time per (mode, size, N, nodes) run."""
    return (df.groupby(["mode", "size_label", "N", "nodes"])["time_ms"]
              .sum().reset_index().rename(columns={"time_ms": "total_ms"}))


# ---------------------------------------------------------------------------
# Metrics table: per (size, task) -> Speedup / Efficiency / Overhead at P=4
# ---------------------------------------------------------------------------
def build_metrics_table(df):
    seq = df[df["mode"] == "seq"][["size_label", "N", "task", "time_ms"]] \
            .rename(columns={"time_ms": "T_seq"})
    mpi4 = df[(df["mode"] == "mpi") & (df["nodes"] == 4)][["size_label", "N", "task", "time_ms"]] \
            .rename(columns={"time_ms": "T_par"})

    merged = pd.merge(seq, mpi4, on=["size_label", "N", "task"], how="inner")
    P = 4
    merged["Speedup"] = merged["T_seq"] / merged["T_par"]
    merged["Efficiency_%"] = merged["Speedup"] / P * 100
    merged["Overhead_ms"] = P * merged["T_par"] - merged["T_seq"]
    merged["size_label"] = pd.Categorical(merged["size_label"], categories=SIZE_ORDER, ordered=True)
    merged = merged.sort_values(["size_label", "task"])
    return merged


# ---------------------------------------------------------------------------
# EXP-5 scaling table: medium dataset, P = 1 (seq baseline), 2, 3, 4
# Also estimates Amdahl's serial fraction f from the P=4 measurement and
# projects the theoretical curve for the overlay chart.
# ---------------------------------------------------------------------------
def build_scaling_table(df):
    tot = totals_by_run(df)
    medium = tot[tot["size_label"] == "medium"]

    seq_row = medium[medium["mode"] == "seq"]
    if seq_row.empty:
        raise RuntimeError("No sequential medium-size run found — cannot build scaling table.")
    T_seq = seq_row["total_ms"].iloc[0]

    rows = [{"nodes": 1, "T_ms": T_seq, "Speedup": 1.0, "Efficiency_%": 100.0}]
    for p in (2, 3, 4):
        r = medium[(medium["mode"] == "mpi") & (medium["nodes"] == p)]
        if r.empty:
            continue
        t_par = r["total_ms"].iloc[0]
        s = T_seq / t_par
        rows.append({"nodes": p, "T_ms": t_par, "Speedup": s, "Efficiency_%": s / p * 100})

    scaling = pd.DataFrame(rows).sort_values("nodes")

    # Estimate serial fraction f from the P=4 measurement:
    # S = 1 / (f + (1-f)/P)  =>  f = (P/S - 1) / (P - 1)
    row4 = scaling[scaling["nodes"] == 4]
    if not row4.empty and row4["Speedup"].iloc[0] > 1.0001:
        S4 = row4["Speedup"].iloc[0]
        P4 = 4
        f_est = (P4 / S4 - 1) / (P4 - 1)
        f_est = min(max(f_est, 0.0), 1.0)  # clamp to valid range
    else:
        f_est = 0.1  # fallback default if data is missing/degenerate

    return scaling, f_est


def amdahl_curve(f, p_values):
    return [1.0 / (f + (1 - f) / p) for p in p_values]


# ---------------------------------------------------------------------------
# Chart 1: Bar chart — Execution time, Sequential vs MPI, per task (medium N)
# ---------------------------------------------------------------------------
def chart1(metrics, out_dir):
    med = metrics[metrics["size_label"] == "medium"].set_index("task").reindex(CORE_TASKS)
    x = range(len(CORE_TASKS))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([i - width/2 for i in x], med["T_seq"], width, label="Sequential")
    ax.bar([i + width/2 for i in x], med["T_par"], width, label="MPI (4 nodes)")
    ax.set_xticks(list(x))
    ax.set_xticklabels([TASK_DISPLAY[t] for t in CORE_TASKS], rotation=20, ha="right")
    ax.set_ylabel("Execution Time (ms)")
    ax.set_title("Execution Time per Task: Sequential vs MPI (Medium dataset, 10M points)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "chart1_bar_seq_vs_mpi_per_task.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Chart 2: Line graph — Speedup vs Dataset Size (overall, P=4)
# ---------------------------------------------------------------------------
def chart2(df, out_dir):
    tot = totals_by_run(df)
    rows = []
    for size in SIZE_ORDER:
        seq_t = tot[(tot["mode"] == "seq") & (tot["size_label"] == size)]
        mpi_t = tot[(tot["mode"] == "mpi") & (tot["size_label"] == size) & (tot["nodes"] == 4)]
        if seq_t.empty or mpi_t.empty:
            continue
        speedup = seq_t["total_ms"].iloc[0] / mpi_t["total_ms"].iloc[0]
        rows.append({"size_label": size, "Speedup": speedup})
    d = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(d["size_label"], d["Speedup"], marker="o", linewidth=2)
    ax.axhline(4, color="gray", linestyle="--", linewidth=1, label="Ideal speedup (P=4)")
    ax.set_xlabel("Dataset Size")
    ax.set_ylabel("Speedup (T_seq / T_par)")
    ax.set_title("Speedup vs Dataset Size (MPI, 4 nodes)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "chart2_line_speedup_vs_datasize.png", dpi=150)
    plt.close(fig)
    return d


# ---------------------------------------------------------------------------
# Chart 3: Grouped bar chart — MPI task-level time breakdown, all 3 sizes
# ---------------------------------------------------------------------------
def chart3(df, out_dir):
    mpi4 = df[(df["mode"] == "mpi") & (df["nodes"] == 4)]
    pivot = mpi4.pivot_table(index="task", columns="size_label", values="time_ms", aggfunc="mean")
    pivot = pivot.reindex(CORE_TASKS)[[c for c in SIZE_ORDER if c in pivot.columns]]

    x = range(len(CORE_TASKS))
    n_sizes = len(pivot.columns)
    width = 0.8 / n_sizes
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, size in enumerate(pivot.columns):
        offset = (i - (n_sizes - 1) / 2) * width
        ax.bar([xi + offset for xi in x], pivot[size], width, label=size)
    ax.set_xticks(list(x))
    ax.set_xticklabels([TASK_DISPLAY[t] for t in CORE_TASKS], rotation=20, ha="right")
    ax.set_ylabel("Execution Time (ms)")
    ax.set_title("MPI Task-Level Time Breakdown (4 nodes) by Dataset Size")
    ax.legend(title="Dataset size")
    fig.tight_layout()
    fig.savefig(out_dir / "chart3_bar_mpi_task_breakdown.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Chart 4: Efficiency (%) per task, medium dataset, P=4
# ---------------------------------------------------------------------------
def chart4(metrics, out_dir):
    med = metrics[metrics["size_label"] == "medium"].set_index("task").reindex(CORE_TASKS)
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar([TASK_DISPLAY[t] for t in CORE_TASKS], med["Efficiency_%"], color="teal")
    ax.axhline(100, color="gray", linestyle="--", linewidth=1, label="Ideal efficiency (100%)")
    ax.set_ylabel("Parallel Efficiency (%)")
    ax.set_title("Parallel Efficiency per Task (4 nodes, Medium dataset)")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    ax.legend()
    for b in bars:
        ax.annotate(f"{b.get_height():.0f}%", (b.get_x() + b.get_width()/2, b.get_height()),
                    ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "chart4_efficiency_per_task.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Chart 5: Amdahl's Law overlay — theoretical vs actual speedup vs node count
# ---------------------------------------------------------------------------
def chart5(scaling, f_est, out_dir):
    p_actual = scaling["nodes"].tolist()
    s_actual = scaling["Speedup"].tolist()

    p_theory = list(range(1, 9))  # project out to 8 nodes for context
    s_theory = amdahl_curve(f_est, p_theory)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(p_theory, s_theory, linestyle="--", color="gray",
            label=f"Amdahl theoretical (f={f_est:.3f})")
    ax.plot(p_actual, s_actual, marker="o", linewidth=2, color="crimson", label="Measured speedup")
    ax.plot(p_theory, p_theory, linestyle=":", color="lightgray", label="Ideal linear speedup")
    ax.set_xlabel("Number of Nodes (P)")
    ax.set_ylabel("Speedup")
    ax.set_title("Amdahl's Law: Theoretical vs Actual Speedup (Medium dataset)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "chart5_amdahl_overlay.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Write-up generator
# ---------------------------------------------------------------------------
def write_summary(metrics, scaling, f_est, speedup_by_size, out_dir):
    med = metrics[metrics["size_label"] == "medium"]
    best_task = med.loc[med["Speedup"].idxmax()]
    worst_task = med.loc[med["Speedup"].idxmin()]
    row4 = scaling[scaling["nodes"] == 4].iloc[0]

    lines = []
    lines.append("# Performance Analysis Summary (auto-generated)\n")
    lines.append("Paste/adapt this into Section 8 (Results and Discussion) of the report.\n")

    lines.append("## EXP-1: Baseline Comparison (Medium dataset, 10M points)\n")
    lines.append(f"At P=4 nodes, overall speedup was **{row4['Speedup']:.2f}x** "
                 f"with **{row4['Efficiency_%']:.1f}% efficiency**.\n")

    lines.append("## EXP-2: Scalability (Speedup vs Dataset Size)\n")
    for _, r in speedup_by_size.iterrows():
        lines.append(f"- {r['size_label'].capitalize()} dataset: speedup = {r['Speedup']:.2f}x\n")
    lines.append("Speedup should generally increase with dataset size, since a larger "
                 "workload amortizes the fixed cost of MPI setup/communication over more "
                 "computation (a smaller *relative* overhead).\n")

    lines.append("## EXP-3: Task-Level Analysis\n")
    lines.append(f"Best-scaling task: **{TASK_DISPLAY[best_task['task']]}** "
                 f"({best_task['Speedup']:.2f}x speedup, {best_task['Efficiency_%']:.1f}% efficiency).\n")
    lines.append(f"Worst-scaling task: **{TASK_DISPLAY[worst_task['task']]}** "
                 f"({worst_task['Speedup']:.2f}x speedup, {worst_task['Efficiency_%']:.1f}% efficiency).\n")
    lines.append("Tasks needing heavy inter-node communication (Sorting's `MPI_Alltoallv` "
                 "redistribution, Moving Average's halo exchange) are expected to scale worse "
                 "than embarrassingly-parallel reductions (Basic Statistics, Histogram, "
                 "Correlation), since communication time does not shrink as data per node "
                 "shrinks — it is bounded by network latency.\n")

    lines.append("## EXP-4: Communication / Parallel Overhead\n")
    lines.append("Overhead is computed as `T_overhead = P * T_par - T_seq` for each task "
                 "(medium dataset):\n\n")
    lines.append("| Task | Overhead (ms) |\n|---|---|\n")
    for _, r in med.iterrows():
        lines.append(f"| {TASK_DISPLAY[r['task']]} | {r['Overhead_ms']:.2f} |\n")
    lines.append("\nA higher overhead value indicates more time consumed by "
                 "communication/synchronization relative to the ideal case, rather than by "
                 "the actual computation — this is what the Node Master/Algorithm Developer "
                 "should target first if the group has more time to optimize.\n")

    lines.append("## EXP-5: Node Failure Simulation (Medium dataset, P = 2, 3, 4)\n")
    lines.append("| Nodes | Total Time (ms) | Speedup | Efficiency (%) |\n|---|---|---|---|\n")
    for _, r in scaling.iterrows():
        lines.append(f"| {int(r['nodes'])} | {r['T_ms']:.2f} | {r['Speedup']:.2f} | {r['Efficiency_%']:.1f} |\n")
    lines.append(f"\nEstimated serial fraction (Amdahl's Law, from the P=4 measurement): "
                 f"**f = {f_est:.3f}**. This means roughly {f_est*100:.1f}% of the workload "
                 "is inherently sequential (data generation, root-only aggregation steps) and "
                 "cannot benefit from adding more nodes — this caps the maximum achievable "
                 f"speedup at 1/f ≈ {1/f_est if f_est > 0 else float('inf'):.1f}x even with "
                 "infinite nodes, which is the standard Amdahl's Law conclusion to discuss in "
                 "the Conclusion/Future Work section.\n")

    # encoding="utf-8" is required here: the write-up text contains non-ASCII
    # characters (e.g. "≈"), and on Windows write_text() defaults to the
    # system locale's codepage (often cp1252), which can't encode them --
    # causing a UnicodeEncodeError. Linux/Mac default to UTF-8 already, so
    # this only bites on Windows, but is set explicitly for both.
    (out_dir / "analysis_summary.md").write_text("".join(lines), encoding="utf-8")


def main():
    OUT_DIR.mkdir(exist_ok=True)
    df = load_data()
    if df.empty:
        print(f"[ERROR] No usable rows found in {RESULTS_CSV}. Run benchmark_runner.py first.")
        return

    metrics = build_metrics_table(df)
    metrics.to_csv(OUT_DIR / "metrics_summary.csv", index=False)

    scaling, f_est = build_scaling_table(df)
    scaling.to_csv(OUT_DIR / "exp5_scaling_table.csv", index=False)

    chart1(metrics, OUT_DIR)
    speedup_by_size = chart2(df, OUT_DIR)
    chart3(df, OUT_DIR)
    chart4(metrics, OUT_DIR)
    chart5(scaling, f_est, OUT_DIR)

    write_summary(metrics, scaling, f_est, speedup_by_size, OUT_DIR)

    print(f"Done. Everything written to ./{OUT_DIR}/")
    print(" - metrics_summary.csv, exp5_scaling_table.csv")
    print(" - chart1..chart5 .png (the 5 required visualizations)")
    print(" - analysis_summary.md (draft text for the report)")


if __name__ == "__main__":
    main()