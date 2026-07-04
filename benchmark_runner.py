"""
benchmark_runner.py
CSC580 Assignment - Performance Analyst deliverable
=====================================================
Runs on the MASTER laptop. Orchestrates every benchmark run required by the
assignment brief (Section 7.2, EXP-1 through EXP-5), captures the console
output of sequential.exe / mpi_parallel.exe, parses the per-task timings out
of it, and appends everything to a single tidy CSV: results.csv

This does NOT touch sequential.cpp or mpi_parallel.cpp — it just runs the
already-compiled .exe files and reads their printed output, so it stays
cleanly in the Performance Analyst's lane (Algorithm Developer owns the .cpp).

--------------------------------------------------------------------------
SETUP (edit these two things before running):
--------------------------------------------------------------------------
1. Put sequential.exe and mpi_parallel.exe in the same folder as this script
   (or edit SEQ_EXE / MPI_EXE below).
2. Make sure hostfile.txt (created by the Node Master) is in the same folder,
   one IP per line, Master's IP first. Example:
       192.168.1.10
       192.168.1.11
       192.168.1.12
       192.168.1.13

Run with:  python benchmark_runner.py
Requires:  Python 3.8+, no extra packages needed for this script.
--------------------------------------------------------------------------
"""

import csv
import re
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEQ_EXE = "sequential_analytics.exe"
MPI_EXE = "mpi_analytics.exe"
HOSTFILE = "hostfile.txt"
RESULTS_CSV = "results.csv"
delete = "nanti"
BINS = 20
WINDOW = 100

# Dataset sizes per the brief's "Data Scale Requirements" table (Section 5.2)
DATASET_SIZES = {
    "small": 1_000_000,
    "medium": 10_000_000,
    "large": 100_000_000,
}

# Per-task line format printed by both programs, e.g.:
#   "1. Basic Statistics                4.045 ms"
#   "3. Sorting (Sample Sort)         469.922 ms"
#   "Data gen + Scatter               109.639 ms"
#   "Data generation                   91.897 ms"
TASK_LINE_RE = re.compile(r"^(.+?)\s+([\d.]+)\s+ms\s*$")

# Maps whatever text prefix appears in the output to a canonical task name,
# so sequential ("3. Sorting") and MPI ("3. Sorting (Sample Sort)") rows line
# up correctly when compared later in analyze_performance.py.
TASK_CANONICAL = {
    "data generation": "DataGen",
    "data gen + scatter": "DataGen",
    "1. basic statistics": "BasicStats",
    "2. histogram": "Histogram",
    "3. sorting": "Sorting",                # matches both "3. Sorting" and "3. Sorting (Sample Sort)" via startswith
    "4. pearson correlation": "Correlation",
    "5. moving average": "MovingAverage",
    "6. outlier detection": "OutlierDetection",
}


def canonical_task_name(raw_label: str) -> str:
    key = raw_label.strip().lower()
    for prefix, canon in TASK_CANONICAL.items():
        if key.startswith(prefix):
            return canon
    return None  # not a task line we care about (skips "Total compute time...")


def read_hosts(hostfile: str):
    path = Path(hostfile)
    if not path.exists():
        print(f"[WARN] {hostfile} not found — MPI runs will be skipped.")
        return []
    lines = [ln.strip() for ln in path.read_text().splitlines()]
    # allow "IP   # comment" lines like the ones from the Node Master's setup
    hosts = [ln.split("#")[0].strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]
    return [h for h in hosts if h]


def parse_output(stdout_text: str):
    """Returns list of (task_name, time_ms) tuples found in the program output."""
    rows = []
    for line in stdout_text.splitlines():
        m = TASK_LINE_RE.match(line.strip())
        if not m:
            continue
        label, ms = m.group(1), float(m.group(2))
        task = canonical_task_name(label)
        if task:
            rows.append((task, ms))
    return rows


def run_sequential(n: int):
    print(f"  -> sequential_analytics.exe {n} {BINS} {WINDOW}")
    try:
        proc = subprocess.run(
            [SEQ_EXE, str(n), str(BINS), str(WINDOW)],
            capture_output=True, text=True, timeout=1200,
        )
    except Exception as e:
        print(f"     [ERROR] {e}")
        return None
    if proc.returncode != 0:
        print(f"     [ERROR] exit code {proc.returncode}: {proc.stderr[:300]}")
        return None
    return proc.stdout


def run_mpi(n: int, num_nodes: int, hosts: list):
    if num_nodes > len(hosts):
        print(f"     [SKIP] requested {num_nodes} nodes but only {len(hosts)} in hostfile")
        return None
    host_args = []
    for h in hosts[:num_nodes]:
        host_args += [h, "1"]
    cmd = ["mpiexec", "-hosts", str(num_nodes)] + host_args + [MPI_EXE, str(n), str(BINS), str(WINDOW)]
    print(f"  -> {' '.join(cmd)}")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    except Exception as e:
        print(f"     [ERROR] {e}")
        return None
    if proc.returncode != 0:
        print(f"     [ERROR] exit code {proc.returncode}: {proc.stderr[:300]}")
        return None
    return proc.stdout


def append_results(writer, mode, n, size_label, nodes, experiment, output_text):
    if output_text is None:
        return 0
    rows = parse_output(output_text)
    for task, ms in rows:
        writer.writerow({
            "experiment": experiment, "mode": mode, "size_label": size_label,
            "N": n, "nodes": nodes, "task": task, "time_ms": ms,
        })
    return len(rows)


def main():
    hosts = read_hosts(HOSTFILE)
    print(f"Found {len(hosts)} hosts in {HOSTFILE}: {hosts}\n")

    new_file = not Path(RESULTS_CSV).exists()
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "experiment", "mode", "size_label", "N", "nodes", "task", "time_ms"
        ])
        if new_file:
            writer.writeheader()

        # ---- EXP-1 / EXP-2 / EXP-3: sequential baseline, all 3 sizes ----
        print("=== Sequential baseline runs (all 3 dataset sizes) ===")
        for label, n in DATASET_SIZES.items():
            out = run_sequential(n)
            count = append_results(writer, "seq", n, label, 1, "EXP1-2-3", out)
            print(f"     logged {count} task rows\n")
            f.flush()

        # ---- EXP-1 / EXP-2 / EXP-3: MPI at full 4 nodes, all 3 sizes ----
        print("=== MPI runs at P=4 (all 3 dataset sizes) ===")
        for label, n in DATASET_SIZES.items():
            out = run_mpi(n, 4, hosts)
            count = append_results(writer, "mpi", n, label, 4, "EXP1-2-3", out)
            print(f"     logged {count} task rows\n")
            f.flush()

        # ---- EXP-5: Node failure simulation, medium dataset, P=2,3,4 ----
        print("=== MPI runs at P=2,3,4 on medium dataset (EXP-5: node failure sim) ===")
        medium_n = DATASET_SIZES["medium"]
        for p in (2, 3, 4):
            out = run_mpi(medium_n, p, hosts)
            count = append_results(writer, "mpi", medium_n, "medium", p, "EXP5", out)
            print(f"     logged {count} task rows\n")
            f.flush()

    print(f"\nAll done. Raw results appended to {RESULTS_CSV}")
    print("Next step: run analyze_performance.py to compute metrics and generate the 5 required charts.")


if __name__ == "__main__":
    main()
