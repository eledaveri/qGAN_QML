# main.py — entry point
# Runs the qGAN benchmark across all configurations defined below.
# For each configuration, N_RUNS independent training runs are executed.
# Per-run plots (PDF+losses, relative entropy) are saved individually.
# After all runs for a configuration are complete, a summary plot is produced
# showing all RE curves overlaid and a bar chart of the final KS statistics.
# Aggregate statistics (mean ± std of KS and RE) are written to a CSV file,
# mirroring the structure of Table 1 in the paper.
#
# Reproduction of: Zoufal, Lucchi, Woerner
# "Quantum Generative Adversarial Networks for learning and loading
#  random distributions", npj Quantum Information (2019)
#
# Deviations from the original paper:
#   [DIFF-1] Qiskit 2.x: assign_parameters() instead of bind_parameters()
#   [DIFF-2] init_mode="random": |0⟩^n + params uniform in [-π, π]
#   [DIFF-3] Parameter Shift Rule (exact gradient for RY) instead of
#            Supplementary Methods B (not publicly available)
#   [DIFF-4] Adam(amsgrad=True) from PyTorch ≡ AMSGRAD lr=1e-4 of the paper
#   [DIFF-5] Training extended to 3000 epochs for harder configs (paper: 2000)

import os
import csv
import warnings
import numpy as np
warnings.filterwarnings("ignore")

from trainer  import train_qgan
from plotting import plot_results, plot_relative_entropy, plot_runs_summary
from config   import N_EPOCHS, BATCH_SIZE, LR, OUTPUT_DIR, KS_THRESHOLD

# ── Number of independent runs per configuration ─────────────────────────────
N_RUNS = 10

# ── The 9 configurations to benchmark ────────────────────────────────────────
# Format: (distribution, init_mode, depth)
# Selected to match the most informative rows of Table 1 in the paper,
# covering all three distributions and all three initialisation strategies.
CONFIGS = [
    # Log-normal
    ("lognormal",  "uniform", 2),
    ("lognormal",  "normal",  1),
    ("lognormal",  "random",  2),
    # Triangular
    ("triangular", "uniform", 2),
    ("triangular", "normal",  1),
    ("triangular", "random",  2),
    # Bimodal
    ("bimodal",    "uniform", 3),
    ("bimodal",    "normal",  2),
    ("bimodal",    "random",  2),
]

# ── Output paths ──────────────────────────────────────────────────────────────
RUNS_DIR  = os.path.join(OUTPUT_DIR, "runs")               # per-run plots
STATS_CSV = os.path.join(OUTPUT_DIR, "results_table.csv")  # aggregate table


def run_config(distribution, init_mode, depth):
    """
    Execute N_RUNS independent training runs for one configuration.

    For each run:
      - trains the qGAN with train_qgan()
      - saves the PDF+loss plot and the RE plot to RUNS_DIR
      - records the final KS and RE values

    After all runs:
      - saves a summary plot (overlaid RE curves + KS bar chart)

    Returns a dict with aggregate statistics for the CSV table.
    """
    ks_list       = []
    re_list       = []
    accepted_list = []
    all_histories = []

    print(f"\n{'=' * 60}")
    print(f"  {distribution.upper()} | init={init_mode} | k={depth}")
    print(f"  Running {N_RUNS} independent training runs")
    print(f"{'=' * 60}")

    for run_idx in range(N_RUNS):
        print(f"\n  -- Run {run_idx + 1}/{N_RUNS} " + "-" * 40)

        generator, discriminator, history, target_probs = train_qgan(
            distribution=distribution,
            init_mode=init_mode,
            depth=depth,
            n_epochs=N_EPOCHS,
            batch_size=BATCH_SIZE,
            lr=LR,
            verbose=True,
        )

        # Record final metrics
        ks_final = history["ks_stat"][-1]
        re_final = history["relative_entropy"][-1]
        accepted = ks_final < KS_THRESHOLD

        ks_list.append(ks_final)
        re_list.append(re_final)
        accepted_list.append(accepted)
        all_histories.append(history)

        status = "ACCEPTED" if accepted else "REJECTED"
        print(f"\n  -> Run {run_idx + 1}: KS={ks_final:.4f}  "
              f"RE={re_final:.4f}  [{status}]")

        # Save per-run plots
        base_name = (f"qgan_{distribution}_{init_mode}"
                     f"_k{depth}_run{run_idx + 1}")

        plot_results(
            generator=generator,
            history=history,
            target_probs=target_probs,
            distribution=distribution,
            init_mode=init_mode,
            depth=depth,
            save_path=os.path.join(RUNS_DIR, f"{base_name}.png"),
        )

        plot_relative_entropy(
            histories=[history],
            labels=[f"{distribution} | init={init_mode} | k={depth} | run {run_idx + 1}"],
            title=f"Relative Entropy — {distribution} (run {run_idx + 1})",
            save_path=os.path.join(RUNS_DIR, f"re_{base_name}.png"),
        )

    # ── Summary plot for this configuration ───────────────────────────────────
    summary_name = f"summary_{distribution}_{init_mode}_k{depth}.png"
    plot_runs_summary(
        all_histories=all_histories,
        all_ks=ks_list,
        distribution=distribution,
        init_mode=init_mode,
        depth=depth,
        save_path=os.path.join(OUTPUT_DIR, summary_name),
    )

    # ── Aggregate statistics ──────────────────────────────────────────────────
    ks_arr     = np.array(ks_list)
    re_arr     = np.array(re_list)
    n_accepted = sum(accepted_list)

    print(f"\n  -- Summary: {distribution} | {init_mode} | k={depth} --")
    print(f"     Accepted  : {n_accepted}/{N_RUNS}")
    print(f"     mu_KS={ks_arr.mean():.4f}   sigma_KS={ks_arr.std():.4f}")
    print(f"     mu_RE={re_arr.mean():.4f}   sigma_RE={re_arr.std():.4f}")

    return {
        "distribution": distribution,
        "init_mode":    init_mode,
        "depth":        depth,
        "n_accepted":   n_accepted,
        "mu_ks":        float(ks_arr.mean()),
        "sigma_ks":     float(ks_arr.std()),
        "mu_re":        float(re_arr.mean()),
        "sigma_re":     float(re_arr.std()),
        # Individual run values stored as semicolon-separated strings in CSV
        "ks_values":    ks_list,
        "re_values":    re_list,
    }


def save_csv(results):
    """
    Write aggregate results to a CSV file.
    Structure mirrors Table 1 of the paper (mu, sigma, n_accepted for KS and RE).
    The file is rewritten after every completed configuration so that data is
    preserved even if the process is interrupted mid-run.
    """
    fieldnames = [
        "distribution", "init_mode", "depth",
        "n_accepted",
        "mu_ks", "sigma_ks",
        "mu_re", "sigma_re",
        "ks_values", "re_values",
    ]
    with open(STATS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = dict(r)
            row["ks_values"] = ";".join(f"{v:.6f}" for v in r["ks_values"])
            row["re_values"] = ";".join(f"{v:.6f}" for v in r["re_values"])
            writer.writerow(row)
    print(f"\n  Table saved to: {STATS_CSV}")


def print_summary_table(results):
    """Print the aggregate results table to stdout (style of Table 1)."""
    print("\n" + "=" * 75)
    print(f"  {'Distrib.':<12} {'Init':<9} {'k':>2}  "
          f"{'n_acc':>5}  {'mu_KS':>7}  {'sig_KS':>7}  "
          f"{'mu_RE':>7}  {'sig_RE':>7}")
    print("=" * 75)
    for r in results:
        print(f"  {r['distribution']:<12} {r['init_mode']:<9} {r['depth']:>2}  "
              f"{r['n_accepted']:>5}  {r['mu_ks']:>7.4f}  {r['sigma_ks']:>7.4f}  "
              f"{r['mu_re']:>7.4f}  {r['sigma_re']:>7.4f}")
    print("=" * 75)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(RUNS_DIR,   exist_ok=True)

    print("=" * 60)
    print(" qGAN — Zoufal, Lucchi, Woerner (npj Quantum Info, 2019)")
    print(" Benchmark: 10 runs per configuration, 9 configurations")
    print("=" * 60)
    print(f"\n  Epochs     : {N_EPOCHS}")
    print(f"  Batch size : {BATCH_SIZE}")
    print(f"  Lr         : {LR}")
    print(f"  Runs/config: {N_RUNS}")
    print(f"  Run plots  : {RUNS_DIR}")
    print(f"  Summaries  : {OUTPUT_DIR}")
    print(f"  CSV        : {STATS_CSV}\n")

    all_results = []

    for distribution, init_mode, depth in CONFIGS:
        result = run_config(distribution, init_mode, depth)
        all_results.append(result)

        # Save CSV incrementally after each completed configuration
        save_csv(all_results)

    print_summary_table(all_results)

    print("\n" + "=" * 60)
    print(" Benchmark complete.")
    print(f"  Run plots  : {RUNS_DIR}")
    print(f"  Summaries  : {OUTPUT_DIR}")
    print(f"  CSV table  : {STATS_CSV}")
    print("=" * 60)


if __name__ == "__main__":
    main()