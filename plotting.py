# plotting.py — result visualisation
# Reproduces the style of Figures 3 and 4 of the paper:
#   - left panel : PDF comparison between trained |g_θ⟩ and target
#   - right panel: loss function curves over training epochs
# Also provides a summary plot that overlays all 10 RE curves for a single
# configuration, illustrating training robustness across independent runs.

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, no tkinter — safe for multi-run loops
import matplotlib.pyplot as plt
from generator import QuantumGenerator
from config    import KS_THRESHOLD, PLOT_DPI


def plot_results(generator:    QuantumGenerator,
                 history:      dict,
                 target_probs: np.ndarray,
                 distribution: str,
                 init_mode:    str,
                 depth:        int,
                 save_path:    str = None) -> None:
    """
    Generate the PDF + loss-curves figure (style of Fig. 3 in the paper).

    Parameters
    ----------
    generator    : trained QuantumGenerator
    history      : dict with keys loss_G, loss_D, relative_entropy, ks_stat
    target_probs : empirical target distribution probabilities
    distribution : distribution name  ("lognormal" | "triangular" | "bimodal")
    init_mode    : initialisation mode ("uniform" | "normal" | "random")
    depth        : circuit depth k
    save_path    : output file path (None = display only)
    """
    n_values    = generator.n_values
    x_vals      = np.arange(n_values)
    final_probs = generator.get_probabilities()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(
        f"qGAN — {distribution}  |  init={init_mode}  |  depth k={depth}",
        fontsize=13, fontweight="bold"
    )

    # ── Left panel: PDF ───────────────────────────────────────────────────────
    ax = axes[0]
    ax.bar(x_vals, final_probs, alpha=0.7,
           label=r"trained $|g_\theta\rangle$", color="steelblue")
    ax.plot(x_vals, target_probs, "o-", color="navy",
            label="target", linewidth=2)
    ax.set_xlabel("x")
    ax.set_ylabel("p(x)")
    ax.set_title("PDF")
    ax.set_xticks(x_vals)
    ax.legend()

    # ── Right panel: loss functions ───────────────────────────────────────────
    ax = axes[1]
    epochs = range(1, len(history["loss_G"]) + 1)
    ax.plot(epochs, history["loss_G"], color="crimson",
            label=f"Gen. loss, depth {depth}", linewidth=1.5)
    ax.plot(epochs, history["loss_D"], color="salmon",
            label=f"Dis. loss, depth {depth}", linewidth=1.5, linestyle="--")
    ax.set_xlabel("time steps")
    ax.set_ylabel("loss")
    ax.set_title("Loss functions")
    ax.set_ylim(auto=True)
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
        print(f"  -> Saved: {save_path}")

    plt.close(fig)
    _print_metrics(history, distribution, init_mode, depth)


def plot_relative_entropy(histories: list,
                           labels:    list,
                           title:     str = "Relative Entropy",
                           save_path: str = None) -> None:
    """
    Plot relative entropy convergence for one or more runs (style of Fig. 4d).

    Parameters
    ----------
    histories : list of history dicts (one per run)
    labels    : legend labels, one per history
    title     : figure title
    save_path : output file path (None = display only)
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["navy", "steelblue", "cornflowerblue",
              "darkorange", "tomato", "salmon"]

    for i, (hist, label) in enumerate(zip(histories, labels)):
        epochs = range(1, len(hist["relative_entropy"]) + 1)
        ax.plot(epochs, hist["relative_entropy"],
                label=label, color=colors[i % len(colors)], linewidth=1.8)

    ax.set_xlabel("time steps")
    ax.set_ylabel("relative entropy")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
        print(f"  -> Saved: {save_path}")

    plt.close(fig)


def plot_runs_summary(all_histories: list,
                      all_ks:        list,
                      distribution:  str,
                      init_mode:     str,
                      depth:         int,
                      save_path:     str = None) -> None:
    """
    Summary plot for a single configuration across N independent runs.
    Produces a 1x2 figure:
      - Left : all N relative entropy curves overlaid (thin coloured lines)
               plus the per-run mean curve (thick dark line)
      - Right: bar chart of the final KS statistic for each run, with the
               acceptance threshold marked as a horizontal dashed line

    This figure is analogous to what Table 1 of the paper summarises
    numerically, but gives a visual impression of run-to-run variability.

    Parameters
    ----------
    all_histories : list of N history dicts, one per run
    all_ks        : list of N final KS values (floats)
    distribution  : distribution name
    init_mode     : initialisation mode
    depth         : circuit depth k
    save_path     : output file path (None = display only)
    """
    n_runs = len(all_histories)
    # Colour palette: thin lines cycle through blues, mean line is dark
    run_colors = plt.cm.Blues(np.linspace(0.35, 0.85, n_runs))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle(
        f"qGAN summary — {distribution}  |  init={init_mode}  |  depth k={depth}"
        f"  ({n_runs} runs)",
        fontsize=12, fontweight="bold"
    )

    # ── Left panel: RE curves for all runs ───────────────────────────────────
    ax = axes[0]
    re_matrix = []
    for i, hist in enumerate(all_histories):
        re_curve = np.array(hist["relative_entropy"])
        re_matrix.append(re_curve)
        epochs = range(1, len(re_curve) + 1)
        ax.plot(epochs, re_curve, color=run_colors[i], linewidth=0.9, alpha=0.7,
                label=f"run {i + 1}" if n_runs <= 5 else None)

    # Mean curve across all runs (pad shorter curves with their last value)
    max_len = max(len(r) for r in re_matrix)
    padded  = np.array([
        np.pad(r, (0, max_len - len(r)), mode="edge") for r in re_matrix
    ])
    mean_re = padded.mean(axis=0)
    ax.plot(range(1, max_len + 1), mean_re, color="navy",
            linewidth=2.2, linestyle="--", label="mean")

    ax.set_xlabel("time steps")
    ax.set_ylabel("relative entropy")
    ax.set_title("Relative Entropy — all runs")
    ax.legend(fontsize=7)

    # ── Right panel: final KS per run ─────────────────────────────────────────
    ax = axes[1]
    run_indices = np.arange(1, n_runs + 1)
    bar_colors  = [
        "steelblue" if ks < KS_THRESHOLD else "tomato" for ks in all_ks
    ]
    ax.bar(run_indices, all_ks, color=bar_colors, alpha=0.85, width=0.6)
    ax.axhline(KS_THRESHOLD, color="crimson", linewidth=1.5,
               linestyle="--", label=f"threshold = {KS_THRESHOLD}")
    ax.set_xlabel("run")
    ax.set_ylabel("KS statistic")
    ax.set_title("Final KS statistic per run")
    ax.set_xticks(run_indices)
    ax.legend(fontsize=9)

    # Annotate each bar with the KS value
    for idx, ks in zip(run_indices, all_ks):
        ax.text(idx, ks + 0.001, f"{ks:.3f}", ha="center", va="bottom",
                fontsize=7)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
        print(f"  -> Saved: {save_path}")

    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Internal utility
# ─────────────────────────────────────────────────────────────────────────────

def _print_metrics(history:      dict,
                   distribution: str,
                   init_mode:    str,
                   depth:        int) -> None:
    """Print final KS statistic and Relative Entropy for a single run."""
    final_ks = history["ks_stat"][-1]
    final_re = history["relative_entropy"][-1]
    accepted = final_ks < KS_THRESHOLD

    print(f"\n  Final metrics — {distribution} | init={init_mode} | k={depth}")
    print(f"    KS statistic    : {final_ks:.4f}  "
          f"({'ACCEPT' if accepted else 'REJECT'}, threshold={KS_THRESHOLD})")
    print(f"    Relative Entropy: {final_re:.4f}")