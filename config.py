# config.py — qGAN's hyperparameters, training settings and experiment configurations
# Zoufal, Lucchi, Woerner - "Quantum Generative Adversarial Networks
# for learning and loading random distributions", npj Quantum Information (2019)

# ── Architecture ─────────────────────────────────────────────────────────────
N_QUBITS      = 3
DISCRIMINATOR_INPUT_DIM = 50

# ── Training ─────────────────────────────────────────────────────────────────
N_EPOCHS      = 3000        # paper 2000
BATCH_SIZE    = 2000
LR            = 1e-4

# ── Adam / AMSGRAD ────────────────────────────────────────────────────────────
BETA1         = 0.9
BETA2         = 0.999
EPS           = 1e-8

# ── Sampling ─────────────────────────────────────────────────────────────
N_TRAINING_SAMPLES  = 20_000
N_PROB_SHOTS        = 16_000   # doubled: reduce the variance in the more complex configurations [8000]
N_GRAD_SHOTS        = 800      # doubled: smoother and cleaner gradients [400]
N_EVAL_SAMPLES      = 1_000

# ── Gradient penalty ─────────────────────────────────────────────────────────
LAMBDA_GP     = 5.0

# ── KS test ──────────────────────────────────────────────────────────────────
KS_THRESHOLD  = 0.0859

# ── Generator initialization ───────────────────────────────────────────────
INIT_DELTA    = 0.1

# ── Experiments ───────────────────────────────────────────────────────────────
# Each tuple: (distribution, init_mode, depth)
CONFIGS = [
    ("lognormal",  "uniform", 2),   # already working — unchanged
    ("triangular", "uniform", 2),   # fix: was (normal, k=1) → mode collapse guaranteed
    ("bimodal",    "uniform", 3),   # k=3 ok, needs more epoch and shot budget
]

# ── Device ────────────────────────────────────────────────────────────────────
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Output ────────────────────────────────────────────────────────────────────
OUTPUT_DIR    = "results_10_06_2024"
PLOT_DPI      = 150