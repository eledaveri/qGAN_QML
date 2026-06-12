# trainer.py — qGAN training loop
# Alternates the discriminator update(PyTorch/AMSGRAD) and the
# quantum generator update(Parameter Shift Rule + AMSGRAD manuale).
#
# [DIFF-4] Uses Adam(amsgrad=True) of PyTorch, same as AMSGRAD
#          with lr=1e-4 described in the paper (section Methods).
# [DIFF-5] Default: 3000 epochs and batch=2000 (paper: 2000/2000)

import numpy as np
import torch
import torch.optim as optim

from generator     import QuantumGenerator
from discriminator import Discriminator
from losses        import generator_loss, discriminator_loss, gradient_penalty
from data          import generate_training_data, empirical_probs
from config        import (
    N_EPOCHS, BATCH_SIZE, LR,
    BETA1, BETA2, EPS,
    N_GRAD_SHOTS, N_EVAL_SAMPLES, N_TRAINING_SAMPLES,
    KS_THRESHOLD,
)


def train_qgan(distribution: str  = "lognormal",
               init_mode:    str  = "uniform",
               n_qubits:     int  = 3,
               depth:        int  = 1,
               n_epochs:     int  = N_EPOCHS,
               batch_size:   int  = BATCH_SIZE,
               lr:           float = LR,
               verbose:      bool  = True):
    """
    Principal training loop for the qGAN.

    Parameters
    ----------
    distribution : "lognormal" | "triangular" | "bimodal"
    init_mode    : "uniform" | "normal" | "random"
    n_qubits     : number of qubits in the generator
    depth        : k = number of entangling blocks
    n_epochs     : number of epochs  [DIFF-5]
    batch_size   : size of the batch  [DIFF-5]
    lr           : learning rate for AMSGRAD
    verbose      : print progress every 50 epochs

    Returns
    -------
    generator    : QuantumGenerator trained
    discriminator: Discriminator trained
    history      : dict with loss_G, loss_D, relative_entropy, ks_stat
    target_probs : empirical probabilities of the training dataset
    """
    n_values = 2 ** n_qubits

    # ── Dataset ───────────────────────────────────────────────────────────────
    training_data = generate_training_data(distribution,
                                            n_samples=N_TRAINING_SAMPLES,
                                            n_values=n_values)
    target_mean = training_data.mean()
    target_std  = training_data.std()

    # ── Models ───────────────────────────────────────────────────────────────
    generator     = QuantumGenerator(n_qubits, depth, init_mode,
                                     target_mean=target_mean,
                                     target_std=target_std,
                                     n_values=n_values)
    discriminator = Discriminator()

    # ── Optimizer for the discriminator: AMSGRAD [DIFF-4] ────────────────────────
    optimizer_D = optim.Adam(discriminator.parameters(),
                             lr=lr, amsgrad=True)

    # ── Manual AMSGRAD state for the generator ───────────────────────────────
    n_params   = len(generator.params)
    m_t        = np.zeros(n_params)   # primo momento
    v_t        = np.zeros(n_params)   # secondo momento
    v_hat_max  = np.zeros(n_params)   # v̂_max per AMSGRAD

    history = {
        "loss_G":          [],
        "loss_D":          [],
        "relative_entropy":[],
        "ks_stat":         [],
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f" qGAN  |  dist={distribution}  init={init_mode}  k={depth}")
        print(f" Epoche={n_epochs}  Batch={batch_size}  lr={lr}")
        print(f"{'='*60}")

    # ── Epoch loop ────────────────────────────────────────────────────────────
    for epoch in range(n_epochs):
        np.random.shuffle(training_data)
        epoch_loss_G, epoch_loss_D = [], []

        # ── Batch loop ────────────────────────────────────────────────────────
        for start in range(0, len(training_data), batch_size):
            real_batch = training_data[start : start + batch_size]
            if len(real_batch) < 10:
                continue

            gen_samples = generator.sample(len(real_batch))

            # ── Update Discriminator ───────────────────────────────────────
            optimizer_D.zero_grad()
            loss_D = discriminator_loss(discriminator, real_batch,
                                         gen_samples, n_values)
            gp     = gradient_penalty(discriminator, real_batch,
                                       gen_samples, n_values)
            (loss_D + gp).backward()
            optimizer_D.step()
            epoch_loss_D.append(loss_D.item())

            # ── Update Generator (Parameter Shift Rule) ────────────────────
            def gen_loss_fn(samples: np.ndarray) -> float:
                """Loss scalare del generatore per il PSR."""
                with torch.no_grad():
                    x = discriminator.encode(samples, n_values)
                    d = discriminator(x)
                    return -torch.mean(torch.log(d + 1e-8)).item()

            grad = generator.compute_gradient(gen_loss_fn,
                                               n_shots=N_GRAD_SHOTS)

            # Manual AMSGRAD
            t       = epoch + 1
            m_t     = BETA1 * m_t + (1 - BETA1) * grad
            v_t     = BETA2 * v_t + (1 - BETA2) * grad ** 2
            m_hat   = m_t / (1 - BETA1 ** t)
            v_hat   = v_t / (1 - BETA2 ** t)
            v_hat_max = np.maximum(v_hat_max, v_hat)
            generator.params -= lr * m_hat / (np.sqrt(v_hat_max) + EPS)

            # Current generator loss (post-update)
            loss_G_val = gen_loss_fn(generator.sample(len(real_batch)))
            epoch_loss_G.append(loss_G_val)

        # ── Metrics per epoch ────────────────────────────────────────────────
        gen_probs    = generator.get_probabilities()
        target_fresh = generate_training_data(distribution,
                                               n_samples=10_000,
                                               n_values=n_values)
        target_probs = empirical_probs(target_fresh, n_values)

        # Relative Entropy (KL divergence: target ‖ gen)
        safe_gen    = np.clip(gen_probs,    1e-10, 1)
        safe_target = np.clip(target_probs, 1e-10, 1)
        rel_entropy = float(np.sum(safe_target * np.log(safe_target / safe_gen)))

        # KS statistic = max|CDF_gen - CDF_target| [DIFF-3: cdf invece di pdf per KS]
        gen_cdf    = np.cumsum(gen_probs)
        target_cdf = np.cumsum(target_probs)
        ks_stat    = float(np.max(np.abs(gen_cdf - target_cdf)))

        mean_G = float(np.mean(epoch_loss_G)) if epoch_loss_G else 0.0
        mean_D = float(np.mean(epoch_loss_D)) if epoch_loss_D else 0.0

        history["loss_G"].append(mean_G)
        history["loss_D"].append(mean_D)
        history["relative_entropy"].append(rel_entropy)
        history["ks_stat"].append(ks_stat)

        if verbose and (epoch % 50 == 0 or epoch == n_epochs - 1):
            accepted = "✓" if ks_stat < KS_THRESHOLD else "✗"
            print(f"  Epoch {epoch+1:4d}/{n_epochs} | "
                  f"L_G={mean_G:.4f}  L_D={mean_D:.4f} | "
                  f"RE={rel_entropy:.4f}  KS={ks_stat:.4f} {accepted}")

    return generator, discriminator, history, target_probs