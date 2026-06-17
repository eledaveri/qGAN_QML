# qGAN_QML — Reproduction of Zoufal, Lucchi & Woerner (2019)

A faithful reproduction of the quantum Generative Adversarial Network (qGAN)
described in **C. Zoufal, A. Lucchi, S. Woerner, _"Quantum Generative
Adversarial Networks for Learning and Loading Random Distributions"_, npj
Quantum Information 5:103 (2019)** — [doi:10.1038/s41534-019-0223-2](https://doi.org/10.1038/s41534-019-0223-2).

A qGAN trains a shallow parametrised quantum circuit so that its measurement
statistics approximate a target probability distribution. Once trained, the
circuit acts as a reusable, low-depth quantum data-loading channel — avoiding
the `O(2ⁿ)` cost of exact state preparation.

This project reproduces the architecture and the simulation study (contributions
1–2 of the paper). The quantum-finance application (option pricing via QAE) is
out of scope.

---

## What it does

- Builds the variational quantum generator from the paper: an initial `RY`
  layer followed by `k` blocks of (cyclic `CZ` entangler + `RY` layer), on
  `n = 3` qubits (discrete grid `{0, …, 7}`).
- Trains it adversarially against a classical 3-layer PyTorch discriminator,
  using the non-saturating losses and a gradient penalty.
- Benchmarks **9 configurations** (3 distributions × 3 initialisations) with
  **10 independent runs each** — 90 training runs total — and reports
  Kolmogorov–Smirnov and relative-entropy statistics in the style of the
  paper's Table 1.

---

## Repository structure

| File | Responsibility |
|------|----------------|
| `config.py` | Hyperparameters, sampling budgets, output paths, device selection |
| `data.py` | Sampling of the three discrete target distributions |
| `generator.py` | Variational circuit + `QuantumGenerator` (sampling, probabilities, PSR gradients) |
| `discriminator.py` | Classical discriminator (`Linear 50→50→20→1` + LeakyReLU/Sigmoid) |
| `losses.py` | Generator/discriminator losses (paper Eqs. 8–9) + gradient penalty |
| `trainer.py` | Alternating training loop (discriminator AMSGRAD + generator PSR/AMSGRAD) |
| `main.py` | Benchmark entry point: runs all configurations and writes the results table |
| `plotting.py` | Per-run PDF/loss plots, RE curves, and per-configuration summary plots |

---

## Requirements & installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install torch                # see note below
```

The code uses **Qiskit 2.x**, **Qiskit Aer**, NumPy, SciPy, Matplotlib, and
**PyTorch** (for the discriminator and its optimiser). The quantum generator
runs on CPU via Qiskit Aer; PyTorch automatically uses a GPU for the
discriminator if one is available (`config.DEVICE`).

> **Note:** `requirements.txt` currently lists only the Qiskit/SciPy stack.
> PyTorch (`torch`) is imported by `config.py`, `discriminator.py`,
> `losses.py`, and `trainer.py`, so it must be installed too — either add a
> line `torch>=2.0` to `requirements.txt` or install it manually as shown
> above.

---

## Usage

Run the full benchmark (all 9 configurations, 10 runs each):

```bash
python main.py
```

This will:

- train each configuration for `N_EPOCHS` epochs, `N_RUNS` times;
- save per-run plots to `results_10_06_2024/runs/`
  (`qgan_<dist>_<init>_k<k>_run<i>.png` and the matching `re_…` plot);
- save a per-configuration summary plot to `results_10_06_2024/`;
- write aggregate statistics incrementally to
  `results_10_06_2024/results_table.csv` (rewritten after each configuration,
  so partial results survive an interruption).

The benchmark is CPU-bound on the quantum side and takes a long time: each
gradient step costs `2 · n_params` circuit evaluations (Parameter Shift Rule).
Expect the full 90-run sweep to run over the course of (several) days.

---

## Configurations

The 9 benchmarked configurations are defined as `CONFIGS` in `main.py`
(format `(distribution, init_mode, depth)`), chosen to match informative rows
of the paper's Table 1:

| Distribution | Uniform | Normal | Random |
|--------------|:-------:|:------:|:------:|
| Log-normal   | `k=2`   | `k=1`  | `k=2`  |
| Triangular   | `k=2`   | `k=1`  | `k=2`  |
| Bimodal      | `k=3`   | `k=2`  | `k=2`  |

Key hyperparameters (`config.py`): 20 000 training samples, batch size 2 000,
learning rate `1e-4`, AMSGRAD (`β₁=0.9, β₂=0.999`), 16 000 shots for
probability estimation, 800 shots per gradient evaluation, gradient-penalty
weight `λ=5`, KS acceptance threshold `0.0859`.

---

## Metrics

- **Relative entropy** — `KL(p_target ‖ p_θ)`, tracked every epoch.
- **KS statistic** — `max_j |F_θ(j) − F_target(j)|`; a run is *accepted* at
  95 % confidence when `D_KS < 0.0859`.

---

## Deviations from the paper

Five departures from the original work are tagged `[DIFF-N]` in the source:

| # | Component | Type | Impact |
|---|-----------|------|--------|
| 1 | Qiskit API | API change | None — `assign_parameters()` replaces the removed `bind_parameters()`; identical semantics |
| 2 | Random initialisation | Clarification | None — matches the reference Qiskit qGAN default (`|0⟩ⁿ` + uniform angles in `[−π, π]`, no `δ`); `δ` is applied only to the structured `uniform`/`normal` inits |
| 3 | Generator gradients | Supplement unavailable | None — Parameter Shift Rule is exact for `RY` gates; replaces the unpublished Supplementary Methods B |
| 4 | Discriminator optimiser | Implementation | None — PyTorch `Adam(amsgrad=True)` is the same AMSGRAD algorithm |
| 5 | Training length | Hardware constraint | Positive — 3 000 epochs instead of 2 000 for better convergence on the harder targets |

See the project report for the full discussion of each deviation and its
effect on the results.

---

## Results (summary)

Across the 90 runs, the reproduction confirms the paper's central claims: the
qGAN learns and loads all three target distributions, and successful runs reach
KS statistics well below the acceptance threshold. Aggregate statistics show
log-normal results matching or exceeding the paper, while the bimodal and
triangular–uniform settings exhibit larger run-to-run variability (occasional
mode collapse), traceable to the shot noise of the PSR-based gradient
estimator. Full numbers are in `results_10_06_2024/results_table.csv` and the
report.

---

## License

Not yet chosen. **MIT** or **Apache 2.0** are both good fits for an academic
reproduction (Qiskit itself is Apache 2.0). Before publishing, check your
institution's regulations on ownership of student coursework. Note that the
original paper PDF and any figures reproduced from it are **not** covered by
your own license — keep only your own code and results in the repository.
