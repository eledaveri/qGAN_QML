# generator.py — Varational quantum circuit and sampling on AerSimulator
# Reproduce Fig. 2 of the paper: k+1 layer RY + k entangling block.
#
# [DIFF-1] Qiskit 2.x: bind_parameters() → assign_parameters({ParameterVector[i]: val})
# [DIFF-2] init_mode="random": |0⟩^n + uniform parameters in [-π, π]
# [DIFF-3] Parameter Shift Rule instead of analytic gradients (Supplementary B)

import numpy as np
from scipy.stats import norm as scipy_norm
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_aer import AerSimulator

from config import (
    N_PROB_SHOTS, N_GRAD_SHOTS, INIT_DELTA,
)


# ─────────────────────────────────────────────────────────────────────────────
# Variational Circuit
# ─────────────────────────────────────────────────────────────────────────────

def build_variational_circuit(n_qubits: int,
                               depth: int) -> tuple[QuantumCircuit, ParameterVector]:
    """
    Builds the variational circuit described in Fig. 2 of the paper.

    Structure
    ---------
    - 1 initial layer of RY(θ)
    - depth repetitions of: [entangling block CZ cyclic] + [layer RY(θ)]
    - Total parameters: (depth + 1) * n_qubits  (only RY are parametric)

    [DIFF-1] In Qiskit 2.x the parameters are assigned with assign_parameters().
    """
    n_params = (depth + 1) * n_qubits
    theta = ParameterVector("θ", n_params)
    qc = QuantumCircuit(n_qubits)

    param_idx = 0

    # First layer RY
    for i in range(n_qubits):
        qc.ry(theta[param_idx], i)
        param_idx += 1

    # k repetitions: entangling block + layer RY
    for _ in range(depth):
        # Entangling block: CZ cyclic  i → (i+1) mod n
        for i in range(n_qubits):
            qc.cz(i, (i + 1) % n_qubits)
        # Layer RY
        for i in range(n_qubits):
            qc.ry(theta[param_idx], i)
            param_idx += 1

    qc.measure_all()
    return qc, theta


# ─────────────────────────────────────────────────────────────────────────────
# Quantum Generator
# ─────────────────────────────────────────────────────────────────────────────

class QuantumGenerator:
    """
    Quantum Generator: variational circuit + sampling on AerSimulator.

    init_mode
    ---------
    "uniform"  → |ψ_in⟩ = uniform (RY(π/2) ≈ Hadamard)
    "normal"   → |ψ_in⟩ ≈ gaussian discrete (angles from normal CDF)
    "random"   → |0⟩^n with uniform parameters in [-π, π]  [DIFF-2]

    In all cases the parameters are perturbed by ±INIT_DELTA around
    the initial state, to break the symmetry.
    """

    def __init__(self,
                 n_qubits: int,
                 depth: int,
                 init_mode: str = "uniform",
                 target_mean: float = None,
                 target_std: float = None,
                 n_values: int = None):
        self.n_qubits = n_qubits
        self.depth    = depth
        self.n_values = n_values or 2 ** n_qubits
        self.simulator = AerSimulator()

        self.base_circuit, self.theta_vec = build_variational_circuit(n_qubits, depth)
        n_params = (depth + 1) * n_qubits

        if init_mode == "uniform":
            base = self._uniform_angles(n_params)
            self.params = base + np.random.uniform(-INIT_DELTA, INIT_DELTA, n_params)
        elif init_mode == "normal":
            base = self._normal_angles(n_params, target_mean, target_std)
            self.params = base + np.random.uniform(-INIT_DELTA, INIT_DELTA, n_params)
        elif init_mode == "random":
            self.params = np.random.uniform(-np.pi, np.pi, n_params)  # [DIFF-2]
        else:
            raise ValueError(
                f"init_mode '{init_mode}' not recognized. "
                "Choose from: 'uniform', 'normal', 'random'."
            )

    # ── Initializers for the angles ──────────────────────────────────────────

    def _uniform_angles(self, n_params: int) -> np.ndarray:
        """RY(π/2) in the first layer → uniform distribution (≈ Hadamard)."""
        angles = np.zeros(n_params)
        angles[: self.n_qubits] = np.pi / 2
        return angles

    def _normal_angles(self, n_params: int,
                       mean: float, std: float) -> np.ndarray:
        """
        Approximates a discrete Gaussian by setting the angles according to the normal CDF.
        Simplified solution  (Supplementary Methods A non pubblicato).  [DIFF-3]
        """
        angles = np.zeros(n_params)
        x_vals = np.arange(self.n_values)

        if mean is not None and std is not None:
            probs = scipy_norm.pdf(x_vals, loc=mean, scale=std)
            probs /= probs.sum()
        else:
            probs = np.ones(self.n_values) / self.n_values

        cdf = np.cumsum(probs)
        for i in range(self.n_qubits):
            idx = min(i, len(cdf) - 1)
            angles[i] = 2 * np.arcsin(np.sqrt(np.clip(cdf[idx], 0, 1)))

        return angles

    # ── Sampling ─────────────────────────────────────────────────────────

    def sample(self, n_shots: int) -> np.ndarray:
        """
        Executes the circuit and returns an array of integers (sampled values).
        Qiskit returns bitstrings in little-endian order → inversion [::-1].
        """
        param_dict  = dict(zip(self.theta_vec, self.params))
        bound_qc    = self.base_circuit.assign_parameters(param_dict)  # [DIFF-1]
        job         = self.simulator.run(bound_qc, shots=n_shots)
        counts      = job.result().get_counts()

        samples = []
        for bitstring, count in counts.items():
            value = int(bitstring[::-1], 2)
            samples.extend([value] * count)

        return np.array(samples)

    def get_probabilities(self) -> np.ndarray:
        """Estimates p_j^θ via sampling (N_PROB_SHOTS shots)."""
        samples = self.sample(N_PROB_SHOTS)
        probs   = np.zeros(self.n_values)
        for s in samples:
            if s < self.n_values:
                probs[s] += 1
        return probs / probs.sum()

    # ── Gradients via Parameter Shift Rule ───────────────────────────────────

    def compute_gradient(self, loss_fn, n_shots: int = N_GRAD_SHOTS) -> np.ndarray:
        """
        Computes ∂L/∂θ_i = [L(θ_i + π/2) - L(θ_i - π/2)] / 2  for each i.

        [DIFF-3] The paper cites analytic gradients (Supplementary Methods B,
        not published). The PSR is mathematically equivalent for RY gates.
        """
        grad  = np.zeros_like(self.params)
        shift = np.pi / 2
        original_params = self.params.copy()

        for i in range(len(self.params)):
            # θ_i + π/2
            self.params = original_params.copy()
            self.params[i] += shift
            loss_plus = loss_fn(self.sample(n_shots // 2))

            # θ_i - π/2
            self.params = original_params.copy()
            self.params[i] -= shift
            loss_minus = loss_fn(self.sample(n_shots // 2))

            grad[i] = (loss_plus - loss_minus) / 2

        self.params = original_params
        return grad