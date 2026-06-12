# data.py — training data generation 
# Discrete distributions tested (Fig. 3): log-normal, triangolare, bimodale.

import numpy as np
from scipy.stats import triang


def generate_training_data(distribution: str,
                            n_samples: int = 20_000,
                            n_values: int = 8) -> np.ndarray:
    """
    Generate discrete training data from the specified distribution.
    The raw samples are generated from the continuous distribution, then
    clipped to [0, n_values-1] and rounded to integers. 
    Args:
        distribution: "lognormal", "triangular" o "bimodal"
        n_samples: number of samples to generate
        n_values: number of discrete values (0, 1, ..., n_values-1)
    Returns:
        Array of integers with the discrete samples.
    """
    if distribution == "lognormal":
        # μ=1, σ=1 (log-normal)
        raw = np.random.lognormal(mean=1.0, sigma=1.0, size=n_samples)

    elif distribution == "triangular":
        # lower=0, upper=7, mode=2
        raw = triang.rvs(c=2 / 7, loc=0, scale=7, size=n_samples)

    elif distribution == "bimodal":
        # Two gaussians: μ1=0.5, σ1=1  e  μ2=3.5, σ2=0.5
        n1 = n_samples // 2
        g1 = np.random.normal(0.5, 1.0, n1)
        g2 = np.random.normal(3.5, 0.5, n_samples - n1)
        raw = np.concatenate([g1, g2])
        np.random.shuffle(raw)

    else:
        raise ValueError(
            f"Distribution '{distribution}' not recognized. "
            "Choose from: 'lognormal', 'triangular', 'bimodal'."
        )

    # Truncate to [0, n_values-1] and round to integers
    clipped = np.clip(raw, 0, n_values - 1)
    return np.round(clipped).astype(int)


def empirical_probs(data: np.ndarray, n_values: int) -> np.ndarray:
    """Estimate empirical probabilities from an array of integer samples."""
    counts = np.bincount(data, minlength=n_values)
    return counts / counts.sum()