# losses.py — loss functions and gradient penalty
# Implements the Eq. 8 and 9 of the paper + gradient penalty (section Methods).

import torch
import numpy as np
from discriminator import Discriminator
from config import LAMBDA_GP


def generator_loss(discriminator: Discriminator,
                   gen_samples: np.ndarray,
                   n_values: int) -> torch.Tensor:
    """
    Loss of the generator  [Eq. 8]:
        L_G = -1/m  Σ  log D(g^l)

    The generator maximizes the probability that the discriminator
    classifies its samples as real.
    """
    x_gen = discriminator.encode(gen_samples, n_values)
    d_gen = discriminator(x_gen)
    return -torch.mean(torch.log(d_gen + 1e-8))


def discriminator_loss(discriminator: Discriminator,
                       real_samples: np.ndarray,
                       gen_samples: np.ndarray,
                       n_values: int) -> torch.Tensor:
    """
    Loss of the discriminator  [Eq. 9]:
        L_D = -1/m  Σ  [ log D(x^l) + log(1 - D(g^l)) ]

    The discriminator maximizes the probability of distinguishing
    real samples from generated samples.
    """
    x_real = discriminator.encode(real_samples, n_values)
    x_gen  = discriminator.encode(gen_samples,  n_values)

    d_real = discriminator(x_real)
    d_gen  = discriminator(x_gen)

    return -torch.mean(
        torch.log(d_real + 1e-8) + torch.log(1 - d_gen + 1e-8)
    )


def gradient_penalty(discriminator: Discriminator,
                     real_samples: np.ndarray,
                     gen_samples: np.ndarray,
                     n_values: int,
                     lambda_gp: float = LAMBDA_GP) -> torch.Tensor:
    """
    Gradient penalty on the discriminator to stabilize the training.
    [Paper, section Methods: "gradient penalty on the discriminator's loss function"]
    Ref. Kodali et al. (2017), Roth et al. (2017).

    Penalizes the deviations of the gradient norm from the value 1
    on points interpolated between real and generated samples.
    """
    x_real = discriminator.encode(real_samples, n_values)
    x_gen  = discriminator.encode(gen_samples,  n_values)

    # Convex interpolation between real and generated samples
    batch = min(len(real_samples), len(gen_samples))
    alpha = torch.rand(batch, 1, device=x_real.device)
    interpolated = (alpha * x_real[:batch] + (1 - alpha) * x_gen[:batch])
    interpolated = interpolated.requires_grad_(True)

    d_interp = discriminator(interpolated)
    grad = torch.autograd.grad(
        outputs=d_interp,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True,
        retain_graph=True,
    )[0]

    gp = lambda_gp * ((grad.norm(2, dim=1) - 1) ** 2).mean()
    return gp