"""LoRA² utilities: discretized exponential, quantile rank, regularization losses.

Reference: "Not All Layers Are Created Equal: Adaptive LoRA Ranks for
Personalized Image Generation" (Shenaj et al., arXiv:2603.21884)
"""

import math
import torch
from typing import Optional


def discretized_exponential(x: torch.Tensor, nu: torch.Tensor) -> torch.Tensor:
    """Compute f(x; ν) = exp(−ν·x) − exp(−ν·(x+1)).

    A monotonically decreasing importance distribution over rank indices.
    Rank index 1 has the highest importance; later indices have progressively
    lower importance. This ordering allows the effective rank to be determined
    by truncating at a quantile threshold.

    Args:
        x: Rank indices (1-based), shape (*,).
        nu: Positive scalar controlling the distribution spread.
            Larger ν → faster decay → lower effective rank.

    Returns:
        Importance weights f(x; ν), same shape as x.
    """
    return torch.exp(-nu * x) - torch.exp(-nu * (x + 1))


def compute_effective_rank(
    nu: float | torch.Tensor,
    quantile: float = 0.9,
    max_rank: int = 512,
) -> int:
    """Compute effective rank D from ν.

    D = min{k : Σ_{j=1}^{k} f(j; ν) ≥ quantile}

    Closed-form: D = ⌈−ln(1−q) / ν⌉  (from CDF of the exponential)

    Args:
        nu: Learnable rate parameter (positive scalar).
        quantile: Target cumulative probability (default 0.9).
        max_rank: Upper bound on returned rank.

    Returns:
        Effective rank D, clamped to [1, max_rank].
    """
    nu_val = nu.item() if isinstance(nu, torch.Tensor) else float(nu)
    if nu_val <= 0:
        return max_rank
    d = math.ceil(-math.log(1.0 - quantile) / nu_val)
    return min(max(d, 1), max_rank)


def compute_lambda_diag(nu: torch.Tensor, d: int) -> torch.Tensor:
    """Compute importance diagonal Λ = [f(1;ν), ..., f(D;ν)].

    Args:
        nu: Positive scalar tensor.
        d: Number of rank positions to compute.

    Returns:
        1-D tensor of shape (d,) with importance values.
    """
    indices = torch.arange(1, d + 1, device=nu.device, dtype=nu.dtype)
    return discretized_exponential(indices, nu)


def compute_nu_target(r_target: int, quantile: float = 0.9) -> float:
    """Compute the ν value that corresponds to a target rank.

    ν_target = −ln(1−q) / r_target

    Used for rank regularization: L_reg = |ν − ν_target|.

    Args:
        r_target: Desired target rank.
        quantile: Quantile used for rank computation (default 0.9).

    Returns:
        The ν value such that compute_effective_rank(ν_target) ≈ r_target.
    """
    return -math.log(1.0 - quantile) / r_target


def rescaled_kaiming_std(nu: torch.Tensor, d: int) -> float:
    """Compute rescaled Kaiming initialization standard deviation for A weights.

    Per the paper (Section 3.3):
        std = √2 / √(Σ_{j=1}^{D} f²(j; ν))

    This counteracts the gradient rescaling effect of the Λ diagonal,
    ensuring stable convergence.

    Args:
        nu: Current ν parameter tensor.
        d: Current effective rank.

    Returns:
        Standard deviation for weight initialization.
    """
    indices = torch.arange(1, d + 1, device=nu.device, dtype=nu.dtype)
    f_vals = discretized_exponential(indices, nu)
    sum_sq = (f_vals ** 2).sum().item()
    if sum_sq <= 0:
        return math.sqrt(2.0)
    return math.sqrt(2.0) / math.sqrt(sum_sq)
