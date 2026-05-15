"""
PiSSA (Principal Singular values and Singular vectors Adaptation) utilities.

Provides standalone functions for PiSSA initialization, fast randomized SVD,
QPiSSA iterative quantization, and PiSSA→LoRA conversion.

References:
    Meng et al., "PiSSA: Principal Singular Values and Singular Vectors
    Adaptation of Large Language Models", arXiv:2404.02948, 2024.
"""
import torch
from typing import Tuple, Optional
from ..logging import logger


@torch.no_grad()
def pissa_svd(
    weight: torch.Tensor,
    r: int,
    fast_niter: int = 0,
    n_oversamples: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Perform PiSSA decomposition of a weight matrix.

    Decomposes *weight* into principal components (A, B) and a residual
    matrix W^res such that ``W = W^res + A @ B``.

    Args:
        weight: Pretrained weight matrix of shape ``(out, in)``.
        r: Rank for low-rank decomposition.
        fast_niter: If > 0, use fast randomized SVD with this many power iterations.
        n_oversamples: Oversampling parameter for randomized SVD.

    Returns:
        ``(A, B, W_res)`` tuple where:
            - *A*: ``(out, r)`` principal left singular vectors scaled by sqrt(S)
            - *B*: ``(r, in)`` principal right singular vectors scaled by sqrt(S)
            - *W_res*: ``(out, in)`` residual matrix (frozen during training)
    """
    dtype = weight.float()
    m, n = weight.shape

    if fast_niter <= 0:
        # Exact SVD
        U, S, Vh = torch.linalg.svd(dtype, full_matrices=False)
    else:
        # Fast randomized SVD
        r_oversampled = min(r + n_oversamples, min(m, n))
        Omega = torch.randn((n, r_oversampled), dtype=dtype, device=weight.device)
        Y = dtype @ Omega
        for _ in range(fast_niter):
            Y = dtype @ (dtype.T @ Y)
        Q, _ = torch.linalg.qr(Y)
        B_proj = Q.T @ dtype
        Ub, S, Vh = torch.linalg.svd(B_proj, full_matrices=False)
        U = Q @ Ub

    # Extract top-r principal components
    U_r = U[:, :r]
    S_r = S[:r]
    Vh_r = Vh[:r, :]

    sqrt_S_r = torch.sqrt(S_r)
    A = U_r * sqrt_S_r.unsqueeze(0)       # (m, r)
    B = sqrt_S_r.unsqueeze(1) * Vh_r      # (r, n)
    W_res = weight - A @ B

    return A.to(weight.dtype), B.to(weight.dtype), W_res.to(weight.dtype)


@torch.no_grad()
def convert_pissa_to_lora(
    A_trained: torch.Tensor,
    B_trained: torch.Tensor,
    A_init: torch.Tensor,
    B_init: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert trained PiSSA adapter parameters to portable LoRA format.

    Uses the identity:
        ΔW = A'·B' - A₀·B₀ = [A' | A₀] · [B' | -B₀]^T

    This produces a standard LoRA adapter (rank 2r) that can be loaded onto
    the *original* (non-decomposed) pretrained model without requiring SVD.

    Args:
        A_trained: Trained PiSSA ``lora_up`` weight ``(out, r)``.
        B_trained: Trained PiSSA ``lora_down`` weight ``(r, in)``.
        A_init: Initial PiSSA ``lora_up`` weight from SVD ``(out, r)``.
        B_init: Initial PiSSA ``lora_down`` weight from SVD ``(r, in)``.

    Returns:
        ``(delta_A, delta_B)`` where:
            - *delta_A*: ``(out, 2r)`` — concatenated LoRA A matrix
            - *delta_B*: ``(2r, in)`` — concatenated LoRA B matrix
    """
    delta_A = torch.cat([A_trained, A_init], dim=1)    # (out, 2r)
    delta_B = torch.cat([B_trained, -B_init], dim=0)   # (2r, in)
    return delta_A, delta_B


@torch.no_grad()
def qpissa_iterative(
    weight: torch.Tensor,
    r: int,
    niter: int = 5,
    quant_fn=None,
    fast_niter: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """QPiSSA-T-iters: iterative quantization-aware PiSSA decomposition.

    Iteratively refines the low-rank decomposition and quantized residual
    to minimize quantization error. Based on Algorithm 1 from the PiSSA paper.

    Args:
        weight: Pretrained weight matrix of shape ``(out, in)``.
        r: Rank for low-rank decomposition.
        niter: Number of alternating SVD+quantization iterations.
        quant_fn: Quantization function ``(tensor) -> (quantized, dequantized)``.
                  If None, assumes the weight is already quantized and uses
                  a no-op identity function.
        fast_niter: If > 0, use fast randomized SVD for iterations.

    Returns:
        ``(quantized_residual, dequantized_residual, A, B)`` tuple.
    """
    if quant_fn is None:
        def quant_fn(x):
            return x, x

    res = weight.float()
    A, B = None, None

    for i in range(niter):
        # SVD of current residual
        U, S, Vh = torch.linalg.svd(res, full_matrices=False)
        U_r = U[:, :r]
        S_r = S[:r]
        Vh_r = Vh[:r, :]

        sqrt_S_r = torch.sqrt(S_r)
        A = U_r * sqrt_S_r.unsqueeze(0)
        B = sqrt_S_r.unsqueeze(1) * Vh_r

        # Compute residual
        res = weight - A @ B

        # Quantize and get error
        quantized, dequantized = quant_fn(res)
        res = weight - dequantized

    # Final quantize
    res = weight - A @ B
    quantized_res, dequantized_res = quant_fn(res)

    logger.info(
        f"QPiSSA: {niter} iterations, rank={r}, "
        f"final residual norm={res.norm().item():.6f}"
    )
    return (
        quantized_res.to(weight.dtype),
        dequantized_res.to(weight.dtype),
        A.to(weight.dtype),
        B.to(weight.dtype),
    )


@torch.no_grad()
def compute_svd_error_ratio(
    weight: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    W_residual: torch.Tensor,
) -> float:
    """Compute the SVD reconstruction error ratio.

    Used to verify that ``W = W_res + A @ B`` holds after PiSSA decomposition.

    Returns the relative Frobenius norm error.
    """
    reconstructed = W_residual + A @ B
    error = (weight - reconstructed).norm().item()
    original_norm = weight.norm().item()
    if original_norm == 0:
        return 0.0
    return error / original_norm
