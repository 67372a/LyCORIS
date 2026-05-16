"""
RaLoRA: Rank-Aligned LoRA with Gradient Intrinsic Dimensionality — Utility Functions

Implements:
- Entropy-based effective rank (GID estimator)
- Threshold-based and cumulative-variance alternative erank methods
- Importance computation: I(W) = avg(|W ⊙ G|) for RaLoRA-Pro
- Inter-layer rank allocation from parameter budget
- Intra-layer n_split allocation from effective rank
- Gradient accumulation hooks (backward → CPU offload)
- Main entry point: ralora_precompute_gradients()

Reference: RaLoRA paper (ICLR 2026)
  "Gradient Intrinsic Dimensionality Alignment"
"""

import math
import torch
import torch.nn as nn
import torch.distributed as dist

from typing import Callable, Dict, List, Optional, OrderedDict, Tuple
from collections import OrderedDict
from ..logging import logger


# ---------------------------------------------------------------------------
# Gradient Hook
# ---------------------------------------------------------------------------

def _record_gradient_hook_factory(weight: nn.Parameter, world_size: int = 1, rank: int = 0):
    """Create a backward hook that accumulates gradients onto weight.grad_stored (CPU).

    On each backward pass:
      - If distributed: all-reduce the gradient across workers
      - Accumulate onto weight.grad_stored (on CPU)
      - Increment weight.iters
      - Clear GPU gradient
    """
    def record_gradient_hook(grad: torch.Tensor) -> torch.Tensor:
        # All-reduce across distributed workers
        if world_size > 1 and grad is not None:
            dist.all_reduce(grad, op=dist.ReduceOp.SUM)
            grad = grad / world_size

        if rank == 0 and grad is not None:
            grad_cpu = grad.detach().cpu()
            # Support both direct weight parameters and modules with org_weight
            target = weight
            if hasattr(weight, 'org_weight') and not hasattr(weight, 'grad_stored'):
                target = weight.org_weight
            if not hasattr(target, 'grad_stored') or target.grad_stored is None:
                target.grad_stored = grad_cpu
                target.iters = 1
            else:
                target.grad_stored = target.grad_stored + grad_cpu
                target.iters += 1

        return grad

    return record_gradient_hook


# ---------------------------------------------------------------------------
# Effective Rank (GID) Estimation
# ---------------------------------------------------------------------------

def compute_effective_rank(gradient_matrix: torch.Tensor, dtype=torch.float32, eps=1e-10) -> float:
    """Compute the entropy-based effective rank of a gradient matrix.

    Uses the method from Roy & Vetterli (2007):
        erank(G) = exp(-Σ p_i log p_i)
    where p_i = σ_i / Σ σ_j are the normalized singular values.

    This is the primary GID estimator used by RaLoRA (Section 3.2, Eq. 3).

    Args:
        gradient_matrix: 2D gradient tensor (shape: [m, n]).
        dtype: Working precision for SVD.
        eps: Small value to avoid log(0).

    Returns:
        Effective rank (float, ≥ 1.0).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    gradient_matrix = gradient_matrix.to(dtype=dtype, device=device)

    if gradient_matrix.dim() != 2:
        raise ValueError("Input gradient_matrix must be a 2D tensor")

    # Perform Singular Value Decomposition (SVD)
    try:
        U, S, Vh = torch.linalg.svd(gradient_matrix)
    except RuntimeError as e:
        logger.warning(f"RaLoRA: SVD computation failed: {e}")
        return 1.0

    if S.numel() == 0:
        return 1.0

    # Compute L1 norm of singular values
    l1_norm = torch.sum(S)

    # Compute normalized singular values: p_i = σ_i / Σ σ_j
    p = S / l1_norm

    # Compute Shannon entropy: H = -sum(p_k * log(p_k))
    entropy = -torch.sum(p * torch.log(p + eps))

    # Compute effective rank: erank = exp(H)
    effective_rank = torch.exp(entropy).item()

    del U, S, Vh, gradient_matrix

    return max(1.0, effective_rank)


def count_singular_values_above_threshold(
    gradient_matrix: torch.Tensor,
    threshold: float = 1e-2,
    dtype=torch.float32,
) -> float:
    """Count singular values above a fixed threshold (alternative erank method).

    Args:
        gradient_matrix: 2D gradient tensor.
        threshold: Absolute threshold ε for singular values.
        dtype: Working precision.

    Returns:
        Count of singular values > threshold (≥ 1.0).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    gradient_matrix = gradient_matrix.to(dtype=dtype, device=device)

    if gradient_matrix.dim() != 2:
        raise ValueError("Input gradient_matrix must be a 2D tensor")

    try:
        _, S, _ = torch.svd_lowrank(gradient_matrix, q=min(gradient_matrix.shape))
    except RuntimeError:
        return 1.0

    count = torch.sum(S > threshold).item()

    del S, gradient_matrix

    return max(1.0, count)


def count_singular_values_by_variance_threshold(
    gradient_matrix: torch.Tensor,
    cumulative_variance_threshold: float = 0.99,
    dtype=torch.float32,
) -> float:
    """Count singular values needed to explain a fraction of total variance.

    Args:
        gradient_matrix: 2D gradient tensor.
        cumulative_variance_threshold: Fraction of variance to explain (0–1).
        dtype: Working precision.

    Returns:
        Number of singular values needed (≥ 1.0).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    gradient_matrix = gradient_matrix.to(dtype=dtype, device=device)

    if gradient_matrix.dim() != 2:
        raise ValueError("Input gradient_matrix must be a 2D tensor")

    if not (0.0 <= cumulative_variance_threshold <= 1.0):
        raise ValueError("cumulative_variance_threshold must be in [0.0, 1.0]")

    try:
        _, S, _ = torch.svd_lowrank(gradient_matrix, q=min(gradient_matrix.shape))
    except RuntimeError:
        return 1.0

    if S.numel() == 0:
        return 1.0

    singular_values_squared = S.pow(2)
    total_variance = torch.sum(singular_values_squared)

    if total_variance < 1e-10:
        return 1.0

    variance_ratios = singular_values_squared / total_variance
    cumulative_variance_ratios = torch.cumsum(variance_ratios, dim=0)
    indices_above = (cumulative_variance_ratios >= cumulative_variance_threshold).nonzero(as_tuple=True)[0]

    if indices_above.numel() == 0:
        count = float(S.numel())
    else:
        count = float(indices_above.min().item() + 1)

    del S, singular_values_squared, total_variance, variance_ratios, cumulative_variance_ratios, gradient_matrix

    return max(1.0, count)


def _get_erank(gradient_matrix: torch.Tensor, erank_method: str = "entropy",
               svd_threshold: float = 0.0, cumulative_variance: float = 0.0) -> float:
    """Dispatch to the appropriate erank computation method.

    Args:
        gradient_matrix: 2D gradient tensor.
        erank_method: "entropy", "threshold", or "cumulative_variance".
        svd_threshold: Threshold for "threshold" method.
        cumulative_variance: Threshold for "cumulative_variance" method.

    Returns:
        Effective rank estimate.
    """
    if svd_threshold > 0:
        logger.debug(f"RaLoRA: Using threshold-based erank with ε={svd_threshold}")
        return count_singular_values_above_threshold(gradient_matrix, threshold=svd_threshold)
    elif cumulative_variance > 0:
        logger.debug(f"RaLoRA: Using cumulative-variance erank with τ={cumulative_variance}")
        return count_singular_values_by_variance_threshold(
            gradient_matrix, cumulative_variance_threshold=cumulative_variance
        )
    else:
        return compute_effective_rank(gradient_matrix)


# ---------------------------------------------------------------------------
# Importance (Loss Sensitivity) for RaLoRA-Pro
# ---------------------------------------------------------------------------

def compute_importance(param: torch.Tensor, grad_stored: torch.Tensor) -> float:
    """Compute loss sensitivity importance score for RaLoRA-Pro.

    I(W_l) = avg(|W_l ⊙ G_l|)  —  Equation 5 in the paper.

    Args:
        param: Pre-trained weight W of shape (out_features, in_features) or >2-D.
        grad_stored: Accumulated gradient G, same shape as param.

    Returns:
        Scalar importance score.
    """
    param = param.float()
    grad_stored = grad_stored.float().to(param.device)
    importance = torch.mean(torch.abs(param * grad_stored)).item()
    return importance


# ---------------------------------------------------------------------------
# Importance Normalization
# ---------------------------------------------------------------------------

def get_normalized_importances(importances_tensor: torch.Tensor) -> torch.Tensor:
    """Normalize importance scores: α_l = I_l / Σ I_k  (Equation 6).

    Args:
        importances_tensor: Raw importance values.

    Returns:
        Normalized importance tensor summing to 1.
    """
    total = importances_tensor.sum()
    if total > 0:
        return importances_tensor / total
    return torch.ones_like(importances_tensor) / len(importances_tensor)


# ---------------------------------------------------------------------------
# Inter-Layer Rank Allocation (RaLoRA-Pro)
# ---------------------------------------------------------------------------

def get_allocated_rank(
    modules: List,
    ref_rank: int,
    min_rank: Optional[int] = None,
    max_rank: Optional[int] = None,
    features_func: Optional[str] = None,
) -> Tuple[Dict[str, int], float, float, OrderedDict]:
    """RaLoRA-Pro: Allocate per-layer ranks from parameter budget.

    Uses dimensionality-smoothed budget (paper Eq. 7-8):
        P_total = Σ sqrt(d_in^l + d_out^l) × r_ref
        r_l = clip(round(P_total × α_l / sqrt(d_in^l + d_out^l)), r_min, r_max)

    Args:
        modules: List of RaLoRAModule instances with grad_stored.
        ref_rank: Reference rank r_ref for budget calculation.
        min_rank: Minimum rank per layer.
        max_rank: Maximum rank per layer.
        features_func: Feature adjustment ("sqrt", "log1p", or None). None uses
                       paper's sqrt(d_in + d_out) smoothing.

    Returns:
        (named_ranks, total_budget, actual_trainable, named_importances)
    """
    feature_adjust_func: Callable = {
        'sqrt': math.sqrt,
        'log1p': math.log1p,
        None: math.sqrt,  # Paper default: sqrt(d_in + d_out)
    }.get(features_func, math.sqrt)

    named_importances: OrderedDict[str, float] = OrderedDict()
    named_ranks: Dict[str, int] = {}
    named_smooth_features: Dict[str, float] = {}
    named_features: Dict[str, float] = {}
    smooth_total_budget = 0.0
    total_budget = 0.0
    actual_trainable = 0.0

    for module in modules:
        name = module.lora_name
        org_w_check = module.org_weight if hasattr(module, 'org_weight') else module.weight
        if not hasattr(org_w_check, 'grad_stored') or org_w_check.grad_stored is None:
            logger.warning(f"RaLoRA: Module {name} has no stored gradients, skipping.")
            continue

        features = module.in_features + module.out_features
        grad_stored = org_w_check.grad_stored.to(
            org_w_check.data.device
        ) / org_w_check.iters

        importance = compute_importance(org_w_check.data, grad_stored)
        named_importances[name] = importance

        # Dimensionality smoothing: sqrt(d_in + d_out) per paper Eq. 7-8
        adjusted_features = feature_adjust_func(features)
        named_smooth_features[name] = adjusted_features
        named_features[name] = features
        smooth_total_budget += adjusted_features * ref_rank
        total_budget += features * ref_rank

        del grad_stored

    if not named_importances:
        raise ValueError("RaLoRA: No modules with stored gradients found.")

    # Normalize importance scores (Eq. 6)
    importances_tensor = torch.tensor(list(named_importances.values()))
    normalized_importances = get_normalized_importances(importances_tensor)

    for name, norm_imp in zip(named_importances.keys(), normalized_importances):
        smooth_trainable = round(smooth_total_budget * norm_imp.item())
        rank = smooth_trainable // named_smooth_features[name]
        rank = max(1, int(rank))

        if min_rank is not None and max_rank is not None:
            rank = min(max(round(rank), min_rank), max_rank)

        named_ranks[name] = rank
        actual_trainable += rank * named_features[name]

    return named_ranks, total_budget, actual_trainable, named_importances


# ---------------------------------------------------------------------------
# n_split (Block-Diagonal) Allocation — Core RaLoRA
# ---------------------------------------------------------------------------

def compute_n_split_allocations(
    modules: List,
    named_ranks: Dict[str, int],
    n_max: int = 32,
    erank_method: str = "entropy",
    svd_threshold: float = 0.0,
    cumulative_variance: float = 0.0,
) -> Dict[str, int]:
    """Compute the number of diagonal blocks n_l per layer.

    Formula (paper Eq. 4):
        e_l = floor(log₂(erank(G_l) / r_l))
        n_l = 2^clip(e_l, 1, n_max)

    Args:
        modules: List of RaLoRAModule instances.
        named_ranks: {lora_name: allocated_rank} dict.
        n_max: Maximum expansion factor (n_max in paper).
        erank_method: GID estimation method.
        svd_threshold: Threshold for threshold-based erank.
        cumulative_variance: Threshold for cumulative-variance erank.

    Returns:
        {lora_name: n_split} dict (each n_split is a power of 2, ≥ 1).
    """
    named_eranks: Dict[str, float] = {}
    named_n_splits: Dict[str, int] = {}

    for module in modules:
        name = module.lora_name

        org_w = module.org_weight if hasattr(module, 'org_weight') else module.weight
        if not hasattr(org_w, 'grad_stored') or org_w.grad_stored is None:
            logger.warning(f"RaLoRA: Module {name} has no stored gradients, skipping n_split.")
            named_n_splits[name] = 1
            continue

        # Compute effective rank from accumulated gradient
        grad_2d = _get_2d_gradient(org_w)
        erank = _get_erank(grad_2d, erank_method, svd_threshold, cumulative_variance)
        named_eranks[name] = erank

        # Compute n_split from erank and allocated rank
        rank = named_ranks.get(name, module.lora_dim)
        if rank > 0:
            n_splits_power = math.floor(math.log2(max(erank / rank, 1.0)))
        else:
            n_splits_power = 0

        # Clamp: max_power = log2(n_max)
        max_power = int(math.log2(n_max))
        n_splits_power = max(0, min(n_splits_power, max_power))
        n_split = 2 ** n_splits_power

        named_n_splits[name] = n_split
        logger.debug(
            f"RaLoRA: Module {name}: erank={erank:.1f}, rank={rank}, "
            f"n_split={n_split}"
        )

    return named_n_splits, named_eranks


def _get_2d_gradient(org_w) -> torch.Tensor:
    """Get accumulated gradient as 2D float32 tensor for SVD."""
    grad = org_w.grad_stored.float() / org_w.iters
    if grad.dim() > 2:
        grad = grad.reshape(grad.shape[0], -1)
    return grad


# ---------------------------------------------------------------------------
# Main Precomputation Entry Point
# ---------------------------------------------------------------------------

def ralora_precompute_gradients(
    modules: List,
    dataloader,
    forward_fn: Callable,
    model: Optional[nn.Module] = None,
    ref_rank: int = 8,
    min_rank: Optional[int] = None,
    max_rank: Optional[int] = None,
    n_max: int = 32,
    pro_mode: bool = False,
    erank_method: str = "entropy",
    svd_threshold: float = 0.0,
    cumulative_variance: float = 0.0,
    max_steps: int = 64,
    world_size: int = 1,
    global_rank: int = 0,
    device: Optional[torch.device] = None,
    save_dir: Optional[str] = None,
) -> Dict[str, int]:
    """Master precomputation function for RaLoRA / RaLoRA-Pro.

    Call this ONCE before starting the main training loop.

    Phases:
      1. Register backward hooks on pretrained weights of all modules
      2. For up to max_steps mini-batches:
         a. Forward + backward pass
         b. Hooks accumulate gradients to CPU (weight.grad_stored)
      3. All-reduce gradients across distributed workers
      4. If pro_mode (RaLoRA-Pro):
         a. Compute importance scores I(W_l) = avg(|W_l ⊙ G_l|)
         b. Allocate per-layer rank r_l from dimensionality-smoothed budget
      5. Compute effective rank (GID) of each layer's gradient
      6. Compute n_split per layer: n_l = 2^clip(floor(log2(erank/r_l)), 0, log2(n_max))
      7. Call dynamic_init on each module
      8. Save rank.json / n_splits.json / importance.json (if save_dir set)

    Args:
        modules: List of RaLoRAModule instances.
        dataloader: Training dataloader.
        forward_fn: Callable(model, batch) -> loss.
        ref_rank: Reference rank for parameter budget.
        min_rank: Minimum rank per layer (RaLoRA-Pro).
        max_rank: Maximum rank per layer (RaLoRA-Pro).
        n_max: Maximum expansion factor.
        pro_mode: Enable RaLoRA-Pro (inter-layer reallocation).
        erank_method: "entropy", "threshold", or "cumulative_variance".
        svd_threshold: Threshold for threshold-based erank.
        cumulative_variance: Threshold for cumulative-variance erank.
        max_steps: Maximum gradient accumulation steps.
        world_size, global_rank: Distributed info.
        device: Compute device.
        save_dir: Directory to save metadata JSON files.

    Returns:
        {lora_name: allocated_rank} dict.
    """
    import time
    import json
    import os

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not modules:
        raise RuntimeError("RaLoRA: No modules provided for precomputation.")

    logger.info(
        f"RaLoRA: Starting gradient pre-computation. "
        f"max_steps={max_steps}, pro_mode={pro_mode}, n_max={n_max}, "
        f"erank_method={erank_method}"
    )

    # --- Phase 1: Register gradient hooks ---
    hooks = []
    for module in modules:
        # Hook on org_weight to capture gradient w.r.t. pretrained weights
        org_w = module.org_weight if hasattr(module, 'org_weight') else module.weight
        org_w.requires_grad = True
        hook = org_w.register_hook(
            _record_gradient_hook_factory(org_w, world_size, global_rank)
        )
        hooks.append(hook)

    # Make sure all other parameters are frozen during precomputation
    for module in modules:
        for param in module.parameters():
            param.requires_grad = False

    # --- Phase 2: Accumulate gradients ---
    t_start = time.time()
    for idx, batch in enumerate(dataloader):
        if idx >= max_steps:
            break

        # Move batch to device
        if isinstance(batch, dict):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
        elif isinstance(batch, torch.Tensor):
            batch = batch.to(device)

        # Forward + backward
        if model is not None:
            result = forward_fn(model, batch)
        else:
            result = forward_fn(batch)
        if isinstance(result, (tuple, list)):
            loss = result[0]
        else:
            loss = result

        if hasattr(loss, 'backward'):
            loss.backward()

        if idx % 10 == 0 or idx == max_steps - 1:
            logger.info(
                f"RaLoRA: Gradient step {idx + 1}/{max_steps}, "
                f"loss={loss.item():.4f}"
            )

    t_elapsed = time.time() - t_start
    logger.info(f"RaLoRA: Gradient accumulation complete in {t_elapsed:.1f}s")

    # --- Remove hooks ---
    for hook in hooks:
        hook.remove()

    # Clear any remaining GPU gradients
    for module in modules:
        for p in module.parameters():
            p.grad = None

    # --- Sync distributed workers ---
    if world_size > 1:
        dist.barrier()

    # --- Phase 3: Average gradients (on rank 0 only) ---
    if global_rank == 0:
        for module in modules:
            org_w = module.org_weight if hasattr(module, 'org_weight') else module.weight
            if hasattr(org_w, 'grad_stored') and hasattr(org_w, 'iters'):
                org_w.grad_stored = org_w.grad_stored / org_w.iters

    # --- Phase 4: RaLoRA-Pro inter-layer rank allocation ---
    named_importances = OrderedDict()

    if pro_mode:
        logger.info("RaLoRA-Pro: Computing loss sensitivity + allocating per-layer ranks.")
        named_ranks, total_budget, actual_trainable, named_importances = get_allocated_rank(
            modules,
            ref_rank=ref_rank,
            min_rank=min_rank,
            max_rank=max_rank,
        )
        logger.info(
            f"RaLoRA-Pro: Budget total={total_budget:.0f}, "
            f"actual_trainable={actual_trainable:.0f}"
        )
    else:
        # RaLoRA (non-Pro): uniform rank for all layers
        named_ranks = {}
        for module in modules:
            named_ranks[module.lora_name] = ref_rank

    # --- Phase 5: Compute n_split from effective rank ---
    logger.info("RaLoRA: Computing per-layer n_split from gradient intrinsic dimensionality.")
    named_n_splits, named_eranks = compute_n_split_allocations(
        modules,
        named_ranks,
        n_max=n_max,
        erank_method=erank_method,
        svd_threshold=svd_threshold,
        cumulative_variance=cumulative_variance,
    )

    # --- Phase 6: Dynamic initialization ---
    logger.info("RaLoRA: Running dynamic_init on all modules.")
    for module in modules:
        name = module.lora_name
        if name not in named_ranks:
            continue

        rank = named_ranks[name]
        n_split = named_n_splits.get(name, 1)
        avg_rank = ref_rank  # For scaling

        logger.info(f"RaLoRA: Init {name}: rank={rank}, n_split={n_split}")
        module.dynamic_init(avg_rank=avg_rank, rank=rank, n_split=n_split)

        # Clean up gradient storage
        org_w = module.org_weight if hasattr(module, 'org_weight') else module.weight
        if hasattr(org_w, 'grad_stored'):
            del org_w.grad_stored
            org_w.grad_stored = None
        if hasattr(org_w, 'iters'):
            del org_w.iters

    # --- Phase 7: Save metadata ---
    if save_dir is not None and global_rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, 'rank.json'), 'w') as f:
            json.dump({k: int(v) for k, v in named_ranks.items()}, f)
        with open(os.path.join(save_dir, 'n_splits.json'), 'w') as f:
            json.dump({k: int(v) for k, v in named_n_splits.items()}, f)
        if named_importances:
            with open(os.path.join(save_dir, 'importance.json'), 'w') as f:
                json.dump(named_importances, f)
        logger.info(f"RaLoRA: Saved metadata to {save_dir}")

    logger.info("RaLoRA: Precomputation complete. Ready for training.")
    return named_ranks
