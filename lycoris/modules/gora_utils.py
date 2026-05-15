"""
GoRA: Gradient-driven Adaptive Low Rank Adaptation — Utility Functions

Implements:
- Gradient accumulation hooks (backward → CPU offload)
- Importance computation: I(W) = avg(|W ⊙ G|) and other metrics
- Rank allocation from parameter budget
- Adaptive N: early stopping when layer importances converge
- Adaptive γ: grid search on first batch to minimize loss
- Main entry point: gora_precompute_gradients()

Reference: GoRA paper (https://arxiv.org/abs/2502.12171)
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
    """Create a backward hook that accumulates gradients onto `weight.grad_stored` (CPU).

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
# Importance Metrics
# ---------------------------------------------------------------------------

def compute_importance(
    param: torch.Tensor,
    grad_stored: torch.Tensor,
    importance_type: str = "union_mean",
    lora_rank: Optional[int] = None,
    max_lora_rank: Optional[int] = None,
    scale_features: bool = False,
) -> Tuple[bool, float]:
    """Compute importance score for a weight matrix.

    Args:
        param: Pre-trained weight W of shape (out_features, in_features) or >2-D.
        grad_stored: Accumulated gradient G, same shape as param.
        importance_type: Metric to use (see below).
        lora_rank: Reference rank for estimated nuclear norm.
        max_lora_rank: Override rank for SVD-based metrics.
        scale_features: If True, apply sqrt to the importance score.

    Supported importance_type values:
      - ``"union_mean"``: avg(|W ⊙ G|)  [GoRA paper default]
      - ``"union_frobenius_norm"``: ||W ⊙ G||_F
      - ``"union_2ord_norm"``: mean of row-wise L2 norms of W ⊙ G
      - ``"union_nuc_norm"``: nuclear norm of W ⊙ G
      - ``"grad_nuc_norm"``: nuclear norm of G
      - ``"grad_est_nuc_norm"``: estimated nuclear norm of G (fast SVD)
      - ``"union_est_nuc_norm"``: estimated nuclear norm of W ⊙ G
      - ``"grad_frobenius_norm"``: ||G||_F
      - ``"grad_mean"``: avg(|G|)
      - ``"grad_entropy"``: entropy of G assuming Gaussian
      - ``"union_mean_grad_nuc_norm"``: tuple (avg(|W⊙G|), est_nuc_norm(G))
      - ``"union_mean_union_nuc_norm"``: tuple
      - ``"grad_mean_grad_nuc_norm"``: tuple

    Returns:
        (is_tuple: bool, importance: float or tuple of floats)
    """
    param = param.float()
    grad_stored = grad_stored.float().to(param.device)

    if max_lora_rank:
        svd_rank = max_lora_rank
    else:
        svd_rank = min(4 * (lora_rank or 8), min(param.shape))

    if importance_type == 'union_frobenius_norm':
        importance = torch.linalg.matrix_norm(param * grad_stored).item()
    elif importance_type == 'union_2ord_norm':
        importance = torch.mean(torch.linalg.norm(param * grad_stored, dim=1)).item()
    elif importance_type == 'union_mean':
        importance = torch.mean(torch.abs(param * grad_stored)).item()
    elif importance_type == 'union_nuc_norm':
        importance = torch.linalg.matrix_norm(param * grad_stored, ord='nuc').item()
    elif importance_type == 'grad_nuc_norm':
        importance = torch.linalg.matrix_norm(grad_stored, ord='nuc').item()
    elif importance_type == 'grad_est_nuc_norm':
        importance = _est_nuc_norm(grad_stored, svd_rank)
    elif importance_type == 'union_est_nuc_norm':
        importance = _est_nuc_norm(param * grad_stored, svd_rank)
    elif importance_type == 'grad_frobenius_norm':
        importance = torch.linalg.matrix_norm(grad_stored).item()
    elif importance_type == 'grad_mean':
        importance = torch.mean(torch.abs(grad_stored)).item()
    elif importance_type == 'grad_entropy':
        importance = _get_entropy(grad_stored)
    elif importance_type == 'union_mean_grad_nuc_norm':
        importance = (
            torch.mean(torch.abs(param * grad_stored)).item(),
            _est_nuc_norm(grad_stored, svd_rank),
        )
    elif importance_type == 'union_mean_union_nuc_norm':
        param_grad = param * grad_stored
        importance = (
            torch.mean(torch.abs(param_grad)).item(),
            _est_nuc_norm(param_grad, svd_rank),
        )
    elif importance_type == 'grad_mean_grad_nuc_norm':
        importance = (
            torch.mean(torch.abs(grad_stored)).item(),
            _est_nuc_norm(grad_stored, svd_rank),
        )
    else:
        raise ValueError(f"Unknown importance_type: {importance_type}")

    if scale_features:
        if isinstance(importance, tuple):
            importance = tuple(math.sqrt(i) for i in importance)
        else:
            importance = math.sqrt(importance)

    return isinstance(importance, tuple), importance


def _est_nuc_norm(tensor: torch.Tensor, rank: int) -> float:
    """Estimate nuclear norm via fast randomized SVD."""
    try:
        _, Sr, _ = torch.svd_lowrank(tensor, rank, niter=8)
        return torch.sum(torch.log1p(Sr)).item()
    except Exception:
        return 0.0


def _get_entropy(tensor: torch.Tensor) -> float:
    """Gaussian entropy of flattened tensor values."""
    flat = tensor.flatten()
    sigma = torch.std(flat)
    eps = 1e-8
    sigma = torch.clamp(sigma, min=eps)
    return (torch.log(sigma) + 0.5 * (torch.log(torch.tensor(2 * torch.pi)) + 1)).item()


# ---------------------------------------------------------------------------
# Importance Normalization
# ---------------------------------------------------------------------------

def get_normalized_importances(
    importances_tensor: torch.Tensor,
    softmax: bool = False,
    temperature: float = 0.5,
) -> torch.Tensor:
    """Normalize importance scores to sum to 1.

    Args:
        importances_tensor: Raw importance values.
        softmax: If True, apply softmax with temperature.
        temperature: Temperature for softmax.

    Returns:
        Normalized importance tensor summing to 1.
    """
    if softmax:
        shifted = (importances_tensor - importances_tensor.min())
        divisor = (importances_tensor.max() - importances_tensor.min())
        if divisor > 0:
            shifted = shifted / (divisor * temperature)
        normalized = torch.softmax(shifted, dim=0)
    else:
        total = importances_tensor.sum()
        if total > 0:
            normalized = importances_tensor / total
        else:
            normalized = torch.ones_like(importances_tensor) / len(importances_tensor)
    return normalized


# ---------------------------------------------------------------------------
# Rank Allocation
# ---------------------------------------------------------------------------

def allocate_ranks(
    modules: List,
    ref_rank: int,
    min_rank: int = 1,
    max_rank: int = 32,
    importance_type: str = "union_mean",
    softmax_importance: bool = False,
    temperature: float = 0.5,
    scale_importance: bool = False,
    features_func: Optional[str] = None,
    allocate_strategy: str = "moderate",
    device: Optional[torch.device] = None,
) -> Tuple[Dict[str, int], float, float, OrderedDict]:
    """Allocate ranks to GoRA modules based on gradient importance.

    Args:
        modules: List of GoRAModule instances with weight.grad_stored.
        ref_rank: Reference rank r^ref for budget calculation.
        min_rank: Minimum rank per adapter.
        max_rank: Maximum rank per adapter.
        importance_type: Importance metric name.
        softmax_importance: Use softmax normalization.
        temperature: Softmax temperature.
        scale_importance: Apply sqrt to importance scores.
        features_func: Feature adjustment function ("sqrt", "log1p", or None).
        allocate_strategy: Rounding strategy — "radical" (ceil), "moderate" (round), "conserved" (floor).
        device: Device for tensor operations.

    Returns:
        (named_ranks, total_budget, actual_trainable, named_importances)
    """
    allocate_func: Callable = {
        'radical': math.ceil,
        'moderate': round,
        'conserved': math.floor,
    }.get(allocate_strategy, round)

    feature_adjust_func: Callable = {
        'sqrt': math.sqrt,
        'log1p': math.log1p,
        None: lambda x: x,
    }.get(features_func, lambda x: x)

    named_importances: OrderedDict[str, float] = OrderedDict()
    named_ranks: Dict[str, int] = {}
    named_features: Dict[str, float] = {}
    named_smooth_features: Dict[str, float] = {}
    total_budget = 0.0
    smooth_total_budget = 0.0
    actual_trainable = 0.0
    is_tuple = False

    for module in modules:
        name = module.lora_name
        org_w_check = module.org_weight if hasattr(module, 'org_weight') else module.weight
        if not hasattr(org_w_check, 'grad_stored') or org_w_check.grad_stored is None:
            logger.warning(f"GoRA: Module {name} has no stored gradients, skipping.")
            continue

        features = module.in_features + module.out_features
        grad_stored = org_w_check.grad_stored.to(device) / org_w_check.iters

        is_tuple_local, importance = compute_importance(
            org_w_check.data,
            grad_stored,
            importance_type=importance_type,
            lora_rank=ref_rank,
            max_lora_rank=max_rank,
            scale_features=scale_importance,
        )
        is_tuple = is_tuple_local

        named_importances[name] = importance
        adjusted_features = feature_adjust_func(features)
        named_smooth_features[name] = adjusted_features
        named_features[name] = features
        smooth_total_budget += adjusted_features * ref_rank
        total_budget += features * ref_rank

        # Clean up GPU
        del grad_stored

    if not named_importances:
        raise ValueError("GoRA: No modules with stored gradients found.")

    # Normalize importances (handle tuple case for dual metrics)
    if is_tuple:
        first_vals = torch.tensor([v[0] for v in named_importances.values()])
        second_vals = torch.tensor([v[1] for v in named_importances.values()])
        first_norm = get_normalized_importances(first_vals, softmax_importance, temperature)
        second_norm = get_normalized_importances(second_vals, softmax_importance, temperature)
        normalized = torch.tensor([0.5 * a + 0.5 * b for a, b in zip(first_norm, second_norm)])
    else:
        vals = torch.tensor(list(named_importances.values()))
        normalized = get_normalized_importances(vals, softmax_importance, temperature)

    for name, norm_imp in zip(named_importances.keys(), normalized):
        smooth_trainable = allocate_func(smooth_total_budget * norm_imp.item())
        rank = smooth_trainable // named_smooth_features[name]
        rank = max(1, int(rank))  # ensure at least rank 1
        if min_rank is not None and max_rank is not None:
            rank = min(max(allocate_func(rank), min_rank), max_rank)
        named_ranks[name] = rank
        actual_trainable += rank * named_features[name]

    return named_ranks, total_budget, actual_trainable, named_importances


# ---------------------------------------------------------------------------
# Importance Convergence Check (Adaptive N)
# ---------------------------------------------------------------------------

def check_importance_convergence(
    current: OrderedDict[str, float],
    previous: OrderedDict[str, float],
    threshold: float = 0.01,
) -> bool:
    """Check if importance scores have converged across all layers.

    Args:
        current: Current importance scores per layer.
        previous: Previous importance scores per layer.
        threshold: Relative change threshold.

    Returns:
        True if all layers have converged.
    """
    if previous is None or not previous:
        return False

    if len(current) != len(previous):
        return False

    for name in current:
        if name not in previous:
            return False
        curr_val = current[name]
        prev_val = previous[name]

        if isinstance(curr_val, tuple):
            for c, p in zip(curr_val, prev_val):
                if abs(c - p) / (p + 1e-8) > threshold:
                    return False
        else:
            if abs(curr_val - prev_val) / (prev_val + 1e-8) > threshold:
                return False
    return True


# ---------------------------------------------------------------------------
# Adaptive Gamma (Scaling Factor) Auto-Tuning
# ---------------------------------------------------------------------------

@torch.no_grad()
def adaptive_gamma_selection(
    modules: List,
    forward_fn: Callable[[], torch.Tensor],
    gamma_init: float = 1.0,
    gamma_decay: float = 0.8,
    gamma_min: float = 1e-5,
    scaling_alpha: float = 1.0,
) -> float:
    """Auto-tune γ by grid search on first training batch.

    Scales each module's lora_up (weight_b) by candidate γ values,
    evaluates loss on the same batch, and picks the γ with lowest loss.

    Args:
        modules: List of GoRAModule instances (already initialized).
        forward_fn: Callable that returns scalar loss on first batch.
        gamma_init: Starting γ value.
        gamma_decay: Multiplicative decay factor.
        gamma_min: Minimum γ to try.
        scaling_alpha: α used in scaling formula.

    Returns:
        Best γ value.
    """
    # Build candidate list
    candidates = []
    current = gamma_init
    while current >= gamma_min:
        candidates.append(current)
        current *= gamma_decay
    if not candidates or candidates[-1] > gamma_min:
        candidates.append(gamma_min)

    # Baseline loss (γ = scaling_alpha, i.e., no extra scaling)
    base_loss = forward_fn()
    if hasattr(base_loss, 'item'):
        base_loss = base_loss.item()

    best_loss = float('inf')
    best_gamma = 0.0

    for gamma in candidates:
        # Scale all lora_up weights by gamma
        for mod in modules:
            m = mod.out_features
            alpha = mod.scaling_alpha if hasattr(mod, 'scaling_alpha') else 1.0
            scale = (gamma * math.sqrt(m)) / alpha
            mod.lora_up.weight.data *= scale

        current_loss = forward_fn()
        if hasattr(current_loss, 'item'):
            current_loss = current_loss.item()

        # Undo scaling
        for mod in modules:
            m = mod.out_features
            alpha = mod.scaling_alpha if hasattr(mod, 'scaling_alpha') else 1.0
            scale = (gamma * math.sqrt(m)) / alpha
            mod.lora_up.weight.data /= scale

        logger.debug(f"GoRA γ={gamma:.6f}  loss={current_loss:.6f}")

        if current_loss < best_loss and current_loss < base_loss:
            best_loss = current_loss
            best_gamma = gamma

    logger.info(f"GoRA adaptive γ: selected γ={best_gamma:.6f} (base_loss={base_loss:.6f}, best_loss={best_loss:.6f})")
    return best_gamma


# ---------------------------------------------------------------------------
# Dynamic Initialization of a Single Module
# ---------------------------------------------------------------------------

@torch.no_grad()
def gora_dynamic_init(
    modules: List,
    named_ranks: Dict[str, int],
    ref_rank: int,
    scaling_alpha: float,
    stable_gamma: float = 0.05,
    scale_by_lr: bool = False,
    lr: float = 1e-3,
    weight_a_init_method: str = 'kaiming',  # 'kaiming', 'weight_svd', 'grad_svd'
    fast_svd_niter: int = 16,
) -> None:
    """Initialize GoRA adapter weights for all modules with allocated ranks.

    For each module:
      1. Set lora_dim to allocated rank
      2. Initialize A₀ (lora_down) via Kaiming uniform or SVD
      3. Compute B₀ = G @ A₀ᵀ @ (A₀ @ A₀ᵀ)⁻¹ (pseudo-inverse projection)
      4. Scale B₀ by ξ = (stable_gamma * √m) / α
      5. Set lora_up = scaled B₀, lora_down = A₀
      6. Clean up grad_stored

    Args:
        modules: List of GoRAModule instances.
        named_ranks: {lora_name: allocated_rank} dict.
        ref_rank: Reference rank (unused in scaling if rs_lora is active).
        scaling_alpha: α hyperparameter for scaling.
        stable_gamma: γ — scaling factor for initialization magnitude.
        scale_by_lr: Use learning rate in scaling formula.
        lr: Learning rate for scale_by_lr mode.
        weight_a_init_method: How to initialize A₀.
        fast_svd_niter: Power iterations for fast SVD.
    """
    for module in modules:
        name = module.lora_name
        if name not in named_ranks:
            continue

        rank = named_ranks[name]
        if rank <= 0:
            logger.warning(f"GoRA: Module {name} allocated rank <= 0, skipping.")
            continue

        org_w = module.org_weight if hasattr(module, 'org_weight') else module.weight
        if not hasattr(org_w, 'grad_stored') or org_w.grad_stored is None:
            logger.warning(f"GoRA: Module {name} has no stored gradients, skipping.")
            continue

        _grad_compress_init_single(
            module, rank, stable_gamma, scaling_alpha,
            scale_by_lr, lr, weight_a_init_method, fast_svd_niter,
        )

        # Free gradient storage
        if hasattr(org_w, 'grad_stored'):
            del org_w.grad_stored
            org_w.grad_stored = None
        if hasattr(org_w, 'iters'):
            del org_w.iters


def _grad_compress_init_single(
    module,
    rank: int,
    stable_gamma: float,
    scaling_alpha: float,
    scale_by_lr: bool,
    lr: float,
    weight_a_init_method: str,
    fast_svd_niter: int,
) -> None:
    """Initialize a single GoRAModule's lora weights via gradient compression.

    Formula (matching GoRA paper Eq. 9):
      A₀ ~ Kaiming uniform (or SVD)
      B₀ = G @ A₀ᵀ @ (A₀ @ A₀ᵀ + εI)⁻¹
      B₀ *= (stable_gamma / scaling_alpha)   [or lr-based variant]

    Note: LyCORIS convention is lora_up (A) and lora_down (B), where
          ΔW = lora_up @ lora_down.
          In GoRA paper: A ∈ R^{m×r} (up), B ∈ R^{r×n} (down).
          Here we initialize:
            lora_down (A₀): Kaiming uniform  (r × in_features)
            lora_up   (B₀): G @ A₀ᵀ @ (A₀A₀ᵀ)⁻¹  scaled  (out_features × r)
    """
    dtype = module.lora_down.weight.dtype
    device = module.lora_down.weight.device
    in_features = module.in_features
    out_features = module.out_features
    m = out_features  # paper notation
    n = in_features   # paper notation

    # Resize lora_down/lora_up if allocated rank differs from initial
    if module.lora_down.out_features != rank:
        old_down_weight = module.lora_down.weight.data
        module.lora_down = type(module.lora_down)(
            in_features, rank, bias=False,
        ).to(device=device, dtype=dtype)
        # Don't copy — will be overwritten below
    if module.lora_up.in_features != rank:
        module.lora_up = type(module.lora_up)(
            rank, out_features, bias=False,
        ).to(device=device, dtype=dtype)

    # Get accumulated gradient as float32 on correct device
    org_w = module.org_weight if hasattr(module, 'org_weight') else module.weight
    grad_stored = org_w.grad_stored.to(dtype=torch.float32, device=device)
    grad_stored = grad_stored / org_w.iters  # average over accumulation steps

    # --- Initialize A₀ (lora_down: r × in_features) ---
    lora_down_2d = torch.empty((rank, n), dtype=torch.float32, device=device)

    if weight_a_init_method == 'weight_svd':
        # Use SVD of pre-trained weight for A₀
        try:
            _, Sr, Ur = torch.svd_lowrank(
                org_w.data.float(), rank, niter=fast_svd_niter,
            )
            Uhr = Ur.t()  # (rank, in_features)
            lora_down_2d = torch.diag(Sr) @ Uhr
        except Exception:
            torch.nn.init.kaiming_uniform_(lora_down_2d, a=math.sqrt(5))
    elif weight_a_init_method == 'grad_svd':
        # Use SVD of gradient for A₀
        try:
            _, Sr, Ur = torch.svd_lowrank(grad_stored, rank, niter=fast_svd_niter)
            Uhr = Ur.t()
            lora_down_2d = torch.diag(Sr) @ Uhr
        except Exception:
            torch.nn.init.kaiming_uniform_(lora_down_2d, a=math.sqrt(5))
    else:
        # Default: Kaiming uniform (paper's standard init for A₀)
        torch.nn.init.kaiming_uniform_(lora_down_2d, a=math.sqrt(5))

    # --- Compute B₀ = G @ A₀ᵀ @ (A₀ @ A₀ᵀ + εI)⁻¹ ---
    # A₀ shape: (r, in_features) = (r, n)
    # A₀ᵀ shape: (n, r)
    # A₀ @ A₀ᵀ shape: (r, r)
    # G shape: (out_features, in_features) = (m, n)
    # B₀ = G @ A₀ᵀ @ (A₀A₀ᵀ)⁻¹  → shape (m, r)

    A0 = lora_down_2d  # (r, n)
    A0T = A0.T          # (n, r)
    A0A0T = A0 @ A0T    # (r, r)
    epsilon = 1e-8 * torch.eye(rank, device=device, dtype=torch.float32)
    try:
        A0A0T_inv = torch.linalg.pinv(A0A0T + epsilon)
    except Exception:
        # Fallback: use solve on regularized matrix
        A0A0T_reg = A0A0T + epsilon
        A0A0T_inv = torch.linalg.inv(A0A0T_reg)
    A0A0T_inv_A0T = A0T @ A0A0T_inv  # (n, r)
    lora_up_2d = grad_stored @ A0A0T_inv_A0T  # (m, r)

    # --- Scaling ---
    if scale_by_lr:
        # ξ = (lr / √(r/n)) * scale_rank  [from MyTransformers codebase]
        stable_gamma_effective = (lr / math.sqrt(rank / n)) * scaling_alpha
    else:
        stable_gamma_effective = stable_gamma

    # Paper Eq. 10: ξ = (γ · √m) / α  when using rsLoRA (α/√r forward)
    # With rsLoRA, scale = (α / √r), and ξ = (γ · √m) / α
    # So: B₀ *= (stable_gamma_effective * √m) / scaling_alpha
    lora_up_2d *= (stable_gamma_effective * math.sqrt(m)) / scaling_alpha

    # --- Compute reconstruction error ---
    # A₀B₀ should approximate -γ·G (one SGD step)
    reconstruction = lora_up_2d @ A0  # (m, r) @ (r, n) = (m, n)
    target = -stable_gamma_effective * grad_stored
    recon_error = torch.norm(target - reconstruction, p='fro').item()
    relative_error = recon_error / (torch.norm(grad_stored, p='fro').item() + 1e-8)
    module._gora_recon_error = recon_error
    module._gora_relative_error = relative_error

    # --- Overwrite module parameters ---
    # lora_down = A₀  (r, in_features) — already computed
    module.lora_down.weight.data.copy_(lora_down_2d.to(dtype))
    # lora_up = B₀  (out_features, r)
    module.lora_up.weight.data.copy_(lora_up_2d.to(dtype))

    # Update lora_dim to allocated rank
    module.lora_dim = rank

    # Update scale for rsLoRA: scale = α / √r
    r_factor = math.sqrt(rank)
    module.scale = scaling_alpha / r_factor
    module.alpha.copy_(torch.tensor(scaling_alpha * (rank / r_factor)))

    logger.info(
        f"GoRA init: {module.lora_name}  rank={rank}  "
        f"recon_error={recon_error:.6f}  relative_error={relative_error:.6f}"
    )


# ---------------------------------------------------------------------------
# Main Entry Point: GoRA Pre-compute Gradients
# ---------------------------------------------------------------------------

def gora_precompute_gradients(
    modules: List,
    dataloader,
    forward_fn: Callable[[], Tuple[torch.Tensor, ...]],
    ref_rank: int = 8,
    min_rank: int = 1,
    max_rank: int = 32,
    importance_type: str = "union_mean",
    scaling_alpha: float = 1.0,
    stable_gamma: float = 0.05,
    max_steps: int = 64,
    adaptive_n: bool = True,
    convergence_threshold: float = 0.01,
    min_steps: int = 3,
    adaptive_gamma: bool = False,
    gamma_init: float = 1.0,
    gamma_decay: float = 0.8,
    gamma_min: float = 1e-5,
    softmax_importance: bool = False,
    temperature: float = 0.5,
    scale_importance: bool = False,
    features_func: Optional[str] = None,
    allocate_strategy: str = "moderate",
    weight_a_init_method: str = "kaiming",
    fast_svd_niter: int = 16,
    scale_by_lr: bool = False,
    lr: float = 1e-3,
    world_size: int = 1,
    global_rank: int = 0,
    device: Optional[torch.device] = None,
    save_dir: Optional[str] = None,
) -> Dict[str, int]:
    """Run GoRA pre-computation phase: gradient accumulation, rank allocation, initialization.

    Call this ONCE before starting the main training loop.

    Steps:
      1. Register backward hooks on all modules
      2. For up to max_steps:
         a. Forward + backward on a batch
         b. Hooks accumulate gradients to CPU
         c. (Optional) Check importance convergence for adaptive N
      3. Allocate ranks based on importance scores
      4. Initialize all module lora weights via grad_compress_init
      5. (Optional) Auto-tune γ on first batch

    Args:
        modules: List of GoRAModule instances.
        dataloader: Training dataloader (must yield batches).
        forward_fn: Callable that takes (model, batch) and returns (loss, ...).
        ref_rank: Reference rank r^ref.
        min_rank: Minimum rank per adapter.
        max_rank: Maximum rank per adapter.
        importance_type: Importance metric name.
        scaling_alpha: α hyperparameter.
        stable_gamma: γ scaling factor for initialization.
        max_steps: Maximum gradient accumulation steps.
        adaptive_n: Enable adaptive N (early stopping on importance convergence).
        convergence_threshold: Threshold for adaptive N.
        min_steps: Minimum steps before checking convergence.
        adaptive_gamma: Enable adaptive γ auto-tuning.
        gamma_init: Initial γ for grid search.
        gamma_decay: Decay factor for γ candidates.
        gamma_min: Minimum γ to try.
        softmax_importance: Use softmax for importance normalization.
        temperature: Temperature for softmax.
        scale_importance: Apply sqrt to importance.
        features_func: Feature adjustment function.
        allocate_strategy: Rounding strategy.
        weight_a_init_method: How to init A₀.
        fast_svd_niter: Power iterations for fast SVD.
        scale_by_lr: Use lr-based scaling.
        lr: Learning rate for lr-based scaling.
        world_size: Distributed world size.
        global_rank: Current process rank.
        device: Device for computations.
        save_dir: If not None, save rank.json and importance.json here.

    Returns:
        {lora_name: allocated_rank} dict.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info(f"GoRA: Pre-computing gradients (max_steps={max_steps}, adaptive_n={adaptive_n})")

    # --- Phase 1: Gradient Accumulation ---
    hooks = []
    for module in modules:
        # Temporarily enable grad on weight
        org_w = module.org_weight if hasattr(module, 'org_weight') else module.weight
        org_w.requires_grad = True
        # Register backward hook on weight
        hook = org_w.register_hook(
            _record_gradient_hook_factory(org_w, world_size, global_rank)
        )
        hooks.append((module, hook))

    import time
    prev_importances = None
    named_ranks = {}
    named_importances = OrderedDict()

    for step_idx, batch in enumerate(dataloader):
        if step_idx >= max_steps:
            break

        # Move batch to device
        batch = _to_device(batch, device)

        # Forward + backward
        result = forward_fn(batch)
        if isinstance(result, (tuple, list)):
            loss = result[0]
        else:
            loss = result
        loss.backward()

        # Zero out grads of parameters (we just used them)
        for module in modules:
            org_w = module.org_weight if hasattr(module, 'org_weight') else module.weight
            for param in module.parameters():
                if param.grad is not None and param is not org_w:
                    param.grad = None
            # weight.grad is cleared in the hook after accumulation

        if global_rank == 0:
            elapsed = time.time()
            logger.info(
                f"GoRA grad step {step_idx+1}/{max_steps}: "
                f"loss={loss.item():.4f}"
            )

        # Adaptive N: check convergence
        if adaptive_n and global_rank == 0 and (step_idx + 1) >= min_steps:
            _, _, _, curr_importances = allocate_ranks(
                modules, ref_rank, min_rank, max_rank,
                importance_type, softmax_importance, temperature,
                scale_importance, features_func, allocate_strategy, device,
            )
            if check_importance_convergence(curr_importances, prev_importances, convergence_threshold):
                logger.info(f"GoRA: Importance scores converged at step {step_idx+1}. Stopping accumulation.")
                break
            prev_importances = curr_importances

    # Remove hooks
    for module, hook in hooks:
        hook.remove()
        org_w = module.org_weight if hasattr(module, 'org_weight') else module.weight
        org_w.requires_grad = False

    # Finalize: average accumulated gradients
    for module in modules:
        org_w = module.org_weight if hasattr(module, 'org_weight') else module.weight
        if hasattr(org_w, 'grad_stored') and org_w.grad_stored is not None:
            org_w.grad_stored = org_w.grad_stored / org_w.iters

    # --- Phase 2: Rank Allocation ---
    named_ranks, total_budget, actual_trainable, named_importances = allocate_ranks(
        modules, ref_rank, min_rank, max_rank,
        importance_type, softmax_importance, temperature,
        scale_importance, features_func, allocate_strategy, device,
    )
    logger.info(
        f"GoRA: Rank allocation complete. "
        f"total_budget={total_budget:.0f} actual_trainable={actual_trainable:.0f}"
    )

    # --- Phase 3: Dynamic Initialization ---
    gora_dynamic_init(
        modules, named_ranks, ref_rank, scaling_alpha,
        stable_gamma, scale_by_lr, lr, weight_a_init_method, fast_svd_niter,
    )

    # Log average reconstruction error
    errors = [m._gora_recon_error for m in modules if hasattr(m, '_gora_recon_error')]
    rel_errors = [m._gora_relative_error for m in modules if hasattr(m, '_gora_relative_error')]
    if errors:
        logger.info(
            f"GoRA: avg recon_error={sum(errors)/len(errors):.6f}  "
            f"avg relative_error={sum(rel_errors)/len(rel_errors):.6f}"
        )

    # --- Phase 4: Adaptive Gamma (optional) ---
    if adaptive_gamma:
        # Get first batch for loss evaluation
        first_batch = _to_device(next(iter(dataloader)), device)

        def eval_loss():
            result = forward_fn(first_batch)
            if isinstance(result, (tuple, list)):
                return result[0]
            return result

        best_gamma = adaptive_gamma_selection(
            modules, eval_loss, gamma_init, gamma_decay, gamma_min, scaling_alpha,
        )
        # Apply best gamma scaling
        for module in modules:
            m = module.out_features
            alpha = scaling_alpha
            best_scale = (best_gamma * math.sqrt(m)) / alpha
            module.lora_up.weight.data *= best_scale

    # --- Save rank/importance metadata ---
    if save_dir is not None and global_rank == 0:
        import json, os
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, 'rank.json'), 'w') as f:
            json.dump(named_ranks, f)
        with open(os.path.join(save_dir, 'importance.json'), 'w') as f:
            json.dump(
                {k: (list(v) if isinstance(v, tuple) else v) for k, v in named_importances.items()},
                f,
            )
        logger.info(f"GoRA: Saved rank/importance metadata to {save_dir}")

    return named_ranks


# ---------------------------------------------------------------------------
# Helper: move batch to device
# ---------------------------------------------------------------------------

def _to_device(batch, device):
    """Move batch dict or tensor to device."""
    if isinstance(batch, dict):
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    elif isinstance(batch, torch.Tensor):
        return batch.to(device)
    return batch
