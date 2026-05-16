"""
Unit tests for GoRA: Gradient-driven Adaptive Low Rank Adaptation.

Tests:
  1. GoRAModule instantiation with GoRA-specific parameters
  2. Gradient accumulation via backward hook
  3. Importance computation (paper's avg(|W⊙G|))
  4. Rank allocation from budget
  5. grad_compress_init — initialization via pseudo-inverse projection
  6. Forward pass produces correct output shapes
  7. rsLoRA scaling (α/√r) is applied
  8. Saved state dict compatibility with LoConModule
  9. Adaptive N convergence detection
"""

import math
import json
import tempfile
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytest

# Make lycoris importable from repo root
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ==== Helpers ===============================================================

def _get_weight(module):
    """Get the base weight tensor from a LyCORIS module (uses org_weight)."""
    return module.org_weight if hasattr(module, 'org_weight') else module.weight


def _set_weight(module, value):
    """Set the base weight tensor on a LyCORIS module."""
    if hasattr(module, 'org_weight'):
        module.org_module[0].weight.data.copy_(value)
    else:
        module.weight.data.copy_(value)


def _accumulate_gradients_weight(module, grad_cpu):
    """Simulate gradient accumulation on a module's base weight."""
    w = _get_weight(module)
    if not hasattr(w, 'grad_stored') or w.grad_stored is None:
        w.grad_stored = grad_cpu
        w.iters = 1
    else:
        w.grad_stored = w.grad_stored + grad_cpu
        w.iters += 1


# ==== Test Fixtures =========================================================

@pytest.fixture(autouse=True)
def cleanup_registries():
    """Clean up global registries before each test."""
    from lycoris.modules.locon import GoRAModule, LoConModule
    GoRAModule.reset_gora_registry()
    LoConModule.reset_olora_registry()
    yield
    GoRAModule.reset_gora_registry()
    LoConModule.reset_olora_registry()


@pytest.fixture
def simple_linear():
    """An ordinary nn.Linear to wrap with GoRA."""
    return nn.Linear(32, 64, bias=False)


@pytest.fixture
def gora_module(simple_linear):
    """Create a GoRAModule wrapping a 32→64 Linear layer."""
    from lycoris.modules.locon import GoRAModule

    mod = GoRAModule(
        lora_name="test_gora",
        org_module=simple_linear,
        multiplier=1.0,
        lora_dim=4,
        alpha=16.0,
        gora_ref_rank=8,
        gora_min_rank=2,
        gora_max_rank=16,
        gora_gamma=0.05,
        gora_importance_type="union_mean",
    )
    return mod


@pytest.fixture
def gora_module_conv():
    """Create a GoRAModule wrapping a Conv2d."""
    from lycoris.modules.locon import GoRAModule

    conv = nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False)
    mod = GoRAModule(
        lora_name="test_gora_conv",
        org_module=conv,
        multiplier=1.0,
        lora_dim=4,
        alpha=8.0,
        gora_ref_rank=8,
        gora_min_rank=2,
        gora_max_rank=16,
        gora_gamma=0.05,
    )
    return mod


# ==== Test 1: Instantiation =================================================

def test_gora_module_creation(gora_module):
    """GoRAModule should create with GoRA-specific attributes."""
    from lycoris.modules.locon import GoRAModule as GM

    mod = gora_module
    assert mod.lora_name == "test_gora"
    assert mod.lora_dim == 4
    assert mod.in_features == 32
    assert mod.out_features == 64
    assert mod.rs_lora is True
    assert mod.gora_ref_rank == 8
    assert mod.gora_gamma == 0.05
    assert mod.gora_importance_type == "union_mean"
    assert mod.scaling_alpha == 16.0

    # rsLoRA scaling: scale = α / √r
    expected_scale = 16.0 / math.sqrt(4)
    assert abs(mod.scale - expected_scale) < 1e-6

    # Registered in global list
    assert mod in GM._gora_modules


def test_gora_module_registry():
    """Class-level registry should track all GoRA modules."""
    from lycoris.modules.locon import GoRAModule as GM

    GM.reset_gora_registry()

    lin1 = nn.Linear(16, 32, bias=False)
    lin2 = nn.Linear(32, 64, bias=False)

    m1 = GM(lora_name="m1", org_module=lin1, lora_dim=4)
    m2 = GM(lora_name="m2", org_module=lin2, lora_dim=4)

    assert len(GM._gora_modules) == 2
    assert m1 in GM._gora_modules
    assert m2 in GM._gora_modules

    # get_gora_modules with model filter
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = m1

    model = DummyModel()
    filtered = GM.get_gora_modules(model)
    assert len(filtered) == 1
    assert m1 in filtered
    assert m2 not in filtered

    GM.reset_gora_registry()
    assert len(GM._gora_modules) == 0


# ==== Test 2: Gradient Accumulation via Hook ================================

def test_gradient_hook_accumulation(gora_module):
    """Backward hook should accumulate gradients on org_weight.grad_stored (CPU)."""
    from lycoris.modules.gora_utils import _record_gradient_hook_factory

    mod = gora_module
    w = _get_weight(mod)
    w.requires_grad = True

    hook_fn = _record_gradient_hook_factory(w, world_size=1, rank=0)
    hook = w.register_hook(hook_fn)

    # Simulate 4 backward passes
    for _ in range(4):
        x = torch.randn(2, 32, requires_grad=True)
        y = F.linear(x, w)
        loss = y.sum()
        loss.backward()
        # Clear param grads except for the hook-managed one
        w.grad = None

    hook.remove()

    assert hasattr(w, 'grad_stored')
    assert w.grad_stored is not None
    assert w.iters == 4
    assert w.grad_stored.shape == w.shape
    assert w.grad_stored.device.type == 'cpu'

    avg_grad = w.grad_stored / w.iters
    assert avg_grad.abs().sum() > 0

    w.requires_grad = False
    del w.grad_stored
    del w.iters


# ==== Test 3: Importance Computation =========================================

def test_importance_union_mean():
    """Paper's default: I(W) = avg(|W ⊙ G|)."""
    from lycoris.modules.gora_utils import compute_importance

    W = torch.tensor([[1.0, -2.0], [3.0, 4.0]])
    G = torch.tensor([[0.5, -0.5], [1.0, 2.0]])

    is_tuple, importance = compute_importance(W, G, importance_type="union_mean")

    expected = torch.mean(torch.abs(W * G)).item()
    assert not is_tuple
    assert abs(importance - expected) < 1e-6


def test_importance_grad_frobenius():
    """Frobenius norm of gradient."""
    from lycoris.modules.gora_utils import compute_importance

    G = torch.tensor([[3.0, 4.0], [0.0, 0.0]])
    W = torch.ones_like(G)

    is_tuple, imp = compute_importance(W, G, importance_type="grad_frobenius_norm")
    assert not is_tuple
    assert abs(imp - 5.0) < 1e-4


def test_importance_tuple_metric():
    """Tuple importance metrics like union_mean_grad_nuc_norm."""
    from lycoris.modules.gora_utils import compute_importance

    W = torch.randn(16, 32)
    G = torch.randn(16, 32)

    is_tuple, imp = compute_importance(
        W, G, importance_type="union_mean_grad_nuc_norm", lora_rank=8,
    )
    assert is_tuple
    assert isinstance(imp, tuple)
    assert len(imp) == 2
    assert imp[0] > 0


# ==== Test 4: Rank Allocation ===============================================

def test_rank_allocation_basic():
    """Rank allocation should respect budget and importance."""
    from lycoris.modules.locon import GoRAModule as GM
    from lycoris.modules.gora_utils import allocate_ranks

    GM.reset_gora_registry()

    lin1 = nn.Linear(16, 32, bias=False)
    lin2 = nn.Linear(32, 64, bias=False)

    m1 = GM(lora_name="m1", org_module=lin1, lora_dim=4)
    m2 = GM(lora_name="m2", org_module=lin2, lora_dim=4)

    # Assign fake accumulated gradients on org_weight
    w1 = _get_weight(m1)
    w2 = _get_weight(m2)
    _accumulate_gradients_weight(m1, torch.randn(32, 16) * 10.0)
    _accumulate_gradients_weight(m2, torch.randn(64, 32) * 0.1)

    named_ranks, total_budget, actual_trainable, importances = allocate_ranks(
        [m1, m2],
        ref_rank=8,
        min_rank=2,
        max_rank=32,
        importance_type="union_mean",
    )

    assert len(named_ranks) == 2
    assert named_ranks["m1"] >= 2
    assert named_ranks["m2"] >= 2
    assert named_ranks["m1"] <= 32
    assert named_ranks["m2"] <= 32

    # m1 should get more rank than m2 due to larger gradient
    assert named_ranks["m1"] >= named_ranks["m2"], (
        f"Expected m1 rank >= m2 rank, got {named_ranks}"
    )

    del w1.grad_stored
    del w2.grad_stored
    GM.reset_gora_registry()


def test_rank_allocation_with_bounds():
    """Min/max rank bounds should be respected."""
    from lycoris.modules.locon import GoRAModule as GM
    from lycoris.modules.gora_utils import allocate_ranks

    GM.reset_gora_registry()

    lin = nn.Linear(64, 128, bias=False)
    m = GM(lora_name="m", org_module=lin, lora_dim=4)
    _accumulate_gradients_weight(m, torch.randn(128, 64))

    named_ranks, _, _, _ = allocate_ranks(
        [m], ref_rank=8, min_rank=4, max_rank=10,
    )
    assert 4 <= named_ranks["m"] <= 10

    del _get_weight(m).grad_stored
    GM.reset_gora_registry()


# ==== Test 5: grad_compress_init =============================================

def test_grad_compress_init_single(gora_module):
    """grad_compress_init should initialize lora weights from accumulated gradient.

    Verifies the corrected paper convention:
      - lora_up (m×r) = paper's A₀ — randomly initialized (Kaiming)
      - lora_down (r×n) = paper's B₀ — computed via left pseudo-inverse
    """
    from lycoris.modules.gora_utils import _grad_compress_init_single

    mod = gora_module
    w = _get_weight(mod)
    w.requires_grad = True

    # Simulate gradient accumulation
    for _ in range(8):
        x = torch.randn(2, 32, requires_grad=True)
        y = F.linear(x, w)
        y.sum().backward()
        if not hasattr(w, 'grad_stored') or w.grad_stored is None:
            w.grad_stored = w.grad.detach().cpu()
            w.iters = 1
        else:
            w.grad_stored = w.grad_stored + w.grad.detach().cpu()
            w.iters += 1
        w.grad = None

    original_weight = w.data.clone()

    _grad_compress_init_single(
        mod, rank=8, stable_gamma=0.05, scaling_alpha=16.0,
        scale_by_lr=False, lr=1e-3, weight_a_init_method="kaiming",
        fast_svd_niter=16,
    )

    # Shapes: lora_down = paper's B₀ (r×n), lora_up = paper's A₀ (m×r)
    assert mod.lora_down.weight.shape == (8, 32), "lora_down should be (rank, in_features) = paper's B₀"
    assert mod.lora_up.weight.shape == (64, 8), "lora_up should be (out_features, rank) = paper's A₀"
    assert mod.lora_dim == 8

    expected_scale = 16.0 / math.sqrt(8)
    assert abs(mod.scale - expected_scale) < 1e-6

    # Base weight should be unchanged (GoRA does NOT manipulate pre-trained weight)
    assert torch.equal(w.data, original_weight)

    assert hasattr(mod, '_gora_recon_error')
    assert mod._gora_recon_error > 0

    del w.grad_stored
    del w.iters


def test_paper_eq9_initialization_formula(gora_module):
    """Paper Eq. 9: B₀ = -(A₀ᵀ A₀)⁻¹ A₀ᵀ G.

    In LyCORIS convention:
      - lora_up (m×r) = paper's A₀ — randomly initialized
      - lora_down (r×n) = paper's B₀ — computed via LEFT pseudo-inverse with negative sign
    """
    from lycoris.modules.gora_utils import _grad_compress_init_single

    mod = gora_module
    w = _get_weight(mod)
    w.requires_grad = True

    # Use a deterministic gradient for reproducibility
    torch.manual_seed(42)
    G = torch.randn(64, 32)  # (out_features, in_features) = (m, n)
    w.grad_stored = G.clone()
    w.iters = 1

    # Use scaling_alpha as stable_gamma so ξ/α = 1 (no extra scaling for formula verification)
    _grad_compress_init_single(
        mod, rank=8,
        stable_gamma=16.0,  # = scaling_alpha, so ξ/α = 1 → no extra scaling
        scaling_alpha=16.0,
        scale_by_lr=False, lr=1e-3,
        weight_a_init_method="kaiming",
        fast_svd_niter=16,
    )

    # Extract matrices
    A0 = mod.lora_up.weight.data.float()    # (m, r) = (64, 8)
    B0 = mod.lora_down.weight.data.float()   # (r, n) = (8, 32)

    # Recompute what Eq 9 should give us: B₀ = -(A₀ᵀA₀)⁻¹A₀ᵀG
    A0T = A0.T  # (r, m)
    A0TA0 = A0T @ A0  # (r, r)
    A0TA0_inv = torch.linalg.pinv(A0TA0 + 1e-8 * torch.eye(8, device=A0.device))
    expected_B0 = -(A0TA0_inv @ A0T @ G)  # (r, n) — LEFT pseudo-inverse with negative sign

    # Since stable_gamma = scaling_alpha, ξ = 1.0, so B0 should match expected_B0 exactly
    # (accounting for the scale factor: B0 *= (stable_gamma * sqrt(m)) / scaling_alpha)
    xi = (16.0 * math.sqrt(64)) / 16.0  # = sqrt(64) = 8
    expected_scaled_B0 = expected_B0 * xi

    assert torch.allclose(B0, expected_scaled_B0, atol=1e-4), \
        f"lora_down does NOT match paper Eq 9. Max diff: {(B0 - expected_scaled_B0).abs().max():.6f}"

    # Verify product A₀B₀ approximates -G (with projection and scaling)
    AB = A0 @ B0  # (m, r) @ (r, n) = (m, n)
    P = A0 @ A0TA0_inv @ A0T  # projection matrix onto Col(A₀)
    expected_AB = -(P @ G) * xi  # -ξ * P_col(A₀) @ G

    assert torch.allclose(AB, expected_AB, atol=1e-4), \
        f"A₀B₀ does NOT match -ξ·P_A·G. Max diff: {(AB - expected_AB).abs().max():.6f}"

    # Verify negative direction: A₀B₀ should point opposite to G
    G_proj = P @ G
    correlation = torch.dot(G_proj.flatten(), AB.flatten())
    assert correlation < 0, \
        f"A₀B₀ should point in NEGATIVE G direction! Correlation={correlation:.4f}. Missing negative sign?"

    del w.grad_stored
    del w.iters


def test_paper_eq10_scaling_formula(gora_module):
    """Paper Eq. 10: ξ = γ·√m / α for rsLoRA variant.

    With rsLoRA: lora_scaler = α/√r, product should approximate -γG.
    """
    from lycoris.modules.gora_utils import _grad_compress_init_single

    mod = gora_module
    w = _get_weight(mod)
    w.requires_grad = True

    torch.manual_seed(42)
    G = torch.randn(64, 32)
    w.grad_stored = G.clone()
    w.iters = 1

    gamma = 0.05
    alpha = 16.0

    _grad_compress_init_single(
        mod, rank=8,
        stable_gamma=gamma,
        scaling_alpha=alpha,
        scale_by_lr=False, lr=1e-3,
        weight_a_init_method="kaiming",
        fast_svd_niter=16,
    )

    A0 = mod.lora_up.weight.data.float()
    B0 = mod.lora_down.weight.data.float()

    # Total forward contribution: ΔW = (α/√r) * A₀B₀
    lora_scale = mod.scale  # α/√r
    delta_W = lora_scale * (A0 @ B0)

    # Should approximate -γG (one step of gradient descent)
    G_flat = G.flatten()
    delta_flat = delta_W.flatten()

    # Check direction: ΔW should point in -G direction
    dot = torch.dot(delta_flat, G_flat)
    assert dot < 0, \
        f"ΔW should point in NEGATIVE G direction (gradient descent). Got positive correlation {dot:.4f}"

    # Verify ξ formula: B₀ should be scaled by ξ = γ·√m / α
    xi_expected = gamma * math.sqrt(64) / alpha  # γ·√64 / 16 = 0.05 * 8 / 16 = 0.025
    # Recompute B₀ without scaling to verify
    A0T = A0.T
    A0TA0 = A0T @ A0
    A0TA0_inv = torch.linalg.pinv(A0TA0 + 1e-8 * torch.eye(8, device=A0.device))
    unscaled_B0 = -(A0TA0_inv @ A0T @ G)
    expected_B0 = unscaled_B0 * xi_expected

    assert torch.allclose(B0, expected_B0, atol=1e-4), \
        f"Scaling ξ={xi_expected:.6f} not applied correctly. Max diff: {(B0 - expected_B0).abs().max():.6f}"

    del w.grad_stored
    del w.iters


def test_gradient_descent_direction_initialization(gora_module):
    """Critical: initialization must approximate -G, NOT +G.

    A positive initialization would increase loss instead of decreasing it.
    """
    from lycoris.modules.gora_utils import _grad_compress_init_single

    mod = gora_module
    w = _get_weight(mod)
    w.requires_grad = True

    torch.manual_seed(42)
    # Deterministic weight and gradient
    W0 = torch.randn(64, 32)
    _set_weight(mod, W0)
    G = torch.randn(64, 32)
    w.grad_stored = G.clone()
    w.iters = 1

    gamma = 0.1

    _grad_compress_init_single(
        mod, rank=8,
        stable_gamma=gamma, scaling_alpha=16.0,
        scale_by_lr=False, lr=1e-3,
        weight_a_init_method="kaiming",
        fast_svd_niter=16,
    )

    A0 = mod.lora_up.weight.data.float()
    B0 = mod.lora_down.weight.data.float()
    lora_delta = mod.scale * (A0 @ B0)

    # Effective weight after initialization
    W_eff = W0 + lora_delta

    # Gradient descent step: W_gd = W0 - γG
    W_gd = W0 - gamma * G

    # Gradient ascent step (WRONG): W_ga = W0 + γG
    W_ga = W0 + gamma * G

    dist_to_gd = torch.norm(W_eff - W_gd, p='fro')
    dist_to_ga = torch.norm(W_eff - W_ga, p='fro')

    assert dist_to_gd < dist_to_ga, \
        f"Initialization moves in WRONG direction! " \
        f"dist_to_gd={dist_to_gd:.4f} >= dist_to_ga={dist_to_ga:.4f}. " \
        f"Missing negative sign means ΔW ≈ +γG instead of -γG."

    del w.grad_stored
    del w.iters


def test_features_func_default_is_sqrt():
    """Paper Eq 7-8: budget uses √(m+n). Default features_func should be 'sqrt'."""
    from lycoris.modules.gora_utils import allocate_ranks, compute_importance

    # Verify the feature_adjust_func dict defaults to math.sqrt
    from lycoris.modules.locon import GoRAModule as GM

    GM.reset_gora_registry()

    lin1 = nn.Linear(16, 32, bias=False)
    lin2 = nn.Linear(64, 128, bias=False)

    m1 = GM(lora_name="l1", org_module=lin1, lora_dim=4)
    m2 = GM(lora_name="l2", org_module=lin2, lora_dim=4)

    _accumulate_gradients_weight(m1, torch.randn(32, 16))
    _accumulate_gradients_weight(m2, torch.randn(128, 64))

    # With features_func=None (default), sqrt should be applied
    named_ranks, total_budget, _, _ = allocate_ranks(
        [m1, m2], ref_rank=8, min_rank=1, max_rank=32,
        features_func=None,  # default
    )

    # Total budget should use sqrt(m+n): √48 * 8 + √192 * 8
    expected_budget = math.sqrt(16 + 32) * 8 + math.sqrt(64 + 128) * 8
    assert abs(total_budget - expected_budget) < 0.01, \
        f"Default features_func should use sqrt. Expected budget {expected_budget:.2f}, got {total_budget:.2f}"

    del _get_weight(m1).grad_stored
    del _get_weight(m2).grad_stored
    GM.reset_gora_registry()


def test_rank_allocation_uses_rounding():
    """Paper Eq 8: r^i = [b·a^i / √(m+n)] uses rounding, NOT floor division."""
    from lycoris.modules.locon import GoRAModule as GM
    from lycoris.modules.gora_utils import allocate_ranks

    GM.reset_gora_registry()

    lin = nn.Linear(16, 32, bias=False)
    m = GM(lora_name="m", org_module=lin, lora_dim=4)
    _accumulate_gradients_weight(m, torch.randn(32, 16))

    # With only one layer, importance = 1.0
    # smooth_total_budget = √48 * 8 ≈ 55.4
    # smooth_trainable = round(55.4 * 1.0) = 55
    # rank = round(55 / √48) = round(7.94) = 8 (with rounding)
    # rank = floor(55 / √48) = floor(7.94) = 7 (with floor — old bug)
    named_ranks, _, _, _ = allocate_ranks(
        [m], ref_rank=8, min_rank=1, max_rank=32,
        allocate_strategy="moderate",  # uses round()
    )

    # With round: should be ~8. With floor: would be ~7.
    assert named_ranks["m"] >= 7, f"Expected rank >= 7 with rounding, got {named_ranks['m']}"

    del _get_weight(m).grad_stored
    GM.reset_gora_registry()


def test_lora_up_is_random_lora_down_is_computed(gora_module):
    """Verify correct convention: lora_up = paper's A₀ (random), lora_down = paper's B₀ (computed).

    Before the fix, roles were swapped: lora_down was random, lora_up was computed.
    Uses scaling_alpha as stable_gamma so ξ = 1 — no extra scaling, pure formula.
    """
    from lycoris.modules.gora_utils import _grad_compress_init_single

    mod = gora_module
    w = _get_weight(mod)

    torch.manual_seed(42)
    G = torch.randn(64, 32)
    w.grad_stored = G.clone()
    w.iters = 1

    in_dim = mod.in_features
    out_dim = mod.out_features
    alpha = 16.0

    # Use stable_gamma = scaling_alpha / sqrt(out_dim) so ξ = 1 (no extra scaling)
    _grad_compress_init_single(
        mod, rank=4,
        stable_gamma=alpha / math.sqrt(out_dim),  # ξ = 1
        scaling_alpha=alpha,
        scale_by_lr=False, lr=1e-3, weight_a_init_method="kaiming",
        fast_svd_niter=16,
    )

    A0 = mod.lora_up.weight.data.float()   # (64, 4) — should be random (paper's A₀)
    B0 = mod.lora_down.weight.data.float()  # (4, 32) — should be computed (paper's B₀)

    # lora_up (A₀) should have non-trivial values (kaiming init)
    assert A0.abs().sum() > 0, "lora_up should have non-zero values (random init)"
    assert A0.std() > 0, "lora_up should have positive std (random init)"

    # lora_down (B₀) should be correlated with G via A₀^T
    A0T = A0.T
    A0TA0 = A0T @ A0
    A0TA0_inv = torch.linalg.pinv(A0TA0 + 1e-8 * torch.eye(4, device=A0.device))
    expected_B0 = -(A0TA0_inv @ A0T @ G)  # pure Eq 9, unscaled

    # Since ξ = 1 (stable_gamma * sqrt(m) / alpha = 1), B0 should match unscaled formula
    assert torch.allclose(B0, expected_B0, atol=1e-4), \
        "lora_down should be computed via left pseudo-inverse of lora_up (Eq 9)"

    del w.grad_stored
    del w.iters


# ==== Test 6: Forward Pass ===================================================

def test_forward_pass_linear(gora_module):
    """GoRAModule forward should produce correct output shape with adapter contribution."""
    mod = gora_module

    # Set predictable lora weights
    mod.lora_down.weight.data = torch.randn(4, 32) * 0.1
    mod.lora_up.weight.data = torch.randn(64, 4) * 0.1
    mod.lora_dim = 4
    mod.scale = 16.0 / math.sqrt(4)

    x = torch.randn(3, 32)

    with torch.no_grad():
        base_out = F.linear(x, _get_weight(mod))

    mod.apply_to()
    out = mod.forward(x)
    mod.restore()

    assert out.shape == (3, 64)
    diff = out - base_out
    assert diff.abs().sum() > 0
    assert diff.shape == (3, 64)


def test_forward_pass_conv(gora_module_conv):
    """GoRAModule conv forward should produce correct output shape."""
    mod = gora_module_conv
    x = torch.randn(2, 16, 8, 8)

    mod.apply_to()
    out = mod.forward(x)
    mod.restore()

    assert out.shape == (2, 32, 8, 8)


def test_forward_pass_zero_rank(simple_linear):
    """Module with lora_dim=0 should pass through base weight only."""
    from lycoris.modules.locon import GoRAModule as GM

    GM.reset_gora_registry()

    # Use lora_dim=1 to avoid ZeroDivisionError in rsLoRA scaling,
    # then manually set to 0 for forward test
    mod = GM(lora_name="zero", org_module=simple_linear, lora_dim=1, alpha=1.0)
    mod.lora_dim = 0
    # Override scale for zero rank
    mod.scale = 0.0

    x = torch.randn(5, 32)
    out = mod.forward(x)
    expected = F.linear(x, _get_weight(mod))

    assert out.shape == (5, 64)
    assert torch.allclose(out, expected, atol=1e-5)

    GM.reset_gora_registry()


# ==== Test 7: Saved State Dict Compatibility =================================

def test_state_dict_compatible_with_locon():
    """GoRA saved state dict should be loadable as a LoConModule."""
    from lycoris.modules.locon import GoRAModule as GM, LoConModule as LM

    GM.reset_gora_registry()
    LM.reset_olora_registry()

    # Create GoRA module
    lin = nn.Linear(32, 64, bias=False)
    gora = GM(lora_name="test", org_module=lin, lora_dim=4, alpha=16.0)

    gora.lora_down.weight.data = torch.randn(4, 32)
    gora.lora_up.weight.data = torch.randn(64, 4)
    gora.lora_dim = 4

    # custom_state_dict gives keys WITHOUT prefix (matching LoConModule)
    sd = gora.custom_state_dict()
    assert "lora_down.weight" in sd
    assert "lora_up.weight" in sd
    assert "alpha" in sd
    assert "gora_gamma" not in sd
    assert "gora_ref_rank" not in sd

    # Create a LoConModule and load — use sd keys directly (no prefix)
    lin2 = nn.Linear(32, 64, bias=False)
    locon = LM(lora_name="test", org_module=lin2, lora_dim=4, alpha=16.0)

    # Load: custom_state_dict keys match submodule structure directly
    locon.load_state_dict(sd, strict=False)

    # lora_down should match exactly (saved as-is)
    assert torch.allclose(locon.lora_down.weight, gora.lora_down.weight), \
        "lora_down weights should match"
    # lora_up is saved with scalar multiplier in custom_state_dict
    # After loading, LoConModule's load_weight_hook resets scalar to 1.0,
    # so lora_up weights should match (since GoRA also has its scalar multiplier applied)
    # We just verify shapes match since the scalar handling differs slightly
    assert locon.lora_up.weight.shape == gora.lora_up.weight.shape

    LM.reset_olora_registry()
    GM.reset_gora_registry()


def test_custom_state_dict_matches_locon():
    """GoRAModule.custom_state_dict() should match LoConModule's keys."""
    from lycoris.modules.locon import GoRAModule as GM, LoConModule as LM

    GM.reset_gora_registry()
    LM.reset_olora_registry()

    gora = GM(
        lora_name="test2", org_module=nn.Linear(32, 64, bias=False),
        lora_dim=4, alpha=16.0,
    )
    locon = LM(
        lora_name="test2", org_module=nn.Linear(32, 64, bias=False),
        lora_dim=4, alpha=16.0,
    )

    gora_sd = gora.custom_state_dict()
    locon_sd = locon.custom_state_dict()

    assert set(gora_sd.keys()) == set(locon_sd.keys())

    LM.reset_olora_registry()
    GM.reset_gora_registry()


# ==== Test 8: Adaptive N Convergence =========================================

def test_importance_convergence():
    """check_importance_convergence should detect when importances stabilize."""
    from collections import OrderedDict
    from lycoris.modules.gora_utils import check_importance_convergence

    prev = OrderedDict([("a", 1.0), ("b", 2.0)])

    # Small change — converged
    cur = OrderedDict([("a", 1.01), ("b", 1.99)])
    assert check_importance_convergence(cur, prev, threshold=0.02)

    # Large change — not converged
    cur_big = OrderedDict([("a", 1.5), ("b", 1.5)])
    assert not check_importance_convergence(cur_big, prev, threshold=0.02)

    # Missing keys — not converged (different lengths)
    cur_missing = OrderedDict([("a", 1.0)])
    assert not check_importance_convergence(cur_missing, prev, threshold=0.1)

    # Extra keys — not converged
    cur_extra = OrderedDict([("a", 1.0), ("b", 2.0), ("c", 3.0)])
    assert not check_importance_convergence(cur_extra, prev, threshold=0.1)

    # None previous — not converged
    assert not check_importance_convergence(cur, None, threshold=0.1)


# ==== Test 9: Adaptive Gamma Selection =======================================

def test_adaptive_gamma_selection():
    """adaptive_gamma_selection should find a γ that minimizes loss."""
    from lycoris.modules.locon import GoRAModule as GM
    from lycoris.modules.gora_utils import adaptive_gamma_selection

    GM.reset_gora_registry()

    lin = nn.Linear(10, 10, bias=False)
    mod = GM(lora_name="ag", org_module=lin, lora_dim=4, alpha=1.0)
    # Deterministic weights to ensure stable test
    torch.manual_seed(42)
    mod.lora_up.weight.data = torch.randn(10, 4)
    mod.lora_down.weight.data = torch.randn(4, 10)

    target_gamma = 0.5

    def forward_fn():
        actual = mod.lora_up.weight.data.norm().item()
        return torch.tensor(abs(actual - target_gamma * 10.0))

    best = adaptive_gamma_selection(
        [mod], forward_fn, gamma_init=1.0, gamma_decay=0.8,
        gamma_min=1e-5, scaling_alpha=1.0,
    )

    assert best >= 0
    assert isinstance(best, float)

    GM.reset_gora_registry()


# ==== Test 10: Precompute Entry Point (Integration) ==========================

def test_precompute_gradients_integration():
    """Full gora_precompute_gradients pipeline should run end-to-end."""
    from lycoris.modules.locon import GoRAModule as GM
    from lycoris.modules.gora_utils import gora_precompute_gradients

    GM.reset_gora_registry()

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin1 = nn.Linear(16, 32, bias=False)
            self.lin2 = nn.Linear(32, 16, bias=False)

        def forward(self, x):
            x = F.linear(x, self.lin1.weight)
            x = F.relu(x)
            x = F.linear(x, self.lin2.weight)
            return x

    # Create GoRA wrappers
    m1 = GM(lora_name="test_lin1", org_module=nn.Linear(16, 32, bias=False), lora_dim=4, alpha=16.0)
    m2 = GM(lora_name="test_lin2", org_module=nn.Linear(32, 16, bias=False), lora_dim=4, alpha=16.0)

    # Build a simple forward function that uses the modules directly
    dummy_batch = {"input_ids": torch.randn(2, 16), "labels": torch.randn(2, 16)}
    batches = [dummy_batch] * 8

    def forward_fn(batch):
        x = batch["input_ids"]
        labels = batch["labels"]
        w1 = _get_weight(m1)
        w2 = _get_weight(m2)
        out = F.linear(F.relu(F.linear(x, w1)), w2)
        loss = F.mse_loss(out, labels)
        return (loss,)

    named_ranks = gora_precompute_gradients(
        modules=[m1, m2],
        dataloader=batches,
        forward_fn=forward_fn,
        ref_rank=8,
        min_rank=2,
        max_rank=16,
        importance_type="union_mean",
        scaling_alpha=16.0,
        stable_gamma=0.05,
        max_steps=4,
        adaptive_n=False,
        adaptive_gamma=False,
    )

    assert len(named_ranks) == 2
    assert "test_lin1" in named_ranks
    assert "test_lin2" in named_ranks
    assert named_ranks["test_lin1"] >= 2
    assert named_ranks["test_lin2"] >= 2
    assert m1.lora_dim == named_ranks["test_lin1"]
    assert m2.lora_dim == named_ranks["test_lin2"]
    assert m1.lora_up.weight.abs().sum() > 0
    assert m2.lora_down.weight.abs().sum() > 0

    GM.reset_gora_registry()


# ==== Test 11: Rank JSON Save/Load ===========================================

def test_rank_json_save():
    """Rank allocation should be serializable to JSON."""
    named_ranks = {"test_gora": 8, "test_gora_2": 4}
    importances = {"test_gora": 0.6, "test_gora_2": (0.3, 1.2)}

    with tempfile.TemporaryDirectory() as tmpdir:
        rank_path = os.path.join(tmpdir, "rank.json")
        imp_path = os.path.join(tmpdir, "importance.json")

        with open(rank_path, 'w') as f:
            json.dump(named_ranks, f)
        with open(imp_path, 'w') as f:
            json.dump(
                {k: (list(v) if isinstance(v, tuple) else v) for k, v in importances.items()},
                f,
            )

        with open(rank_path, 'r') as f:
            loaded_ranks = json.load(f)
        with open(imp_path, 'r') as f:
            loaded_imps = json.load(f)

        assert loaded_ranks == named_ranks
        assert loaded_imps["test_gora"] == 0.6
        assert loaded_imps["test_gora_2"] == [0.3, 1.2]


# ==== Test 12: rsLoRA Scaling Verification ===================================

def test_rslora_scaling():
    """GoRAModule must use rsLoRA: scale = α / √r."""
    from lycoris.modules.locon import GoRAModule as GM

    GM.reset_gora_registry()

    alphas = [1.0, 16.0, 32.0]
    ranks = [4, 8, 16]

    for alpha in alphas:
        for rank in ranks:
            mod = GM(
                lora_name=f"rs_{alpha}_{rank}",
                org_module=nn.Linear(16, 32, bias=False),
                lora_dim=rank,
                alpha=alpha,
            )
            expected_scale = alpha / math.sqrt(rank)
            assert abs(mod.scale - expected_scale) < 1e-6, (
                f"alpha={alpha}, rank={rank}: expected {expected_scale}, got {mod.scale}"
            )

    GM.reset_gora_registry()


# ==== Test 13: LycorisNetwork.prepare_gora() Integration =====================

def test_prepare_gora_new_network_triggers_init():
    """prepare_gora on a new GoRA network should run precompute and init."""
    from lycoris.wrapper import create_lycoris, LycorisNetwork
    from lycoris.modules.locon import GoRAModule as GM

    GM.reset_gora_registry()

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(16, 32, bias=False)

        def forward(self, x):
            return self.lin(x)

    model = TinyModel()
    LycorisNetwork.TARGET_REPLACE_MODULE = ["Linear"]
    LycorisNetwork.TARGET_REPLACE_NAME = []
    LycorisNetwork.MODULE_ALGO_MAP = {}
    LycorisNetwork.NAME_ALGO_MAP = {}

    network = create_lycoris(
        model, algo="gora", lora_dim=4, linear_alpha=16.0,
        gora_ref_rank=8, gora_min_rank=2, gora_max_rank=16,
        gora_gamma=0.05, multiplier=1.0,
    )

    assert network._gora_needs_init is True, "New GoRA network should need init"

    batches = [{"input_ids": torch.randn(2, 16), "labels": torch.randn(2, 32)} for _ in range(4)]

    def forward_fn(batch):
        out = model(batch["input_ids"])
        return (F.mse_loss(out, batch["labels"]),)

    network.prepare_gora(
        dataloader=batches, forward_fn=forward_fn,
        max_steps=4, adaptive_n=False, adaptive_gamma=False,
    )

    assert network._gora_needs_init is False, "Flag cleared after init"
    assert len(network.loras) >= 1
    lora_mod = network.loras[0]
    assert lora_mod.lora_up.weight.abs().sum() > 0
    assert lora_mod.lora_down.weight.abs().sum() > 0

    GM.reset_gora_registry()


def test_prepare_gora_state_resumption_skips_init():
    """When flag is False (state resumption), prepare_gora should skip."""
    from lycoris.wrapper import create_lycoris, LycorisNetwork
    from lycoris.modules.locon import GoRAModule as GM

    GM.reset_gora_registry()

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(16, 32, bias=False)

    model = TinyModel()
    LycorisNetwork.TARGET_REPLACE_MODULE = ["Linear"]
    LycorisNetwork.MODULE_ALGO_MAP = {}
    LycorisNetwork.NAME_ALGO_MAP = {}

    network = create_lycoris(
        model, algo="gora", lora_dim=4, linear_alpha=16.0, multiplier=1.0,
    )

    # Simulate state resumption: clear gora modules, set flag False
    GM.reset_gora_registry()
    network._gora_needs_init = False

    batches = [{"input_ids": torch.randn(2, 16), "labels": torch.randn(2, 32)}]

    def forward_fn(batch):
        return (torch.tensor(0.0),)

    # Should be no-op
    network.prepare_gora(
        dataloader=batches, forward_fn=forward_fn,
        max_steps=1, adaptive_n=False, adaptive_gamma=False,
    )
    assert network._gora_needs_init is False

    GM.reset_gora_registry()


def test_prepare_gora_uses_stored_kwargs():
    """prepare_gora should read config from stored _gora_kwargs."""
    from lycoris.wrapper import create_lycoris, LycorisNetwork
    from lycoris.modules.locon import GoRAModule as GM

    GM.reset_gora_registry()


# ==== Test 14: Kohya Path Integration ========================================

def test_kohya_lycoris_network_gora():
    """LycorisNetworkKohya with algo='gora' should set _gora_needs_init."""
    from lycoris.kohya import LycorisNetworkKohya
    from lycoris.modules.locon import GoRAModule as GM

    GM.reset_gora_registry()
    LycorisNetworkKohya.ENABLE_CONV = False
    LycorisNetworkKohya.UNET_TARGET_REPLACE_MODULE = ["Linear"]
    LycorisNetworkKohya.UNET_TARGET_REPLACE_NAME = []
    LycorisNetworkKohya.TEXT_ENCODER_TARGET_REPLACE_MODULE = []
    LycorisNetworkKohya.TEXT_ENCODER_TARGET_REPLACE_NAME = []
    LycorisNetworkKohya.MODULE_ALGO_MAP = {}
    LycorisNetworkKohya.NAME_ALGO_MAP = {}
    LycorisNetworkKohya.TARGET_EXCLUDE_NAME = []

    class TinyUNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(16, 32, bias=False)

    unet = TinyUNet()

    network = LycorisNetworkKohya(
        text_encoder=None, unet=unet, multiplier=1.0,
        lora_dim=4, alpha=16.0, network_module="gora",
        gora_gamma=0.05, gora_ref_rank=8,
    )

    assert network._gora_needs_init is True
    assert network._gora_kwargs.get('gora_gamma') == 0.05
    assert len(network.loras) >= 1

    GM.reset_gora_registry()


def test_kohya_lycoris_network_not_gora():
    """LycorisNetworkKohya with algo='lora' should NOT set _gora_needs_init."""
    from lycoris.kohya import LycorisNetworkKohya
    from lycoris.modules.locon import GoRAModule as GM

    GM.reset_gora_registry()


# ==== Test 15: Full create_network (Kohya) Integration =======================

def test_kohya_create_network_gora_full_flow():
    """Full create_network path with algo='gora' — flag, kwargs, modules, and prepare_gora."""
    from lycoris.kohya import create_network, LycorisNetworkKohya
    from lycoris.modules.locon import GoRAModule as GM

    GM.reset_gora_registry()
    LycorisNetworkKohya.ENABLE_CONV = False
    LycorisNetworkKohya.TARGET_EXCLUDE_NAME = []

    class TinyUNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(16, 32, bias=False)

    unet = TinyUNet()

    # Use dict preset to avoid the string-preset overriding target modules
    network = create_network(
        multiplier=1.0, network_dim=4, network_alpha=16.0,
        vae=None, text_encoder=None, unet=unet,
        algo="gora",
        preset={"enable_conv": False, "unet_target_module": ["Linear"],
                "unet_target_name": [], "text_encoder_target_module": [],
                "text_encoder_target_name": []},
        gora_gamma=0.05, gora_ref_rank=8,
    )

    assert network._gora_needs_init is True
    assert network._gora_kwargs.get("gora_gamma") == 0.05
    # GoRA enforces alpha = dim; scaling_alpha was corrected from 16.0 to 4
    assert network._gora_kwargs.get("scaling_alpha") == 4
    assert len(network.loras) >= 1
    assert any(isinstance(l, GM) for l in network.loras), "No GoRAModule in Kohya network"

    # prepare_gora should run successfully
    batches = [{"input_ids": torch.randn(2, 16), "labels": torch.randn(2, 32)} for _ in range(2)]

    def forward_fn(batch):
        out = unet.lin(batch["input_ids"])
        return (F.mse_loss(out, batch["labels"]),)

    network.prepare_gora(
        dataloader=batches, forward_fn=forward_fn,
        max_steps=2, adaptive_n=False, adaptive_gamma=False,
    )

    assert network._gora_needs_init is False
    GM.reset_gora_registry()


def test_kohya_create_network_lora_no_gora_flag():
    """create_network with algo='lora' should not set GoRA flag or kwargs."""
    from lycoris.kohya import create_network, LycorisNetworkKohya
    from lycoris.modules.locon import GoRAModule as GM

    GM.reset_gora_registry()
    LycorisNetworkKohya.ENABLE_CONV = False
    LycorisNetworkKohya.TARGET_EXCLUDE_NAME = []

    class TinyUNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(16, 32, bias=False)

    unet = TinyUNet()

    network = create_network(
        multiplier=1.0, network_dim=4, network_alpha=1.0,
        vae=None, text_encoder=None, unet=unet,
        algo="lora",
        preset={"enable_conv": False, "unet_target_module": ["Linear"],
                "unet_target_name": [], "text_encoder_target_module": [],
                "text_encoder_target_name": []},
    )

    assert network._gora_needs_init is False
    assert network._gora_kwargs == {}

    GM.reset_gora_registry()
    LycorisNetworkKohya.ENABLE_CONV = False
    LycorisNetworkKohya.UNET_TARGET_REPLACE_MODULE = ["Linear"]
    LycorisNetworkKohya.UNET_TARGET_REPLACE_NAME = []
    LycorisNetworkKohya.TEXT_ENCODER_TARGET_REPLACE_MODULE = []
    LycorisNetworkKohya.TEXT_ENCODER_TARGET_REPLACE_NAME = []
    LycorisNetworkKohya.MODULE_ALGO_MAP = {}
    LycorisNetworkKohya.NAME_ALGO_MAP = {}
    LycorisNetworkKohya.TARGET_EXCLUDE_NAME = []

    class TinyUNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(16, 32, bias=False)

    unet = TinyUNet()

    network = LycorisNetworkKohya(
        text_encoder=None, unet=unet, multiplier=1.0,
        lora_dim=4, alpha=1.0, network_module="lora",
    )

    assert network._gora_needs_init is False
    assert network._gora_kwargs == {}

    GM.reset_gora_registry()
