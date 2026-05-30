"""
Verification tests for SVD initialization correctness in LoConModule.

Tests cover:
1. Core SVD decomposition math (lora_up @ lora_down = rank-r approximation)
2. Residual subtraction correctness for all segment types
3. PiSSA conversion correctness (custom_state_dict vs convert_pissa_to_lora)
4. QPiSSA initialization correctness
5. Conv layer SVD initialization
6. Scalar handling in PiSSA conversion
"""

import math
import torch
import torch.nn as nn
import pytest

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32


def _make_linear(in_f=64, out_f=32, device=DEVICE):
    """Create a linear module with random weights."""
    mod = nn.Linear(in_f, out_f, bias=False).to(device)
    nn.init.kaiming_uniform_(mod.weight, a=math.sqrt(5))
    return mod


def _make_conv2d(in_c=8, out_c=16, k=3, device=DEVICE):
    """Create a conv2d module with random weights."""
    mod = nn.Conv2d(in_c, out_c, k, padding=k // 2, bias=False).to(device)
    nn.init.kaiming_uniform_(mod.weight, a=math.sqrt(5))
    return mod


def _make_locon(org_module, lora_dim=4, alpha=None, use_scalar=False,
                pissa_niter=0, segment=None, **kwargs):
    """Create a LoConModule with SVD init if segment is specified.

    Moves the LoCon module to the same device as org_module.
    """
    from lycoris.modules.locon import LoConModule

    if alpha is None:
        alpha = lora_dim  # default: scale = 1.0

    extra_kwargs = {}
    if segment is not None:
        extra_kwargs["svd_segment"] = segment
    extra_kwargs.update(kwargs)

    module = LoConModule(
        "test_layer",
        org_module,
        multiplier=1.0,
        lora_dim=lora_dim,
        alpha=alpha,
        use_scalar=use_scalar,
        pissa_niter=pissa_niter,
        **extra_kwargs,
    )
    # Move to same device as org_module (LoConModule is not auto-moved)
    module = module.to(org_module.weight.device)
    # Re-ensure pissa buffers are on the right device
    if hasattr(module, "pissa_A_init") and module.pissa_A_init is not None:
        module.pissa_A_init = module.pissa_A_init.to(org_module.weight.device)
    if hasattr(module, "pissa_B_init") and module.pissa_B_init is not None:
        module.pissa_B_init = module.pissa_B_init.to(org_module.weight.device)
    return module


def _to_cpu(t):
    """Move tensor to CPU for comparison."""
    if isinstance(t, torch.Tensor):
        return t.detach().float().cpu()
    return t


# --------------------------------------------------------------------------
# Test 1: Core SVD math — lora_up @ lora_down reconstructs rank-r approx
# --------------------------------------------------------------------------

class TestSVDDecomposition:
    """Verify that lora_up @ lora_down = rank-r SVD approximation."""

    @pytest.mark.parametrize("segment", ["top", "middle", "bottom"])
    def test_linear_svd_reconstruction(self, segment):
        """For linear layers, lora_up @ lora_down should equal the
        rank-r SVD approximation of the selected segment."""
        org = _make_linear(64, 32)
        orig_weight = org.weight.data.clone()
        lora_dim = 4

        mod = _make_locon(org, lora_dim=lora_dim, segment=segment)

        # Compute expected rank-r SVD approximation (on same device)
        W = orig_weight.float()
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        h = len(S)
        if segment == "top":
            start = 0
        elif segment == "middle":
            start = (h - lora_dim) // 2
        else:  # bottom
            start = h - lora_dim

        U_r = U[:, start:start + lora_dim]
        S_r = S[start:start + lora_dim]
        Vh_r = Vh[start:start + lora_dim]

        expected_approx = U_r @ torch.diag(S_r) @ Vh_r  # (out, in)

        # Compute actual from lora weights (on same device as module)
        up = mod.lora_up.weight.data.float()   # (out, r)
        down = mod.lora_down.weight.data.float()  # (r, in)
        actual_approx = up @ down  # (out, in)

        torch.testing.assert_close(
            _to_cpu(actual_approx), _to_cpu(expected_approx),
            atol=1e-4, rtol=1e-4,
            msg=f"SVD reconstruction mismatch for segment={segment}"
        )

    @pytest.mark.parametrize("segment", ["top", "middle", "bottom"])
    def test_conv2d_svd_reconstruction(self, segment):
        """For conv2d layers, the flattened lora_up @ lora_down should equal
        the rank-r SVD approximation of the flattened weight."""
        org = _make_conv2d(8, 16, 3)
        orig_weight = org.weight.data.clone()
        lora_dim = 4

        mod = _make_locon(org, lora_dim=lora_dim, segment=segment)

        # Flatten original weight to 2D
        W_2d = orig_weight.float().reshape(orig_weight.shape[0], -1)
        U, S, Vh = torch.linalg.svd(W_2d, full_matrices=False)
        h = len(S)
        if segment == "top":
            start = 0
        elif segment == "middle":
            start = (h - lora_dim) // 2
        else:
            start = h - lora_dim

        U_r = U[:, start:start + lora_dim]
        S_r = S[start:start + lora_dim]
        Vh_r = Vh[start:start + lora_dim]
        expected_approx = U_r @ torch.diag(S_r) @ Vh_r  # (out, in*kh*kw)

        # Flatten lora weights
        up = mod.lora_up.weight.data.float().reshape(mod.lora_up.weight.shape[0], -1)
        down = mod.lora_down.weight.data.float().reshape(mod.lora_down.weight.shape[0], -1)
        actual_approx = up @ down

        torch.testing.assert_close(
            _to_cpu(actual_approx), _to_cpu(expected_approx),
            atol=1e-4, rtol=1e-4,
            msg=f"Conv2d SVD reconstruction mismatch for segment={segment}"
        )


# --------------------------------------------------------------------------
# Test 2: Residual subtraction — base_weight + adapter ≈ original_weight
# --------------------------------------------------------------------------

class TestResidualSubtraction:
    """After SVD init, base_weight + adapter_weight should ≈ original_weight."""

    def _check_residual(self, org_module, locon_module, orig_weight):
        """Check that org_weight + locon_diff ≈ original_weight."""
        with torch.no_grad():
            diff = locon_module.make_weight().float() * locon_module.scale
            base_weight = org_module.weight.data.float()
            reconstructed = base_weight + diff.reshape(base_weight.shape)
            torch.testing.assert_close(
                _to_cpu(reconstructed), _to_cpu(orig_weight.float()),
                atol=1e-3, rtol=1e-3,
                msg="Residual: base + adapter != original"
            )

    @pytest.mark.parametrize("segment", ["top", "middle", "bottom"])
    def test_linear_residual(self, segment):
        org = _make_linear(64, 32)
        orig_weight = org.weight.data.clone()
        alpha = 4  # scale = 1.0
        mod = _make_locon(org, lora_dim=4, alpha=alpha, segment=segment)
        self._check_residual(org, mod, orig_weight)

    @pytest.mark.parametrize("segment", ["top", "middle", "bottom"])
    def test_conv2d_residual(self, segment):
        org = _make_conv2d(8, 16, 3)
        orig_weight = org.weight.data.clone()
        alpha = 4  # scale = 1.0
        mod = _make_locon(org, lora_dim=4, alpha=alpha, segment=segment)
        self._check_residual(org, mod, orig_weight)


# --------------------------------------------------------------------------
# Test 3: PiSSA init (top segment) stores correct initial weights
# --------------------------------------------------------------------------

class TestPiSSAInit:
    """Verify PiSSA-specific initialization (top segment, pissa_A_init, pissa_B_init)."""

    def test_pissa_stores_initial_weights(self):
        org = _make_linear(64, 32)
        mod = _make_locon(org, lora_dim=4, segment="top")

        assert mod.is_pissa is True
        assert mod.pissa_A_init is not None
        assert mod.pissa_B_init is not None

        # pissa_A_init should match lora_up at init time
        torch.testing.assert_close(
            _to_cpu(mod.pissa_A_init),
            _to_cpu(mod.lora_up.weight.data),
            atol=1e-6, rtol=1e-6,
        )
        # pissa_B_init should match lora_down at init time
        torch.testing.assert_close(
            _to_cpu(mod.pissa_B_init),
            _to_cpu(mod.lora_down.weight.data),
            atol=1e-6, rtol=1e-6,
        )

    def test_pissa_residual_is_W_minus_Wr(self):
        """PiSSA base weight should be W - W_r (principal residual)."""
        org = _make_linear(64, 32)
        orig_weight = org.weight.data.clone()
        lora_dim = 4

        mod = _make_locon(org, lora_dim=lora_dim, segment="top")

        # Expected residual
        W = orig_weight.float()
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        U_r, S_r, Vh_r = U[:, :lora_dim], S[:lora_dim], Vh[:lora_dim]
        W_r = U_r @ torch.diag(S_r) @ Vh_r
        expected_residual = W - W_r

        actual_residual = org.weight.data.float()
        torch.testing.assert_close(
            _to_cpu(actual_residual), _to_cpu(expected_residual),
            atol=1e-3, rtol=1e-3,
            msg="PiSSA residual != W - W_r"
        )

    def test_pissa_alpha_not_equal_lora_dim_breaks_reconstruction(self):
        """When alpha != lora_dim, PiSSA reconstruction is NOT exact.
        This documents the known limitation."""
        org = _make_linear(64, 32)
        orig_weight = org.weight.data.clone()

        # alpha=8, lora_dim=4 → scale=2.0
        mod = _make_locon(org, lora_dim=4, alpha=8, segment="top")

        with torch.no_grad():
            diff = mod.make_weight().float() * mod.scale
            base_weight = org.weight.data.float()
            reconstructed = base_weight + diff.reshape(base_weight.shape)

        # This will NOT be equal to orig_weight because scale != 1
        error = (_to_cpu(reconstructed) - _to_cpu(orig_weight.float())).abs().max().item()
        assert error > 0.01, (
            f"Expected nonzero error when alpha != lora_dim, got {error}"
        )


# --------------------------------------------------------------------------
# Test 4: convert_pissa_to_lora scalar handling
# --------------------------------------------------------------------------

class TestPiSSAConversion:
    """Verify PiSSA→LoRA conversion correctness, especially with scalar."""

    def test_conversion_without_scalar(self):
        """Without use_scalar, convert_pissa_to_lora should preserve the
        effective output (base + adapter) across the conversion boundary."""
        org = _make_linear(64, 32)
        orig_weight = org.weight.data.clone()

        mod = _make_locon(org, lora_dim=4, alpha=4, segment="top", use_scalar=False)

        # Simulate a training step: perturb lora_up slightly
        with torch.no_grad():
            mod.lora_up.weight.data += torch.randn_like(mod.lora_up.weight) * 0.01

        # Record the effective forward BEFORE conversion:
        #   effective = base_weight + adapter * scalar * scale
        with torch.no_grad():
            adapter_before = mod.lora_up.weight.float() @ mod.lora_down.weight.float()
            base_before = org.weight.data.float()
            effective_before = base_before + adapter_before.reshape(base_before.shape) * mod.scalar.float() * mod.scale

        # Convert
        result = mod.convert_pissa_to_lora()
        assert result is True

        # After conversion, base should be restored to original
        torch.testing.assert_close(
            _to_cpu(org.weight.data), _to_cpu(orig_weight),
            atol=1e-3, rtol=1e-3,
            msg="Base weight not restored after conversion"
        )

        # Record the effective forward AFTER conversion:
        #   effective = base_weight + adapter * scalar * scale
        with torch.no_grad():
            adapter_after = mod.lora_up.weight.float() @ mod.lora_down.weight.float()
            base_after = org.weight.data.float()
            effective_after = base_after + adapter_after.reshape(base_after.shape) * mod.scalar.float() * mod.scale

        # The effective weight should be the same before and after conversion
        torch.testing.assert_close(
            _to_cpu(effective_before), _to_cpu(effective_after),
            atol=1e-3, rtol=1e-3,
            msg="Effective weight changed after convert_pissa_to_lora (no scalar)"
        )

    def test_conversion_with_scalar_custom_state_dict(self):
        """custom_state_dict correctly bakes scalar into PiSSA conversion."""
        org = _make_linear(64, 32)
        orig_weight = org.weight.data.clone()

        mod = _make_locon(org, lora_dim=4, alpha=4, segment="top", use_scalar=True)

        # Set scalar to a non-trivial value
        with torch.no_grad():
            mod.scalar.data.fill_(2.5)
            mod.lora_up.weight.data += torch.randn_like(mod.lora_up.weight) * 0.01

        # Record the trained adapter effect:
        # delta = scalar * A'B' - A₀B₀
        pissa_A0 = mod.pissa_A_init.float()
        pissa_B0 = mod.pissa_B_init.float()
        with torch.no_grad():
            trained_effect = (
                mod.scalar.float() * mod.lora_up.weight.float() @ mod.lora_down.weight.float()
                - pissa_A0 @ pissa_B0
            )

        # Test custom_state_dict conversion (correct path)
        state_dict = mod.custom_state_dict()
        # On load, scalar is reset to 1.0, so forward = up @ down * 1.0
        sd_effect = state_dict["lora_up.weight"].float() @ state_dict["lora_down.weight"].float()

        torch.testing.assert_close(
            _to_cpu(sd_effect.reshape(orig_weight.shape)),
            _to_cpu(trained_effect.reshape(orig_weight.shape)),
            atol=1e-3, rtol=1e-3,
            msg="custom_state_dict PiSSA conversion doesn't match trained effect"
        )

    def test_conversion_with_scalar_method(self):
        """convert_pissa_to_lora() now correctly bakes scalar into lora_up.

        With use_scalar=True and scalar != 1, the conversion absorbs scalar
        into lora_up and resets scalar to 1.0, matching custom_state_dict.
        """
        org = _make_linear(64, 32)
        orig_weight = org.weight.data.clone()

        mod = _make_locon(org, lora_dim=4, alpha=4, segment="top", use_scalar=True)

        with torch.no_grad():
            mod.scalar.data.fill_(2.5)
            mod.lora_up.weight.data += torch.randn_like(mod.lora_up.weight) * 0.01

        # Record the effective output BEFORE conversion
        with torch.no_grad():
            adapter_before = mod.lora_up.weight.float() @ mod.lora_down.weight.float()
            base_before = org.weight.data.float()
            effective_before = (base_before
                                + adapter_before.reshape(base_before.shape)
                                * mod.scalar.float() * mod.scale)

        mod.convert_pissa_to_lora()

        # After conversion, scalar should be reset to 1.0
        assert abs(float(mod.scalar) - 1.0) < 1e-6, (
            f"scalar should be 1.0 after conversion, got {float(mod.scalar)}"
        )

        # The effective output should be preserved
        with torch.no_grad():
            adapter_after = mod.lora_up.weight.float() @ mod.lora_down.weight.float()
            base_after = org.weight.data.float()
            effective_after = (base_after
                               + adapter_after.reshape(base_after.shape)
                               * mod.scalar.float() * mod.scale)

        torch.testing.assert_close(
            _to_cpu(effective_before), _to_cpu(effective_after),
            atol=1e-3, rtol=1e-3,
            msg="Effective weight changed after convert_pissa_to_lora (with scalar)"
        )


# --------------------------------------------------------------------------
# Test 5: QPiSSA initialization
# --------------------------------------------------------------------------

class TestQPiSSAInit:
    """Verify QPiSSA iterative initialization."""

    def test_qpissa_residual_quality(self):
        """QPiSSA should produce a residual that when combined with adapter
        reconstructs the original weight."""
        org = _make_linear(64, 32)
        orig_weight = org.weight.data.clone()

        def identity_quant(w):
            return w, w

        mod = _make_locon(
            org, lora_dim=4, alpha=4,
            qpissa_iter=3, quant_fn=identity_quant,
        )

        assert mod.is_pissa is True

        # With identity quant, residual + adapter should ≈ original weight
        with torch.no_grad():
            adapter = mod.lora_up.weight.float() @ mod.lora_down.weight.float()
            base = org.weight.data.float()
            reconstructed = base + adapter.reshape(base.shape)

        torch.testing.assert_close(
            _to_cpu(reconstructed), _to_cpu(orig_weight.float()),
            atol=1e-2, rtol=1e-2,
            msg="QPiSSA with identity quant: residual + adapter != original"
        )

    def test_qpissa_adapter_reconstruction(self):
        """QPiSSA adapter weights should reconstruct a valid SVD approximation."""
        org = _make_linear(64, 32)

        def identity_quant(w):
            return w, w

        mod = _make_locon(
            org, lora_dim=4, alpha=4,
            qpissa_iter=1, quant_fn=identity_quant,
        )

        with torch.no_grad():
            approx = mod.lora_up.weight.float() @ mod.lora_down.weight.float()

        # The adapter shouldn't be zero
        assert approx.abs().sum().item() > 0, "QPiSSA adapter is all zeros"


# --------------------------------------------------------------------------
# Test 6: Randomized SVD (fast PiSSA) correctness
# --------------------------------------------------------------------------

class TestRandomizedSVD:
    """Verify that randomized SVD (pissa_niter > 0) gives reasonable results."""

    def test_randomized_svd_close_to_exact(self):
        """Randomized SVD should produce a result close to exact SVD."""
        org = _make_linear(128, 64)
        orig_weight = org.weight.data.clone()

        # Exact SVD init
        org_exact = _make_linear(128, 64)
        org_exact.weight.data.copy_(orig_weight)
        mod_exact = _make_locon(org_exact, lora_dim=8, segment="top", pissa_niter=0)

        # Randomized SVD init
        org_fast = _make_linear(128, 64)
        org_fast.weight.data.copy_(orig_weight)
        mod_fast = _make_locon(org_fast, lora_dim=8, segment="top", pissa_niter=5)

        # Both should produce valid approximations
        W = orig_weight.float()
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        W_r = U[:, :8] @ torch.diag(S[:8]) @ Vh[:8]

        exact_approx = _to_cpu(mod_exact.lora_up.weight.float()
                               @ mod_exact.lora_down.weight.float())
        fast_approx = _to_cpu(mod_fast.lora_up.weight.float()
                              @ mod_fast.lora_down.weight.float())
        W_r_cpu = _to_cpu(W_r)

        # Exact should be very close to W_r
        exact_error = (exact_approx - W_r_cpu).abs().max().item()
        assert exact_error < 1e-3, f"Exact SVD error too large: {exact_error}"

        # Randomized should be reasonably close (but not exact)
        fast_error = (fast_approx - W_r_cpu).abs().max().item()
        assert fast_error < 0.5, f"Randomized SVD error too large: {fast_error}"


# --------------------------------------------------------------------------
# Test 7: Middle/bottom segment correctness
# --------------------------------------------------------------------------

class TestSegmentSelection:
    """Verify SVD segments select the correct singular value ranges."""

    def test_top_segment_selects_largest_singular_values(self):
        org = _make_linear(64, 32)
        mod = _make_locon(org, lora_dim=4, segment="top")

        # Reconstruct full weight
        with torch.no_grad():
            adapter = mod.lora_up.weight.float() @ mod.lora_down.weight.float()
            base = org.weight.data.float()
            full_W = base + adapter.reshape(base.shape)

        U, S, Vh = torch.linalg.svd(full_W, full_matrices=False)

        # The adapter should capture the top-4 singular values
        adapter_svd = torch.linalg.svd(adapter, full_matrices=False)[1]

        torch.testing.assert_close(
            _to_cpu(adapter_svd[:4]), _to_cpu(S[:4]),
            atol=1e-2, rtol=1e-2,
            msg="Top segment doesn't capture largest singular values"
        )

    def test_bottom_segment_selects_smallest_singular_values(self):
        org = _make_linear(64, 32)
        lora_dim = 4

        mod = _make_locon(org, lora_dim=lora_dim, segment="bottom")

        # Reconstruct full weight
        with torch.no_grad():
            adapter = mod.lora_up.weight.float() @ mod.lora_down.weight.float()
            base = org.weight.data.float()
            full_W = base + adapter.reshape(base.shape)

        U, S, Vh = torch.linalg.svd(full_W, full_matrices=False)
        h = len(S)

        # The adapter should capture the bottom-4 singular values
        adapter_svd = torch.linalg.svd(adapter, full_matrices=False)[1]

        expected_sv = S[h - lora_dim:]
        actual_sv = adapter_svd[:lora_dim]

        torch.testing.assert_close(
            _to_cpu(actual_sv), _to_cpu(expected_sv),
            atol=1e-2, rtol=1e-2,
            msg="Bottom segment doesn't capture smallest singular values"
        )


# --------------------------------------------------------------------------
# Test 8: Scale interaction with SVD init
# --------------------------------------------------------------------------

class TestScaleInteraction:
    """Verify that the scale factor interacts correctly with SVD init."""

    @pytest.mark.parametrize("alpha,lora_dim,expected_scale", [
        (4, 4, 1.0),      # alpha == lora_dim → scale = 1.0
        (8, 4, 2.0),      # alpha = 2 * lora_dim → scale = 2.0
        (2, 4, 0.5),      # alpha = lora_dim/2 → scale = 0.5
    ])
    def test_non_pissa_scale_residual(self, alpha, lora_dim, expected_scale):
        """For non-PiSSA segments, residual = W - W_segment * scale,
        so forward at multiplier=1 gives W back."""
        org = _make_linear(64, 32)
        orig_weight = org.weight.data.clone()

        mod = _make_locon(org, lora_dim=lora_dim, alpha=alpha, segment="middle")

        assert abs(mod.scale - expected_scale) < 1e-6, (
            f"Expected scale={expected_scale}, got {mod.scale}"
        )

        with torch.no_grad():
            diff = mod.make_weight().float() * mod.scale
            base = org.weight.data.float()
            reconstructed = base + diff.reshape(base.shape)

        torch.testing.assert_close(
            _to_cpu(reconstructed), _to_cpu(orig_weight.float()),
            atol=1e-3, rtol=1e-3,
            msg=f"Non-PiSSA residual incorrect with alpha={alpha}, lora_dim={lora_dim}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
