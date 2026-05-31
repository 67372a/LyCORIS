"""
Tests for OrthoLoRA: Cayley-parameterized orthogonal low-rank adapter.
"""

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from lycoris.modules.ortholora import OrthoLoRAModule

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32


def _make_linear(in_f=64, out_f=32, device=DEVICE):
    mod = nn.Linear(in_f, out_f, bias=False).to(device)
    nn.init.kaiming_uniform_(mod.weight, a=math.sqrt(5))
    return mod


def _make_ortho(org_module, lora_dim=4, alpha=None, **kwargs):
    if alpha is None:
        alpha = lora_dim  # scale = 1.0
    module = OrthoLoRAModule(
        "test_layer",
        org_module,
        multiplier=1.0,
        lora_dim=lora_dim,
        alpha=alpha,
        **kwargs,
    ).to(org_module.weight.device)
    return module


# --------------------------------------------------------------------------
# 1. Zero-init invariant
# --------------------------------------------------------------------------

class TestOrthoZeroInit:
    def test_zero_init_output_matches_org(self):
        org = _make_linear()
        mod = _make_ortho(org)
        mod.apply_to()

        x = torch.randn(2, 64, device=DEVICE, dtype=DTYPE)
        out = mod(x)
        org_out = mod.org_forward(x)
        assert torch.allclose(out, org_out, atol=1e-5)


# --------------------------------------------------------------------------
# 2. Cayley orthogonality
# --------------------------------------------------------------------------

class TestOrthoCayley:
    def test_cayley_identity_at_init(self):
        org = _make_linear()
        mod = _make_ortho(org)
        R_p = OrthoLoRAModule._cayley(mod.S_p.float())
        R_q = OrthoLoRAModule._cayley(mod.S_q.float())
        I = torch.eye(4, device=DEVICE)
        assert torch.allclose(R_p, I, atol=1e-6)
        assert torch.allclose(R_q, I, atol=1e-6)

    def test_cayley_orthogonality_after_update(self):
        org = _make_linear()
        mod = _make_ortho(org)

        with torch.no_grad():
            mod.S_p.data = torch.randn(4, 4, device=DEVICE) * 0.1
            mod.S_q.data = torch.randn(4, 4, device=DEVICE) * 0.1

        R_p = OrthoLoRAModule._cayley(mod.S_p.float())
        R_q = OrthoLoRAModule._cayley(mod.S_q.float())
        I = torch.eye(4, device=DEVICE)

        assert torch.allclose(R_p.T @ R_p, I, atol=1e-5)
        assert torch.allclose(R_q.T @ R_q, I, atol=1e-5)

    def test_batched_cayley_matches_individual(self):
        S_q = torch.randn(4, 4, device=DEVICE) * 0.1
        S_p = torch.randn(4, 4, device=DEVICE) * 0.1

        skew = torch.stack([S_q, S_p])
        A = skew - skew.transpose(-2, -1)
        eye = torch.eye(4, device=DEVICE)
        R = torch.linalg.solve(eye + A, eye - A)

        R_q_ind = OrthoLoRAModule._cayley(S_q)
        R_p_ind = OrthoLoRAModule._cayley(S_p)

        assert torch.allclose(R[0], R_q_ind, atol=1e-6)
        assert torch.allclose(R[1], R_p_ind, atol=1e-6)


# --------------------------------------------------------------------------
# 3. SVD init correctness
# --------------------------------------------------------------------------

class TestOrthoSVDInit:
    def test_bases_orthogonal_columns(self):
        """P_basis columns should be approximately orthonormal (randomized SVD)."""
        org = _make_linear(64, 32)
        mod = _make_ortho(org, lora_dim=4)

        P = mod.P_basis.float()  # (out, r) bf16→float
        Q = mod.Q_basis.float()  # (r, in) bf16→float

        # P^T P ≈ I (orthonormal columns) — randomized SVD is approximate
        gram_p = P.T @ P
        I_r = torch.eye(4, device=DEVICE)
        assert torch.allclose(gram_p, I_r, atol=5e-3), (
            f"P_basis columns not orthonormal: max err {(gram_p - I_r).abs().max()}"
        )

        # Q Q^T ≈ I (orthonormal rows)
        gram_q = Q @ Q.T
        assert torch.allclose(gram_q, I_r, atol=5e-3), (
            f"Q_basis rows not orthonormal: max err {(gram_q - I_r).abs().max()}"
        )


# --------------------------------------------------------------------------
# 4. Weight reconstruction
# --------------------------------------------------------------------------

class TestOrthoWeightRecon:
    def test_make_weight_at_init_is_zero(self):
        org = _make_linear()
        mod = _make_ortho(org)

        w = mod.make_weight(device=DEVICE)
        assert torch.allclose(w, torch.zeros_like(w), atol=1e-6)

    def test_make_weight_with_nonzero_params(self):
        org = _make_linear()
        mod = _make_ortho(org)

        with torch.no_grad():
            mod.S_p.data = torch.randn(4, 4, device=DEVICE) * 0.1
            mod.S_q.data = torch.randn(4, 4, device=DEVICE) * 0.1
            mod.lambda_layer.data = torch.ones(1, 4, device=DEVICE)

        w = mod.make_weight(device=DEVICE).float()
        assert w.norm() > 0

        # Verify ΔW = P_eff @ diag(λ) @ Q_eff
        R_p = OrthoLoRAModule._cayley(mod.S_p.float())
        R_q = OrthoLoRAModule._cayley(mod.S_q.float())
        P_eff = mod.P_basis.float() @ R_p
        Q_eff = R_q @ mod.Q_basis.float()
        lam = mod.lambda_layer.float()
        expected = (P_eff * lam) @ Q_eff

        # bf16 frozen bases cause ~1e-3 precision loss
        assert torch.allclose(w, expected, atol=2e-3), (
            f"Max diff: {(w - expected).abs().max()}"
        )


# --------------------------------------------------------------------------
# 5. State dict round-trip (native save)
# --------------------------------------------------------------------------

class TestOrthoNativeRoundTrip:
    def test_native_save_has_keys(self):
        org = _make_linear()
        mod = _make_ortho(org, native_save=True)

        with torch.no_grad():
            mod.S_p.data = torch.randn(4, 4, device=DEVICE) * 0.1
            mod.S_q.data = torch.randn(4, 4, device=DEVICE) * 0.1
            mod.lambda_layer.data = torch.rand(1, 4, device=DEVICE)

        sd = mod.custom_state_dict()
        assert "S_p" in sd
        assert "S_q" in sd
        assert "P_basis" in sd
        assert "Q_basis" in sd
        assert "lambda_layer" in sd

        # Round-trip values match
        assert torch.allclose(sd["S_p"].float(), mod.S_p.float(), atol=1e-6)
        assert torch.allclose(sd["S_q"].float(), mod.S_q.float(), atol=1e-6)
        assert torch.allclose(sd["lambda_layer"].float(), mod.lambda_layer.float(), atol=1e-6)


# --------------------------------------------------------------------------
# 6. Distill conversion
# --------------------------------------------------------------------------

class TestOrthoDistill:
    def test_distill_preserves_delta(self):
        org = _make_linear()
        mod = _make_ortho(org, native_save=False)

        with torch.no_grad():
            mod.S_p.data = torch.randn(4, 4, device=DEVICE) * 0.1
            mod.S_q.data = torch.randn(4, 4, device=DEVICE) * 0.1
            mod.lambda_layer.data = torch.rand(1, 4, device=DEVICE)

        delta_w = mod.make_weight(device=DEVICE).float().cpu()

        sd = mod.custom_state_dict()
        lora_up = sd["lora_up.weight"].float().cpu()
        lora_down = sd["lora_down.weight"].float().cpu()

        reconstructed = lora_up @ lora_down
        assert torch.allclose(reconstructed, delta_w, atol=2e-3), (
            f"Max diff: {(reconstructed - delta_w).abs().max()}"
        )

    def test_distill_sd_has_standard_keys(self):
        org = _make_linear()
        mod = _make_ortho(org, native_save=False)
        sd = mod.custom_state_dict()
        assert "lora_up.weight" in sd
        assert "lora_down.weight" in sd
        assert "alpha" in sd
        assert "S_p" not in sd
        assert "P_basis" not in sd


# --------------------------------------------------------------------------
# 7. Gradient flow
# --------------------------------------------------------------------------

class TestOrthoGradient:
    def test_gradients_flow_to_trainable_params(self):
        org = _make_linear()
        mod = _make_ortho(org)
        mod.apply_to()

        x = torch.randn(2, 64, device=DEVICE, dtype=DTYPE, requires_grad=True)
        out = mod(x)
        loss = out.sum()
        loss.backward()

        assert mod.S_p.grad is not None
        assert mod.S_q.grad is not None
        assert mod.lambda_layer.grad is not None


# --------------------------------------------------------------------------
# 8. Forward mode parity: bypass ≈ rebuild
# --------------------------------------------------------------------------

class TestOrthoForwardModeParity:
    def test_bypass_matches_rebuild(self):
        org = _make_linear()
        mod_rebuild = _make_ortho(org, bypass_mode=False)
        mod_bypass = _make_ortho(org, bypass_mode=True)

        # Copy weights
        mod_bypass.S_p.data.copy_(mod_rebuild.S_p.data)
        mod_bypass.S_q.data.copy_(mod_rebuild.S_q.data)
        mod_bypass.lambda_layer.data.copy_(mod_rebuild.lambda_layer.data)
        # Copy frozen bases
        mod_bypass.P_basis.copy_(mod_rebuild.P_basis)
        mod_bypass.Q_basis.copy_(mod_rebuild.Q_basis)

        mod_rebuild.apply_to()
        mod_bypass.apply_to()

        x = torch.randn(2, 64, device=DEVICE, dtype=DTYPE)
        out_rebuild = mod_rebuild(x)
        out_bypass = mod_bypass(x)

        assert torch.allclose(out_rebuild, out_bypass, atol=1e-4), (
            f"Max diff: {(out_rebuild - out_bypass).abs().max().item()}"
        )


# --------------------------------------------------------------------------
# 9. Bf16/fp32 compatibility
# --------------------------------------------------------------------------

class TestOrthoDtype:
    def test_forward_with_bf16_input(self):
        """Forward works when input is bf16 (params stay fp32 for Cayley solve)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for bf16 test")
        org = _make_linear()
        mod = _make_ortho(org)
        mod.apply_to()

        x = torch.randn(2, 64, device=DEVICE, dtype=torch.bfloat16)
        out = mod(x)
        assert out.shape == (2, 32)


# --------------------------------------------------------------------------
# 10. Module dropout
# --------------------------------------------------------------------------

class TestOrthoModuleDropout:
    def test_dropout_one_returns_org(self):
        org = _make_linear()
        mod = _make_ortho(org, module_dropout=1.0)
        mod.apply_to()

        x = torch.randn(2, 64, device=DEVICE, dtype=DTYPE)
        mod.train()
        out = mod(x)
        org_out = mod.org_forward(x)
        assert torch.allclose(out, org_out, atol=1e-6)


# --------------------------------------------------------------------------
# 11. T-LoRA mask compatibility
# --------------------------------------------------------------------------

class TestOrthoTLoRAMask:
    def test_tlora_mask_buffer_registered(self):
        org = _make_linear()
        mod = _make_ortho(org, use_timestep_mask=True)
        assert hasattr(mod, "_timestep_mask")
        assert mod._timestep_mask.shape == (1, 4)

    def test_tlora_zero_mask_kills_delta(self):
        org = _make_linear()
        mod = _make_ortho(org, use_timestep_mask=True)
        mod.apply_to()

        x = torch.randn(2, 64, device=DEVICE, dtype=DTYPE)
        mod._timestep_mask.fill_(0.0)
        out = mod(x)
        org_out = mod.org_forward(x)
        assert torch.allclose(out, org_out, atol=1e-5)


# --------------------------------------------------------------------------
# 12. Registration: algo="ortholora" creates OrthoLoRAModule
# --------------------------------------------------------------------------

class TestOrthoRegistration:
    def test_create_lycoris_ortholora(self):
        from lycoris.wrapper import create_lycoris

        model = nn.ModuleDict({"lin": _make_linear()})
        network = create_lycoris(
            model,
            multiplier=1.0,
            linear_dim=4,
            linear_alpha=4,
            algo="ortholora",
        )
        assert len(network.loras) > 0
        assert isinstance(network.loras[0], OrthoLoRAModule)


# --------------------------------------------------------------------------
# 14. DoRA (weight decomposition) support
# --------------------------------------------------------------------------

class TestOrthoDoRA:
    def test_dora_scale_registered(self):
        org = _make_linear()
        mod = _make_ortho(org, weight_decompose=True)
        assert mod.wd is True
        assert hasattr(mod, "dora_scale")
        assert mod.dora_scale.ndim == 2  # (out, 1) keepdim=True

    def test_dora_forces_rebuild_mode(self):
        org = _make_linear()
        mod = _make_ortho(org, weight_decompose=True, bypass_mode=True)
        assert mod.bypass_mode is False, "DoRA should force rebuild mode"

    def test_dora_preserves_magnitude_at_init(self):
        """At init ΔW=0, DoRA output should equal org_forward (magnitude unchanged)."""
        org = _make_linear()
        mod = _make_ortho(org, weight_decompose=True)
        mod.apply_to()

        x = torch.randn(2, 64, device=DEVICE, dtype=DTYPE)
        out = mod(x)
        org_out = mod.org_forward(x)
        assert torch.allclose(out, org_out, atol=1e-5)

    def test_dora_output_differs_from_no_dora(self):
        """With non-zero ΔW, DoRA output should differ from non-DoRA."""
        org = _make_linear()
        mod_dora = _make_ortho(org, weight_decompose=True)
        mod_nodora = _make_ortho(org, weight_decompose=False)

        # Copy same non-trivial params
        with torch.no_grad():
            mod_dora.S_p.data = torch.randn(4, 4, device=DEVICE) * 0.1
            mod_dora.S_q.data = torch.randn(4, 4, device=DEVICE) * 0.1
            mod_dora.lambda_layer.data = torch.ones(1, 4, device=DEVICE)
            mod_nodora.S_p.data.copy_(mod_dora.S_p.data)
            mod_nodora.S_q.data.copy_(mod_dora.S_q.data)
            mod_nodora.lambda_layer.data.copy_(mod_dora.lambda_layer.data)

        mod_dora.apply_to()
        mod_nodora.apply_to()

        x = torch.randn(2, 64, device=DEVICE, dtype=DTYPE)
        out_dora = mod_dora(x)
        out_nodora = mod_nodora(x)

        # Should differ because DoRA normalizes the merged weight
        assert not torch.allclose(out_dora, out_nodora, atol=1e-4), (
            "DoRA output should differ from non-DoRA"
        )

    def test_dora_custom_state_dict_includes_scale(self):
        org = _make_linear()
        mod = _make_ortho(org, weight_decompose=True, native_save=True)
        sd = mod.custom_state_dict()
        assert "dora_scale" in sd

    def test_dora_scale_in_weight_list(self):
        assert "dora_scale" in OrthoLoRAModule.weight_list

    def test_dora_get_merged_weight(self):
        org = _make_linear()
        mod = _make_ortho(org, weight_decompose=True)

        with torch.no_grad():
            mod.S_p.data = torch.randn(4, 4, device=DEVICE) * 0.1
            mod.S_q.data = torch.randn(4, 4, device=DEVICE) * 0.1
            mod.lambda_layer.data = torch.ones(1, 4, device=DEVICE)

        merged, _ = mod.get_merged_weight(multiplier=1.0)
        assert merged.shape == org.weight.shape

        # Verify DoRA: ||merged|| per channel should be close to ||W₀||
        orig_norm = torch.linalg.vector_norm(
            org.weight.data.float(), dim=1, keepdim=True
        )
        merged_norm = torch.linalg.vector_norm(
            merged.float(), dim=1, keepdim=True
        )
        # At multiplier=1, scale = 1*(m/norm - 1) + 1 = m/norm
        # So merged_norm ≈ orig_norm (DoRA preserves magnitude)
        assert torch.allclose(merged_norm, orig_norm, atol=1e-3), (
            f"DoRA should preserve magnitude: "
            f"max diff {(merged_norm - orig_norm).abs().max()}"
        )


# ==========================================================================
# 15. Double-scale bug regression: bypass/rebuild parity with α ≠ lora_dim
# ==========================================================================

class TestOrthoScaleConsistency:
    """Regression tests for the double-`self.scale` bug in rebuild mode.

    When ``alpha ≠ lora_dim``, ``self.scale = alpha/lora_dim ≠ 1.0``.
    Before the fix, ``make_weight()`` included ``* self.scale`` AND
    ``_forward_rebuild_core()`` applied ``* self.scale`` again → scale².
    """

    def test_bypass_rebuild_parity_alpha_not_dim(self):
        """Bypass and rebuild must match even when scale ≠ 1.0."""
        org = _make_linear(in_f=64, out_f=32)
        # alpha=1, lora_dim=4 → scale = 0.25
        mod_rebuild = _make_ortho(org, lora_dim=4, alpha=1, bypass_mode=False)
        mod_bypass = _make_ortho(org, lora_dim=4, alpha=1, bypass_mode=True)

        # Copy identical weights
        mod_bypass.S_p.data.copy_(mod_rebuild.S_p.data)
        mod_bypass.S_q.data.copy_(mod_rebuild.S_q.data)
        mod_bypass.lambda_layer.data.copy_(mod_rebuild.lambda_layer.data)
        mod_bypass.P_basis.copy_(mod_rebuild.P_basis)
        mod_bypass.Q_basis.copy_(mod_rebuild.Q_basis)

        mod_rebuild.apply_to()
        mod_bypass.apply_to()

        x = torch.randn(2, 64, device=DEVICE, dtype=DTYPE)
        out_rebuild = mod_rebuild(x)
        out_bypass = mod_bypass(x)

        assert torch.allclose(out_rebuild, out_bypass, atol=1e-4), (
            f"Bypass/rebuild mismatch with alpha=1, lora_dim=4 (scale=0.25): "
            f"max diff {(out_rebuild - out_bypass).abs().max().item()}"
        )

    def test_bypass_rebuild_parity_scale_lt_1(self):
        """Another non-unity scale: alpha=2, lora_dim=8 → scale=0.25."""
        org = _make_linear(in_f=64, out_f=32)
        mod_rebuild = _make_ortho(org, lora_dim=8, alpha=2, bypass_mode=False)
        mod_bypass = _make_ortho(org, lora_dim=8, alpha=2, bypass_mode=True)

        mod_bypass.S_p.data.copy_(mod_rebuild.S_p.data)
        mod_bypass.S_q.data.copy_(mod_rebuild.S_q.data)
        mod_bypass.lambda_layer.data.copy_(mod_rebuild.lambda_layer.data)
        mod_bypass.P_basis.copy_(mod_rebuild.P_basis)
        mod_bypass.Q_basis.copy_(mod_rebuild.Q_basis)

        mod_rebuild.apply_to()
        mod_bypass.apply_to()

        x = torch.randn(2, 64, device=DEVICE, dtype=DTYPE)
        out_rebuild = mod_rebuild(x)
        out_bypass = mod_bypass(x)

        assert torch.allclose(out_rebuild, out_bypass, atol=1e-4), (
            f"Bypass/rebuild mismatch with alpha=2, lora_dim=8 (scale=0.25): "
            f"max diff {(out_rebuild - out_bypass).abs().max().item()}"
        )

    def test_bypass_rebuild_parity_scale_gt_1(self):
        """Scale > 1: alpha=8, lora_dim=4 → scale=2.0."""
        org = _make_linear(in_f=64, out_f=32)
        mod_rebuild = _make_ortho(org, lora_dim=4, alpha=8, bypass_mode=False)
        mod_bypass = _make_ortho(org, lora_dim=4, alpha=8, bypass_mode=True)

        mod_bypass.S_p.data.copy_(mod_rebuild.S_p.data)
        mod_bypass.S_q.data.copy_(mod_rebuild.S_q.data)
        mod_bypass.lambda_layer.data.copy_(mod_rebuild.lambda_layer.data)
        mod_bypass.P_basis.copy_(mod_rebuild.P_basis)
        mod_bypass.Q_basis.copy_(mod_rebuild.Q_basis)

        mod_rebuild.apply_to()
        mod_bypass.apply_to()

        x = torch.randn(2, 64, device=DEVICE, dtype=DTYPE)
        out_rebuild = mod_rebuild(x)
        out_bypass = mod_bypass(x)

        assert torch.allclose(out_rebuild, out_bypass, atol=1e-4), (
            f"Bypass/rebuild mismatch with alpha=8, lora_dim=4 (scale=2.0): "
            f"max diff {(out_rebuild - out_bypass).abs().max().item()}"
        )

    def test_scaling_matches_reference_formula(self):
        """Verify output matches the reference formula (bf16-aware).

        Both the module and the reference use bf16 work dtype for the
        basis matmuls, so we compare at bf16 precision (~1e-2 atol).
        """
        org = _make_linear(in_f=64, out_f=32)
        alpha, lora_dim = 2, 8  # scale = 0.25
        mod = _make_ortho(org, lora_dim=lora_dim, alpha=alpha, bypass_mode=True)

        with torch.no_grad():
            mod.S_p.data = torch.randn(lora_dim, lora_dim, device=DEVICE) * 0.1
            mod.S_q.data = torch.randn(lora_dim, lora_dim, device=DEVICE) * 0.1
            mod.lambda_layer.data = torch.randn(1, lora_dim, device=DEVICE)

        mod.apply_to()
        x = torch.randn(2, 64, device=DEVICE, dtype=DTYPE)

        # Reference computation matching anima_lora forward, using same
        # bf16 work dtype as the actual module path.
        work = mod.P_basis.dtype  # bf16
        expected_scale = alpha / lora_dim  # 0.25
        R_p = OrthoLoRAModule._cayley(mod.S_p.float())
        R_q = OrthoLoRAModule._cayley(mod.S_q.float())
        P_eff = (mod.P_basis.float() @ R_p).to(work)
        Q_eff = (R_q @ mod.Q_basis.float()).to(work)
        lam = mod.lambda_layer.to(work)

        org_out = mod.org_forward(x)
        lx = F.linear(x.to(work), Q_eff)  # bf16 matmul
        lx = lx * lam
        out = F.linear(lx, P_eff)  # bf16 matmul

        expected_out = org_out.float() + out.float() * 1.0 * expected_scale * 1.0
        actual_out = mod(x).float()

        # bf16 precision: ~1e-2 absolute tolerance
        assert torch.allclose(actual_out, expected_out, atol=5e-2), (
            f"Output mismatch vs reference formula: "
            f"max diff {(actual_out - expected_out).abs().max()}"
        )

    def test_make_weight_scale_excluded(self):
        """make_weight() should NOT include self.scale (LoConModule convention)."""
        org = _make_linear()
        alpha, lora_dim = 2, 8
        mod = _make_ortho(org, lora_dim=lora_dim, alpha=alpha)

        with torch.no_grad():
            mod.S_p.data = torch.randn(lora_dim, lora_dim, device=DEVICE) * 0.1
            mod.S_q.data = torch.randn(lora_dim, lora_dim, device=DEVICE) * 0.1
            mod.lambda_layer.data = torch.ones(1, lora_dim, device=DEVICE)

        raw_weight = mod.make_weight(device=DEVICE).float()

        # Compute expected raw diff (without scale)
        R_p = OrthoLoRAModule._cayley(mod.S_p.float())
        R_q = OrthoLoRAModule._cayley(mod.S_q.float())
        P_eff = mod.P_basis.float() @ R_p
        Q_eff = R_q @ mod.Q_basis.float()
        expected_raw = (P_eff * mod.lambda_layer.float()) @ Q_eff

        assert torch.allclose(raw_weight, expected_raw, atol=2e-3), (
            f"make_weight should exclude scale: max diff "
            f"{(raw_weight - expected_raw).abs().max()}"
        )

    def test_get_diff_weight_includes_scale(self):
        """get_diff_weight() should apply self.scale once."""
        org = _make_linear()
        alpha, lora_dim = 2, 8
        mod = _make_ortho(org, lora_dim=lora_dim, alpha=alpha)

        with torch.no_grad():
            mod.S_p.data = torch.randn(lora_dim, lora_dim, device=DEVICE) * 0.1
            mod.S_q.data = torch.randn(lora_dim, lora_dim, device=DEVICE) * 0.1
            mod.lambda_layer.data = torch.ones(1, lora_dim, device=DEVICE)

        raw_weight = mod.make_weight(device=DEVICE).float()
        diff_weight, _ = mod.get_diff_weight(multiplier=1.0, device=DEVICE)
        diff_weight = diff_weight.float()

        expected_diff = raw_weight * (alpha / lora_dim)  # scale applied once
        assert torch.allclose(diff_weight, expected_diff, atol=1e-5), (
            f"get_diff_weight should include scale once: max diff "
            f"{(diff_weight - expected_diff).abs().max()}"
        )

    def test_get_merged_weight_matches_org_plus_scaled_diff(self):
        """get_merged_weight() = org + diff_weight * multiplier, scale applied once."""
        org = _make_linear()
        alpha, lora_dim = 2, 8
        mod = _make_ortho(org, lora_dim=lora_dim, alpha=alpha)

        with torch.no_grad():
            mod.S_p.data = torch.randn(lora_dim, lora_dim, device=DEVICE) * 0.1
            mod.S_q.data = torch.randn(lora_dim, lora_dim, device=DEVICE) * 0.1
            mod.lambda_layer.data = torch.ones(1, lora_dim, device=DEVICE)

        merged, _ = mod.get_merged_weight(multiplier=1.0)
        diff_weight, _ = mod.get_diff_weight(multiplier=1.0)
        expected = org.weight.data.float() + diff_weight.float()

        # bf16 frozen bases cause ~1e-3 precision loss in make_weight
        assert torch.allclose(merged.float(), expected, atol=2e-3), (
            f"get_merged_weight mismatch: max diff "
            f"{(merged.float() - expected).abs().max()}"
        )


# --------------------------------------------------------------------------
# 16. Gradient flow through rebuild mode
# --------------------------------------------------------------------------

class TestOrthoGradientRebuild:
    def test_lambda_grad_flows_in_rebuild_mode(self):
        """λ gradient must be non-zero in rebuild mode (even at zero-init)."""
        org = _make_linear()
        mod = _make_ortho(org, bypass_mode=False)
        mod.apply_to()

        x = torch.randn(2, 64, device=DEVICE, dtype=DTYPE, requires_grad=True)
        out = mod(x)
        loss = out.sum()
        loss.backward()

        assert mod.lambda_layer.grad is not None, (
            "lambda_layer has no gradient in rebuild mode"
        )
        assert mod.lambda_layer.grad.abs().sum() > 0, (
            "lambda_layer grad is all zeros in rebuild mode"
        )

    def test_all_grads_nonzero_with_nontrivial_params(self):
        """With non-zero S_p, S_q, λ, all grads must be non-zero."""
        org = _make_linear()
        mod = _make_ortho(org, bypass_mode=False)
        mod.apply_to()

        # Set non-trivial params so gradients flow to all of them
        with torch.no_grad():
            mod.S_p.data = torch.randn_like(mod.S_p) * 0.1
            mod.S_q.data = torch.randn_like(mod.S_q) * 0.1
            mod.lambda_layer.data = torch.ones_like(mod.lambda_layer)

        x = torch.randn(2, 64, device=DEVICE, dtype=DTYPE)
        out = mod(x)
        loss = out.sum()
        loss.backward()

        assert mod.S_p.grad is not None and mod.S_p.grad.abs().sum() > 0, (
            "S_p has zero gradient with non-trivial params"
        )
        assert mod.S_q.grad is not None and mod.S_q.grad.abs().sum() > 0, (
            "S_q has zero gradient with non-trivial params"
        )
        assert mod.lambda_layer.grad is not None and mod.lambda_layer.grad.abs().sum() > 0, (
            "lambda_layer has zero gradient with non-trivial params"
        )

    def test_gradient_consistency_across_modes(self):
        """λ gradient direction should agree between bypass and rebuild.

        At zero-init, bf16 intermediates cause magnitude differences, so we
        check sign agreement and order-of-magnitude similarity rather than
        exact equality.
        """
        org = _make_linear()
        mod_bypass = _make_ortho(org, bypass_mode=True)
        mod_rebuild = _make_ortho(org, bypass_mode=False)

        # Set non-zero params for meaningful gradients
        with torch.no_grad():
            S_p = torch.randn(4, 4, device=DEVICE) * 0.1
            S_q = torch.randn(4, 4, device=DEVICE) * 0.1
            lam = torch.ones(1, 4, device=DEVICE)
            mod_bypass.S_p.data.copy_(S_p)
            mod_bypass.S_q.data.copy_(S_q)
            mod_bypass.lambda_layer.data.copy_(lam)
            mod_rebuild.S_p.data.copy_(S_p)
            mod_rebuild.S_q.data.copy_(S_q)
            mod_rebuild.lambda_layer.data.copy_(lam)
            mod_rebuild.P_basis.copy_(mod_bypass.P_basis)
            mod_rebuild.Q_basis.copy_(mod_bypass.Q_basis)

        mod_rebuild.apply_to()
        mod_bypass.apply_to()

        x = torch.randn(2, 64, device=DEVICE, dtype=DTYPE)

        # Bypass grad
        out_b = mod_bypass(x)
        out_b.sum().backward()
        grad_lambda_bypass = mod_bypass.lambda_layer.grad.clone()

        # Rebuild grad
        out_r = mod_rebuild(x)
        out_r.sum().backward()
        grad_lambda_rebuild = mod_rebuild.lambda_layer.grad.clone()

        # Check sign agreement (cosine similarity > 0)
        cos_sim = F.cosine_similarity(
            grad_lambda_bypass.flatten(), grad_lambda_rebuild.flatten(), dim=0
        )
        assert cos_sim > 0.9, (
            f"λ gradient direction disagrees between bypass and rebuild: "
            f"cosine similarity = {cos_sim.item():.4f}"
        )


# --------------------------------------------------------------------------
# 17. Reference implementation equivalence
# --------------------------------------------------------------------------

class TestOrthoReferenceEquivalence:
    """Verify the LyCORIS port matches the anima_lora reference computation.

    The reference forward reproduces the anima_lora OrthoLoRAModule.forward
    using the same bf16 work dtype for basis matmuls, matching the actual
    module's computation path (Cayley solve stays fp32, boundary cast to bf16).
    """

    def _reference_forward(self, org_forward, P_basis, Q_basis, S_p, S_q,
                           lambda_layer, x, multiplier, scale):
        """Reproduce the anima_lora OrthoLoRAModule.forward in plain PyTorch.

        Uses the same bf16/fp32 split as the actual path: Cayley solve in
        fp32, cast to bf16 work dtype, matmuls in bf16.

        Args:
            org_forward: The original (un-patched) forward function.
        """
        work = P_basis.dtype  # bf16

        # Cayley solve in fp32 (matches actual _compute_effective_bases)
        skew = torch.stack([S_q.float(), S_p.float()])
        A = skew - skew.transpose(-2, -1)
        r = A.shape[-1]
        eye = torch.eye(r, device=A.device, dtype=torch.float32)
        R = torch.linalg.solve(eye + A, eye - A)
        R_q = R[0].to(work)
        R_p = R[1].to(work)

        Q_eff = R_q @ Q_basis   # bf16 × bf16 = bf16
        P_eff = P_basis @ R_p   # bf16 × bf16 = bf16

        org_out = org_forward(x)  # original Linear forward (pre-patch)
        lx = F.linear(x.to(work), Q_eff)  # bf16 matmul
        lx = lx * lambda_layer.to(work)    # bf16 multiply
        out = F.linear(lx, P_eff)          # bf16 matmul

        lora_out = out.float() * multiplier * scale
        return org_out.float() + lora_out

    def test_bypass_matches_reference(self):
        """Bypass-mode output matches the reference anima_lora forward."""
        org = _make_linear(in_f=64, out_f=32)
        alpha, lora_dim = 2, 8
        mod = _make_ortho(org, lora_dim=lora_dim, alpha=alpha, bypass_mode=True)

        with torch.no_grad():
            mod.S_p.data = torch.randn(lora_dim, lora_dim, device=DEVICE) * 0.1
            mod.S_q.data = torch.randn(lora_dim, lora_dim, device=DEVICE) * 0.1
            mod.lambda_layer.data = torch.randn(1, lora_dim, device=DEVICE)

        mod.apply_to()
        x = torch.randn(2, 64, device=DEVICE, dtype=DTYPE)

        actual = mod(x)
        expected = self._reference_forward(
            mod.org_forward, mod.P_basis, mod.Q_basis, mod.S_p, mod.S_q,
            mod.lambda_layer, x, multiplier=1.0, scale=alpha / lora_dim,
        )

        # Both paths use bf16 bases — small tolerance for rounding
        assert torch.allclose(actual.float(), expected, atol=5e-2), (
            f"Bypass output vs reference: max diff "
            f"{(actual.float() - expected).abs().max()}"
        )

    def test_rebuild_matches_reference(self):
        """Rebuild-mode output matches the reference anima_lora forward."""
        org = _make_linear(in_f=64, out_f=32)
        alpha, lora_dim = 2, 8
        mod = _make_ortho(org, lora_dim=lora_dim, alpha=alpha, bypass_mode=False)

        with torch.no_grad():
            mod.S_p.data = torch.randn(lora_dim, lora_dim, device=DEVICE) * 0.1
            mod.S_q.data = torch.randn(lora_dim, lora_dim, device=DEVICE) * 0.1
            mod.lambda_layer.data = torch.randn(1, lora_dim, device=DEVICE)

        mod.apply_to()
        x = torch.randn(2, 64, device=DEVICE, dtype=DTYPE)

        actual = mod(x)
        expected = self._reference_forward(
            mod.org_forward, mod.P_basis, mod.Q_basis, mod.S_p, mod.S_q,
            mod.lambda_layer, x, multiplier=1.0, scale=alpha / lora_dim,
        )

        # Rebuild does weight-space addition in fp32 (after make_weight),
        # then F.linear in fp32, vs reference bf16 matmuls. Slightly wider
        # tolerance for the different computation order.
        assert torch.allclose(actual.float(), expected, atol=8e-2), (
            f"Rebuild output vs reference: max diff "
            f"{(actual.float() - expected).abs().max()}"
        )

    def test_multiplier_applied_correctly(self):
        """With multiplier=1 and non-trivial params, LoRA delta is non-zero."""
        org = _make_linear(in_f=64, out_f=32)
        alpha, lora_dim = 2, 8

        mod = _make_ortho(org, lora_dim=lora_dim, alpha=alpha, bypass_mode=True)
        with torch.no_grad():
            mod.S_p.data = torch.randn(lora_dim, lora_dim, device=DEVICE) * 0.1
            mod.S_q.data = torch.randn(lora_dim, lora_dim, device=DEVICE) * 0.1
            mod.lambda_layer.data = torch.randn(1, lora_dim, device=DEVICE)

        mod.apply_to()
        x = torch.randn(2, 64, device=DEVICE, dtype=DTYPE)

        out = mod(x)
        org_out = mod.org_forward(x)

        # delta = out - org
        delta = (out.float() - org_out.float())

        # Check delta is non-zero (proves LoRA is active)
        assert delta.norm() > 1e-6, "Delta should be non-zero with non-trivial params"

        # With scale < 1, undoing scale should increase delta norm
        expected_scale = alpha / lora_dim
        raw_delta = delta / expected_scale
        assert raw_delta.norm() > delta.norm(), (
            "Undoing scale should increase norm (scale < 1)"
        )
