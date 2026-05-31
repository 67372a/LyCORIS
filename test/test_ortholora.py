"""
Tests for OrthoLoRA: Cayley-parameterized orthogonal low-rank adapter.
"""

import math

import pytest
import torch
import torch.nn as nn

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
        assert torch.allclose(reconstructed, delta_w, atol=1e-3), (
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
