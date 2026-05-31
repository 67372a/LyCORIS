"""
Tests for T-LoRA timestep rank masking flag on LoConModule.

Ports the lightweight anima_lora T-LoRA approach: a training-time bottleneck
mask on standard LoRA that produces portable lora_up/lora_down checkpoints.

Test matrix:
 1. Default: use_timestep_mask=False → no buffer, forward unchanged
 2. use_timestep_mask=True → buffer registered, default all-ones → same output
 3. Zero mask → output equals org_forward(x) (ΔW=0)
 4. Mask position: applied between lora_down and lora_up
 5. set_timestep_mask + clear_timestep_mask round-trip
 6. Shared buffer: one mask tensor shared across all modules
 7. persistent=False: mask not in state_dict()
 8. Mask schedule: t=max_timestep → few ranks, t=0 → all ranks
 9. Bypass mode parity: mask applies in bypass path
10. Rebuild mode parity: mask applies in rebuild path (via make_weight)
11. Multi-module: setting mask updates all LoConModule instances
12. Conv2d + T-LoRA: mask applies correctly to conv rank dimension
13. Gradient flow with T-LoRA mask
"""

import math

import pytest
import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32


def _make_linear(in_f=64, out_f=32, device=DEVICE):
    mod = nn.Linear(in_f, out_f, bias=False).to(device)
    nn.init.kaiming_uniform_(mod.weight, a=math.sqrt(5))
    return mod


def _make_conv2d(in_c=8, out_c=16, k=3, device=DEVICE):
    mod = nn.Conv2d(in_c, out_c, k, padding=k // 2, bias=False).to(device)
    nn.init.kaiming_uniform_(mod.weight, a=math.sqrt(5))
    return mod


def _make_locon(org_module, lora_dim=4, alpha=None, use_timestep_mask=False, **kwargs):
    from lycoris.modules.locon import LoConModule

    if alpha is None:
        alpha = lora_dim  # scale = 1.0
    module = LoConModule(
        "test_layer",
        org_module,
        multiplier=1.0,
        lora_dim=lora_dim,
        alpha=alpha,
        use_timestep_mask=use_timestep_mask,
        **kwargs,
    )
    # Move to same device as org_module (LoConModule is not auto-moved)
    module = module.to(org_module.weight.device)
    return module


def _seed_lora_weights(mod, seed=42):
    """Set non-zero lora_up weights so the LoRA delta is non-trivial.

    By default lora_up is zero-init, making the delta zero regardless of
    lora_down or the mask.  This helper breaks the zero-init for testing.
    """
    with torch.no_grad():
        torch.manual_seed(seed)
        mod.lora_up.weight.data = torch.randn_like(mod.lora_up.weight) * 0.01


# --------------------------------------------------------------------------
# 1. Default: use_timestep_mask=False → no buffer, forward unchanged
# --------------------------------------------------------------------------

class TestTLoRADefaultOff:
    def test_no_buffer_when_disabled(self):
        org = _make_linear()
        mod = _make_locon(org, use_timestep_mask=False)
        assert not getattr(mod, "use_timestep_mask", False)

    def test_forward_adds_delta_when_disabled(self):
        """LoCon with non-zero lora_up should differ from org_forward."""
        org = _make_linear()
        x = torch.randn(2, 64, device=DEVICE, dtype=DTYPE)
        org_out = org(x).detach()

        mod = _make_locon(org, use_timestep_mask=False)
        _seed_lora_weights(mod)
        mod.apply_to()
        out = mod(x)
        assert not torch.allclose(out, org_out, atol=1e-6)


# --------------------------------------------------------------------------
# 2. use_timestep_mask=True → buffer registered, default all-ones → same output
# --------------------------------------------------------------------------

class TestTLoRAEnabled:
    def test_buffer_registered(self):
        org = _make_linear()
        mod = _make_locon(org, use_timestep_mask=True)
        assert hasattr(mod, "_timestep_mask")
        assert mod.use_timestep_mask is True
        assert mod._timestep_mask.shape == (1, 4)  # lora_dim=4
        assert torch.allclose(
            mod._timestep_mask,
            torch.ones(1, 4, device=mod._timestep_mask.device),
        )

    def test_output_matches_no_mask_when_all_ones(self):
        """With all-ones mask, T-LoRA output should be identical to no-mask."""
        org = _make_linear()
        x = torch.randn(2, 64, device=DEVICE, dtype=DTYPE)

        # Create two modules with identical weights
        mod_no_mask = _make_locon(org, use_timestep_mask=False)
        mod_with_mask = _make_locon(org, use_timestep_mask=True)
        # Copy weights so both have identical lora_down and lora_up
        mod_with_mask.lora_down.weight.data.copy_(mod_no_mask.lora_down.weight.data)
        mod_with_mask.lora_up.weight.data.copy_(mod_no_mask.lora_up.weight.data)
        _seed_lora_weights(mod_no_mask, seed=42)
        _seed_lora_weights(mod_with_mask, seed=42)

        mod_no_mask.apply_to()
        mod_with_mask.apply_to()

        out_no = mod_no_mask(x)
        out_yes = mod_with_mask(x)
        assert torch.allclose(out_no, out_yes, atol=1e-5), (
            f"Max diff: {(out_no - out_yes).abs().max().item()}"
        )


# --------------------------------------------------------------------------
# 3. Zero mask → output equals org_forward(x) (ΔW=0)
# --------------------------------------------------------------------------

class TestTLoRAZeroMask:
    def test_zero_mask_kills_delta(self):
        org = _make_linear()
        mod = _make_locon(org, use_timestep_mask=True)
        _seed_lora_weights(mod)
        mod.apply_to()

        x = torch.randn(2, 64, device=DEVICE, dtype=DTYPE)

        # Set mask to all zeros
        mod._timestep_mask.fill_(0.0)

        out = mod(x)
        org_out = mod.org_forward(x)
        assert torch.allclose(out, org_out, atol=1e-5), (
            f"Max diff: {(out - org_out).abs().max().item()}"
        )


# --------------------------------------------------------------------------
# 4. Mask position: applied between lora_down and lora_up
# --------------------------------------------------------------------------

class TestTLoRAMaskPosition:
    def test_partial_mask_selectively_zeros_rank(self):
        """Mask [1,0,0,0] should keep only rank-0, zeroing ranks 1-3."""
        org = _make_linear()
        mod = _make_locon(org, use_timestep_mask=True, alpha=4)
        _seed_lora_weights(mod)
        mod.apply_to()

        x = torch.randn(2, 64, device=DEVICE, dtype=DTYPE)

        # Full mask
        mod._timestep_mask.fill_(1.0)
        out_full = mod(x).detach().clone()

        # Partial mask: only rank 0
        mask = torch.zeros(1, 4, device=DEVICE)
        mask[0, 0] = 1.0
        mod._timestep_mask.copy_(mask)
        out_partial = mod(x).detach()

        # Should differ (partial mask kills some rank dims)
        assert not torch.allclose(out_full, out_partial, atol=1e-6)

        # And partial should NOT be zero (rank 0 is still active)
        org_out = mod.org_forward(x)
        assert not torch.allclose(out_partial, org_out, atol=1e-6)


# --------------------------------------------------------------------------
# 5. set_timestep_mask + clear_timestep_mask round-trip
# --------------------------------------------------------------------------

class TestTLoRASetClear:
    def test_set_and_clear_roundtrip(self):
        from lycoris.wrapper import LycorisNetwork

        org = _make_linear()
        # Use a wrapper module so LycorisNetwork can find named children
        model = nn.ModuleDict({"lin": org})
        network = LycorisNetwork(
            model,
            lora_dim=4,
            alpha=4,
            network_module="locon",
            use_timestep_mask=True,
        )
        network.apply_to()

        # Seed weights on all modules
        for lora in network.loras:
            _seed_lora_weights(lora)

        x = torch.randn(2, 64, device=DEVICE, dtype=DTYPE)

        # Initialize the shared buffer by calling set_timestep_mask once
        t = torch.tensor(0.0, device=DEVICE)
        network.set_timestep_mask(t, max_timestep=1.0)
        # At t=0 with min_rank=1: all ranks active → full delta

        out_clear = network.loras[0](x).detach()
        org_out = network.loras[0].org_forward(x)

        # At t=0 all ranks active → non-zero delta
        assert not torch.allclose(out_clear, org_out, atol=1e-4), (
            "Full mask should produce non-zero delta"
        )

        # Zero the shared mask directly
        network._shared_timestep_mask.fill_(0.0)
        out_zero = network.loras[0](x).detach()

        # Zero mask should kill delta
        assert torch.allclose(out_zero, org_out, atol=1e-4), (
            "Zero mask should kill delta"
        )

        # clear_timestep_mask restores all-ones
        network.clear_timestep_mask()
        out_clear2 = network.loras[0](x).detach()
        assert torch.allclose(out_clear, out_clear2, atol=1e-5), (
            "Clear should restore original output"
        )


# --------------------------------------------------------------------------
# 6. Shared buffer: one mask tensor shared across all modules
# --------------------------------------------------------------------------

class TestTLoRASharedBuffer:
    def test_shared_buffer_across_modules(self):
        from lycoris.wrapper import LycorisNetwork

        model = nn.ModuleDict({
            "lin1": _make_linear(64, 32),
            "lin2": _make_linear(64, 32),
        })
        network = LycorisNetwork(
            model,
            lora_dim=4,
            alpha=4,
            network_module="locon",
            use_timestep_mask=True,
        )
        network.apply_to()

        # After set_timestep_mask, all modules should share the same buffer
        t = torch.tensor(0.5, device=DEVICE)
        network.set_timestep_mask(t, max_timestep=1.0)

        masks = []
        for lora in network.loras:
            if getattr(lora, "use_timestep_mask", False):
                masks.append(lora._timestep_mask)

        assert len(masks) == 2
        # Both should be the exact same tensor (same data pointer)
        assert masks[0] is masks[1]


# --------------------------------------------------------------------------
# 7. persistent=False: mask not in state_dict()
# --------------------------------------------------------------------------

class TestTLoRANotInStateDict:
    def test_mask_excluded_from_state_dict(self):
        org = _make_linear()
        mod = _make_locon(org, use_timestep_mask=True)
        sd = mod.state_dict()
        for key in sd:
            assert "timestep_mask" not in key, f"Unexpected key: {key}"


# --------------------------------------------------------------------------
# 8. Mask schedule: t=max_timestep → few ranks, t=0 → all ranks
# --------------------------------------------------------------------------

class TestTLoRASchedule:
    def test_schedule_high_noise_few_ranks(self):
        from lycoris.wrapper import LycorisNetwork

        org = _make_linear()
        model = nn.ModuleDict({"lin": org})
        network = LycorisNetwork(
            model,
            lora_dim=4,
            alpha=4,
            network_module="locon",
            use_timestep_mask=True,
            tlora_min_rank=1,
            tlora_alpha=1.0,
        )
        network.apply_to()

        # At max timestep (high noise): should have few ranks (min_rank=1)
        t_max = torch.tensor(1.0, device=DEVICE)
        network.set_timestep_mask(t_max, max_timestep=1.0)
        mask_high = network.loras[0]._timestep_mask.clone()
        assert mask_high.sum() <= 1.5, f"Expected ≤1 rank active, got {mask_high.sum()}"

        # At t=0 (low noise): should have all ranks
        t_zero = torch.tensor(0.0, device=DEVICE)
        network.set_timestep_mask(t_zero, max_timestep=1.0)
        mask_low = network.loras[0]._timestep_mask.clone()
        assert mask_low.sum() == 4, f"Expected 4 ranks active, got {mask_low.sum()}"

    def test_compute_timestep_mask_static(self):
        from lycoris.wrapper import LycorisNetwork

        mask = LycorisNetwork.compute_timestep_mask(
            timestep=0.5, max_timestep=1.0, max_rank=8, min_rank=1, alpha=1.0
        )
        assert mask.shape == (1, 8)
        # At t=0.5, alpha=1.0: r = 0.5 * (8-1) + 1 = 4.5 → 4
        assert mask.sum() == 4.0


# --------------------------------------------------------------------------
# 9. Bypass mode parity: mask applies in bypass path
# --------------------------------------------------------------------------

class TestTLoRABypassMode:
    def test_bypass_zero_mask_kills_delta(self):
        org = _make_linear()
        mod = _make_locon(org, use_timestep_mask=True, bypass_mode=True)
        _seed_lora_weights(mod)
        mod.apply_to()

        x = torch.randn(2, 64, device=DEVICE, dtype=DTYPE)

        # Zero mask
        mod._timestep_mask.fill_(0.0)
        out = mod(x)
        org_out = mod.org_forward(x)
        assert torch.allclose(out, org_out, atol=1e-5)


# --------------------------------------------------------------------------
# 10. Rebuild mode parity: mask applies in rebuild path
# --------------------------------------------------------------------------

class TestTLoRARebuildMode:
    def test_rebuild_zero_mask_kills_delta(self):
        org = _make_linear()
        # bypass_mode=False → rebuild mode
        mod = _make_locon(org, use_timestep_mask=True, bypass_mode=False)
        _seed_lora_weights(mod)
        mod.apply_to()

        x = torch.randn(2, 64, device=DEVICE, dtype=DTYPE)

        # Zero mask
        mod._timestep_mask.fill_(0.0)
        out = mod(x)
        org_out = mod.org_forward(x)
        assert torch.allclose(out, org_out, atol=1e-5)


# --------------------------------------------------------------------------
# 11. Multi-module: setting mask updates all LoConModule instances
# --------------------------------------------------------------------------

class TestTLoRAMultiModule:
    def test_all_modules_see_mask_update(self):
        from lycoris.wrapper import LycorisNetwork

        model = nn.ModuleDict({
            "a": _make_linear(64, 32),
            "b": _make_linear(64, 32),
            "c": _make_linear(64, 32),
        })
        network = LycorisNetwork(
            model,
            lora_dim=4,
            alpha=4,
            network_module="locon",
            use_timestep_mask=True,
            tlora_min_rank=1,
            tlora_alpha=1.0,
        )
        network.apply_to()

        # Set a specific mask
        t = torch.tensor(0.75, device=DEVICE)
        network.set_timestep_mask(t, max_timestep=1.0)

        # All T-LoRA modules should see the same mask values
        for lora in network.loras:
            if getattr(lora, "use_timestep_mask", False):
                assert lora._timestep_mask.sum() == network._shared_timestep_mask.sum()


# --------------------------------------------------------------------------
# 12. Conv2d + T-LoRA: mask applies correctly
# --------------------------------------------------------------------------

class TestTLoRAConv2d:
    def test_conv2d_zero_mask_kills_delta(self):
        org = _make_conv2d()
        mod = _make_locon(org, lora_dim=4, alpha=4, use_timestep_mask=True)
        _seed_lora_weights(mod)
        mod.apply_to()

        x = torch.randn(1, 8, 16, 16, device=DEVICE, dtype=DTYPE)

        # Zero mask
        mod._timestep_mask.fill_(0.0)
        out = mod(x)
        org_out = mod.org_forward(x)
        assert torch.allclose(out, org_out, atol=1e-4)

    def test_conv2d_buffer_shape(self):
        org = _make_conv2d()
        mod = _make_locon(org, lora_dim=4, use_timestep_mask=True)
        assert mod._timestep_mask.shape == (1, 4)


# --------------------------------------------------------------------------
# 13. Gradient flow with T-LoRA mask
# --------------------------------------------------------------------------

class TestTLoRAGradient:
    def test_gradient_flows_through_mask(self):
        org = _make_linear()
        mod = _make_locon(org, use_timestep_mask=True)
        _seed_lora_weights(mod)  # Break zero-init so gradients are non-zero
        mod.apply_to()

        x = torch.randn(2, 64, device=DEVICE, dtype=DTYPE, requires_grad=True)

        # Partial mask
        mask = torch.zeros(1, 4, device=DEVICE)
        mask[0, 0] = 1.0
        mask[0, 1] = 1.0
        mod._timestep_mask.copy_(mask)

        out = mod(x)
        loss = out.sum()
        loss.backward()

        # lora_down and lora_up should have non-zero gradients
        assert mod.lora_down.weight.grad is not None, "lora_down has no grad"
        assert mod.lora_up.weight.grad is not None, "lora_up has no grad"
        assert mod.lora_down.weight.grad.abs().sum() > 0, "lora_down grad is zero"
        assert mod.lora_up.weight.grad.abs().sum() > 0, "lora_up grad is zero"
