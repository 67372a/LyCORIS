"""Tests for LoRA²: Adaptive Rank LoRA.

Tests cover:
1. lora2_utils.py: discretized_exponential, compute_effective_rank,
   compute_lambda_diag, compute_nu_target, rescaled_kaiming_std
2. lora2.py: LoRA2Module forward pass, serialization, loss computation,
   gradient flow through ν, bypass and rebuild modes
"""

import math
import unittest

import torch
import torch.nn as nn

from lycoris.modules.lora2_utils import (
    discretized_exponential,
    compute_effective_rank,
    compute_lambda_diag,
    compute_nu_target,
    rescaled_kaiming_std,
)
from lycoris.modules.lora2 import LoRA2Module


class TestDiscretizedExponential(unittest.TestCase):
    """Test f(x; ν) = exp(−ν·x) − exp(−ν·(x+1))."""

    def test_basic_values(self):
        """Verify f(x; ν) for known inputs."""
        nu = torch.tensor(1.0)
        # f(1; 1) = exp(-1) - exp(-2) ≈ 0.3679 - 0.1353 ≈ 0.2325
        result = discretized_exponential(torch.tensor(1.0), nu)
        expected = math.exp(-1) - math.exp(-2)
        self.assertAlmostEqual(result.item(), expected, places=5)

    def test_monotonically_decreasing(self):
        """f should decrease as x increases."""
        nu = torch.tensor(0.5)
        x = torch.arange(1, 20, dtype=torch.float32)
        f_vals = discretized_exponential(x, nu)
        for i in range(len(f_vals) - 1):
            self.assertGreater(f_vals[i].item(), f_vals[i + 1].item())

    def test_sum_approaches_one(self):
        """Σ_{j=0}^{∞} f(j; ν) should equal 1 (telescoping series)."""
        # f(x;ν) = exp(-νx) - exp(-ν(x+1)), summing from x=0: telescopes to 1
        nu = torch.tensor(0.1)
        x = torch.arange(0, 10000, dtype=torch.float32)
        f_vals = discretized_exponential(x, nu)
        self.assertAlmostEqual(f_vals.sum().item(), 1.0, places=4)

    def test_batch_computation(self):
        """Should work with batched inputs."""
        nu = torch.tensor(0.3)
        x = torch.arange(1, 10, dtype=torch.float32)
        result = discretized_exponential(x, nu)
        self.assertEqual(result.shape, (9,))
        self.assertTrue((result > 0).all())

    def test_large_nu_concentrates(self):
        """Large ν should give very small effective rank (mass concentrated early)."""
        nu = torch.tensor(5.0)
        d = compute_effective_rank(nu, 0.9, 512)
        # With ν=5, D = ⌈2.3026/5⌉ = 1
        self.assertLessEqual(d, 2)


class TestComputeEffectiveRank(unittest.TestCase):
    """Test D = ⌈−ln(0.1) / ν⌉."""

    def test_known_values(self):
        """Verify effective rank for known ν values."""
        # ν = -ln(0.1)/10 ≈ 0.2303 → D = 10
        nu_target = compute_nu_target(10, 0.9)
        d = compute_effective_rank(nu_target, 0.9, 512)
        self.assertEqual(d, 10)

    def test_small_nu_large_rank(self):
        """Small ν → large rank."""
        d = compute_effective_rank(torch.tensor(0.01), 0.9, 512)
        self.assertGreater(d, 100)

    def test_large_nu_small_rank(self):
        """Large ν → small rank."""
        d = compute_effective_rank(torch.tensor(10.0), 0.9, 512)
        self.assertLessEqual(d, 5)

    def test_clamped_to_max_rank(self):
        """Result should not exceed max_rank."""
        d = compute_effective_rank(torch.tensor(0.001), 0.9, 64)
        self.assertLessEqual(d, 64)

    def test_clamped_to_one(self):
        """Result should be at least 1."""
        d = compute_effective_rank(torch.tensor(100.0), 0.9, 512)
        self.assertGreaterEqual(d, 1)

    def test_negative_nu(self):
        """Negative ν should return max_rank."""
        d = compute_effective_rank(torch.tensor(-1.0), 0.9, 512)
        self.assertEqual(d, 512)


class TestComputeLambdaDiag(unittest.TestCase):
    """Test Λ diagonal computation."""

    def test_shape(self):
        """Λ should have shape (d,)."""
        nu = torch.tensor(0.3)
        lam = compute_lambda_diag(nu, 16)
        self.assertEqual(lam.shape, (16,))

    def test_monotonically_decreasing(self):
        """Λ values should decrease."""
        nu = torch.tensor(0.2)
        lam = compute_lambda_diag(nu, 32)
        for i in range(len(lam) - 1):
            self.assertGreater(lam[i].item(), lam[i + 1].item())

    def test_positive_values(self):
        """All Λ values should be positive."""
        nu = torch.tensor(0.1)
        lam = compute_lambda_diag(nu, 64)
        self.assertTrue((lam > 0).all())

    def test_differentiable(self):
        """Λ should be differentiable w.r.t. ν."""
        nu = torch.tensor(0.3, requires_grad=True)
        lam = compute_lambda_diag(nu, 16)
        loss = lam.sum()
        loss.backward()
        self.assertIsNotNone(nu.grad)
        self.assertFalse(torch.isnan(nu.grad).any())


class TestComputeNuTarget(unittest.TestCase):
    """Test ν_target computation."""

    def test_roundtrip_with_effective_rank(self):
        """compute_nu_target(r) should give ν such that compute_effective_rank(ν) ≈ r."""
        for r_target in [4, 8, 16, 32, 64, 128, 256]:
            nu = compute_nu_target(r_target, 0.9)
            d = compute_effective_rank(nu, 0.9, 512)
            # Should be exactly equal for integer targets
            self.assertEqual(d, r_target, f"r_target={r_target}: got D={d}")

    def test_positive_output(self):
        """ν_target should always be positive for positive r_target."""
        nu = compute_nu_target(32, 0.9)
        self.assertGreater(nu, 0)


class TestRescaledKaimingStd(unittest.TestCase):
    """Test rescaled Kaiming initialization standard deviation."""

    def test_positive_std(self):
        """Std should be positive."""
        nu = torch.tensor(0.3)
        std = rescaled_kaiming_std(nu, 16)
        self.assertGreater(std, 0)

    def test_finite(self):
        """Std should be finite."""
        nu = torch.tensor(0.1)
        std = rescaled_kaiming_std(nu, 64)
        self.assertTrue(math.isfinite(std))

    def test_decreases_with_rank(self):
        """Larger d → smaller std (more Λ terms in denominator)."""
        nu = torch.tensor(0.3)
        std_small = rescaled_kaiming_std(nu, 8)
        std_large = rescaled_kaiming_std(nu, 64)
        self.assertLess(std_large, std_small)


class TestLoRA2Module(unittest.TestCase):
    """Test LoRA2Module (extends LoConModule)."""

    def _make_linear_module(self, in_features=64, out_features=128):
        """Create a simple Linear module for testing."""
        return nn.Linear(in_features, out_features, bias=False)

    def _make_lora2(
        self,
        org_module=None,
        lora_dim=16,
        alpha=16,
        bypass_mode=False,
        **kwargs,
    ):
        """Create a LoRA2Module for testing."""
        if org_module is None:
            org_module = self._make_linear_module()
        module = LoRA2Module(
            lora_name="test_lora2",
            org_module=org_module,
            multiplier=1.0,
            lora_dim=lora_dim,
            alpha=alpha,
            bypass_mode=bypass_mode,
            **kwargs,
        )
        return module

    def tearDown(self):
        """Clear the LoRA² registry between tests."""
        LoRA2Module.reset_lora2_registry()

    def test_creation(self):
        """Module should be created without errors."""
        mod = self._make_lora2()
        self.assertIsInstance(mod, LoRA2Module)
        self.assertTrue(hasattr(mod, 'lora2_nu'))
        self.assertTrue(isinstance(mod.lora2_nu, nn.Parameter))

    def test_initial_rank(self):
        """Initial effective rank should match r_target."""
        mod = self._make_lora2(lora_dim=32)
        d = mod.compute_effective_rank()
        # r_target defaults to lora_dim
        self.assertEqual(d, 32)

    def test_forward_rebuild_mode(self):
        """Forward pass in rebuild mode should produce correct output shape."""
        org = self._make_linear_module(in_features=64, out_features=128)
        mod = self._make_lora2(org_module=org, bypass_mode=False)
        x = torch.randn(2, 64)
        out = mod(x)
        self.assertEqual(out.shape, (2, 128))

    def test_forward_bypass_mode(self):
        """Forward pass in bypass mode should produce correct output shape."""
        org = self._make_linear_module(in_features=64, out_features=128)
        mod = self._make_lora2(org_module=org, bypass_mode=True)
        x = torch.randn(2, 64)
        out = mod(x)
        self.assertEqual(out.shape, (2, 128))

    def test_gradient_flow_through_nu(self):
        """Gradients should flow through ν."""
        mod = self._make_lora2()
        x = torch.randn(2, 64)
        out = mod(x)
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(mod.lora2_nu.grad)

    def test_gradient_flow_through_weights(self):
        """Gradients should flow through lora_up and lora_down."""
        mod = self._make_lora2()
        x = torch.randn(2, 64)
        out = mod(x)
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(mod.lora_up.weight.grad)
        self.assertIsNotNone(mod.lora_down.weight.grad)

    def test_rank_reg_loss(self):
        """Rank regularization loss should be non-negative."""
        mod = self._make_lora2()
        reg_loss = mod.get_rank_reg_loss()
        self.assertGreaterEqual(reg_loss.item(), 0)

    def test_rank_reg_loss_zero_at_target(self):
        """L_reg should be zero when ν = ν_target."""
        mod = self._make_lora2()
        # Set ν to its target value
        mod.lora2_nu.data.fill_(mod.nu_target)
        reg_loss = mod.get_rank_reg_loss()
        self.assertAlmostEqual(reg_loss.item(), 0.0, places=6)

    def test_total_rank_reg_loss(self):
        """Static method should aggregate losses across all modules."""
        LoRA2Module.reset_lora2_registry()
        mod1 = self._make_lora2()
        mod2 = self._make_lora2(self._make_linear_module(32, 64))
        total = LoRA2Module.get_total_rank_reg_loss()
        expected = mod1.get_rank_reg_loss() + mod2.get_rank_reg_loss()
        self.assertAlmostEqual(total.item(), expected.item(), places=5)

    def test_make_weight_shape(self):
        """make_weight should return weight with correct shape."""
        mod = self._make_lora2()
        weight = mod.make_weight()
        self.assertEqual(weight.shape, (128, 64))

    def test_make_weight_changes_with_nu(self):
        """Changing ν should change the weight (when B is non-zero)."""
        mod = self._make_lora2()
        # B (lora_up) is zero-initialized, so weight is always 0 regardless of ν.
        # Initialize B with random values to test the Λ effect.
        with torch.no_grad():
            nn.init.normal_(mod.lora_up.weight, mean=0.0, std=0.1)
        w1 = mod.make_weight().detach().clone()
        # Change ν significantly
        mod.lora2_nu.data.fill_(1.0)
        w2 = mod.make_weight().detach().clone()
        self.assertFalse(torch.allclose(w1, w2))

    def test_custom_state_dict_truncates(self):
        """custom_state_dict should truncate to effective rank."""
        mod = self._make_lora2(lora_dim=64)
        # Set ν so effective rank is small
        nu_val = compute_nu_target(16, 0.9)
        mod.lora2_nu.data.fill_(nu_val)
        sd = mod.custom_state_dict()
        # lora_up should be truncated to rank 16
        self.assertEqual(sd["lora_up.weight"].shape[1], 16)
        self.assertEqual(sd["lora_down.weight"].shape[0], 16)

    def test_roundtrip_state_dict(self):
        """Saving and loading should preserve the module."""
        org = self._make_linear_module(in_features=64, out_features=128)
        mod = self._make_lora2(org_module=org, lora_dim=32)

        # Set specific ν value (use a power of 2 rank for clean roundtrip)
        nu_val = compute_nu_target(16, 0.9)
        mod.lora2_nu.data.fill_(nu_val)
        expected_rank = mod.compute_effective_rank()

        # Save
        sd = mod.custom_state_dict()

        # Reconstruct
        mod2 = LoRA2Module.make_module_from_state_dict(
            "test_lora2", org,
            sd["lora_up.weight"],
            sd["lora_down.weight"],
            sd["lora2_nu"],
            sd["alpha"],
            sd.get("dora_scale"),
        )

        # Check ν matches
        self.assertAlmostEqual(
            mod2.lora2_nu.item(), nu_val, places=5
        )

        # Check effective rank matches what the original module had
        self.assertEqual(mod2.compute_effective_rank(), expected_rank)

    def test_algo_check(self):
        """algo_check should identify LoRA² state dicts."""
        mod = self._make_lora2()
        sd = mod.custom_state_dict()
        # Prefix keys with module name
        prefixed = {f"test_lora2.{k}": v for k, v in sd.items()}
        self.assertTrue(LoRA2Module.algo_check(prefixed, "test_lora2"))

    def test_algo_check_negative(self):
        """algo_check should not match plain LoRA state dicts."""
        plain_sd = {"test.lora_up.weight": torch.randn(10, 4)}
        self.assertFalse(LoRA2Module.algo_check(plain_sd, "test"))

    def test_zero_init_b(self):
        """lora_up (B) should be zero-initialized (standard LoRA convention)."""
        mod = self._make_lora2()
        self.assertTrue(torch.all(mod.lora_up.weight == 0))

    def test_nu_is_learnable(self):
        """ν should have requires_grad=True."""
        mod = self._make_lora2()
        self.assertTrue(mod.lora2_nu.requires_grad)

    def test_adaptive_rank_forward_consistency(self):
        """Output should be consistent for same input across calls."""
        mod = self._make_lora2()
        mod.eval()
        x = torch.randn(2, 64)
        out1 = mod(x)
        out2 = mod(x)
        self.assertTrue(torch.allclose(out1, out2))

    def test_nu_changes_rank(self):
        """Changing ν should change effective rank."""
        mod = self._make_lora2(lora_dim=64)
        # Start at rank 64
        d1 = mod.compute_effective_rank()
        self.assertEqual(d1, 64)

        # Set large ν → small rank
        mod.lora2_nu.data.fill_(5.0)
        d2 = mod.compute_effective_rank()
        self.assertLess(d2, d1)


class TestLoRA2Conv(unittest.TestCase):
    """Test LoRA2Module with Conv2d layers."""

    def tearDown(self):
        LoRA2Module.reset_lora2_registry()

    def test_conv2d_forward(self):
        """Should work with Conv2d modules."""
        org = nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False)
        mod = LoRA2Module(
            lora_name="test_conv",
            org_module=org,
            lora_dim=8,
            alpha=8,
        )
        x = torch.randn(1, 16, 8, 8)
        out = mod(x)
        self.assertEqual(out.shape, (1, 32, 8, 8))

    def test_conv2d_bypass_forward(self):
        """Should work with Conv2d in bypass mode."""
        org = nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False)
        mod = LoRA2Module(
            lora_name="test_conv_bypass",
            org_module=org,
            lora_dim=8,
            alpha=8,
            bypass_mode=True,
        )
        x = torch.randn(1, 16, 8, 8)
        out = mod(x)
        self.assertEqual(out.shape, (1, 32, 8, 8))


if __name__ == "__main__":
    unittest.main()
