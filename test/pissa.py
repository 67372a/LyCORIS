"""
Unit tests for PiSSA (Principal Singular values and Singular vectors Adaptation)
integration into LyCORIS LoCon.

Tests cover:
- PiSSA exact SVD initialization and reconstruction fidelity
- PiSSA fast randomized SVD (various niter values)
- PiSSA→LoRA portable conversion and identity at initialization
- custom_state_dict with PiSSA conversion on save
- Backward compatibility with standard LoRA initialization
- pissa_utils standalone functions
- Forward pass correctness with PiSSA-initialized modules
"""
import unittest

import torch
import torch.nn as nn
from parameterized import parameterized

from lycoris.modules.locon import LoConModule
from lycoris.modules.pissa_utils import (
    pissa_svd,
    convert_pissa_to_lora,
    compute_svd_error_ratio,
)


class PiSSAModuleTests(unittest.TestCase):
    """Tests for PiSSA initialization and conversion on LoConModule."""

    def setUp(self):
        torch.manual_seed(42)

    # ------------------------------------------------------------------
    # Backward compatibility
    # ------------------------------------------------------------------
    def test_standard_lora_init_unaffected(self):
        """Standard LoRA (kaiming init) must still work unchanged."""
        linear = nn.Linear(64, 32, bias=False)
        linear.weight.data.normal_()
        w_orig = linear.weight.data.clone()

        module = LoConModule(
            "test", linear, multiplier=1.0, lora_dim=8, alpha=8
        )

        # Base weight must be untouched (standard LoRA freezes W)
        self.assertTrue(
            torch.allclose(linear.weight.data, w_orig),
            "Standard LoRA must not modify the base weight",
        )
        # Adapter weights must be initialized (not zero-shape)
        self.assertEqual(module.lora_up.weight.shape, (32, 8))
        self.assertEqual(module.lora_down.weight.shape, (8, 64))
        self.assertFalse(
            module.is_pissa, "Standard LoRA should not be marked as PiSSA"
        )

    # ------------------------------------------------------------------
    # PiSSA exact SVD
    # ------------------------------------------------------------------
    def test_pissa_exact_svd_reconstruction(self):
        """W_res + A @ B must equal the original weight W."""
        linear = nn.Linear(64, 32, bias=False)
        linear.weight.data.normal_()
        w_orig = linear.weight.data.clone()

        module = LoConModule(
            "test",
            linear,
            multiplier=1.0,
            lora_dim=4,
            alpha=4,
            svd_segment="top",
        )

        self.assertTrue(module.is_pissa, "Top SVD segment should mark PiSSA")

        w_res = linear.weight.data
        reconstructed = w_res + module.lora_up.weight @ module.lora_down.weight
        error = (w_orig - reconstructed).norm().item()
        rel_error = error / w_orig.norm().item()

        self.assertLess(rel_error, 1e-4,
                        f"Reconstruction error {rel_error:.2e} exceeds tolerance")

    def test_pissa_exact_svd_on_conv2d(self):
        """PiSSA exact SVD must work on Conv2d layers."""
        conv = nn.Conv2d(16, 16, 3, padding=1, bias=False)
        conv.weight.data.normal_()
        w_orig = conv.weight.data.clone()

        module = LoConModule(
            "test",
            conv,
            multiplier=1.0,
            lora_dim=4,
            alpha=4,
            svd_segment="top",
        )

        self.assertTrue(module.is_pissa)
        w_res = conv.weight.data
        # For conv, we compute diff via make_weight
        diff = module.make_weight(device=w_res.device)
        reconstructed = w_res + diff
        error = (w_orig - reconstructed).norm().item()
        rel_error = error / w_orig.norm().item()
        self.assertLess(rel_error, 1e-4,
                        f"Conv2d reconstruction error {rel_error:.2e} exceeds tolerance")

    def test_pissa_stores_init_weights(self):
        """PiSSA must store initial A, B for later LoRA conversion."""
        linear = nn.Linear(64, 32, bias=False)
        linear.weight.data.normal_()

        module = LoConModule(
            "test",
            linear,
            multiplier=1.0,
            lora_dim=4,
            alpha=4,
            svd_segment="top",
        )

        self.assertIsNotNone(module.pissa_A_init)
        self.assertIsNotNone(module.pissa_B_init)
        self.assertEqual(module.pissa_A_init.shape, (32, 4))
        self.assertEqual(module.pissa_B_init.shape, (4, 64))

        # Init weights must match current adapter weights at init time
        self.assertTrue(
            torch.allclose(module.pissa_A_init, module.lora_up.weight.data),
            "pissa_A_init must equal initial lora_up at creation",
        )
        self.assertTrue(
            torch.allclose(module.pissa_B_init, module.lora_down.weight.data),
            "pissa_B_init must equal initial lora_down at creation",
        )

    # ------------------------------------------------------------------
    # PiSSA fast randomized SVD
    # ------------------------------------------------------------------
    @parameterized.expand([(1,), (2,), (4,), (8,)])
    def test_pissa_fast_svd_reconstruction(self, niter):
        """Fast randomized SVD with varying niter must approximate W.

        Note: niter >= 16 can cause numerical overflow in power iterations
        for small matrices. The PiSSA paper recommends niter <= 8 for most
        practical use cases.
        """
        linear = nn.Linear(128, 64, bias=False)
        linear.weight.data.normal_()
        w_orig = linear.weight.data.clone()

        module = LoConModule(
            "test",
            linear,
            multiplier=1.0,
            lora_dim=8,
            alpha=8,
            svd_segment="top",
            pissa_niter=niter,
        )

        self.assertTrue(module.is_pissa)
        w_res = linear.weight.data
        reconstructed = w_res + module.lora_up.weight @ module.lora_down.weight
        error = (w_orig - reconstructed).norm().item()
        rel_error = error / w_orig.norm().item()

        # Fast SVD error decreases with more iterations
        # At niter=1: typically ~1e-3, at niter=8: ~1e-6
        max_err = {1: 5e-2, 2: 1e-2, 4: 5e-3, 8: 3e-3}
        self.assertLess(rel_error, max_err[niter],
                        f"Fast SVD niter={niter}: error {rel_error:.2e} > {max_err[niter]}")

    # ------------------------------------------------------------------
    # PiSSA → LoRA portable conversion
    # ------------------------------------------------------------------
    def test_pissa_to_lora_conversion_restores_weight(self):
        """After conversion, base weight must be restored to original."""
        linear = nn.Linear(64, 32, bias=False)
        linear.weight.data.normal_()
        w_orig = linear.weight.data.clone()

        module = LoConModule(
            "test",
            linear,
            multiplier=1.0,
            lora_dim=4,
            alpha=4,
            svd_segment="top",
        )

        pissa_A_init = module.pissa_A_init.clone()
        pissa_B_init = module.pissa_B_init.clone()
        w_res_before = linear.weight.data.clone()

        success = module.convert_pissa_to_lora()
        self.assertTrue(success, "Conversion must succeed for a PiSSA module")
        self.assertFalse(
            module.is_pissa, "Module must not be PiSSA after conversion"
        )
        self.assertIsNone(module.pissa_A_init)
        self.assertIsNone(module.pissa_B_init)

        # Weight restored: W = W_res + A_init @ B_init
        w_expected = w_res_before + pissa_A_init @ pissa_B_init
        self.assertTrue(
            torch.allclose(linear.weight.data, w_expected),
            "Base weight must be restored after PiSSA→LoRA conversion",
        )

    def test_pissa_to_lora_delta_zero_at_init(self):
        """At initialization, converted LoRA delta must be zero."""
        linear = nn.Linear(64, 32, bias=False)
        linear.weight.data.normal_()

        module = LoConModule(
            "test",
            linear,
            multiplier=1.0,
            lora_dim=4,
            alpha=4,
            svd_segment="top",
            pissa_convert=True,
        )
        module.convert_pissa_to_lora()

        # ΔW = [A | A0] @ [B | -B0] = A@B - A0@B0 = 0 at init
        delta = module.lora_up.weight @ module.lora_down.weight
        self.assertLess(delta.norm().item(), 1e-4,
                        f"Converted LoRA delta must be near-zero at init, got {delta.norm().item():.2e}")

    def test_pissa_to_lora_doubles_rank(self):
        """After conversion, rank must double from r to 2r."""
        linear = nn.Linear(64, 32, bias=False)
        linear.weight.data.normal_()
        r = 4

        module = LoConModule(
            "test",
            linear,
            multiplier=1.0,
            lora_dim=r,
            alpha=r,
            svd_segment="top",
        )
        module.convert_pissa_to_lora()

        self.assertEqual(module.lora_up.weight.shape, (32, 2 * r))
        self.assertEqual(module.lora_down.weight.shape, (2 * r, 64))
        self.assertEqual(module.lora_dim, 2 * r)

    def test_convert_non_pissa_noop(self):
        """Calling convert_pissa_to_lora on non-PiSSA module must return False."""
        linear = nn.Linear(64, 32, bias=False)
        linear.weight.data.normal_()

        module = LoConModule(
            "test", linear, multiplier=1.0, lora_dim=4, alpha=4
        )
        success = module.convert_pissa_to_lora()
        self.assertFalse(success, "convert on non-PiSSA must return False")

    # ------------------------------------------------------------------
    # custom_state_dict with PiSSA conversion
    # ------------------------------------------------------------------
    def test_custom_state_dict_pissa_converted(self):
        """custom_state_dict must optionally export PiSSA as doubled LoRA."""
        linear = nn.Linear(64, 32, bias=False)
        linear.weight.data.normal_()

        module = LoConModule(
            "test",
            linear,
            multiplier=1.0,
            lora_dim=4,
            alpha=4,
            svd_segment="top",
            pissa_convert=True,
        )
        sd = module.custom_state_dict()

        self.assertIn("pissa_converted", sd)
        self.assertEqual(sd["lora_up.weight"].shape, (32, 8))
        self.assertEqual(sd["lora_down.weight"].shape, (8, 64))

    def test_custom_state_dict_pissa_raw(self):
        """custom_state_dict with pissa_convert=False keeps raw A,B."""
        linear = nn.Linear(64, 32, bias=False)
        linear.weight.data.normal_()

        module = LoConModule(
            "test",
            linear,
            multiplier=1.0,
            lora_dim=4,
            alpha=4,
            svd_segment="top",
            pissa_convert=False,
        )
        sd = module.custom_state_dict()

        self.assertNotIn("pissa_converted", sd)
        self.assertEqual(sd["lora_up.weight"].shape, (32, 4))
        self.assertEqual(sd["lora_down.weight"].shape, (4, 64))

    # ------------------------------------------------------------------
    # Forward pass correctness
    # ------------------------------------------------------------------
    def test_pissa_forward_output_equals_original(self):
        """At init, PiSSA module output must equal original module output."""
        linear = nn.Linear(64, 32, bias=False)
        linear.weight.data.normal_()

        # Record original forward
        x = torch.randn(2, 64)
        with torch.no_grad():
            original_output = linear(x).clone()

        module = LoConModule(
            "test",
            linear,
            multiplier=1.0,
            lora_dim=4,
            alpha=4,
            svd_segment="top",
        )
        module.apply_to()

        with torch.no_grad():
            patched_output = linear(x).clone()

        self.assertTrue(
            torch.allclose(original_output, patched_output, atol=1e-5),
            "PiSSA-patched output must match original at initialization",
        )
        module.restore()

    def test_pissa_forward_gradient_flows(self):
        """Gradients must flow through PiSSA adapter parameters."""
        linear = nn.Linear(64, 32, bias=False)
        linear.weight.data.normal_()

        module = LoConModule(
            "test",
            linear,
            multiplier=1.0,
            lora_dim=4,
            alpha=4,
            svd_segment="top",
        )
        module.apply_to()

        x = torch.randn(2, 64)

        output = linear(x)
        loss = output.sum()
        loss.backward()

        self.assertIsNotNone(module.lora_up.weight.grad,
                             "lora_up must have gradient")
        self.assertIsNotNone(module.lora_down.weight.grad,
                             "lora_down must have gradient")
        # PiSSA adapter parameters must be trainable
        self.assertTrue(module.lora_up.weight.requires_grad)
        self.assertTrue(module.lora_down.weight.requires_grad)
        module.restore()

    # ------------------------------------------------------------------
    # pissa_utils standalone
    # ------------------------------------------------------------------
    def test_pissa_svd_standalone_reconstruction(self):
        """pissa_utils.pissa_svd must produce W = W_res + A @ B."""
        W = torch.randn(32, 64)
        for r in [1, 2, 4, 8]:
            A, B, W_res = pissa_svd(W, r=r, fast_niter=0)
            ratio = compute_svd_error_ratio(W, A, B, W_res)
            self.assertLess(ratio, 1e-4,
                            f"r={r}: reconstruction error {ratio:.2e} too high")

    def test_pissa_svd_standalone_shapes(self):
        """pissa_utils.pissa_svd must return correctly-shaped tensors."""
        W = torch.randn(48, 96)
        r = 7
        A, B, W_res = pissa_svd(W, r=r)
        self.assertEqual(A.shape, (48, r))
        self.assertEqual(B.shape, (r, 96))
        self.assertEqual(W_res.shape, (48, 96))

    def test_convert_pissa_to_lora_identity(self):
        """At init (A=A0, B=B0), converted LoRA delta must be zero."""
        W = torch.randn(32, 64)
        A, B, _ = pissa_svd(W, r=8, fast_niter=0)
        delta_A, delta_B = convert_pissa_to_lora(A, B, A, B)
        delta = delta_A @ delta_B
        self.assertLess(delta.norm().item(), 1e-4,
                        f"Identity conversion must yield zero delta, got {delta.norm().item():.2e}")

    def test_convert_pissa_to_lora_shapes(self):
        """Conversion must double the rank in the concatenated matrices."""
        W = torch.randn(32, 64)
        A, B, _ = pissa_svd(W, r=4, fast_niter=0)
        delta_A, delta_B = convert_pissa_to_lora(A, B, A, B)
        self.assertEqual(delta_A.shape, (32, 8))
        self.assertEqual(delta_B.shape, (8, 64))

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------
    def test_pissa_rank_too_large_falls_back(self):
        """When rank > min(m,n), SVD segment init should warn and skip."""
        linear = nn.Linear(16, 8, bias=False)
        linear.weight.data.normal_()
        w_orig = linear.weight.data.clone()

        # min(m,n) = 8, asking for rank 10
        module = LoConModule(
            "test",
            linear,
            multiplier=1.0,
            lora_dim=10,
            alpha=10,
            svd_segment="top",
        )
        # Should warn but not crash; weights should fall back to standard init
        self.assertFalse(module.is_pissa)

    def test_pissa_middle_segment_not_pissa(self):
        """Only 'top' segment is considered PiSSA for conversion purposes."""
        linear = nn.Linear(64, 32, bias=False)
        linear.weight.data.normal_()

        module = LoConModule(
            "test",
            linear,
            multiplier=1.0,
            lora_dim=4,
            alpha=4,
            svd_segment="middle",
        )
        self.assertFalse(module.is_pissa)
        self.assertIsNone(module.pissa_A_init)
        self.assertIsNone(module.pissa_B_init)

    def test_pissa_with_orthogonal_init_mutually_exclusive(self):
        """SVD segment init must override orthogonal_init (with warning)."""
        linear = nn.Linear(64, 32, bias=False)
        linear.weight.data.normal_()

        module = LoConModule(
            "test",
            linear,
            multiplier=1.0,
            lora_dim=4,
            alpha=4,
            svd_segment="top",
            orthogonal_init=True,
        )
        # Should not crash and should be PiSSA
        self.assertTrue(module.is_pissa)

    def test_pissa_load_weight_hook_tolerates_missing_buffers(self):
        """load_weight_hook must tolerate missing PiSSA init buffers.

        When loading a standard LoRA state dict (no pissa_A_init or pissa_B_init)
        into a PiSSA module, the load_weight_hook should handle the missing
        keys gracefully without errors.
        """
        linear = nn.Linear(64, 32, bias=False)
        linear.weight.data.normal_()

        # Create a PiSSA module
        module = LoConModule(
            "test",
            linear,
            multiplier=1.0,
            lora_dim=4,
            alpha=4,
            svd_segment="top",
        )

        # Build a state dict that only has lora weights (no PiSSA buffers)
        # This simulates loading a standard LoRA checkpoint
        partial_sd = {
            "lora_up.weight": module.lora_up.weight.data.clone(),
            "lora_down.weight": module.lora_down.weight.data.clone(),
            "alpha": module.alpha.clone(),
        }
        missing, unexpected = module.load_state_dict(partial_sd, strict=False)

        # The load_weight_hook should have removed pissa_* keys from missing
        self.assertTrue(all("pissa_" not in k for k in missing),
                       f"PiSSA keys should be tolerated but found in missing: {missing}")
        self.assertIsNotNone(module.pissa_A_init,
                           "pissa_A_init should still be set from init")
        self.assertIsNotNone(module.pissa_B_init,
                           "pissa_B_init should still be set from init")


if __name__ == "__main__":
    unittest.main()
