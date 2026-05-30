"""Tests for the optimized apply_weight_decompose (DoRA) in LoConModule.

Verifies that the torch.linalg.vector_norm + pre-computed dim tuple approach
produces numerically identical results to the original reshape-norm-reshape
implementation across:
  - wd_on_output=True and wd_on_output=False
  - Linear (2D) and Conv2d (4D) weight shapes
  - float32, float16, bfloat16 dtypes
  - Various multiplier values
  - Both the __init__ (dora_scale) and forward (apply_weight_decompose) paths
"""

import unittest
import torch
import torch.nn as nn
from itertools import product

from lycoris.modules.locon import LoConModule


def _apply_weight_decompose_reference(weight, dora_scale, dora_norm_dims, multiplier=1, wd_on_output=True):
    """Reference (original) implementation for comparison."""
    weight = weight.to(dora_scale.dtype)
    if wd_on_output:
        weight_norm = (
            weight.reshape(weight.shape[0], -1)
            .norm(dim=1)
            .reshape(weight.shape[0], *[1] * dora_norm_dims)
        ) + torch.finfo(weight.dtype).eps
    else:
        weight_norm = (
            weight.transpose(0, 1)
            .reshape(weight.shape[1], -1)
            .norm(dim=1, keepdim=True)
            .reshape(weight.shape[1], *[1] * dora_norm_dims)
            .transpose(0, 1)
        ) + torch.finfo(weight.dtype).eps

    scale = dora_scale.to(weight.device) / weight_norm
    scale = multiplier * (scale - 1) + 1
    return weight * scale


class TestDoRAOptimization(unittest.TestCase):
    """Test that optimized apply_weight_decompose matches reference."""

    def _create_module(self, base_module, wd_on_output, dtype, device="cuda"):
        """Create a LoConModule with weight_decompose enabled."""
        base = base_module.to(device, dtype)
        mod = LoConModule(
            "test",
            base,
            multiplier=1.0,
            lora_dim=4,
            alpha=1,
            weight_decompose=True,
            wd_on_output=wd_on_output,
        ).to(device, dtype)
        mod.apply_to()
        return mod

    # ------------------------------------------------------------------
    # Test: dora_scale initialization matches reference
    # ------------------------------------------------------------------

    def _test_dora_scale_init(self, base_fn, wd_on_output, dtype, device="cuda"):
        """Verify dora_scale is initialized identically by both approaches."""
        base, _ = base_fn(16)
        base = base.to(device, dtype)
        org_weight = base.weight.data.clone().float()
        ndim = org_weight.dim()

        # Reference
        if wd_on_output:
            ref_dora_scale = torch.norm(
                org_weight.reshape(org_weight.shape[0], -1), dim=1, keepdim=True
            ).reshape(org_weight.shape[0], *[1] * (ndim - 1))
        else:
            ref_dora_scale = (
                torch.norm(
                    org_weight.transpose(1, 0).reshape(org_weight.shape[1], -1),
                    dim=1, keepdim=True,
                )
                .reshape(org_weight.shape[1], *[1] * (ndim - 1))
                .transpose(1, 0)
            )

        # Optimized (via module init) — use the SAME base module
        mod = self._create_module(base, wd_on_output, dtype, device)

        torch.testing.assert_close(
            mod.dora_scale.data,
            ref_dora_scale.float(),
            rtol=1e-5,
            atol=1e-6,
            msg=f"dora_scale mismatch for wd_on_output={wd_on_output}, dtype={dtype}",
        )

    def test_dora_scale_init_linear_output(self):
        self._test_dora_scale_init(
            lambda dim: (nn.Linear(dim, dim), torch.randn(1, dim)),
            wd_on_output=True, dtype=torch.float32,
        )

    def test_dora_scale_init_linear_input(self):
        self._test_dora_scale_init(
            lambda dim: (nn.Linear(dim, dim), torch.randn(1, dim)),
            wd_on_output=False, dtype=torch.float32,
        )

    def test_dora_scale_init_conv2d_output(self):
        self._test_dora_scale_init(
            lambda dim: (nn.Conv2d(dim, dim, 3, 1, 1), torch.randn(1, dim, 8, 8)),
            wd_on_output=True, dtype=torch.float32,
        )

    def test_dora_scale_init_conv2d_input(self):
        self._test_dora_scale_init(
            lambda dim: (nn.Conv2d(dim, dim, 3, 1, 1), torch.randn(1, dim, 8, 8)),
            wd_on_output=False, dtype=torch.float32,
        )

    # ------------------------------------------------------------------
    # Test: apply_weight_decompose matches reference
    # ------------------------------------------------------------------

    def _test_apply_weight_decompose(
        self, base_fn, wd_on_output, dtype, multiplier=1.0, device="cuda"
    ):
        """Verify optimized apply_weight_decompose matches reference."""
        base, _ = base_fn(16)
        base = base.to(device, dtype)
        mod = self._create_module(base_fn(16)[0], wd_on_output, dtype, device)

        # Create a random merged weight (simulating org_weight + diff_weight)
        merged_weight = torch.randn_like(base.weight.data)
        if dtype != torch.float32:
            merged_weight = merged_weight.to(dtype)

        # Reference
        ndim = merged_weight.dim()
        dora_norm_dims = ndim - 1
        ref_result = _apply_weight_decompose_reference(
            merged_weight.clone(),
            mod.dora_scale.data,
            dora_norm_dims,
            multiplier=multiplier,
            wd_on_output=wd_on_output,
        )

        # Optimized
        opt_result = mod.apply_weight_decompose(merged_weight.clone(), multiplier)

        # bfloat16 has only 7 mantissa bits (vs fp16's 10); the different
        # reduction order in torch.linalg.vector_norm vs reshape().norm(dim=1)
        # can cause small numerical differences amplified by the division.
        # Measured max errors (single-run, 16×16 linear):
        #   bf16: abs=7.8e-3, rel=3.0e-2
        #   fp16: abs=1.5e-3, rel=3.9e-3
        if dtype == torch.bfloat16:
            rtol, atol = 5e-2, 1e-2
        elif dtype == torch.float16:
            rtol, atol = 1e-2, 3e-3
        else:
            rtol, atol = 1e-5, 1e-6

        torch.testing.assert_close(
            opt_result,
            ref_result,
            rtol=rtol,
            atol=atol,
            msg=(
                f"apply_weight_decompose mismatch for wd_on_output={wd_on_output}, "
                f"dtype={dtype}, multiplier={multiplier}"
            ),
        )

    def test_apply_wd_linear_f32_output_m1(self):
        self._test_apply_weight_decompose(
            lambda dim: (nn.Linear(dim, dim), torch.randn(1, dim)),
            wd_on_output=True, dtype=torch.float32, multiplier=1.0,
        )

    def test_apply_wd_linear_f32_input_m1(self):
        self._test_apply_weight_decompose(
            lambda dim: (nn.Linear(dim, dim), torch.randn(1, dim)),
            wd_on_output=False, dtype=torch.float32, multiplier=1.0,
        )

    def test_apply_wd_linear_f32_output_m05(self):
        self._test_apply_weight_decompose(
            lambda dim: (nn.Linear(dim, dim), torch.randn(1, dim)),
            wd_on_output=True, dtype=torch.float32, multiplier=0.5,
        )

    def test_apply_wd_linear_f32_input_m05(self):
        self._test_apply_weight_decompose(
            lambda dim: (nn.Linear(dim, dim), torch.randn(1, dim)),
            wd_on_output=False, dtype=torch.float32, multiplier=0.5,
        )

    def test_apply_wd_conv2d_f32_output_m1(self):
        self._test_apply_weight_decompose(
            lambda dim: (nn.Conv2d(dim, dim, 3, 1, 1), torch.randn(1, dim, 8, 8)),
            wd_on_output=True, dtype=torch.float32, multiplier=1.0,
        )

    def test_apply_wd_conv2d_f32_input_m1(self):
        self._test_apply_weight_decompose(
            lambda dim: (nn.Conv2d(dim, dim, 3, 1, 1), torch.randn(1, dim, 8, 8)),
            wd_on_output=False, dtype=torch.float32, multiplier=1.0,
        )

    def test_apply_wd_conv2d_f32_output_m03(self):
        self._test_apply_weight_decompose(
            lambda dim: (nn.Conv2d(dim, dim, 3, 1, 1), torch.randn(1, dim, 8, 8)),
            wd_on_output=True, dtype=torch.float32, multiplier=0.3,
        )

    def test_apply_wd_conv2d_f32_input_m03(self):
        self._test_apply_weight_decompose(
            lambda dim: (nn.Conv2d(dim, dim, 3, 1, 1), torch.randn(1, dim, 8, 8)),
            wd_on_output=False, dtype=torch.float32, multiplier=0.3,
        )

    # ------------------------------------------------------------------
    # Test: mixed precision (fp16/bf16 input, fp32 dora_scale)
    # ------------------------------------------------------------------

    def test_apply_wd_linear_f16_output(self):
        self._test_apply_weight_decompose(
            lambda dim: (nn.Linear(dim, dim), torch.randn(1, dim)),
            wd_on_output=True, dtype=torch.float16, multiplier=1.0,
        )

    def test_apply_wd_linear_bf16_output(self):
        self._test_apply_weight_decompose(
            lambda dim: (nn.Linear(dim, dim), torch.randn(1, dim)),
            wd_on_output=True, dtype=torch.bfloat16, multiplier=1.0,
        )

    def test_apply_wd_conv2d_f16_input(self):
        self._test_apply_weight_decompose(
            lambda dim: (nn.Conv2d(dim, dim, 3, 1, 1), torch.randn(1, dim, 8, 8)),
            wd_on_output=False, dtype=torch.float16, multiplier=1.0,
        )

    def test_apply_wd_conv2d_bf16_input(self):
        self._test_apply_weight_decompose(
            lambda dim: (nn.Conv2d(dim, dim, 3, 1, 1), torch.randn(1, dim, 8, 8)),
            wd_on_output=False, dtype=torch.bfloat16, multiplier=1.0,
        )

    # ------------------------------------------------------------------
    # Test: pre-computed attributes exist and are correct
    # ------------------------------------------------------------------

    def test_precomputed_attrs_linear(self):
        base = nn.Linear(16, 16).cuda()
        mod = LoConModule(
            "test", base, weight_decompose=True, wd_on_output=True,
        ).cuda()
        self.assertEqual(mod._dora_norm_dims, (1,))
        self.assertTrue(hasattr(mod, "_dora_eps"))
        self.assertAlmostEqual(mod._dora_eps.item(), torch.finfo(torch.float32).eps, places=20)

    def test_precomputed_attrs_conv2d_output(self):
        base = nn.Conv2d(16, 16, 3, 1, 1).cuda()
        mod = LoConModule(
            "test", base, weight_decompose=True, wd_on_output=True,
        ).cuda()
        self.assertEqual(mod._dora_norm_dims, (1, 2, 3))

    def test_precomputed_attrs_conv2d_input(self):
        base = nn.Conv2d(16, 16, 3, 1, 1).cuda()
        mod = LoConModule(
            "test", base, weight_decompose=True, wd_on_output=False,
        ).cuda()
        self.assertEqual(mod._dora_norm_dims, (0, 2, 3))

    def test_precomputed_attrs_conv1d_output(self):
        base = nn.Conv1d(16, 16, 3, 1, 1).cuda()
        mod = LoConModule(
            "test", base, weight_decompose=True, wd_on_output=True,
        ).cuda()
        self.assertEqual(mod._dora_norm_dims, (1, 2))

    def test_precomputed_attrs_conv3d_output(self):
        base = nn.Conv3d(16, 16, 3, 1, 1).cuda()
        mod = LoConModule(
            "test", base, weight_decompose=True, wd_on_output=True,
        ).cuda()
        self.assertEqual(mod._dora_norm_dims, (1, 2, 3, 4))

    # ------------------------------------------------------------------
    # Test: End-to-end forward pass with DoRA
    # ------------------------------------------------------------------

    def _test_forward_e2e(self, base_fn, wd_on_output, dtype, device="cuda"):
        """End-to-end forward pass should not crash and produce finite output."""
        base, test_input = base_fn(16)
        base = base.to(device, dtype)
        test_input = test_input.to(device, dtype)
        mod = LoConModule(
            "test",
            base,
            multiplier=1.0,
            lora_dim=4,
            alpha=1,
            weight_decompose=True,
            wd_on_output=wd_on_output,
        ).to(device, dtype)
        mod.apply_to()

        with torch.no_grad():
            output = mod(test_input)

        self.assertTrue(torch.isfinite(output).all(), "Output contains non-finite values")
        self.assertEqual(output.dtype, dtype, f"Output dtype {output.dtype} != expected {dtype}")

    def test_forward_e2e_linear_f32_output(self):
        self._test_forward_e2e(
            lambda dim: (nn.Linear(dim, dim), torch.randn(1, dim)),
            wd_on_output=True, dtype=torch.float32,
        )

    def test_forward_e2e_linear_f32_input(self):
        self._test_forward_e2e(
            lambda dim: (nn.Linear(dim, dim), torch.randn(1, dim)),
            wd_on_output=False, dtype=torch.float32,
        )

    def test_forward_e2e_conv2d_f32_output(self):
        self._test_forward_e2e(
            lambda dim: (nn.Conv2d(dim, dim, 3, 1, 1), torch.randn(1, dim, 8, 8)),
            wd_on_output=True, dtype=torch.float32,
        )

    def test_forward_e2e_conv2d_f32_input(self):
        self._test_forward_e2e(
            lambda dim: (nn.Conv2d(dim, dim, 3, 1, 1), torch.randn(1, dim, 8, 8)),
            wd_on_output=False, dtype=torch.float32,
        )

    def test_forward_e2e_linear_bf16_output(self):
        self._test_forward_e2e(
            lambda dim: (nn.Linear(dim, dim), torch.randn(1, dim)),
            wd_on_output=True, dtype=torch.bfloat16,
        )

    def test_forward_e2e_conv2d_bf16_input(self):
        self._test_forward_e2e(
            lambda dim: (nn.Conv2d(dim, dim, 3, 1, 1), torch.randn(1, dim, 8, 8)),
            wd_on_output=False, dtype=torch.bfloat16,
        )

    # ------------------------------------------------------------------
    # Test: get_merged_weight with DoRA
    # ------------------------------------------------------------------

    def test_get_merged_weight_linear(self):
        base = nn.Linear(16, 16).cuda()
        mod = LoConModule(
            "test", base, lora_dim=4, alpha=1, weight_decompose=True,
            wd_on_output=True,
        ).cuda()
        mod.apply_to()

        merged, _ = mod.get_merged_weight(multiplier=1.0)
        self.assertTrue(torch.isfinite(merged).all())
        self.assertEqual(merged.shape, base.weight.shape)

    def test_get_merged_weight_conv2d(self):
        base = nn.Conv2d(16, 16, 3, 1, 1).cuda()
        mod = LoConModule(
            "test", base, lora_dim=4, alpha=1, weight_decompose=True,
            wd_on_output=False,
        ).cuda()
        mod.apply_to()

        merged, _ = mod.get_merged_weight(multiplier=0.5)
        self.assertTrue(torch.isfinite(merged).all())
        self.assertEqual(merged.shape, base.weight.shape)

    # ------------------------------------------------------------------
    # Test: multiplier=0 should return identity (no DoRA scaling)
    # ------------------------------------------------------------------

    def test_multiplier_zero_is_identity(self):
        base = nn.Linear(16, 16).cuda()
        mod = LoConModule(
            "test", base, lora_dim=4, alpha=1, weight_decompose=True,
            wd_on_output=True,
        ).cuda()
        mod.apply_to()

        weight = torch.randn(16, 16, device="cuda", dtype=torch.float32)
        result = mod.apply_weight_decompose(weight.clone(), multiplier=0.0)

        # When multiplier=0, scale = 0*(scale-1)+1 = 1, so output = weight * 1
        torch.testing.assert_close(
            result, weight.to(torch.float32), rtol=1e-6, atol=1e-6,
        )

    # ------------------------------------------------------------------
    # Test: gradient flows through apply_weight_decompose
    # ------------------------------------------------------------------

    def test_gradient_flows(self):
        base = nn.Linear(16, 16).cuda()
        mod = LoConModule(
            "test", base, lora_dim=4, alpha=1, weight_decompose=True,
            wd_on_output=True,
        ).cuda()
        mod.apply_to()

        x = torch.randn(1, 16, device="cuda", requires_grad=True)
        output = mod(x)
        loss = output.sum()
        loss.backward()

        # dora_scale should have gradients
        self.assertIsNotNone(mod.dora_scale.grad)
        # lora weights should have gradients
        self.assertIsNotNone(mod.lora_up.weight.grad)
        self.assertIsNotNone(mod.lora_down.weight.grad)


if __name__ == "__main__":
    unittest.main()
