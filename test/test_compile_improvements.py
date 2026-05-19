"""Tests for torch.compile improvements in base.py and locon.py.

Verifies:
  - _cached_dtype stays in sync after .to()
  - _call_op produces correct results for linear and conv modules
  - Bias pre-resolution (no numel check in compiled graph)
  - _forward_bypass_core produces correct results
  - multiplier_buf tensor and property setter sync
  - Compiled forward equivalence with new code paths
"""

import unittest
import torch
import torch.nn as nn

from lycoris.modules.locon import LoConModule
from lycoris.modules.loha import LohaModule
from lycoris.modules.lokr import LokrModule
from lycoris.modules.abba import AbbaModule
from lycoris.modules.glora import GLoRAModule
from lycoris.modules.ia3 import IA3Module
from lycoris.modules.diag_oft import DiagOFTModule
from lycoris.modules.boft import ButterflyOFTModule


CUDA_AVAILABLE = torch.cuda.is_available()


def _device():
    return torch.device("cuda") if CUDA_AVAILABLE else torch.device("cpu")


# ===========================================================================
# 1.  Cached dtype sync
# ===========================================================================

class CachedDtypeTests(unittest.TestCase):
    """Verify _cached_dtype stays in sync with dtype_tensor after .to()."""

    def test_cached_dtype_matches_property(self):
        device = _device()
        base = nn.Linear(16, 16).to(device)
        net = LoConModule("test", base, lora_dim=4, alpha=1).to(device)
        self.assertEqual(net._cached_dtype, net.dtype_tensor.dtype)
        self.assertEqual(net._cached_dtype, net.dtype)

    def test_cached_dtype_after_to_bf16(self):
        if not CUDA_AVAILABLE:
            self.skipTest("CUDA required for bf16")
        base = nn.Linear(16, 16).to("cuda")
        net = LoConModule("test", base, lora_dim=4, alpha=1).to("cuda")
        net = net.to(torch.bfloat16)
        self.assertEqual(net._cached_dtype, torch.bfloat16)
        self.assertEqual(net._cached_dtype, net.dtype)

    def test_cached_dtype_after_to_fp64(self):
        device = _device()
        base = nn.Linear(16, 16).to(device)
        net = LoConModule("test", base, lora_dim=4, alpha=1).to(device)
        net = net.to(torch.float64)
        self.assertEqual(net._cached_dtype, torch.float64)
        self.assertEqual(net._cached_dtype, net.dtype)

    def test_cached_dtype_loha(self):
        device = _device()
        base = nn.Linear(16, 16).to(device)
        net = LohaModule("test", base, lora_dim=4, alpha=1).to(device)
        self.assertEqual(net._cached_dtype, net.dtype)

    def test_cached_dtype_conv2d(self):
        device = _device()
        base = nn.Conv2d(16, 16, 3, 1, 1).to(device)
        net = LoConModule("test", base, lora_dim=4, alpha=1).to(device)
        self.assertEqual(net._cached_dtype, net.dtype)


# ===========================================================================
# 2.  _call_op correctness
# ===========================================================================

class CallOpTests(unittest.TestCase):
    """Verify _call_op produces the same results as direct op calls."""

    def test_call_op_linear(self):
        device = _device()
        dtype = torch.float32
        base = nn.Linear(16, 16).to(device, dtype)
        net = LoConModule("test", base, lora_dim=4, alpha=1).to(device, dtype)
        net.apply_to()

        x = torch.randn(2, 16, device=device, dtype=dtype)
        w = torch.randn(16, 16, device=device, dtype=dtype)
        b = torch.randn(16, device=device, dtype=dtype)

        out_op = net._call_op(x, w, b)
        out_ref = torch.nn.functional.linear(x, w, b)
        torch.testing.assert_close(out_op, out_ref)

    def test_call_op_linear_no_bias(self):
        device = _device()
        dtype = torch.float32
        base = nn.Linear(16, 16).to(device, dtype)
        net = LoConModule("test", base, lora_dim=4, alpha=1).to(device, dtype)

        x = torch.randn(2, 16, device=device, dtype=dtype)
        w = torch.randn(16, 16, device=device, dtype=dtype)

        out_op = net._call_op(x, w)
        out_ref = torch.nn.functional.linear(x, w, None)
        torch.testing.assert_close(out_op, out_ref)

    def test_call_op_conv2d(self):
        device = _device()
        dtype = torch.float32
        base = nn.Conv2d(16, 16, 3, 1, 1).to(device, dtype)
        net = LoConModule("test", base, lora_dim=4, alpha=1).to(device, dtype)

        x = torch.randn(2, 16, 8, 8, device=device, dtype=dtype)
        w = torch.randn(16, 16, 3, 3, device=device, dtype=dtype)
        b = torch.randn(16, device=device, dtype=dtype)

        out_op = net._call_op(x, w, b)
        out_ref = torch.nn.functional.conv2d(
            x, w, b, stride=base.stride, padding=base.padding,
            dilation=base.dilation, groups=base.groups,
        )
        torch.testing.assert_close(out_op, out_ref)

    def test_call_op_conv1d(self):
        device = _device()
        dtype = torch.float32
        base = nn.Conv1d(16, 16, 3, 1, 1).to(device, dtype)
        net = LoConModule("test", base, lora_dim=4, alpha=1).to(device, dtype)

        x = torch.randn(2, 16, 8, device=device, dtype=dtype)
        w = torch.randn(16, 16, 3, device=device, dtype=dtype)

        out_op = net._call_op(x, w)
        out_ref = torch.nn.functional.conv1d(
            x, w, None, stride=base.stride, padding=base.padding,
        )
        torch.testing.assert_close(out_op, out_ref)

    def test_call_op_matches_kw_dict(self):
        """_call_op must produce identical output to the old **kw_dict pattern."""
        device = _device()
        dtype = torch.float32
        base = nn.Conv2d(16, 32, 3, 1, 1, bias=False).to(device, dtype)
        net = LoConModule("test", base, lora_dim=4, alpha=1).to(device, dtype)

        x = torch.randn(1, 16, 8, 8, device=device, dtype=dtype)
        # Use a weight with correct shape for conv2d (out=32, in=16, k=3)
        w = torch.randn(32, 16, 3, 3, device=device, dtype=dtype)

        out_new = net._call_op(x, w, None)
        out_old = net.op(x, w, None, **net.kw_dict)
        torch.testing.assert_close(out_new, out_old)


# ===========================================================================
# 3.  Bias pre-resolution
# ===========================================================================

class BiasPreResolutionTests(unittest.TestCase):
    """Verify bias is correctly pre-resolved in forward() (no numel check)."""

    def test_linear_with_bias(self):
        device = _device()
        dtype = torch.float32
        base = nn.Linear(16, 16, bias=True).to(device, dtype)
        net = LoConModule("test", base, lora_dim=4, alpha=1).to(device, dtype)
        net.apply_to()

        x = torch.randn(2, 16, device=device, dtype=dtype)
        out = base(x)
        self.assertEqual(out.shape, (2, 16))

    def test_linear_without_bias(self):
        device = _device()
        dtype = torch.float32
        base = nn.Linear(16, 16, bias=False).to(device, dtype)
        net = LoConModule("test", base, lora_dim=4, alpha=1).to(device, dtype)
        net.apply_to()

        x = torch.randn(2, 16, device=device, dtype=dtype)
        out = base(x)
        self.assertEqual(out.shape, (2, 16))

    def test_conv2d_with_bias(self):
        device = _device()
        dtype = torch.float32
        base = nn.Conv2d(16, 16, 3, 1, 1, bias=True).to(device, dtype)
        net = LoConModule("test", base, lora_dim=4, alpha=1).to(device, dtype)
        net.apply_to()

        x = torch.randn(1, 16, 8, 8, device=device, dtype=dtype)
        out = base(x)
        self.assertEqual(out.shape, (1, 16, 8, 8))

    def test_bias_none_handling(self):
        """Forward must work when org_bias is None (no bias on original module)."""
        device = _device()
        dtype = torch.float32
        base = nn.Linear(16, 16, bias=False).to(device, dtype)
        net = LoConModule("test", base, lora_dim=4, alpha=1).to(device, dtype)
        net.apply_to()

        x = torch.randn(2, 16, device=device, dtype=dtype)
        out = base(x)
        # No exception means bias=None was handled correctly
        self.assertEqual(out.shape, (2, 16))


# ===========================================================================
# 4.  Multiplier buffer tensor
# ===========================================================================

class MultiplierBufferTests(unittest.TestCase):
    """Verify multiplier_buf stays in sync and works as tensor."""

    def test_multiplier_buf_exists(self):
        device = _device()
        base = nn.Linear(16, 16).to(device)
        net = LoConModule("test", base, lora_dim=4, alpha=1, multiplier=0.8).to(device)
        self.assertIsInstance(net.multiplier_buf, torch.Tensor)
        self.assertAlmostEqual(net.multiplier_buf.item(), 0.8, places=5)

    def test_multiplier_property_getter(self):
        device = _device()
        base = nn.Linear(16, 16).to(device)
        net = LoConModule("test", base, lora_dim=4, alpha=1, multiplier=0.5).to(device)
        self.assertIsInstance(net.multiplier, float)
        self.assertAlmostEqual(net.multiplier, 0.5, places=5)

    def test_multiplier_property_setter_syncs_buf(self):
        device = _device()
        base = nn.Linear(16, 16).to(device)
        net = LoConModule("test", base, lora_dim=4, alpha=1).to(device)
        net.multiplier = 0.7
        self.assertAlmostEqual(net.multiplier, 0.7, places=5)
        self.assertAlmostEqual(net.multiplier_buf.item(), 0.7, places=5)

    def test_multiplier_buf_on_device(self):
        device = _device()
        base = nn.Linear(16, 16).to(device)
        net = LoConModule("test", base, lora_dim=4, alpha=1).to(device)
        self.assertEqual(net.multiplier_buf.device.type, device.type)

    def test_multiplier_forward_correctness(self):
        """Forward with different multipliers should produce different results."""
        device = _device()
        dtype = torch.float32
        base = nn.Linear(16, 16).to(device, dtype)
        # use_scalar=True so lora_up is not zero-initialized
        net = LoConModule("test", base, lora_dim=4, alpha=1, multiplier=1.0,
                          use_scalar=True).to(device, dtype)
        net.apply_to()

        x = torch.randn(2, 16, device=device, dtype=dtype)

        net.multiplier = 1.0
        out1 = base(x)

        net.multiplier = 0.5
        out2 = base(x)

        # Results should differ because multipliers differ
        self.assertFalse(torch.allclose(out1, out2, atol=1e-6))

    def test_multiplier_buf_is_buffer(self):
        """multiplier_buf should be a registered buffer (not a parameter)."""
        device = _device()
        base = nn.Linear(16, 16).to(device)
        net = LoConModule("test", base, lora_dim=4, alpha=1).to(device)
        self.assertIn("multiplier_buf", dict(net.named_buffers()))
        self.assertNotIn("multiplier_buf", dict(net.named_parameters()))


# ===========================================================================
# 5.  _forward_bypass_core (LoCon only)
# ===========================================================================

class BypassCoreTests(unittest.TestCase):
    """Verify _forward_bypass_core produces correct outputs for LoCon."""

    def test_bypass_core_linear(self):
        device = _device()
        dtype = torch.float32
        base = nn.Linear(16, 16).to(device, dtype)
        net = LoConModule("test", base, lora_dim=4, alpha=1).to(device, dtype)

        x = torch.randn(2, 16, device=device, dtype=dtype)
        down_w = net.lora_down.weight.to(device, dtype)
        up_w = net.lora_up.weight.to(device, dtype)
        scale = net.scalar * net.scale * 1.0

        out_core = net._forward_bypass_core(x, down_w, up_w, None, scale)

        # Compare with manual computation
        mid = torch.nn.functional.linear(x, down_w)
        up = torch.nn.functional.linear(mid, up_w)
        out_ref = up * scale

        torch.testing.assert_close(out_core, out_ref)

    def test_bypass_core_equivalence_with_bypass_forward_diff(self):
        """_forward_bypass_core should match _bypass_forward_diff_single output."""
        device = _device()
        dtype = torch.float32
        base = nn.Linear(16, 16).to(device, dtype)
        net = LoConModule("test", base, lora_dim=4, alpha=1).to(device, dtype)

        x = torch.randn(2, 16, device=device, dtype=dtype)
        down_w = net.lora_down.weight.to(device, dtype)
        up_w = net.lora_up.weight.to(device, dtype)
        scale_val = net.scalar * net.scale * 1.0

        out_core = net._forward_bypass_core(x, down_w, up_w, None, scale_val)

        # The bypass_forward_diff_single applies drop() at the end,
        # which is nn.Identity when dropout=0, so outputs should match.
        out_bypass = net._bypass_forward_diff_single(x, scale=1.0)
        torch.testing.assert_close(out_core, out_bypass)

    def test_bypass_core_method_exists(self):
        """_forward_bypass_core should exist on LoConModule."""
        self.assertTrue(hasattr(LoConModule, '_forward_bypass_core'))


# ===========================================================================
# 6.  Conv params cached on base
# ===========================================================================

class ConvParamsCacheTests(unittest.TestCase):
    """Verify _conv_stride etc. are set for conv modules."""

    def test_conv2d_params(self):
        device = _device()
        base = nn.Conv2d(16, 32, 3, stride=2, padding=1, dilation=1, groups=1).to(device)
        net = LoConModule("test", base, lora_dim=4, alpha=1).to(device)
        self.assertEqual(net._conv_stride, (2, 2))
        self.assertEqual(net._conv_padding, (1, 1))
        self.assertEqual(net._conv_dilation, (1, 1))
        self.assertEqual(net._conv_groups, 1)

    def test_conv1d_params(self):
        device = _device()
        base = nn.Conv1d(16, 32, 3, stride=1, padding=1).to(device)
        net = LoConModule("test", base, lora_dim=4, alpha=1).to(device)
        self.assertEqual(net._conv_stride, (1,))
        self.assertEqual(net._conv_padding, (1,))

    def test_linear_no_conv_params(self):
        device = _device()
        base = nn.Linear(16, 16).to(device)
        net = LoConModule("test", base, lora_dim=4, alpha=1).to(device)
        self.assertFalse(hasattr(net, '_conv_stride'))


# ===========================================================================
# 7.  Compile equivalence with improvements
# ===========================================================================

class CompileWithImprovementsTests(unittest.TestCase):
    """Verify compiled and non-compiled outputs match with new code paths."""

    def _compile_kwargs():
        if CUDA_AVAILABLE:
            return dict(mode="default", dynamic=True, fullgraph=False)
        return dict(backend="eager", fullgraph=False)

    def test_compile_linear_with_dora(self):
        device = _device()
        dtype = torch.float32
        dim = 32

        torch.manual_seed(42)
        base_a = nn.Linear(dim, dim, bias=True).to(device, dtype)
        net_a = LoConModule("test", base_a, lora_dim=4, alpha=1,
                            weight_decompose=True).to(device, dtype)
        net_a.apply_to()

        torch.manual_seed(42)
        base_b = nn.Linear(dim, dim, bias=True).to(device, dtype)
        base_b.load_state_dict(base_a.state_dict())
        net_b = LoConModule("test", base_b, lora_dim=4, alpha=1,
                            weight_decompose=True).to(device, dtype)
        net_b.apply_to()
        net_b.compile_forward(**CompileWithImprovementsTests._compile_kwargs())

        x = torch.randn(3, dim, device=device, dtype=dtype)
        with torch.no_grad():
            out_a = base_a(x)
            out_b = base_b(x)

        torch.testing.assert_close(out_a, out_b)

    def test_compile_linear_multiplied(self):
        """Compiled output should match with different multiplier values."""
        device = _device()
        dtype = torch.float32
        dim = 32

        for mult in [0.5, 1.0, 1.5]:
            torch.manual_seed(42)
            base_a = nn.Linear(dim, dim).to(device, dtype)
            net_a = LoConModule("test", base_a, lora_dim=4, alpha=1,
                                multiplier=mult).to(device, dtype)
            net_a.apply_to()

            torch.manual_seed(42)
            base_b = nn.Linear(dim, dim).to(device, dtype)
            base_b.load_state_dict(base_a.state_dict())
            net_b = LoConModule("test", base_b, lora_dim=4, alpha=1,
                                multiplier=mult).to(device, dtype)
            net_b.apply_to()
            net_b.compile_forward(**CompileWithImprovementsTests._compile_kwargs())

            x = torch.randn(3, dim, device=device, dtype=dtype)
            with torch.no_grad():
                out_a = base_a(x)
                out_b = base_b(x)

            torch.testing.assert_close(out_a, out_b, atol=1e-5, rtol=1e-5,
                                        msg=f"Mismatch at multiplier={mult}")

    def test_compile_conv2d(self):
        device = _device()
        dtype = torch.float32
        ch = 16

        torch.manual_seed(42)
        base_a = nn.Conv2d(ch, ch, 3, 1, 1).to(device, dtype)
        net_a = LoConModule("test", base_a, lora_dim=4, alpha=1).to(device, dtype)
        net_a.apply_to()

        torch.manual_seed(42)
        base_b = nn.Conv2d(ch, ch, 3, 1, 1).to(device, dtype)
        base_b.load_state_dict(base_a.state_dict())
        net_b = LoConModule("test", base_b, lora_dim=4, alpha=1).to(device, dtype)
        net_b.apply_to()
        net_b.compile_forward(**CompileWithImprovementsTests._compile_kwargs())

        x = torch.randn(1, ch, 8, 8, device=device, dtype=dtype)
        with torch.no_grad():
            out_a = base_a(x)
            out_b = base_b(x)

        torch.testing.assert_close(out_a, out_b)


if __name__ == "__main__":
    unittest.main()
