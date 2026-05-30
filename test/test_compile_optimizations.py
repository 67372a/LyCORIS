"""Tests for torch.compile optimizations in base.py and locon.py.

Verifies:
  - **kw_dict fallback removed from _call_op and _call_op_1x1 (raises NotImplementedError)
  - _scale_in_diff cached in __init__ and used correctly in _forward_rebuild_core
  - Compiled forward equivalence with all code paths still holds
"""

import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from lycoris.modules.locon import LoConModule
from lycoris.modules.loha import LohaModule


CUDA_AVAILABLE = torch.cuda.is_available()


def _device():
    return torch.device("cuda") if CUDA_AVAILABLE else torch.device("cpu")


def _compile_kwargs():
    if CUDA_AVAILABLE:
        return dict(mode="default", dynamic=True, fullgraph=False)
    return dict(backend="eager", fullgraph=False)


# ===========================================================================
# 1.  _call_op / _call_op_1x1 — **kw_dict fallback removed
# ===========================================================================

class CallOpFallbackTests(unittest.TestCase):
    """Verify that _call_op and _call_op_1x1 raise NotImplementedError
    for unsupported module types instead of using **kw_dict fallback."""

    def test_call_op_raises_for_unsupported_type(self):
        """_call_op should raise NotImplementedError for unknown module_type."""
        device = _device()
        base = nn.Linear(16, 16).to(device)
        net = LoConModule("test", base, lora_dim=4, alpha=1).to(device)

        # Override module_type to an unsupported value
        net.module_type = "unsupported_type"

        x = torch.randn(2, 16, device=device)
        weight = torch.randn(16, 16, device=device)

        with self.assertRaises(NotImplementedError) as ctx:
            net._call_op(x, weight)
        self.assertIn("unsupported_type", str(ctx.exception))
        self.assertIn("_call_op", str(ctx.exception))

    def test_call_op_raises_for_unsupported_type_with_bias(self):
        """_call_op should raise NotImplementedError even when bias is provided."""
        device = _device()
        base = nn.Linear(16, 16).to(device)
        net = LoConModule("test", base, lora_dim=4, alpha=1).to(device)

        net.module_type = "unsupported_type"

        x = torch.randn(2, 16, device=device)
        weight = torch.randn(16, 16, device=device)
        bias = torch.randn(16, device=device)

        with self.assertRaises(NotImplementedError):
            net._call_op(x, weight, bias)

    def test_call_op_1x1_raises_for_unsupported_type(self):
        """_call_op_1x1 should raise NotImplementedError for unknown module_type."""
        device = _device()
        base = nn.Linear(16, 16).to(device)
        net = LoConModule("test", base, lora_dim=4, alpha=1).to(device)

        net.module_type = "unsupported_type"

        x = torch.randn(2, 16, device=device)
        weight = torch.randn(16, 16, device=device)

        with self.assertRaises(NotImplementedError) as ctx:
            net._call_op_1x1(x, weight)
        self.assertIn("unsupported_type", str(ctx.exception))
        self.assertIn("_call_op_1x1", str(ctx.exception))

    def test_call_op_linear_still_works(self):
        """_call_op should still produce correct results for linear."""
        device = _device()
        dtype = torch.float32
        base = nn.Linear(16, 16).to(device, dtype)
        net = LoConModule("test", base, lora_dim=4, alpha=1).to(device, dtype)

        x = torch.randn(2, 16, device=device, dtype=dtype)
        weight = torch.randn(16, 16, device=device, dtype=dtype)
        bias = torch.randn(16, device=device, dtype=dtype)

        out = net._call_op(x, weight, bias)
        expected = F.linear(x, weight, bias)
        torch.testing.assert_close(out, expected)

    def test_call_op_conv2d_still_works(self):
        """_call_op should still produce correct results for conv2d."""
        device = _device()
        dtype = torch.float32
        base = nn.Conv2d(16, 16, 3, 1, 1).to(device, dtype)
        net = LoConModule("test", base, lora_dim=4, alpha=1).to(device, dtype)

        x = torch.randn(1, 16, 8, 8, device=device, dtype=dtype)
        weight = torch.randn(16, 16, 3, 3, device=device, dtype=dtype)

        out = net._call_op(x, weight)
        expected = F.conv2d(x, weight, stride=net._conv_stride,
                            padding=net._conv_padding,
                            dilation=net._conv_dilation,
                            groups=net._conv_groups)
        torch.testing.assert_close(out, expected)

    def test_call_op_1x1_linear_still_works(self):
        """_call_op_1x1 should still produce correct results for linear."""
        device = _device()
        dtype = torch.float32
        base = nn.Linear(16, 16).to(device, dtype)
        net = LoConModule("test", base, lora_dim=4, alpha=1).to(device, dtype)

        x = torch.randn(2, 16, device=device, dtype=dtype)
        weight = torch.randn(16, 16, device=device, dtype=dtype)

        out = net._call_op_1x1(x, weight)
        expected = F.linear(x, weight)
        torch.testing.assert_close(out, expected)

    def test_call_op_1x1_conv2d_still_works(self):
        """_call_op_1x1 should still produce correct results for conv2d (1x1)."""
        device = _device()
        dtype = torch.float32
        base = nn.Conv2d(16, 16, 3, 1, 1).to(device, dtype)
        net = LoConModule("test", base, lora_dim=4, alpha=1).to(device, dtype)

        x = torch.randn(1, 16, 8, 8, device=device, dtype=dtype)
        weight = torch.randn(16, 16, 1, 1, device=device, dtype=dtype)

        out = net._call_op_1x1(x, weight)
        expected = F.conv2d(x, weight)
        torch.testing.assert_close(out, expected)


# ===========================================================================
# 2.  _scale_in_diff cached attribute
# ===========================================================================

class ScaleInDiffCacheTests(unittest.TestCase):
    """Verify _scale_in_diff is cached correctly at init time."""

    def test_scale_in_diff_false_for_standard_lora(self):
        """Standard LoRA (no tucker, no rank_dropout, no wd): _scale_in_diff=False."""
        device = _device()
        base = nn.Linear(16, 16).to(device)
        net = LoConModule("test", base, lora_dim=4, alpha=1).to(device)
        self.assertFalse(net._scale_in_diff)

    def test_scale_in_diff_true_with_tucker(self):
        """Tucker mode: _scale_in_diff=True (tucker=True, wd=False)."""
        device = _device()
        base = nn.Conv2d(16, 16, 3, 1, 1).to(device)
        net = LoConModule("test", base, lora_dim=4, alpha=1,
                          use_tucker=True).to(device)
        self.assertTrue(net._scale_in_diff)

    def test_scale_in_diff_true_with_rank_dropout(self):
        """Rank dropout: _scale_in_diff=True."""
        device = _device()
        base = nn.Linear(16, 16).to(device)
        net = LoConModule("test", base, lora_dim=4, alpha=1,
                          rank_dropout=0.1).to(device)
        self.assertTrue(net._scale_in_diff)

    def test_scale_in_diff_false_with_wd(self):
        """Weight decompose (DoRA): _scale_in_diff=False even with rank_dropout."""
        device = _device()
        base = nn.Linear(16, 16).to(device)
        net = LoConModule("test", base, lora_dim=4, alpha=1,
                          weight_decompose=True, rank_dropout=0.1).to(device)
        self.assertFalse(net._scale_in_diff)

    def test_scale_in_diff_false_with_wd_and_tucker(self):
        """Weight decompose + tucker: _scale_in_diff=False (wd takes priority)."""
        device = _device()
        base = nn.Conv2d(16, 16, 3, 1, 1).to(device)
        net = LoConModule("test", base, lora_dim=4, alpha=1,
                          weight_decompose=True, use_tucker=True).to(device)
        self.assertFalse(net._scale_in_diff)

    def test_scale_in_diff_matches_manual_computation(self):
        """_scale_in_diff should equal the manual expression for all combos."""
        device = _device()
        for wd in [False, True]:
            for tucker in [False, True]:
                for rd in [0.0, 0.1]:
                    if tucker:
                        base = nn.Conv2d(16, 16, 3, 1, 1).to(device)
                    else:
                        base = nn.Linear(16, 16).to(device)
                    net = LoConModule(
                        "test", base, lora_dim=4, alpha=1,
                        weight_decompose=wd, use_tucker=tucker,
                        rank_dropout=rd,
                    ).to(device)
                    expected = (not wd) and (tucker or rd)
                    self.assertEqual(
                        net._scale_in_diff, expected,
                        f"Failed for wd={wd}, tucker={tucker}, rd={rd}: "
                        f"expected {expected}, got {net._scale_in_diff}"
                    )


# ===========================================================================
# 3.  Compiled forward equivalence with optimizations
# ===========================================================================

class CompiledEquivalenceTests(unittest.TestCase):
    """Verify compiled forward produces identical results after optimizations."""

    def test_compile_linear_standard(self):
        """Standard LoRA linear: compiled matches non-compiled."""
        device = _device()
        dtype = torch.float32
        dim = 32

        torch.manual_seed(42)
        base_a = nn.Linear(dim, dim).to(device, dtype)
        net_a = LoConModule("test", base_a, lora_dim=4, alpha=1,
                            use_scalar=True).to(device, dtype)
        net_a.apply_to()

        torch.manual_seed(42)
        base_b = nn.Linear(dim, dim).to(device, dtype)
        base_b.load_state_dict(base_a.state_dict())
        net_b = LoConModule("test", base_b, lora_dim=4, alpha=1,
                            use_scalar=True).to(device, dtype)
        net_b.apply_to()
        net_b.compile_forward(**_compile_kwargs())

        x = torch.randn(3, dim, device=device, dtype=dtype)
        with torch.no_grad():
            out_a = base_a(x)
            out_b = base_b(x)

        torch.testing.assert_close(out_a, out_b)

    def test_compile_bypass_mode(self):
        """Bypass mode: compiled matches non-compiled."""
        device = _device()
        dtype = torch.float32
        dim = 32

        torch.manual_seed(42)
        base_a = nn.Linear(dim, dim).to(device, dtype)
        net_a = LoConModule("test", base_a, lora_dim=4, alpha=1,
                            use_scalar=True, bypass_mode=True).to(device, dtype)
        net_a.apply_to()

        torch.manual_seed(42)
        base_b = nn.Linear(dim, dim).to(device, dtype)
        base_b.load_state_dict(base_a.state_dict())
        net_b = LoConModule("test", base_b, lora_dim=4, alpha=1,
                            use_scalar=True, bypass_mode=True).to(device, dtype)
        net_b.apply_to()
        net_b.compile_forward(**_compile_kwargs())

        x = torch.randn(3, dim, device=device, dtype=dtype)
        with torch.no_grad():
            out_a = base_a(x)
            out_b = base_b(x)

        torch.testing.assert_close(out_a, out_b)

    def test_compile_dora(self):
        """DoRA (weight_decompose): compiled matches non-compiled."""
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
        net_b.compile_forward(**_compile_kwargs())

        x = torch.randn(3, dim, device=device, dtype=dtype)
        with torch.no_grad():
            out_a = base_a(x)
            out_b = base_b(x)

        torch.testing.assert_close(out_a, out_b)

    def test_compile_tucker(self):
        """Tucker mode: compiled matches non-compiled."""
        device = _device()
        dtype = torch.float32
        dim = 16

        torch.manual_seed(42)
        base_a = nn.Conv2d(dim, dim, 3, 1, 1).to(device, dtype)
        net_a = LoConModule("test", base_a, lora_dim=4, alpha=1,
                            use_tucker=True, use_scalar=True).to(device, dtype)
        net_a.apply_to()

        torch.manual_seed(42)
        base_b = nn.Conv2d(dim, dim, 3, 1, 1).to(device, dtype)
        base_b.load_state_dict(base_a.state_dict())
        net_b = LoConModule("test", base_b, lora_dim=4, alpha=1,
                            use_tucker=True, use_scalar=True).to(device, dtype)
        net_b.apply_to()
        net_b.compile_forward(**_compile_kwargs())

        x = torch.randn(1, dim, 8, 8, device=device, dtype=dtype)
        with torch.no_grad():
            out_a = base_a(x)
            out_b = base_b(x)

        torch.testing.assert_close(out_a, out_b)

    def test_compile_rank_dropout_eval(self):
        """Rank dropout in eval mode: compiled matches non-compiled."""
        device = _device()
        dtype = torch.float32
        dim = 32

        torch.manual_seed(42)
        base_a = nn.Linear(dim, dim).to(device, dtype)
        net_a = LoConModule("test", base_a, lora_dim=4, alpha=1,
                            rank_dropout=0.1, use_scalar=True).to(device, dtype)
        net_a.apply_to()
        net_a.eval()

        torch.manual_seed(42)
        base_b = nn.Linear(dim, dim).to(device, dtype)
        base_b.load_state_dict(base_a.state_dict())
        net_b = LoConModule("test", base_b, lora_dim=4, alpha=1,
                            rank_dropout=0.1, use_scalar=True).to(device, dtype)
        net_b.apply_to()
        net_b.eval()
        net_b.compile_forward(**_compile_kwargs())

        x = torch.randn(3, dim, device=device, dtype=dtype)
        with torch.no_grad():
            out_a = base_a(x)
            out_b = base_b(x)

        torch.testing.assert_close(out_a, out_b)

    def test_training_with_compiled_backward(self):
        """Verify gradients flow correctly through compiled forward."""
        device = _device()
        dtype = torch.float32
        base = nn.Linear(16, 16).to(device, dtype)
        net = LoConModule("test", base, lora_dim=4, alpha=1,
                          use_scalar=True).to(device, dtype)
        net.compile_forward(**_compile_kwargs())
        net.apply_to()

        x = torch.randn(2, 16, device=device, dtype=dtype)
        out = base(x)
        loss = out.sum()
        loss.backward()

        for name, p in net.named_parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad, f"{name} has no grad")


if __name__ == "__main__":
    unittest.main()
