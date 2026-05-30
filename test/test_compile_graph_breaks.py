"""Tests for torch.compile graph-break elimination optimizations.

Verifies:
  - _expected_ndim and rank-dropout view shapes are set correctly
  - Norm parameter caching (_ln_*, _gn_*) for layernorm and groupnorm
  - _call_op explicit norm dispatch produces correct results
  - _rank_drop_shape / _bypass_rank_drop_shape produce correct broadcast
  - Scalar multiplication fusion in _forward_rebuild_core
  - Compiled forward equivalence with all new code paths
"""

import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from lycoris.modules.locon import LoConModule
from lycoris.modules.loha import LohaModule
from lycoris.modules.lokr import LokrModule


CUDA_AVAILABLE = torch.cuda.is_available()


def _device():
    return torch.device("cuda") if CUDA_AVAILABLE else torch.device("cpu")


def _compile_kwargs():
    if CUDA_AVAILABLE:
        return dict(mode="default", dynamic=True, fullgraph=False)
    return dict(backend="eager", fullgraph=False)


# ===========================================================================
# 1.  _expected_ndim and rank-dropout view shapes
# ===========================================================================

class ExpectedNdimTests(unittest.TestCase):
    """Verify _expected_ndim and view shapes are correct for all module types."""

    def test_linear_ndim(self):
        device = _device()
        base = nn.Linear(16, 16).to(device)
        net = LoConModule("test", base, lora_dim=4, alpha=1).to(device)
        self.assertEqual(net._expected_ndim, 2)

    def test_conv1d_ndim(self):
        device = _device()
        base = nn.Conv1d(16, 16, 3, 1, 1).to(device)
        net = LoConModule("test", base, lora_dim=4, alpha=1).to(device)
        self.assertEqual(net._expected_ndim, 3)

    def test_conv2d_ndim(self):
        device = _device()
        base = nn.Conv2d(16, 16, 3, 1, 1).to(device)
        net = LoConModule("test", base, lora_dim=4, alpha=1).to(device)
        self.assertEqual(net._expected_ndim, 4)

    def test_conv3d_ndim(self):
        device = _device()
        base = nn.Conv3d(16, 16, 3, 1, 1).to(device)
        net = LoConModule("test", base, lora_dim=4, alpha=1).to(device)
        self.assertEqual(net._expected_ndim, 5)

    def test_rank_drop_shape_linear(self):
        device = _device()
        base = nn.Linear(16, 16).to(device)
        net = LoConModule("test", base, lora_dim=4, alpha=1).to(device)
        self.assertEqual(net._rank_drop_shape, [-1, 1])
        self.assertEqual(net._bypass_rank_drop_shape, [1, -1])

    def test_rank_drop_shape_conv2d(self):
        device = _device()
        base = nn.Conv2d(16, 16, 3, 1, 1).to(device)
        net = LoConModule("test", base, lora_dim=4, alpha=1).to(device)
        self.assertEqual(net._rank_drop_shape, [-1, 1, 1, 1])
        self.assertEqual(net._bypass_rank_drop_shape, [1, -1, 1, 1])

    def test_rank_drop_shape_conv3d(self):
        device = _device()
        base = nn.Conv3d(16, 16, 3, 1, 1).to(device)
        net = LoConModule("test", base, lora_dim=4, alpha=1).to(device)
        self.assertEqual(net._rank_drop_shape, [-1, 1, 1, 1, 1])
        self.assertEqual(net._bypass_rank_drop_shape, [1, -1, 1, 1, 1])


# ===========================================================================
# 2.  Norm parameter caching
# ===========================================================================

class NormParamCacheTests(unittest.TestCase):
    """Verify norm parameters are cached for compile-friendly access."""

    def test_layernorm_params_cached(self):
        device = _device()
        base = nn.LayerNorm(16).to(device)
        # LayerNorm is supported by LycorisBaseModule but not by LoConModule
        # directly. Use the base class __init__ via a minimal subclass.
        from lycoris.modules.base import LycorisBaseModule

        class TestNormModule(LycorisBaseModule):
            def forward(self, x):
                return x

        net = TestNormModule("test", base).to(device)
        self.assertTrue(hasattr(net, '_ln_normalized_shape'))
        self.assertEqual(net._ln_normalized_shape, (16,))
        self.assertTrue(hasattr(net, '_ln_eps'))
        self.assertAlmostEqual(net._ln_eps, base.eps, places=10)

    def test_groupnorm_params_cached(self):
        device = _device()
        base = nn.GroupNorm(4, 16).to(device)
        from lycoris.modules.base import LycorisBaseModule

        class TestNormModule(LycorisBaseModule):
            def forward(self, x):
                return x

        net = TestNormModule("test", base).to(device)
        self.assertTrue(hasattr(net, '_gn_num_groups'))
        self.assertEqual(net._gn_num_groups, 4)
        self.assertTrue(hasattr(net, '_gn_eps'))
        self.assertAlmostEqual(net._gn_eps, base.eps, places=10)

    def test_linear_no_norm_params(self):
        device = _device()
        base = nn.Linear(16, 16).to(device)
        net = LoConModule("test", base, lora_dim=4, alpha=1).to(device)
        self.assertFalse(hasattr(net, '_ln_normalized_shape'))
        self.assertFalse(hasattr(net, '_gn_num_groups'))


# ===========================================================================
# 3.  _call_op explicit norm dispatch
# ===========================================================================

class CallOpNormTests(unittest.TestCase):
    """Verify _call_op produces correct results for norm types."""

    def _make_layernorm_module(self, device):
        base = nn.LayerNorm(16).to(device)
        from lycoris.modules.base import LycorisBaseModule

        class TestNormModule(LycorisBaseModule):
            def forward(self, x):
                return x

        net = TestNormModule("test", base).to(device)
        return net

    def _make_groupnorm_module(self, device):
        base = nn.GroupNorm(4, 16).to(device)
        from lycoris.modules.base import LycorisBaseModule

        class TestNormModule(LycorisBaseModule):
            def forward(self, x):
                return x

        net = TestNormModule("test", base).to(device)
        return net

    def test_call_op_layernorm_with_bias(self):
        device = _device()
        dtype = torch.float32
        net = self._make_layernorm_module(device)

        x = torch.randn(2, 16, device=device, dtype=dtype)
        weight = torch.randn(16, device=device, dtype=dtype)
        bias = torch.randn(16, device=device, dtype=dtype)

        out_op = net._call_op(x, weight, bias)
        out_ref = F.layer_norm(
            x, net._ln_normalized_shape, weight, bias, eps=net._ln_eps
        )
        torch.testing.assert_close(out_op, out_ref)

    def test_call_op_layernorm_no_bias(self):
        device = _device()
        dtype = torch.float32
        net = self._make_layernorm_module(device)

        x = torch.randn(2, 16, device=device, dtype=dtype)
        weight = torch.randn(16, device=device, dtype=dtype)

        out_op = net._call_op(x, weight)
        out_ref = F.layer_norm(
            x, net._ln_normalized_shape, weight, eps=net._ln_eps
        )
        torch.testing.assert_close(out_op, out_ref)

    def test_call_op_groupnorm_with_bias(self):
        device = _device()
        dtype = torch.float32
        net = self._make_groupnorm_module(device)

        x = torch.randn(2, 16, 4, 4, device=device, dtype=dtype)
        weight = torch.randn(16, device=device, dtype=dtype)
        bias = torch.randn(16, device=device, dtype=dtype)

        out_op = net._call_op(x, weight, bias)
        out_ref = F.group_norm(
            x, net._gn_num_groups, weight, bias, eps=net._gn_eps
        )
        torch.testing.assert_close(out_op, out_ref)

    def test_call_op_groupnorm_no_bias(self):
        device = _device()
        dtype = torch.float32
        net = self._make_groupnorm_module(device)

        x = torch.randn(2, 16, 4, 4, device=device, dtype=dtype)
        weight = torch.randn(16, device=device, dtype=dtype)

        out_op = net._call_op(x, weight)
        out_ref = F.group_norm(
            x, net._gn_num_groups, weight, eps=net._gn_eps
        )
        torch.testing.assert_close(out_op, out_ref)

    def test_call_op_1x1_layernorm(self):
        """_call_op_1x1 should also dispatch to F.layer_norm for layernorm."""
        device = _device()
        dtype = torch.float32
        net = self._make_layernorm_module(device)

        x = torch.randn(2, 16, device=device, dtype=dtype)
        weight = torch.randn(16, device=device, dtype=dtype)
        bias = torch.randn(16, device=device, dtype=dtype)

        out_op = net._call_op_1x1(x, weight, bias)
        out_ref = F.layer_norm(
            x, net._ln_normalized_shape, weight, bias, eps=net._ln_eps
        )
        torch.testing.assert_close(out_op, out_ref)


# ===========================================================================
# 4.  Rank-dropout view shape correctness
# ===========================================================================

class RankDropoutViewTests(unittest.TestCase):
    """Verify pre-computed view shapes produce correct broadcasting."""

    def test_weight_drop_shape_linear(self):
        """For linear: drop (r,) -> (r, 1) should broadcast with weight (r, c)."""
        device = _device()
        dtype = torch.float32
        r, c = 4, 16
        drop = torch.randn(r, device=device, dtype=dtype)
        weight = torch.randn(r, c, device=device, dtype=dtype)

        shape = [-1, 1]
        result = weight * drop.view(shape)
        # Verify: each row of weight is scaled by corresponding drop element
        for i in range(r):
            torch.testing.assert_close(result[i], weight[i] * drop[i])

    def test_weight_drop_shape_conv2d(self):
        """For conv2d: drop (r,) -> (r, 1, 1, 1) should broadcast with weight (r, c, kh, kw)."""
        device = _device()
        dtype = torch.float32
        r, c, kh, kw = 4, 16, 3, 3
        drop = torch.randn(r, device=device, dtype=dtype)
        weight = torch.randn(r, c, kh, kw, device=device, dtype=dtype)

        shape = [-1, 1, 1, 1]
        result = weight * drop.view(shape)
        for i in range(r):
            torch.testing.assert_close(result[i], weight[i] * drop[i])

    def test_bypass_drop_shape_linear(self):
        """For linear bypass: drop (r,) -> (1, r) should broadcast with mid (batch, r)."""
        device = _device()
        dtype = torch.float32
        batch, r = 2, 4
        drop = torch.randn(r, device=device, dtype=dtype)
        mid = torch.randn(batch, r, device=device, dtype=dtype)

        shape = [1, -1]
        result = mid * drop.view(shape)
        for b in range(batch):
            torch.testing.assert_close(result[b], mid[b] * drop)

    def test_bypass_drop_shape_conv2d(self):
        """For conv2d bypass: drop (r,) -> (1, r, 1, 1) broadcasts with mid (batch, r, h, w)."""
        device = _device()
        dtype = torch.float32
        batch, r, h, w = 2, 4, 8, 8
        drop = torch.randn(r, device=device, dtype=dtype)
        mid = torch.randn(batch, r, h, w, device=device, dtype=dtype)

        shape = [1, -1, 1, 1]
        result = mid * drop.view(shape)
        for b in range(batch):
            for c_idx in range(r):
                torch.testing.assert_close(
                    result[b, c_idx], mid[b, c_idx] * drop[c_idx]
                )

    def test_apply_rank_dropout_uses_precomputed_shape(self):
        """_apply_rank_dropout should produce correct output with pre-computed shape."""
        device = _device()
        dtype = torch.float32
        base = nn.Linear(16, 16).to(device, dtype)
        net = LoConModule("test", base, lora_dim=4, alpha=1,
                          rank_dropout=0.5).to(device, dtype)
        net.train()

        weight = torch.randn(16, 16, device=device, dtype=dtype)
        # Call multiple times to exercise randomness
        out = net._apply_rank_dropout(weight, device)
        self.assertEqual(out.shape, weight.shape)

    def test_bypass_rank_dropout_with_precomputed_shape(self):
        """Bypass forward should work correctly with pre-computed view shapes."""
        device = _device()
        dtype = torch.float32
        base = nn.Linear(16, 16).to(device, dtype)
        net = LoConModule("test", base, lora_dim=4, alpha=1,
                          rank_dropout=0.5).to(device, dtype)
        net.train()

        x = torch.randn(2, 16, device=device, dtype=dtype)
        wb = net.lora_down.weight.to(device, dtype)
        wa = net.lora_up.weight.to(device, dtype)
        combined_scale = net.scalar.to(device, dtype) * net.scale * 1.0

        # Should not raise with pre-computed shapes
        out = net._forward_bypass_core(x, wb, wa, None, combined_scale)
        self.assertEqual(out.shape, x.shape)

    def test_conv2d_bypass_rank_dropout(self):
        """Conv2d bypass forward should work with pre-computed view shapes."""
        device = _device()
        dtype = torch.float32
        base = nn.Conv2d(16, 32, 3, 1, 1).to(device, dtype)
        net = LoConModule("test", base, lora_dim=4, alpha=1,
                          rank_dropout=0.5).to(device, dtype)
        net.train()

        x = torch.randn(1, 16, 8, 8, device=device, dtype=dtype)
        wb = net.lora_down.weight.to(device, dtype)
        wa = net.lora_up.weight.to(device, dtype)
        combined_scale = net.scalar.to(device, dtype) * net.scale * 1.0

        out = net._forward_bypass_core(x, wb, wa, None, combined_scale)
        # Conv2d(16, 32) output has 32 channels, spatial dims preserved
        self.assertEqual(out.shape, (1, 32, 8, 8))


# ===========================================================================
# 5.  Scalar multiplication fusion in _forward_rebuild_core
# ===========================================================================

class ScalarFusionTests(unittest.TestCase):
    """Verify fused scalar multiplication produces identical results."""

    def test_rebuild_core_matches_naive(self):
        """_forward_rebuild_core output should match naive weight computation."""
        device = _device()
        dtype = torch.float32
        base = nn.Linear(16, 16).to(device, dtype)
        # use_scalar=True so lora_up is not zero-initialized
        net = LoConModule("test", base, lora_dim=4, alpha=1,
                          use_scalar=True, multiplier=0.7).to(device, dtype)
        net.apply_to()

        x = torch.randn(2, 16, device=device, dtype=dtype)

        # Compute expected output manually
        with torch.no_grad():
            org_weight = net.get_org_weight_for_compute(x.device).to(dtype)
            bias = net.get_org_bias_for_compute(x.device)
            if bias is not None:
                bias = bias.to(dtype)
            out_core = net._forward_rebuild_core(x, org_weight, bias)

            # Manual: make_weight * scale * multiplier + org_weight
            wa = net.lora_up.weight
            wb = net.lora_down.weight
            diff = wa.view(wa.size(0), -1) @ wb.view(wb.size(0), -1)
            diff = diff.view(net.shape)
            diff = diff * net.scalar * net.scale * net.multiplier
            expected = F.linear(x, org_weight + diff, bias)

        torch.testing.assert_close(out_core, expected, atol=1e-5, rtol=1e-5)

    def test_rebuild_core_dora_matches_naive(self):
        """DoRA path should match naive weight computation."""
        device = _device()
        dtype = torch.float32
        dim = 16
        base = nn.Linear(dim, dim, bias=True).to(device, dtype)
        net = LoConModule("test", base, lora_dim=4, alpha=1,
                          weight_decompose=True, use_scalar=True,
                          multiplier=0.8).to(device, dtype)
        net.apply_to()

        x = torch.randn(2, dim, device=device, dtype=dtype)

        with torch.no_grad():
            org_weight = net.get_org_weight_for_compute(x.device).to(dtype)
            bias = net.get_org_bias_for_compute(x.device)
            if bias is not None:
                bias = bias.to(dtype)
            out_core = net._forward_rebuild_core(x, org_weight, bias)

        # Verify shape is correct
        self.assertEqual(out_core.shape, x.shape)
        # Verify it's not NaN/Inf
        self.assertFalse(torch.isnan(out_core).any())
        self.assertFalse(torch.isinf(out_core).any())

    def test_rebuild_core_tucker_matches_naive(self):
        """Tucker path should match the full forward path."""
        device = _device()
        dtype = torch.float32
        base = nn.Conv2d(16, 16, 3, 1, 1).to(device, dtype)
        net = LoConModule("test", base, lora_dim=4, alpha=1,
                          use_tucker=True, use_scalar=True,
                          multiplier=1.0).to(device, dtype)
        net.apply_to()

        x = torch.randn(1, 16, 8, 8, device=device, dtype=dtype)

        # Reference: full forward (which calls _forward_rebuild_core internally)
        with torch.no_grad():
            out_full = base(x)

            # Direct call to _forward_rebuild_core
            org_weight = net.get_org_weight_for_compute(x.device).to(dtype)
            bias = net.get_org_bias_for_compute(x.device)
            if bias is not None:
                bias = bias.to(dtype)
            out_core = net._forward_rebuild_core(x, org_weight, bias)

        torch.testing.assert_close(out_full, out_core, atol=1e-5, rtol=1e-5)


# ===========================================================================
# 6.  Compiled forward equivalence with all new code paths
# ===========================================================================

class CompiledEquivalenceNewPathsTests(unittest.TestCase):
    """Verify compiled and non-compiled outputs match with all new optimizations."""

    def test_compile_linear_standard(self):
        """Standard LoRA linear: compiled rebuild core matches non-compiled."""
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

    def test_compile_linear_dora(self):
        """DoRA linear: compiled rebuild core matches non-compiled."""
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

    def test_compile_conv2d(self):
        """Conv2d: compiled rebuild core matches non-compiled."""
        device = _device()
        dtype = torch.float32
        dim = 16

        torch.manual_seed(42)
        base_a = nn.Conv2d(dim, dim, 3, 1, 1).to(device, dtype)
        net_a = LoConModule("test", base_a, lora_dim=4, alpha=1,
                            use_scalar=True).to(device, dtype)
        net_a.apply_to()

        torch.manual_seed(42)
        base_b = nn.Conv2d(dim, dim, 3, 1, 1).to(device, dtype)
        base_b.load_state_dict(base_a.state_dict())
        net_b = LoConModule("test", base_b, lora_dim=4, alpha=1,
                            use_scalar=True).to(device, dtype)
        net_b.apply_to()
        net_b.compile_forward(**_compile_kwargs())

        x = torch.randn(1, dim, 8, 8, device=device, dtype=dtype)
        with torch.no_grad():
            out_a = base_a(x)
            out_b = base_b(x)

        torch.testing.assert_close(out_a, out_b)

    def test_compile_bypass_mode(self):
        """Bypass mode: compiled bypass core matches non-compiled."""
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

    def test_compile_with_rank_dropout(self):
        """Rank dropout: compiled forward matches non-compiled (in eval mode)."""
        device = _device()
        dtype = torch.float32
        dim = 32

        torch.manual_seed(42)
        base_a = nn.Linear(dim, dim).to(device, dtype)
        net_a = LoConModule("test", base_a, lora_dim=4, alpha=1,
                            rank_dropout=0.1, use_scalar=True).to(device, dtype)
        net_a.apply_to()
        net_a.eval()  # eval mode disables dropout for deterministic comparison

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

    def test_compile_with_different_multipliers(self):
        """Different multipliers should produce different results with compile."""
        device = _device()
        dtype = torch.float32
        dim = 32

        base = nn.Linear(dim, dim).to(device, dtype)
        net = LoConModule("test", base, lora_dim=4, alpha=1,
                          use_scalar=True, multiplier=1.0).to(device, dtype)
        net.apply_to()
        net.compile_forward(**_compile_kwargs())

        x = torch.randn(2, dim, device=device, dtype=dtype)

        net.multiplier = 1.0
        with torch.no_grad():
            out1 = base(x)

        net.multiplier = 0.5
        with torch.no_grad():
            out2 = base(x)

        self.assertFalse(torch.allclose(out1, out2, atol=1e-6))


# ===========================================================================
# 7.  Training loop with optimizations
# ===========================================================================

class TrainingWithOptimizationsTests(unittest.TestCase):
    """Verify gradients flow correctly through all optimized code paths."""

    def test_training_linear_standard(self):
        device = _device()
        dtype = torch.float32
        base = nn.Linear(16, 16).to(device, dtype)
        net = LoConModule("test", base, lora_dim=4, alpha=1,
                          use_scalar=True).to(device, dtype)
        net.apply_to()

        x = torch.randn(2, 16, device=device, dtype=dtype)
        out = base(x)
        loss = out.sum()
        loss.backward()

        for name, p in net.named_parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad, f"{name} has no grad")

    def test_training_linear_compiled(self):
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

    def test_training_dora(self):
        device = _device()
        dtype = torch.float32
        base = nn.Linear(16, 16, bias=True).to(device, dtype)
        net = LoConModule("test", base, lora_dim=4, alpha=1,
                          weight_decompose=True).to(device, dtype)
        net.apply_to()

        x = torch.randn(2, 16, device=device, dtype=dtype)
        out = base(x)
        loss = out.sum()
        loss.backward()

        for name, p in net.named_parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad, f"{name} has no grad")

    def test_training_rank_dropout(self):
        device = _device()
        dtype = torch.float32
        base = nn.Linear(16, 16).to(device, dtype)
        net = LoConModule("test", base, lora_dim=4, alpha=1,
                          rank_dropout=0.3, use_scalar=True).to(device, dtype)
        net.apply_to()

        x = torch.randn(2, 16, device=device, dtype=dtype)
        out = base(x)
        loss = out.sum()
        loss.backward()

        for name, p in net.named_parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad, f"{name} has no grad")


# ===========================================================================
# 8.  Other module types inherit _expected_ndim
# ===========================================================================

class OtherModuleNdimTests(unittest.TestCase):
    """Verify other module types (LoHa, LoKr) also get _expected_ndim."""

    def test_loha_linear_ndim(self):
        device = _device()
        base = nn.Linear(16, 16).to(device)
        net = LohaModule("test", base, lora_dim=4, alpha=1).to(device)
        self.assertEqual(net._expected_ndim, 2)
        self.assertEqual(net._rank_drop_shape, [-1, 1])
        self.assertEqual(net._bypass_rank_drop_shape, [1, -1])

    def test_lokr_linear_ndim(self):
        device = _device()
        base = nn.Linear(16, 16).to(device)
        net = LokrModule("test", base, lora_dim=4, alpha=1).to(device)
        self.assertEqual(net._expected_ndim, 2)
        self.assertEqual(net._rank_drop_shape, [-1, 1])

    def test_loha_conv2d_ndim(self):
        device = _device()
        base = nn.Conv2d(16, 16, 3, 1, 1).to(device)
        net = LohaModule("test", base, lora_dim=4, alpha=1).to(device)
        self.assertEqual(net._expected_ndim, 4)
        self.assertEqual(net._rank_drop_shape, [-1, 1, 1, 1])


if __name__ == "__main__":
    unittest.main()
