"""Tests for compile-friendly optimizations in base.py, loha.py, and general.py.

Verifies:
  - Fix 1: _orthogonalize() uses _qr_ndim instead of len(shape)
  - Fix 2: LoHa get_weight() uses _rank_drop_shape instead of len(weight.shape[1:])
  - Fix 3: rebuild_tucker() produces identical results with tensordot vs einsum
  - Compiled forward equivalence for Tucker mode (LoCon + LoHa)
  - Compiled forward equivalence for orthogonalize mode (LoCon)
"""

import math
import unittest

import torch
import torch.nn as nn

from lycoris.modules.locon import LoConModule
from lycoris.modules.loha import LohaModule
from lycoris.functional.general import rebuild_tucker


CUDA_AVAILABLE = torch.cuda.is_available()


def _device():
    return torch.device("cuda") if CUDA_AVAILABLE else torch.device("cpu")


def _compile_kwargs():
    if CUDA_AVAILABLE:
        return dict(mode="default", dynamic=True, fullgraph=False)
    return dict(backend="eager", fullgraph=False)


# ===========================================================================
# 1.  rebuild_tucker() — tensordot vs einsum equivalence
# ===========================================================================

class RebuildTuckerTests(unittest.TestCase):
    """Verify the new tensordot-based rebuild_tucker matches the old einsum."""

    def _einsum_reference(self, t, wa, wb):
        """Old einsum-based implementation for comparison."""
        return torch.einsum("i j ..., i p, j r -> p r ...", t, wa, wb)

    def test_linear_tucker_2d(self):
        """2D (no kernel): t(i,j), wa(i,p), wb(j,r) → result(p,r)."""
        device = _device()
        t = torch.randn(4, 3, device=device)
        wa = torch.randn(4, 8, device=device)
        wb = torch.randn(3, 6, device=device)
        result = rebuild_tucker(t, wa, wb)
        expected = self._einsum_reference(t, wa, wb)
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)
        self.assertEqual(result.shape, (8, 6))

    def test_conv1d_tucker(self):
        """1D kernel: t(i,j,k), wa(i,p), wb(j,r) → result(p,r,k)."""
        device = _device()
        t = torch.randn(4, 3, 5, device=device)
        wa = torch.randn(4, 8, device=device)
        wb = torch.randn(3, 6, device=device)
        result = rebuild_tucker(t, wa, wb)
        expected = self._einsum_reference(t, wa, wb)
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)
        self.assertEqual(result.shape, (8, 6, 5))

    def test_conv2d_tucker(self):
        """2D kernel: t(i,j,k1,k2), wa(i,p), wb(j,r) → result(p,r,k1,k2)."""
        device = _device()
        t = torch.randn(4, 3, 5, 5, device=device)
        wa = torch.randn(4, 8, device=device)
        wb = torch.randn(3, 6, device=device)
        result = rebuild_tucker(t, wa, wb)
        expected = self._einsum_reference(t, wa, wb)
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)
        self.assertEqual(result.shape, (8, 6, 5, 5))

    def test_conv3d_tucker(self):
        """3D kernel: t(i,j,k1,k2,k3), wa(i,p), wb(j,r) → result(p,r,k1,k2,k3)."""
        device = _device()
        t = torch.randn(4, 3, 3, 3, 3, device=device)
        wa = torch.randn(4, 8, device=device)
        wb = torch.randn(3, 6, device=device)
        result = rebuild_tucker(t, wa, wb)
        expected = self._einsum_reference(t, wa, wb)
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)
        self.assertEqual(result.shape, (8, 6, 3, 3, 3))

    def test_rank_1_tucker(self):
        """Edge case: rank=1."""
        device = _device()
        t = torch.randn(1, 1, 3, 3, device=device)
        wa = torch.randn(1, 4, device=device)
        wb = torch.randn(1, 8, device=device)
        result = rebuild_tucker(t, wa, wb)
        expected = self._einsum_reference(t, wa, wb)
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_grad_flows_through_tensordot(self):
        """Verify gradients flow through the new tensordot path."""
        device = _device()
        t = torch.randn(4, 3, 5, 5, device=device, requires_grad=True)
        wa = torch.randn(4, 8, device=device, requires_grad=True)
        wb = torch.randn(3, 6, device=device, requires_grad=True)
        result = rebuild_tucker(t, wa, wb)
        loss = result.sum()
        loss.backward()
        self.assertIsNotNone(t.grad)
        self.assertIsNotNone(wa.grad)
        self.assertIsNotNone(wb.grad)

    def test_tucker_in_locon_forward(self):
        """Tucker-mode LoCon forward should produce correct results."""
        device = _device()
        dtype = torch.float32
        base = nn.Conv2d(16, 16, 3, 1, 1).to(device, dtype)
        net = LoConModule("test", base, lora_dim=4, alpha=1,
                          use_tucker=True, use_scalar=True).to(device, dtype)
        net.apply_to()

        x = torch.randn(1, 16, 8, 8, device=device, dtype=dtype)
        out = base(x)
        self.assertEqual(out.shape, (1, 16, 8, 8))
        self.assertFalse(torch.isnan(out).any())

    def test_tucker_in_locon_compiled(self):
        """Tucker-mode LoCon compiled forward matches non-compiled."""
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
        torch.testing.assert_close(out_a, out_b, atol=1e-5, rtol=1e-5)


# ===========================================================================
# 2.  _orthogonalize() — compile-friendly path
# ===========================================================================

class OrthogonalizeCacheTests(unittest.TestCase):
    """Verify _orthogonalize uses _qr_ndim and produces correct results."""

    def test_qr_ndim_set_for_linear(self):
        """_qr_ndim should be set to _expected_ndim for linear modules."""
        device = _device()
        base = nn.Linear(16, 16).to(device)
        net = LoConModule("test", base, lora_dim=4, alpha=1,
                          orthogonalize=True).to(device)
        self.assertEqual(net._qr_ndim, 2)
        self.assertEqual(net._qr_ndim, net._expected_ndim)

    def test_qr_ndim_set_for_conv2d(self):
        """_qr_ndim should be set to _expected_ndim for conv2d modules."""
        device = _device()
        base = nn.Conv2d(16, 16, 3, 1, 1).to(device)
        net = LoConModule("test", base, lora_dim=4, alpha=1,
                          orthogonalize=True).to(device)
        self.assertEqual(net._qr_ndim, 4)
        self.assertEqual(net._qr_ndim, net._expected_ndim)

    def test_orthogonalize_linear_2d(self):
        """_orthogonalize on 2D weight should produce correct result."""
        device = _device()
        base = nn.Linear(16, 16).to(device)
        net = LoConModule("test", base, lora_dim=4, alpha=1,
                          orthogonalize=True).to(device)
        net.train()

        w = torch.randn(16, 4, device=device)
        result = net._orthogonalize(w)
        self.assertEqual(result.shape, w.shape)
        self.assertFalse(torch.isnan(result).any())

    def test_orthogonalize_conv_4d(self):
        """_orthogonalize on 4D weight should produce correct result."""
        device = _device()
        base = nn.Conv2d(16, 16, 3, 1, 1).to(device)
        net = LoConModule("test", base, lora_dim=4, alpha=1,
                          orthogonalize=True).to(device)
        net.train()

        w = torch.randn(16, 4, 1, 1, device=device)
        result = net._orthogonalize(w)
        self.assertEqual(result.shape, w.shape)
        self.assertFalse(torch.isnan(result).any())

    def test_orthogonalize_noop_when_disabled(self):
        """_orthogonalize should return input unchanged when disabled."""
        device = _device()
        base = nn.Linear(16, 16).to(device)
        net = LoConModule("test", base, lora_dim=4, alpha=1,
                          orthogonalize=False).to(device)
        net.train()

        w = torch.randn(16, 4, device=device)
        result = net._orthogonalize(w)
        self.assertTrue(torch.equal(result, w))

    def test_orthogonalize_noop_when_eval(self):
        """_orthogonalize should return input unchanged in eval mode."""
        device = _device()
        base = nn.Linear(16, 16).to(device)
        net = LoConModule("test", base, lora_dim=4, alpha=1,
                          orthogonalize=True).to(device)
        net.eval()

        w = torch.randn(16, 4, device=device)
        result = net._orthogonalize(w)
        self.assertTrue(torch.equal(result, w))

    def test_orthogonalize_rows_lt_cols(self):
        """_orthogonalize handles rows < cols (transpose path)."""
        device = _device()
        base = nn.Linear(16, 16).to(device)
        net = LoConModule("test", base, lora_dim=4, alpha=1,
                          orthogonalize=True).to(device)
        net.train()

        # lora_down shape: (rank, in_features) → rows < cols
        w = torch.randn(4, 16, device=device)
        result = net._orthogonalize(w)
        self.assertEqual(result.shape, w.shape)
        self.assertFalse(torch.isnan(result).any())

    def test_compile_with_orthogonalize(self):
        """Compiled forward with orthogonalize should match non-compiled."""
        device = _device()
        dtype = torch.float32
        dim = 32

        torch.manual_seed(42)
        base_a = nn.Linear(dim, dim).to(device, dtype)
        net_a = LoConModule("test", base_a, lora_dim=4, alpha=1,
                            orthogonalize=True, use_scalar=True).to(device, dtype)
        net_a.apply_to()

        torch.manual_seed(42)
        base_b = nn.Linear(dim, dim).to(device, dtype)
        base_b.load_state_dict(base_a.state_dict())
        net_b = LoConModule("test", base_b, lora_dim=4, alpha=1,
                            orthogonalize=True, use_scalar=True).to(device, dtype)
        net_b.apply_to()
        net_b.compile_forward(**_compile_kwargs())

        x = torch.randn(2, dim, device=device, dtype=dtype)
        with torch.no_grad():
            out_a = base_a(x)
            out_b = base_b(x)
        torch.testing.assert_close(out_a, out_b, atol=1e-5, rtol=1e-5)


# ===========================================================================
# 3.  LoHa rank dropout — pre-computed _rank_drop_shape
# ===========================================================================

class LoHaRankDropoutTests(unittest.TestCase):
    """Verify LoHa get_weight() rank dropout uses _rank_drop_shape correctly."""

    def test_rank_drop_shape_set_for_loha_linear(self):
        """_rank_drop_shape should be set for LoHa linear modules."""
        device = _device()
        base = nn.Linear(16, 16).to(device)
        net = LohaModule("test", base, lora_dim=4, alpha=1,
                         rank_dropout=0.5).to(device)
        self.assertEqual(net._rank_drop_shape, [-1, 1])

    def test_rank_drop_shape_set_for_loha_conv2d(self):
        """_rank_drop_shape should be set for LoHa conv2d modules."""
        device = _device()
        base = nn.Conv2d(16, 16, 3, 1, 1).to(device)
        net = LohaModule("test", base, lora_dim=4, alpha=1,
                         rank_dropout=0.5).to(device)
        self.assertEqual(net._rank_drop_shape, [-1, 1, 1, 1])

    def test_get_weight_with_rank_dropout_linear(self):
        """get_weight() should produce correct shapes with rank_dropout."""
        device = _device()
        base = nn.Linear(16, 16).to(device)
        net = LohaModule("test", base, lora_dim=4, alpha=1,
                         rank_dropout=0.5, use_scalar=True).to(device)
        net.train()

        weight = net.get_weight(net.shape)
        self.assertEqual(weight.shape, net.shape)
        self.assertFalse(torch.isnan(weight).any())

    def test_get_weight_with_rank_dropout_conv2d(self):
        """get_weight() conv2d should produce correct shapes with rank_dropout."""
        device = _device()
        base = nn.Conv2d(16, 16, 3, 1, 1).to(device)
        net = LohaModule("test", base, lora_dim=4, alpha=1,
                         rank_dropout=0.5, use_scalar=True).to(device)
        net.train()

        weight = net.get_weight(net.shape)
        self.assertEqual(weight.shape, net.shape)
        self.assertFalse(torch.isnan(weight).any())

    def test_loha_compile_equivalence(self):
        """Compiled LoHa forward should match non-compiled."""
        device = _device()
        dtype = torch.float32
        dim = 32

        torch.manual_seed(42)
        base_a = nn.Linear(dim, dim).to(device, dtype)
        net_a = LohaModule("test", base_a, lora_dim=4, alpha=1,
                           use_scalar=True).to(device, dtype)
        net_a.apply_to()

        torch.manual_seed(42)
        base_b = nn.Linear(dim, dim).to(device, dtype)
        base_b.load_state_dict(base_a.state_dict())
        net_b = LohaModule("test", base_b, lora_dim=4, alpha=1,
                           use_scalar=True).to(device, dtype)
        net_b.apply_to()
        net_b.compile_forward(**_compile_kwargs())

        x = torch.randn(2, dim, device=device, dtype=dtype)
        with torch.no_grad():
            out_a = base_a(x)
            out_b = base_b(x)
        torch.testing.assert_close(out_a, out_b, atol=1e-5, rtol=1e-5)

    def test_loha_compile_with_rank_dropout(self):
        """Compiled LoHa with rank_dropout should work (eval mode)."""
        device = _device()
        dtype = torch.float32
        dim = 32

        torch.manual_seed(42)
        base_a = nn.Linear(dim, dim).to(device, dtype)
        net_a = LohaModule("test", base_a, lora_dim=4, alpha=1,
                           rank_dropout=0.1, use_scalar=True).to(device, dtype)
        net_a.eval()
        net_a.apply_to()

        torch.manual_seed(42)
        base_b = nn.Linear(dim, dim).to(device, dtype)
        base_b.load_state_dict(base_a.state_dict())
        net_b = LohaModule("test", base_b, lora_dim=4, alpha=1,
                           rank_dropout=0.1, use_scalar=True).to(device, dtype)
        net_b.eval()
        net_b.apply_to()
        net_b.compile_forward(**_compile_kwargs())

        x = torch.randn(2, dim, device=device, dtype=dtype)
        with torch.no_grad():
            out_a = base_a(x)
            out_b = base_b(x)
        torch.testing.assert_close(out_a, out_b, atol=1e-5, rtol=1e-5)


# ===========================================================================
# 4.  Integration: Tucker + Orthogonalize + Compile
# ===========================================================================

class TuckerOrthogonalizeCompileTests(unittest.TestCase):
    """Integration test: Tucker mode + orthogonalize + compiled forward."""

    def test_tucker_orthogonalize_training_forward(self):
        """Tucker + orthogonalize should produce valid outputs in training."""
        device = _device()
        dtype = torch.float32
        base = nn.Conv2d(16, 16, 3, 1, 1).to(device, dtype)
        net = LoConModule("test", base, lora_dim=4, alpha=1,
                          use_tucker=True, orthogonalize=True,
                          use_scalar=True).to(device, dtype)
        net.apply_to()
        net.train()

        x = torch.randn(1, 16, 8, 8, device=device, dtype=dtype)
        out = base(x)
        self.assertEqual(out.shape, (1, 16, 8, 8))
        self.assertFalse(torch.isnan(out).any())

        # Verify gradients flow
        loss = out.sum()
        loss.backward()
        for name, p in net.named_parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad, f"{name} has no grad")

    def test_tucker_orthogonalize_compiled(self):
        """Tucker + orthogonalize compiled should match non-compiled."""
        device = _device()
        dtype = torch.float32
        dim = 16

        torch.manual_seed(42)
        base_a = nn.Conv2d(dim, dim, 3, 1, 1).to(device, dtype)
        net_a = LoConModule("test", base_a, lora_dim=4, alpha=1,
                            use_tucker=True, orthogonalize=True,
                            use_scalar=True).to(device, dtype)
        net_a.apply_to()

        torch.manual_seed(42)
        base_b = nn.Conv2d(dim, dim, 3, 1, 1).to(device, dtype)
        base_b.load_state_dict(base_a.state_dict())
        net_b = LoConModule("test", base_b, lora_dim=4, alpha=1,
                            use_tucker=True, orthogonalize=True,
                            use_scalar=True).to(device, dtype)
        net_b.apply_to()
        net_b.compile_forward(**_compile_kwargs())

        x = torch.randn(1, dim, 8, 8, device=device, dtype=dtype)
        with torch.no_grad():
            out_a = base_a(x)
            out_b = base_b(x)
        torch.testing.assert_close(out_a, out_b, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
