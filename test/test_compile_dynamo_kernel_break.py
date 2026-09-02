"""Regression: fused-kernel (triton/tilelang) entry points vs torch.compile.

Before the fix, a ``torch.compile``-wrapped module forward
(``_forward_rebuild_core``) traced through the autograd kernel entry points
into the triton launch path, where symbolic grid math crashed dynamo with
``InternalTorchDynamoError: 'SymNodeVariable' object has no attribute 'value'``.

The kernel entry points are now marked eager-under-dynamo (``torch.compiler.disable``),
so dynamo graph-breaks around the fused op and runs it eagerly. These tests
verify that compiled module forwards with the triton backend:

  - run without dynamo errors (forward AND backward),
  - match eager results within tolerance,
  - survive varying batch sizes (dynamic shapes),
  - and that the kernel functions themselves compile without being traced
    into the launch machinery.
"""

import copy
import os
import unittest

# Must be set before any lycoris import: backend resolution is process-wide.
os.environ.setdefault("LYCORIS_KERNEL_BACKEND", "triton")

import torch
import torch.nn as nn

from lycoris.kernels.autograd import locon as ag_locon
from lycoris.kernels.dispatch import available_backends, eager_under_dynamo
from lycoris.modules.locon import LoConModule
from lycoris.modules.loha import LohaModule
from lycoris.modules.lokr import LokrModule

CUDA_AVAILABLE = torch.cuda.is_available()
TRITON_AVAILABLE = "triton" in available_backends()

DEVICE = torch.device("cuda" if CUDA_AVAILABLE else "cpu")


def _compile_kwargs():
    if CUDA_AVAILABLE:
        return dict(mode="default", dynamic=True, fullgraph=False)
    return dict(backend="eager", fullgraph=False)


@unittest.skipUnless(CUDA_AVAILABLE, "CUDA required")
@unittest.skipUnless(TRITON_AVAILABLE, "triton backend required")
class CompiledModuleForwardTests(unittest.TestCase):
    """Compiled rebuild-mode forwards must run and match eager on CUDA/triton."""

    def _run_module(self, net, xs, targets):
        """Forward+backward per input; returns (losses, param grads snapshot)."""
        net.zero_grad(set_to_none=True)
        losses = []
        for x, t in zip(xs, targets):
            out = net(x)
            loss = (out - t).pow(2).mean()
            loss.backward()
            losses.append(loss.detach())
        grads = {
            name: (
                p.grad.detach().clone()
                if p.grad is not None
                else torch.zeros_like(p)
            )
            for name, p in net.named_parameters()
            if p.requires_grad
        }
        return torch.stack(losses), grads

    def _check_module(self, make_net, in_features=32, out_features=48):
        torch.manual_seed(0)
        eager_net = make_net().to(DEVICE)
        # Deep copy guarantees identical base (org) weights too —
        # load_state_dict only covers the adapter parameters.
        compiled_net = copy.deepcopy(eager_net)

        # Varying batch sizes exercise dynamic shapes under dynamo.
        xs = [
            torch.randn(2, in_features, device=DEVICE),
            torch.randn(3, in_features, device=DEVICE),
        ]
        targets = [torch.randn(x.shape[0], out_features, device=DEVICE) for x in xs]

        eager_losses, eager_grads = self._run_module(eager_net, xs, targets)

        compiled_net.compile_forward(**_compile_kwargs())
        # Warmup + steady state: the first call compiles, the second runs the
        # cached graph with a new symbolic batch size.
        compiled_losses, compiled_grads = self._run_module(compiled_net, xs, targets)

        self.assertFalse(
            any(torch.isnan(l) for l in compiled_losses), "NaN loss in compiled path"
        )
        torch.testing.assert_close(compiled_losses, eager_losses, rtol=1e-4, atol=1e-5)
        for name in eager_grads:
            torch.testing.assert_close(
                compiled_grads[name], eager_grads[name], rtol=1e-3, atol=1e-4
            )

    def test_locon_linear(self):
        self._check_module(
            lambda: LoConModule("t", nn.Linear(32, 48), lora_dim=4, alpha=2)
        )

    def test_loha_linear(self):
        self._check_module(
            lambda: LohaModule("t", nn.Linear(32, 48), lora_dim=4, alpha=2)
        )

    def test_lokr_linear(self):
        self._check_module(
            lambda: LokrModule("t", nn.Linear(64, 64), lora_dim=4, alpha=2, factor=16),
            in_features=64,
            out_features=64,
        )


@unittest.skipUnless(CUDA_AVAILABLE, "CUDA required")
@unittest.skipUnless(TRITON_AVAILABLE, "triton backend required")
class FusedKernelEagerUnderDynamoTests(unittest.TestCase):
    """The kernel entry points compile as opaque eager calls, results match."""

    def test_locon_diff_weight_compiled(self):
        torch.manual_seed(0)
        down = torch.randn(4, 16, device=DEVICE, requires_grad=True)
        up = torch.randn(48, 4, device=DEVICE, requires_grad=True)

        eager = ag_locon.locon_diff_weight(down, up, gamma=1.0)

        compiled_fn = torch.compile(ag_locon.locon_diff_weight, **_compile_kwargs())
        compiled = compiled_fn(down, up, gamma=1.0)

        torch.testing.assert_close(compiled, eager, rtol=1e-4, atol=1e-5)

    def test_locon_diff_weight_grad_flows_through_compile(self):
        torch.manual_seed(1)
        down = torch.randn(4, 16, device=DEVICE)
        up = torch.randn(48, 4, device=DEVICE)

        down_e = down.clone().requires_grad_(True)
        up_e = up.clone().requires_grad_(True)
        out = ag_locon.locon_diff_weight(down_e, up_e, gamma=1.0)
        out.sum().backward()

        down_c = down.clone().requires_grad_(True)
        up_c = up.clone().requires_grad_(True)
        compiled_fn = torch.compile(ag_locon.locon_diff_weight, **_compile_kwargs())
        compiled_fn(down_c, up_c, gamma=1.0).sum().backward()

        torch.testing.assert_close(down_c.grad, down_e.grad, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(up_c.grad, up_e.grad, rtol=1e-4, atol=1e-5)


class DecoratorBehaviorTests(unittest.TestCase):
    """The decorator wraps plain functions and stays transparent when eager."""

    def test_returns_callable_preserving_result(self):
        @eager_under_dynamo
        def add(a, b):
            return a + b

        self.assertEqual(add(2, 3), 5)

    def test_known_entries_are_marked(self):
        # Every entry point that reaches a fused kernel launch must be
        # wrapped, or dynamo can trace into the launch path and crash.
        from lycoris.kernels.autograd import (
            boft,
            dora,
            glora,
            ia3,
            loha,
            lokr,
            norms,
            full as full_ag,
            diag_oft,
        )

        for fn in (
            ag_locon.locon_diff_weight,
            ag_locon.locon_bypass_diff,
            loha.loha_diff_weight,
            loha.loha_bypass_diff,
            lokr.lokr_kron_weight,
            lokr.lokr_kron_bypass,
            glora.glora_diff_weight,
            dora.apply_dora,
            full_ag.full_diff_weight,
            norms.norm_diff_weights,
            ia3.ia3_diff_weight,
            ia3.ia3_bypass,
            diag_oft.diag_oft_diff_weight,
            diag_oft.diag_oft_bypass_diff,
            boft.boft_diff_weight,
            boft.boft_bypass_diff,
        ):
            self.assertTrue(
                getattr(fn, "_torchdynamo_disable", False)
                or getattr(fn, "__torch_dynamo_disable__", False),
                f"{fn.__name__} is not dynamo-disabled",
            )


if __name__ == "__main__":
    unittest.main()
