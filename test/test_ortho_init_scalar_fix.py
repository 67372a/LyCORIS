"""
Tests for the orthogonal-init / scalar interaction fix.

Before the fix, ``orthogonal_init=True`` initialized the "up" (second)
weight matrix with orthogonal init even when ``use_scalar=False``.  That
breaks the standard LoRA convention that the adapter produces a zero delta
at initialization, because the down and up matrices are then both non-zero.

The fix makes ``use_scalar`` the outer condition: when no scalar is used,
the second matrix is zero-initialized (delta = 0 at init), and orthogonal
init is only applied when a scalar is actually used.
"""

import torch
import torch.nn as nn

from lycoris.modules import GLoRAModule, LohaModule, LokrModule, LoConModule

DEVICE = torch.device("cuda")

_MODULES = (LoConModule, LohaModule, LokrModule, GLoRAModule)


def _make_linear(in_f=16, out_f=16):
    mod = nn.Linear(in_f, out_f, bias=False).to(DEVICE)
    nn.init.kaiming_uniform_(mod.weight, a=5 ** 0.5)
    return mod


def _make_module(module_cls, org_module, *, use_scalar, orthogonal_init):
    return module_cls(
        "test_layer",
        org_module,
        multiplier=1.0,
        lora_dim=4,
        alpha=1.0,
        use_scalar=use_scalar,
        orthogonal_init=orthogonal_init,
    ).to(DEVICE)


def test_no_scalar_orthogonal_init_produces_zero_diff():
    """use_scalar=False + orthogonal_init=True must still zero-init the adapter."""
    for module_cls in _MODULES:
        org = _make_linear()
        mod = _make_module(
            module_cls, org, use_scalar=False, orthogonal_init=True
        )

        diff, _ = mod.get_diff_weight(multiplier=1.0, device=DEVICE)
        assert torch.allclose(
            diff.float(), torch.zeros_like(diff.float()), atol=1e-6
        ), f"{module_cls.__name__}: expected zero diff weight, got max={diff.abs().max().item():.3e}"


def test_scalar_orthogonal_init_produces_nonzero_diff():
    """use_scalar=True + orthogonal_init=True should keep orthogonal init."""
    for module_cls in _MODULES:
        org = _make_linear()
        mod = _make_module(
            module_cls, org, use_scalar=True, orthogonal_init=True
        )

        diff, _ = mod.get_diff_weight(multiplier=1.0, device=DEVICE)
        assert diff.float().abs().max() > 1e-6, (
            f"{module_cls.__name__}: expected non-zero diff weight with "
            f"use_scalar=True + orthogonal_init=True"
        )


def test_no_scalar_orthogonal_init_forward_matches_org():
    """Forward output must equal the base module output at initialization."""
    for module_cls in _MODULES:
        org = _make_linear()
        mod = _make_module(
            module_cls, org, use_scalar=False, orthogonal_init=True
        )

        x = torch.randn(2, 16, device=DEVICE)
        expected = org(x).detach().clone()

        mod.apply_to()
        actual = mod(x)

        assert torch.allclose(actual, expected, atol=1e-5), (
            f"{module_cls.__name__}: forward output diverged from base at init, "
            f"max diff={(actual - expected).abs().max().item():.3e}"
        )
