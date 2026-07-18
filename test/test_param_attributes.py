"""
Tests for optimizer-relevant parameter attributes on LyCORIS modules.

Verifies that ``LycorisBaseModule.tag_parameters()`` (called via
``LycorisNetworkKohya._tag_all_parameters()`` from ``prepare_optimizer_params``
and ``prepare_grad_etc``) sets the following attributes on the correct
``nn.Parameter`` objects so that Advanced_Optimizers can identify each
parameter's role:

    _is_dora_scale  -> DoRA magnitude scale
    _is_oft         -> OFT skew-symmetric blocks
    _is_lora_A      -> LoRA down/A factor
    _is_lora_B      -> LoRA up/B factor
    is_hidden       -> generic 2D hidden-layer weight
    is_vector       -> logically-vector parameter (multi-dim)

Also verifies the attributes survive a device move (``.to()``) because
``prepare_optimizer_params`` re-applies them after the move.
"""

import math

import pytest
import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32


# We need a wrapper whose class name is in UNET_TARGET_REPLACE_MODULE.
class Transformer2DModel(nn.Module):
    """Mock transformer block so LycorisNetworkKohya finds target modules."""

    def __init__(self, in_f=64, out_f=32):
        super().__init__()
        self.linear = nn.Linear(in_f, out_f, bias=False)


def _make_model():
    return nn.ModuleDict({
        "block1": Transformer2DModel(64, 32),
        "block2": Transformer2DModel(64, 32),
    }).to(DEVICE)


def _make_kohya_network(algo="locon", weight_decompose=False, **kwargs):
    from lycoris.kohya import LycorisNetworkKohya

    model = _make_model()
    network = LycorisNetworkKohya(
        None,  # text_encoder
        model,  # unet
        multiplier=1.0,
        lora_dim=4,
        alpha=4,
        network_module=algo,
        weight_decompose=weight_decompose,
        **kwargs,
    )
    # Register the lora modules as submodules so named_parameters() sees them.
    network.apply_to(None, model, False, True)
    return network


def _prepare_optimizer_params(net, lr=1e-3):
    """Call LycorisNetworkKohya.prepare_optimizer_params with a valid unet_lr.

    The Kohya variant returns a (param_groups, lr_descriptions) tuple and
    requires an explicit unet_lr (or learning_rate fallback) to actually
    create parameter groups.
    """
    return net.prepare_optimizer_params(unet_lr=lr)


def _first_module_of_type(network, cls_name):
    """Return the first lora module whose class name matches cls_name."""
    for lora in network.loras:
        if lora.__class__.__name__ == cls_name:
            return lora
    return None


# ---------------------------------------------------------------------------
# 1. Plain LoRA (locon): lora_down -> A, lora_up -> B
# ---------------------------------------------------------------------------

class TestLoConAttributes:
    def test_lora_down_is_lora_a(self):
        net = _make_kohya_network(algo="locon")
        _prepare_optimizer_params(net)
        mod = _first_module_of_type(net, "LoConModule")
        assert mod is not None
        assert getattr(mod.lora_down.weight, "_is_lora_A", False) is True
        assert getattr(mod.lora_down.weight, "is_hidden", False) is True

    def test_lora_up_is_lora_b(self):
        net = _make_kohya_network(algo="locon")
        _prepare_optimizer_params(net)
        mod = _first_module_of_type(net, "LoConModule")
        assert mod is not None
        assert getattr(mod.lora_up.weight, "_is_lora_B", False) is True
        assert getattr(mod.lora_up.weight, "is_hidden", False) is True

    def test_no_dora_scale_without_wd(self):
        net = _make_kohya_network(algo="locon", weight_decompose=False)
        _prepare_optimizer_params(net)
        mod = _first_module_of_type(net, "LoConModule")
        assert mod is not None
        assert not hasattr(mod, "dora_scale") or not isinstance(
            getattr(mod, "dora_scale", None), nn.Parameter
        )

    def test_no_oft_blocks(self):
        net = _make_kohya_network(algo="locon")
        _prepare_optimizer_params(net)
        mod = _first_module_of_type(net, "LoConModule")
        assert mod is not None
        assert not hasattr(mod, "oft_blocks")


# ---------------------------------------------------------------------------
# 2. DoRA (locon + weight_decompose): dora_scale tagged
# ---------------------------------------------------------------------------

class TestDoRAAttributes:
    def test_dora_scale_tagged(self):
        net = _make_kohya_network(algo="locon", weight_decompose=True)
        _prepare_optimizer_params(net)
        mod = _first_module_of_type(net, "LoConModule")
        assert mod is not None
        assert isinstance(mod.dora_scale, nn.Parameter)
        assert getattr(mod.dora_scale, "_is_dora_scale", False) is True
        assert getattr(mod.dora_scale, "is_vector", False) is True

    def test_dora_lora_factors_still_tagged(self):
        net = _make_kohya_network(algo="locon", weight_decompose=True)
        _prepare_optimizer_params(net)
        mod = _first_module_of_type(net, "LoConModule")
        assert mod is not None
        assert getattr(mod.lora_down.weight, "_is_lora_A", False) is True
        assert getattr(mod.lora_up.weight, "_is_lora_B", False) is True


# ---------------------------------------------------------------------------
# 3. Diag-OFT: oft_blocks tagged, rescale tagged as vector
# ---------------------------------------------------------------------------

class TestDiagOFTAttributes:
    def test_oft_blocks_tagged(self):
        net = _make_kohya_network(algo="diag-oft")
        _prepare_optimizer_params(net)
        mod = _first_module_of_type(net, "DiagOFTModule")
        assert mod is not None
        assert isinstance(mod.oft_blocks, nn.Parameter)
        assert getattr(mod.oft_blocks, "_is_oft", False) is True

    def test_rescale_tagged_as_vector(self):
        net = _make_kohya_network(algo="diag-oft", rescaled=True)
        _prepare_optimizer_params(net)
        mod = _first_module_of_type(net, "DiagOFTModule")
        assert mod is not None
        assert isinstance(mod.rescale, nn.Parameter)
        assert getattr(mod.rescale, "is_vector", False) is True

    def test_oft_not_tagged_as_lora(self):
        net = _make_kohya_network(algo="diag-oft")
        _prepare_optimizer_params(net)
        mod = _first_module_of_type(net, "DiagOFTModule")
        assert mod is not None
        assert getattr(mod.oft_blocks, "_is_lora_A", False) is False
        assert getattr(mod.oft_blocks, "_is_lora_B", False) is False


# ---------------------------------------------------------------------------
# 4. BOFT: oft_blocks tagged
# ---------------------------------------------------------------------------

class TestBOFTAttributes:
    def test_boft_blocks_tagged(self):
        net = _make_kohya_network(algo="boft")
        _prepare_optimizer_params(net)
        mod = _first_module_of_type(net, "ButterflyOFTModule")
        assert mod is not None
        assert isinstance(mod.oft_blocks, nn.Parameter)
        assert getattr(mod.oft_blocks, "_is_oft", False) is True


# ---------------------------------------------------------------------------
# 5. Full: diff weight tagged as is_hidden
# ---------------------------------------------------------------------------

class TestFullAttributes:
    def test_full_diff_is_hidden(self):
        net = _make_kohya_network(algo="full")
        _prepare_optimizer_params(net)
        mod = _first_module_of_type(net, "FullModule")
        assert mod is not None
        assert isinstance(mod.weight, nn.Parameter)
        assert getattr(mod.weight, "is_hidden", False) is True
        # Should NOT be tagged as lora/oft/dora
        assert getattr(mod.weight, "_is_lora_A", False) is False
        assert getattr(mod.weight, "_is_lora_B", False) is False
        assert getattr(mod.weight, "_is_oft", False) is False
        assert getattr(mod.weight, "_is_dora_scale", False) is False


# ---------------------------------------------------------------------------
# 6. Attributes survive device move (re-applied by prepare_optimizer_params)
# ---------------------------------------------------------------------------

class TestAttributesSurviveDeviceMove:
    def test_attributes_after_to(self):
        net = _make_kohya_network(algo="locon", weight_decompose=True)
        # Move to a (possibly different) device — .to() replaces Parameters
        net.to(DEVICE)
        # prepare_optimizer_params must re-apply the attributes
        _prepare_optimizer_params(net)
        mod = _first_module_of_type(net, "LoConModule")
        assert mod is not None
        assert getattr(mod.lora_down.weight, "_is_lora_A", False) is True
        assert getattr(mod.lora_up.weight, "_is_lora_B", False) is True
        assert getattr(mod.dora_scale, "_is_dora_scale", False) is True
        assert getattr(mod.dora_scale, "is_vector", False) is True

    def test_prepare_grad_etc_also_tags(self):
        net = _make_kohya_network(algo="locon")
        net.to(DEVICE)
        net.prepare_grad_etc()
        mod = _first_module_of_type(net, "LoConModule")
        assert mod is not None
        assert getattr(mod.lora_down.weight, "_is_lora_A", False) is True
        assert getattr(mod.lora_up.weight, "_is_lora_B", False) is True


# ---------------------------------------------------------------------------
# 7. tag_parameters is idempotent (safe to call multiple times)
# ---------------------------------------------------------------------------

class TestIdempotent:
    def test_double_call_safe(self):
        net = _make_kohya_network(algo="locon", weight_decompose=True)
        net._tag_all_parameters()
        net._tag_all_parameters()
        mod = _first_module_of_type(net, "LoConModule")
        assert mod is not None
        assert getattr(mod.lora_down.weight, "_is_lora_A", False) is True
        assert getattr(mod.dora_scale, "_is_dora_scale", False) is True


# ---------------------------------------------------------------------------
# 8. All trainable params handed to optimizer carry expected tags
# ---------------------------------------------------------------------------

class TestAllParamsTagged:
    def test_every_2d_param_has_role(self):
        """Every 2D trainable parameter should be tagged with at least one
        role attribute (_is_oft, _is_lora_A, _is_lora_B, _is_dora_scale,
        or is_hidden)."""
        for algo in ("locon", "diag-oft", "boft", "full"):
            net = _make_kohya_network(algo=algo, weight_decompose=(algo == "locon"))
            groups, _ = _prepare_optimizer_params(net)
            for group in groups:
                for p in group["params"]:
                    if p.ndim < 2:
                        continue
                    has_role = (
                        getattr(p, "_is_oft", False)
                        or getattr(p, "_is_lora_A", False)
                        or getattr(p, "_is_lora_B", False)
                        or getattr(p, "_is_dora_scale", False)
                        or getattr(p, "is_hidden", False)
                    )
                    assert has_role, (
                        f"algo={algo}: 2D param with shape {tuple(p.shape)} "
                        f"has no role attribute"
                    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
