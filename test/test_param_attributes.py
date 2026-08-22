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


# ---------------------------------------------------------------------------
# 9. is_hidden heuristic via original_name
# ---------------------------------------------------------------------------

class TestIsHiddenHeuristic:
    def test_hidden_layer_defaults_true(self):
        """Modules with no original_name default to is_hidden=True."""
        net = _make_kohya_network(algo="locon")
        _prepare_optimizer_params(net)
        mod = _first_module_of_type(net, "LoConModule")
        assert mod is not None
        # The mock model uses Transformer2DModel → original_name contains
        # 'block1.linear' or similar → not a non-hidden prefix → is_hidden=True.
        assert getattr(mod.lora_down.weight, "is_hidden", False) is True

    def test_non_hidden_time_embedding(self):
        """A module whose original_name starts with 'time_embedding' should
        have is_hidden=False."""
        from lycoris.modules.locon import LoConModule

        net = _make_kohya_network(algo="locon")
        _prepare_optimizer_params(net)
        mod = _first_module_of_type(net, "LoConModule")
        assert mod is not None
        # Simulate a non-hidden layer by setting original_name
        mod.original_name = "time_embedding.linear_1"
        mod.tag_parameters()
        assert getattr(mod.lora_down.weight, "is_hidden", True) is False

    def test_non_hidden_conv_in(self):
        from lycoris.modules.locon import LoConModule

        net = _make_kohya_network(algo="locon")
        _prepare_optimizer_params(net)
        mod = _first_module_of_type(net, "LoConModule")
        mod.original_name = "conv_in"
        mod.tag_parameters()
        assert getattr(mod.lora_down.weight, "is_hidden", True) is False

    def test_non_hidden_final_layer(self):
        from lycoris.modules.locon import LoConModule

        net = _make_kohya_network(algo="locon")
        _prepare_optimizer_params(net)
        mod = _first_module_of_type(net, "LoConModule")
        mod.original_name = "final_layer.linear"
        mod.tag_parameters()
        assert getattr(mod.lora_down.weight, "is_hidden", True) is False

    def test_hidden_layer_nested_path(self):
        """A nested path like 'down_blocks.0.attentions.0...' is hidden."""
        from lycoris.modules.locon import LoConModule

        net = _make_kohya_network(algo="locon")
        _prepare_optimizer_params(net)
        mod = _first_module_of_type(net, "LoConModule")
        mod.original_name = "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q"
        mod.tag_parameters()
        assert getattr(mod.lora_down.weight, "is_hidden", False) is True

    def test_non_hidden_img_in(self):
        from lycoris.modules.locon import LoConModule

        net = _make_kohya_network(algo="locon")
        _prepare_optimizer_params(net)
        mod = _first_module_of_type(net, "LoConModule")
        mod.original_name = "img_in"
        mod.tag_parameters()
        assert getattr(mod.lora_down.weight, "is_hidden", True) is False


# ---------------------------------------------------------------------------
# 10. is_norm, is_scalar, is_bias, and weight_decay_ratio tags
# ---------------------------------------------------------------------------

class TestNormScalarBiasAndWeightDecayTags:
    def test_norm_module_tagged(self):
        """NormModule w_norm and b_norm should have is_norm=True, weight_decay_ratio=0.0."""
        from lycoris.modules.norms import NormModule

        ln = nn.LayerNorm(32).to(DEVICE)
        mod = NormModule("test_norm", ln, multiplier=1.0).to(DEVICE)
        mod.tag_parameters()

        assert getattr(mod.w_norm, "is_norm", False) is True
        assert getattr(mod.w_norm, "is_bias", True) is False
        assert getattr(mod.w_norm, "weight_decay_ratio", None) == 0.0

        assert getattr(mod.b_norm, "is_norm", False) is True
        assert getattr(mod.b_norm, "is_bias", False) is True
        assert getattr(mod.b_norm, "weight_decay_ratio", None) == 0.0

    def test_adaln_modulation_tagged_as_norm(self):
        """Modules targeting AdaLN / modulation projections should have is_norm=True and weight_decay_ratio=0.0."""
        net = _make_kohya_network(algo="locon")
        _prepare_optimizer_params(net)
        mod = _first_module_of_type(net, "LoConModule")
        assert mod is not None

        # Simulate AdaLN modulation layer in DiT
        mod.original_name = "blocks.0.adaLN_modulation.1"
        mod.tag_parameters()

        assert getattr(mod.lora_down.weight, "is_norm", False) is True
        assert getattr(mod.lora_up.weight, "is_norm", False) is True
        assert getattr(mod.lora_down.weight, "weight_decay_ratio", None) == 0.0
        assert getattr(mod.lora_up.weight, "weight_decay_ratio", None) == 0.0

    def test_scalar_tagged(self):
        """Scalar parameters (like use_scalar or lora2_nu) should have is_scalar=True, weight_decay_ratio=0.0."""
        from lycoris.modules.locon import LoConModule

        linear = nn.Linear(64, 32, bias=False).to(DEVICE)
        mod = LoConModule("test_scalar", linear, lora_dim=4, alpha=4, use_scalar=True).to(DEVICE)
        mod.tag_parameters()

        assert isinstance(mod.scalar, nn.Parameter)
        assert getattr(mod.scalar, "is_scalar", False) is True
        assert getattr(mod.scalar, "is_bias", True) is False
        assert getattr(mod.scalar, "weight_decay_ratio", None) == 0.0

        # LoRA A and B should have is_scalar=False and weight_decay_ratio=1.0
        assert getattr(mod.lora_down.weight, "is_scalar", True) is False
        assert getattr(mod.lora_down.weight, "weight_decay_ratio", None) == 1.0
        assert getattr(mod.lora_up.weight, "is_scalar", True) is False
        assert getattr(mod.lora_up.weight, "weight_decay_ratio", None) == 1.0

    def test_full_diff_bias_tagged(self):
        """FullModule bias parameter should have is_bias=True, weight_decay_ratio=0.0."""
        from lycoris.modules.full import FullModule

        linear_with_bias = nn.Linear(64, 32, bias=True).to(DEVICE)
        mod = FullModule("test_full", linear_with_bias, multiplier=1.0).to(DEVICE)
        mod.tag_parameters()

        assert isinstance(mod.bias, nn.Parameter)
        assert getattr(mod.bias, "is_bias", False) is True
        assert getattr(mod.bias, "weight_decay_ratio", None) == 0.0

        assert isinstance(mod.weight, nn.Parameter)
        assert getattr(mod.weight, "is_bias", True) is False
        assert getattr(mod.weight, "weight_decay_ratio", None) == 1.0

    def test_dora_scale_weight_decay_ratio(self):
        """DoRA scale vector should have weight_decay_ratio=0.0."""
        net = _make_kohya_network(algo="locon", weight_decompose=True)
        _prepare_optimizer_params(net)
        mod = _first_module_of_type(net, "LoConModule")
        assert mod is not None

        assert getattr(mod.dora_scale, "weight_decay_ratio", None) == 0.0
        assert getattr(mod.lora_down.weight, "weight_decay_ratio", None) == 1.0
        assert getattr(mod.lora_up.weight, "weight_decay_ratio", None) == 1.0

    def test_custom_weight_decay_ratio_override(self):
        """Custom weight_decay_ratio on module or parameter should override default."""
        net = _make_kohya_network(algo="locon")
        _prepare_optimizer_params(net)
        mod = _first_module_of_type(net, "LoConModule")
        assert mod is not None

        # Module-level override
        mod.weight_decay_ratio = 0.5
        mod.tag_parameters()
        assert getattr(mod.lora_down.weight, "weight_decay_ratio", None) == 0.5
        assert getattr(mod.lora_up.weight, "weight_decay_ratio", None) == 0.5

        # Parameter-level override
        mod.lora_down.weight.custom_weight_decay_ratio = 0.25
        mod.tag_parameters()
        assert getattr(mod.lora_down.weight, "weight_decay_ratio", None) == 0.25


# ---------------------------------------------------------------------------
# 11. Test tag_lora_module_params from network_base.py
# ---------------------------------------------------------------------------

class TestNetworkBaseTagLoraModuleParams:
    def _get_tag_fn(self):
        import importlib.util
        import sys
        from pathlib import Path
        sd_scripts_dir = Path(__file__).parent.parent.parent / "LoRA_Easy_Training_Scripts" / "backend" / "sd_scripts"
        path = sd_scripts_dir / "networks" / "network_base.py"
        if not path.exists():
            pytest.skip("network_base.py not found at expected path")
        if str(sd_scripts_dir) not in sys.path:
            sys.path.insert(0, str(sd_scripts_dir))
        spec = importlib.util.spec_from_file_location("network_base", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.tag_lora_module_params

    def test_network_base_tagging_lora(self):
        tag_lora_module_params = self._get_tag_fn()

        class MockLoRA(nn.Module):
            def __init__(self):
                super().__init__()
                self.lora_down = nn.Linear(64, 4, bias=False)
                self.lora_up = nn.Linear(4, 32, bias=False)
                self.dora_scale = nn.Parameter(torch.ones(32))
                self.original_name = "blocks.0.attn1.to_q"

        mod = MockLoRA().to(DEVICE)
        tag_lora_module_params(mod)

        assert getattr(mod.lora_down.weight, "_is_lora_A", False) is True
        assert getattr(mod.lora_down.weight, "is_hidden", False) is True
        assert getattr(mod.lora_down.weight, "is_norm", True) is False
        assert getattr(mod.lora_down.weight, "is_scalar", True) is False
        assert getattr(mod.lora_down.weight, "is_bias", True) is False
        assert getattr(mod.lora_down.weight, "weight_decay_ratio", None) == 1.0

        assert getattr(mod.lora_up.weight, "_is_lora_B", False) is True
        assert getattr(mod.lora_up.weight, "weight_decay_ratio", None) == 1.0

        assert getattr(mod.dora_scale, "_is_dora_scale", False) is True
        assert getattr(mod.dora_scale, "is_vector", False) is True
        assert getattr(mod.dora_scale, "weight_decay_ratio", None) == 0.0

    def test_network_base_tagging_adaln_norm(self):
        tag_lora_module_params = self._get_tag_fn()

        class MockAdaLN(nn.Module):
            def __init__(self):
                super().__init__()
                self.lora_down = nn.Linear(64, 4, bias=False)
                self.lora_up = nn.Linear(4, 32, bias=False)
                self.original_name = "blocks.0.adaLN_modulation.1"

        mod = MockAdaLN().to(DEVICE)
        tag_lora_module_params(mod)

        assert getattr(mod.lora_down.weight, "is_norm", False) is True
        assert getattr(mod.lora_up.weight, "is_norm", False) is True
        assert getattr(mod.lora_down.weight, "weight_decay_ratio", None) == 0.0
        assert getattr(mod.lora_up.weight, "weight_decay_ratio", None) == 0.0


# ---------------------------------------------------------------------------
# 12. Test WarpAINO with weight_decay_ratio
# ---------------------------------------------------------------------------

class TestWarpAINOWeightDecayRatio:
    def _get_warpaino_cls(self):
        import importlib.util
        from pathlib import Path
        path = Path(__file__).parent.parent.parent / "LoRA_Easy_Training_Scripts" / "backend" / "custom_scheduler" / "LoraEasyCustomOptimizer" / "warpaino.py"
        if not path.exists():
            pytest.skip("warpaino.py not found at expected path")
        spec = importlib.util.spec_from_file_location("warpaino", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.WarpAINO

    def test_warpaino_weight_decay_ratio_native(self):
        WarpAINO = self._get_warpaino_cls()

        # Create three identical 2D parameters on CUDA
        p_full = nn.Parameter(torch.ones(16, 16, device=DEVICE) * 2.0)
        p_zero = nn.Parameter(torch.ones(16, 16, device=DEVICE) * 2.0)
        p_half = nn.Parameter(torch.ones(16, 16, device=DEVICE) * 2.0)

        p_full.weight_decay_ratio = 1.0
        p_zero.weight_decay_ratio = 0.0
        p_half.weight_decay_ratio = 0.5

        # Provide zero gradients so only weight decay drives parameter changes
        p_full.grad = torch.zeros_like(p_full)
        p_zero.grad = torch.zeros_like(p_zero)
        p_half.grad = torch.zeros_like(p_half)

        opt = WarpAINO(
            [p_full, p_zero, p_half],
            lr=0.01,
            weight_decay=0.1,
            cautious_wd=False,
            foreach=False,
        )

        opt.step()

        # p_zero should not have decayed (stayed at 2.0)
        assert torch.allclose(p_zero.data, torch.ones(16, 16, device=DEVICE) * 2.0)

        # p_full should have decayed: p - lr * wd * p = 2.0 - 0.01 * 0.1 * 2.0 = 1.998
        decay_full = 2.0 - p_full.data.mean().item()
        decay_half = 2.0 - p_half.data.mean().item()

        assert decay_full > 0.001
        assert abs(decay_half - decay_full * 0.5) < 1e-4

    def test_warpaino_weight_decay_ratio_foreach(self):
        WarpAINO = self._get_warpaino_cls()

        p_full = nn.Parameter(torch.ones(16, 16, device=DEVICE) * 2.0)
        p_zero = nn.Parameter(torch.ones(16, 16, device=DEVICE) * 2.0)
        p_zero_1d = nn.Parameter(torch.ones(16, device=DEVICE) * 2.0)

        p_full.weight_decay_ratio = 1.0
        p_zero.weight_decay_ratio = 0.0
        p_zero_1d.weight_decay_ratio = 0.0

        p_full.grad = torch.zeros_like(p_full)
        p_zero.grad = torch.zeros_like(p_zero)
        p_zero_1d.grad = torch.zeros_like(p_zero_1d)

        opt = WarpAINO(
            [p_full, p_zero, p_zero_1d],
            lr=0.01,
            weight_decay=0.1,
            cautious_wd=False,
            foreach=True,
        )

        opt.step()

        assert torch.allclose(p_zero.data, torch.ones(16, 16, device=DEVICE) * 2.0)
        assert torch.allclose(p_zero_1d.data, torch.ones(16, device=DEVICE) * 2.0)
        assert p_full.data.mean().item() < 2.0

    def test_warpaino_skips_warp_on_1d_and_scalars(self):
        WarpAINO = self._get_warpaino_cls()

        p_bias = nn.Parameter(torch.ones(16, device=DEVICE))
        p_bias.is_bias = True
        p_bias.grad = torch.randn_like(p_bias)

        p_scalar = nn.Parameter(torch.tensor(1.0, device=DEVICE))
        p_scalar.is_scalar = True
        p_scalar.grad = torch.randn_like(p_scalar)

        p_2d = nn.Parameter(torch.ones(16, 16, device=DEVICE))
        p_2d.grad = torch.randn_like(p_2d)

        opt = WarpAINO([p_bias, p_scalar, p_2d], lr=0.01, warp_mode="dense")
        opt.step()

        # Warp matrix should not be allocated for 1D bias or scalar
        assert "warp" not in opt.state[p_bias]
        assert "warp" not in opt.state[p_scalar]
        # Warp matrix should be allocated for 2D matrix
        assert "warp" in opt.state[p_2d]

    def test_warpaino_unilateral_spectral_on_non_hidden(self):
        WarpAINO = self._get_warpaino_cls()

        p_hidden = nn.Parameter(torch.ones(32, 16, device=DEVICE))
        p_hidden.is_hidden = True
        p_hidden.grad = torch.randn_like(p_hidden)

        p_non_hidden = nn.Parameter(torch.ones(32, 16, device=DEVICE))
        p_non_hidden.is_hidden = False
        p_non_hidden.grad = torch.randn_like(p_non_hidden)

        opt = WarpAINO(
            [p_hidden, p_non_hidden],
            lr=0.01,
            warp_mode="spectral",
            spectral_bilateral=True,
        )
        opt.step()

        # Hidden layer should have bilateral spectral warp (left and right)
        assert "spectral_log_left" in opt.state[p_hidden]
        assert "spectral_log_right" in opt.state[p_hidden]

        # Non-hidden layer (e.g. embeddings / output heads) should only have unilateral left warp
        assert "spectral_log_left" in opt.state[p_non_hidden]
        assert "spectral_log_right" not in opt.state[p_non_hidden]


# ---------------------------------------------------------------------------
# 13. Test Advanced_Optimizers adjust_wds with weight_decay_ratio
# ---------------------------------------------------------------------------

class TestAdjustWDsWeightDecayRatio:
    def test_adjust_wds_ratio(self):
        import importlib.util
        import sys
        from pathlib import Path
        adv_optm_root = Path(__file__).parent.parent.parent / "Advanced_Optimizers"
        if not adv_optm_root.exists():
            pytest.skip("Advanced_Optimizers not found at expected path")
        if str(adv_optm_root) not in sys.path:
            sys.path.insert(0, str(adv_optm_root))
        from adv_optm.util.scaled_optm import adjust_wds

        p = nn.Parameter(torch.ones(16, 16, device=DEVICE))
        p.weight_decay_ratio = 0.5

        wd, cwd = adjust_wds(0.1, 0.05, p)
        assert abs(wd - 0.05) < 1e-6
        assert abs(cwd - 0.05) < 1e-6

        p.weight_decay_ratio = 0.0
        wd0, cwd0 = adjust_wds(0.1, 0.05, p)
        assert abs(wd0 - 0.0) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
