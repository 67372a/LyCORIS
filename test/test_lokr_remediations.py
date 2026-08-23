"""Regression tests for LoKr reconstruction, bypass, and serialization."""

import copy
import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from lycoris.modules.lokr import LokrModule


CUDA = torch.device("cuda")


def _require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("LoKr regression tests require CUDA")


def _copy_adapter(source, target):
    for name in (
        "lokr_w1", "lokr_w1_a", "lokr_w1_b", "lokr_w2", "lokr_w2_a",
        "lokr_w2_b", "lokr_t2",
    ):
        if hasattr(source, name) and hasattr(target, name):
            getattr(target, name).data.copy_(getattr(source, name).data)
    if isinstance(source.scalar, nn.Parameter) and isinstance(target.scalar, nn.Parameter):
        target.scalar.data.copy_(source.scalar.data)


def _linear_pair(**kwargs):
    base = nn.Linear(16, 16, bias=False, device=CUDA)
    rebuild_base = copy.deepcopy(base)
    bypass_base = copy.deepcopy(base)
    rebuild = LokrModule("rebuild", rebuild_base, multiplier=1.0, **kwargs).to(CUDA)
    bypass = LokrModule("bypass", bypass_base, multiplier=1.0, bypass_mode=True, **kwargs).to(CUDA)
    _copy_adapter(rebuild, bypass)
    rebuild.eval()
    bypass.eval()
    rebuild.apply_to()
    bypass.apply_to()
    return rebuild_base, bypass_base, rebuild, bypass


def _conv_pair(dim, **kwargs):
    conv_cls = (nn.Conv1d, nn.Conv2d, nn.Conv3d)[dim - 1]
    kernel = (3,) * dim
    common = dict(
        in_channels=4,
        out_channels=8,
        kernel_size=kernel,
        stride=(2,) * dim,
        padding=(2,) * dim,
        dilation=(2,) * dim,
        bias=False,
    )
    base = conv_cls(**common, device=CUDA)
    rebuild_base = copy.deepcopy(base)
    bypass_base = copy.deepcopy(base)
    rebuild = LokrModule("rebuild", rebuild_base, multiplier=1.0, **kwargs).to(CUDA)
    bypass = LokrModule("bypass", bypass_base, multiplier=1.0, bypass_mode=True, **kwargs).to(CUDA)
    _copy_adapter(rebuild, bypass)
    rebuild.eval()
    bypass.eval()
    rebuild.apply_to()
    bypass.apply_to()
    spatial = (15,) * dim
    x = torch.randn((2, 4, *spatial), device=CUDA)
    return rebuild_base, bypass_base, rebuild, bypass, x


def test_lokr_scale_is_applied_once_for_rebuild_merge_and_parametrize():
    _require_cuda()
    rebuild_base, _, rebuild, _, = _linear_pair(
        lora_dim=1, alpha=2, factor=4, use_scalar=False
    )
    with torch.no_grad():
        rebuild.lokr_w2_a.normal_()
        rebuild.lokr_w2_b.normal_()
        expected = rebuild_base.weight + rebuild.get_weight(rebuild.shape)
        merged, _ = rebuild.get_merged_weight(shape=rebuild.shape, device=CUDA)
    torch.testing.assert_close(merged, expected, atol=1e-5, rtol=1e-5)

    x = torch.randn(3, 16, device=CUDA)
    rebuild_base.weight.data.copy_(rebuild_base.weight.data)
    rebuild_base.eval()
    with torch.no_grad():
        out_rebuild = rebuild_base(x)
    torch.testing.assert_close(out_rebuild, F.linear(x, expected), atol=1e-5, rtol=1e-5)

    param_base = nn.Linear(16, 16, bias=False, device=CUDA)
    param = LokrModule.parametrize(
        param_base, "weight", lora_dim=1, alpha=2, factor=4, use_scalar=False
    )
    with torch.no_grad():
        param.lokr_w2_a.copy_(rebuild.lokr_w2_a)
        param.lokr_w2_b.copy_(rebuild.lokr_w2_b)
        param.lokr_w1.copy_(rebuild.lokr_w1)
    original = param_base.parametrizations.weight.original
    expected_param, _ = param.get_merged_weight(
        multiplier=param.multiplier, shape=original.shape, device=CUDA
    )
    torch.testing.assert_close(param_base.weight, expected_param, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_lokr_low_rank_conv_bypass_matches_rebuild_with_nondefault_params(dim):
    _require_cuda()
    rebuild_base, bypass_base, rebuild, bypass, x = _conv_pair(
        dim, lora_dim=1, alpha=2, factor=2, use_tucker=False
    )
    with torch.no_grad():
        for name in ("lokr_w1", "lokr_w2_a", "lokr_w2_b"):
            if hasattr(rebuild, name):
                getattr(rebuild, name).normal_(std=0.1)
        _copy_adapter(rebuild, bypass)
        out_rebuild = rebuild_base(x)
        out_bypass = bypass_base(x)
    torch.testing.assert_close(out_bypass, out_rebuild, atol=2e-3, rtol=2e-4)


def test_lokr_tucker_conv_bypass_uses_transposed_output_factor_and_1x1_ops():
    _require_cuda()
    rebuild_base, bypass_base, rebuild, bypass, x = _conv_pair(
        2, lora_dim=1, alpha=2, factor=2, use_tucker=True
    )
    assert rebuild.tucker
    with torch.no_grad():
        rebuild.lokr_w1.normal_(std=0.1)
        rebuild.lokr_w2_a.normal_(std=0.1)
        rebuild.lokr_w2_b.normal_(std=0.1)
        rebuild.lokr_t2.normal_(std=0.1)
        _copy_adapter(rebuild, bypass)
        out_rebuild = rebuild_base(x)
        out_bypass = bypass_base(x)
    torch.testing.assert_close(out_bypass, out_rebuild, atol=2e-5, rtol=2e-5)


def test_lokr_conv_svd_segment_initialization_is_kernel_aware():
    _require_cuda()
    conv = nn.Conv2d(8, 16, 3, padding=1, bias=False, device=CUDA)
    module = LokrModule(
        "svd", conv, lora_dim=4, alpha=4, factor=-1,
        use_tucker=False, svd_segment="top",
    ).to(CUDA)
    assert module.get_weight(module.shape).shape == conv.weight.shape
    assert torch.isfinite(module.get_weight(module.shape)).all()


def test_lokr_svd_segment_accounts_for_trainable_scalar():
    _require_cuda()
    conv = nn.Conv2d(8, 16, 3, padding=1, bias=False, device=CUDA)
    original = conv.weight.detach().clone()
    module = LokrModule(
        "svd_scalar", conv, lora_dim=4, alpha=4, factor=-1,
        use_tucker=False, use_scalar=True, svd_segment="top",
    ).to(CUDA)
    module.eval()
    reconstructed = conv.weight + module.get_weight(module.shape) * module.scalar
    torch.testing.assert_close(reconstructed, original, atol=2e-5, rtol=2e-5)


def test_lokr_dora_forces_rebuild_even_when_constructed_directly():
    _require_cuda()
    module = LokrModule(
        "dora", nn.Linear(16, 16, bias=False, device=CUDA),
        lora_dim=1, alpha=2, factor=4,
        weight_decompose=True, bypass_mode=True,
    ).to(CUDA)
    assert module.bypass_mode is False


def test_lokr_rank_dropout_is_supported_in_both_forward_modes():
    _require_cuda()
    base_rebuild = nn.Linear(16, 16, bias=False, device=CUDA)
    base_bypass = copy.deepcopy(base_rebuild)
    rebuild = LokrModule(
        "rank_rebuild", base_rebuild, lora_dim=1, alpha=2, factor=4,
        rank_dropout=0.5, rank_dropout_scale=True,
    ).to(CUDA)
    bypass = LokrModule(
        "rank_bypass", base_bypass, lora_dim=1, alpha=2, factor=4,
        rank_dropout=0.5, rank_dropout_scale=True, bypass_mode=True,
    ).to(CUDA)
    _copy_adapter(rebuild, bypass)
    rebuild.train()
    bypass.train()
    rebuild.apply_to()
    bypass.apply_to()
    x = torch.randn(2, 16, device=CUDA)
    assert torch.isfinite(base_rebuild(x)).all()
    assert torch.isfinite(base_bypass(x)).all()


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(lora_dim=1, alpha=1, factor=4, use_tucker=False),
        dict(lora_dim=4, alpha=2, factor=4, use_tucker=False),
        dict(lora_dim=1, alpha=1, factor=2, use_tucker=True),
        dict(
            lora_dim=4, alpha=2, factor=8, use_tucker=False,
            decompose_both=True, unbalanced_factorization=True,
        ),
    ],
)
def test_lokr_state_dict_factory_round_trip(kwargs):
    _require_cuda()
    base = nn.Conv2d(4, 8, 3, padding=1, bias=False, device=CUDA) if kwargs["use_tucker"] else nn.Linear(64, 128, bias=False, device=CUDA)
    module = LokrModule("roundtrip", base, **kwargs).to(CUDA)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.normal_()
    state = module.state_dict()
    params = [state.get(key) for key in LokrModule.weight_list]
    loaded = LokrModule.make_module_from_state_dict("loaded", base, *params).to(CUDA)
    loaded.eval()
    module.eval()
    assert loaded.use_w1 == module.use_w1
    assert loaded.use_w2 == module.use_w2
    assert loaded.tucker == module.tucker
    assert loaded.unbalanced_factorization == module.unbalanced_factorization
    torch.testing.assert_close(
        loaded.get_weight(loaded.shape), module.get_weight(module.shape),
        atol=1e-5, rtol=1e-5,
    )

    # The optional metadata is not a live buffer, so strict loading must also
    # accept the serialized representation without reporting it as unexpected.
    if module.unbalanced_factorization:
        fresh = LokrModule("fresh", base, **kwargs).to(CUDA)
        fresh.load_state_dict(state)
