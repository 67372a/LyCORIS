"""CUDA regression tests for NoRA on LoCon and its RaLoRA variant."""

import pytest
import torch
import torch.nn as nn

from lycoris.modules.locon import LoConModule, RaLoRAModule
from lycoris.wrapper import LycorisNetwork


DEVICE = torch.device("cuda")


def _make_locon(base, **kwargs):
    return LoConModule("nora_test", base, lora_dim=4, alpha=4, **kwargs).to(DEVICE)


def _set_nonzero_factors(module, seed=0):
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    with torch.no_grad():
        module.lora_down.weight.copy_(
            torch.randn(module.lora_down.weight.shape, device=DEVICE, generator=generator)
        )
        module.lora_up.weight.copy_(
            torch.randn(module.lora_up.weight.shape, device=DEVICE, generator=generator)
        )
        if module.tucker:
            module.lora_mid.weight.copy_(
                torch.randn(module.lora_mid.weight.shape, device=DEVICE, generator=generator)
            )


def _manual_nora(weight):
    norm = torch.linalg.vector_norm(weight.float(), ord=2, dim=0, keepdim=True)
    return weight / (norm.to(weight.dtype) + 1e-6)


def test_defaults_and_init_only_normalization():
    default = _make_locon(nn.Linear(12, 8, bias=False).to(DEVICE))
    assert not default.use_nora
    assert not default.use_nora_init

    init_only = _make_locon(
        nn.Linear(12, 8, bias=False).to(DEVICE), nora_init=True
    )
    assert init_only.use_nora_init
    assert not init_only.use_nora
    norms = torch.linalg.vector_norm(init_only.lora_down.weight.float(), dim=0)
    torch.testing.assert_close(norms, torch.ones_like(norms), atol=3e-5, rtol=3e-5)

    _set_nonzero_factors(init_only)
    before = init_only.make_weight().detach().clone()
    with torch.no_grad():
        init_only.lora_down.weight.mul_(3.0)
    after = init_only.make_weight().detach()
    torch.testing.assert_close(after, before * 3.0, atol=2e-5, rtol=2e-5)


def test_full_nora_normalizes_dynamically_and_preserves_gradients():
    module = _make_locon(nn.Linear(12, 8, bias=False).to(DEVICE), nora=True)
    assert module.use_nora
    assert module.use_nora_init
    _set_nonzero_factors(module, seed=5)

    raw_down = module.lora_down.weight.detach().clone()
    expected = module.lora_up.weight @ _manual_nora(raw_down)
    torch.testing.assert_close(module.make_weight(), expected, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(module.lora_down.weight, raw_down)

    first = module.make_weight().detach()
    with torch.no_grad():
        module.lora_down.weight.mul_(3.0)
    second = module.make_weight().detach()
    torch.testing.assert_close(second, first, atol=2e-5, rtol=2e-5)

    module.zero_grad(set_to_none=True)
    module.make_weight().square().sum().backward()
    assert module.lora_down.weight.grad is not None
    assert torch.isfinite(module.lora_down.weight.grad).all()
    assert module.lora_down.weight.grad.norm() > 0


def test_nora_composes_with_runtime_orthogonalization():
    module = _make_locon(
        nn.Linear(12, 8, bias=False).to(DEVICE),
        nora=True,
        orthogonalize=True,
    )
    _set_nonzero_factors(module, seed=9)
    module.train()
    effective_down = module._maybe_nora_down(module.lora_down.weight)
    norms = torch.linalg.vector_norm(effective_down.float(), dim=0)
    torch.testing.assert_close(norms, torch.ones_like(norms), atol=3e-5, rtol=3e-5)
    assert torch.isfinite(module.make_weight()).all()


@pytest.mark.parametrize("kind", ["linear", "conv", "tucker"])
def test_rebuild_bypass_merge_and_export_parity(kind):
    if kind == "linear":
        base = nn.Linear(12, 8).to(DEVICE)
        x = torch.randn(2, 12, device=DEVICE)
    else:
        base = nn.Conv2d(4, 8, 3, padding=1).to(DEVICE)
        x = torch.randn(2, 4, 9, 9, device=DEVICE)

    module = _make_locon(base, nora=True, use_tucker=(kind == "tucker"))
    _set_nonzero_factors(module, seed=11)
    module.train()

    rebuilt = module._forward_rebuild_core(x, base.weight, base.bias)
    bypass = base(x) + module._bypass_forward_diff_single(x)
    torch.testing.assert_close(rebuilt, bypass, atol=2e-5, rtol=2e-5)

    merged_weight, _ = module.get_merged_weight(multiplier=1.0)
    merged = module._call_op(x, merged_weight, base.bias)
    torch.testing.assert_close(rebuilt, merged, atol=2e-5, rtol=2e-5)
    state = module.custom_state_dict()
    torch.testing.assert_close(
        state["lora_down.weight"], _manual_nora(module.lora_down.weight)
    )


def test_compiled_full_nora_forward_matches_eager():
    dim = 16
    torch.manual_seed(41)
    base_eager = nn.Linear(dim, dim).to(DEVICE)
    eager = _make_locon(base_eager, nora=True, use_scalar=True)
    eager.apply_to()

    torch.manual_seed(41)
    base_compiled = nn.Linear(dim, dim).to(DEVICE)
    base_compiled.load_state_dict(base_eager.state_dict())
    compiled = _make_locon(base_compiled, nora=True, use_scalar=True)
    compiled.apply_to()
    compiled.compile_forward(mode="default", dynamic=True, fullgraph=False)

    x = torch.randn(2, dim, device=DEVICE)
    with torch.no_grad():
        eager_output = base_eager(x)
        compiled_output = base_compiled(x)
    torch.testing.assert_close(eager_output, compiled_output, atol=1e-5, rtol=1e-5)


def test_nora_initialization_preserves_pissa_initial_weight():
    base = nn.Linear(16, 12, bias=False).to(DEVICE)
    original_weight = base.weight.detach().clone()
    module = LoConModule(
        "nora_pissa_test",
        base,
        lora_dim=4,
        alpha=4,
        svd_segment="top",
        nora=True,
    ).to(DEVICE)
    merged, _ = module.get_merged_weight(multiplier=1.0)
    torch.testing.assert_close(merged, original_weight, atol=3e-5, rtol=3e-5)


def test_ralora_block_factors_use_dynamic_nora():
    module = RaLoRAModule(
        "nora_ralora_test",
        nn.Linear(16, 16, bias=False).to(DEVICE),
        lora_dim=4,
        alpha=4,
        nora=True,
    ).to(DEVICE)
    module.dynamic_init(avg_rank=4, rank=4, n_split=2)
    module = module.to(DEVICE)

    assert len(module._mini_lora_A) == 2
    for factor in module._mini_lora_A:
        norms = torch.linalg.vector_norm(factor.float(), dim=0)
        torch.testing.assert_close(norms, torch.ones_like(norms), atol=3e-5, rtol=3e-5)

    with torch.no_grad():
        for factor in module._mini_lora_B:
            factor.normal_()
    first = module.make_weight(device=DEVICE).detach()
    with torch.no_grad():
        module._mini_lora_A[0].mul_(2.0)
    second = module.make_weight(device=DEVICE).detach()
    torch.testing.assert_close(second, first, atol=2e-5, rtol=2e-5)


def test_olora_multitask_down_factors_use_dynamic_nora():
    LoConModule.reset_olora_registry()
    try:
        module = _make_locon(
            nn.Linear(12, 8, bias=False).to(DEVICE), nora=True, olora=True
        )
        module.add_task(1)
        module = module.to(DEVICE)
        with torch.no_grad():
            for down, up in zip(module.lora_down_modules, module.lora_up_modules):
                down.weight.normal_()
                up.weight.normal_()
        first = module.make_weight(device=DEVICE).detach()
        with torch.no_grad():
            module.lora_down_modules[0].weight.mul_(2.0)
        second = module.make_weight(device=DEVICE).detach()
        torch.testing.assert_close(second, first, atol=2e-5, rtol=2e-5)
    finally:
        LoConModule.reset_olora_registry()


@pytest.mark.parametrize("suffix", [".safetensors", ".pt"])
def test_nora_configuration_round_trips_with_network_weights(tmp_path, suffix):
    source = LycorisNetwork(
        nn.Identity().to(DEVICE), init_only=True, nora=True
    )
    source.add_module(
        "adapter", _make_locon(nn.Linear(12, 8, bias=False).to(DEVICE), nora=True)
    )
    checkpoint = tmp_path / f"nora{suffix}"
    source.save_weights(str(checkpoint), torch.float32, {})

    target = LycorisNetwork(nn.Identity().to(DEVICE), init_only=True)
    target_module = _make_locon(nn.Linear(12, 8, bias=False).to(DEVICE))
    target.add_module("adapter", target_module)
    target.load_weights(str(checkpoint))

    assert target.nora
    assert target.nora_init
    assert target_module.use_nora
    assert target_module.use_nora_init
