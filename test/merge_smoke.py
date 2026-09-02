"""Post-merge smoke test: validates the merged (upstream-kernels + fork-features)
architecture end-to-end on CUDA. Deletes itself conceptually via __main__ guard."""

import torch

import lycoris
from lycoris import (
    LoConModule,
    LokrModule,
    LohaModule,
    TLoraModule,
    TSMModule,
    OrthoLoRAModule,
    GLoRAModule,
    compute_timestep_mask,
    compute_timestep_mask_batch,
    set_timestep_mask,
    clear_timestep_mask,
)
from lycoris.config import list_builtin_presets
from lycoris.config_sdk import PresetConfig
from lycoris.wrapper import create_lycoris

DEVICE = "cuda"
FAILURES = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f" ({detail})" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


def smoke_module(module_cls, org, name, extra_kwargs=None, algo="lora"):
    kwargs = dict(multiplier=1.0, lora_dim=4, alpha=4)
    if extra_kwargs:
        kwargs.update(extra_kwargs)
    net = module_cls(name, org, **kwargs)
    net.to(DEVICE)
    x = torch.randn(2, 8, 32, 32, device=DEVICE)
    y = net(x)
    check(f"{algo} forward shape", y.shape == org(x).shape)
    # state dict round trip via custom_state_dict
    sd = net.custom_state_dict()
    check(f"{algo} custom_state_dict non-empty", len(sd) > 0)
    # merged weight path
    merged, _ = net.get_merged_weight(multiplier=1.0)
    check(f"{algo} get_merged_weight", torch.isfinite(merged).all().item())
    return net


def main():
    check("CUDA available", torch.cuda.is_available())
    lin = torch.nn.Linear(32, 32).to(DEVICE)
    conv = torch.nn.Conv2d(8, 8, 3, padding=1).to(DEVICE)

    # --- Core algorithms (upstream kernel dispatch + fork features) ---
    smoke_module(LoConModule, lin, "lc_lin", algo="locon-linear")
    smoke_module(LoConModule, conv, "lc_conv", algo="locon-conv")
    smoke_module(LokrModule, lin, "lk_lin", algo="lokr-linear")
    smoke_module(LokrModule, conv, "lk_conv", algo="lokr-conv")
    smoke_module(LohaModule, lin, "lh_lin", algo="loha")

    # --- Fork feature: runtime orthogonalization ---
    net = LoConModule(
        "ortho", lin, multiplier=1.0, lora_dim=4, alpha=4,
        orthogonalize=True, use_scalar=False,
    )
    net.to(DEVICE)
    check("orthogonalize forces init", net.use_orthogonal_init)
    check("orthogonalize forces scalar param", isinstance(net.scalar, torch.nn.Parameter))
    x = torch.randn(2, 32, device=DEVICE)
    y = net(x)
    check("orthogonalize forward", torch.isfinite(y).all().item())

    # --- Fork feature: T-LoRA incl. batched masks ---
    tl = TLoraModule("tlora", lin, multiplier=1.0, lora_dim=4, alpha=4)
    tl.to(DEVICE)
    mask = compute_timestep_mask(500, 1000, 4)
    set_timestep_mask(mask)
    x = torch.randn(2, 32, device=DEVICE)
    y = tl(x)
    check("tlora forward with mask", torch.isfinite(y).all().item())
    bmask = compute_timestep_mask_batch(torch.tensor([100, 900], device=DEVICE), 1000, 4)
    check("tlora batched mask shape", bmask.shape == (2, 4))
    set_timestep_mask(bmask)
    y2 = tl(x)
    check("tlora forward with batched mask", torch.isfinite(y2).all().item())
    clear_timestep_mask()

    # --- Fork module: TSM + OrthoLoRA registry ---
    check("tsm registered", TSMModule is not None)
    o = OrthoLoRAModule("ortho_lora", lin, multiplier=1.0, lora_dim=4, alpha=4)
    o.to(DEVICE)
    yo = o(torch.randn(2, 32, device=DEVICE))
    check("ortholora forward", torch.isfinite(yo).all().item())

    # --- Upstream feature: high precision merge context ---
    from lycoris.modules.base import LycorisBaseModule
    check("base has _prepare_merge_context", hasattr(LycorisBaseModule, "_prepare_merge_context"))
    check("base has compile_forward", hasattr(LycorisBaseModule, "compile_forward"))

    # --- Wrapper-level network with preset ---
    class MiniModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blk = torch.nn.Sequential(
                torch.nn.Linear(32, 64), torch.nn.GELU(), torch.nn.Linear(64, 32)
            )

        def forward(self, x):
            return self.blk(x)

    model = MiniModel().to(DEVICE)
    network = create_lycoris(model, multiplier=1.0, linear_dim=4, linear_alpha=4, algo="lokr")
    check("network created modules", len(network.loras) > 0, f"got {len(network.loras)}")

    # --- Presets: union (upstream + fork Anima presets) with extras round-trip ---
    presets = list_builtin_presets()
    check("anima preset restored", "anima" in presets)
    check("anima-inpaint preset restored", "anima-inpaint" in presets)
    anima_inpaint = presets["anima-inpaint"]
    d = anima_inpaint.to_dict()
    check("anima-inpaint extras round-trip", "network_reg_dims" in d and "include_patterns" in d)

    # --- config_sdk validation of fork keys ---
    cfg = PresetConfig.from_dict(
        {"unet_target_module": ["Block"], "network_reg_dims": {"a.*": 8}}
    )
    check("config_sdk accepts fork keys", cfg.extra.get("network_reg_dims") == {"a.*": 8})
    check("config_sdk fork keys round-trip", "network_reg_dims" in cfg.to_dict())

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURES: {FAILURES}")
        raise SystemExit(1)
    print("ALL MERGE SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
