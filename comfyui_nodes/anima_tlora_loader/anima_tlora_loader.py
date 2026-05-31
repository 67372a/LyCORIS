"""
Anima T-LoRA Loader for ComfyUI
================================
Loads LyCORIS T-LoRA checkpoints trained with Machina's Anima fork,
supporting both Anima/MMDiT (SD3, Anima) and SDXL (classic UNet) architectures.

Uses ComfyUI's BypassInjectionManager for true per-step timestep-dependent
rank masking during inference — not a static approximation.

Architecture support:
  - Anima / MMDiT (SD3, Anima):  ``lora_unet_blocks_N_*`` checkpoint keys
  - SDXL / classic UNet (SDXL, SD1.5, SD2):  ``lora_unet_input_blocks_*`` /
    ``lora_unet_middle_block_*`` / ``lora_unet_output_blocks_*`` keys

Installation:
1. Copy this file and __init__.py to ComfyUI/custom_nodes/tlora/
2. Restart ComfyUI
3. Search for "Load Anima T-LoRA" in the node menu under loaders/T-LoRA

Based on bghira's ComfyUI-T-LoRA implementation, adapted for Anima LyCORIS
checkpoint format.
"""

import logging
import math
import threading
import uuid
from collections import defaultdict

import folder_paths
import torch
import torch.nn.functional as F

import comfy.lora
import comfy.patcher_extension
import comfy.utils
import comfy.weight_adapter


# ── Thread-local mask state ────────────────────────────────────────────────────

_TLORA_STATE = threading.local()
_ANIMA_TLORA_CONFIG_KEY = "anima_tlora_runtime_config"
_ANIMA_TLORA_WRAPPER_KEY = "anima_tlora_predict_noise_wrapper"
_ANIMA_TLORA_INJECTION_PREFIX = "anima_tlora_bypass"


def _set_mask(mask):
    _TLORA_STATE.mask = mask


def _get_mask():
    return getattr(_TLORA_STATE, "mask", None)


def _clear_mask():
    for attr in ("mask", "debug_step", "adapter_log_emitted"):
        if hasattr(_TLORA_STATE, attr):
            delattr(_TLORA_STATE, attr)


# ── Rank computation ───────────────────────────────────────────────────────────

def _compute_active_rank(timestep, max_timestep, max_rank, min_rank, alpha=1.0):
    """
    r(t) = floor(((max_t - t) / max_t)^alpha * (max_rank - min_rank)) + min_rank
    At t=max_timestep: r = min_rank  (structural protection)
    At t=0:            r = max_rank  (full style detail)
    """
    min_rank = max(0, min(max_rank, min_rank))
    if max_timestep <= 0:
        return min_rank
    t = max(0.0, min(float(max_timestep), float(timestep)))
    progress = ((float(max_timestep) - t) / float(max_timestep)) ** float(alpha)
    progress = max(0.0, min(1.0, progress))
    r = int(progress * (max_rank - min_rank)) + min_rank
    return max(0, min(max_rank, r))


def _rank_mask_tensor(active_rank, max_rank, device, dtype):
    mask = torch.zeros((1, max_rank), device=device, dtype=dtype)
    if active_rank > 0:
        mask[:, :active_rank] = 1.0
    return mask


def _prepare_rank_mask(rank, reference):
    """Get current thread-local mask, reshaped to (1, rank)."""
    mask = _get_mask()
    if mask is None:
        return torch.ones((1, rank), device=reference.device, dtype=reference.dtype)
    mask = mask.to(device=reference.device, dtype=reference.dtype)
    if mask.ndim == 1:
        mask = mask.view(1, -1)
    if mask.shape[1] < rank:
        pad = torch.ones((1, rank - mask.shape[1]), device=mask.device, dtype=mask.dtype)
        mask = torch.cat([mask, pad], dim=1)
    elif mask.shape[1] > rank:
        mask = mask[:, :rank]
    return mask


# ── T-LoRA adapter (orthogonal, Anima LyCORIS format) ─────────────────────────

class _AnimaTLoraAdapter:
    """
    T-LoRA forward adapter for Anima LyCORIS checkpoint format (linear layers).

    Checkpoint keys per module:
      q_layer.weight  (rank, in)   -- Q down projection
      p_layer.weight  (out, rank)  -- P up projection
      lambda_layer    (1, rank)    -- learnable singular values
      base_q          (rank, in)   -- base state Q
      base_p          (out, rank)  -- base state P
      base_lambda     (1, rank)    -- base state lambda
      alpha           scalar       -- alpha scaling

    Forward:
      lam_masked      = lambda_layer * mask
      lam_base_masked = base_lambda  * mask
      curr  = P @ diag(lam_masked)      @ Q @ x
      base  = P_base @ diag(lam_base_masked) @ Q_base @ x
      delta = (curr - base) * scale * multiplier
    """

    def __init__(self, q, p, lam, base_q, base_p, base_lam, scale):
        self.rank = int(q.shape[0])
        self.q        = q
        self.p        = p
        self.lam      = lam
        self.base_q   = base_q
        self.base_p   = base_p
        self.base_lam = base_lam
        self.scale    = float(scale)
        self.multiplier = 1.0

    def h(self, x, _base_out):
        orig_dtype = x.dtype
        dtype = self.q.dtype
        x_c = x.to(dtype)
        dev = x_c.device

        q        = self.q.to(device=dev, dtype=dtype)
        p        = self.p.to(device=dev, dtype=dtype)
        lam      = self.lam.to(device=dev, dtype=x_c.dtype)
        base_q   = self.base_q.to(device=dev, dtype=dtype)
        base_p   = self.base_p.to(device=dev, dtype=dtype)
        base_lam = self.base_lam.to(device=dev, dtype=x_c.dtype)

        mask = _prepare_rank_mask(self.rank, x_c)

        lam_m      = lam      * mask
        lam_base_m = base_lam * mask

        curr_hidden = F.linear(x_c, q)
        curr_scaled = curr_hidden * lam_m
        curr_out    = F.linear(curr_scaled, p)

        base_hidden = F.linear(x_c, base_q)
        base_scaled = base_hidden * lam_base_m
        base_out    = F.linear(base_scaled, base_p)

        result = (curr_out - base_out) * self.scale
        return result.to(orig_dtype) * float(self.multiplier)

    def g(self, y):
        return y


class _AnimaTLoraAdapterConv(_AnimaTLoraAdapter):
    """
    T-LoRA forward adapter with Conv2d support for SDXL classic UNet layers.

    Extends ``_AnimaTLoraAdapter`` with a convolution-aware forward path.

    For Conv2d target modules, the checkpoint stores Q/P as 2D matrices
    (``Linear`` projection weights).  During forward we reshape them to 4D
    1×1 or k×k convolution kernels and use ``F.conv2d``.

    The BypassForwardHook sets ``is_conv``, ``kw_dict`` (stride/padding/
    dilation/groups), ``kernel_size``, and ``in_channels`` on this adapter.
    """

    def h(self, x, _base_out):
        orig_dtype = x.dtype
        dtype = self.q.dtype
        x_c = x.to(dtype)
        dev = x_c.device

        q        = self.q.to(device=dev, dtype=dtype)
        p        = self.p.to(device=dev, dtype=dtype)
        lam      = self.lam.to(device=dev, dtype=x_c.dtype)
        base_q   = self.base_q.to(device=dev, dtype=dtype)
        base_p   = self.base_p.to(device=dev, dtype=dtype)
        base_lam = self.base_lam.to(device=dev, dtype=x_c.dtype)

        mask = _prepare_rank_mask(self.rank, x_c)

        is_conv = getattr(self, 'is_conv', False)
        if not is_conv:
            # ── Linear forward (delegate to parent) ──────────────────
            lam_m      = lam      * mask
            lam_base_m = base_lam * mask

            curr_hidden = F.linear(x_c, q)
            curr_scaled = curr_hidden * lam_m
            curr_out    = F.linear(curr_scaled, p)

            base_hidden = F.linear(x_c, base_q)
            base_scaled = base_hidden * lam_base_m
            base_out    = F.linear(base_scaled, base_p)

            result = (curr_out - base_out) * self.scale
            return result.to(orig_dtype) * float(self.multiplier)

        # ── Conv2d forward ───────────────────────────────────────────
        kw = getattr(self, 'kw_dict', {})
        stride  = kw.get('stride',  (1, 1))
        padding = kw.get('padding', (0, 0))
        dilation = kw.get('dilation', (1, 1))
        groups  = kw.get('groups',  1)
        target_ks = getattr(self, 'kernel_size', (1, 1))
        C_in    = x_c.shape[1]
        C_out   = p.shape[0]  # output channels from P projection

        # ── Resolve effective kernel size from q's stored dimensions ──
        # q is (rank, in_features).  For a conv layer, in_features should be
        # C_in * kH * kW (or C_in/groups * kH * kW for grouped convs).
        # However, training code may store q with in_features == C_in
        # (1×1 effective kernel), even for k×k target layers.
        # We derive the effective kernel from q's actual size so we never
        # attempt a .view() with the wrong number of elements.
        q_in_features = int(q.shape[1])
        target_kernel_area = C_in * target_ks[0] * target_ks[1]

        if q_in_features == target_kernel_area:
            # q dimensions match the target kernel → use target conv params
            effective_ks = target_ks
            eff_padding  = padding
            eff_dilation = dilation
        elif q_in_features == C_in:
            # q was stored as a 1×1 conv (in_features == C_in)
            # Use 1×1 kernel with the target's stride; padding/dilation
            # are zeroed because a 1×1 kernel doesn't need them.
            effective_ks = (1, 1)
            eff_padding  = (0, 0)
            eff_dilation = (1, 1)
        else:
            # Try to infer kernel from q's in_features dimension.
            # in_features should be divisible by C_in (or C_in/groups).
            k_area = q_in_features // C_in
            if q_in_features % C_in == 0 and k_area > 0:
                k = int(k_area ** 0.5)
                if k * k == k_area:
                    effective_ks = (k, k)
                else:
                    effective_ks = (k_area, 1)
                # Adjust padding to preserve spatial size with the
                # effective kernel (assumes odd kernel, same H/W).
                if effective_ks[0] != target_ks[0] or effective_ks[1] != target_ks[1]:
                    eff_padding = (0, 0) if effective_ks == (1, 1) else padding
                else:
                    eff_padding = padding
                eff_dilation = (1, 1) if effective_ks == (1, 1) else dilation
            else:
                # Last resort: treat as 1×1
                effective_ks = (1, 1)
                eff_padding  = (0, 0)
                eff_dilation = (1, 1)

        # --- curr path: Q -> λ -> P ---
        # Q: (rank, in_features) → (rank, C_in, kH, kW)
        q_4d = q.view(self.rank, C_in, *effective_ks).contiguous()
        q_out = F.conv2d(x_c, q_4d, stride=stride, padding=eff_padding,
                         dilation=eff_dilation)
        # q_out: (B, rank, H_out, W_out)

        lam_m = lam * mask  # (1, rank)
        q_scaled = q_out * lam_m.view(1, self.rank, 1, 1)

        # P: (C_out, rank) → (C_out, rank, 1, 1)
        p_4d = p.view(C_out, self.rank, 1, 1).contiguous()
        curr_out = F.conv2d(q_scaled, p_4d, stride=1, padding=0)
        # curr_out: (B, C_out, H_out, W_out)

        # --- base path: Q_base -> λ_base -> P_base ---
        base_q_4d = base_q.view(self.rank, C_in, *effective_ks).contiguous()
        base_q_out = F.conv2d(x_c, base_q_4d, stride=stride, padding=eff_padding,
                              dilation=eff_dilation)

        lam_base_m = base_lam * mask
        base_q_scaled = base_q_out * lam_base_m.view(1, self.rank, 1, 1)

        base_p_4d = base_p.view(C_out, self.rank, 1, 1).contiguous()
        base_out = F.conv2d(base_q_scaled, base_p_4d, stride=1, padding=0)

        result = (curr_out - base_out) * self.scale
        return result.to(orig_dtype) * float(self.multiplier)


# ── Checkpoint key parsing (Anima/MMDiT format) ─────────────────────────────

def _group_checkpoint_keys(state_dict):
    """Group checkpoint keys by module name."""
    modules = defaultdict(dict)
    for key, tensor in state_dict.items():
        if key.endswith(".p_layer.weight"):
            module = key[:-len(".p_layer.weight")]
            suffix = "p_layer.weight"
        elif key.endswith(".q_layer.weight"):
            module = key[:-len(".q_layer.weight")]
            suffix = "q_layer.weight"
        else:
            dot_idx = key.rfind(".")
            if dot_idx == -1:
                continue
            module = key[:dot_idx]
            suffix = key[dot_idx + 1:]
        modules[module][suffix] = tensor
    return modules


def _parse_module_to_model_key(checkpoint_key):
    """
    Convert checkpoint module key to ComfyUI model weight key (Anima/MMDiT).

    ``lora_unet_blocks_0_cross_attn_k_proj``
      → ``diffusion_model.blocks.0.cross_attn.k_proj.weight``

    ``lora_unet_blocks_0_mlp_layer1``
      → ``diffusion_model.blocks.0.mlp.layer1.weight``
    """
    if not checkpoint_key.startswith("lora_unet_"):
        return None

    name = checkpoint_key[len("lora_unet_"):]
    parts = name.split("_")

    if len(parts) < 2 or parts[0] != "blocks":
        return None

    block_num = parts[1]
    rest = "_".join(parts[2:])

    layer_map = {
        "cross_attn_k_proj":      "cross_attn.k_proj",
        "cross_attn_q_proj":      "cross_attn.q_proj",
        "cross_attn_v_proj":      "cross_attn.v_proj",
        "cross_attn_output_proj": "cross_attn.output_proj",
        "self_attn_k_proj":       "self_attn.k_proj",
        "self_attn_q_proj":       "self_attn.q_proj",
        "self_attn_v_proj":       "self_attn.v_proj",
        "self_attn_output_proj":  "self_attn.output_proj",
        "mlp_layer1":             "mlp.layer1",
        "mlp_layer2":             "mlp.layer2",
    }

    layer_path = layer_map.get(rest)
    if layer_path is None:
        return None

    return f"diffusion_model.blocks.{block_num}.{layer_path}.weight"


# ── Checkpoint key parsing (SDXL / classic UNet format) ─────────────────────

def _build_sdxl_key_map(model_sd_keys):
    """
    Build a reverse lookup from flattened checkpoint-style keys to
    ComfyUI model weight keys for SDXL classic UNet.

    The model stores keys as::

      diffusion_model.input_blocks.4.1.transformer_blocks.0.attn1.to_q.weight

    The checkpoint flattens dots to underscores::

      lora_unet_input_blocks_4_1_transformer_blocks_0_attn1_to_q.p_layer.weight

    This function creates a ``{flattened_key: model_key}`` mapping so the
    parser can do a simple O(1) dict lookup.
    """
    key_map = {}
    for k in model_sd_keys:
        if k.startswith("diffusion_model.") and k.endswith(".weight"):
            # diffusion_model.input_blocks.4.1.foo.weight
            # → input_blocks_4_1_foo
            stripped = k[len("diffusion_model."):-len(".weight")]
            flattened = stripped.replace(".", "_")
            key_map[flattened] = k
    return key_map


def _parse_sdxl_module_key(checkpoint_key, key_map):
    """
    Convert an SDXL checkpoint module key to a ComfyUI model weight key.

    Strips the ``lora_unet_`` prefix and looks up the rest in ``key_map``
    (built by ``_build_sdxl_key_map``).
    """
    if not checkpoint_key.startswith("lora_unet_"):
        return None
    flattened = checkpoint_key[len("lora_unet_"):]
    return key_map.get(flattened, None)


# ── Sigma / timestep helpers ───────────────────────────────────────────────────

def _extract_sigma_scalar(timestep):
    if isinstance(timestep, torch.Tensor):
        if timestep.numel() == 0:
            return None
        return float(timestep.reshape(-1)[0].detach().float().cpu())
    try:
        return float(timestep)
    except Exception:
        return None


def _sigma_to_timestep(model_sampling, sigma_value):
    if model_sampling is None or sigma_value is None:
        return None
    if not hasattr(model_sampling, "timestep"):
        return sigma_value
    try:
        sigma = torch.tensor([sigma_value], dtype=torch.float32)
        t = model_sampling.timestep(sigma)
        if isinstance(t, torch.Tensor) and t.numel() > 0:
            return float(t.reshape(-1)[0].detach().float().cpu())
    except Exception:
        return sigma_value
    return sigma_value


def _resolve_max_timestep(model_patcher, requested):
    """
    Resolve the maximum timestep for rank scheduling.

    Falls back to 1000 (standard for SDXL, SD1.5, SD2) if model does not
    report one.
    """
    if requested is not None and int(requested) > 0:
        return int(requested)
    model_sampling = model_patcher.get_model_object("model_sampling")
    for attr in ("num_timesteps", "multiplier"):
        if hasattr(model_sampling, attr):
            v = int(getattr(model_sampling, attr))
            if v > 0:
                return v
    return 1000


# ── Predict noise wrapper (true per-step masking) ──────────────────────────────

def _anima_tlora_predict_noise_wrapper(executor, x, timestep, model_options=None, seed=None):
    """
    Wraps the model's predict_noise call to inject the correct rank mask
    for the current denoising timestep before each forward pass.
    """
    model_options = model_options or {}
    model_patcher = executor.class_obj.model_patcher
    config = model_patcher.get_attachment(_ANIMA_TLORA_CONFIG_KEY)
    if config is None:
        return executor(x, timestep, model_options=model_options, seed=seed)

    sigma = _extract_sigma_scalar(timestep)
    model_sampling = model_patcher.get_model_object("model_sampling")
    t_value = _sigma_to_timestep(model_sampling, sigma)
    if t_value is None:
        return executor(x, timestep, model_options=model_options, seed=seed)

    active_rank = _compute_active_rank(
        timestep=t_value,
        max_timestep=config["max_timestep"],
        max_rank=config["max_rank"],
        min_rank=config["min_rank"],
        alpha=config["alpha"],
    )

    config["step_counter"] = int(config.get("step_counter", 0)) + 1
    step = config["step_counter"]

    if config.get("debug", False) and (step == 1 or step % config.get("debug_every", 1) == 0):
        logging.info(
            "[Anima T-LoRA] step=%d sigma=%.4f t=%.1f active_rank=%d/%d",
            step, float(sigma or 0), float(t_value), active_rank, config["max_rank"]
        )

    mask = _rank_mask_tensor(active_rank, config["max_rank"], device=x.device, dtype=x.dtype)
    _set_mask(mask)
    _TLORA_STATE.debug_step = {"step": step}
    _TLORA_STATE.adapter_log_emitted = False

    try:
        return executor(x, timestep, model_options=model_options, seed=seed)
    finally:
        _clear_mask()


def _configure_runtime(model_patcher, max_rank, min_rank, alpha, max_timestep, debug=False, debug_every=1):
    resolved = _resolve_max_timestep(model_patcher, max_timestep)
    config = {
        "max_rank":     int(max_rank),
        "min_rank":     int(min_rank),
        "alpha":        float(alpha),
        "max_timestep": int(resolved),
        "debug":        bool(debug),
        "debug_every":  int(debug_every),
        "step_counter": 0,
    }
    model_patcher.set_attachments(_ANIMA_TLORA_CONFIG_KEY, config)
    model_patcher.remove_wrappers_with_key(
        comfy.patcher_extension.WrappersMP.PREDICT_NOISE,
        _ANIMA_TLORA_WRAPPER_KEY,
    )
    model_patcher.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.PREDICT_NOISE,
        _ANIMA_TLORA_WRAPPER_KEY,
        _anima_tlora_predict_noise_wrapper,
    )


# ── Model loading (Anima / MMDiT) ─────────────────────────────────────────────

def _load_anima_tlora(model, state_dict, strength):
    """
    Parse Anima LyCORIS T-LoRA checkpoint, build adapters,
    and inject them via ComfyUI's BypassInjectionManager.

    Handles ``lora_unet_blocks_N_*`` checkpoint keys (Anima, SD3, MMDiT).
    Uses the standard ``_AnimaTLoraAdapter`` (linear-only).
    """
    modules = _group_checkpoint_keys(state_dict)
    logging.info("[Anima T-LoRA] Found %d modules in checkpoint", len(modules))

    model_lora = model.clone()
    model_sd_keys = set(model_lora.model.state_dict().keys())

    manager = comfy.weight_adapter.BypassInjectionManager()
    loaded = 0
    skipped = 0
    max_rank = 0

    for checkpoint_key, module_data in modules.items():
        required = {"p_layer.weight", "q_layer.weight", "lambda_layer", "base_p", "base_q", "base_lambda", "alpha"}
        if not required.issubset(set(module_data.keys())):
            skipped += 1
            continue

        model_weight_key = _parse_module_to_model_key(checkpoint_key)
        if model_weight_key is None:
            logging.debug("[Anima T-LoRA] Could not parse key: %s", checkpoint_key)
            skipped += 1
            continue

        if model_weight_key not in model_sd_keys:
            logging.debug("[Anima T-LoRA] Key not in model: %s", model_weight_key)
            skipped += 1
            continue

        try:
            alpha_val = module_data["alpha"].float().item()
            rank = module_data["q_layer.weight"].shape[0]
            scale = alpha_val / rank

            adapter = _AnimaTLoraAdapter(
                q        = module_data["q_layer.weight"].detach().clone(),
                p        = module_data["p_layer.weight"].detach().clone(),
                lam      = module_data["lambda_layer"].detach().clone(),
                base_q   = module_data["base_q"].detach().clone(),
                base_p   = module_data["base_p"].detach().clone(),
                base_lam = module_data["base_lambda"].detach().clone(),
                scale    = scale,
            )

            manager.add_adapter(model_weight_key, adapter, strength=strength)
            loaded += 1
            max_rank = max(max_rank, rank)

        except Exception as e:
            logging.warning("[Anima T-LoRA] Skipping %s: %s", checkpoint_key, e)
            skipped += 1

    logging.info("[Anima T-LoRA] Loaded=%d, Skipped=%d, max_rank=%d", loaded, skipped, max_rank)

    if loaded == 0:
        raise ValueError("[Anima T-LoRA] No adapters were created. Check checkpoint format.")

    injections = manager.create_injections(model_lora.model)
    injection_key = f"{_ANIMA_TLORA_INJECTION_PREFIX}_{uuid.uuid4().hex}"
    model_lora.set_injections(injection_key, injections)

    return model_lora, loaded, max_rank


# ── Model loading (SDXL / classic UNet) ────────────────────────────────────────

def _load_anima_tlora_sdxl(model, state_dict, strength):
    """
    Parse an Anima LyCORIS T-LoRA checkpoint trained for SDXL (classic UNet),
    build adapters (with Conv2d support), and inject them via
    ComfyUI's BypassInjectionManager.

    Handles ``lora_unet_input_blocks_*``, ``lora_unet_middle_block_*``, and
    ``lora_unet_output_blocks_*`` checkpoint keys.

    Uses ``_AnimaTLoraAdapterConv`` which supports both linear and conv2d
    target layers.
    """
    modules = _group_checkpoint_keys(state_dict)
    logging.info("[Anima T-LoRA SDXL] Found %d modules in checkpoint", len(modules))

    model_lora = model.clone()
    model_sd_keys = set(model_lora.model.state_dict().keys())

    # Build the reverse key map from the SDXL UNet model state dict
    sdxl_key_map = _build_sdxl_key_map(model_sd_keys)

    manager = comfy.weight_adapter.BypassInjectionManager()
    loaded = 0
    skipped = 0
    max_rank = 0

    for checkpoint_key, module_data in modules.items():
        required = {"p_layer.weight", "q_layer.weight", "lambda_layer", "base_p", "base_q", "base_lambda", "alpha"}
        if not required.issubset(set(module_data.keys())):
            skipped += 1
            continue

        model_weight_key = _parse_sdxl_module_key(checkpoint_key, sdxl_key_map)
        if model_weight_key is None:
            logging.debug("[Anima T-LoRA SDXL] Could not parse key: %s", checkpoint_key)
            skipped += 1
            continue

        if model_weight_key not in model_sd_keys:
            logging.debug("[Anima T-LoRA SDXL] Key not in model: %s", model_weight_key)
            skipped += 1
            continue

        try:
            alpha_val = module_data["alpha"].float().item()
            rank = module_data["q_layer.weight"].shape[0]
            scale = alpha_val / rank

            adapter = _AnimaTLoraAdapterConv(
                q        = module_data["q_layer.weight"].detach().clone(),
                p        = module_data["p_layer.weight"].detach().clone(),
                lam      = module_data["lambda_layer"].detach().clone(),
                base_q   = module_data["base_q"].detach().clone(),
                base_p   = module_data["base_p"].detach().clone(),
                base_lam = module_data["base_lambda"].detach().clone(),
                scale    = scale,
            )

            manager.add_adapter(model_weight_key, adapter, strength=strength)
            loaded += 1
            max_rank = max(max_rank, rank)

        except Exception as e:
            logging.warning("[Anima T-LoRA SDXL] Skipping %s: %s", checkpoint_key, e)
            skipped += 1

    logging.info("[Anima T-LoRA SDXL] Loaded=%d, Skipped=%d, max_rank=%d", loaded, skipped, max_rank)

    if loaded == 0:
        raise ValueError("[Anima T-LoRA SDXL] No adapters were created. Check checkpoint format.")

    injections = manager.create_injections(model_lora.model)
    injection_key = f"{_ANIMA_TLORA_INJECTION_PREFIX}_{uuid.uuid4().hex}"
    model_lora.set_injections(injection_key, injections)

    return model_lora, loaded, max_rank


# ── ComfyUI Node (Anima / MMDiT) ──────────────────────────────────────────────

class AnimaTLoraLoader:
    """
    Loads a LyCORIS T-LoRA checkpoint trained with Machina's Anima fork.

    Architecture: Anima / SD3 / MMDiT (``lora_unet_blocks_N_*`` keys).

    Uses ComfyUI's BypassInjectionManager and predict_noise wrapper for
    true per-step timestep-dependent rank masking during inference.

    At high noise timesteps: few ranks active (structural protection)
    At low noise timesteps:  all ranks active (full style detail)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": -2.0,
                    "max": 2.0,
                    "step": 0.01,
                    "display": "slider"
                }),
                "min_rank": ("INT", {
                    "default": 4,
                    "min": 0,
                    "max": 128,
                    "step": 1,
                    "tooltip": "Minimum active ranks at highest noise timestep. Match your training min_rank."
                }),
                "max_rank": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 128,
                    "step": 1,
                    "tooltip": "0 = infer from checkpoint automatically"
                }),
                "alpha": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.05,
                    "max": 8.0,
                    "step": 0.05,
                    "tooltip": "Controls rank scaling curve. 1.0 = linear. Match your training mask_alpha."
                }),
                "max_timestep": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10000,
                    "step": 1,
                    "tooltip": "0 = infer from model automatically"
                }),
                "debug": ("BOOLEAN", {"default": False}),
                "debug_every": ("INT", {"default": 5, "min": 1, "max": 100}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_tlora"
    CATEGORY = "loaders/T-LoRA"
    DESCRIPTION = "Loads Anima LyCORIS T-LoRA (Anima/SD3/MMDiT) with true per-step timestep-dependent rank masking."

    def load_tlora(self, model, lora_name, strength, min_rank, max_rank, alpha, max_timestep, debug, debug_every):
        lora_path = folder_paths.get_full_path("loras", lora_name)
        if lora_path is None:
            raise ValueError(f"[Anima T-LoRA] Could not find: {lora_name}")

        logging.info("[Anima T-LoRA] Loading: %s", lora_path)
        logging.info("[Anima T-LoRA] strength=%.3f min_rank=%d alpha=%.2f", strength, min_rank, alpha)

        state_dict = comfy.utils.load_torch_file(lora_path, safe_load=True)
        model_out, loaded_count, inferred_rank = _load_anima_tlora(model, state_dict, strength)

        runtime_max_rank = int(max_rank) if int(max_rank) > 0 else int(inferred_rank)
        if runtime_max_rank <= 0:
            raise ValueError("[Anima T-LoRA] Could not determine max_rank. Set it manually.")

        clamped_min_rank = max(0, min(runtime_max_rank, int(min_rank)))

        _configure_runtime(
            model_patcher=model_out,
            max_rank=runtime_max_rank,
            min_rank=clamped_min_rank,
            alpha=alpha,
            max_timestep=max_timestep,
            debug=debug,
            debug_every=debug_every,
        )

        logging.info(
            "[Anima T-LoRA] Ready. %d adapters, max_rank=%d, min_rank=%d",
            loaded_count, runtime_max_rank, clamped_min_rank
        )
        return (model_out,)


# ── ComfyUI Node (SDXL / classic UNet) ────────────────────────────────────────

class AnimaTLoraLoaderSDXL:
    """
    Loads a LyCORIS T-LoRA checkpoint for SDXL (classic UNet).

    Architecture: SDXL, SD1.5, SD2 (``lora_unet_input_blocks_*``,
    ``lora_unet_middle_block_*``, ``lora_unet_output_blocks_*`` keys).

    Supports both linear layers (attention projections, FF layers) and
    Conv2d layers (resnet blocks, down/upsamplers) via
    ``_AnimaTLoraAdapterConv``.

    Uses ComfyUI's BypassInjectionManager and predict_noise wrapper for
    true per-step timestep-dependent rank masking during inference.

    At high noise timesteps: few ranks active (structural protection)
    At low noise timesteps:  all ranks active (full style detail)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": -2.0,
                    "max": 2.0,
                    "step": 0.01,
                    "display": "slider"
                }),
                "min_rank": ("INT", {
                    "default": 4,
                    "min": 0,
                    "max": 128,
                    "step": 1,
                    "tooltip": "Minimum active ranks at highest noise timestep. Match your training min_rank."
                }),
                "max_rank": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 128,
                    "step": 1,
                    "tooltip": "0 = infer from checkpoint automatically"
                }),
                "alpha": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.05,
                    "max": 8.0,
                    "step": 0.05,
                    "tooltip": "Controls rank scaling curve. 1.0 = linear. Match your training mask_alpha."
                }),
                "max_timestep": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10000,
                    "step": 1,
                    "tooltip": "0 = infer from model automatically"
                }),
                "debug": ("BOOLEAN", {"default": False}),
                "debug_every": ("INT", {"default": 5, "min": 1, "max": 100}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_tlora"
    CATEGORY = "loaders/T-LoRA"
    DESCRIPTION = "Loads Anima LyCORIS T-LoRA (SDXL/SD1.5/SD2 classic UNet) with Conv2d support and true per-step timestep-dependent rank masking."

    def load_tlora(self, model, lora_name, strength, min_rank, max_rank, alpha, max_timestep, debug, debug_every):
        lora_path = folder_paths.get_full_path("loras", lora_name)
        if lora_path is None:
            raise ValueError(f"[Anima T-LoRA SDXL] Could not find: {lora_name}")

        logging.info("[Anima T-LoRA SDXL] Loading: %s", lora_path)
        logging.info("[Anima T-LoRA SDXL] strength=%.3f min_rank=%d alpha=%.2f", strength, min_rank, alpha)

        state_dict = comfy.utils.load_torch_file(lora_path, safe_load=True)
        model_out, loaded_count, inferred_rank = _load_anima_tlora_sdxl(model, state_dict, strength)

        runtime_max_rank = int(max_rank) if int(max_rank) > 0 else int(inferred_rank)
        if runtime_max_rank <= 0:
            raise ValueError("[Anima T-LoRA SDXL] Could not determine max_rank. Set it manually.")

        clamped_min_rank = max(0, min(runtime_max_rank, int(min_rank)))

        _configure_runtime(
            model_patcher=model_out,
            max_rank=runtime_max_rank,
            min_rank=clamped_min_rank,
            alpha=alpha,
            max_timestep=max_timestep,
            debug=debug,
            debug_every=debug_every,
        )

        logging.info(
            "[Anima T-LoRA SDXL] Ready. %d adapters, max_rank=%d, min_rank=%d",
            loaded_count, runtime_max_rank, clamped_min_rank
        )
        return (model_out,)


# ── Node registration ──────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "AnimaTLoraLoader":      AnimaTLoraLoader,
    "AnimaTLoraLoaderSDXL":  AnimaTLoraLoaderSDXL,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimaTLoraLoader":      "Load Anima T-LoRA",
    "AnimaTLoraLoaderSDXL":  "Load Anima T-LoRA (SDXL)",
}
