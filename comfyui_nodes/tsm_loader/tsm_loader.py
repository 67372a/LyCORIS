"""
TimeStep Master (TSM) Loader for ComfyUI
==========================================

Loads LyCORIS TSM checkpoints and performs per-step asymmetric mixture of
timestep LoRA experts during inference.

Uses ComfyUI's BypassInjectionManager for true per-step timestep-dependent
expert selection and gating — not a static approximation.

TSM inference (paper Eq. 5):
    ΔW*x = B_{i1}A_{i1}*x + Σ_{j=2}^m G_j ⊙ B_{ij}A_{ij}*x
    where G_j = F(z_t) + ε(t)  (no sigmoid, paper Eq. 6)
    and i_j = ceil(t/T * n_j)  (paper Eq. 7)

The gate depends on BOTH input features z_t AND timestep t, so weights
cannot be pre-merged — per-step adapter computation is required.

Installation:
1. Copy this file and __init__.py to ComfyUI/custom_nodes/tsm_loader/
2. Restart ComfyUI
3. Search for "Load TSM" in the node menu under loaders/TSM

Based on anima_tlora_loader.py pattern, adapted for TSM checkpoint format.
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


# ── Thread-local timestep state ───────────────────────────────────────────────

_TSM_STATE = threading.local()
_TSM_CONFIG_KEY = "tsm_runtime_config"
_TSM_WRAPPER_KEY = "tsm_predict_noise_wrapper"
_TSM_INJECTION_PREFIX = "tsm_bypass"


def _set_timestep(timestep):
    _TSM_STATE.timestep = timestep


def _get_timestep():
    return getattr(_TSM_STATE, "timestep", None)


def _clear_timestep():
    for attr in ("timestep", "debug_step"):
        if hasattr(_TSM_STATE, attr):
            delattr(_TSM_STATE, attr)


# ── Expert index computation (paper Eq. 7) ───────────────────────────────────

def _expert_index(timestep, num_timesteps, n_j):
    """i_j = ceil(t / T * n_j), converted to 0-indexed.

    Paper Eq. 7: i_j = ⌈t/T · n_j⌉, 1-indexed.
    """
    if num_timesteps <= 0 or n_j <= 0:
        return 0
    idx_1indexed = math.ceil(timestep / num_timesteps * n_j)
    idx_1indexed = max(1, min(idx_1indexed, n_j))
    return idx_1indexed - 1


# ── TSM adapter (per-module asymmetric expert mixture) ───────────────────────

class _TSMAdapter:
    """
    TSM forward adapter for a single module.

    Checkpoint keys per module:
        experts.{scale}.{idx}.down.weight   -- A matrix (r × k)
        experts.{scale}.{idx}.up.weight     -- B matrix (d × r), scalar baked in
        router_fc.weight                     -- FC layer weight
        router_fc.bias                       -- FC layer bias
        timestep_embed.weight                -- T × (m-1) embedding
        alpha                                -- scalar

    Forward (paper Eq. 5-6):
        core_delta = B_{i1} A_{i1} x                           (ungated)
        gates = FC(z_t) + embed(t)                              (no sigmoid)
        ctx_delta = Σ G_j * B_{ij} A_{ij} x                    (gated)
        output = (core_delta + ctx_delta) * scale * multiplier
    """

    def __init__(self, experts, router_fc_w, router_fc_b,
                 timestep_embed_w, scale, n_scales, num_timesteps,
                 router_input_mode="input"):
        """
        Args:
            experts: dict of (scale_idx, expert_idx) -> (down_w, up_w)
            router_fc_w: FC layer weight (num_context, in_dim) or (num_context, lora_dim)
            router_fc_b: FC layer bias (num_context,)
            timestep_embed_w: T × (m-1) embedding matrix
            scale: alpha / lora_dim
            n_scales: list of expert counts per scale, e.g. [8, 1]
            num_timesteps: T (total diffusion timesteps)
            router_input_mode: "input" (use z_t) or "bottleneck" (use A*x)
        """
        self.experts = experts
        self.router_fc_w = router_fc_w
        self.router_fc_b = router_fc_b
        self.timestep_embed_w = timestep_embed_w
        self.scale = float(scale)
        self.n_scales = list(n_scales)
        self.num_timesteps = int(num_timesteps)
        self.router_input_mode = router_input_mode
        self.multiplier = 1.0
        self.current_timestep = None  # Set by wrapper before each step

    def h(self, x, _base_out):
        """Compute TSM delta: core expert (ungated) + gated context experts.

        Args:
            x: module input tensor (batch, in_dim) for linear
            _base_out: base model output (unused, TSM computes independently)

        Returns:
            TSM weight delta applied to input x.
        """
        timestep = self.current_timestep
        if timestep is None:
            timestep = self.num_timesteps  # Default: max noise

        orig_dtype = x.dtype
        device = x.device

        # --- Core expert (scale 0, ungated) ---
        core_idx = _expert_index(timestep, self.num_timesteps, self.n_scales[0])
        core_down, core_up = self.experts[(0, core_idx)]
        core_down = core_down.to(device=device, dtype=orig_dtype)
        core_up = core_up.to(device=device, dtype=orig_dtype)
        core_mid = F.linear(x, core_down)
        core_out = F.linear(core_mid, core_up)

        # --- Router gates: G(z_t, t) = F(z_t) + ε(t) ---
        num_context = len(self.n_scales) - 1
        if num_context > 0:
            # Pool input features for router: z_t
            z = x.detach().float()
            if z.dim() > 1:
                z = z.mean(dim=0)  # (in_dim,)

            fc_w = self.router_fc_w.to(device=device, dtype=torch.float32)
            fc_b = self.router_fc_b.to(device=device, dtype=torch.float32)
            embed_w = self.timestep_embed_w.to(device=device, dtype=torch.float32)

            # For bottleneck mode, project z through first core expert's A
            if self.router_input_mode == "bottleneck":
                core_a = self.experts[(0, core_idx)][0]  # down = A
                core_a = core_a.to(device=device, dtype=torch.float32)
                z = F.linear(z, core_a)  # (lora_dim,)

            # Ensure z matches FC input dim
            z = z[: fc_w.shape[1]]
            if z.shape[0] < fc_w.shape[1]:
                z = F.pad(z, (0, fc_w.shape[1] - z.shape[0]))

            fc_out = F.linear(z, fc_w, fc_b)

            # ε(t): extract t-th row (1-indexed → 0-indexed)
            t_idx = min(max(int(timestep) - 1, 0), self.num_timesteps - 1)
            t_embed = embed_w[t_idx]

            gates = fc_out + t_embed  # (num_context,) — no sigmoid

            # --- Context experts (gated) ---
            ctx_out = torch.zeros_like(core_out)
            for ctx_idx in range(num_context):
                scale_idx = ctx_idx + 1  # Skip core scale
                expert_idx = _expert_index(
                    timestep, self.num_timesteps, self.n_scales[scale_idx]
                )
                ctx_down, ctx_up = self.experts[(scale_idx, expert_idx)]
                ctx_down = ctx_down.to(device=device, dtype=orig_dtype)
                ctx_up = ctx_up.to(device=device, dtype=orig_dtype)
                ctx_mid = F.linear(x, ctx_down)
                ctx_delta = F.linear(ctx_mid, ctx_up)
                ctx_out = ctx_out + gates[ctx_idx].to(orig_dtype) * ctx_delta

            result = core_out + ctx_out
        else:
            result = core_out

        return result.to(orig_dtype) * self.scale * float(self.multiplier)

    def g(self, y):
        return y


# ── Checkpoint key parsing ───────────────────────────────────────────────────

def _group_tsm_keys(state_dict):
    """Group TSM checkpoint keys by module name.

    TSM keys are identified by containing '.experts.' in the suffix.

    Returns:
        dict: module_name -> dict of parsed expert/router/config data
    """
    modules = defaultdict(dict)
    for key, tensor in state_dict.items():
        # Find module boundary: split at '.experts.', '.router_fc.', '.timestep_embed.', '.alpha'
        for marker in [".experts.", ".router_fc.", ".timestep_embed.", ".alpha"]:
            if marker in key:
                idx = key.index(marker)
                module = key[:idx]
                suffix = key[idx + 1:]  # Remove leading dot
                modules[module][suffix] = tensor
                break
    return modules


def _parse_expert_structure(module_data):
    """Determine n_scales from expert keys.

    Returns:
        n_scales: list of expert counts per scale, e.g. [8, 1]
    """
    expert_keys = [k for k in module_data if k.startswith("experts.")]
    if not expert_keys:
        return None

    max_scale = -1
    experts_per_scale = defaultdict(int)
    for k in expert_keys:
        parts = k.split(".")
        # experts.{scale}.{idx}.{up|down}.weight
        if len(parts) >= 4:
            scale_idx = int(parts[1])
            expert_idx = int(parts[2])
            max_scale = max(max_scale, scale_idx)
            experts_per_scale[scale_idx] = max(
                experts_per_scale[scale_idx], expert_idx + 1
            )

    if max_scale < 0:
        return None

    return [experts_per_scale[s] for s in range(max_scale + 1)]


# ── Key mapping: LyCORIS → ComfyUI model ────────────────────────────────────

def _parse_module_to_model_key(checkpoint_key):
    """
    Convert LyCORIS kohya-format module key to ComfyUI model weight key.

    Examples:
        lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q
          → diffusion_model.down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.weight

        lora_te1_text_model_encoder_layers_0_self_attn_q_proj
          → diffusion_model.encoder.layers.0.self_attn.q_proj.weight (CLIP)

    Returns model weight key or None if not parseable.
    """
    if checkpoint_key.startswith("lora_unet_"):
        name = checkpoint_key[len("lora_unet_"):]
        model_key = "diffusion_model." + _convert_kohya_unet_path(name)
        return model_key + ".weight"
    elif checkpoint_key.startswith("lora_te"):
        # CLIP text encoder: lora_te{N}_...
        # Try to find the te number and rest
        underscore_idx = checkpoint_key.index("_", len("lora_te"))
        name = checkpoint_key[underscore_idx + 1:]
        model_key = "diffusion_model." + _convert_kohya_te_path(name)
        return model_key + ".weight"
    return None


def _convert_kohya_unet_path(name):
    """Convert kohya UNet path to ComfyUI path.

    lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q
      → down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q
    """
    parts = name.split("_")
    result = []
    i = 0
    while i < len(parts):
        part = parts[i]

        # Known multi-word components
        if part == "down" and i + 1 < len(parts) and parts[i + 1] == "blocks":
            result.append("down_blocks")
            i += 2
        elif part == "up" and i + 1 < len(parts) and parts[i + 1] == "blocks":
            result.append("up_blocks")
            i += 2
        elif part == "mid" and i + 1 < len(parts) and parts[i + 1] == "block":
            result.append("mid_block")
            i += 2
        elif part == "transformer" and i + 1 < len(parts) and parts[i + 1] == "blocks":
            result.append("transformer_blocks")
            i += 2
        elif part == "attentions":
            result.append("attentions")
            i += 1
        elif part == "resnets":
            result.append("resnets")
            i += 1
        elif part == "upsamplers":
            result.append("upsamplers")
            i += 1
        elif part == "downsamplers":
            result.append("downsamplers")
            i += 1
        elif part == "crossattn" and i + 1 < len(parts) and parts[i + 1] == "k":
            # crossattn_k_proj → cross_attn.k_proj (but this is uncommon in kohya)
            result.append("cross_attn")
            i += 1
        elif part == "selfattn" and i + 1 < len(parts):
            result.append("self_attn")
            i += 1
        # Attention projection components
        elif part == "to" and i + 1 < len(parts) and parts[i + 1] == "q":
            result.append("to_q")
            i += 2
        elif part == "to" and i + 1 < len(parts) and parts[i + 1] == "k":
            result.append("to_k")
            i += 2
        elif part == "to" and i + 1 < len(parts) and parts[i + 1] == "v":
            result.append("to_v")
            i += 2
        elif part == "to" and i + 1 < len(parts) and parts[i + 1] == "out":
            result.append("to_out")
            i += 2
        elif part == "to" and i + 1 < len(parts) and parts[i + 1] == "0":
            result.append("to_out.0")
            i += 2
        # Numeric parts (block numbers)
        elif part.isdigit():
            result.append(part)
            i += 1
        # FF layers
        elif part == "ff" and i + 1 < len(parts) and parts[i + 1] == "net":
            result.append("ff.net")
            i += 2
        elif part == "proj" and i + 1 < len(parts) and parts[i + 1] == "in":
            result.append("proj_in")
            i += 2
        elif part == "proj" and i + 1 < len(parts) and parts[i + 1] == "out":
            result.append("proj_out")
            i += 2
        # Conv layers
        elif part == "conv" and i + 1 < len(parts) and parts[i + 1] == "shortcut":
            result.append("conv_shortcut")
            i += 2
        elif part == "conv1":
            result.append("conv1")
            i += 1
        elif part == "conv2":
            result.append("conv2")
            i += 1
        elif part == "time" and i + 1 < len(parts) and parts[i + 1] == "emb" and i + 2 < len(parts) and parts[i + 2] == "proj":
            result.append("time_emb_proj")
            i += 3
        # Norm layers
        elif part == "norm":
            result.append("norm")
            i += 1
        elif part == "group" and i + 1 < len(parts) and parts[i + 1] == "norm":
            result.append("group_norm")
            i += 2
        elif part == "layer" and i + 1 < len(parts) and parts[i + 1] == "norm":
            result.append("layer_norm")
            i += 2
        else:
            result.append(part)
            i += 1

    return ".".join(result)


def _convert_kohya_te_path(name):
    """Convert kohya text encoder path to ComfyUI path.

    lora_te1_text_model_encoder_layers_0_self_attn_q_proj
      → text_model.encoder.layers.0.self_attn.q_proj
    """
    parts = name.split("_")
    result = []
    i = 0
    while i < len(parts):
        part = parts[i]
        if part == "text" and i + 1 < len(parts) and parts[i + 1] == "model":
            result.append("text_model")
            i += 2
        elif part == "encoder" and i + 1 < len(parts) and parts[i + 1] == "layers":
            result.append("encoder.layers")
            i += 2
        elif part == "self" and i + 1 < len(parts) and parts[i + 1] == "attn":
            result.append("self_attn")
            i += 2
        elif part == "mlp" and i + 1 < len(parts) and parts[i + 1] == "fc1":
            result.append("mlp.fc1")
            i += 2
        elif part == "mlp" and i + 1 < len(parts) and parts[i + 1] == "fc2":
            result.append("mlp.fc2")
            i += 2
        elif part == "q" and i + 1 < len(parts) and parts[i + 1] == "proj":
            result.append("q_proj")
            i += 2
        elif part == "k" and i + 1 < len(parts) and parts[i + 1] == "proj":
            result.append("k_proj")
            i += 2
        elif part == "v" and i + 1 < len(parts) and parts[i + 1] == "proj":
            result.append("v_proj")
            i += 2
        elif part == "out" and i + 1 < len(parts) and parts[i + 1] == "proj":
            result.append("out_proj")
            i += 2
        elif part == "layer" and i + 1 < len(parts) and parts[i + 1] == "norm1":
            result.append("layer_norm1")
            i += 2
        elif part == "layer" and i + 1 < len(parts) and parts[i + 1] == "norm2":
            result.append("layer_norm2")
            i += 2
        elif part.isdigit():
            result.append(part)
            i += 1
        else:
            result.append(part)
            i += 1
    return ".".join(result)


# ── Sigma / timestep helpers ─────────────────────────────────────────────────

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


def _resolve_num_timesteps(model_patcher, requested):
    if requested is not None and int(requested) > 0:
        return int(requested)
    model_sampling = model_patcher.get_model_object("model_sampling")
    for attr in ("num_timesteps", "multiplier"):
        if hasattr(model_sampling, attr):
            v = int(getattr(model_sampling, attr))
            if v > 0:
                return v
    return 1000


# ── Predict noise wrapper (per-step timestep injection) ─────────────────────

def _tsm_predict_noise_wrapper(executor, x, timestep, model_options=None, seed=None):
    """
    Wraps the model's predict_noise call to set the current timestep
    on all TSM adapters before each forward pass.
    """
    model_options = model_options or {}
    model_patcher = executor.class_obj.model_patcher
    config = model_patcher.get_attachment(_TSM_CONFIG_KEY)
    if config is None:
        return executor(x, timestep, model_options=model_options, seed=seed)

    sigma = _extract_sigma_scalar(timestep)
    model_sampling = model_patcher.get_model_object("model_sampling")
    t_value = _sigma_to_timestep(model_sampling, sigma)
    if t_value is None:
        return executor(x, timestep, model_options=model_options, seed=seed)

    # Set timestep on all adapters
    adapters = config.get("adapters", [])
    for adapter in adapters:
        adapter.current_timestep = int(round(t_value))

    config["step_counter"] = int(config.get("step_counter", 0)) + 1
    step = config["step_counter"]

    if config.get("debug", False) and (step == 1 or step % config.get("debug_every", 1) == 0):
        logging.info(
            "[TSM] step=%d sigma=%.4f t=%.1f num_adapters=%d",
            step, float(sigma or 0), float(t_value), len(adapters)
        )

    _set_timestep(int(round(t_value)))

    try:
        return executor(x, timestep, model_options=model_options, seed=seed)
    finally:
        _clear_timestep()
        for adapter in adapters:
            adapter.current_timestep = None


def _configure_runtime(model_patcher, num_timesteps, adapters, debug=False, debug_every=1):
    config = {
        "num_timesteps": int(num_timesteps),
        "adapters": adapters,
        "debug": bool(debug),
        "debug_every": int(debug_every),
        "step_counter": 0,
    }
    model_patcher.set_attachments(_TSM_CONFIG_KEY, config)
    model_patcher.remove_wrappers_with_key(
        comfy.patcher_extension.WrappersMP.PREDICT_NOISE,
        _TSM_WRAPPER_KEY,
    )
    model_patcher.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.PREDICT_NOISE,
        _TSM_WRAPPER_KEY,
        _tsm_predict_noise_wrapper,
    )


# ── Model loading ────────────────────────────────────────────────────────────

def _load_tsm(model, state_dict, strength):
    """
    Parse TSM checkpoint, build adapters, and inject via BypassInjectionManager.
    """
    modules = _group_tsm_keys(state_dict)
    logging.info("[TSM] Found %d modules in checkpoint", len(modules))

    model_lora = model.clone()
    model_sd_keys = set(model_lora.model.state_dict().keys())

    manager = comfy.weight_adapter.BypassInjectionManager()
    loaded = 0
    skipped = 0
    all_adapters = []
    num_timesteps = 0

    for checkpoint_key, module_data in modules.items():
        # Check required keys
        n_scales = _parse_expert_structure(module_data)
        if n_scales is None:
            skipped += 1
            continue

        required = {"router_fc.weight", "router_fc.bias",
                     "timestep_embed.weight", "alpha"}
        if not required.issubset(set(module_data.keys())):
            logging.debug("[TSM] Missing required keys for %s", checkpoint_key)
            skipped += 1
            continue

        # Map to model weight key
        model_weight_key = _parse_module_to_model_key(checkpoint_key)
        if model_weight_key is None:
            logging.debug("[TSM] Could not parse key: %s", checkpoint_key)
            skipped += 1
            continue

        if model_weight_key not in model_sd_keys:
            logging.debug("[TSM] Key not in model: %s", model_weight_key)
            skipped += 1
            continue

        try:
            alpha_val = module_data["alpha"].float().item()

            # Collect all experts
            experts = {}
            for scale_idx, n_experts in enumerate(n_scales):
                for expert_idx in range(n_experts):
                    down_key = f"experts.{scale_idx}.{expert_idx}.down.weight"
                    up_key = f"experts.{scale_idx}.{expert_idx}.up.weight"
                    if down_key not in module_data or up_key not in module_data:
                        raise KeyError(f"Missing expert [{scale_idx}][{expert_idx}]")
                    experts[(scale_idx, expert_idx)] = (
                        module_data[down_key].detach().clone(),
                        module_data[up_key].detach().clone(),
                    )

            # Determine lora_dim from first expert's down weight
            first_down = experts[(0, 0)][0]
            lora_dim = first_down.shape[0]
            scale = alpha_val / lora_dim

            # Determine router input mode from FC weight shape
            router_fc_w = module_data["router_fc.weight"]
            router_in_dim = router_fc_w.shape[1]
            if router_in_dim == lora_dim:
                router_input_mode = "bottleneck"
            else:
                router_input_mode = "input"

            # Get num_timesteps from embedding shape
            embed_w = module_data["timestep_embed.weight"]
            timesteps = embed_w.shape[0]
            num_timesteps = max(num_timesteps, timesteps)

            adapter = _TSMAdapter(
                experts=experts,
                router_fc_w=router_fc_w.detach().clone(),
                router_fc_b=module_data["router_fc.bias"].detach().clone(),
                timestep_embed_w=embed_w.detach().clone(),
                scale=scale,
                n_scales=n_scales,
                num_timesteps=timesteps,
                router_input_mode=router_input_mode,
            )
            adapter.multiplier = strength

            manager.add_adapter(model_weight_key, adapter, strength=1.0)
            all_adapters.append(adapter)
            loaded += 1

        except Exception as e:
            logging.warning("[TSM] Skipping %s: %s", checkpoint_key, e)
            skipped += 1

    logging.info("[TSM] Loaded=%d, Skipped=%d", loaded, skipped)

    if loaded == 0:
        raise ValueError("[TSM] No adapters were created. Check checkpoint format.")

    injections = manager.create_injections(model_lora.model)
    injection_key = f"{_TSM_INJECTION_PREFIX}_{uuid.uuid4().hex}"
    model_lora.set_injections(injection_key, injections)

    return model_lora, loaded, num_timesteps, all_adapters


# ── ComfyUI Node ─────────────────────────────────────────────────────────────

class TimeStepMasterLoader:
    """
    Loads a LyCORIS TSM checkpoint for per-step asymmetric mixture of
    timestep LoRA experts during inference.

    At each denoising step, the node:
    1. Selects the core expert (finest interval) — ungated
    2. Computes router gates G(z_t, t) = F(z_t) + ε(t)
    3. Gates context experts (coarser intervals) adaptively

    Requires a TSM checkpoint trained with LyCORIS (algo='tsm').
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
                "num_timesteps": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 1000,
                    "step": 1,
                    "tooltip": "Total diffusion timesteps T. 0 = auto-detect from model."
                }),
                "debug": ("BOOLEAN", {"default": False}),
                "debug_every": ("INT", {"default": 5, "min": 1, "max": 100}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_tsm"
    CATEGORY = "loaders/TSM"
    DESCRIPTION = "Loads LyCORIS TSM checkpoint with per-step asymmetric expert mixture."

    def load_tsm(self, model, lora_name, strength, num_timesteps, debug, debug_every):
        lora_path = folder_paths.get_full_path("loras", lora_name)
        if lora_path is None:
            raise ValueError(f"[TSM] Could not find: {lora_name}")

        logging.info("[TSM] Loading: %s", lora_path)
        logging.info("[TSM] strength=%.3f num_timesteps=%d", strength, num_timesteps)

        state_dict = comfy.utils.load_torch_file(lora_path, safe_load=True)
        model_out, loaded_count, inferred_timesteps, adapters = _load_tsm(
            model, state_dict, strength
        )

        resolved_timesteps = _resolve_num_timesteps(model_out, num_timesteps)
        if resolved_timesteps <= 0:
            resolved_timesteps = inferred_timesteps

        _configure_runtime(
            model_patcher=model_out,
            num_timesteps=resolved_timesteps,
            adapters=adapters,
            debug=debug,
            debug_every=debug_every,
        )

        logging.info(
            "[TSM] Ready. %d adapters, num_timesteps=%d",
            loaded_count, resolved_timesteps
        )
        return (model_out,)


# ── Node registration ────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "TimeStepMasterLoader": TimeStepMasterLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TimeStepMasterLoader": "Load TSM (TimeStep Master)",
}
