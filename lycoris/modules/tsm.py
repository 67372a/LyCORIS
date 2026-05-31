"""
TimeStep Master (TSM) — Asymmetrical Mixture of TimeStep LoRA Experts.

Based on "TimeStep Master: Asymmetrical Mixture of Timestep LoRA Experts
for Versatile and Efficient Diffusion Models in Vision" (arXiv:2503.07416)

Key features:
1. Multiple LoRA experts at different timestep interval scales
2. Core expert (finest granularity, ungated) + context experts (gated)
3. Timestep-dependent router: G(z_t, t) = F(z_t) + ε(t)
4. Two-stage training: fostering (expert training) and assembling (router training)

Timestep context is passed via global state, following the T-LoRA pattern.
"""

import math
from functools import cache
from typing import List, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import LycorisBaseModule
from ..logging import logger


RouterInputType = Literal["bottleneck", "input"]

# Thread-local storage for TSM timestep context (set by training loop)
_tsm_timestep_storage: dict[int, int] = {}


def set_tsm_timestep(timestep: int, group_id: int = 0) -> None:
    """Set the current diffusion timestep for TSM modules.

    Call this before each forward pass with the current timestep.

    Args:
        timestep: Current denoising timestep (0 to T-1 or 1 to T).
        group_id: Optional group ID for multi-network scenarios.
    """
    _tsm_timestep_storage[group_id] = timestep


def get_tsm_timestep(group_id: int = 0) -> Optional[int]:
    """Get the current TSM timestep, or None if not set."""
    return _tsm_timestep_storage.get(group_id, None)


def clear_tsm_timestep(group_id: int = 0) -> None:
    """Clear the TSM timestep after the forward pass."""
    _tsm_timestep_storage.pop(group_id, None)


@cache
def _log_tsm_init():
    return logger.info(
        "TSM: TimeStep Master — Asymmetrical Mixture of TimeStep LoRA Experts"
    )


class TSMModule(LycorisBaseModule):
    """TSM module implementing asymmetrical mixture of timestep LoRA experts.

    The weight delta is computed as:

    Stage 1 (fostering):
        ΔW = B_{i} A_{i}  (single expert for current interval)

    Stage 2 (assembling):
        ΔW = B_{i1} A_{i1} + Σ_{j=2}^{m} G_j ⊙ B_{ij} A_{ij}

    Where i1 is the core expert (finest interval, ungated), and
    G_j = FC(z_t) + embed(t) are context expert gates (no sigmoid, per paper Eq. 6).

    Args:
        lora_name: Unique name for this module.
        org_module: Original module to wrap.
        multiplier: Output multiplier.
        lora_dim: Rank of each LoRA expert.
        alpha: Alpha scaling factor.
        dropout: Dropout probability.
        rank_dropout: Rank-wise dropout probability.
        module_dropout: Module-level dropout probability.
        bypass_mode: Use bypass forward mode.
        tsm_n_scales: List of interval counts per scale, e.g. [8, 1].
            The first element is the core scale (finest granularity).
            Each subsequent element is a context scale.
        tsm_num_timesteps: Total number of diffusion timesteps T.
        tsm_stage: Training stage (1=fostering, 2=assembling).
        tsm_router_input: Router input mode ("bottleneck" or "input").
            Paper uses "input" (z_t is the input feature, dimension k).
    """

    name = "tsm"
    support_module = {
        "linear",
        "conv1d",
        "conv2d",
        "conv3d",
    }
    weight_list = [
        "alpha",
    ]
    weight_list_det = ["alpha"]  # Unique identifier for TSM

    def __init__(
        self,
        lora_name: str,
        org_module: nn.Module,
        multiplier: float = 1.0,
        lora_dim: int = 4,
        alpha: float = 1,
        dropout: float = 0.0,
        rank_dropout: float = 0.0,
        module_dropout: float = 0.0,
        use_tucker: bool = False,
        use_scalar: bool = False,
        scalar_init_value: float = None,
        rank_dropout_scale: bool = False,
        bypass_mode: bool = None,
        # TSM-specific parameters
        tsm_n_scales: Optional[List[int]] = None,
        tsm_num_timesteps: int = 1000,
        tsm_stage: int = 1,
        tsm_router_input: RouterInputType = "input",
        ggpo_beta: Optional[float] = None,
        ggpo_sigma: Optional[float] = None,
        ggpo_conv: bool = False,
        ggpo_conv_weight_sample_size: int = 100,
        **kwargs,
    ):
        super().__init__(
            lora_name,
            org_module,
            multiplier,
            dropout,
            rank_dropout,
            module_dropout,
            rank_dropout_scale,
            bypass_mode,
            ggpo_beta,
            ggpo_sigma,
            ggpo_conv,
            ggpo_conv_weight_sample_size,
        )

        if self.module_type not in self.support_module:
            raise ValueError(f"{self.module_type} is not supported in TSM algo.")

        _log_tsm_init()

        # Parse n_scales — default to [8, 1] (paper's simplest effective config)
        if tsm_n_scales is None:
            tsm_n_scales = [8, 1]
        self.tsm_n_scales = list(tsm_n_scales)
        self.tsm_num_timesteps = tsm_num_timesteps
        self.tsm_stage = tsm_stage
        self.tsm_router_input = tsm_router_input
        self.lora_dim = lora_dim
        self.mask_group_id = 0

        if len(self.tsm_n_scales) < 2:
            raise ValueError(
                f"TSM requires at least 2 scales (core + context), got {len(tsm_n_scales)}"
            )

        num_context_scales = len(self.tsm_n_scales) - 1

        # --- Dimensions ---
        if self.module_type.startswith("conv"):
            self.isconv = True
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels
            k_size = org_module.kernel_size
            stride = org_module.stride
            padding = org_module.padding

            self.conv_shape = (out_dim, in_dim, *k_size)
            self.down_op = self.op
            self.up_op = self.op
            self.kw_dict_down = {
                "stride": stride,
                "padding": padding,
                "dilation": org_module.dilation,
                "groups": org_module.groups,
            }
            self.kw_dict_up = {
                "stride": (1,) * len(k_size),
                "padding": (0,) * len(k_size),
                "dilation": (1,) * len(k_size),
                "groups": 1,
            }
        else:
            self.isconv = False
            in_dim = org_module.in_features
            out_dim = org_module.out_features
            self.down_op = F.linear
            self.up_op = F.linear
            self.kw_dict_down = {}
            self.kw_dict_up = {}

        self.in_dim = in_dim
        self.out_dim = out_dim

        # --- Create experts organized by scale ---
        # experts[scale_idx] = nn.ModuleList of expert pairs
        # Each expert has "down" and "up" sub-modules
        self.experts = nn.ModuleList()
        for scale_idx, n_experts in enumerate(self.tsm_n_scales):
            scale_experts = nn.ModuleList()
            for _ in range(n_experts):
                expert = self._create_expert(in_dim, out_dim, org_module)
                scale_experts.append(expert)
            self.experts.append(scale_experts)

        # --- Router (only used in stage 2) ---
        if self.tsm_router_input == "bottleneck":
            router_in_dim = lora_dim
        else:
            router_in_dim = in_dim

        self.router_fc = nn.Linear(router_in_dim, num_context_scales, bias=True)
        self.timestep_embed = nn.Embedding(tsm_num_timesteps, num_context_scales)

        # Initialize router to produce zero gates at start
        nn.init.zeros_(self.router_fc.weight)
        nn.init.zeros_(self.router_fc.bias)
        nn.init.zeros_(self.timestep_embed.weight)

        # --- Alpha scaling ---
        if isinstance(alpha, torch.Tensor):
            alpha = alpha.detach().float().numpy()
        alpha = lora_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / lora_dim
        self.register_buffer("alpha", torch.tensor(alpha))

        # Scalar
        if use_scalar:
            init_val = scalar_init_value if scalar_init_value is not None else 0.1
            self.scalar = nn.Parameter(torch.tensor(init_val))
        else:
            self.register_buffer("scalar", torch.tensor(1.0), persistent=False)

        # Set initial stage (freezes/unfreezes appropriate params)
        self.set_stage(tsm_stage)

    def _create_expert(
        self, in_dim: int, out_dim: int, org_module: nn.Module
    ) -> nn.ModuleDict:
        """Create a single LoRA expert (down + up pair)."""
        if self.isconv:
            k_size = org_module.kernel_size
            stride = org_module.stride
            padding = org_module.padding
            down = self.module(
                in_dim, self.lora_dim, k_size, stride, padding, bias=False
            )
            up = self.module(self.lora_dim, out_dim, 1, bias=False)
        else:
            down = nn.Linear(in_dim, self.lora_dim, bias=False)
            up = nn.Linear(self.lora_dim, out_dim, bias=False)

        # Initialize: A (down) with Kaiming, B (up) with zeros
        nn.init.kaiming_uniform_(down.weight, a=math.sqrt(5))
        nn.init.zeros_(up.weight)

        return nn.ModuleDict({"down": down, "up": up})

    def set_stage(self, stage: int) -> None:
        """Switch between fostering (1) and assembling (2) stages.

        In stage 1, all experts are trainable and the router is frozen.
        In stage 2, all experts are frozen and the router is trainable.
        """
        self.tsm_stage = stage
        if stage == 2:
            # Freeze all experts
            for scale_experts in self.experts:
                for expert in scale_experts:
                    for p in expert.parameters():
                        p.requires_grad = False
            # Unfreeze router
            self.router_fc.requires_grad_(True)
            self.timestep_embed.requires_grad_(True)
        else:
            # Stage 1: all experts trainable
            for scale_experts in self.experts:
                for expert in scale_experts:
                    for p in expert.parameters():
                        p.requires_grad = True
            # Freeze router in stage 1
            self.router_fc.requires_grad_(False)
            self.timestep_embed.requires_grad_(False)

    def _get_expert_index(self, timestep: int, scale_idx: int) -> int:
        """Get the expert index for a given timestep and scale.

        Per paper Eq. 3/5: i_j = ceil(t / T * n_j), 1-indexed.
        We convert to 0-indexed for array access.

        Args:
            timestep: Current timestep (1 to T).
            scale_idx: Index into tsm_n_scales.

        Returns:
            0-indexed expert index for this scale.
        """
        n_j = self.tsm_n_scales[scale_idx]
        # i_j = ceil(t / T * n_j), 1-indexed
        expert_idx_1indexed = math.ceil(timestep / self.tsm_num_timesteps * n_j)
        # Clamp to valid range and convert to 0-indexed
        expert_idx_1indexed = max(1, min(expert_idx_1indexed, n_j))
        return expert_idx_1indexed - 1

    def _get_timestep(self) -> int:
        """Get current timestep from global state, defaulting to T (max noise)."""
        t = get_tsm_timestep(self.mask_group_id)
        if t is None:
            # Default: use max timestep (all experts active)
            return self.tsm_num_timesteps
        return t

    def _compute_gates(
        self,
        bottleneck: torch.Tensor,
        input_features: torch.Tensor,
        timestep: int,
    ) -> torch.Tensor:
        """Compute context expert gates via the router (paper Eq. 6).

        G(z_t, t) = FC(z_t) + ε(t)    (no sigmoid — raw unbounded gates)

        Args:
            bottleneck: Core expert's down-projection output.
            input_features: Original input to the module.
            timestep: Current diffusion timestep.

        Returns:
            Gate tensor of shape (num_context_scales,).
        """
        num_context = len(self.tsm_n_scales) - 1
        device = bottleneck.device if self.tsm_router_input == "bottleneck" else input_features.device

        if self.tsm_router_input == "bottleneck":
            # Pool bottleneck to (lora_dim,)
            z = bottleneck.detach().float()
            # Reduce all dims except the rank dim to get (lora_dim,)
            if z.dim() == 1:
                z = z  # Already (lora_dim,)
            elif z.dim() == 2:
                z = z.mean(dim=0)  # (lora_dim,)
            else:
                # Conv: (B, lora_dim, *spatial) -> (lora_dim,)
                reduce_dims = list(range(0, z.dim()))
                reduce_dims.remove(1)  # Keep dim 1 (lora_dim)
                z = z.mean(dim=reduce_dims)
            z = z[: self.lora_dim]
        else:
            # Pool input features to (in_dim,)
            z = input_features.detach().float()
            if z.dim() == 1:
                z = z
            elif z.dim() == 2:
                z = z.mean(dim=0)
            else:
                reduce_dims = list(range(0, z.dim()))
                reduce_dims.remove(1)
                z = z.mean(dim=reduce_dims)
            z = z[: self.in_dim]

        # Router: FC(z) + embed(t)
        # Move router to same device as z in case module was created on different device
        fc_w = self.router_fc.weight.to(device)
        fc_b = self.router_fc.bias.to(device)
        embed_w = self.timestep_embed.weight.to(device)

        fc_out = F.linear(z.to(fc_w.dtype), fc_w, fc_b)
        # Paper Eq. 6: G(z_t, t) = F(z_t) + ε(t)
        # ε(t) extracts the t-th row of the T×(m-1) embedding matrix.
        # Timesteps are 1-indexed (t ∈ [1, T]), so embedding index = t - 1.
        t_embed_idx = min(max(timestep - 1, 0), self.tsm_num_timesteps - 1)
        t_embed = embed_w[torch.tensor(t_embed_idx, device=device, dtype=torch.long)]
        # No sigmoid — paper uses raw sum as gating values (unbounded)
        gates = fc_out + t_embed
        return gates

    def _apply_expert(
        self, expert: nn.ModuleDict, x: torch.Tensor, dtype: torch.dtype
    ) -> torch.Tensor:
        """Apply a single LoRA expert to input x: down -> up."""
        down_w = expert["down"].weight.to(dtype)
        up_w = expert["up"].weight.to(dtype)

        if self.isconv:
            mid = self.down_op(x, down_w, None, **self.kw_dict_down)
            out = self.up_op(mid, up_w, None, **self.kw_dict_up)
        else:
            mid = self.down_op(x, down_w, None)
            out = self.up_op(mid, up_w, None)

        return out

    def _forward_stage1(self, x: torch.Tensor) -> torch.Tensor:
        """Stage 1 (fostering): forward through one active expert per scale.

        For each scale, the expert matching the current timestep is selected.
        All scales contribute so that every expert receives gradients and learns
        to handle its assigned timestep interval.  This is critical: context
        scale experts must have non-zero B matrices before stage 2, otherwise
        the router's gating has nothing to gate.

        Per paper Eq. 3–4: each expert trains on its interval's timesteps.
        """
        timestep = self._get_timestep()
        dtype = x.dtype

        total_delta = None
        for scale_idx in range(len(self.tsm_n_scales)):
            expert_idx = self._get_expert_index(timestep, scale_idx)
            expert = self.experts[scale_idx][expert_idx]
            delta = self._apply_expert(expert, x, dtype)
            if total_delta is None:
                total_delta = delta
            else:
                total_delta = total_delta + delta

        return total_delta * self.scalar.to(x.device) * self.scale

    def _forward_stage2(self, x: torch.Tensor) -> torch.Tensor:
        """Stage 2 (assembling): asymmetrical mixture of experts.

        Core expert (finest interval, ungated) + gated context experts.
        ΔW*x = B_{i1}A_{i1}*x + Σ_{j=2}^{m} G_j ⊙ B_{ij}A_{ij}*x
        """
        timestep = self._get_timestep()
        dtype = x.dtype
        device = x.device

        # --- Core expert (scale 0, ungated) ---
        core_expert_idx = self._get_expert_index(timestep, 0)
        core_expert = self.experts[0][core_expert_idx]

        # Compute core bottleneck for router input
        core_down_w = core_expert["down"].weight.to(dtype)
        if self.isconv:
            core_bottleneck = self.down_op(x, core_down_w, None, **self.kw_dict_down)
        else:
            core_bottleneck = self.down_op(x, core_down_w, None)

        # Compute core output
        core_up_w = core_expert["up"].weight.to(dtype)
        if self.isconv:
            core_out = self.up_op(core_bottleneck, core_up_w, None, **self.kw_dict_up)
        else:
            core_out = self.up_op(core_bottleneck, core_up_w, None)

        # --- Context experts (gated) ---
        gates = self._compute_gates(core_bottleneck, x, timestep)

        context_out = None
        for ctx_idx in range(len(self.tsm_n_scales) - 1):
            scale_idx = ctx_idx + 1  # Skip core scale (0)
            expert_idx = self._get_expert_index(timestep, scale_idx)
            ctx_expert = self.experts[scale_idx][expert_idx]

            ctx_delta = self._apply_expert(ctx_expert, x, dtype)
            gate = gates[ctx_idx]

            gated_delta = ctx_delta * gate
            if context_out is None:
                context_out = gated_delta
            else:
                context_out = context_out + gated_delta

        # Combine
        delta = core_out
        if context_out is not None:
            delta = delta + context_out

        return delta * self.scalar.to(device) * self.scale

    def get_diff_weight(self, multiplier=1.0, shape=None, device=None):
        """Compute weight difference for rebuild mode.

        Reconstructs the full ΔW from all active experts.
        """
        if device is None:
            device = self.experts[0][0]["down"].weight.device

        timestep = self._get_timestep()

        if self.tsm_stage == 1:
            # Stage 1: all scales contribute (one expert per scale)
            diff = None
            for scale_idx in range(len(self.tsm_n_scales)):
                expert_idx = self._get_expert_index(timestep, scale_idx)
                expert = self.experts[scale_idx][expert_idx]
                down_w = expert["down"].weight.to(device)
                up_w = expert["up"].weight.to(device)

                if self.isconv:
                    up_2d = up_w.view(up_w.size(0), -1)
                    down_2d = down_w.view(down_w.size(0), -1)
                    scale_diff = (up_2d @ down_2d).view(self.shape)
                else:
                    scale_diff = (
                        up_w.view(up_w.size(0), -1)
                        @ down_w.view(down_w.size(0), -1)
                    ).view(self.shape)

                if diff is None:
                    diff = scale_diff
                else:
                    diff = diff + scale_diff

        else:
            # Stage 2: asymmetrical mixture
            core_expert_idx = self._get_expert_index(timestep, 0)
            core_expert = self.experts[0][core_expert_idx]
            core_down_w = core_expert["down"].weight.to(device)
            core_up_w = core_expert["up"].weight.to(device)

            if self.isconv:
                core_up_2d = core_up_w.view(core_up_w.size(0), -1)
                core_down_2d = core_down_w.view(core_down_w.size(0), -1)
                diff = (core_up_2d @ core_down_2d).view(self.shape)
            else:
                diff = (
                    core_up_w.view(core_up_w.size(0), -1)
                    @ core_down_w.view(core_down_w.size(0), -1)
                ).view(self.shape)

            # Context experts with gates
            # We need bottleneck for router — approximate with zeros if not in bypass
            # For rebuild mode, compute bottleneck from core expert
            dummy_bottleneck = torch.zeros(
                self.lora_dim, device=device, dtype=core_down_w.dtype
            )
            gates = self._compute_gates(
                dummy_bottleneck,
                torch.zeros(self.in_dim, device=device, dtype=core_down_w.dtype),
                timestep,
            )

            for ctx_idx in range(len(self.tsm_n_scales) - 1):
                scale_idx = ctx_idx + 1
                expert_idx = self._get_expert_index(timestep, scale_idx)
                ctx_expert = self.experts[scale_idx][expert_idx]
                ctx_down_w = ctx_expert["down"].weight.to(device)
                ctx_up_w = ctx_expert["up"].weight.to(device)

                if self.isconv:
                    ctx_up_2d = ctx_up_w.view(ctx_up_w.size(0), -1)
                    ctx_down_2d = ctx_down_w.view(ctx_down_w.size(0), -1)
                    ctx_diff = (ctx_up_2d @ ctx_down_2d).view(self.shape)
                else:
                    ctx_diff = (
                        ctx_up_w.view(ctx_up_w.size(0), -1)
                        @ ctx_down_w.view(ctx_down_w.size(0), -1)
                    ).view(self.shape)

                diff = diff + gates[ctx_idx] * ctx_diff

        diff = diff * self.scalar.to(device) * self.scale * multiplier

        if shape is not None:
            diff = diff.view(shape)

        return diff, None

    def get_merged_weight(self, multiplier=1.0, shape=None, device=None):
        """Get original weight + LoRA delta."""
        diff, _ = self.get_diff_weight(multiplier=multiplier, shape=shape, device=device)
        weight = self.get_org_weight_for_compute(diff.device)
        if weight.dtype != diff.dtype:
            weight = weight.to(diff.dtype)
        merged = weight + diff
        return merged, None

    def bypass_forward_diff(self, x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        """Compute LoRA contribution in bypass mode."""
        if self.tsm_stage == 1:
            delta = self._forward_stage1(x)
        else:
            delta = self._forward_stage2(x)
        return self.drop(delta * scale)

    def bypass_forward(self, x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        """Forward with bypass mode (compute LoRA separately)."""
        return self.org_forward(x) + self.bypass_forward_diff(x, scale=scale)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Forward pass with TSM expert routing."""
        if self.module_dropout and self.training:
            if torch.rand(1) < self.module_dropout:
                return self.org_forward(x, *args, **kwargs)

        if self.bypass_mode:
            return self.bypass_forward(x, scale=self.multiplier)

        # Rebuild mode
        base = self.org_forward(x, *args, **kwargs)
        diff_weight, _ = self.get_diff_weight(multiplier=self.multiplier, device=x.device)
        diff_weight = diff_weight.to(x.dtype)
        delta = self.op(x, diff_weight, None, **self.kw_dict)
        return base + delta

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def custom_state_dict(self):
        """Serialize all experts + router + config."""
        destination = {}
        scalar = self.scalar.to(
            device=self.experts[0][0]["up"].weight.device, non_blocking=True
        )

        # Serialize each expert
        for scale_idx, scale_experts in enumerate(self.experts):
            for expert_idx, expert in enumerate(scale_experts):
                prefix = f"experts.{scale_idx}.{expert_idx}"
                # Bake scalar into up weight for portability
                destination[f"{prefix}.up.weight"] = (
                    expert["up"].weight * scalar
                )
                destination[f"{prefix}.down.weight"] = expert["down"].weight

        # Serialize router
        destination["router_fc.weight"] = self.router_fc.weight
        destination["router_fc.bias"] = self.router_fc.bias
        destination["timestep_embed.weight"] = self.timestep_embed.weight

        # Serialize config
        destination["alpha"] = self.alpha

        return destination

    @classmethod
    def extract_state_dict(cls, lyco_state_dict, lora_name):
        """Extract TSM state dict entries for this module."""
        prefix = f"{lora_name}."
        result = {}

        # Find all keys with this prefix
        expert_downs = {}
        expert_ups = {}
        router_fc_w = None
        router_fc_b = None
        timestep_embed_w = None
        alpha = None

        for key, value in lyco_state_dict.items():
            if not key.startswith(prefix):
                continue
            subkey = key[len(prefix) :]

            if subkey.startswith("experts."):
                parts = subkey.split(".")
                # experts.{scale}.{idx}.{up|down}.weight
                scale_idx = int(parts[1])
                expert_idx = int(parts[2])
                direction = parts[3]  # "up" or "down"
                ekey = (scale_idx, expert_idx)
                if direction == "down":
                    expert_downs[ekey] = value
                elif direction == "up":
                    expert_ups[ekey] = value
            elif subkey == "router_fc.weight":
                router_fc_w = value
            elif subkey == "router_fc.bias":
                router_fc_b = value
            elif subkey == "timestep_embed.weight":
                timestep_embed_w = value
            elif subkey == "alpha":
                alpha = value

        if alpha is None:
            return None  # Not a TSM checkpoint

        # Pack into ordered tuple for make_module_from_state_dict
        # Determine scales from expert keys
        if not expert_downs:
            return None

        max_scale = max(k[0] for k in expert_downs.keys())
        max_expert = max(k[1] for k in expert_downs.keys())

        # Build ordered list: [alpha, router_fc_w, router_fc_b, timestep_embed_w,
        #                      scale0_expert0_down, scale0_expert0_up, ...]
        params = [alpha, router_fc_w, router_fc_b, timestep_embed_w]
        n_scales = []
        for s in range(max_scale + 1):
            n_experts_for_scale = max(
                (k[1] for k in expert_downs.keys() if k[0] == s), default=-1
            ) + 1
            n_scales.append(n_experts_for_scale)
            for e in range(n_experts_for_scale):
                params.append(expert_downs.get((s, e)))
                params.append(expert_ups.get((s, e)))

        params.insert(0, n_scales)  # Prepend n_scales list
        return tuple(params)

    @classmethod
    def algo_check(cls, lyco_state_dict, lora_name):
        """Check if this state dict belongs to a TSM module."""
        prefix = f"{lora_name}."
        for key in lyco_state_dict:
            if key.startswith(prefix) and "experts." in key[len(prefix) :]:
                return True
        return False

    @classmethod
    def make_module_from_state_dict(
        cls,
        lora_name: str,
        orig_module: nn.Module,
        n_scales,
        alpha,
        router_fc_w,
        router_fc_b,
        timestep_embed_w,
        *expert_weights,
    ):
        """Reconstruct TSM module from saved state dict."""
        # Determine lora_dim from first expert's down weight
        first_down = None
        for w in expert_weights:
            if w is not None:
                first_down = w
                break
        if first_down is None:
            raise ValueError("No expert weights found in state dict")

        lora_dim = first_down.shape[0] if first_down.dim() >= 2 else first_down.shape[0]

        module = cls(
            lora_name,
            orig_module,
            multiplier=1.0,
            lora_dim=lora_dim,
            alpha=float(alpha),
            tsm_n_scales=n_scales,
            tsm_num_timesteps=timestep_embed_w.shape[0] if timestep_embed_w is not None else 1000,
            tsm_stage=1,
        )

        # Load expert weights
        weight_idx = 0
        for scale_idx in range(len(n_scales)):
            for expert_idx in range(n_scales[scale_idx]):
                if weight_idx < len(expert_weights):
                    down_w = expert_weights[weight_idx]
                    up_w = expert_weights[weight_idx + 1] if weight_idx + 1 < len(expert_weights) else None
                    weight_idx += 2

                    expert = module.experts[scale_idx][expert_idx]
                    if down_w is not None:
                        expert["down"].weight.data.copy_(down_w)
                    if up_w is not None:
                        # up was saved with scalar baked in; we store scalar separately
                        expert["up"].weight.data.copy_(up_w)

        # Load router
        if router_fc_w is not None:
            module.router_fc.weight.data.copy_(router_fc_w)
        if router_fc_b is not None:
            module.router_fc.bias.data.copy_(router_fc_b)
        if timestep_embed_w is not None:
            module.timestep_embed.weight.data.copy_(timestep_embed_w)

        return module

    def load_weight_hook(self, module: nn.Module, incompatible_keys):
        """Handle missing/extra keys on load."""
        missing_keys = incompatible_keys.missing_keys
        for key in list(missing_keys):
            if "scalar" in key:
                missing_keys.remove(key)
        # Reset scalar to 1.0 on load
        if isinstance(self.scalar, nn.Parameter):
            self.scalar.data.copy_(torch.ones_like(self.scalar))
        elif getattr(self, "scalar", None) is not None:
            self.scalar.copy_(torch.ones_like(self.scalar))

    @torch.no_grad()
    def apply_max_norm(self, max_norm: float, device=None):
        """Apply max norm regularization to all expert weights."""
        # Compute total norm across all experts
        total_norm_sq = torch.tensor(0.0)
        for scale_experts in self.experts:
            for expert in scale_experts:
                down_norm = expert["down"].weight.float().norm()
                up_norm = expert["up"].weight.float().norm()
                total_norm_sq += down_norm**2 + up_norm**2
        orig_norm = total_norm_sq.sqrt() * self.scale

        norm = torch.clamp(orig_norm, max_norm / 2)
        desired = torch.clamp(norm, max=max_norm)
        ratio = desired.cpu() / norm.cpu()

        scaled = norm != desired
        if scaled:
            self.scalar *= ratio

        return scaled, orig_norm * ratio
