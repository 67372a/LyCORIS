"""LoRA²: Adaptive Rank LoRA module.

Each LoRA component learns its own rank during training by placing a
learnable diagonal importance matrix Λ between B and A:

    ΔW = B · diag(Λ) · A

The rank is controlled by a single learnable parameter ν that
determines a discretized exponential distribution over rank positions.
Early rank positions (high importance) are preferentially used, allowing
the model to self-select an appropriate capacity.

Reference: "Not All Layers Are Created Equal: Adaptive LoRA Ranks for
Personalized Image Generation" (Shenaj et al., arXiv:2603.21884)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .locon import LoConModule
from .lora2_utils import (
    compute_effective_rank,
    compute_lambda_diag,
    compute_nu_target,
    rescaled_kaiming_std,
)
from ..logging import logger
from typing import Optional


class LoRA2Module(LoConModule):
    """LoRA²: Adaptive Rank LoRA.

    Extends LoConModule with a learnable diagonal importance matrix
    that adapts the effective rank per component during training.

    Key properties:
      - Each component has a learnable ν parameter controlling rank
      - Forward: ΔW = B · diag(Λ) · A where Λ depends on ν
      - Rank regularization loss keeps ν near a target
      - At save time, weights are truncated to effective rank
      - Inference is identical to standard LoRA (no Λ needed)

    Args:
        lora2_nu_init: Initial value for ν (default: auto from r_target).
        lora2_nu_target: Target rank for regularization (default: lora_dim).
        lora2_quantile: Quantile for effective rank computation (default: 0.9).
        lora2_lambda_r: Weight for rank regularization loss.
        lora2_lambda_e: Weight for entropy loss (external, requires attention maps).
    """

    name = "lora2"
    support_module = {"linear", "conv1d", "conv2d", "conv3d"}

    # weight_list defines the serialization order.
    # extract_state_dict returns values in this order.
    # make_module_from_state_dict receives them as positional args.
    weight_list = [
        "lora_up.weight",
        "lora_down.weight",
        "lora2_nu",
        "alpha",
        "dora_scale",
    ]
    # algo_check looks for any of these keys to identify this module type.
    # lora2_nu is unique to LoRA² — distinguishes it from plain LoCon.
    weight_list_det = ["lora2_nu"]

    # Class-level registry for aggregating regularization losses across all modules.
    _lora2_modules: list = []

    def __init__(
        self,
        lora_name,
        org_module: nn.Module,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
        dropout=0.0,
        rank_dropout=0.0,
        module_dropout=0.0,
        use_tucker=False,
        use_scalar=False,
        scalar_init_value=None,
        rank_dropout_scale=False,
        weight_decompose=False,
        wd_on_output=True,
        bypass_mode=None,
        rs_lora=False,
        ggpo_beta: Optional[float] = None,
        ggpo_sigma: Optional[float] = None,
        ggpo_conv: bool = False,
        ggpo_conv_weight_sample_size: int = 100,
        orthogonalize=False,
        orthogonal_init=False,
        # --- LoRA² specific ---
        lora2_nu_init=None,
        lora2_nu_target=None,
        lora2_quantile=0.9,
        lora2_lambda_r=1e-4,
        lora2_lambda_e=1e-4,
        **kwargs,
    ):
        # Tucker decomposition is incompatible with adaptive rank Λ
        if use_tucker:
            logger.warning(
                f"LoRA² does not support Tucker decomposition "
                f"(module {lora_name}), disabling Tucker."
            )
            use_tucker = False

        super().__init__(
            lora_name=lora_name,
            org_module=org_module,
            multiplier=multiplier,
            lora_dim=lora_dim,
            alpha=alpha,
            dropout=dropout,
            rank_dropout=rank_dropout,
            module_dropout=module_dropout,
            use_tucker=False,  # Force disable
            use_scalar=use_scalar,
            scalar_init_value=scalar_init_value,
            rank_dropout_scale=rank_dropout_scale,
            weight_decompose=weight_decompose,
            wd_on_output=wd_on_output,
            bypass_mode=bypass_mode,
            rs_lora=rs_lora,
            ggpo_beta=ggpo_beta,
            ggpo_sigma=ggpo_sigma,
            ggpo_conv=ggpo_conv,
            ggpo_conv_weight_sample_size=ggpo_conv_weight_sample_size,
            orthogonalize=orthogonalize,
            orthogonal_init=orthogonal_init,
            **kwargs,
        )

        # Store LoRA² configuration
        self.lora2_quantile = lora2_quantile
        self.lora2_lambda_r = lora2_lambda_r
        self.lora2_lambda_e = lora2_lambda_e
        self.lora2_r_target = lora2_nu_target if lora2_nu_target is not None else lora_dim

        # Compute ν target for regularization
        self.nu_target = compute_nu_target(self.lora2_r_target, lora2_quantile)

        # Initialize ν as a learnable scalar parameter
        if lora2_nu_init is None:
            nu_init_val = self.nu_target
        else:
            nu_init_val = float(lora2_nu_init)
        self.lora2_nu = nn.Parameter(torch.tensor(nu_init_val, dtype=torch.float32))

        # Track current effective rank (updated each forward)
        self._current_d = compute_effective_rank(
            self.lora2_nu, lora2_quantile, lora_dim
        )

        # Re-initialize lora_down (A) with rescaled Kaiming per paper Section 3.3.
        # lora_up (B) stays zero-initialized (from parent __init__).
        self._reinit_lora_down()

        # Register in global LoRA² registry for loss aggregation
        LoRA2Module._lora2_modules.append(self)

    def _reinit_lora_down(self):
        """Re-initialize lora_down (A) with rescaled Kaiming per paper.

        Paper convention: ΔW = B · Λ · A
        LyCORIS convention: ΔW = lora_up · lora_down
        Mapping: lora_up = B, lora_down = A

        Per paper: B = zero-initialized, A = rescaled Kaiming.
        """
        d = compute_effective_rank(self.lora2_nu, self.lora2_quantile, self.lora_dim)
        std = rescaled_kaiming_std(self.lora2_nu, d)
        with torch.no_grad():
            nn.init.normal_(self.lora_down.weight, mean=0.0, std=std)

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    def get_rank_reg_loss(self) -> torch.Tensor:
        """L_reg = |ν − ν_target| for this component."""
        return torch.abs(self.lora2_nu - self.nu_target)

    @staticmethod
    def get_total_rank_reg_loss() -> torch.Tensor:
        """Aggregate rank regularization loss across all LoRA² modules.

        Call from training loop::

            rank_reg_loss = LoRA2Module.get_total_rank_reg_loss()
            total_loss = task_loss + lambda_r * rank_reg_loss
        """
        if not LoRA2Module._lora2_modules:
            return torch.tensor(0.0)
        first_device = LoRA2Module._lora2_modules[0].lora2_nu.device
        total = torch.tensor(0.0, device=first_device)
        for mod in LoRA2Module._lora2_modules:
            total = total + mod.get_rank_reg_loss()
        return total

    @classmethod
    def get_lora2_modules(cls) -> list:
        """Get all registered LoRA² modules."""
        return list(cls._lora2_modules)

    @classmethod
    def reset_lora2_registry(cls):
        """Clear the LoRA² module registry."""
        cls._lora2_modules.clear()

    # ------------------------------------------------------------------
    # Adaptive rank computation
    # ------------------------------------------------------------------

    def compute_effective_rank(self) -> int:
        """Compute and cache the current effective rank from ν."""
        self._current_d = compute_effective_rank(
            self.lora2_nu, self.lora2_quantile, self.lora_dim
        )
        return self._current_d

    def get_lambda_for_rank(self, d: int, device=None, dtype=None) -> torch.Tensor:
        """Compute Λ diagonal values for rank d.

        Args:
            d: Number of rank positions.
            device: Target device.
            dtype: Target dtype.

        Returns:
            1-D tensor of shape (d,) with importance weights.
        """
        nu = self.lora2_nu
        if device is not None:
            nu = nu.to(device=device)
        if dtype is not None:
            nu = nu.to(dtype=dtype)
        return compute_lambda_diag(nu, d)

    # ------------------------------------------------------------------
    # Weight computation overrides (rebuild mode)
    # ------------------------------------------------------------------

    def make_weight(self, device=None):
        """Compute ΔW = B · diag(Λ) · A with adaptive rank.

        Overrides parent to insert the Λ importance diagonal.
        Handles both linear (2D) and conv (>2D) weight shapes by
        flattening to 2D for the matmul, then reshaping back.
        """
        wa = self.lora_up.weight.to(device)    # B: (out, r) or (out, r, 1, 1) for conv
        wb = self.lora_down.weight.to(device)  # A: (r, in) or (r, in, kH, kW) for conv

        d = self.compute_effective_rank()
        lam = self.get_lambda_for_rank(d, device)  # (d,)

        # Flatten to 2D, truncate to effective rank, apply Λ
        # ΔW = (B[:, :d] * Λ) @ A[:d, :]
        wa_2d = wa.view(wa.size(0), -1)[:, :d] * lam  # (out, d)
        wb_2d = wb.view(wb.size(0), -1)[:d, :]         # (d, in_flat)

        weight = (wa_2d @ wb_2d).view(self.shape)
        weight = self._apply_rank_dropout(weight, device)

        return weight * self.scalar.to(device)

    def _compute_diff_weight_single(self, device, dtype):
        """Override: compute diff weight with Λ for rank_dropout path.

        Same as make_weight but returns weight * scalar * scale
        (the caller handles merge with org_weight).
        """
        wa = self.lora_up.weight.to(device=device, dtype=dtype)
        wb = self.lora_down.weight.to(device=device, dtype=dtype)

        d = self.compute_effective_rank()
        lam = self.get_lambda_for_rank(d, device, dtype)

        wa_2d = wa.view(wa.size(0), -1)[:, :d] * lam
        wb_2d = wb.view(wb.size(0), -1)[:d, :]
        diff_weight = (wa_2d @ wb_2d).view(self.shape)
        diff_weight = self._apply_rank_dropout(diff_weight, device)

        diff_weight = (diff_weight * self.scalar.to(device=device, dtype=dtype)) * self.scale
        return diff_weight

    # ------------------------------------------------------------------
    # Bypass mode overrides
    # ------------------------------------------------------------------

    def _bypass_forward_diff_single(self, x, scale=1):
        """Bypass forward with Λ applied between down and up projections.

        In standard LoRA: mid = A @ x, out = B @ mid
        In LoRA²: mid = A[:d,:] @ x, mid_scaled = Λ * mid, out = B[:,:d] @ mid_scaled
        """
        d = self.compute_effective_rank()
        lam = self.get_lambda_for_rank(d, x.device, x.dtype)  # (d,)

        # Down projection: use only first d rank dimensions
        wb = self.lora_down.weight[:d, :].to(x.device, dtype=x.dtype)
        wa = self.lora_up.weight[:, :d].to(x.device, dtype=x.dtype)

        if self.isconv:
            mid = self.down_op(
                x, wb, bias=None,
                stride=self.lora_down.stride,
                padding=self.lora_down.padding,
                dilation=self.lora_down.dilation,
                groups=self.lora_down.groups,
            )
        else:
            mid = self.down_op(x, wb)

        # Apply Λ: scale rank dimension by importance
        if self.isconv:
            # mid shape: (batch, d, H', W')
            mid = mid * lam.view(1, -1, 1, 1)
        else:
            # mid shape: (batch, d)
            mid = mid * lam.unsqueeze(0)

        # Rank dropout (applied to the Λ-scaled activations)
        if self.rank_dropout and self.training:
            drop = (torch.rand(d, device=mid.device) > self.rank_dropout).to(mid.dtype)
            if self.rank_dropout_scale:
                drop /= drop.mean()
            if self.isconv:
                mid = mid * drop.view(1, -1, 1, 1)
            else:
                dims = len(x.shape)
                mid = mid * drop.view(*[1] * (dims - 1), -1)

        # Up projection
        if self.isconv:
            up = self.up_op(
                mid, wa, bias=None,
                stride=self.lora_up.stride,
                padding=self.lora_up.padding,
                dilation=self.lora_up.dilation,
                groups=self.lora_up.groups,
            )
        else:
            up = self.up_op(mid, wa)

        return self.drop(up * self.scalar * self.scale * scale)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def custom_state_dict(self):
        """Save truncated to effective rank D.

        Truncates lora_up, lora_down to D dimensions and saves ν.
        The Λ diagonal can be recomputed from ν at load time, so it
        is not persisted.
        """
        d = self.compute_effective_rank()

        destination = {}
        destination["alpha"] = self.alpha
        destination["lora2_nu"] = self.lora2_nu

        # Truncate weights to effective rank
        destination["lora_up.weight"] = self.lora_up.weight[:, :d]
        destination["lora_down.weight"] = self.lora_down.weight[:d]

        if self.wd:
            destination["dora_scale"] = self.dora_scale

        return destination

    @classmethod
    def make_module_from_state_dict(
        cls, lora_name, orig_module, up, down, nu, alpha, dora_scale
    ):
        """Reconstruct LoRA² module from saved state dict.

        The saved weights are truncated to effective rank D. We reconstruct
        with lora_dim = D so the module starts at the saved capacity.
        """
        d = down.size(0) if down.dim() >= 2 else down.size(0)
        module = cls(
            lora_name,
            orig_module,
            1.0,
            lora_dim=d,
            alpha=float(alpha),
            weight_decompose=dora_scale is not None,
            lora2_nu_target=d,  # Target rank = saved rank
        )
        # Copy saved weights
        module.lora_up.weight.data[:, :d].copy_(up)
        module.lora_down.weight.data[:d].copy_(down)
        module.lora2_nu.data.copy_(nu)
        if dora_scale is not None:
            module.dora_scale.copy_(dora_scale)
        return module

    @classmethod
    def algo_check(cls, state_dict, lora_name):
        """Check if state dict contains LoRA² weights (has lora2_nu key)."""
        return f"{lora_name}.lora2_nu" in state_dict

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def in_features(self) -> int:
        """Input feature dimension."""
        return self.shape[1] if len(self.shape) >= 2 else self.shape[0]

    @property
    def out_features(self) -> int:
        """Output feature dimension."""
        return self.shape[0] if len(self.shape) >= 2 else 0
