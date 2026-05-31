"""
OrthoLoRA: Cayley-parameterized orthogonal low-rank adapter (PSOFT-inspired).

Frozen SVD bases P_basis (out, r) and Q_basis (r, in) from the pretrained
weight's top-r SVD, rotated by Cayley(S_q) / Cayley(S_p) where
R = (I - A)(I + A)^{-1}, A = S - S^T.  Trainable: S_p, S_q (r×r),
lambda_layer (1, r).

    out = x @ Q_eff^T @ diag(λ) @ P_eff^T
    where Q_eff = cayley(S_q) @ Q_basis, P_eff = P_basis @ cayley(S_p)

Zero-init S_p, S_q, λ → ΔW=0 at step 0.  Orthogonality is structural
(Cayley guarantees R^T R = I), not regularized.

Saves as standard lora_up/lora_down by default (distill mode) for
portability, or as native S_p/S_q/P_basis/Q_basis/lambda_layer for
checkpoint resume.

Ref: PSOFT (Wu et al., ICLR 2026).
Ported from anima_lora/networks/lora_modules/ortho.py.
"""

import math
from functools import cache
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import LycorisBaseModule
from ..logging import logger


@cache
def log_ortholora_init():
    return logger.info(
        "OrthoLoRA: Cayley-parameterized orthogonal adapter with SVD-informed init"
    )


class OrthoLoRAModule(LycorisBaseModule):
    name = "ortholora"
    support_module = {"linear"}  # Linear only (no Conv2d support)
    weight_list = [
        "S_p",
        "S_q",
        "P_basis",
        "Q_basis",
        "lambda_layer",
        "alpha",
        "dora_scale",
        # Distilled form (after custom_state_dict distill mode)
        "lora_up.weight",
        "lora_down.weight",
    ]
    weight_list_det = ["S_p"]  # Unique discriminator (2D = OrthoLoRA)

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
        weight_decompose: bool = False,
        bypass_mode: bool = None,
        use_timestep_mask: bool = False,
        native_save: bool = False,
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
        )

        if self.module_type not in self.support_module:
            raise ValueError(
                f"{self.module_type} is not supported in OrthoLoRA algo."
            )

        self.lora_dim = lora_dim
        self.native_save = native_save

        log_ortholora_init()

        # --- SVD-informed init ---
        # Randomized lowrank is ~10-100× faster than full SVD at r ≪ min(m,n).
        init_device = "cuda" if torch.cuda.is_available() else "cpu"
        W = org_module.weight.data.float().to(init_device)
        q = min(lora_dim + 6, min(W.shape))
        U, _S_vals, V = torch.svd_lowrank(W, q=q, niter=2)
        P_init = U[:, :lora_dim].clone().contiguous()  # (out, r)
        Q_init = V[:, :lora_dim].T.clone().contiguous()  # (r, in)
        del U, _S_vals, V, W

        # Frozen subspace bases; Cayley rotates within them.
        self.register_buffer("P_basis", P_init.cpu())
        self.register_buffer("Q_basis", Q_init.cpu())

        # Cayley(0) = I → at init P_eff = P_basis, Q_eff = Q_basis.
        self.S_p = nn.Parameter(torch.zeros(lora_dim, lora_dim))
        self.S_q = nn.Parameter(torch.zeros(lora_dim, lora_dim))

        # ΔW = 0 at init (standard LoRA convention).
        self.lambda_layer = nn.Parameter(torch.zeros(1, lora_dim))

        # Frozen bases → bf16. Cayley solve stays fp32 in forward.
        self.P_basis = self.P_basis.to(torch.bfloat16)
        self.Q_basis = self.Q_basis.to(torch.bfloat16)

        # Pre-allocated identity for the batched Cayley solve.
        self.register_buffer(
            "_eye_r",
            torch.eye(lora_dim, dtype=torch.float32),
            persistent=False,
        )

        # DoRA (weight decomposition): magnitude-direction normalization.
        self.wd = weight_decompose
        if self.wd:
            org_weight = org_module.weight.data.float()
            ndim = org_weight.dim()
            self._dora_norm_dims = tuple(range(1, ndim))  # norm along all dims except output
            self.register_buffer(
                "_dora_eps",
                torch.tensor(torch.finfo(torch.float32).eps),
                persistent=False,
            )
            self.dora_scale = nn.Parameter(
                torch.linalg.vector_norm(
                    org_weight, dim=self._dora_norm_dims, keepdim=True
                )
            ).float()
            # DoRA forces rebuild mode (bypass can't normalize the merged weight)
            self.bypass_mode = False

        # Alpha scaling
        if isinstance(alpha, torch.Tensor):
            alpha = alpha.detach().float().item()
        alpha = lora_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / lora_dim
        self.register_buffer("alpha", torch.tensor(alpha))

        # Scalar (not used by default, but available for compatibility)
        if use_scalar:
            init_val = scalar_init_value if scalar_init_value is not None else 0.1
            self.scalar = nn.Parameter(torch.tensor(init_val))
        else:
            self.register_buffer("scalar", torch.tensor(1.0), persistent=False)

        # T-LoRA timestep mask
        self.use_timestep_mask = use_timestep_mask
        if use_timestep_mask:
            self.register_buffer(
                "_timestep_mask",
                torch.ones(1, lora_dim, dtype=torch.float32),
                persistent=False,
            )

    @staticmethod
    def _cayley(S: torch.Tensor) -> torch.Tensor:
        """R = (I - A)(I + A)^{-1}, A = S - S^T.  2D or batched 3D."""
        A = S - S.transpose(-2, -1)
        r = A.shape[-1]
        eye = torch.eye(r, device=A.device, dtype=A.dtype)
        if A.dim() == 3:
            eye = eye.unsqueeze(0).expand_as(A)
        return torch.linalg.solve(eye + A, eye - A)

    def _compute_effective_bases(self):
        """Compute Q_eff and P_eff via batched Cayley solve.

        Returns (Q_eff, P_eff) in bf16 work dtype.
        """
        work = self.P_basis.dtype  # bf16

        # Stack S_q + S_p into one (2, r, r) solve — halves LU/TRSM launches.
        skew = torch.stack([self.S_q, self.S_p])
        A = skew - skew.transpose(-2, -1)
        R = torch.linalg.solve(self._eye_r + A, self._eye_r - A)
        R_q = R[0].to(work)
        R_p = R[1].to(work)

        Q_eff = R_q @ self.Q_basis  # (r, in) bf16
        P_eff = self.P_basis @ R_p  # (out, r) bf16
        return Q_eff, P_eff

    def make_weight(self, device=None):
        """Compute ΔW = P_eff @ diag(λ) @ Q_eff as a full weight tensor.

        Returns raw diff * scalar (NO scale).  Callers (get_diff_weight,
        _forward_rebuild_core) apply ``self.scale`` — matching the LoConModule
        convention where ``make_weight`` omits ``scale`` so it's applied once.
        """
        Q_eff, P_eff = self._compute_effective_bases()
        lam = self.lambda_layer.to(Q_eff.dtype)  # (1, r)

        # Apply T-LoRA mask if active
        if self.use_timestep_mask and self.training:
            lam = lam * self._timestep_mask.to(lam)

        # ΔW = P_eff @ diag(λ) @ Q_eff
        weight = (P_eff * lam) @ Q_eff  # (out, in)
        weight = weight.view(self.shape)
        return weight * self.scalar.to(device)

    def get_diff_weight(self, multiplier=1.0, shape=None, device=None):
        diff = self.make_weight(device=device) * self.scale * multiplier
        if shape is not None:
            diff = diff.view(shape)
        return diff, None

    def get_merged_weight(self, multiplier=1.0, shape=None, device=None):
        diff, _ = self.get_diff_weight(multiplier=1, shape=shape, device=device)
        weight = self.get_org_weight_for_compute(diff.device)
        if weight.dtype != diff.dtype:
            weight = weight.to(diff.dtype)
        if self.wd:
            return self.apply_weight_decompose(weight + diff, multiplier), None
        return weight + diff * multiplier, None

    def apply_weight_decompose(self, weight, multiplier=1):
        """DoRA: normalize merged weight to preserve pretrained magnitude.

        output = x @ (merged * ||W₀|| / ||merged||)
        where multiplier interpolation: scale = m * (s-1) + 1
        """
        if weight.dtype != self.dora_scale.dtype:
            weight = weight.to(self.dora_scale.dtype)
        weight_norm = (
            torch.linalg.vector_norm(
                weight, dim=self._dora_norm_dims, keepdim=True
            ) + self._dora_eps
        )
        scale = self.dora_scale.to(weight.device, non_blocking=True) / weight_norm
        scale = multiplier * (scale - 1) + 1
        return weight * scale

    def bypass_forward_diff(self, x, scale=1):
        """Bypass-mode diff: x → Q_eff → λ → P_eff."""
        work = self.P_basis.dtype  # bf16
        Q_eff, P_eff = self._compute_effective_bases()

        lx = F.linear(x.to(work), Q_eff)  # (B, *, r)
        lam = self.lambda_layer.to(work)
        if self.use_timestep_mask and self.training:
            lam = lam * self._timestep_mask.to(work)
        lx = lx * lam

        if self.dropout and self.training:
            lx = F.dropout(lx, p=self.dropout)

        out = F.linear(lx, P_eff)  # (B, *, out)
        return self.drop(out * self.scalar.to(x.device) * self.scale * scale)

    def bypass_forward(self, x, scale=1):
        return self.org_forward(x) + self.bypass_forward_diff(x, scale=scale)

    def _forward_rebuild_core(self, x, org_weight, bias):
        """Rebuild-mode forward — torch.compile target."""
        diff_weight = self.make_weight(device=x.device).to(x.dtype)
        if self.wd:
            # DoRA: magnitude-direction normalization
            weight = self.apply_weight_decompose(
                org_weight + diff_weight * self.scale, self.multiplier
            )
            x = self.drop(x)  # Input dropout for DoRA (per paper)
        else:
            weight = org_weight + diff_weight * self.scale * self.multiplier
        return self._call_op(x, weight, bias)

    def forward(self, x, *args, **kwargs):
        if self.module_dropout and self.training:
            if torch.rand(1) < self.module_dropout:
                return self.org_forward(x, *args, **kwargs)

        if self.bypass_mode:
            return self.bypass_forward(x, scale=self.multiplier)

        x = x.to(self._cached_dtype)
        org_weight = self.get_org_weight_for_compute(x.device)
        org_bias = self.get_org_bias_for_compute(x.device)
        bias = org_bias.to(x.dtype, non_blocking=True) if org_bias is not None else None

        return self._forward_rebuild_core(x, org_weight, bias)

    def custom_state_dict(self):
        """Save state dict.

        Two modes:
        - Distill mode (default): converts Cayley → lora_up/lora_down for
          portability.  Any standard LoRA loader can consume these keys.
        - Native mode (native_save=True): saves S_p, S_q, P_basis, Q_basis,
          lambda_layer for checkpoint resume with full fidelity.
        """
        if self.native_save:
            destination = {
                "S_p": self.S_p,
                "S_q": self.S_q,
                "P_basis": self.P_basis,
                "Q_basis": self.Q_basis,
                "lambda_layer": self.lambda_layer,
                "alpha": self.alpha,
            }
            if self.wd:
                destination["dora_scale"] = self.dora_scale
            return destination

        # Distill mode: Cayley + frozen SVD → lora_down/lora_up
        R_p = self._cayley(self.S_p.float())
        R_q = self._cayley(self.S_q.float())
        P_eff = self.P_basis.float() @ R_p  # (out, r)
        Q_eff = R_q @ self.Q_basis.float()  # (r, in)

        lam_1d = self.lambda_layer.squeeze(0).float()
        lam_abs = lam_1d.abs()
        lam_sign = lam_1d.sign()
        lam_sqrt = lam_abs.sqrt()

        # sqrt-split λ so ΔW = lora_up @ lora_down is preserved
        save_dtype = self.P_basis.dtype  # bf16
        lora_up = (
            (P_eff * (lam_sqrt * lam_sign).unsqueeze(0))
            .to(save_dtype)
            .cpu()
            .contiguous()
        )
        lora_down = (
            (Q_eff * lam_sqrt.unsqueeze(1))
            .to(save_dtype)
            .cpu()
            .contiguous()
        )

        return {
            "lora_up.weight": lora_up,
            "lora_down.weight": lora_down,
            "alpha": self.alpha,
        }

    @classmethod
    def algo_check(cls, state_dict, lora_name):
        """Detect OrthoLoRA in a state dict.

        Discriminator: .S_p key with dim == 2.
        """
        key = f"{lora_name}.S_p"
        if key in state_dict:
            return state_dict[key].ndim == 2
        # Also detect distilled form (lora_up + lora_down without S_p)
        # In that case, fall through to LoConModule detection.
        return False

    @classmethod
    def make_module_from_state_dict(
        cls, lora_name, orig_module, S_p, S_q, P_basis, Q_basis, lambda_layer, alpha
    ):
        """Reconstruct OrthoLoRA module from native checkpoint."""
        lora_dim = S_p.shape[0]
        module = cls(
            lora_name,
            orig_module,
            multiplier=1.0,
            lora_dim=lora_dim,
            alpha=float(alpha) if alpha is not None else lora_dim,
            native_save=True,
        )
        module.S_p.data.copy_(S_p)
        module.S_q.data.copy_(S_q)
        module.P_basis.copy_(P_basis)
        module.Q_basis.copy_(Q_basis)
        module.lambda_layer.data.copy_(lambda_layer)
        return module

    @classmethod
    def extract_state_dict(cls, state_dict, lora_name):
        """Extract weights matching weight_list order."""
        return [state_dict.get(f"{lora_name}.{k}", None) for k in cls.weight_list]

    def load_weight_hook(self, module: nn.Module, incompatible_keys):
        missing_keys = incompatible_keys.missing_keys
        for key in missing_keys[:]:
            if "scalar" in key or "timestep_mask" in key or "dora_scale" in key:
                missing_keys.remove(key)
        if isinstance(self.scalar, nn.Parameter):
            self.scalar.data.copy_(torch.ones_like(self.scalar))

    @torch.no_grad()
    def apply_max_norm(self, max_norm, device=None):
        diff, _ = self.get_diff_weight(multiplier=1.0, device=device)
        orig_norm = diff.norm() * self.scale
        norm = torch.clamp(orig_norm, max=max_norm / 2)
        desired = torch.clamp(norm, max=max_norm)
        ratio = desired.cpu() / norm.cpu()
        scaled = norm != desired
        if scaled:
            self.scalar *= ratio
            return scaled, orig_norm * ratio
        return 0, orig_norm

    @torch.no_grad()
    def get_norm(self, device=None):
        diff, _ = self.get_diff_weight(multiplier=1.0, device=device)
        return diff.norm()

    def regularization(self):
        """No-op: Cayley guarantees orthogonality structurally."""
        zero = torch.tensor(0.0, device=self.S_p.device)
        return zero, zero
