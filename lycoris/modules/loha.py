import math

import torch
import torch.nn as nn

from .base import LycorisBaseModule
from ..functional.loha import diff_weight as loha_diff_weight
from ..logging import logger

from typing import Optional


class LohaModule(LycorisBaseModule):
    name = "loha"
    support_module = {
        "linear",
        "conv1d",
        "conv2d",
        "conv3d",
    }
    weight_list = [
        "hada_w1_a",
        "hada_w1_b",
        "hada_w2_a",
        "hada_w2_b",
        "hada_t1",
        "hada_t2",
        "alpha",
        "dora_scale",
    ]
    weight_list_det = ["hada_w1_a"]

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
        orthogonalize=False,
        orthogonal_init=False,
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
            ggpo_sigma
        )
        if self.module_type not in self.support_module:
            raise ValueError(f"{self.module_type} is not supported in LoHa algo.")
        self.lora_name = lora_name
        self.lora_dim = lora_dim
        self.tucker = False
        self.rs_lora = rs_lora
        self.use_orthogonal_weights = orthogonalize
        if orthogonalize and not orthogonal_init:
            orthogonal_init = True
        self.use_orthogonal_init = orthogonal_init
        if self.use_orthogonal_weights and not use_scalar:
            use_scalar = True

        w_shape = self.shape
        if self.module_type.startswith("conv"):
            in_dim = org_module.in_channels
            k_size = org_module.kernel_size
            out_dim = org_module.out_channels
            self.shape = (out_dim, in_dim, *k_size)
            self.tucker = use_tucker and any(i != 1 for i in k_size)
            if self.tucker:
                w_shape = (out_dim, in_dim, *k_size)
            else:
                w_shape = (out_dim, in_dim * torch.tensor(k_size).prod().item())

        if self.tucker:
            self.hada_t1 = nn.Parameter(torch.empty(lora_dim, lora_dim, *w_shape[2:]))
            self.hada_w1_a = nn.Parameter(
                torch.empty(lora_dim, w_shape[0])
            )  # out_dim, 1-mode
            self.hada_w1_b = nn.Parameter(
                torch.empty(lora_dim, w_shape[1])
            )  # in_dim , 2-mode

            self.hada_t2 = nn.Parameter(torch.empty(lora_dim, lora_dim, *w_shape[2:]))
            self.hada_w2_a = nn.Parameter(
                torch.empty(lora_dim, w_shape[0])
            )  # out_dim, 1-mode
            self.hada_w2_b = nn.Parameter(
                torch.empty(lora_dim, w_shape[1])
            )  # in_dim , 2-mode
        else:
            self.hada_w1_a = nn.Parameter(torch.empty(w_shape[0], lora_dim))
            self.hada_w1_b = nn.Parameter(torch.empty(lora_dim, w_shape[1]))

            self.hada_w2_a = nn.Parameter(torch.empty(w_shape[0], lora_dim))
            self.hada_w2_b = nn.Parameter(torch.empty(lora_dim, w_shape[1]))

        self.wd = weight_decompose
        self.wd_on_output = wd_on_output
        if self.wd:
            org_weight = org_module.weight.cpu().clone().float()
            ndim = org_weight.dim()
            self.dora_norm_dims = ndim - 1

            # Pre-compute dimension tuples for torch.linalg.vector_norm.
            if wd_on_output:
                self._dora_norm_dims = tuple(range(1, ndim))
            else:
                self._dora_norm_dims = (0,) + tuple(range(2, ndim))

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

        if self.dropout:
            print("[WARN]LoHa/LoKr haven't implemented normal dropout yet.")

        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().float().numpy()  # without casting, bf16 causes error
        alpha = lora_dim if alpha is None or alpha == 0 else alpha

        r_factor = lora_dim
        if self.rs_lora:
            r_factor = math.sqrt(r_factor)

        self.scale = alpha / r_factor

        self.register_buffer("alpha", torch.tensor(alpha * (lora_dim / r_factor)))

        if use_scalar:
            init_val = scalar_init_value if scalar_init_value is not None else 0.1
            self.scalar = nn.Parameter(torch.tensor(init_val))
        else:
            self.register_buffer("scalar", torch.tensor(1.0), persistent=False)
        # Need more experiments on init method
        if self.use_orthogonal_init:
            if self.tucker:
                torch.nn.init.orthogonal_(self.hada_t1)
                torch.nn.init.orthogonal_(self.hada_t2)
            torch.nn.init.orthogonal_(self.hada_w1_b)
            torch.nn.init.orthogonal_(self.hada_w1_a)
            torch.nn.init.orthogonal_(self.hada_w2_b)
            torch.nn.init.orthogonal_(self.hada_w2_a)
        else:
            if self.tucker:
                torch.nn.init.normal_(self.hada_t1, std=0.1)
                torch.nn.init.normal_(self.hada_t2, std=0.1)
            torch.nn.init.normal_(self.hada_w1_b, std=1)
            torch.nn.init.normal_(self.hada_w1_a, std=0.1)
            torch.nn.init.normal_(self.hada_w2_b, std=1)
            if use_scalar:
                torch.nn.init.normal_(self.hada_w2_a, std=0.1)
            else:
                torch.nn.init.constant_(self.hada_w2_a, 0)

        # SVD segment initialization
        svd_segment = kwargs.get("svd_segment", None)
        if svd_segment is not None:
            if self.use_orthogonal_init:
                logger.warning(
                    f"svd_segment='{svd_segment}' and orthogonal_init=True are mutually exclusive. "
                    f"SVD segment init will replace orthogonal initialization for {self.lora_name}."
                )
            self._init_svd_segment(svd_segment)

    @torch.no_grad()
    def _init_svd_segment(self, segment: str):
        """Initialize LoHa weights from a segment of the SVD spectrum.

        Strategy: set the first Hadamard pair (w1a, w1b) to the SVD segment,
        and the second pair (w2a, w2b) to produce an all-ones matrix so that
        the Hadamard product equals the SVD segment.
        """
        if self.tucker:
            logger.warning(
                f"SVD segment init is not supported for tucker decomposition "
                f"(module {self.lora_name}), skipping."
            )
            return

        org_weight_2d = self._get_weight_2d(self.org_module[0])
        result = self._compute_svd_segment(org_weight_2d, self.lora_dim, segment)
        if result is None:
            logger.warning(
                f"Weight {self.lora_name} has fewer singular values than "
                f"lora_dim={self.lora_dim}, skipping SVD segment init."
            )
            return
        Vr, Sr, Uhr = result

        sqrt_Sr = torch.sqrt(Sr)

        # First pair: encode the SVD segment
        self.hada_w1_a.data.copy_((Vr @ torch.diag(sqrt_Sr)).to(self.hada_w1_a.dtype))
        self.hada_w1_b.data.copy_((torch.diag(sqrt_Sr) @ Uhr).to(self.hada_w1_b.dtype))

        # Second pair: produce all-ones matrix via outer product
        # ones(out, r)/r @ ones(r, in) = ones(out, in)
        self.hada_w2_a.data.fill_(1.0 / self.lora_dim)
        self.hada_w2_b.data.fill_(1.0)

        orig_shape = self.org_module[0].weight.shape
        diff = (Vr @ torch.diag(Sr) @ Uhr).reshape(orig_shape)
        self.org_module[0].weight.data -= (
            diff.to(self.org_module[0].weight.dtype) * self.scale
        )
        logger.info(f"SVD segment init ({segment}): {self.lora_name}")

    @classmethod
    def make_module_from_state_dict(
        cls, lora_name, orig_module, w1a, w1b, w2a, w2b, t1, t2, alpha, dora_scale
    ):
        module = cls(
            lora_name,
            orig_module,
            1,
            w1b.size(0),
            float(alpha),
            use_tucker=t1 is not None,
            weight_decompose=dora_scale is not None,
        )
        module.hada_w1_a.copy_(w1a)
        module.hada_w1_b.copy_(w1b)
        module.hada_w2_a.copy_(w2a)
        module.hada_w2_b.copy_(w2b)
        if t1 is not None:
            module.hada_t1.copy_(t1)
            module.hada_t2.copy_(t2)
        if dora_scale is not None:
            module.dora_scale.copy_(dora_scale)
        return module

    def load_weight_hook(self, module: nn.Module, incompatible_keys):
        missing_keys = incompatible_keys.missing_keys
        for key in missing_keys:
            if "scalar" in key:
                del missing_keys[missing_keys.index(key)]
        if isinstance(self.scalar, nn.Parameter):
            self.scalar.data.copy_(torch.ones_like(self.scalar))
        elif getattr(self, "scalar", None) is not None:
            self.scalar.copy_(torch.ones_like(self.scalar))
        else:
            self.register_buffer(
                "scalar", torch.ones_like(self.scalar), persistent=False
            )

    def get_weight(self, shape):
        scale = torch.tensor(
            self.scale, dtype=self.hada_w1_b.dtype, device=self.hada_w1_b.device
        )
        # Orthogonalize weights on the fly if runtime orthogonalization is enabled
        w1_b = self._orthogonalize(self.hada_w1_b)
        w1_a = self._orthogonalize(self.hada_w1_a)
        w2_b = self._orthogonalize(self.hada_w2_b)
        w2_a = self._orthogonalize(self.hada_w2_a)
        if self.tucker:
            t1 = self._orthogonalize(self.hada_t1)
            t2 = self._orthogonalize(self.hada_t2)
            weight = loha_diff_weight(
                w1_b,
                w1_a,
                w2_b,
                w2_a,
                t1,
                t2,
                gamma=scale,
            )
        else:
            weight = loha_diff_weight(
                w1_b,
                w1_a,
                w2_b,
                w2_a,
                None,
                None,
                gamma=scale,
            )
        if shape is not None:
            weight = weight.reshape(shape)
        if self.training and self.rank_dropout:
            drop = (torch.rand(weight.size(0)) > self.rank_dropout).to(weight.dtype)
            # Use pre-computed _rank_drop_shape instead of dynamic
            # len(weight.shape[1:]) splat — avoids graph breaks from
            # Python container traversal on symbolic tensor shapes.
            drop = drop.view(self._rank_drop_shape).to(weight.device)
            if self.rank_dropout_scale:
                drop /= drop.mean()
            weight *= drop
        return weight

    def get_diff_weight(self, multiplier=1, shape=None, device=None):
        scale = self.scale * multiplier
        diff = self.get_weight(shape) * scale
        if device is not None:
            diff = diff.to(device)
        return diff, None

    def get_merged_weight(self, multiplier=1, shape=None, device=None):
        diff = self.get_diff_weight(multiplier=1, shape=shape, device=device)[0]

        weight = self.get_org_weight_for_compute(diff.device)

        if weight.dtype != diff.dtype:
            weight = weight.to(diff.dtype)

        if self.wd:
            merged = self.apply_weight_decompose(weight + diff, multiplier)
        else:
            merged = weight + diff * multiplier
        return merged, None

    def apply_weight_decompose(self, weight, multiplier=1):
        if weight.dtype != self.dora_scale.dtype:
            weight = weight.to(self.dora_scale.dtype)

        # Use torch.linalg.vector_norm with pre-computed dim tuple and
        # keepdim=True.  Eliminates the reshape→norm→reshape + transpose chain
        # that materialises a full contiguous copy for wd_on_output=False.
        weight_norm = (
            torch.linalg.vector_norm(
                weight, dim=self._dora_norm_dims, keepdim=True
            ) + self._dora_eps
        )

        scale = self.dora_scale.to(weight.device, non_blocking=True) / weight_norm
        # Always apply: when multiplier==1 this simplifies to scale unchanged.
        # Avoids data-dependent branch that causes torch.compile graph breaks.
        scale = multiplier * (scale - 1) + 1

        return weight * scale

    def custom_state_dict(self):
        destination = {}
        destination["alpha"] = self.alpha
        if self.wd:
            destination["dora_scale"] = self.dora_scale
        destination["hada_w1_a"] = self.hada_w1_a * self.scalar.to(device=self.hada_w1_a.device, non_blocking=True)
        destination["hada_w1_b"] = self.hada_w1_b
        destination["hada_w2_a"] = self.hada_w2_a
        destination["hada_w2_b"] = self.hada_w2_b
        if self.tucker:
            destination["hada_t1"] = self.hada_t1
            destination["hada_t2"] = self.hada_t2
        return destination

    @torch.no_grad()
    def apply_max_norm(self, max_norm, device=None):
        orig_norm = (self.get_weight(self.shape) * self.scalar).norm()
        norm = torch.clamp(orig_norm, max_norm / 2)
        desired = torch.clamp(norm, max=max_norm)
        ratio = desired.cpu() / norm.cpu()

        scaled = norm != desired
        if scaled:
            self.scalar *= ratio
            return scaled, orig_norm * ratio
        else:
            return 0, orig_norm
        
    @torch.no_grad()
    def get_norm(self, device=None):
        weight = self.get_weight(self.shape)
        # Norm before scale determined by self.scalar
        unscaled_norm = weight.norm()
        return unscaled_norm

    def bypass_forward_diff(self, x, scale=1):
        diff_weight = self.get_weight(self.shape) * self.scalar * scale
        return self.drop(self._call_op(x, diff_weight))

    def bypass_forward(self, x, scale=1):
        return self.org_forward(x) + self.bypass_forward_diff(x, scale=scale)

    def _forward_rebuild_core(self, x, org_weight, bias):
        """Rebuild-mode forward pass — the torch.compile target.

        Computes diff weight via Hadamard product of two low-rank pairs,
        merges with the pre-fetched original weight, and runs the fused
        linear/conv operation.  All inputs are pre-fetched GPU tensors.
        """
        diff_weight = self.get_weight(self.shape).to(self._cached_dtype) * self.scalar
        weight = org_weight
        multiplier = self.multiplier_buf
        if self.wd:
            weight = self.apply_weight_decompose(weight + diff_weight, multiplier)
        else:
            weight = weight + diff_weight * multiplier
        return self._call_op(x, weight, bias)

    def forward(self, x: torch.Tensor, *args, **kwargs):
        if self.module_dropout and self.training:
            if torch.rand(1) < self.module_dropout:
                bias = self.get_org_bias_for_compute(x.device)
                if bias is not None:
                    bias = bias.to(x.dtype, non_blocking=True)

                return self._call_op(
                    x,
                    self.get_org_weight_for_compute(x.device).to(self._cached_dtype, non_blocking=True).data,
                    bias,
                )
        if self.bypass_mode:
            return self.bypass_forward(x, scale=self.multiplier)

        x = x.to(self._cached_dtype)
        org_weight = self.get_org_weight_for_compute(x.device).data.to(self._cached_dtype, non_blocking=True)
        org_bias = self.get_org_bias_for_compute(x.device)
        # Pre-resolve bias to real tensor or None (avoids numel check in compiled graph)
        if org_bias is not None:
            bias = org_bias.to(x.dtype, non_blocking=True)
        else:
            bias = None

        return self._forward_rebuild_core(x, org_weight, bias)
