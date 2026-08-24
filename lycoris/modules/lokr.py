import math
from functools import cache

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import LycorisBaseModule
from ..functional import factorization, rebuild_tucker
from ..functional.lokr import make_kron
from ..logging import logger

from typing import Optional


@cache
def logging_force_full_matrix(lora_dim, dim, factor):
    logger.warning(
        f"lora_dim {lora_dim} is too large for"
        f" dim={dim} and {factor=}"
        ", using full matrix mode."
    )


def factorization_with_warning(
    dimension, factor, lora_name, dimension_name, unbalanced=False
):
    factors = factorization(dimension, factor)
    if unbalanced:
        factors = factors[::-1]

    if factor > 0 and dimension % factor != 0:
        logger.warning(
            f"LoKr module '{lora_name}': requested factor={factor} does not "
            f"evenly divide {dimension_name} dimension={dimension}; using "
            f"factor pair={factors} (effective factor={factors[0]})."
        )
    return factors


class LokrModule(LycorisBaseModule):
    name = "kron"
    support_module = {
        "linear",
        "conv1d",
        "conv2d",
        "conv3d",
    }
    weight_list = [
        "lokr_w1",
        "lokr_w1_a",
        "lokr_w1_b",
        "lokr_w2",
        "lokr_w2_a",
        "lokr_w2_b",
        "lokr_t1",
        "lokr_t2",
        "alpha",
        "dora_scale",
        # These are optional compatibility metadata.  They are emitted only
        # for configurations whose orientation cannot be recovered from the
        # factor shapes (currently unbalanced factorization).
        "lokr_factor",
        "lokr_unbalanced",
    ]
    weight_list_det = ["lokr_w1", "lokr_w1_a"]

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
        decompose_both=False,
        factor: int = -1,  # factorization factor
        rank_dropout_scale=False,
        weight_decompose=False,
        wd_on_output=True,
        full_matrix=False,
        bypass_mode=None,
        rs_lora=False,
        unbalanced_factorization=False,
        ggpo_beta: Optional[float] = None,
        ggpo_sigma: Optional[float] = None,
        orthogonalize=False,
        orthogonal_init=False,
        scalar_type: str = "scalar",
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
        # DoRA is a weight-space operation.  Applying it to a direct-factor
        # bypass would omit the magnitude normalization, so keep the same
        # invariant here as the network wrappers do.
        if weight_decompose and self.bypass_mode:
            self.bypass_mode = False
            logger.info(
                "Because weight decomposition (DoRA) is enabled, bypass mode "
                "has been disabled"
            )
        if self.module_type not in self.support_module:
            raise ValueError(f"{self.module_type} is not supported in LoKr algo.")

        factor = int(factor)
        self.lokr_factor = factor
        self.lora_dim = lora_dim
        self.tucker = False
        self.use_w1 = False
        self.use_w2 = False
        self.full_matrix = full_matrix
        self.rs_lora = rs_lora
        self.unbalanced_factorization = bool(unbalanced_factorization)
        self.use_orthogonal_weights = orthogonalize
        if orthogonalize and not orthogonal_init:
            orthogonal_init = True
        self.use_orthogonal_init = orthogonal_init
        if self.use_orthogonal_weights and not use_scalar:
            use_scalar = True

        if self.module_type.startswith("conv"):
            in_dim = org_module.in_channels
            k_size = org_module.kernel_size
            out_dim = org_module.out_channels
            self.shape = (out_dim, in_dim, *k_size)

            in_m, in_n = factorization_with_warning(
                in_dim, factor, lora_name, "input"
            )
            out_l, out_k = factorization_with_warning(
                out_dim, factor, lora_name, "output", unbalanced_factorization
            )
            shape = ((out_l, out_k), (in_m, in_n), *k_size)  # ((a, b), (c, d), *k_size)
            self.tucker = use_tucker and any(i != 1 for i in k_size)
            if (
                decompose_both
                and lora_dim < max(shape[0][0], shape[1][0]) / 2
                and not self.full_matrix
            ):
                self.lokr_w1_a = nn.Parameter(torch.empty(shape[0][0], lora_dim))
                self.lokr_w1_b = nn.Parameter(torch.empty(lora_dim, shape[1][0]))
            else:
                self.use_w1 = True
                self.lokr_w1 = nn.Parameter(
                    torch.empty(shape[0][0], shape[1][0])
                )  # a*c, 1-mode

            if lora_dim >= max(shape[0][1], shape[1][1]) / 2 or self.full_matrix:
                if not self.full_matrix:
                    logging_force_full_matrix(lora_dim, max(in_dim, out_dim), factor)
                self.use_w2 = True
                self.lokr_w2 = nn.Parameter(
                    torch.empty(shape[0][1], shape[1][1], *k_size)
                )
            elif self.tucker:
                self.lokr_t2 = nn.Parameter(torch.empty(lora_dim, lora_dim, *shape[2:]))
                self.lokr_w2_a = nn.Parameter(
                    torch.empty(lora_dim, shape[0][1])
                )  # b, 1-mode
                self.lokr_w2_b = nn.Parameter(
                    torch.empty(lora_dim, shape[1][1])
                )  # d, 2-mode
            else:  # Conv2d not tucker
                # bigger part. weight and LoRA. [b, dim] x [dim, d*k1*k2]
                self.lokr_w2_a = nn.Parameter(torch.empty(shape[0][1], lora_dim))
                self.lokr_w2_b = nn.Parameter(
                    torch.empty(
                        lora_dim, shape[1][1] * math.prod(shape[2:])
                    )
                )
                # w1 ⊗ (w2_a x w2_b) = (a, b)⊗((c, dim)x(dim, d*k1*k2)) = (a, b)⊗(c, d*k1*k2) = (ac, bd*k1*k2)
        else:  # Linear
            in_dim = org_module.in_features
            out_dim = org_module.out_features
            self.shape = (out_dim, in_dim)

            in_m, in_n = factorization_with_warning(
                in_dim, factor, lora_name, "input"
            )
            out_l, out_k = factorization_with_warning(
                out_dim, factor, lora_name, "output", unbalanced_factorization
            )
            shape = (
                (out_l, out_k),
                (in_m, in_n),
            )  # ((a, b), (c, d)), out_dim = a*c, in_dim = b*d
            # smaller part. weight scale
            if (
                decompose_both
                and lora_dim < max(shape[0][0], shape[1][0]) / 2
                and not self.full_matrix
            ):
                self.lokr_w1_a = nn.Parameter(torch.empty(shape[0][0], lora_dim))
                self.lokr_w1_b = nn.Parameter(torch.empty(lora_dim, shape[1][0]))
            else:
                self.use_w1 = True
                self.lokr_w1 = nn.Parameter(
                    torch.empty(shape[0][0], shape[1][0])
                )  # a*c, 1-mode
            if lora_dim < max(shape[0][1], shape[1][1]) / 2 and not self.full_matrix:
                # bigger part. weight and LoRA. [b, dim] x [dim, d]
                self.lokr_w2_a = nn.Parameter(torch.empty(shape[0][1], lora_dim))
                self.lokr_w2_b = nn.Parameter(torch.empty(lora_dim, shape[1][1]))
                # w1 ⊗ (w2_a x w2_b) = (a, b)⊗((c, dim)x(dim, d)) = (a, b)⊗(c, d) = (ac, bd)
            else:
                if not self.full_matrix:
                    logging_force_full_matrix(lora_dim, max(in_dim, out_dim), factor)
                self.use_w2 = True
                self.lokr_w2 = nn.Parameter(torch.empty(shape[0][1], shape[1][1]))

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

        self.dropout = dropout
        if dropout:
            print("[WARN]LoHa/LoKr haven't implemented normal dropout yet.")
        self.rank_dropout = rank_dropout
        self.rank_dropout_scale = rank_dropout_scale
        self.module_dropout = module_dropout

        if isinstance(alpha, torch.Tensor):
            alpha = alpha.detach().float().numpy()  # without casting, bf16 causes error
        alpha = lora_dim if alpha is None or alpha == 0 else alpha
        if self.use_w2 and self.use_w1:
            # use scale = 1
            alpha = lora_dim

        r_factor = lora_dim
        if self.rs_lora:
            r_factor = math.sqrt(r_factor)

        self.scale = alpha / r_factor

        self.register_buffer("alpha", torch.tensor(alpha * (lora_dim / r_factor)))

        # LoKr: vector scalar modes not supported due to Kronecker structure
        if use_scalar and scalar_type in ("row", "column", "row_column"):
            logger.warning(
                f"LoKr: scalar_type='{scalar_type}' is not supported due to Kronecker structure. "
                f"Falling back to scalar_type='scalar'."
            )
            scalar_type = "scalar"

        if use_scalar:
            init_val = scalar_init_value if scalar_init_value is not None else 0.1
            self.scalar = nn.Parameter(torch.tensor(init_val))
        else:
            self.register_buffer("scalar", torch.tensor(1.0), persistent=False)

        # Weight initialization
        if self.use_w2:
            if use_scalar:
                if self.use_orthogonal_init:
                    torch.nn.init.orthogonal_(self.lokr_w2)
                else:
                    torch.nn.init.kaiming_uniform_(self.lokr_w2, a=math.sqrt(5))
            else:
                torch.nn.init.constant_(self.lokr_w2, 0)
        else:
            if self.tucker:
                if self.use_orthogonal_init:
                    torch.nn.init.orthogonal_(self.lokr_t2)
                else:
                    torch.nn.init.kaiming_uniform_(self.lokr_t2, a=math.sqrt(5))

            if self.use_orthogonal_init:
                torch.nn.init.orthogonal_(self.lokr_w2_a)
            else:
                torch.nn.init.kaiming_uniform_(self.lokr_w2_a, a=math.sqrt(5))

            if use_scalar:
                if self.use_orthogonal_init:
                    torch.nn.init.orthogonal_(self.lokr_w2_b)
                else:
                    torch.nn.init.kaiming_uniform_(self.lokr_w2_b, a=math.sqrt(5))
            else:
                torch.nn.init.constant_(self.lokr_w2_b, 0)

        if self.use_w1:
            if self.use_orthogonal_init:
                torch.nn.init.orthogonal_(self.lokr_w1)
            else:
                torch.nn.init.kaiming_uniform_(self.lokr_w1, a=math.sqrt(5))
        else:
            if self.use_orthogonal_init:
                torch.nn.init.orthogonal_(self.lokr_w1_a)
                torch.nn.init.orthogonal_(self.lokr_w1_b)
            else:
                torch.nn.init.kaiming_uniform_(self.lokr_w1_a, a=math.sqrt(5))
                torch.nn.init.kaiming_uniform_(self.lokr_w1_b, a=math.sqrt(5))

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
        """Initialize LoKr weights from a segment of the SVD spectrum.

        Uses a Kronecker-SVD approximation: reshape the SVD segment into the
        Kronecker factorization structure and use rank-1 SVD to split into
        w1 and w2 factors.  Tucker convolutions are skipped.
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

        # Build the SVD segment matrix M = V_r * diag(S_r) * U_r^T
        M = Vr @ torch.diag(Sr) @ Uhr  # (out_dim, in_dim)

        orig_shape = self.org_module[0].weight.shape
        out_dim, in_dim = orig_shape[0], orig_shape[1]
        k = orig_shape[2:] if len(orig_shape) > 2 else ()

        # Determine Kronecker factorization dimensions
        factor = self.lokr_factor
        from ..functional import factorization
        in_m, in_n = factorization(in_dim, factor)
        out_l, out_k = factorization(out_dim, factor)
        if self.unbalanced_factorization:
            out_l, out_k = out_k, out_l

        # Keep convolution kernel elements in the second Kronecker factor.
        # The old implementation discarded them while reshaping M, which
        # both changed the element count and made ordinary k×k convolutions
        # fail before the factors were initialized.
        M_reshaped = M.reshape(out_l, out_k, in_m, in_n, *k)
        permute_order = (0, 2, 1, 3, *range(4, 4 + len(k)))
        M_perm = M_reshaped.permute(*permute_order).reshape(
            out_l * in_m, out_k * in_n * (math.prod(k) if k else 1)
        )
        U_k, S_k, Vh_k = torch.linalg.svd(M_perm.float(), full_matrices=False)

        sqrt_S0 = torch.sqrt(S_k[0])
        w1_init = (U_k[:, 0] * sqrt_S0).reshape(out_l, in_m)
        w2_init = (Vh_k[0, :] * sqrt_S0).reshape(out_k, in_n, *k)

        # Apply to the appropriate weight parameters
        if self.use_w1:
            self.lokr_w1.data.copy_(w1_init.to(self.lokr_w1.dtype))
        else:
            # Factor w1 into w1_a @ w1_b via SVD
            U1, S1, Vh1 = torch.linalg.svd(w1_init, full_matrices=False)
            r1 = min(self.lokr_w1_a.shape[1], len(S1))
            self.lokr_w1_a.data.zero_()
            self.lokr_w1_b.data.zero_()
            self.lokr_w1_a.data[:, :r1].copy_(
                (U1[:, :r1] @ torch.diag(torch.sqrt(S1[:r1]))).to(self.lokr_w1_a.dtype)
            )
            self.lokr_w1_b.data[:r1].copy_(
                (torch.diag(torch.sqrt(S1[:r1])) @ Vh1[:r1, :]).to(self.lokr_w1_b.dtype)
            )

        if self.use_w2:
            self.lokr_w2.data.copy_(w2_init.to(self.lokr_w2.dtype))
        else:
            # Factor w2 into w2_a @ w2_b via SVD
            w2_init_2d = w2_init.reshape(out_k, -1)
            U2, S2, Vh2 = torch.linalg.svd(w2_init_2d, full_matrices=False)
            r2 = min(self.lokr_w2_a.shape[0], len(S2))
            self.lokr_w2_a.data.zero_()
            self.lokr_w2_b.data.zero_()
            self.lokr_w2_a.data[:r2].copy_(
                (U2[:, :r2] @ torch.diag(torch.sqrt(S2[:r2]))).T.to(self.lokr_w2_a.dtype)
            )
            w2b_2d = torch.diag(torch.sqrt(S2[:r2])) @ Vh2[:r2, :]
            self.lokr_w2_b.data[:r2].copy_(
                w2b_2d.reshape(r2, -1).to(self.lokr_w2_b.dtype)
            )

        # Adjust org_weight with the *actual* initialized factors.  Using
        # w1_init/w2_init here is wrong when one of the factor matrices has
        # insufficient rank, and also ignored the optional scalar parameter.
        from ..functional.lokr import make_kron
        if self.use_w1:
            w1_actual = self.lokr_w1
        else:
            w1_actual = self.lokr_w1_a @ self.lokr_w1_b
        if self.use_w2:
            w2_actual = self.lokr_w2
        else:
            w2_actual = self.lokr_w2_a @ self.lokr_w2_b
            if k:
                w2_actual = w2_actual.view(out_k, in_n, *k)
        diff = make_kron(w1_actual, w2_actual, 1.0).to(
            self.org_module[0].weight.device, dtype=M.dtype
        )
        scalar = self.scalar.to(diff.device, dtype=diff.dtype)
        self.org_module[0].weight.data -= (
            diff * self.scale * scalar
        ).to(self.org_module[0].weight.dtype)
        logger.info(f"SVD segment init ({segment}): {self.lora_name}")

    @classmethod
    def make_module_from_state_dict(
        cls,
        lora_name,
        orig_module,
        w1,
        w1a,
        w1b,
        w2,
        w2a,
        w2b,
        _,
        t2,
        alpha,
        dora_scale,
        lokr_factor=None,
        lokr_unbalanced=None,
    ):
        is_conv = isinstance(orig_module, (nn.Conv1d, nn.Conv2d, nn.Conv3d))
        kernel_elements = math.prod(orig_module.kernel_size) if is_conv else 1

        # Functional LoKr checkpoints may retain the convolution kernel
        # dimensions on w2_b, while module checkpoints flatten them.  Use one
        # canonical representation before inferring channel factors and
        # copying into the module's 2-D parameter.
        if is_conv and t2 is None and w2b is not None and w2b.dim() > 2:
            w2b = w2b.reshape(w2b.size(0), -1)

        # Tucker stores rank in dimension 0; the non-Tucker convolution
        # representation stores it in w2a's last dimension.
        if w1a is not None:
            lora_dim = w1a.size(1)
        elif t2 is not None:
            lora_dim = t2.size(0)
        elif w2a is not None:
            lora_dim = w2a.size(1)
        else:
            lora_dim = 1

        w1_shape = w1.shape if w1 is not None else (w1a.size(0), w1b.size(1))
        if w2 is not None:
            w2_shape = (w2.size(0), w2.size(1))
        elif t2 is not None:
            w2_shape = (w2a.size(1), w2b.size(1))
        elif is_conv:
            # w2b is (rank, in_n * kernel_elements) in the serialized
            # non-Tucker convolution format.
            w2_shape = (w2a.size(0), w2b.size(1) // kernel_elements)
        else:
            w2_shape = (w2a.size(0), w2b.size(1))

        out_dim = orig_module.out_channels if is_conv else orig_module.out_features
        in_dim = orig_module.in_channels if is_conv else orig_module.in_features
        out_pair = (w1_shape[0], w2_shape[0])
        in_pair = (w1_shape[1], w2_shape[1])

        if lokr_factor is not None:
            factor = int(lokr_factor.item())
        else:
            # Recover an effective factor from the channel dimensions only.
            # In particular, never use max(w2.shape), since conv kernel
            # dimensions are not factorization dimensions.
            candidates = set()
            for dimension in (in_dim, out_dim):
                for value in range(1, int(math.sqrt(dimension)) + 1):
                    if dimension % value == 0:
                        candidates.add(value)
                        candidates.add(dimension // value)
            factor = None
            for candidate in sorted(candidates):
                if factorization(in_dim, candidate) == tuple(sorted(in_pair)) and factorization(out_dim, candidate) == tuple(sorted(out_pair)):
                    factor = candidate
                    break
            if factor is None:
                factor = min(in_pair)

        unbalanced = bool(lokr_unbalanced.item()) if lokr_unbalanced is not None else False
        full_matrix = w2 is not None and w1a is None

        module = cls(
            lora_name,
            orig_module,
            1,
            lora_dim,
            float(alpha),
            use_tucker=t2 is not None,
            decompose_both=w1a is not None,
            factor=factor,
            weight_decompose=dora_scale is not None,
            full_matrix=full_matrix,
            unbalanced_factorization=unbalanced,
        )
        if w1 is not None:
            module.lokr_w1.data.copy_(w1)
        else:
            module.lokr_w1_a.data.copy_(w1a)
            module.lokr_w1_b.data.copy_(w1b)
        if w2 is not None:
            module.lokr_w2.data.copy_(w2)
        else:
            module.lokr_w2_a.data.copy_(w2a)
            module.lokr_w2_b.data.copy_(w2b)
        if t2 is not None:
            module.lokr_t2.data.copy_(t2)
        if dora_scale is not None:
            module.dora_scale.data.copy_(dora_scale)
        # Scalars are baked into the serialized factors, matching the normal
        # load-state-dict hook.  Reset here as well for direct factory users.
        if isinstance(module.scalar, nn.Parameter):
            module.scalar.data.fill_(1)
        return module

    def load_weight_hook(self, module: nn.Module, incompatible_keys):
        missing_keys = incompatible_keys.missing_keys
        missing_keys[:] = [
            key for key in missing_keys
            if "scalar" not in key and "lokr_factor" not in key
            and "lokr_unbalanced" not in key
        ]
        # Optional orientation metadata is consumed while constructing a
        # module from a checkpoint and is intentionally not a live parameter
        # or buffer.  Ignore it when loading into an already-created module.
        incompatible_keys.unexpected_keys[:] = [
            key for key in incompatible_keys.unexpected_keys
            if not key.endswith("lokr_factor")
            and not key.endswith("lokr_unbalanced")
        ]
        if isinstance(self.scalar, nn.Parameter):
            self.scalar.data.copy_(torch.ones_like(self.scalar))
        elif getattr(self, "scalar", None) is not None:
            self.scalar.copy_(torch.ones_like(self.scalar))
        else:
            self.register_buffer(
                "scalar", torch.ones_like(self.scalar), persistent=False
            )

    def get_weight(self, shape):
        """Return the scaled Kronecker update.

        This method owns ``self.scale``.  Callers add only their runtime
        multiplier, which keeps rebuild, merge, parametrization, and bypass
        numerically consistent.
        """
        if self.use_w1:
            w1 = self._orthogonalize(self.lokr_w1)
        else:
            w1_a_ortho = self._orthogonalize(self.lokr_w1_a)
            w1_b_ortho = self._orthogonalize(self.lokr_w1_b)
            w1 = w1_a_ortho @ w1_b_ortho

        if self.use_w2:
            w2 = self._orthogonalize(self.lokr_w2)
        else:
            w2_a_ortho = self._orthogonalize(self.lokr_w2_a)
            w2_b_ortho = self._orthogonalize(self.lokr_w2_b)
            if self.tucker:
                # We don't orthogonalize the core tensor `lokr_t2`
                w2 = rebuild_tucker(self.lokr_t2, w2_a_ortho, w2_b_ortho)
            else:
                w2 = w2_a_ortho @ w2_b_ortho
        
        weight = make_kron(w1, w2, self.scale)
        dtype = weight.dtype
        if shape is not None:
            weight = weight.view(shape)
        if self.training and self.rank_dropout:
            drop = (torch.rand(weight.size(0), device=weight.device) > self.rank_dropout).to(dtype)
            drop = drop.view(-1, *[1] * len(weight.shape[1:]))
            if self.rank_dropout_scale:
                drop /= drop.mean().clamp_min(torch.finfo(dtype).eps)
            weight *= drop
        if self.training and self.dropout:
            weight = self.drop(weight)
        return weight

    def get_diff_weight(self, multiplier=1, shape=None, device=None):
        # get_weight() already applies alpha/rank.  Do not apply self.scale a
        # second time in merge or parametrization paths.
        diff = self.get_weight(shape) * self.scalar * multiplier
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
        if self.use_w1:
            destination["lokr_w1"] = self.lokr_w1 * self.scalar.to(device=self.lokr_w1.device, non_blocking=True)
        else:
            destination["lokr_w1_a"] = self.lokr_w1_a * self.scalar.to(device=self.lokr_w1_a.device, non_blocking=True)
            destination["lokr_w1_b"] = self.lokr_w1_b

        if self.use_w2:
            destination["lokr_w2"] = self.lokr_w2
        else:
            destination["lokr_w2_a"] = self.lokr_w2_a
            destination["lokr_w2_b"] = self.lokr_w2_b
            if self.tucker:
                destination["lokr_t2"] = self.lokr_t2
        if self.unbalanced_factorization:
            destination["lokr_factor"] = torch.tensor(
                self.lokr_factor, dtype=torch.int64, device=self.alpha.device
            )
            destination["lokr_unbalanced"] = torch.tensor(
                1, dtype=torch.int64, device=self.alpha.device
            )
        return destination

    @torch.no_grad()
    def apply_max_norm(self, max_norm, device=None):
        orig_norm = self.get_weight(self.shape).norm()
        norm = torch.clamp(orig_norm, max_norm / 2)
        desired = torch.clamp(norm, max=max_norm)
        ratio = desired.cpu() / norm.cpu()

        scaled = norm != desired
        if scaled:
            modules = 4 - self.use_w1 - self.use_w2 + (not self.use_w2 and self.tucker)
            if self.use_w1:
                self.lokr_w1 *= ratio ** (1 / modules)
            else:
                self.lokr_w1_a *= ratio ** (1 / modules)
                self.lokr_w1_b *= ratio ** (1 / modules)

            if self.use_w2:
                self.lokr_w2 *= ratio ** (1 / modules)
            else:
                if self.tucker:
                    self.lokr_t2 *= ratio ** (1 / modules)
                self.lokr_w2_a *= ratio ** (1 / modules)
                self.lokr_w2_b *= ratio ** (1 / modules)
            return scaled, orig_norm * ratio
        else:
            return 0, orig_norm
        
    @torch.no_grad()
    def get_norm(self, device=None):
        weight = self.get_weight(self.shape)
        unscaled_norm = weight.norm()
        return unscaled_norm

    def bypass_forward_diff(self, h, scale=1):
        is_conv = self.module_type.startswith("conv")
        if self.use_w2:
            ba = self._orthogonalize(self.lokr_w2)
        else:
            a = self._orthogonalize(self.lokr_w2_b)
            b = self._orthogonalize(self.lokr_w2_a)

            if self.tucker:
                t = self.lokr_t2
                a = a.view(*a.shape, *[1] * (len(t.shape) - 2))
                # w2_a is stored as (rank, out_factor), but convolution
                # weights use (out_channels, in_channels, ...).
                b = b.transpose(0, 1).contiguous()
                b = b.view(*b.shape, *[1] * (len(t.shape) - 2))
            elif is_conv:
                # Non-Tucker convolution w2_b is serialized as a flattened
                # (rank, in_factor * kernel_elements) matrix.
                kernel_elements = math.prod(self.shape[2:])
                if a.shape[1] % kernel_elements != 0:
                    raise RuntimeError(
                        "Invalid LoKr convolution factor shape: flattened input "
                        f"elements ({a.shape[1]}) are not divisible by kernel "
                        f"size ({kernel_elements})"
                    )
                a = a.view(
                    a.shape[0], a.shape[1] // kernel_elements, *self.shape[2:]
                )
                b = b.view(*b.shape, *[1] * (len(self.shape) - 2))

        if self.use_w1:
            c = self._orthogonalize(self.lokr_w1)
        else:
            w1_a_ortho = self._orthogonalize(self.lokr_w1_a)
            w1_b_ortho = self._orthogonalize(self.lokr_w1_b)
            c = w1_a_ortho @ w1_b_ortho
        uq = c.size(1)

        if is_conv:
            # (b, uq), vq, ...
            batch_size, _, *rest = h.shape
            h_in_group = h.reshape(batch_size * uq, -1, *rest)
        else:
            # b, ..., uq, vq
            h_in_group = h.reshape(*h.shape[:-1], uq, -1)

        if self.use_w2:
            hb = self._call_op(h_in_group, ba)
        else:
            if is_conv:
                if self.tucker:
                    ha = self._call_op_1x1(h_in_group, a)
                    ht = self._call_op(ha, t)
                    hb = self._call_op_1x1(ht, b)
                else:
                    ha = self._call_op(h_in_group, a)
                    hb = self._call_op_1x1(ha, b)
            else:
                ha = self._call_op(h_in_group, a)
                hb = self._call_op(ha, b)

        if is_conv:
            # (b, uq), vp, ..., f
            # -> b, uq, vp, ..., f
            # -> b, f, vp, ..., uq
            hb = hb.view(batch_size, -1, *hb.shape[1:])
            h_cross_group = hb.transpose(1, -1)
        else:
            # b, ..., uq, vq
            # -> b, ..., vq, uq
            h_cross_group = hb.transpose(-1, -2)

        hc = F.linear(h_cross_group, c)
        if is_conv:
            # b, f, vp, ..., up
            # -> b, up, vp, ... ,f
            # -> b, c, ..., f
            hc = hc.transpose(1, -1)
            h = hc.reshape(batch_size, -1, *hc.shape[3:])
        else:
            # b, ..., vp, up
            # -> b, ..., up, vp
            # -> b, ..., c
            hc = hc.transpose(-1, -2)
            h = hc.reshape(*hc.shape[:-2], -1)

        # get_weight() applies the alpha/rank scale in rebuild mode.  Apply
        # that factor here too, followed only by the runtime multiplier.
        h = h * scale * self.scale * self.scalar
        if self.training and self.rank_dropout:
            drop_size = h.shape[1] if is_conv else h.shape[-1]
            drop = (torch.rand(drop_size, device=h.device) > self.rank_dropout).to(h.dtype)
            if self.rank_dropout_scale:
                drop /= drop.mean().clamp_min(torch.finfo(h.dtype).eps)
            if is_conv:
                drop = drop.view(1, -1, *([1] * (h.dim() - 2)))
            else:
                drop = drop.view(*([1] * (h.dim() - 1)), -1)
            h = h * drop
        return self.drop(h)

    def bypass_forward(self, x, scale=1):
        return self.org_forward(x) + self.bypass_forward_diff(x, scale=scale)

    def _forward_rebuild_core(self, x, org_weight, bias):
        """Rebuild-mode forward pass — the torch.compile target.

        Computes diff weight via Kronecker product decomposition, merges
        with the pre-fetched original weight, and runs the fused
        linear/conv operation.  All inputs are pre-fetched GPU tensors.
        """
        diff_weight = self.get_weight(self.shape).to(self._cached_dtype) * self.scalar
        weight = org_weight
        multiplier = self.multiplier_buf
        if self.wd:
            weight = self.apply_weight_decompose(weight + diff_weight, multiplier)
        else:
            # Always apply multiplier (no-op when multiplier==1 since x*1=x).
            # Avoids data-dependent branch that causes torch.compile graph breaks.
            weight = weight + diff_weight * multiplier
        return self._call_op(x, weight, bias)

    def forward(self, x: torch.Tensor, *args, **kwargs):
        if self.module_dropout and self.training:
            if torch.rand(1) < self.module_dropout:
                return self.org_forward(x)
        if self.bypass_mode:
            return self.bypass_forward(x, self.multiplier)

        x = x.to(self._cached_dtype)
        org_weight = self.get_org_weight_for_compute(x.device).data.to(self._cached_dtype, non_blocking=True)
        org_bias = self.get_org_bias_for_compute(x.device)
        # Pre-resolve bias to real tensor or None (avoids numel check in compiled graph)
        if org_bias is not None:
            bias = org_bias.to(x.dtype, non_blocking=True)
        else:
            bias = None

        return self._forward_rebuild_core(x, org_weight, bias)


if __name__ == "__main__":
    base = nn.Conv2d(128, 128, 3, 1, 1)
    net = LokrModule(
        "",
        base,
        multiplier=1,
        lora_dim=4,
        alpha=1,
        weight_decompose=False,
        use_tucker=False,
        use_scalar=False,
        decompose_both=True,
    )
    net.apply_to()
    sd = net.state_dict()
    for key in sd:
        if key != "alpha":
            sd[key] = torch.randn_like(sd[key])
    net.load_state_dict(sd)

    test_input = torch.randn(1, 128, 16, 16)
    test_output = net(test_input)
    print(test_output.shape)

    net2 = LokrModule(
        "",
        base,
        multiplier=1,
        lora_dim=4,
        alpha=1,
        weight_decompose=False,
        use_tucker=False,
        use_scalar=False,
        bypass_mode=True,
        decompose_both=True,
    )
    net2.apply_to()
    net2.load_state_dict(sd)
    print(net2)

    test_output2 = net2(test_input)
    print(F.mse_loss(test_output, test_output2))
