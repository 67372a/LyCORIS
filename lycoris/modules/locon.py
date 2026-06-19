import math
from functools import cache

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import LycorisBaseModule
from ..functional.general import rebuild_tucker
from ..logging import logger

from typing import Optional, Callable, Dict

@cache
def log_wd():
    return logger.warning(
        "Using weight_decompose=True with LoRA (DoRA) will cause network dropout to be applied to the forward input, "
        "instead of to the layers, as per the DoRA paper."
    )


class LoConModule(LycorisBaseModule):
    name = "locon"
    support_module = {
        "linear",
        "conv1d",
        "conv2d",
        "conv3d",
    }
    weight_list = [
        "lora_up.weight",
        "lora_down.weight",
        "lora_mid.weight",
        "alpha",
        "dora_scale",
        "pissa_A_init",
        "pissa_B_init",
        "pissa_converted",
    ]
    weight_list_det = ["lora_up.weight"]

    # O-LoRA: registry of all LoConModule instances for orthogonality loss aggregation.
    _olora_modules: list = []

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
        pissa_niter: int = 0,
        pissa_convert: bool = True,
        olora: bool = False,
        olora_lambda: float = 0.5,
        olora_task_id: int = 0,
        use_timestep_mask: bool = False,
        scalar_type: str = "scalar",
        **kwargs,
    ):
        """if alpha == 0 or None, alpha is rank (no scaling)."""
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
            ggpo_conv_weight_sample_size
        )
        if self.module_type not in self.support_module:
            raise ValueError(f"{self.module_type} is not supported in LoRA/LoCon algo.")
        self.lora_dim = lora_dim
        self.tucker = False
        self.rs_lora = rs_lora
        self.use_orthogonal_weights = orthogonalize
        if orthogonalize and not orthogonal_init:
            orthogonal_init = True
        self.use_orthogonal_init = orthogonal_init
        if self.use_orthogonal_weights and not use_scalar:
            use_scalar = True

        # O-LoRA configuration
        self.olora = olora
        self.olora_lambda = olora_lambda
        self.olora_task_id = olora_task_id
        if self.olora:
            LoConModule._olora_modules.append(self)

        if self.module_type.startswith("conv"):
            self.isconv = True
            # For general LoCon
            in_dim = org_module.in_channels
            k_size = org_module.kernel_size
            stride = org_module.stride
            padding = org_module.padding
            out_dim = org_module.out_channels
            use_tucker = use_tucker and any(i != 1 for i in k_size)
            self.down_op = self.op
            self.up_op = self.op
            if use_tucker:
                self.lora_down = self.module(in_dim, lora_dim, 1, bias=False)
                self.lora_mid = self.module(
                    lora_dim, lora_dim, k_size, stride, padding, bias=False
                )
                self.tucker = True
            else:
                self.lora_down = self.module(
                    in_dim, lora_dim, k_size, stride, padding, bias=False
                )
            self.lora_up = self.module(lora_dim, out_dim, 1, bias=False)
        elif self.module_type == "linear" or isinstance(org_module, nn.Linear):
            self.isconv = False
            self.down_op = F.linear
            self.up_op = F.linear
            in_dim = org_module.in_features
            out_dim = org_module.out_features
            self.lora_down = nn.Linear(in_dim, lora_dim, bias=False)
            self.lora_up = nn.Linear(lora_dim, out_dim, bias=False)
        else:
            raise NotImplementedError

        # O-LoRA multi-task containers: wrap the freshly created modules.
        # self.lora_down / self.lora_up are kept as direct references to the
        # current (trainable) task's modules for backward compatibility with
        # all existing code paths (init, forward, merge, state_dict, etc.).
        if self.olora:
            self.lora_down_modules = nn.ModuleList([self.lora_down])
            self.lora_up_modules = nn.ModuleList([self.lora_up])
            self.lora_scalar_list = nn.ParameterList()
            self.lora_mid_modules = nn.ModuleList()
            if self.tucker:
                self.lora_mid_modules.append(self.lora_mid)
        else:
            self.lora_down_modules = nn.ModuleList()
            self.lora_up_modules = nn.ModuleList()
            self.lora_scalar_list = nn.ParameterList()
            self.lora_mid_modules = nn.ModuleList()

        self.wd = weight_decompose
        self.wd_on_output = wd_on_output
        if self.wd:
            # Compute norm on the original device to avoid CPU↔GPU transfer.
            # torch.norm is read-only so no clone needed; .float() creates a
            # new tensor for numerical precision.
            org_weight = org_module.weight.data.float()
            ndim = org_weight.dim()
            self.dora_norm_dims = ndim - 1

            # Pre-compute dimension tuples for torch.linalg.vector_norm in
            # apply_weight_decompose — avoids runtime tuple construction and
            # eliminates the transpose→reshape→norm→reshape→transpose chain
            # that materialises a full contiguous copy for non-wd_on_output.
            #   wd_on_output=True : norm along all dims except dim 0 (output)
            #   wd_on_output=False: norm along all dims except dim 1 (input)
            if wd_on_output:
                self._dora_norm_dims = tuple(range(1, ndim))
            else:
                self._dora_norm_dims = (0,) + tuple(range(2, ndim))

            # Pre-compute eps to avoid torch.finfo lookup every forward pass.
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

        if dropout and self.wd:
            log_wd()

        if isinstance(alpha,torch.Tensor):
            alpha = alpha.detach().float().item()  # without casting, bf16 causes error
        alpha = lora_dim if alpha is None or alpha == 0 else alpha

        r_factor = lora_dim
        if self.rs_lora:
            r_factor = math.sqrt(r_factor)

        self.scale = alpha / r_factor

        self.register_buffer("alpha", torch.tensor(alpha * (lora_dim / r_factor)))

        if use_scalar:
            # O-LoRA only supports scalar_type="scalar"
            if self.olora and scalar_type != "scalar":
                logger.warning(
                    f"O-LoRA: scalar_type='{scalar_type}' is not supported. "
                    f"Falling back to scalar_type='scalar'."
                )
                scalar_type = "scalar"
            self.scalar_type = scalar_type
            if scalar_type == "scalar":
                init_val = scalar_init_value if scalar_init_value is not None else 0.1
                self.scalar = nn.Parameter(torch.tensor(init_val))
                if self.olora:
                    self.lora_scalar_list.append(self.scalar)
            elif scalar_type == "row":
                init_val = scalar_init_value if scalar_init_value is not None else 0.1
                self.row_scalar = nn.Parameter(torch.full((self.shape[0],), init_val))
            elif scalar_type == "column":
                init_val = scalar_init_value if scalar_init_value is not None else 0.1
                self.col_scalar = nn.Parameter(torch.full((self.shape[1],), init_val))
            elif scalar_type == "row_column":
                init_val = scalar_init_value if scalar_init_value is not None else math.sqrt(0.1)
                self.row_scalar = nn.Parameter(torch.full((self.shape[0],), init_val))
                self.col_scalar = nn.Parameter(torch.full((self.shape[1],), init_val))
        else:
            self.scalar_type = "none"
            self.register_buffer("scalar", torch.tensor(1.0), persistent=False)

        # Weight initialization
        if self.use_orthogonal_init:
            torch.nn.init.orthogonal_(self.lora_down.weight)
        else:
            torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))

        if self.use_orthogonal_init:
            torch.nn.init.orthogonal_(self.lora_up.weight)
        else:
            if use_scalar:
                torch.nn.init.kaiming_uniform_(self.lora_up.weight, a=math.sqrt(5))
            else:
                torch.nn.init.constant_(self.lora_up.weight, 0)

        if self.tucker:
            if self.use_orthogonal_init:
                torch.nn.init.orthogonal_(self.lora_mid.weight)
            else:
                torch.nn.init.kaiming_uniform_(self.lora_mid.weight, a=math.sqrt(5))

        self.init_ggpo()

        # T-LoRA: timestep-dependent rank masking (anima_lora port).
        # All-ones default → identity multiply; the training loop rebinds
        # _timestep_mask to a shared live-updated buffer via
        # LycorisNetwork.set_timestep_mask.  persistent=False so the mask
        # never appears in checkpoints — trained weights are full-rank.
        self.use_timestep_mask = use_timestep_mask
        if use_timestep_mask:
            self.register_buffer(
                "_timestep_mask",
                torch.ones(1, lora_dim, dtype=torch.float32),
                persistent=False,
            )

        # Cache _scale_in_diff: determines which weight-construction path
        # is taken in _forward_rebuild_core.  All values are init-time
        # constants so this can be computed once.
        self._scale_in_diff = (
            not self.wd and (self.tucker or self.rank_dropout)
        )

        # Cache the static portion of the apply_ggpo check to avoid
        # recomputing 4 invariant conditions on every forward pass.
        self._ggpo_enabled = (
            self.ggpo_sigma is not None
            and self.ggpo_beta is not None
            and (self.module_type == "linear"
                 or (self.module_type.startswith("conv") and self.ggpo_conv))
        )

        # Store PiSSA-specific config
        self.pissa_niter = pissa_niter
        self.pissa_convert = pissa_convert
        self.is_pissa = False  # True when PiSSA init was used (top SVD + residual base)
        self.pissa_A_init: Optional[torch.Tensor] = None
        self.pissa_B_init: Optional[torch.Tensor] = None

        # QPiSSA: iterative quantization-aware SVD
        qpissa_iter = kwargs.get("qpissa_iter", 0)
        quant_fn = kwargs.get("quant_fn", None)

        if quant_fn is not None and qpissa_iter > 0:
            # QPiSSA: iterative quantization-aware initialization
            if self.use_orthogonal_init:
                logger.warning(
                    f"QPiSSA init and orthogonal_init=True are mutually exclusive. "
                    f"QPiSSA will replace orthogonal initialization for {self.lora_name}."
                )
            self._init_qpissa(quant_fn, niter=qpissa_iter)
        else:
            # Standard SVD segment initialization (PiSSA-style)
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
        """Initialize LoCon weights from a segment of the SVD spectrum.

        Supports PiSSA-style initialization when *segment* is ``"top"``:
        the principal components become the trainable adapter (A,B) and the
        residual is stored in the frozen base weight.

        When ``self.pissa_niter > 0``, uses fast randomized SVD instead of
        exact SVD for PiSSA initialization (only applicable for ``"top"``).
        """
        if self.tucker:
            logger.warning(
                f"SVD segment init is not supported for tucker decomposition "
                f"(module {self.lora_name}), skipping."
            )
            return

        org_weight_2d = self._get_weight_2d(self.org_module[0])

        # Use PiSSA fast SVD when requested and segment is "top"
        if segment == "top" and self.pissa_niter > 0:
            result = self._compute_svd_pissa(
                org_weight_2d, self.lora_dim, niter=self.pissa_niter
            )
            logger.info(
                f"PiSSA fast SVD init (niter={self.pissa_niter}): {self.lora_name}"
            )
        else:
            result = self._compute_svd_segment(org_weight_2d, self.lora_dim, segment)
            logger.info(f"SVD segment init ({segment}): {self.lora_name}")

        if result is None:
            logger.warning(
                f"Weight {self.lora_name} has fewer singular values than "
                f"lora_dim={self.lora_dim}, skipping SVD segment init."
            )
            return
        Vr, Sr, Uhr = result

        sqrt_Sr = torch.sqrt(Sr)
        lora_down_2d = torch.diag(sqrt_Sr) @ Uhr   # (r, in)
        lora_up_2d = Vr @ torch.diag(sqrt_Sr)      # (out, r)

        orig_shape = self.org_module[0].weight.shape
        if self.isconv:
            self.lora_down.weight.data.copy_(
                lora_down_2d.reshape(self.lora_dim, *orig_shape[1:])
            )
        else:
            self.lora_down.weight.data.copy_(lora_down_2d)
        self.lora_up.weight.data.copy_(lora_up_2d.reshape(self.lora_up.weight.shape))

        # For PiSSA (top segment), store initial weights for later conversion
        if segment == "top":
            self.is_pissa = True
            self.pissa_A_init = (
                lora_up_2d.detach().to(self.org_module[0].weight.dtype).clone()
            )
            self.pissa_B_init = (
                lora_down_2d.detach().to(self.org_module[0].weight.dtype).clone()
            )

        diff = (lora_up_2d @ lora_down_2d).reshape(orig_shape)
        # For PiSSA, the singular values already encode scale, so we subtract
        # the unscaled diff (scale should be 1.0 when alpha == lora_dim).
        # For other segments, we use the existing scaling.
        if segment == "top" and self.pissa_niter >= 0:
            # PiSSA mode: singular values encode scale, use unscaled diff
            self.org_module[0].weight.data -= diff.to(self.org_module[0].weight.dtype)
        else:
            self.org_module[0].weight.data -= (
                diff.to(self.org_module[0].weight.dtype) * self.scale
            )

    @torch.no_grad()
    def _init_qpissa(self, quant_fn, niter: int = 5):
        """Initialize with QPiSSA: iterative quantization-aware SVD.

        Performs the QPiSSA-T-iters algorithm (Algorithm 1 from the PiSSA paper):
        1. SVD on the weight matrix
        2. Extract principal components into A, B
        3. Compute residual = W - A @ B
        4. Quantize/dequantize the residual
        5. Compute error = W - dequantized_residual
        6. SVD on error → refine A, B
        7. Repeat for *niter* iterations

        This significantly reduces quantization error compared to QLoRA by
        quantizing only the residual (which has a narrower distribution)
        rather than the full weight matrix.

        Args:
            quant_fn: Callable ``(weight) -> (quantized, dequantized)`` that
                      quantizes and dequantizes a weight tensor.
            niter: Number of alternating SVD+quantization iterations (default 5).
        """
        if self.tucker:
            logger.warning(
                f"QPiSSA init is not supported for tucker decomposition "
                f"(module {self.lora_name}), skipping."
            )
            return

        from .pissa_utils import qpissa_iterative

        org_weight_2d = self._get_weight_2d(self.org_module[0])

        # Run QPiSSA-T-iters
        quant_res, dequant_res, pissa_A, pissa_B = qpissa_iterative(
            org_weight_2d,
            self.lora_dim,
            niter=niter,
            quant_fn=quant_fn,
            fast_niter=self.pissa_niter,
        )

        # Set adapter weights from PiSSA decomposition
        orig_shape = self.org_module[0].weight.shape
        lora_down_2d = pissa_B  # (r, in)
        lora_up_2d = pissa_A    # (out, r)

        if self.isconv:
            self.lora_down.weight.data.copy_(
                lora_down_2d.reshape(self.lora_dim, *orig_shape[1:])
            )
        else:
            self.lora_down.weight.data.copy_(lora_down_2d)
        self.lora_up.weight.data.copy_(lora_up_2d.reshape(self.lora_up.weight.shape))

        # Set base weight to the quantized residual
        self.org_module[0].weight.data.copy_(
            dequant_res.reshape(orig_shape).to(self.org_module[0].weight.dtype)
        )

        # Mark as PiSSA and store initial weights for later conversion
        self.is_pissa = True
        self.pissa_A_init = (
            lora_up_2d.detach().to(self.org_module[0].weight.dtype).clone()
        )
        self.pissa_B_init = (
            lora_down_2d.detach().to(self.org_module[0].weight.dtype).clone()
        )

        logger.info(
            f"QPiSSA init: {self.lora_name} (rank={self.lora_dim}, niter={niter})"
        )

    @classmethod
    def make_module_from_state_dict(
        cls, lora_name, orig_module, up, down, mid, alpha, dora_scale
    ):
        module = cls(
            lora_name,
            orig_module,
            1.0,
            down.size(0),
            float(alpha),
            use_tucker=mid is not None,
            weight_decompose=dora_scale is not None,
        )
        module.lora_up.weight.data.copy_(up)
        module.lora_down.weight.data.copy_(down)
        if mid is not None:
            module.lora_mid.weight.data.copy_(mid)
        if dora_scale is not None:
            module.dora_scale.copy_(dora_scale)
        return module

    @staticmethod
    def _should_skip_missing_key(key: str) -> bool:
        """Check if a missing key is expected to be absent (e.g., optional features)."""
        if "scalar" in key:
            return True
        # O(n) substring checks — fast for short key strings
        for tag in ("pissa_A_init", "pissa_B_init", "pissa_converted",
                     "olora_task_id",
                     "lora_down_task", "lora_up_task", "lora_mid_task"):
            if tag in key:
                return True
        return False

    def load_weight_hook(self, module: nn.Module, incompatible_keys):
        missing_keys = incompatible_keys.missing_keys
        # Filter expected missing keys in O(n) instead of O(n²)
        missing_keys[:] = [k for k in missing_keys if not self._should_skip_missing_key(k)]
        # Reset scalars to 1.0 (baked into weights during save)
        for attr in ("scalar", "row_scalar", "col_scalar"):
            param = getattr(self, attr, None)
            if param is not None:
                if isinstance(param, nn.Parameter):
                    param.data.copy_(torch.ones_like(param))
                else:
                    param.copy_(torch.ones_like(param))
        # Initialize PiSSA buffers if not loaded from state dict
        if not hasattr(self, "pissa_A_init") or self.pissa_A_init is None:
            self.pissa_A_init = None
        if not hasattr(self, "pissa_B_init") or self.pissa_B_init is None:
            self.pissa_B_init = None
        # Initialize O-LoRA task ID if not loaded from state dict
        if not hasattr(self, "olora_task_id"):
            self.olora_task_id = 0

    def _apply_rank_dropout(self, weight, device):
        """Apply rank dropout to a weight tensor.

        Extracted from inline code to eliminate duplication across
        _make_weight_single, _make_weight_multitask,
        _compute_diff_weight_single, _compute_diff_weight_multitask.

        Args:
            weight: Tensor whose first dimension is the rank/dropout dimension.
            device: Target device for the dropout mask.

        Returns:
            Weight with dropout applied, or the original weight unchanged.
        """
        if not (self.training and self.rank_dropout):
            return weight
        drop = (torch.rand(weight.size(0), device=device) > self.rank_dropout).to(
            weight.dtype
        )
        drop = drop.view(self._rank_drop_shape)
        if self.rank_dropout_scale:
            drop /= drop.mean()
        return weight * drop

    def _maybe_orthogonalize(self, weight):
        """Orthogonalize only if use_orthogonal_weights is active.

        Avoids the function-call + branch overhead of _orthogonalize() in the
        common case where orthogonalization is disabled (the fast path).
        """
        if self.use_orthogonal_weights and self.training:
            return self._orthogonalize(weight)
        return weight

    def _apply_scalar(self, weight, device=None, dtype=None):
        """Apply the appropriate scalar (single, row, col, or row+col) to weight."""
        if self.scalar_type == "scalar":
            return weight * self.scalar.to(device=device, dtype=dtype)
        elif self.scalar_type == "row":
            return weight * self.row_scalar.to(device=device, dtype=dtype).view(-1, *([1] * (weight.dim() - 1)))
        elif self.scalar_type == "column":
            return weight * self.col_scalar.to(device=device, dtype=dtype).view(1, -1, *([1] * (weight.dim() - 2)))
        elif self.scalar_type == "row_column":
            r = self.row_scalar.to(device=device, dtype=dtype).view(-1, *([1] * (weight.dim() - 1)))
            c = self.col_scalar.to(device=device, dtype=dtype).view(1, -1, *([1] * (weight.dim() - 2)))
            return weight * r * c
        return weight

    def _apply_row_scalar(self, weight):
        """Bake row scalar into weight for saving. For scalar type, bakes the single scalar."""
        if self.scalar_type == "scalar":
            return weight * self.scalar.to(device=weight.device, non_blocking=True)
        elif self.scalar_type in ("row", "row_column"):
            return weight * self.row_scalar.to(device=weight.device, non_blocking=True).view(-1, *([1] * (weight.dim() - 1)))
        return weight

    def _apply_col_scalar(self, weight):
        """Bake col scalar into weight for saving."""
        if self.scalar_type in ("column", "row_column"):
            return weight * self.col_scalar.to(device=weight.device, non_blocking=True).view(1, -1, *([1] * (weight.dim() - 2)))
        return weight

    def make_weight(self, device=None):
        if self.olora:
            return self._make_weight_multitask(device)
        return self._make_weight_single(device)

    def _make_weight_single(self, device=None):
        """Original single-task weight computation (used when olora=False)."""
        wa = self._maybe_orthogonalize(self.lora_up.weight.to(device))
        wb = self._maybe_orthogonalize(self.lora_down.weight.to(device))
        # T-LoRA: zero out masked rank dimensions in the down weight so the
        # up×down product has those rank components zeroed.  Equivalent to
        # masking the bottleneck activation but works in rebuild mode.
        if self.use_timestep_mask and self.training:
            wb = wb * self._timestep_mask.to(wb).view(-1, *([1] * (wb.dim() - 1)))
        if self.tucker:
            t = self._maybe_orthogonalize(self.lora_mid.weight.to(device))
            wa = wa.view(wa.size(0), -1).transpose(0, 1)
            wb = wb.view(wb.size(0), -1)
            weight = rebuild_tucker(t, wa, wb)
        else:
            weight = wa.view(wa.size(0), -1) @ wb.view(wb.size(0), -1)

        weight = weight.view(self.shape)
        weight = self._apply_rank_dropout(weight, device)

        return self._apply_scalar(weight, device)

    def _make_weight_multitask(self, device=None):
        """Multi-task O-LoRA weight: sum of w_a @ w_b across all tasks."""
        total = None
        num_tasks = len(self.lora_down_modules)
        for idx in range(num_tasks):
            wa = self._maybe_orthogonalize(self.lora_up_modules[idx].weight.to(device))
            wb = self._maybe_orthogonalize(self.lora_down_modules[idx].weight.to(device))
            if self.tucker and len(self.lora_mid_modules) > idx:
                t = self._maybe_orthogonalize(self.lora_mid_modules[idx].weight.to(device))
                wa = wa.view(wa.size(0), -1).transpose(0, 1)
                wb = wb.view(wb.size(0), -1)
                task_weight = rebuild_tucker(t, wa, wb)
            else:
                task_weight = wa.view(wa.size(0), -1) @ wb.view(wb.size(0), -1)
            task_weight = task_weight.view(self.shape)
            task_weight = self._apply_rank_dropout(task_weight, device)
            scalar = self.lora_scalar_list[idx] if idx < len(self.lora_scalar_list) else self.scalar
            task_weight = task_weight * scalar.to(device)
            if total is None:
                total = task_weight
            else:
                total = total + task_weight
        return total

    def _compute_diff_weight_single(self, device, dtype):
        """Single-task diff_weight for non-bypass forward (tucker or rank_dropout case)."""
        wa = self._maybe_orthogonalize(self.lora_up.weight).to(device=device, dtype=dtype)
        wb = self._maybe_orthogonalize(self.lora_down.weight).to(device=device, dtype=dtype)
        # T-LoRA: mask rank dimensions in the down weight (rebuild-mode path).
        if self.use_timestep_mask and self.training:
            wb = wb * self._timestep_mask.to(device=device, dtype=dtype).view(-1, *([1] * (wb.dim() - 1)))

        if self.tucker:
            t = self._maybe_orthogonalize(self.lora_mid.weight).to(device=device, dtype=dtype)
            wa = wa.view(wa.size(0), -1).transpose(0, 1)
            wb = wb.view(wb.size(0), -1)
            diff_weight = rebuild_tucker(t, wa, wb)
        else:
            diff_weight = wa.view(wa.size(0), -1) @ wb.view(wb.size(0), -1)

        diff_weight = diff_weight.view(self.shape)
        diff_weight = self._apply_rank_dropout(diff_weight, device)

        # Cast scalar to target device+dtype once to avoid implicit fp32 upcast then
        # downcast back to dtype on every forward pass.
        diff_weight = self._apply_scalar(diff_weight, device=device, dtype=dtype) * self.scale
        return diff_weight

    def _compute_diff_weight_multitask(self, device, dtype):
        """Multi-task O-LoRA diff_weight summing over all task LoRA pairs."""
        total = None
        num_tasks = len(self.lora_down_modules)
        for idx in range(num_tasks):
            wa = self._maybe_orthogonalize(self.lora_up_modules[idx].weight).to(device=device, dtype=dtype)
            wb = self._maybe_orthogonalize(self.lora_down_modules[idx].weight).to(device=device, dtype=dtype)

            if self.tucker and len(self.lora_mid_modules) > idx:
                t = self._maybe_orthogonalize(self.lora_mid_modules[idx].weight).to(device=device, dtype=dtype)
                wa = wa.view(wa.size(0), -1).transpose(0, 1)
                wb = wb.view(wb.size(0), -1)
                task_weight = rebuild_tucker(t, wa, wb)
            else:
                task_weight = wa.view(wa.size(0), -1) @ wb.view(wb.size(0), -1)

            task_weight = task_weight.view(self.shape)
            task_weight = self._apply_rank_dropout(task_weight, device)

            scalar = self.lora_scalar_list[idx] if idx < len(self.lora_scalar_list) else self.scalar
            # Pre-cast scalar to target device+dtype to avoid implicit fp32 upcast/downcast.
            task_weight = (task_weight * scalar.to(device=device, dtype=dtype)) * self.scale
            if total is None:
                total = task_weight
            else:
                total = total + task_weight
        return total

    def add_task(self, task_id: int):
        """Create a new LoRA pair for the incoming task and freeze previous ones.

        This follows the O-LoRA algorithm: previous LoRA parameters are fixed,
        and only the current task's LoRA is trained. Orthogonality between the
        new subspace and previous subspaces is enforced via the loss term
        (see :meth:`get_olora_orthogonality_loss`).

        Args:
            task_id: Zero-based task index for the new task.
        """
        if not self.olora:
            raise RuntimeError("add_task() called on a non-O-LoRA LoConModule.")

        # 1. Freeze all existing LoRA modules
        for module in self.lora_down_modules:
            for p in module.parameters():
                p.requires_grad = False
        for module in self.lora_up_modules:
            for p in module.parameters():
                p.requires_grad = False
        for module in self.lora_mid_modules:
            for p in module.parameters():
                p.requires_grad = False
        for scalar in self.lora_scalar_list:
            scalar.requires_grad = False

        # 2. Create new trainable LoRA pair
        if self.isconv:
            in_dim = self.lora_down_modules[0].weight.shape[1]  # in_channels
            out_dim = self.lora_up_modules[0].weight.shape[0]    # out_channels
            k_size = self.lora_down.kernel_size
            stride = self.lora_down.stride
            padding = self.lora_down.padding
            new_down = self.module(in_dim, self.lora_dim, k_size, stride, padding, bias=False)
            new_up = self.module(self.lora_dim, out_dim, 1, bias=False)
        else:
            in_dim = self.lora_down_modules[0].weight.shape[1]
            out_dim = self.lora_up_modules[0].weight.shape[0]
            new_down = nn.Linear(in_dim, self.lora_dim, bias=False)
            new_up = nn.Linear(self.lora_dim, out_dim, bias=False)

        # 3. Initialize new weights
        if self.use_orthogonal_init:
            nn.init.orthogonal_(new_down.weight)
            nn.init.orthogonal_(new_up.weight)
        else:
            nn.init.kaiming_uniform_(new_down.weight, a=math.sqrt(5))
            nn.init.zeros_(new_up.weight)

        # 4. Register new modules
        self.lora_down_modules.append(new_down)
        self.lora_up_modules.append(new_up)

        # 5. Add new scalar if use_scalar is active
        if isinstance(self.scalar, nn.Parameter):
            new_scalar = nn.Parameter(torch.tensor(0.1))
            self.lora_scalar_list.append(new_scalar)

        # 6. Handle tucker mid module
        if self.tucker:
            new_mid = self.module(
                self.lora_dim, self.lora_dim,
                self.lora_mid.kernel_size,
                self.lora_mid.stride,
                self.lora_mid.padding,
                bias=False,
            )
            if self.use_orthogonal_init:
                nn.init.orthogonal_(new_mid.weight)
            else:
                nn.init.kaiming_uniform_(new_mid.weight, a=math.sqrt(5))
            self.lora_mid_modules.append(new_mid)

        # 7. Update backward-compatible references to point to the new trainable task
        self.lora_down = new_down
        self.lora_up = new_up
        if self.tucker:
            self.lora_mid = new_mid

        # 8. Update task ID
        self.olora_task_id = task_id

    def get_diff_weight(self, multiplier=1, shape=None, device=None):
        scale = self.scale * multiplier
        diff = self.make_weight(device=device) * scale
        if shape is not None:
            diff = diff.view(shape)
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
        # keepdim=True.  This replaces the old
        #   reshape→norm(dim=1)→reshape + transpose chain
        # with a single fused call that:
        #  (a) eliminates intermediate tensor allocations from reshape/transpose
        #  (b) avoids the O(n) contiguous copy forced by transpose→reshape
        #      in the wd_on_output=False path (transpose makes the tensor
        #      non-contiguous, so reshape must copy the entire weight)
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
        if self.wd:
            destination["dora_scale"] = self.dora_scale
        destination["alpha"] = self.alpha

        # O-LoRA multi-task serialization
        if self.olora:
            for task_idx in range(len(self.lora_down_modules)):
                scalar = (
                    self.lora_scalar_list[task_idx]
                    if task_idx < len(self.lora_scalar_list)
                    else self.scalar
                )
                destination[f"lora_up_task{task_idx}.weight"] = (
                    self.lora_up_modules[task_idx].weight
                    * scalar.to(device=self.lora_up_modules[task_idx].weight.device, non_blocking=True)
                )
                destination[f"lora_down_task{task_idx}.weight"] = (
                    self.lora_down_modules[task_idx].weight
                )
                if self.tucker and task_idx < len(self.lora_mid_modules):
                    destination[f"lora_mid_task{task_idx}.weight"] = (
                        self.lora_mid_modules[task_idx].weight
                    )
            destination["olora_task_id"] = torch.tensor(self.olora_task_id)
            return destination

        # Non-O-LoRA path (existing logic)
        if self.is_pissa and self.pissa_convert and self.pissa_A_init is not None:
            # PiSSA→LoRA conversion on save:
            # ΔW = A'B' - A₀B₀ = [A' | A₀] · [B' | -B₀]^T
            lora_up_w = self._apply_row_scalar(self.lora_up.weight)
            # Concatenate: trained A (up) with initial A₀ (up init)
            converted_up = torch.cat(
                [lora_up_w, self.pissa_A_init.to(lora_up_w.device)], dim=1
            )
            # Concatenate: trained B (down) with negated initial B₀ (down init)
            lora_down_w = self._apply_col_scalar(self.lora_down.weight)
            converted_down = torch.cat(
                [lora_down_w, -self.pissa_B_init.to(lora_down_w.device)], dim=0
            )
            destination["lora_up.weight"] = converted_up
            destination["lora_down.weight"] = converted_down
            destination["pissa_converted"] = torch.tensor(1.0)
            logger.info(
                f"PiSSA→LoRA conversion on save: {self.lora_name} "
                f"(rank {self.lora_dim} → {2 * self.lora_dim})"
            )
        else:
            destination["lora_up.weight"] = self._apply_row_scalar(self.lora_up.weight)
            destination["lora_down.weight"] = self._apply_col_scalar(self.lora_down.weight)
            # Preserve PiSSA init weights in state dict for round-trip loading
            if self.is_pissa and self.pissa_A_init is not None:
                destination["pissa_A_init"] = self.pissa_A_init
                destination["pissa_B_init"] = self.pissa_B_init

        if self.tucker:
            destination["lora_mid.weight"] = self.lora_mid.weight
        return destination

    @torch.no_grad()
    def convert_pissa_to_lora(self):
        """Convert trained PiSSA adapter to portable LoRA format.

        Uses the identity:
            ΔW = A'B' - A₀B₀ = [A' | A₀] · [B' | -B₀]^T

        After conversion, the module behaves as a standard LoRA adapter
        that can be loaded onto the original (non-decomposed) pretrained model
        without requiring SVD.

        Returns:
            ``True`` if conversion was performed, ``False`` if the module
            was not a PiSSA adapter.
        """
        if not self.is_pissa or self.pissa_A_init is None or self.pissa_B_init is None:
            logger.warning(
                f"convert_pissa_to_lora: {self.lora_name} is not a PiSSA adapter, skipping."
            )
            return False

        # Compute portable LoRA weights.
        # Bake scalar into lora_up before concatenation so the converted
        # adapter produces  scalar*A'B' - A₀B₀  (matching custom_state_dict).
        lora_up_curr = self._apply_row_scalar(self.lora_up.weight.data.clone())
        lora_down_curr = self._apply_col_scalar(self.lora_down.weight.data.clone())
        pissa_up_init = self.pissa_A_init.to(lora_up_curr.device)
        pissa_down_init = self.pissa_B_init.to(lora_down_curr.device)

        # ΔA = [scalar*A' | A₀]  → lora_up shape becomes (out, 2*r)
        delta_up = torch.cat([lora_up_curr, pissa_up_init], dim=1)
        # ΔB = [B' | -B₀]       → lora_down shape becomes (2*r, in)
        delta_down = torch.cat([lora_down_curr, -pissa_down_init], dim=0)

        # Replace adapter weights
        self.lora_up.weight.data = delta_up
        self.lora_down.weight.data = delta_down
        self.lora_dim = 2 * self.lora_dim

        # Reset scalars to 1.0 since they have been absorbed into weights
        for attr in ("scalar", "row_scalar", "col_scalar"):
            param = getattr(self, attr, None)
            if param is not None:
                if isinstance(param, nn.Parameter):
                    param.data.fill_(1.0)
                else:
                    param.fill_(1.0)

        # Restore original weight: W = W^res + A₀B₀
        # The base weight currently holds W^res, so we add back A₀B₀
        orig_shape = self.org_module[0].weight.shape
        a0b0 = (pissa_up_init @ pissa_down_init).reshape(orig_shape)
        self.org_module[0].weight.data.add_(a0b0.to(self.org_module[0].weight.dtype))

        # Clear PiSSA state
        self.is_pissa = False
        self.pissa_A_init = None
        self.pissa_B_init = None

        logger.info(
            f"PiSSA→LoRA converted: {self.lora_name} "
            f"(new rank={self.lora_dim})"
        )
        return True

    @torch.no_grad()
    def apply_max_norm(self, max_norm, device=None):
        orig_norm = self.make_weight(device).norm() * self.scale
        norm = torch.clamp(orig_norm, max_norm / 2)
        desired = torch.clamp(norm, max=max_norm)
        ratio = desired.cpu() / norm.cpu()

        scaled = norm != desired
        if scaled:
            if self.scalar_type == "scalar":
                self.scalar *= ratio
            elif self.scalar_type == "row":
                self.row_scalar *= ratio
            elif self.scalar_type == "column":
                self.col_scalar *= ratio
            elif self.scalar_type == "row_column":
                ratio_sqrt = ratio.sqrt()
                self.row_scalar *= ratio_sqrt
                self.col_scalar *= ratio_sqrt
            return scaled, orig_norm * ratio
        else:
            return 0, orig_norm
        
    @torch.no_grad()
    def get_norm(self, device=None):
        # Norm before scale determined by alpha / r_factor
        unscaled_norm = self.make_weight(device).norm()
        return unscaled_norm

    # ------------------------------------------------------------------
    # O-LoRA: Orthogonality Loss
    # ------------------------------------------------------------------

    def get_olora_orthogonality_loss(self) -> torch.Tensor:
        """Compute L1 orthogonality loss between the current (trainable) and all
        frozen A matrices.

        Matches the reference O-LoRA implementation (uie_trainer_lora.py:96):
            L_orth = Σ_{i=1}^{t-1} Σ_{j,k} |(A_i @ A_t^T)[j,k]|

        Returns:
            Scalar loss tensor (0.0 when O-LoRA is disabled or only one task).
        """
        if not self.olora or len(self.lora_down_modules) <= 1:
            return torch.tensor(0.0, device=self.device)

        current_A = self._get_down_weight_2d(self.lora_down_modules[-1])
        orth_loss = torch.tensor(0.0, device=current_A.device)

        for i in range(len(self.lora_down_modules) - 1):
            old_A = self._get_down_weight_2d(self.lora_down_modules[i])
            cross = old_A @ current_A.T
            orth_loss = orth_loss + torch.abs(cross).sum()

        return orth_loss

    def _get_down_weight_2d(self, down_module: nn.Module) -> torch.Tensor:
        """Flatten down-projection weight to (r, in_dim) for orthogonality computation.

        Handles both linear (2D) and convolutional (>2D) weight shapes.
        """
        w = down_module.weight  # (r, in_dim) or (r, in_channels, k1, k2, ...)
        return w.view(w.size(0), -1)

    @staticmethod
    def get_total_olora_loss() -> torch.Tensor:
        """Aggregate orthogonality loss across all registered O-LoRA modules.

        Call this from the training loop:
            total_loss = task_loss + olora_lambda * LoConModule.get_total_olora_loss()
        """
        if not LoConModule._olora_modules:
            return torch.tensor(0.0)
        # Create initial tensor on the first module's device to avoid
        # repeated CPU→GPU implicit transfers in the accumulation loop.
        first_device = LoConModule._olora_modules[0].device
        total = torch.tensor(0.0, device=first_device)
        for mod in LoConModule._olora_modules:
            total = total + mod.get_olora_orthogonality_loss()
        return total

    @staticmethod
    def reset_olora_registry():
        """Clear the O-LoRA module registry (e.g., before starting a fresh run)."""
        LoConModule._olora_modules.clear()

    # ------------------------------------------------------------------

    @torch.no_grad()
    def merge_old_tasks_to_base(self):
        """Merge all frozen (non-current) task LoRA weights into the base weight.

        Implements Equation 9 from the O-LoRA paper:
            W_init := W_init + Σ_{i=1}^{t-1} A_i B_i

        After merging, the frozen task modules are removed from the module lists,
        freeing GPU memory. The current (trainable) task remains.
        """
        if not self.olora or len(self.lora_down_modules) <= 1:
            return

        # Merge all tasks except the last (current, trainable) one
        for task_idx in range(len(self.lora_down_modules) - 1):
            wa = self.lora_up_modules[task_idx].weight.data
            wb = self.lora_down_modules[task_idx].weight.data
            delta = (wa.view(wa.size(0), -1) @ wb.view(wb.size(0), -1)).view(self.shape)
            delta = delta * self.scale
            scalar = (
                self.lora_scalar_list[task_idx]
                if task_idx < len(self.lora_scalar_list)
                else self.scalar
            )
            self.org_module[0].weight.data += (
                (delta * scalar).to(self.org_module[0].weight.dtype)
            )

        # Remove merged modules (keep only the last one)
        keep_idx = len(self.lora_down_modules) - 1
        self.lora_down = self.lora_down_modules[keep_idx]
        self.lora_up = self.lora_up_modules[keep_idx]
        # nn.ModuleList supports __delitem__, but nn.ParameterList does not.
        # Rebuild containers with only the kept (current) module.
        keep_down = [self.lora_down_modules[keep_idx]]
        keep_up = [self.lora_up_modules[keep_idx]]
        self.lora_down_modules = nn.ModuleList(keep_down)
        self.lora_up_modules = nn.ModuleList(keep_up)
        if self.tucker and len(self.lora_mid_modules) > keep_idx:
            self.lora_mid = self.lora_mid_modules[keep_idx]
            self.lora_mid_modules = nn.ModuleList([self.lora_mid_modules[keep_idx]])
        if len(self.lora_scalar_list) > keep_idx:
            self.scalar = self.lora_scalar_list[keep_idx]
            self.lora_scalar_list = nn.ParameterList([self.lora_scalar_list[keep_idx]])

        # Reset task ID to 0 (only one task left, which is now "task 0")
        self.olora_task_id = 0

    # ------------------------------------------------------------------

    def bypass_forward_diff(self, x, scale=1):
        if self.olora:
            return self._bypass_forward_diff_multitask(x, scale)
        return self._bypass_forward_diff_single(x, scale)

    def _bypass_forward_diff_single(self, x, scale=1):
        """Single-task bypass forward diff — pre-fetches weights, delegates to compiled core."""
        # Pre-fetch and orthogonalize weights outside the compiled region
        # to avoid graph breaks from conditional device transfers.
        wb = self._maybe_orthogonalize(self.lora_down.weight).to(x.device, dtype=x.dtype)
        wa = self._maybe_orthogonalize(self.lora_up.weight).to(x.device, dtype=x.dtype)
        wc = None
        if self.tucker:
            wc = self._maybe_orthogonalize(self.lora_mid.weight).to(x.device, dtype=x.dtype)

        if self.scalar_type == "scalar":
            # Pre-combined scalar to avoid triple multiply inside compiled region
            combined_scale = self.scalar.to(device=x.device, dtype=x.dtype) * self.scale * scale
            return self._forward_bypass_core(x, wb, wa, wc, combined_scale)
        else:
            # Apply col_scalar to input, row_scalar to output.
            # Linear features live on the last dim; Conv channels live on dim 1.
            if self.scalar_type in ("column", "row_column"):
                cs = self.col_scalar.to(device=x.device, dtype=x.dtype)
                if self.module_type == "linear":
                    x = x * cs.view(*([1] * (x.dim() - 1)), -1)
                else:
                    x = x * cs.view(1, -1, *([1] * (x.dim() - 2)))
            out = self._forward_bypass_core(x, wb, wa, wc, self.scale * scale)
            if self.scalar_type in ("row", "row_column"):
                rs = self.row_scalar.to(device=out.device, dtype=out.dtype)
                if self.module_type == "linear":
                    out = out * rs.view(*([1] * (out.dim() - 1)), -1)
                else:
                    out = out * rs.view(1, -1, *([1] * (out.dim() - 2)))
            return out

    def _bypass_forward_diff_multitask(self, x, scale=1):
        """Multi-task O-LoRA bypass forward diff: sum over all task LoRA pairs."""
        total_up = None
        num_tasks = len(self.lora_down_modules)
        for idx in range(num_tasks):
            down_module = self.lora_down_modules[idx]
            up_module = self.lora_up_modules[idx]
            wb = self._maybe_orthogonalize(down_module.weight).to(x.device, dtype=x.dtype)
            wa = self._maybe_orthogonalize(up_module.weight).to(x.device, dtype=x.dtype)

            if self.isconv:
                mid = self.down_op(
                    x, wb, bias=None,
                    stride=down_module.stride,
                    padding=down_module.padding,
                    dilation=down_module.dilation,
                    groups=down_module.groups,
                )
            else:
                mid = self.down_op(x, wb)

            if self.tucker and len(self.lora_mid_modules) > idx:
                wc = self._maybe_orthogonalize(self.lora_mid_modules[idx].weight)
                mid = self.op(
                    mid, wc, bias=None,
                    stride=self.lora_mid_modules[idx].stride,
                    padding=self.lora_mid_modules[idx].padding,
                    dilation=self.lora_mid_modules[idx].dilation,
                    groups=self.lora_mid_modules[idx].groups,
                )

            if self.rank_dropout and self.training:
                drop = (
                    torch.rand(self.lora_dim, device=mid.device) > self.rank_dropout
                ).to(mid.dtype)
                if self.rank_dropout_scale:
                    drop /= drop.mean()
                mid = mid * drop.view(self._bypass_rank_drop_shape)

            if self.isconv:
                up = self.up_op(
                    mid, wa, bias=None,
                    stride=up_module.stride,
                    padding=up_module.padding,
                    dilation=up_module.dilation,
                    groups=up_module.groups,
                )
            else:
                up = self.up_op(mid, wa)

            scalar = self.lora_scalar_list[idx] if idx < len(self.lora_scalar_list) else self.scalar
            task_up = up * scalar * self.scale
            if total_up is None:
                total_up = task_up
            else:
                total_up = total_up + task_up

        return self.drop(total_up * scale)

    def bypass_forward(self, x, scale=1):
        return self.org_forward(x) + self.bypass_forward_diff(x, scale=scale)

    def _forward_rebuild_core(self, x, org_weight, bias):
        """Rebuild-mode forward pass — the torch.compile target.

        Computes lora weight delta, merges with the pre-fetched original weight,
        and runs the fused linear/conv operation. All inputs are pre-fetched GPU
        tensors so no conditional device transfers occur inside this method,
        keeping the compiled graph free of graph breaks.

        Args:
            x: Input tensor.
            org_weight: Original module weight, already on compute device and dtype.
            bias: Pre-resolved bias tensor or None (numel check done in forward()).
        """
        dtype = self._cached_dtype
        device = x.device
        multiplier = self.multiplier_buf

        # Weight difference computation
        # Path 1: _compute_diff_weight_single/multitask already includes
        # scalar * scale in the returned diff_weight.
        # Path 2: make_weight includes scalar but NOT scale; scale is
        # applied below.
        if self._scale_in_diff:
            if self.olora:
                diff_weight = self._compute_diff_weight_multitask(device, dtype)
            else:
                diff_weight = self._compute_diff_weight_single(device, dtype)
        else:
            diff_weight = self.make_weight(device).to(dtype)

        # Merge weights
        weight = org_weight
        if self.wd:
            # DoRA path: _scale_in_diff is always False here, so apply scale
            weight = self.apply_weight_decompose(
                weight + diff_weight * self.scale, multiplier)
            # Input dropout for DoRA
            x = self.drop(x)
        elif self._scale_in_diff:
            # diff_weight already includes scale; just apply multiplier
            weight = weight + diff_weight * multiplier
        else:
            # Fuse scale * multiplier into a single multiplication
            weight = weight + diff_weight * (self.scale * multiplier)

        return self._call_op(x, weight, bias)

    def _forward_bypass_core(self, x, down_weight, up_weight, mid_weight, scale):
        """Compiled bypass diff forward — pure tensor math, no device transfers.

        Uses :meth:`_call_op` (org conv params) for operations whose kernel
        matches the original module, and :meth:`_call_op_1x1` (default
        stride=1, padding=0) for 1×1 convolutions (lora_up, tucker lora_down).

        Args:
            x: Input tensor.
            down_weight: Pre-fetched lora_down weight (orthogonalized if needed).
            up_weight: Pre-fetched lora_up weight (orthogonalized if needed).
            mid_weight: Pre-fetched lora_mid weight or None.
            scale: Scalar multiplier (scalar * self.scale * scale).
        """
        # Down projection:
        #   Tucker mode: lora_down is 1×1 conv → use _call_op_1x1
        #   Non-tucker:  lora_down has org stride/padding → use _call_op
        if self.tucker:
            mid = self._call_op_1x1(x, down_weight)
        else:
            mid = self._call_op(x, down_weight)

        # Optional Tucker mid (has org kernel/stride/padding)
        if self.tucker:
            mid = self._call_op(mid, mid_weight)

        # T-LoRA: mask rank dimensions in the bottleneck activation.
        # Applied after down (+ optional tucker mid), before rank dropout.
        if self.use_timestep_mask and self.training:
            mid = mid * self._timestep_mask.to(mid)

        # Rank dropout — uses pre-computed view shape to avoid
        # len(x.shape) graph breaks in compiled region.
        if self.rank_dropout and self.training:
            drop = (
                torch.rand(self.lora_dim, device=mid.device) > self.rank_dropout
            ).to(mid.dtype)
            if self.rank_dropout_scale:
                drop /= drop.mean()
            mid = mid * drop.view(self._bypass_rank_drop_shape)

        # Up projection — always 1×1 conv (or linear)
        up = self._call_op_1x1(mid, up_weight)

        return self.drop(up * scale)

    def forward(self, x):
        if self.module_dropout and self.training:
            if torch.rand(1) < self.module_dropout:
                return self.org_forward(x)
        
        # Check if perturbation is needed — static conditions cached in _ggpo_enabled
        apply_ggpo = (self._ggpo_enabled
                      and self.training
                      and self.combined_weight_norms is not None
                      and self.grad_norms is not None)
        
        # Handle bypass mode first - simpler path
        if self.bypass_mode:
            result = self.bypass_forward(x, scale=self.multiplier)
            
            if apply_ggpo:
                with torch.no_grad():
                    perturbation_output = self.ggpo_pertubation(x)
                
                if perturbation_output is not None:
                    result = result + perturbation_output
                    
            return result
        
        # Pre-fetch original weights outside the compiled region to avoid
        # graph breaks from conditional device transfers (ramtorch, async).
        dtype = self._cached_dtype
        x = x.to(dtype)
        org_weight = self.get_org_weight_for_compute(x.device).to(dtype, non_blocking=True)
        org_bias = self.get_org_bias_for_compute(x.device)
        # Pre-resolve bias to real tensor or None (avoids numel check in compiled graph)
        if org_bias is not None:
            bias = org_bias.to(dtype, non_blocking=True)
        else:
            bias = None

        # Compiled rebuild path
        result = self._forward_rebuild_core(x, org_weight, bias)
        
        # Apply GGPO perturbation if needed (outside compiled region)
        if apply_ggpo:
            with torch.no_grad():
                perturbation_output = self.ggpo_pertubation(x)
                
            if perturbation_output is not None:
                result = result + perturbation_output
        
        return result

    def ggpo_pertubation(self, x):
        # Optimized perturbation generation based on module type
        if self.module_type == "linear":
            # More efficient scale calculation
            perturbation_scale = (self.ggpo_sigma * torch.sqrt(self.combined_weight_norms**2)) + (self.ggpo_beta * (self.grad_norms**2))
            perturbation_scale_factor = (perturbation_scale * self.perturbation_norm_factor).to(self.device)
            
            # For linear layers, use efficient matrix multiplication
            perturbation = torch.randn(self.org_module_shape, dtype=self.dtype, device=self.device)
            perturbation = perturbation * perturbation_scale_factor.view(-1, 1)
            return x @ perturbation.T
        elif self.module_type.startswith("conv") and self.ggpo_conv:
            # More efficient scale calculation
            perturbation_scale = (self.ggpo_sigma * torch.sqrt(self.combined_weight_norms**2)) + (self.ggpo_beta * (self.grad_norms**2))
            perturbation_scale_factor = (perturbation_scale * self.perturbation_norm_factor).to(self.device)

            # For convolution layers, generate efficient perturbation
            perturbation = torch.randn(self.org_module_shape, dtype=self.dtype, device=self.device)
            
            # Apply scaling with efficient broadcasting
            view_shape = [perturbation.shape[0]] + [1] * (len(perturbation.shape) - 1)
            perturbation = perturbation * perturbation_scale_factor.view(*view_shape)
            
            # Use the appropriate convolution operation
            return self._call_op(x, perturbation, None)
        else:
            return None


class RaLoRAModule(LoConModule):
    """RaLoRA: Rank-Aligned LoRA with Gradient Intrinsic Dimensionality.

    Extends LoConModule with block-diagonal decomposition where the number
    of blocks (n_split) is adaptively determined by the entropy-based
    gradient intrinsic dimensionality (GID) estimator.

    The adapter uses block-diagonal structure:
        ΔW = diag(B₁A₁, B₂A₂, ..., B_nA_n)
    where A_i ∈ R^(r × d_in/n) and B_i ∈ R^(d_out/n × r).

    Equivalent rank = n_split × r, with same parameter count as vanilla LoRA.

    Two modes:
      - RaLoRA (ralora_pro=False): Same rank r for all layers, n_split varies
        per layer based on GID.
      - RaLoRA-Pro (ralora_pro=True): Both rank r_l and n_split vary per layer
        guided by loss sensitivity + GID (dual alignment).

    Reference: "Gradient Intrinsic Dimensionality Alignment" (ICLR 2026)
    """

    name = "ralora"
    support_module = {"linear", "conv1d", "conv2d", "conv3d"}
    weight_list = [
        "lora_up.weight",
        "lora_down.weight",
        "lora_mid.weight",
        "alpha",
        "dora_scale",
        "pissa_A_init",
        "pissa_B_init",
        "pissa_converted",
        "n_split",
        # Block-diagonal weight keys (up to 64 blocks)
        *[f"lora_up_block{i}.weight" for i in range(64)],
        *[f"lora_down_block{i}.weight" for i in range(64)],
    ]
    weight_list_det = ["lora_up.weight", "lora_up_block0.weight"]

    # Registry of all RaLoRAModule instances for cross-module rank/GID allocation.
    _ralora_modules: list = []

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
        pissa_niter: int = 0,
        pissa_convert: bool = True,
        olora: bool = False,
        olora_lambda: float = 0.5,
        olora_task_id: int = 0,
        # --- RaLoRA-specific parameters ---
        ralora_n_max: int = 32,
        ralora_pro: bool = False,
        ralora_ref_rank: Optional[int] = None,
        ralora_min_rank: Optional[int] = None,
        ralora_max_rank: Optional[int] = None,
        ralora_dynamic_scaling: bool = False,
        ralora_rank_stabilize: bool = False,
        ralora_erank_method: str = "entropy",
        ralora_svd_threshold: float = 0.0,
        ralora_cumulative_variance: float = 0.0,
        ralora_forward_method: str = "concat",
        **kwargs,
    ):
        # Capture alpha before super().__init__ modifies it
        self.scaling_alpha = alpha

        super().__init__(
            lora_name=lora_name,
            org_module=org_module,
            multiplier=multiplier,
            lora_dim=lora_dim,
            alpha=alpha,
            dropout=dropout,
            rank_dropout=rank_dropout,
            module_dropout=module_dropout,
            use_tucker=use_tucker,
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
            pissa_niter=pissa_niter,
            pissa_convert=pissa_convert,
            olora=olora,
            olora_lambda=olora_lambda,
            olora_task_id=olora_task_id,
            **kwargs,
        )

        # RaLoRA configuration
        self.ralora_n_max = ralora_n_max
        self.ralora_pro = ralora_pro
        self.ralora_ref_rank = ralora_ref_rank if ralora_ref_rank is not None else lora_dim
        self.ralora_min_rank = ralora_min_rank
        self.ralora_max_rank = ralora_max_rank
        self.ralora_dynamic_scaling = ralora_dynamic_scaling
        self.ralora_rank_stabilize = ralora_rank_stabilize
        self.ralora_erank_method = ralora_erank_method
        self.ralora_svd_threshold = ralora_svd_threshold
        self.ralora_cumulative_variance = ralora_cumulative_variance
        self.ralora_forward_method = ralora_forward_method

        # Block-diagonal state (n_split=1 means vanilla LoRA)
        self.n_split = 1
        self.mini_lora_rank = lora_dim
        self.mini_in_features = self.shape[1] if len(self.shape) >= 2 else self.shape[0]
        self.mini_out_features = self.shape[0] if len(self.shape) >= 2 else 0
        self._mini_lora_A = nn.ParameterList()
        self._mini_lora_B = nn.ParameterList()

        # Register in global RaLoRA module list
        RaLoRAModule._ralora_modules.append(self)

    # ------------------------------------------------------------------
    # Class-level utilities
    # ------------------------------------------------------------------

    @classmethod
    def get_ralora_modules(cls, model: Optional[nn.Module] = None) -> list:
        """Get all RaLoRAModule instances, optionally filtered by model."""
        if model is not None:
            return [
                m for m in cls._ralora_modules
                if any(m is mod for mod in model.modules())
            ]
        return list(cls._ralora_modules)

    @classmethod
    def reset_ralora_registry(cls):
        """Clear the RaLoRA module registry."""
        cls._ralora_modules.clear()

    @classmethod
    @torch.no_grad()
    def precompute_and_init(
        cls,
        model: nn.Module,
        dataloader,
        forward_fn: Callable,
        ref_rank: Optional[int] = None,
        min_rank: Optional[int] = None,
        max_rank: Optional[int] = None,
        n_max: Optional[int] = None,
        pro_mode: Optional[bool] = None,
        erank_method: Optional[str] = None,
        svd_threshold: Optional[float] = None,
        cumulative_variance: Optional[float] = None,
        max_steps: int = 64,
        world_size: int = 1,
        global_rank: int = 0,
        device: Optional[torch.device] = None,
        save_dir: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, int]:
        """Class-level entry point for RaLoRA pre-computation.

        Finds all RaLoRAModule instances in *model*, reads their individual
        config, and delegates to :func:`ralora_utils.ralora_precompute_gradients`.

        Args:
            model: The model containing RaLoRAModule instances.
            dataloader: Training dataloader.
            forward_fn: Forward function (model, batch) -> loss.
            ref_rank: Override reference rank.
            min_rank: Override min rank.
            max_rank: Override max rank.
            n_max: Override max expansion factor.
            pro_mode: Override RaLoRA-Pro mode.
            erank_method: Override GID estimation method.
            svd_threshold: Threshold for threshold-based erank.
            cumulative_variance: Threshold for cumulative-variance erank.
            max_steps: Max gradient accumulation steps.
            world_size, global_rank: Distributed info.
            device: Compute device.
            save_dir: Directory to save rank.json / n_splits.json.

        Returns:
            {lora_name: allocated_rank} dict.
        """
        from .ralora_utils import ralora_precompute_gradients

        modules = cls.get_ralora_modules(model)
        if not modules:
            raise RuntimeError(
                "RaLoRA: No RaLoRAModule instances found in model. "
                "Make sure to create the LyCORIS network with algo='ralora'."
            )

        # Use first module's config as defaults, allow overrides
        first = modules[0]
        ref_rank = ref_rank if ref_rank is not None else first.ralora_ref_rank
        min_rank = min_rank if min_rank is not None else (first.ralora_min_rank or 1)
        max_rank = max_rank if max_rank is not None else (first.ralora_max_rank or first.lora_dim * 4)
        n_max = n_max if n_max is not None else first.ralora_n_max
        pro_mode = pro_mode if pro_mode is not None else first.ralora_pro
        erank_method = erank_method if erank_method is not None else first.ralora_erank_method
        svd_threshold = svd_threshold if svd_threshold is not None else first.ralora_svd_threshold
        cumulative_variance = cumulative_variance if cumulative_variance is not None else first.ralora_cumulative_variance

        return ralora_precompute_gradients(
            modules=modules,
            dataloader=dataloader,
            forward_fn=forward_fn,
            model=model,
            ref_rank=ref_rank,
            min_rank=min_rank,
            max_rank=max_rank,
            n_max=n_max,
            pro_mode=pro_mode,
            erank_method=erank_method,
            svd_threshold=svd_threshold,
            cumulative_variance=cumulative_variance,
            max_steps=max_steps,
            world_size=world_size,
            global_rank=global_rank,
            device=device,
            save_dir=save_dir,
        )

    # ------------------------------------------------------------------
    # Dynamic initialization
    # ------------------------------------------------------------------

    @torch.no_grad()
    def dynamic_init(self, avg_rank: int, rank: int, n_split: int = 1):
        """Rebuild adapter weights for block-diagonal LoRA after precomputation.

        Args:
            avg_rank: Reference/average rank (for scaling factor).
            rank: Per-layer LoRA rank r_l.
            n_split: Number of diagonal blocks n_l (must be a power of 2).
        """
        if n_split < 1:
            raise ValueError(f"n_split must be >= 1, got {n_split}")

        self.n_split = n_split
        self.lora_dim = rank if n_split == 1 else rank * n_split

        if n_split == 1:
            # Vanilla LoRA path — use existing lora_down/lora_up
            self.mini_lora_rank = rank
            self.mini_in_features = self.shape[1] if len(self.shape) >= 2 else self.shape[0]
            self.mini_out_features = self.shape[0] if len(self.shape) >= 2 else 0
            self._mini_lora_A = nn.ParameterList()
            self._mini_lora_B = nn.ParameterList()

            # Resize lora_down/lora_up if allocated rank differs from initial
            if rank != self.lora_down.out_features:
                in_dim = self.shape[1] if len(self.shape) >= 2 else self.shape[0]
                out_dim = self.shape[0] if len(self.shape) >= 2 else 0
                dtype = self.lora_down.weight.dtype
                device = self.lora_down.weight.device
                if self.isconv:
                    self.lora_down = self.module(
                        in_dim, rank, *self.shape[2:],
                        stride=self.lora_down.stride,
                        padding=self.lora_down.padding,
                        bias=False,
                    ).to(device=device, dtype=dtype)
                else:
                    self.lora_down = nn.Linear(in_dim, rank, bias=False).to(device=device, dtype=dtype)
                self.lora_up = nn.Linear(rank, out_dim, bias=False).to(device=device, dtype=dtype)
                # Re-initialize
                if self.use_orthogonal_init:
                    nn.init.orthogonal_(self.lora_down.weight)
                else:
                    nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
                nn.init.zeros_(self.lora_up.weight)

            self._update_scaling(avg_rank, rank)
            return

        # Tucker decomposition is not supported with block-diagonal structure
        if self.tucker:
            logger.warning(
                f"RaLoRA: Tucker decomposition is not supported with n_split > 1 "
                f"(module {self.lora_name}). Falling back to n_split=1."
            )
            self.n_split = 1
            self.lora_dim = rank
            self._update_scaling(avg_rank, rank)
            return

        # Check exact division
        in_features = self.shape[1] if len(self.shape) >= 2 else self.shape[0]
        out_features = self.shape[0] if len(self.shape) >= 2 else 0
        if in_features % n_split != 0:
            raise ValueError(
                f"in_features ({in_features}) must be divisible by n_split ({n_split})"
            )
        if out_features % n_split != 0:
            raise ValueError(
                f"out_features ({out_features}) must be divisible by n_split ({n_split})"
            )

        self.mini_lora_rank = rank
        self.mini_in_features = in_features // n_split
        self.mini_out_features = out_features // n_split

        # Create mini LoRA weights
        self._mini_lora_A = nn.ParameterList()
        self._mini_lora_B = nn.ParameterList()
        for _ in range(n_split):
            if self.isconv:
                # Conv: mini-A has kernel, mini-B is 1x1
                mini_a = nn.Parameter(
                    torch.empty((rank, self.mini_in_features, *self.shape[2:])),
                    requires_grad=True,
                )
                mini_b = nn.Parameter(
                    torch.zeros((self.mini_out_features, rank, *([1] * (len(self.shape) - 2)))),
                    requires_grad=True,
                )
            else:
                mini_a = nn.Parameter(
                    torch.empty((rank, self.mini_in_features)),
                    requires_grad=True,
                )
                mini_b = nn.Parameter(
                    torch.zeros((self.mini_out_features, rank)),
                    requires_grad=True,
                )
            self._mini_lora_A.append(mini_a)
            self._mini_lora_B.append(mini_b)

        # Initialize mini-A with kaiming
        for mini_a in self._mini_lora_A:
            if self.use_orthogonal_init:
                nn.init.orthogonal_(mini_a)
            else:
                nn.init.kaiming_uniform_(mini_a, a=math.sqrt(5))

        # Mini-B stays zero-initialized (standard LoRA convention)

        self._update_scaling(avg_rank, rank)

    def _update_scaling(self, avg_rank: int, rank: int):
        """Update the scaling factor based on rank configuration."""
        if self.ralora_dynamic_scaling:
            scale_rank = rank
        else:
            scale_rank = avg_rank

        if self.ralora_rank_stabilize:
            scale_rank = math.sqrt(scale_rank)

        r_factor = scale_rank
        self.scale = self.scaling_alpha / r_factor
        self.register_buffer("alpha", torch.tensor(self.scaling_alpha * (scale_rank / r_factor)))

        if self.rs_lora:
            self.scale = self.scaling_alpha / math.sqrt(scale_rank)

    # ------------------------------------------------------------------
    # Weight computation (override for block-diagonal)
    # ------------------------------------------------------------------

    def _make_weight_single(self, device=None):
        """Weight computation with block-diagonal support."""
        if self.n_split <= 1:
            return super()._make_weight_single(device)

        # Block-diagonal: assemble via torch.block_diag for linear,
        # manual placement for conv.
        wa_list = [
            self._orthogonalize(a.to(device)) for a in self._mini_lora_A
        ]
        wb_list = [
            self._orthogonalize(b.to(device)) for b in self._mini_lora_B
        ]

        if self.isconv:
            # Conv: assemble block-diagonal weight manually.
            # Each mini-A: (r, in_c/n, k1, k2), mini-B: (out_c/n, r, 1, 1)
            # Produced block: (out_c/n, in_c/n, k1, k2)
            weight = torch.zeros(self.shape, device=device)
            for i in range(self.n_split):
                wa = wa_list[i]  # (r, in_c/n, *kernel)
                wb = wb_list[i]  # (out_c/n, r, 1, ...)
                wa_2d = wa.view(wa.size(0), -1)   # (r, in_c/n * k_size)
                wb_2d = wb.view(wb.size(0), -1)   # (out_c/n, r)
                block_2d = wb_2d @ wa_2d           # (out_c/n, in_c/n * k_size)
                block = block_2d.view(self.mini_out_features, self.mini_in_features, *self.shape[2:])
                out_s = i * self.mini_out_features
                in_s = i * self.mini_in_features
                weight[out_s:out_s + self.mini_out_features,
                       in_s:in_s + self.mini_in_features] = block
        else:
            # Block-wise placement for linear — avoids materializing full
            # block-diagonal matrices (torch.block_diag) and the subsequent
            # dense matmul on the full (d_out, d_in) result.
            weight = torch.zeros(self.shape, device=device)
            for i in range(self.n_split):
                wa = wa_list[i]  # (r, mini_in)
                wb = wb_list[i]  # (mini_out, r)
                block = wb @ wa  # (mini_out, mini_in)
                out_s = i * self.mini_out_features
                in_s = i * self.mini_in_features
                weight[out_s:out_s + self.mini_out_features,
                       in_s:in_s + self.mini_in_features] = block

        weight = self._apply_rank_dropout(weight, device)

        return weight * self.scalar.to(device)

    def _compute_diff_weight_single(self, device, dtype):
        """Diff weight computation with block-diagonal support."""
        if self.n_split <= 1:
            return super()._compute_diff_weight_single(device, dtype)

        wa_list = [
            self._maybe_orthogonalize(a).to(device=device, dtype=dtype) for a in self._mini_lora_A
        ]
        wb_list = [
            self._maybe_orthogonalize(b).to(device=device, dtype=dtype) for b in self._mini_lora_B
        ]

        # Block-wise placement for both conv and linear — avoids materializing
        # full block-diagonal matrices and the dense (d_out, d_in) matmul.
        diff_weight = torch.zeros(self.shape, device=device, dtype=dtype)
        for i in range(self.n_split):
            wa = wa_list[i]  # (r, mini_in) or (r, mini_in, *kernel)
            wb = wb_list[i]  # (mini_out, r) or (mini_out, r, 1, ...)
            wa_2d = wa.view(wa.size(0), -1)   # (r, flat_in)
            wb_2d = wb.view(wb.size(0), -1)   # (mini_out, r)
            block_2d = wb_2d @ wa_2d           # (mini_out, flat_in)
            block = block_2d.view(self.mini_out_features, self.mini_in_features, *self.shape[2:])
            out_s = i * self.mini_out_features
            in_s = i * self.mini_in_features
            diff_weight[out_s:out_s + self.mini_out_features,
                         in_s:in_s + self.mini_in_features] = block

        diff_weight = self._apply_rank_dropout(diff_weight, device)

        diff_weight = (diff_weight * self.scalar.to(device=device)).to(dtype=dtype) * self.scale
        return diff_weight

    # ------------------------------------------------------------------
    # Bypass forward (override for block-diagonal)
    # ------------------------------------------------------------------

    def _bypass_forward_diff_single(self, x, scale=1):
        """Bypass forward with block-diagonal support."""
        if self.n_split <= 1:
            return super()._bypass_forward_diff_single(x, scale)

        device = x.device
        dtype = x.dtype

        if self.isconv:
            # Conv: for-loop over blocks, slice input channels, concat output channels
            up_parts = []
            for i in range(self.n_split):
                mini_x = x[:, i * self.mini_in_features:(i + 1) * self.mini_in_features, ...]
                wa = self._maybe_orthogonalize(self._mini_lora_A[i]).to(device=device, dtype=dtype)
                wb = self._maybe_orthogonalize(self._mini_lora_B[i]).to(device=device, dtype=dtype)
                mid = self.down_op(
                    mini_x, wa,
                    bias=None,
                    stride=self.lora_down.stride,
                    padding=self.lora_down.padding,
                    dilation=self.lora_down.dilation,
                    groups=self.lora_down.groups,
                )
                if self.rank_dropout and self.training:
                    drop = (
                        torch.rand(self.mini_lora_rank, device=device) > self.rank_dropout
                    ).to(dtype)
                    if self.rank_dropout_scale:
                        drop /= drop.mean()
                    drop = drop.view(1, -1, *([1] * (mid.dim() - 2)))
                    mid = mid * drop
                up_i = self.up_op(
                    mid, wb,
                    bias=None,
                    stride=self.lora_up.stride,
                    padding=self.lora_up.padding,
                    dilation=self.lora_up.dilation,
                    groups=self.lora_up.groups,
                )
                up_parts.append(up_i)
            up = torch.cat(up_parts, dim=1)  # concat along channel dim
        else:
            # Linear: block-wise forward without materializing full block-diag.
            # Slice input into blocks, apply per-block matmul, concat outputs.
            up_parts = []
            for i in range(self.n_split):
                in_s = i * self.mini_in_features
                mini_x = x[..., in_s:in_s + self.mini_in_features]
                wa = self._maybe_orthogonalize(self._mini_lora_A[i]).to(device=device, dtype=dtype)
                wb = self._maybe_orthogonalize(self._mini_lora_B[i]).to(device=device, dtype=dtype)
                mid = F.linear(x[..., in_s:in_s + self.mini_in_features], wa)

                if self.rank_dropout and self.training:
                    drop = (
                        torch.rand(self.mini_lora_rank, device=device) > self.rank_dropout
                    ).to(dtype)
                    if self.rank_dropout_scale:
                        drop /= drop.mean()
                    if (dims := len(mid.shape)) == 4:
                        drop = drop.view(1, -1, 1, 1)
                    else:
                        drop = drop.view(*[1] * (dims - 1), -1)
                    mid = mid * drop

                up_i = F.linear(mid, wb)
                up_parts.append(up_i)
            # Concat along the output (last) dimension
            up = torch.cat(up_parts, dim=-1)

        return self.drop(up * self.scalar.to(device) * self.scale * scale)

    # ------------------------------------------------------------------
    # Norm utilities (override for block-diagonal)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def apply_max_norm(self, max_norm, device=None):
        if self.n_split <= 1:
            return super().apply_max_norm(max_norm, device)

        orig_norm = self.make_weight(device).norm() * self.scale
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
        if self.n_split <= 1:
            return super().get_norm(device)
        return self.make_weight(device).norm()

    # ------------------------------------------------------------------
    # Serialization (override for block-diagonal)
    # ------------------------------------------------------------------

    def custom_state_dict(self):
        """Serialization with block-diagonal weight support."""
        if self.n_split <= 1:
            return super().custom_state_dict()

        destination = {}
        if self.wd:
            destination["dora_scale"] = self.dora_scale
        destination["alpha"] = self.alpha
        destination["n_split"] = torch.tensor(self.n_split)

        for i in range(self.n_split):
            scalar_val = self.scalar.to(
                device=self._mini_lora_B[i].device, non_blocking=True
            )
            destination[f"lora_up_block{i}.weight"] = (
                self._mini_lora_B[i] * scalar_val
            )
            destination[f"lora_down_block{i}.weight"] = self._mini_lora_A[i]

        if self.tucker:
            destination["lora_mid.weight"] = self.lora_mid.weight

        return destination

    def load_weight_prehook(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Pre-hook: intercept block-diagonal state dict keys before loading."""
        # Check for block-diagonal checkpoint
        n_split_key = f"{prefix}n_split"
        block_keys_present = (
            f"{prefix}lora_up_block0.weight" in state_dict
            or f"{prefix}lora_down_block0.weight" in state_dict
        )

        if not block_keys_present:
            return  # Not a block-diagonal checkpoint, let parent handle

        # Extract n_split
        n_split = 1
        if n_split_key in state_dict:
            n_split = int(state_dict.pop(n_split_key).item())

        # Extract block weights and build mini-A/B
        mini_a_list = []
        mini_b_list = []
        for i in range(n_split):
            a_key = f"{prefix}lora_down_block{i}.weight"
            b_key = f"{prefix}lora_up_block{i}.weight"
            if a_key in state_dict and b_key in state_dict:
                mini_a_list.append(state_dict.pop(a_key))
                # lora_up was saved with scalar baked in; we store scalar separately
                mini_b_list.append(state_dict.pop(b_key))
            else:
                break

        if len(mini_a_list) == n_split and len(mini_b_list) == n_split:
            # Restore block-diagonal structure
            self.n_split = n_split
            r = mini_a_list[0].size(0) if mini_a_list[0].dim() >= 2 else 0
            in_features = self.shape[1] if len(self.shape) >= 2 else self.shape[0]
            out_features = self.shape[0] if len(self.shape) >= 2 else 0
            self.mini_lora_rank = r
            self.mini_in_features = in_features // n_split
            self.mini_out_features = out_features // n_split
            self.lora_dim = r * n_split

            self._mini_lora_A = nn.ParameterList()
            self._mini_lora_B = nn.ParameterList()
            for i in range(n_split):
                self._mini_lora_A.append(
                    nn.Parameter(mini_a_list[i], requires_grad=True)
                )
                self._mini_lora_B.append(
                    nn.Parameter(mini_b_list[i], requires_grad=True)
                )
            # Reset scalar to 1 since it was baked into lora_up during save
            if isinstance(self.scalar, nn.Parameter):
                self.scalar.data.fill_(1.0)
            elif hasattr(self, "scalar"):
                self.scalar.fill_(1.0)

            # Remove block keys from unexpected (they were popped from state_dict)
            for i in range(n_split):
                a_key = f"{prefix}lora_down_block{i}.weight"
                b_key = f"{prefix}lora_up_block{i}.weight"
                if a_key in unexpected_keys:
                    unexpected_keys.remove(a_key)
                if b_key in unexpected_keys:
                    unexpected_keys.remove(b_key)
            if n_split_key in unexpected_keys:
                unexpected_keys.remove(n_split_key)

    @staticmethod
    def _should_skip_missing_key_ralora(key: str) -> bool:
        """Check if a missing key is expected to be absent in RaLoRA checkpoints."""
        if "scalar" in key:
            return True
        for tag in ("pissa_A_init", "pissa_B_init", "pissa_converted",
                     "olora_task_id", "n_split",
                     "lora_down_block", "lora_up_block",
                     "lora_down_task", "lora_up_task", "lora_mid_task"):
            if tag in key:
                return True
        return False

    def load_weight_hook(self, module: nn.Module, incompatible_keys):
        """Handle block-diagonal weight loading (post-hook)."""
        missing_keys = incompatible_keys.missing_keys

        # Filter expected missing keys in O(n) instead of O(n²)
        missing_keys[:] = [k for k in missing_keys if not self._should_skip_missing_key_ralora(k)]

        # Handle scalar
        if isinstance(self.scalar, nn.Parameter):
            self.scalar.data.copy_(torch.ones_like(self.scalar))
        elif getattr(self, "scalar", None) is not None:
            self.scalar.copy_(torch.ones_like(self.scalar))
        else:
            self.register_buffer(
                "scalar", torch.ones_like(self.scalar), persistent=False
            )

        # Initialize PiSSA buffers if not loaded
        if not hasattr(self, "pissa_A_init") or self.pissa_A_init is None:
            self.pissa_A_init = None
        if not hasattr(self, "pissa_B_init") or self.pissa_B_init is None:
            self.pissa_B_init = None

        # Initialize O-LoRA task ID
        if not hasattr(self, "olora_task_id"):
            self.olora_task_id = 0

    @classmethod
    def make_module_from_state_dict(
        cls, lora_name, orig_module, up, down, mid, alpha, dora_scale, n_split=None
    ):
        """Create module from state dict, handling block-diagonal checkpoints."""
        # If n_split is provided, it's a block-diagonal checkpoint
        if n_split is not None and n_split > 1:
            # Block-diagonal case handled by custom loading path
            # For now, delegate to parent with fallback
            pass

        module = cls(
            lora_name,
            orig_module,
            1.0,
            down.size(0) if down.dim() >= 2 else 0,
            float(alpha),
            use_tucker=mid is not None,
            weight_decompose=dora_scale is not None,
        )
        if hasattr(module, "n_split") and module.n_split <= 1:
            module.lora_up.weight.data.copy_(up)
            module.lora_down.weight.data.copy_(down)
        if mid is not None:
            module.lora_mid.weight.data.copy_(mid)
        if dora_scale is not None:
            module.dora_scale.copy_(dora_scale)
        return module

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def in_features(self) -> int:
        """Input feature dimension."""
        return self.shape[1] if len(self.shape) >= 2 else self.shape[0]

    @property
    def out_features(self) -> int:
        """Output feature dimension."""
        return self.shape[0] if len(self.shape) >= 2 else 0


class GoRAModule(LoConModule):
    """GoRA: Gradient-driven Adaptive Low Rank Adaptation.

    Extends LoConModule with gradient-based rank allocation and initialization.
    The saved checkpoint is identical to a standard LoConModule — only initialization
    and training-time dynamics differ.

    Key properties:
      - Always uses rsLoRA scaling (α / √r) for forward computation.
      - Accumulates pre-trained weight gradients on CPU via backward hooks.
      - Before training, computes layer importance from accumulated gradients
        and allocates ranks adaptively.
      - Initializes A₀ (lora_up) via Kaiming, then computes B₀ (lora_down)
        using the left pseudo-inverse per paper Eq. 9:
        B₀ = -(A₀ᵀ A₀)⁻¹ A₀ᵀ G

    Convention:
        ΔW = lora_up @ lora_down
        lora_up ∈ R^{m×r} = paper's A (randomly initialized)
        lora_down ∈ R^{r×n} = paper's B (computed from gradient)

    Reference: https://arxiv.org/abs/2502.12171

    Args:
        gora_ref_rank: Reference rank r^ref for parameter budget calculation.
        gora_min_rank: Minimum allowed rank per adapter (None = no min).
        gora_max_rank: Maximum allowed rank per adapter (None = no max).
        gora_gamma: Scaling factor γ for initialization magnitude.
        gora_importance_type: Importance metric (default: "union_mean" = avg(|W⊙G|)).
        gora_softmax_importance: Use softmax for importance normalization.
        gora_temperature: Temperature for softmax normalization.
        gora_scale_importance: Apply sqrt to raw importance scores.
        gora_features_func: Feature adjustment ("sqrt", "log1p", or None).
        gora_allocate_strategy: Rounding — "radical", "moderate", "conserved".
        gora_adaptive_gamma: Auto-tune γ on first batch.
        gora_weight_a_init: How to initialize A₀ ("kaiming", "weight_svd", "grad_svd").
        gora_scale_by_lr: Use learning rate in the scaling formula.
        gora_lr: Learning rate for lr-based scaling.
    """

    name = "gora"
    support_module = {"linear", "conv1d", "conv2d", "conv3d"}
    weight_list = [
        "lora_up.weight",
        "lora_down.weight",
        "lora_mid.weight",
        "alpha",
        "dora_scale",
        "pissa_A_init",
        "pissa_B_init",
        "pissa_converted",
    ]
    weight_list_det = ["lora_up.weight"]

    # Registry of all GoRAModule instances for cross-module rank allocation.
    _gora_modules: list = []

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
        rs_lora=None,                    # GoRA always forces rs_lora
        ggpo_beta: Optional[float] = None,
        ggpo_sigma: Optional[float] = None,
        ggpo_conv: bool = False,
        ggpo_conv_weight_sample_size: int = 100,
        orthogonalize=False,
        orthogonal_init=False,
        pissa_niter: int = 0,
        pissa_convert: bool = True,
        olora: bool = False,
        olora_lambda: float = 0.5,
        olora_task_id: int = 0,
        # --- GoRA-specific parameters ---
        gora_ref_rank: Optional[int] = None,
        gora_min_rank: Optional[int] = None,
        gora_max_rank: Optional[int] = None,
        gora_gamma: float = 5e-2,
        gora_importance_type: str = "union_mean",
        gora_softmax_importance: bool = False,
        gora_temperature: float = 0.5,
        gora_scale_importance: bool = False,
        gora_features_func: Optional[str] = None,
        gora_allocate_strategy: str = "moderate",
        gora_adaptive_gamma: bool = False,
        gora_weight_a_init: str = "kaiming",
        gora_scale_by_lr: bool = False,
        gora_lr: float = 1e-3,
        **kwargs,
    ):
        self.scaling_alpha = alpha  # Save before super().__init__ modifies it

        # GoRA always uses rsLoRA scaling (α / √r) as per the paper
        # (Section 3.3, Eq. 10)
        rs_lora = True

        super().__init__(
            lora_name=lora_name,
            org_module=org_module,
            multiplier=multiplier,
            lora_dim=lora_dim,
            alpha=alpha,
            dropout=dropout,
            rank_dropout=rank_dropout,
            module_dropout=module_dropout,
            use_tucker=use_tucker,
            use_scalar=use_scalar,
            scalar_init_value=scalar_init_value,
            rank_dropout_scale=rank_dropout_scale,
            weight_decompose=weight_decompose,
            wd_on_output=wd_on_output,
            bypass_mode=bypass_mode,
            rs_lora=True,
            ggpo_beta=ggpo_beta,
            ggpo_sigma=ggpo_sigma,
            ggpo_conv=ggpo_conv,
            ggpo_conv_weight_sample_size=ggpo_conv_weight_sample_size,
            orthogonalize=orthogonalize,
            orthogonal_init=orthogonal_init,
            pissa_niter=pissa_niter,
            pissa_convert=pissa_convert,
            olora=olora,
            olora_lambda=olora_lambda,
            olora_task_id=olora_task_id,
            **kwargs,
        )

        # Store GoRA configuration
        self.gora_ref_rank = gora_ref_rank if gora_ref_rank is not None else lora_dim
        self.gora_min_rank = gora_min_rank
        self.gora_max_rank = gora_max_rank
        self.gora_gamma = gora_gamma
        self.gora_importance_type = gora_importance_type
        self.gora_softmax_importance = gora_softmax_importance
        self.gora_temperature = gora_temperature
        self.gora_scale_importance = gora_scale_importance
        self.gora_features_func = gora_features_func
        self.gora_allocate_strategy = gora_allocate_strategy
        self.gora_adaptive_gamma = gora_adaptive_gamma
        self.gora_weight_a_init = gora_weight_a_init
        self.gora_scale_by_lr = gora_scale_by_lr
        self.gora_lr = gora_lr

        # Reconstruction error metrics (set during gora_dynamic_init)
        self._gora_recon_error: float = 0.0
        self._gora_relative_error: float = 0.0

        # Register in global GoRA module list
        GoRAModule._gora_modules.append(self)

    @property
    def in_features(self) -> int:
        """Input feature dimension of the weight matrix."""
        return self.shape[1]

    @property
    def out_features(self) -> int:
        """Output feature dimension of the weight matrix."""
        return self.shape[0]

    # ------------------------------------------------------------------
    # Custom state_dict — identical to LoConModule
    # ------------------------------------------------------------------
    def custom_state_dict(self):
        """GoRA saved checkpoint is identical to standard LoCon.

        GoRA-specific parameters (gamma, min_rank, etc.) are training-time only
        and do not need to be serialized. The adapter weights (lora_up, lora_down)
        are all that is needed at inference time.
        """
        # Delegate to parent — GoRA is just LoRA at inference time
        return super().custom_state_dict()

    # ------------------------------------------------------------------
    # Class-level utility: get all registered GoRA modules
    # ------------------------------------------------------------------

    @classmethod
    def get_gora_modules(cls, model: Optional[nn.Module] = None) -> list:
        """Get all GoRAModule instances.
        
        If model is provided, filters to modules within that model.
        """
        if model is not None:
            return [
                m for m in cls._gora_modules
                if any(m is mod for mod in model.modules())
            ]
        return list(cls._gora_modules)

    @classmethod
    def reset_gora_registry(cls):
        """Clear the GoRA module registry (e.g., before starting a fresh run)."""
        cls._gora_modules.clear()

    # ------------------------------------------------------------------
    # Convenience: one-shot precompute + init from class level
    # ------------------------------------------------------------------

    @classmethod
    @torch.no_grad()
    def precompute_and_init(
        cls,
        model: nn.Module,
        dataloader,
        forward_fn: Callable,
        ref_rank: Optional[int] = None,
        min_rank: Optional[int] = None,
        max_rank: Optional[int] = None,
        importance_type: Optional[str] = None,
        scaling_alpha: Optional[float] = None,
        stable_gamma: Optional[float] = None,
        max_steps: int = 64,
        adaptive_n: bool = True,
        convergence_threshold: float = 0.01,
        min_steps: int = 3,
        adaptive_gamma: bool = False,
        gamma_init: float = 1.0,
        gamma_decay: float = 0.8,
        gamma_min: float = 1e-5,
        world_size: int = 1,
        global_rank: int = 0,
        device: Optional[torch.device] = None,
        save_dir: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, int]:
        """Class-level entry point for GoRA pre-computation.

        Finds all GoRAModule instances in *model*, reads their individual
        config, and delegates to :func:`gora_utils.gora_precompute_gradients`.

        Args:
            model: The model containing GoRAModule instances.
            dataloader: Training dataloader.
            forward_fn: Forward function (model, batch) -> loss.
            ref_rank: Override reference rank (uses module config if None).
            min_rank: Override min rank.
            max_rank: Override max rank.
            importance_type: Override importance metric.
            scaling_alpha: Override α.
            stable_gamma: Override γ.
            max_steps: Max gradient accumulation steps.
            adaptive_n: Enable adaptive N.
            convergence_threshold: Threshold for adaptive N.
            min_steps: Min steps before convergence check.
            adaptive_gamma: Enable adaptive γ.
            gamma_init, gamma_decay, gamma_min: γ search params.
            world_size, global_rank: Distributed info.
            device: Compute device.
            save_dir: Directory to save rank.json/importance.json.

        Returns:
            {lora_name: allocated_rank} dict.
        """
        from .gora_utils import gora_precompute_gradients

        modules = cls.get_gora_modules(model)
        if not modules:
            raise RuntimeError(
                "GoRA: No GoRAModule instances found in model. "
                "Make sure to create the LyCORIS network with algo='gora'."
            )

        # Use first module's config as defaults, allow overrides
        first = modules[0]
        ref_rank = ref_rank if ref_rank is not None else first.gora_ref_rank
        min_rank = min_rank if min_rank is not None else (first.gora_min_rank or 1)
        max_rank = max_rank if max_rank is not None else (first.gora_max_rank or 32)
        importance_type = importance_type if importance_type is not None else first.gora_importance_type
        scaling_alpha = scaling_alpha if scaling_alpha is not None else first.scaling_alpha
        stable_gamma = stable_gamma if stable_gamma is not None else first.gora_gamma
        adaptive_gamma = adaptive_gamma or first.gora_adaptive_gamma

        return gora_precompute_gradients(
            modules=modules,
            dataloader=dataloader,
            forward_fn=forward_fn,
            ref_rank=ref_rank,
            min_rank=min_rank,
            max_rank=max_rank,
            importance_type=importance_type,
            scaling_alpha=scaling_alpha,
            stable_gamma=stable_gamma,
            max_steps=max_steps,
            adaptive_n=adaptive_n,
            convergence_threshold=convergence_threshold,
            min_steps=min_steps,
            adaptive_gamma=adaptive_gamma,
            gamma_init=gamma_init,
            gamma_decay=gamma_decay,
            gamma_min=gamma_min,
            softmax_importance=first.gora_softmax_importance,
            temperature=first.gora_temperature,
            scale_importance=first.gora_scale_importance,
            features_func=first.gora_features_func,
            allocate_strategy=first.gora_allocate_strategy,
            weight_a_init_method=first.gora_weight_a_init,
            scale_by_lr=first.gora_scale_by_lr,
            lr=first.gora_lr,
            world_size=world_size,
            global_rank=global_rank,
            device=device,
            save_dir=save_dir,
        )