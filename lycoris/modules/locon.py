import math
from functools import cache

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import LycorisBaseModule
from .pissa_utils import pissa_svd, convert_pissa_to_lora
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
            if use_tucker and any(i != 1 for i in k_size):
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
            org_weight = org_module.weight.cpu().clone().float()
            self.dora_norm_dims = org_weight.dim() - 1
            if self.wd_on_output:
                self.dora_scale = nn.Parameter(
                    torch.norm(
                        org_weight.reshape(org_weight.shape[0], -1),
                        dim=1,
                        keepdim=True,
                    ).reshape(org_weight.shape[0], *[1] * self.dora_norm_dims)
                ).float()
            else:
                self.dora_scale = nn.Parameter(
                    torch.norm(
                        org_weight.transpose(1, 0).reshape(org_weight.shape[1], -1),
                        dim=1,
                        keepdim=True,
                    )
                    .reshape(org_weight.shape[1], *[1] * self.dora_norm_dims)
                    .transpose(1, 0)
                ).float()

        if dropout and self.wd:
            log_wd()

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
            if self.olora:
                self.lora_scalar_list.append(self.scalar)
        else:
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

    def load_weight_hook(self, module: nn.Module, incompatible_keys):
        missing_keys = incompatible_keys.missing_keys
        # Allow missing keys that may not be present in all checkpoint variants
        pissa_keys = {"pissa_A_init", "pissa_B_init", "pissa_converted"}
        olora_keys = {"olora_task_id"}
        for key in list(missing_keys):
            if (
                "scalar" in key
                or any(pk in key for pk in pissa_keys)
                or any(ok in key for ok in olora_keys)
                or ("lora_down_task" in key)
                or ("lora_up_task" in key)
                or ("lora_mid_task" in key)
            ):
                del missing_keys[missing_keys.index(key)]
        if isinstance(self.scalar, nn.Parameter):
            self.scalar.data.copy_(torch.ones_like(self.scalar))
        elif getattr(self, "scalar", None) is not None:
            self.scalar.copy_(torch.ones_like(self.scalar))
        else:
            self.register_buffer(
                "scalar", torch.ones_like(self.scalar), persistent=False
            )
        # Initialize PiSSA buffers if not loaded from state dict
        if not hasattr(self, "pissa_A_init") or self.pissa_A_init is None:
            self.pissa_A_init = None
        if not hasattr(self, "pissa_B_init") or self.pissa_B_init is None:
            self.pissa_B_init = None
        # Initialize O-LoRA task ID if not loaded from state dict
        if not hasattr(self, "olora_task_id"):
            self.olora_task_id = 0

    def make_weight(self, device=None):
        if self.olora:
            return self._make_weight_multitask(device)
        return self._make_weight_single(device)

    def _make_weight_single(self, device=None):
        """Original single-task weight computation (used when olora=False)."""
        wa = self._orthogonalize(self.lora_up.weight.to(device))
        wb = self._orthogonalize(self.lora_down.weight.to(device))
        if self.tucker:
            t = self._orthogonalize(self.lora_mid.weight.to(device))
            wa = wa.view(wa.size(0), -1).transpose(0, 1)
            wb = wb.view(wb.size(0), -1)
            weight = rebuild_tucker(t, wa, wb)
        else:
            weight = wa.view(wa.size(0), -1) @ wb.view(wb.size(0), -1)

        weight = weight.view(self.shape)
        if self.training and self.rank_dropout:
            drop = (torch.rand(weight.size(0), device=device) > self.rank_dropout).to(
                weight.dtype
            )
            drop = drop.view(-1, *[1] * len(weight.shape[1:]))
            if self.rank_dropout_scale:
                drop /= drop.mean()
            weight *= drop

        return weight * self.scalar.to(device)

    def _make_weight_multitask(self, device=None):
        """Multi-task O-LoRA weight: sum of w_a @ w_b across all tasks."""
        total = None
        num_tasks = len(self.lora_down_modules)
        for idx in range(num_tasks):
            wa = self._orthogonalize(self.lora_up_modules[idx].weight.to(device))
            wb = self._orthogonalize(self.lora_down_modules[idx].weight.to(device))
            if self.tucker and len(self.lora_mid_modules) > idx:
                t = self._orthogonalize(self.lora_mid_modules[idx].weight.to(device))
                wa = wa.view(wa.size(0), -1).transpose(0, 1)
                wb = wb.view(wb.size(0), -1)
                task_weight = rebuild_tucker(t, wa, wb)
            else:
                task_weight = wa.view(wa.size(0), -1) @ wb.view(wb.size(0), -1)
            task_weight = task_weight.view(self.shape)
            if self.training and self.rank_dropout:
                drop = (torch.rand(task_weight.size(0), device=device) > self.rank_dropout).to(
                    task_weight.dtype
                )
                drop = drop.view(-1, *[1] * len(task_weight.shape[1:]))
                if self.rank_dropout_scale:
                    drop /= drop.mean()
                task_weight *= drop
            scalar = self.lora_scalar_list[idx] if idx < len(self.lora_scalar_list) else self.scalar
            task_weight = task_weight * scalar.to(device)
            if total is None:
                total = task_weight
            else:
                total = total + task_weight
        return total

    def _compute_diff_weight_single(self, device, dtype):
        """Single-task diff_weight for non-bypass forward (tucker or rank_dropout case)."""
        wa = self._orthogonalize(self.lora_up.weight).to(device=device, dtype=dtype)
        wb = self._orthogonalize(self.lora_down.weight).to(device=device, dtype=dtype)

        if self.tucker:
            t = self._orthogonalize(self.lora_mid.weight).to(device=device, dtype=dtype)
            wa = wa.view(wa.size(0), -1).transpose(0, 1)
            wb = wb.view(wb.size(0), -1)
            diff_weight = rebuild_tucker(t, wa, wb)
        else:
            diff_weight = wa.view(wa.size(0), -1) @ wb.view(wb.size(0), -1)

        diff_weight = diff_weight.view(self.shape)
        if self.training and self.rank_dropout:
            drop = (torch.rand(diff_weight.size(0), device=device) > self.rank_dropout).to(
                diff_weight.dtype
            )
            drop = drop.view(-1, *[1] * len(diff_weight.shape[1:]))
            if self.rank_dropout_scale:
                drop /= drop.mean()
            diff_weight *= drop

        diff_weight = (diff_weight * self.scalar.to(device=device)).to(dtype=dtype) * self.scale
        return diff_weight

    def _compute_diff_weight_multitask(self, device, dtype):
        """Multi-task O-LoRA diff_weight summing over all task LoRA pairs."""
        total = None
        num_tasks = len(self.lora_down_modules)
        for idx in range(num_tasks):
            wa = self._orthogonalize(self.lora_up_modules[idx].weight).to(device=device, dtype=dtype)
            wb = self._orthogonalize(self.lora_down_modules[idx].weight).to(device=device, dtype=dtype)

            if self.tucker and len(self.lora_mid_modules) > idx:
                t = self._orthogonalize(self.lora_mid_modules[idx].weight).to(device=device, dtype=dtype)
                wa = wa.view(wa.size(0), -1).transpose(0, 1)
                wb = wb.view(wb.size(0), -1)
                task_weight = rebuild_tucker(t, wa, wb)
            else:
                task_weight = wa.view(wa.size(0), -1) @ wb.view(wb.size(0), -1)

            task_weight = task_weight.view(self.shape)
            if self.training and self.rank_dropout:
                drop = (torch.rand(task_weight.size(0), device=device) > self.rank_dropout).to(
                    task_weight.dtype
                )
                drop = drop.view(-1, *[1] * len(task_weight.shape[1:]))
                if self.rank_dropout_scale:
                    drop /= drop.mean()
                task_weight *= drop

            scalar = self.lora_scalar_list[idx] if idx < len(self.lora_scalar_list) else self.scalar
            task_weight = (task_weight * scalar.to(device=device)).to(dtype=dtype) * self.scale
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
        weight = weight.to(self.dora_scale.dtype)
        if self.wd_on_output:
            weight_norm = (
                weight.reshape(weight.shape[0], -1)
                .norm(dim=1)
                .reshape(weight.shape[0], *[1] * self.dora_norm_dims)
            ) + torch.finfo(weight.dtype).eps
        else:
            weight_norm = (
                weight.transpose(0, 1)
                .reshape(weight.shape[1], -1)
                .norm(dim=1, keepdim=True)
                .reshape(weight.shape[1], *[1] * self.dora_norm_dims)
                .transpose(0, 1)
            ) + torch.finfo(weight.dtype).eps

        scale = self.dora_scale.to(weight.device) / weight_norm
        if multiplier != 1:
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
            lora_up_w = self.lora_up.weight * self.scalar.to(
                device=self.lora_up.weight.device, non_blocking=True
            )
            # Concatenate: trained A (up) with initial A₀ (up init)
            converted_up = torch.cat(
                [lora_up_w, self.pissa_A_init.to(lora_up_w.device)], dim=1
            )
            # Concatenate: trained B (down) with negated initial B₀ (down init)
            converted_down = torch.cat(
                [self.lora_down.weight, -self.pissa_B_init.to(self.lora_down.weight.device)], dim=0
            )
            destination["lora_up.weight"] = converted_up
            destination["lora_down.weight"] = converted_down
            destination["pissa_converted"] = torch.tensor(1.0)
            logger.info(
                f"PiSSA→LoRA conversion on save: {self.lora_name} "
                f"(rank {self.lora_dim} → {2 * self.lora_dim})"
            )
        else:
            destination["lora_up.weight"] = self.lora_up.weight * self.scalar.to(
                device=self.lora_up.weight.device, non_blocking=True
            )
            destination["lora_down.weight"] = self.lora_down.weight
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

        # Compute portable LoRA weights
        lora_up_curr = self.lora_up.weight.data.clone()
        lora_down_curr = self.lora_down.weight.data.clone()
        pissa_up_init = self.pissa_A_init.to(lora_up_curr.device)
        pissa_down_init = self.pissa_B_init.to(lora_down_curr.device)

        # ΔA = [A' | A₀]  → lora_up shape becomes (out, 2*r)
        delta_up = torch.cat([lora_up_curr, pissa_up_init], dim=1)
        # ΔB = [B' | -B₀] → lora_down shape becomes (2*r, in)
        delta_down = torch.cat([lora_down_curr, -pissa_down_init], dim=0)

        # Replace adapter weights
        self.lora_up.weight.data = delta_up
        self.lora_down.weight.data = delta_down
        self.lora_dim = 2 * self.lora_dim

        # Restore original weight: W = W^res + A₀B₀
        # The base weight currently holds W^res, so we add back A₀B₀
        orig_shape = self.org_module[0].weight.shape
        if self.isconv:
            a0b0 = (pissa_up_init @ pissa_down_init).reshape(orig_shape)
        else:
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
            self.scalar *= ratio
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
        total = torch.tensor(0.0)
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
        """Original single-task bypass forward diff (used when olora=False)."""
        # Orthogonalize weights on the fly for this forward pass.
        # This is only active during training if self.use_orthogonal_weights is True.
        wb = self._orthogonalize(self.lora_down.weight).to(x.device, dtype=x.dtype)
        wa = self._orthogonalize(self.lora_up.weight).to(x.device, dtype=x.dtype)

        # Manually apply the down network using the orthogonalized weight
        if self.isconv:
            # For convolution, we need to pass the module's parameters (stride, padding, etc.)
            mid = self.down_op(
                x,
                wb,
                bias=None,
                stride=self.lora_down.stride,
                padding=self.lora_down.padding,
                dilation=self.lora_down.dilation,
                groups=self.lora_down.groups,
            )
        else: # is linear
            mid = self.down_op(x, wb)

        if self.tucker:
            # CHANGE 3: Apply lora_mid operation manually with orthogonalized weight
            wc = self._orthogonalize(self.lora_mid.weight)
            mid = self.op(
                mid,
                wc,
                bias=None,
                stride=self.lora_mid.stride,
                padding=self.lora_mid.padding,
                dilation=self.lora_mid.dilation,
                groups=self.lora_mid.groups,
            )

        if self.rank_dropout and self.training:
            drop = (
                torch.rand(self.lora_dim, device=mid.device) > self.rank_dropout
            ).to(mid.dtype)
            if self.rank_dropout_scale:
                drop /= drop.mean()
            if (dims := len(x.shape)) == 4:
                drop = drop.view(1, -1, 1, 1)
            else:
                drop = drop.view(*[1] * (dims - 1), -1)
            mid = mid * drop

        # Manually apply the up network using the orthogonalized weight
        if self.isconv:
            # For convolution, we need to pass the module's parameters (stride, padding, etc.)
            up = self.up_op(
                mid,
                wa,
                bias=None,
                stride=self.lora_up.stride,
                padding=self.lora_up.padding,
                dilation=self.lora_up.dilation,
                groups=self.lora_up.groups,
            )
        else: # is linear
            up = self.up_op(mid, wa)

        return self.drop(up * self.scalar * self.scale * scale)

    def _bypass_forward_diff_multitask(self, x, scale=1):
        """Multi-task O-LoRA bypass forward diff: sum over all task LoRA pairs."""
        total_up = None
        num_tasks = len(self.lora_down_modules)
        for idx in range(num_tasks):
            down_module = self.lora_down_modules[idx]
            up_module = self.lora_up_modules[idx]
            wb = self._orthogonalize(down_module.weight).to(x.device, dtype=x.dtype)
            wa = self._orthogonalize(up_module.weight).to(x.device, dtype=x.dtype)

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
                wc = self._orthogonalize(self.lora_mid_modules[idx].weight)
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
                if (dims := len(x.shape)) == 4:
                    drop = drop.view(1, -1, 1, 1)
                else:
                    drop = drop.view(*[1] * (dims - 1), -1)
                mid = mid * drop

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

    def forward(self, x):
        if self.module_dropout and self.training:
            if torch.rand(1) < self.module_dropout:
                return self.org_forward(x)
        
        # Check if perturbation is needed - early return if not in training
        apply_ggpo = (self.training and 
                    self.ggpo_sigma is not None and 
                    self.ggpo_beta is not None and 
                    self.combined_weight_norms is not None and 
                    self.grad_norms is not None and
                    (self.module_type == "linear" or (self.module_type.startswith("conv") and self.ggpo_conv)))
        
        # Handle bypass mode first - simpler path
        if self.bypass_mode:
            result = self.bypass_forward(x, scale=self.multiplier)
            
            if apply_ggpo:
                with torch.no_grad():
                    perturbation_output = self.ggpo_pertubation(x)
                
                if perturbation_output is not None:
                    # Add perturbation to result and return
                    result = result + perturbation_output
                    
            return result
        
        # Non-bypass mode with perturbation
        dtype = self.dtype
        # Non-bypass mode: Get org_weight with async transfer
        org_weight_gpu = self.get_org_weight_for_compute(x.device).to(dtype, non_blocking=True)
        
        # Apply lora dropout during weight computation if enabled
        if (not self.wd and (self.tucker or self.rank_dropout)):
            if self.olora:
                diff_weight = self._compute_diff_weight_multitask(x.device, dtype)
            else:
                diff_weight = self._compute_diff_weight_single(x.device, dtype)
        else:
            diff_weight = self.make_weight(x.device).to(dtype) * self.scale
        
        # Apply the weight to the input
        weight = org_weight_gpu
        
        if self.wd:
            weight = self.apply_weight_decompose(weight + diff_weight, self.multiplier)

            # Input dropout for DoRA
            x = self.drop(x)
        else:
            weight = weight + diff_weight * self.multiplier
        
        # Get bias
        bias = self.get_org_bias_for_compute(x.device)
        if bias is not None:
            bias = bias.to(dtype, non_blocking=True)

        # Apply operation with weights
        result = self.op(x, weight, bias, **self.kw_dict)
        
        # Apply GGPO perturbation if needed
        if apply_ggpo:
            with torch.no_grad():
                perturbation_output = self.ggpo_pertubation(x)
                
            if perturbation_output is not None:
                # Add perturbation to result and return
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
            return self.op(x, perturbation, None, **self.kw_dict)
        else:
            return None


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
      - Initializes B (lora_up) as the pseudo-inverse compressed gradient:
        B₀ = G @ A₀ᵀ @ (A₀ @ A₀ᵀ)⁻¹

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