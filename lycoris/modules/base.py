from collections import OrderedDict
import re
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize

from ..utils.quant import QuantLinears, log_bypass, log_fp8_bypass, log_suspect
from ..logging import logger
from typing import Optional
import math
import torch._dynamo
torch._dynamo.config.recompile_limit = max(32, torch._dynamo.config.recompile_limit)

try:
    from peft.tuners.tuners_utils import BaseTunerLayer
except Exception:  # pragma: no cover - PEFT is optional
    BaseTunerLayer = None


def is_weight_only_fp8_linear(module: nn.Module) -> bool:
    return (
        module.__class__.__name__ == "Fp8Linear"
        and hasattr(module, "in_features")
        and hasattr(module, "out_features")
        and hasattr(module, "weight")
        and hasattr(module, "weight_scale")
    )


def is_linear_like_module(module: nn.Module) -> bool:
    return isinstance(module, nn.Linear) or is_weight_only_fp8_linear(module)


_FP8_BYPASS_ALGOS = frozenset({"lora", "locon", "loha", "lokr", "glora"})


def is_supported_linear_module(
    module: nn.Module, algo_name: str, *, weight_decompose: bool = False
) -> bool:
    if is_weight_only_fp8_linear(module):
        return algo_name in _FP8_BYPASS_ALGOS and (
            algo_name == "lokr" or not weight_decompose
        )
    return isinstance(module, nn.Linear)


def dequantize_weight_only_fp8(module: nn.Module) -> torch.Tensor:
    weight = module.weight.to(torch.float32)
    scale = module.weight_scale.to(device=weight.device, dtype=torch.float32)
    if scale.ndim == 1:
        scale = scale.unsqueeze(1)
    return weight * scale

try:
    from ramtorch.modules.linear import CPUBouncingLinear
except ImportError:
    CPUBouncingLinear = type(None)

class AsyncTensorStreamer:
    def __init__(self, device):
        self.device = device
        self.transfer_stream = torch.cuda.Stream(device=device)
        
        # RING BUFFER SETTINGS
        # Size 3 is safe: [Weight_Layer_N, Bias_Layer_N, Weight_Layer_N+1]
        # This prevents overwriting the weight currently being computed if the 
        # next layer starts transferring immediately.
        self.num_buffers = 3 
        self.idx = 0
        
        # We store (buffer, event) pairs. 
        # buffer: Holds the Tensor memory on GPU
        # event: Records when the Compute Stream is DONE using this buffer
        self.buffers = [None] * self.num_buffers
        self.compute_done_events = [torch.cuda.Event() for _ in range(self.num_buffers)]

    def transfer(self, tensor_cpu: torch.Tensor):
        # 1. Pin Memory (Crucial for Async)
        if not tensor_cpu.is_pinned():
            tensor_cpu = tensor_cpu.pin_memory()

        # 2. Select the next slot in the Ring Buffer
        slot_idx = self.idx
        self.idx = (self.idx + 1) % self.num_buffers
        
        ready_event = self.compute_done_events[slot_idx]
        
        # 3. SYNC: Wait for the PREVIOUS Compute cycle to finish with this specific slot
        # We cannot overwrite this slot if the GPU is still doing Math on the data previously stored here.
        # (For the first run, the event is unrecorded, so this is a no-op).
        self.transfer_stream.wait_event(ready_event)

        with torch.cuda.stream(self.transfer_stream):
            with torch.no_grad():
                # 4. TRANSFER / ALLOCATE
                # We use .to() which uses PyTorch's Caching Allocator. 
                # If self.buffers[slot_idx] existed, it goes back to the pool.
                # We don't manually hold .new_empty() anymore to allow dynamic resizing 
                # if layers have different shapes.
                gpu_tensor = tensor_cpu.to(self.device, non_blocking=True)
                
                # Keep a reference in our ring buffer list so Python doesn't GC it 
                # before the stream operation completes.
                self.buffers[slot_idx] = gpu_tensor
            
            # Record that transfer is finished
            transfer_finished_event = torch.cuda.Event()
            transfer_finished_event.record()

        # 5. SYNC: Tell the Compute Stream (Current Stream) to wait for transfer
        torch.cuda.current_stream().wait_event(transfer_finished_event)
        
        # 6. Mark usage
        # Record an event on the Compute Stream. 
        # The NEXT time we try to write to 'slot_idx', we will wait for this event.
        ready_event.record()
        
        return self.buffers[slot_idx]

# Global registry for multi-gpu support
_STREAMERS = {}

def transfer_ramtensor_to_device(tensor_cpu: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Args:
        tensor_id: Used for debugging/logging, but no longer used for memory allocation keys.
    """
    if not getattr(tensor_cpu, 'is_ramtorch', False):
        return tensor_cpu.to(device, non_blocking=True)
    
    if device.type == 'cpu':
        return tensor_cpu

    if device not in _STREAMERS:
        _STREAMERS[device] = AsyncTensorStreamer(device)
    
    return _STREAMERS[device].transfer(tensor_cpu)

class ModuleCustomSD(nn.Module):
    def __init__(self):
        super().__init__()
        self._register_load_state_dict_pre_hook(self.load_weight_prehook)
        self.register_load_state_dict_post_hook(self.load_weight_hook)

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
        pass

    def load_weight_hook(self, module, incompatible_keys):
        pass

    def custom_state_dict(self):
        return None

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        # TODO: Remove `args` and the parsing logic when BC allows.
        if len(args) > 0:
            if destination is None:
                destination = args[0]
            if len(args) > 1 and prefix == "":
                prefix = args[1]
            if len(args) > 2 and keep_vars is False:
                keep_vars = args[2]
            # DeprecationWarning is ignored by default

        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()

        local_metadata = dict(version=self._version)
        if hasattr(destination, "_metadata"):
            destination._metadata[prefix[:-1]] = local_metadata

        if (custom_sd := self.custom_state_dict()) is not None:
            for k, v in custom_sd.items():
                destination[f"{prefix}{k}"] = v
            return destination
        else:
            return super().state_dict(
                *args, destination=destination, prefix=prefix, keep_vars=keep_vars
            )


@dataclass
class _MergeContext:
    precise: bool
    target_device: torch.device
    target_dtype: torch.dtype
    compute_dtype: torch.dtype
    param_device: torch.device | None
    param_dtype: torch.dtype | None
    module: nn.Module
    weight_param: torch.Tensor
    bias_param: torch.Tensor | None


class LycorisBaseModule(ModuleCustomSD):
    name: str
    dtype_tensor: torch.Tensor
    support_module = {}
    weight_list = []
    weight_list_det = []

    def __init__(
        self,
        lora_name: str,
        org_module: nn.Module,
        multiplier: float = 1.0,
        dropout: float = 0.0,
        rank_dropout: float = 0.0,
        module_dropout: float = 0.0,
        rank_dropout_scale: bool = False,
        bypass_mode: bool = False,
        ggpo_beta: Optional[float] = None,
        ggpo_sigma: Optional[float] = None,
        ggpo_conv: bool = False,
        ggpo_conv_weight_sample_size: int = 100,
        **kwargs,
    ):
        """if alpha == 0 or None, alpha is rank (no scaling)."""
        super().__init__()
        self.lora_name = lora_name
        self.not_supported = False
        self.grad_count = 0
        self.sum_grads = None
        self.sum_squared_grads = None

        self.is_ramtorch_org = isinstance(org_module, CPUBouncingLinear)
        if self.is_ramtorch_org:
            logger.info(f"RamTorch module detected: {lora_name}")

        self.peft_wrapper = None
        if BaseTunerLayer is not None and isinstance(org_module, BaseTunerLayer):
            self.peft_wrapper = org_module
            base_layer = getattr(org_module, "base_layer", None)
            if base_layer is None and hasattr(org_module, "get_base_layer"):
                base_layer = org_module.get_base_layer()
            if base_layer is not None:
                org_module = base_layer

        self.module = type(org_module)
        if is_linear_like_module(org_module) or isinstance(
            org_module, CPUBouncingLinear
        ):
            self.module_type = "linear"
            self.shape = (org_module.out_features, org_module.in_features)
            self.op = F.linear
            self.dim = org_module.out_features
            self.kw_dict = {}
        elif isinstance(org_module, nn.Conv1d):
            self.module_type = "conv1d"
            # Weight layout matches torch.nn.Conv*d: (out, in/groups, *kernel)
            self.shape = (
                org_module.out_channels,
                org_module.in_channels // org_module.groups,
                *org_module.kernel_size,
            )
            self.op = F.conv1d
            self.dim = org_module.out_channels
            self.kw_dict = {
                "stride": org_module.stride,
                "padding": org_module.padding,
                "dilation": org_module.dilation,
                "groups": org_module.groups,
            }
        elif isinstance(org_module, nn.Conv2d):
            self.module_type = "conv2d"
            self.shape = (
                org_module.out_channels,
                org_module.in_channels // org_module.groups,
                *org_module.kernel_size,
            )
            self.op = F.conv2d
            self.dim = org_module.out_channels
            self.kw_dict = {
                "stride": org_module.stride,
                "padding": org_module.padding,
                "dilation": org_module.dilation,
                "groups": org_module.groups,
            }
        elif isinstance(org_module, nn.Conv3d):
            self.module_type = "conv3d"
            self.shape = (
                org_module.out_channels,
                org_module.in_channels // org_module.groups,
                *org_module.kernel_size,
            )
            self.op = F.conv3d
            self.dim = org_module.out_channels
            self.kw_dict = {
                "stride": org_module.stride,
                "padding": org_module.padding,
                "dilation": org_module.dilation,
                "groups": org_module.groups,
            }
        elif isinstance(org_module, nn.LayerNorm):
            self.module_type = "layernorm"
            self.shape = tuple(org_module.normalized_shape)
            self.op = F.layer_norm
            self.dim = org_module.normalized_shape[0]
            self.kw_dict = {
                "normalized_shape": org_module.normalized_shape,
                "eps": org_module.eps,
            }
        elif isinstance(org_module, nn.GroupNorm):
            self.module_type = "groupnorm"
            self.shape = (org_module.num_channels,)
            self.op = F.group_norm
            self.group_num = org_module.num_groups
            self.dim = org_module.num_channels
            self.kw_dict = {"num_groups": org_module.num_groups, "eps": org_module.eps}
        else:
            self.not_supported = True
            self.module_type = "unknown"

        # Compile-friendly conv parameter cache (avoids **kw_dict graph breaks)
        if self.module_type.startswith("conv"):
            self._conv_stride = org_module.stride
            self._conv_padding = org_module.padding
            self._conv_dilation = org_module.dilation
            self._conv_groups = org_module.groups

        # Compile-friendly norm parameter cache (avoids **kw_dict graph breaks)
        if self.module_type == "layernorm":
            self._ln_normalized_shape = org_module.normalized_shape
            self._ln_eps = org_module.eps
        elif self.module_type == "groupnorm":
            self._gn_num_groups = org_module.num_groups
            self._gn_eps = org_module.eps

        # Pre-compute expected input ndim and rank-dropout view shapes.
        # These are constants per module, so torch.compile specializes once
        # and never recompiles due to ndim changes.
        _NDIM_MAP = {"linear": 2, "conv1d": 3, "conv2d": 4, "conv3d": 5}
        self._expected_ndim = _NDIM_MAP.get(self.module_type, None)
        if self._expected_ndim is not None:
            ndim = self._expected_ndim
            # Weight-based rank dropout: drop viewed as (out, 1, 1, ...)
            self._rank_drop_shape = [-1] + [1] * (ndim - 1)
            # Bypass-mode rank dropout: drop viewed as (1, rank, 1, 1, ...)
            self._bypass_rank_drop_shape = [1, -1] + [1] * max(ndim - 2, 0)

        # Compile-friendly QR orthogonalization cache.
        # Cache the weight ndim so _orthogonalize never calls len(shape)
        # inside compiled graphs (avoids Python container traversal on
        # symbolic shapes).  The rows >= cols check on the 2D matrix
        # remains as a Dynamo guard (shapes are static parameters, so
        # Dynamo specializes once without graph breaks).
        self._qr_ndim = self._expected_ndim  # None for norm modules

        self.register_buffer("dtype_tensor", torch.tensor(0.0), persistent=False)
        # Cache dtype as plain attribute to avoid property access in compiled graphs.
        # Kept in sync via _apply() override.
        self._cached_dtype = self.dtype_tensor.dtype

        self.is_quant = False
        if is_weight_only_fp8_linear(org_module):
            if not bypass_mode:
                log_fp8_bypass()
            self.is_quant = True
            bypass_mode = True
        elif isinstance(org_module, QuantLinears):
            if not bypass_mode:
                log_bypass()
            self.is_quant = True
            bypass_mode = True
        if (
            is_linear_like_module(org_module)
            and org_module.__class__.__name__ != "Linear"
        ):
            if bypass_mode is None:
                log_suspect()
                bypass_mode = True
            if bypass_mode == True:
                self.is_quant = True
        self.bypass_mode = bypass_mode
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.rank_dropout_scale = rank_dropout_scale
        self.module_dropout = module_dropout

        ## Dropout things
        # Since LoKr/LoHa/OFT/BOFT are hard to follow the rank_dropout definition from kohya
        # We redefine the dropout procedure here.
        # g(x) = WX + drop(Brank_drop(AX)) for LoCon(lora), bypass
        # g(x) = WX + drop(ΔWX) for any algo except LoCon(lora), bypass
        # g(x) = (W + Brank_drop(A))X for LoCon(lora), rebuid
        # g(x) = (W + rank_drop(ΔW))X for any algo except LoCon(lora), rebuild
        self.drop = nn.Identity() if dropout == 0 else nn.Dropout(dropout)
        self.rank_drop = (
            nn.Identity() if rank_dropout == 0 else nn.Dropout(rank_dropout)
        )

        self._multiplier = float(multiplier)
        self.register_buffer(
            "multiplier_buf",
            torch.tensor(float(multiplier), dtype=torch.float32),
            persistent=False,
        )
        self.org_forward = org_module.forward
        self.org_module = [org_module]

        self.ggpo_sigma = ggpo_sigma
        self.ggpo_beta = ggpo_beta
        self.ggpo_conv = ggpo_conv
        self.ggpo_conv_weight_sample_size = ggpo_conv_weight_sample_size

        # Weight noising config — set after construction by the network
        # wrapper, or passed through **kwargs from subclasses that forward them.
        self.weight_noise_sigma = kwargs.get('weight_noise_sigma', None)
        self.weight_noise_mode = kwargs.get('weight_noise_mode', 'relative')
        self.weight_noise_dynamic_sigma = kwargs.get('weight_noise_dynamic_sigma', False)

    def _call_op(self, x, weight, bias=None):
        """Compile-friendly op dispatch — avoids ``**kw_dict`` graph breaks.

        Uses explicit ``F.linear`` / ``F.conv{1,2,3}d`` calls with cached
        conv parameters instead of ``self.op(x, w, b, **self.kw_dict)``.
        """
        mt = self.module_type
        if mt == "linear":
            return F.linear(x, weight, bias)
        if mt == "conv1d":
            return F.conv1d(
                x, weight, bias,
                stride=self._conv_stride, padding=self._conv_padding,
                dilation=self._conv_dilation, groups=self._conv_groups,
            )
        if mt == "conv2d":
            return F.conv2d(
                x, weight, bias,
                stride=self._conv_stride, padding=self._conv_padding,
                dilation=self._conv_dilation, groups=self._conv_groups,
            )
        if mt == "conv3d":
            return F.conv3d(
                x, weight, bias,
                stride=self._conv_stride, padding=self._conv_padding,
                dilation=self._conv_dilation, groups=self._conv_groups,
            )
        # Explicit norm dispatch — avoids **kw_dict graph breaks
        if mt == "layernorm":
            return F.layer_norm(
                x, self._ln_normalized_shape, weight, bias, eps=self._ln_eps
            )
        if mt == "groupnorm":
            return F.group_norm(
                x, self._gn_num_groups, weight, bias, eps=self._gn_eps
            )
        raise NotImplementedError(
            f"Unsupported module_type '{mt}' in _call_op. "
            f"Supported: linear, conv1d, conv2d, conv3d, layernorm, groupnorm."
        )

    def _call_op_1x1(self, x, weight, bias=None):
        """Compile-friendly 1×1 op dispatch — for 1×1 convolutions.

        Like :meth:`_call_op` but uses default conv params (stride=1,
        padding=0, dilation=1, groups=1).  Correct for lora_up (always
        1×1) and lora_down in tucker mode (1×1).
        """
        mt = self.module_type
        if mt == "linear":
            return F.linear(x, weight, bias)
        if mt == "conv1d":
            return F.conv1d(x, weight, bias)
        if mt == "conv2d":
            return F.conv2d(x, weight, bias)
        if mt == "conv3d":
            return F.conv3d(x, weight, bias)
        # Explicit norm dispatch — avoids **kw_dict graph breaks
        if mt == "layernorm":
            return F.layer_norm(
                x, self._ln_normalized_shape, weight, bias, eps=self._ln_eps
            )
        if mt == "groupnorm":
            return F.group_norm(
                x, self._gn_num_groups, weight, bias, eps=self._gn_eps
            )
        raise NotImplementedError(
            f"Unsupported module_type '{mt}' in _call_op_1x1. "
            f"Supported: linear, conv1d, conv2d, conv3d, layernorm, groupnorm."
        )

    def _orthogonalize(self, weight_matrix: torch.Tensor) -> torch.Tensor:
        """Orthogonalizes the weight matrix using QR decomposition.

        Compile-friendly: uses pre-cached ``_qr_ndim`` (from
        ``_expected_ndim``) instead of ``len(shape)`` to avoid Python
        container traversal on symbolic tensor shapes inside compiled
        graphs.  The ``rows >= cols`` check on the 2-D matrix is a
        static Dynamo guard (weight shapes never change) — not a graph
        break.
        """
        if not self.use_orthogonal_weights or not self.training:
            return weight_matrix

        shape = weight_matrix.shape

        # Use pre-cached ndim when available to avoid len(shape) in
        # compiled graphs.  Fall back to .dim() for unknown module types.
        ndim = self._qr_ndim if self._qr_ndim is not None else weight_matrix.dim()
        if ndim == 0:
            return weight_matrix

        # Reshape to 2-D for QR decomposition.
        # shape[0] avoids len(tensor) which traverses Python container.
        if ndim > 2:
            weight_matrix = weight_matrix.reshape(shape[0], -1)
        elif ndim < 2:
            weight_matrix = weight_matrix.reshape(1, -1)

        # Upcast to fp32 for QR (bf16 CUDA kernel not implemented)
        orig_dtype = weight_matrix.dtype
        weight_matrix_fp32 = weight_matrix.to(torch.float32)

        # For matrices where rows >= cols, QR gives orthonormal columns.
        # For matrices where rows < cols, we transpose to make columns from rows,
        # apply QR, and transpose back. This results in orthonormal rows.
        # NOTE: rows/cols are static nn.Parameter dimensions → Dynamo guards
        # once without graph breaks.
        rows, cols = weight_matrix.shape
        if rows >= cols:
            q, r = torch.linalg.qr(weight_matrix_fp32)
            weight_matrix_fp32 = q * torch.diag(r)
        else:
            q, r = torch.linalg.qr(weight_matrix_fp32.T)
            weight_matrix_fp32 = (q * torch.diag(r)).T
        return weight_matrix_fp32.to(orig_dtype).reshape(shape).contiguous()

    @classmethod
    def parametrize(cls, org_module, attr, *args, **kwargs):
        from .full import FullModule

        if cls is FullModule:
            raise RuntimeError("FullModule cannot be used for parametrize.")
        target_param = getattr(org_module, attr)
        kwargs["bypass_mode"] = False
        if target_param.dim() == 2:
            proxy_module = nn.Linear(
                target_param.shape[0], target_param.shape[1], bias=False
            )
            proxy_module.weight = target_param
        elif target_param.dim() > 2:
            module_type = [
                None,
                None,
                None,
                nn.Conv1d,
                nn.Conv2d,
                nn.Conv3d,
                None,
                None,
            ][target_param.dim()]
            proxy_module = module_type(
                target_param.shape[0],
                target_param.shape[1],
                *target_param.shape[2:],
                bias=False,
            )
            proxy_module.weight = target_param
        module_obj = cls("", proxy_module, *args, **kwargs)
        module_obj.forward = module_obj.parametrize_forward
        module_obj.to(target_param)
        parametrize.register_parametrization(org_module, attr, module_obj)
        return module_obj

    @classmethod
    def algo_check(cls, state_dict, lora_name):
        return any(f"{lora_name}.{k}" in state_dict for k in cls.weight_list_det)

    @classmethod
    def extract_state_dict(cls, state_dict, lora_name):
        return [state_dict.get(f"{lora_name}.{k}", None) for k in cls.weight_list]

    @classmethod
    def make_module_from_state_dict(cls, lora_name, orig_module, *weights):
        raise NotImplementedError

    @property
    def dtype(self):
        return self._cached_dtype

    @property
    def device(self):
        return self.dtype_tensor.device

    def _apply(self, fn, recurse=True):
        """Override to keep _cached_dtype in sync after .to() / .cuda() etc."""
        result = super()._apply(fn, recurse=recurse)
        self._cached_dtype = self.dtype_tensor.dtype
        if hasattr(self, '_multiplier'):
            self._multiplier = float(self.multiplier_buf.item())
        return result

    # Top-level model component prefixes whose parameters are NOT "hidden"
    # layers (embeddings, input/output projections, timestep conditioning).
    # Checked against ``original_name`` which is the dotted path into the
    # root model (e.g. ``time_embedding.linear_1``).
    _NON_HIDDEN_NAME_PREFIXES = (
        'time_embedding', 'time_in', 'timestep_embedding',
        'vector_in', 'guidance_in',
        'img_in', 'txt_in',
        'conv_in', 'conv_out',
        'final_layer',
        'x_embedder', 'pos_embedder', 'patch_embed', 'context_embedder',
    )

    # Patterns matching normalization & AdaLN / modulation components across
    # DiT, Flux, SD3, PixArt, Lumina, Anima, and other transformer models.
    _ADALN_NAME_PATTERN = re.compile(
        r'(?:ada_?ln|modulation|norm\d*_linear|norm\d*_context_linear|norm_linear|_modulation)',
        re.IGNORECASE
    )

    def _is_norm_module(self) -> bool:
        """Determine if this module targets a normalization layer or AdaLN modulation."""
        if self.module_type in ("layernorm", "groupnorm"):
            return True

        if self.__class__.__name__ == "NormModule":
            return True

        mod_cls = getattr(self, "module", None)
        if mod_cls is not None:
            mod_cls_name = getattr(mod_cls, "__name__", "")
            if any(norm_term in mod_cls_name.lower() for norm_term in ("norm", "rmsnorm", "layernorm", "groupnorm")):
                return True

        if len(self.org_module) > 0 and self.org_module[0] is not None:
            org = self.org_module[0]
            org_cls_name = org.__class__.__name__.lower()
            if any(norm_term in org_cls_name for norm_term in ("norm", "rmsnorm", "layernorm", "groupnorm")):
                return True

        original_name = getattr(self, "original_name", "") or ""
        if self._ADALN_NAME_PATTERN.search(original_name):
            return True

        lora_name = getattr(self, "lora_name", "") or ""
        if self._ADALN_NAME_PATTERN.search(lora_name):
            return True

        return False

    def tag_parameters(self):
        """Tag nn.Parameter objects with optimizer-relevant attributes.

        Sets ``_is_dora_scale``, ``_is_oft``, ``_is_lora_A``, ``_is_lora_B``,
        ``is_hidden``, ``is_vector``, ``is_norm``, ``is_scalar``, ``is_bias``,
        and ``weight_decay_ratio`` so that Advanced_Optimizers and parameter
        group builders can identify each parameter's role and apply appropriate
        learning rates, weight decays, and algorithmic adjustments.

        ``is_hidden`` is determined by checking ``original_name`` (the dotted
        path into the root model) against a set of known non-hidden prefixes
        (embeddings, input/output projections, timestep conditioning layers).

        ``is_norm`` is tagged for normalization layers (LayerNorm, RMSNorm,
        GroupNorm, NormModule) and AdaLN / modulation projections.

        ``is_scalar`` is tagged for single-scalar parameters (e.g. ``scalar``,
        ``lora2_nu``, or single-element tensors).

        ``is_bias`` is tagged for additive bias terms (e.g. ``bias``, ``b_norm``,
        ``diff_b``).

        ``weight_decay_ratio`` indicates the multiplier relative to the base
        optimizer's configured weight decay:
        * ``0.0``: No weight decay (biases, normalization weights, scalars,
          DoRA scales, 1D vectors).
        * ``1.0``: Full weight decay (2D+ adapter weight matrices, OFT blocks,
          full diff weights).
        * Custom float: Any pre-existing ``weight_decay_ratio`` set on the
          parameter or module is respected.

        .. note::
            Must be called **after** any device moves (``.to()`` / ``.cuda()``)
            since those replace ``nn.Parameter`` objects via ``_apply`` and
            drop custom tensor attributes. The network calls this from
            ``prepare_optimizer_params()`` / ``prepare_grad_etc()``.
        """
        # Determine if this module targets a hidden layer by checking
        # original_name against known non-hidden top-level components.
        original_name = getattr(self, 'original_name', None) or ''
        is_hidden = not any(original_name.startswith(pfx)
                           for pfx in self._NON_HIDDEN_NAME_PREFIXES)
        is_norm = self._is_norm_module()

        # --- OFT blocks (DiagOFT / BOFT) ---
        oft_blocks = getattr(self, 'oft_blocks', None)
        if isinstance(oft_blocks, nn.Parameter):
            oft_blocks._is_oft = True

        # --- DoRA magnitude scale ---
        dora_scale = getattr(self, 'dora_scale', None)
        if isinstance(dora_scale, nn.Parameter):
            dora_scale._is_dora_scale = True
            dora_scale.is_vector = True

        # --- OFT per-channel rescale (multi-dim vector) ---
        rescale = getattr(self, 'rescale', None)
        if isinstance(rescale, nn.Parameter):
            rescale.is_vector = True

        # --- Standard / ABBA LoRA down(up) sub-modules ---
        # lora_down -> A, lora_up -> B (and the ABBA 1/2 variants)
        _lora_factor_attrs = (
            ('lora_down', '_is_lora_A'),
            ('lora_up', '_is_lora_B'),
            ('lora_down1', '_is_lora_A'),
            ('lora_up1', '_is_lora_B'),
            ('lora_down2', '_is_lora_A'),
            ('lora_up2', '_is_lora_B'),
        )
        for attr, tag in _lora_factor_attrs:
            sub = getattr(self, attr, None)
            if isinstance(sub, nn.Module) and hasattr(sub, 'weight'):
                w = sub.weight
                if isinstance(w, nn.Parameter):
                    setattr(w, tag, True)
                    w.is_hidden = is_hidden

        # --- DyLora ModuleList factors ---
        for list_attr, tag in (
            ('down_list', '_is_lora_A'),
            ('up_list', '_is_lora_B'),
        ):
            lst = getattr(self, list_attr, None)
            if isinstance(lst, nn.ModuleList):
                for mod in lst:
                    if hasattr(mod, 'weight') and isinstance(mod.weight, nn.Parameter):
                        setattr(mod.weight, tag, True)
                        mod.weight.is_hidden = is_hidden

        # --- Block-split mini LoRA factors (lists / ParameterList of raw tensors) ---
        for list_attr, tag in (
            ('_mini_lora_A', '_is_lora_A'),
            ('_mini_lora_B', '_is_lora_B'),
        ):
            lst = getattr(self, list_attr, None)
            if isinstance(lst, (list, nn.ParameterList)):
                for p in lst:
                    if isinstance(p, nn.Parameter):
                        setattr(p, tag, True)
                        p.is_hidden = is_hidden

        # --- Tag all trainable parameters with is_norm, is_scalar, is_bias, and weight_decay_ratio ---
        for name, p in self.named_parameters():
            if not isinstance(p, nn.Parameter):
                continue

            # 1. Bias Tagging
            p_is_bias = (
                name.endswith('bias')
                or name.endswith('b_norm')
                or name.endswith('diff_b')
                or getattr(p, '_is_bias', False)
                or getattr(p, 'is_bias', False)
            )
            p.is_bias = p_is_bias

            # 2. Scalar Tagging
            p_is_scalar = (
                p.numel() == 1
                or name.endswith('scalar')
                or name.endswith('lora2_nu')
                or getattr(p, '_is_scalar', False)
                or getattr(p, 'is_scalar', False)
            )
            p.is_scalar = p_is_scalar

            # 3. Norm Tagging
            p_is_norm = is_norm or name.endswith('w_norm') or name.endswith('b_norm') or getattr(p, 'is_norm', False)
            p.is_norm = p_is_norm

            # 4. Hidden Layer Fallback for 2D+ trainable params
            if (
                p.ndim >= 2
                and not getattr(p, '_is_oft', False)
                and not getattr(p, '_is_lora_A', False)
                and not getattr(p, '_is_lora_B', False)
            ):
                p.is_hidden = is_hidden

            # 5. Weight Decay Ratio Assignment
            if p_is_bias or p_is_norm or p_is_scalar or getattr(p, '_is_dora_scale', False) or getattr(p, 'is_vector', False) or p.ndim <= 1:
                default_wd_ratio = 0.0
            else:
                default_wd_ratio = 1.0

            custom_wd_ratio = getattr(p, '_custom_weight_decay_ratio', None)
            if custom_wd_ratio is None:
                custom_wd_ratio = getattr(p, 'custom_weight_decay_ratio', None)
            if custom_wd_ratio is None:
                custom_wd_ratio = getattr(self, 'weight_decay_ratio', None)

            if custom_wd_ratio is not None:
                p.weight_decay_ratio = float(custom_wd_ratio)
            else:
                p.weight_decay_ratio = default_wd_ratio

    @property
    def multiplier(self):
        return self._multiplier

    @multiplier.setter
    def multiplier(self, value):
        self._multiplier = float(value)
        if hasattr(self, 'multiplier_buf'):
            self.multiplier_buf.fill_(float(value))

    @property
    def org_weight(self):
        return self.org_module[0].weight
    
    def get_org_weight_for_compute(self, device: torch.device):
        """Get org_weight on compute device with async transfer if needed"""
        org_module = self.org_module[0]
        if org_module.weight is None:
            return None
        if is_weight_only_fp8_linear(org_module):
            # Weight-only FP8 weights cannot be materialised in their
            # quantized form; compute against the dequantized weight.
            return dequantize_weight_only_fp8(org_module).to(device)
        weight = org_module.weight
        return transfer_ramtensor_to_device(weight, device)
    
    def get_org_bias_for_compute(self, device: torch.device):
        """Get org_bias on compute device with async transfer if needed"""
        if self.org_module[0].bias is None:
            return None
        bias = self.org_module[0].bias
        return transfer_ramtensor_to_device(bias, device)

    @org_weight.setter
    def org_weight(self, value):
        self.org_module[0].weight.data.copy_(value)

    def _current_weight(self):
        if is_weight_only_fp8_linear(self.org_module[0]):
            return dequantize_weight_only_fp8(self.org_module[0])
        return self.org_module[0].weight.detach()

    def _current_bias(self):
        bias = self.org_module[0].bias
        return None if bias is None else bias.detach()

    def apply_to(self, **kwargs):
        if self.not_supported:
            return

        module = self.org_module[0]
        if not hasattr(module, "_lycoris_original_forward"):
            module._lycoris_original_forward = module.forward

        wrappers = list(getattr(module, "_lycoris_wrappers", []))
        if self in wrappers:
            wrappers.remove(self)

        self.org_forward = module.forward
        wrappers.append(self)

        module._lycoris_wrappers = wrappers
        module.forward = self.forward

    def restore(self):
        if self.not_supported:
            return
        module = self.org_module[0]
        wrappers = list(getattr(module, "_lycoris_wrappers", []))

        if not wrappers:
            module.forward = getattr(
                module, "_lycoris_original_forward", self.org_forward
            )
            return

        try:
            idx = wrappers.index(self)
        except ValueError:
            module.forward = (
                wrappers[-1].forward
                if wrappers
                else getattr(module, "_lycoris_original_forward", self.org_forward)
            )
            return

        wrappers.pop(idx)

        if idx < len(wrappers):
            wrappers[idx].org_forward = self.org_forward

        if wrappers:
            module._lycoris_wrappers = wrappers
            module.forward = wrappers[-1].forward
        else:
            module.forward = getattr(
                module, "_lycoris_original_forward", self.org_forward
            )
            module.__dict__.pop("_lycoris_wrappers", None)
            module.__dict__.pop("_lycoris_original_forward", None)

    def merge_to(self, multiplier=1.0, *, precise: bool = False):
        if self.not_supported:
            return
        if is_weight_only_fp8_linear(self.org_module[0]):
            raise RuntimeError(
                "Merging LyCORIS modules into weight-only FP8 Linear is not supported."
            )

        ctx = self._prepare_merge_context(precise)

        if precise:
            weight_prec, bias_prec = self._compute_precise_result(ctx, multiplier)
            self._apply_precise_weights(ctx, weight_prec, bias_prec)
        else:
            weight, bias = self.get_merged_weight(
                multiplier,
                ctx.weight_param.shape,
                ctx.target_device,
            )
            self._apply_merged_weights(ctx, weight, bias)

        self._restore_merge_context(ctx)

    def onfly_merge(self, multiplier=1.0):
        if self.not_supported:
            return
        if is_weight_only_fp8_linear(self.org_module[0]):
            raise RuntimeError(
                "Merging LyCORIS modules into weight-only FP8 Linear is not supported."
            )
        self_device = next(self.parameters()).device
        self_dtype = next(self.parameters()).dtype
        self.to(self.org_weight)
        self.cached_org_weight = self.org_weight.data.cpu()
        self.cached_org_bias = None
        weight, bias = self.get_merged_weight(
            multiplier, self.org_weight.shape, self.org_weight.device
        )
        self.org_weight = weight
        if bias is not None:
            bias = bias.to(self.org_weight)
            if self.org_module[0].bias is not None:
                self.org_module[0].bias.data.copy_(bias)
                self.cached_org_bias = self.org_module[0].bias.data.cpu()
            else:
                self.org_module[0].bias = nn.Parameter(bias)
        if self.org_module[0].bias is not None:
            self.org_module[0].bias = self.org_module[0].bias.to(self.org_weight)
        self.to(self_device, self_dtype)

    def onfly_restore(self):
        if self.not_supported:
            return
        self.org_weight = self.cached_org_weight.to(self.org_weight)
        if self.cached_org_bias is not None:
            self.org_module[0].bias.data.copy_(self.cached_org_bias.to(self.org_weight))
        del self.cached_org_weight
        del self.cached_org_bias

    def get_diff_weight(self, multiplier=1.0, shape=None, device=None):
        raise NotImplementedError

    def get_merged_weight(self, multiplier=1.0, shape=None, device=None):
        raise NotImplementedError

    @torch.no_grad()
    def apply_max_norm(self, max_norm, device=None):
        return None, None
    
    @torch.no_grad()
    def get_norm(self, device=None):
        return None, None

    def bypass_forward_diff(self, x, scale=1):
        raise NotImplementedError

    def bypass_forward(self, x, scale=1):
        raise NotImplementedError

    def parametrize_forward(self, x: torch.Tensor, *args, **kwargs):
        return self.get_merged_weight(
            multiplier=self.multiplier, shape=x.shape, device=x.device
        )[0].to(x.dtype)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def compile_forward(self, **compile_kwargs):
        """Compile the rebuild-mode and bypass-mode forwards for performance.

        Wraps ``_forward_rebuild_core`` and ``_forward_bypass_core`` with
        ``torch.compile`` so that weight-construction math and the final
        linear/conv op are fused into optimized kernels.  Dispatch logic
        (module_dropout, bypass_mode, GGPO) remains in the uncompiled
        ``forward()`` wrapper.
        """

        if hasattr(self, '_forward_rebuild_core'):
            self._forward_rebuild_core = torch.compile(
                self._forward_rebuild_core, **compile_kwargs
            )
        if hasattr(self, '_forward_bypass_core'):
            self._forward_bypass_core = torch.compile(
                self._forward_bypass_core, **compile_kwargs
            )
    
    @torch.no_grad()
    def initialize_norm_cache(self, org_module_weight: torch.Tensor):
        # Choose a reasonable sample size
        n_rows = org_module_weight.shape[0]
        sample_size = min(2000, n_rows)  # Cap at 2000 samples or use all if smaller

        # Sample random indices across all rows
        indices = torch.randperm(n_rows)[:sample_size]

        # Convert to a supported data type first, then index
        # Use float32 for indexing operations
        weights_float32 = org_module_weight.to(dtype=torch.float32)
        sampled_weights = weights_float32[indices].to(device=self.device)

        # Calculate sampled norms
        sampled_norms = torch.norm(sampled_weights, dim=1, keepdim=True)

        # Store the mean norm as our estimate
        self.org_weight_norm_estimate = sampled_norms.mean()

        # Optional: store standard deviation for confidence intervals
        self.org_weight_norm_std = sampled_norms.std()

        # Free memory
        del sampled_weights, weights_float32

    @torch.no_grad()
    def validate_norm_approximation(self, org_module_weight: torch.Tensor, verbose=True):
        # Calculate the true norm (this will be slow but it's just for validation)
        true_norms = []
        chunk_size = 1024  # Process in chunks to avoid OOM

        for i in range(0, org_module_weight.shape[0], chunk_size):
            end_idx = min(i + chunk_size, org_module_weight.shape[0])
            chunk = org_module_weight[i:end_idx].to(device=self.device, dtype=self.dtype)
            chunk_norms = torch.norm(chunk, dim=1, keepdim=True)
            true_norms.append(chunk_norms.cpu())
            del chunk

        true_norms = torch.cat(true_norms, dim=0)
        true_mean_norm = true_norms.mean().item()

        # Compare with our estimate
        estimated_norm = self.org_weight_norm_estimate.item()

        # Calculate error metrics
        absolute_error = abs(true_mean_norm - estimated_norm)
        relative_error = absolute_error / true_mean_norm * 100  # as percentage

        if verbose:
            logger.info(f"True mean norm: {true_mean_norm:.6f}")
            logger.info(f"Estimated norm: {estimated_norm:.6f}")
            logger.info(f"Absolute error: {absolute_error:.6f}")
            logger.info(f"Relative error: {relative_error:.2f}%")

        return {
            'true_mean_norm': true_mean_norm,
            'estimated_norm': estimated_norm,
            'absolute_error': absolute_error,
            'relative_error': relative_error
        }

    @torch.no_grad()
    def update_norms(self):
        # Early returns for common cases
        if self.ggpo_beta is None or self.ggpo_sigma is None or not self.training:
            return
        
        if not(self.module_type == "linear" or (self.module_type.startswith("conv") and self.ggpo_conv)):
            return
        
        if not (hasattr(self, 'lora_down') and hasattr(self.lora_down, 'weight') and self.lora_down.weight.grad is not None):
            return
        
        if not (hasattr(self, 'lora_up') and hasattr(self.lora_up, 'weight') and self.lora_up.weight.grad is not None):
            return
        
        # Skip update every other step for convolutions to reduce overhead
        if self.module_type != "linear" and hasattr(self, '_skip_counter'):
            self._skip_counter = not self._skip_counter
            if self._skip_counter:
                return
        else:
            self._skip_counter = False
        
        # Fast path for linear layers
        if self.module_type == "linear":
            # Calculate norms directly without forming the full weight matrix
            up_norm = torch.sum(self.lora_up.weight**2)
            down_norm = torch.sum(self.lora_down.weight**2)
            
            # Frobenius norm of the product can be bounded/approximated 
            effect = torch.sqrt(up_norm * down_norm) * self.scale
            
            # Calculate per-output channel distribution (much faster than full matrix mul)
            up_channel_norms = torch.sum(self.lora_up.weight**2, dim=1, keepdim=True)
            total_norm = up_channel_norms.sum()
            
            # Avoid division by zero and normalize
            self.weight_norms = up_channel_norms * (effect / total_norm)
            self.combined_weight_norms = torch.sqrt(
                (self.org_weight_norm_estimate**2) + self.weight_norms**2
            )
            return
        
        if self.module_type.startswith("conv") and self.ggpo_conv:
            # Handle convolution layers - use sampling for efficiency
            # Sample-based estimation for convolution layers
            out_size = self.lora_up.weight.size(0)
            
            # Use a constant estimation factor based on typical CNN properties
            # This avoids expensive reconstruction while capturing essential scaling
            if not hasattr(self, 'conv_norm_estimate'):
                # Cache this value since it's relatively constant
                up = self.lora_up.weight
                down = self.lora_down.weight
                
                # Sample a small subset of weights to estimate norm
                sample_size = min(self.ggpo_conv_weight_sample_size, up.size(0))
                if sample_size < up.size(0):
                    up_indices = torch.randperm(up.size(0))[:sample_size]
                    up_sample = up[up_indices]
                else:
                    up_sample = up
                    
                sample_size = min(self.ggpo_conv_weight_sample_size, down.size(0))
                if sample_size < down.size(0):
                    down_indices = torch.randperm(down.size(0))[:sample_size]
                    down_sample = down[down_indices]
                else:
                    down_sample = down
                
                # Calculate squared Frobenius norms on samples
                up_norm_sq = torch.sum(up_sample**2) * (up.size(0) / up_sample.size(0))
                down_norm_sq = torch.sum(down_sample**2) * (down.size(0) / down_sample.size(0))
                
                # Cache the estimation factor
                self.conv_norm_estimate = torch.sqrt(up_norm_sq * down_norm_sq) * self.scale
            
            # Calculate per-channel output scaling - much faster than full norm calculation
            up_flat = self.lora_up.weight.view(out_size, -1)
            up_channel_norms = torch.sum(up_flat**2, dim=1, keepdim=True)
            channel_sum = up_channel_norms.sum()
            
            # Distribute the precomputed norm across channels
            self.weight_norms = up_channel_norms * (self.conv_norm_estimate / channel_sum)
            self.combined_weight_norms = torch.sqrt(
                (self.org_weight_norm_estimate**2) + self.weight_norms**2
            )


    @torch.no_grad()
    def update_grad_norms(self):
        if not self.training:
            return
        
        if not(self.module_type == "linear" or (self.module_type.startswith("conv") and self.ggpo_conv)):
            return
        
        if not (hasattr(self, 'lora_down') and hasattr(self.lora_down, 'weight') and self.lora_down.weight.grad is not None):
            return
        
        if not (hasattr(self, 'lora_up') and hasattr(self.lora_up, 'weight') and self.lora_down.weight.grad is not None):
            return
            
        # Skip update every other step for convolutions to reduce overhead
        if self.module_type != "linear" and hasattr(self, '_skip_grad_counter'):
            self._skip_grad_counter = not self._skip_grad_counter
            if self._skip_grad_counter:
                return
        else:
            self._skip_grad_counter = False

        # Use direct parameter access instead of named iteration (faster)
        lora_down_grad = self.lora_down.weight.grad
        lora_up_grad = self.lora_up.weight.grad
        
        # Fast path for linear layers
        if self.module_type == "linear":
            # Calculate gradient norms efficiently using matrix properties
            lora_up_weight = self.lora_up.weight
            lora_down_weight = self.lora_down.weight
            
            # For linear layers, directly calculate gradient approximation
            up_down_grad = self.scale * (lora_up_weight @ lora_down_grad)
            up_grad_down = self.scale * (lora_up_grad @ lora_down_weight)
            
            # Sum the gradient components
            approx_grad = up_down_grad + up_grad_down
            
            # Calculate row-wise norms
            self.grad_norms = torch.norm(approx_grad, dim=1, keepdim=True)
        
        if self.module_type.startswith("conv") and self.ggpo_conv:
            # Use a fast approximation for convolution gradients
            out_size = self.lora_up.weight.size(0)
            
            # Calculate gradient magnitude using norm products (faster than reconstruction)
            up_grad_norm = torch.norm(lora_up_grad.view(-1))
            down_weight_norm = torch.norm(self.lora_down.weight.view(-1))
            up_weight_norm = torch.norm(self.lora_up.weight.view(-1))
            down_grad_norm = torch.norm(lora_down_grad.view(-1))
            
            # Approximation of the combined gradient magnitude
            grad_magnitude = self.scale * (up_grad_norm * down_weight_norm + up_weight_norm * down_grad_norm)
            
            # Distribute gradient magnitude across output channels
            # This avoids expensive per-channel calculations while capturing key behavior
            up_channel_magnitudes = torch.norm(self.lora_up.weight.view(out_size, -1), dim=1, keepdim=True)
            magnitude_sum = up_channel_magnitudes.sum()
            
            # Distribute based on weight magnitudes (channels with larger weights get larger gradients)
            self.grad_norms = up_channel_magnitudes * (grad_magnitude / magnitude_sum)

    @staticmethod
    def _compute_svd_segment(org_weight_2d, lora_dim, segment):
        """Compute SVD segment from a 2D weight matrix.

        Slices the SVD spectrum of *org_weight_2d* into a contiguous segment
        of width *lora_dim* determined by *segment*.

        Args:
            org_weight_2d: Weight matrix of shape ``(out, in)``.
            lora_dim: Number of singular values (rank) to select.
            segment: One of ``"top"``, ``"middle"``, or ``"bottom"``.

        Returns:
            ``(Vr, Sr, Uhr)`` tuple, or ``None`` if the matrix has fewer
            singular values than *lora_dim*.
        """
        V, S, Uh = torch.linalg.svd(org_weight_2d.float(), full_matrices=False)
        h = len(S)
        if h < lora_dim:
            return None
        if segment == "top":
            start = 0
        elif segment == "middle":
            start = (h - lora_dim) // 2
        elif segment == "bottom":
            start = h - lora_dim
        else:
            raise ValueError(
                f"Unknown svd_segment '{segment}'. Use 'top', 'middle', or 'bottom'."
            )
        Vr = V[:, start : start + lora_dim]
        Sr = S[start : start + lora_dim]
        Uhr = Uh[start : start + lora_dim]
        return Vr, Sr, Uhr

    @staticmethod
    def _compute_svd_pissa(org_weight_2d, lora_dim, niter=0, n_oversamples=10):
        """Compute PiSSA (top-r) SVD decomposition with optional fast randomized SVD.

        Returns the principal components (top-*lora_dim*) of the weight matrix.
        When *niter* = 0, uses exact SVD via :func:`torch.linalg.svd`.
        When *niter* > 0, uses fast randomized SVD (Halko et al., 2011) with
        *niter* power iterations.

        Args:
            org_weight_2d: Weight matrix of shape ``(out, in)``.
            lora_dim: Number of principal singular values/vectors to extract.
            niter: Number of power iterations for randomized SVD (0 = exact).
            n_oversamples: Extra random samples for randomized SVD stability.

        Returns:
            ``(Vr, Sr, Uhr)`` tuple where:
                - *Vr*: ``(out, lora_dim)`` top-r left singular vectors (``U_{[:,:r]}``)
                - *Sr*: ``(lora_dim,)`` top-r singular values, descending
                - *Uhr*: ``(lora_dim, in)`` top-r right singular vectors transposed (``V^T_{[:r,:]}``)
            or ``None`` if the matrix has fewer singular values than *lora_dim*.

            This naming convention matches the existing
            :meth:`_compute_svd_segment` for consistency within the codebase
            (where ``V, S, Uh = torch.linalg.svd(...)`` maps ``V`` to ``U``
            and ``Uh`` to ``V^T``).
        """
        m, n = org_weight_2d.shape
        h = min(m, n)
        if h < lora_dim:
            return None

        if niter <= 0:
            # Exact SVD path
            V, S, Uh = torch.linalg.svd(org_weight_2d.float(), full_matrices=False)
            Vr = V[:, :lora_dim]
            Sr = S[:lora_dim]
            Uhr = Uh[:lora_dim]
            return Vr, Sr, Uhr

        # Fast randomized SVD path (Halko et al., 2011)
        W_float = org_weight_2d.float()
        r_oversampled = min(lora_dim + n_oversamples, h)

        # Step 1: Random projection
        Omega = torch.randn((n, r_oversampled), dtype=torch.float32, device=W_float.device)
        Y = W_float @ Omega

        # Step 2: Power iterations to improve accuracy
        for _ in range(niter):
            Y = W_float @ (W_float.T @ Y)

        # Step 3: QR decomposition of Y
        Q, _ = torch.linalg.qr(Y)

        # Step 4: Project weight to subspace
        B_proj = Q.T @ W_float  # (r_oversampled, n)

        # Step 5: SVD of the small projected matrix
        Ub, S, Vh = torch.linalg.svd(B_proj, full_matrices=False)

        # Step 6: Transform back
        V = Q @ Ub  # (m, r_oversampled)

        # Step 7: Take top lora_dim components
        Vr = V[:, :lora_dim]
        Sr = S[:lora_dim]
        Uhr = Vh[:lora_dim, :]  # Vh is already (r_oversampled, n)

        return Vr, Sr, Uhr

    @staticmethod
    def _get_weight_2d(org_module):
        """Return the original module weight reshaped to 2-D ``(out, in)``."""
        w = org_module.weight.data.clone()
        if w.dim() > 2:
            w = w.reshape(w.shape[0], -1)
        return w

    @torch.no_grad()
    def init_ggpo(self):
        if self.ggpo_beta is not None and self.ggpo_sigma is not None:
            self.combined_weight_norms = None
            self.grad_norms = None
            self.weight_norms = None
            self.perturbation_norm_factor = 1.0 / math.sqrt(self.org_module[0].weight.shape[0])
            self.initialize_norm_cache(self.org_module[0].weight)
            self.org_module_shape: tuple[int] = self.org_module[0].weight.shape

    @torch.no_grad()
    def accumulate_grad(self):
        for param in self.parameters():
            if param.grad is not None:
                grad = param.grad.detach().flatten()
                self.grad_count += grad.numel()

                # Update running sums
                if self.sum_grads is None:
                    self.sum_grads = grad.sum()
                    self.sum_squared_grads = (grad**2).sum()
                else:
                    self.sum_grads += grad.sum()
                    self.sum_squared_grads += (grad**2).sum()

    def ggpo_pertubation(self, x):
        return None

    @torch.no_grad()
    def inject_weight_noise(self, lr: float = 1e-4, effective_batch_size: int = 1, param_lr_map: dict = None) -> float:
        """Add Gaussian noise directly to trainable parameter values.

        Inspired by ai-toolkit-perceptual's Weight Noising. Adds noise
        after the optimizer step so Adam's loss-minimization corrects
        the drift, causing weights to wander around the optimizer
        trajectory inside a bounded ball.

        Modes:
          - ``'absolute'``: σ fixed at ``weight_noise_sigma``.
          - ``'relative'``: σ = ``weight_noise_sigma`` × per-param
            weight RMS (default). Zero-init params (e.g. LoRA-up)
            get zero noise until they learn something.

        When ``weight_noise_dynamic_sigma`` is True, the computed sigma
        is further scaled by ``lr / √effective_batch_size``, making
        noise magnitude proportional to learning rate and inversely
        proportional to √batch size (consistent with SGLD theory).

        Args:
            lr: Current learning rate (used only when dynamic_sigma=True).
            effective_batch_size: Total effective batch size
                (batch_size × gradient_accumulation_steps).

        Returns:
            Sum of squared noise values (for Frobenius norm computation
            at the network level).
        """
        if self.weight_noise_sigma is None or self.weight_noise_sigma <= 0:
            return 0.0

        # Dynamic sigma scaling: multiply base sigma by lr / √eff_bs
        noise_sq = 0.0
        for p in self.parameters():
            if not p.requires_grad:
                continue
            w = p.data

            # Per-parameter LR: use param_lr_map if available, else fallback to global lr
            p_lr = lr
            if param_lr_map is not None and id(p) in param_lr_map:
                p_lr = param_lr_map[id(p)]

            dyn_scale = 1.0
            if getattr(self, 'weight_noise_dynamic_sigma', False):
                dyn_scale = p_lr / max(math.sqrt(effective_batch_size), 1e-30)

            if self.weight_noise_mode == 'absolute':
                sigma = self.weight_noise_sigma
            elif self.weight_noise_mode == 'relative':
                rms = float(w.detach().pow(2).mean().clamp_min(1e-30).sqrt())
                sigma = self.weight_noise_sigma * rms
            else:
                continue
            sigma = sigma * dyn_scale
            if sigma <= 0:
                continue
            noise = torch.randn_like(w) * sigma
            noise_sq += float(noise.pow(2).sum())
            w.add_(noise)
        return noise_sq

    def _prepare_merge_context(self, precise: bool) -> _MergeContext:
        module = self.org_module[0]
        weight_param = module.weight
        bias_param = module.bias

        params = tuple(self.parameters())
        first_param = params[0] if params else None
        param_device = first_param.device if first_param is not None else None
        param_dtype = first_param.dtype if first_param is not None else None

        target_device = weight_param.device
        target_dtype = weight_param.dtype
        compute_dtype = torch.float64 if precise else target_dtype

        if first_param is not None:
            self.to(device=target_device, dtype=compute_dtype)
        else:
            self.to(target_device)
            if precise:
                self.to(dtype=compute_dtype)

        if precise:
            self._ensure_precise_snapshot(module, weight_param, bias_param)
            self._load_precise_snapshot(
                module,
                weight_param,
                bias_param,
                target_device,
                compute_dtype,
            )

        return _MergeContext(
            precise=precise,
            target_device=target_device,
            target_dtype=target_dtype,
            compute_dtype=compute_dtype,
            param_device=param_device,
            param_dtype=param_dtype,
            module=module,
            weight_param=weight_param,
            bias_param=bias_param,
        )

    def _apply_merged_weights(
        self,
        ctx: _MergeContext,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> None:
        merged_weight = weight.to(ctx.target_dtype)
        ctx.weight_param.data.copy_(merged_weight)

        if bias is not None:
            merged_bias = bias.to(ctx.target_dtype)
            if ctx.bias_param is not None:
                ctx.bias_param.data.copy_(merged_bias)
            else:
                ctx.module.bias = nn.Parameter(merged_bias)
        elif ctx.bias_param is None:
            ctx.module.bias = None

        if ctx.precise:
            ctx.module._lycoris_precise_weight_current = weight.to(torch.float64).cpu()
            if ctx.bias_param is not None:
                if bias is not None:
                    ctx.module._lycoris_precise_bias_current = bias.to(
                        torch.float64
                    ).cpu()
                else:
                    ctx.module._lycoris_precise_bias_current = (
                        ctx.module._lycoris_precise_bias_base
                    )

    def _restore_merge_context(self, ctx: _MergeContext) -> None:
        if ctx.param_device is not None and ctx.param_dtype is not None:
            self.to(device=ctx.param_device, dtype=ctx.param_dtype)
        elif ctx.param_device is not None:
            self.to(ctx.param_device)
        elif ctx.param_dtype is not None:
            self.to(dtype=ctx.param_dtype)

    def _compute_precise_result(
        self, ctx: _MergeContext, multiplier: float
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        base_weight = ctx.module._lycoris_precise_weight_current
        diff_weight, diff_bias = self.get_diff_weight(
            multiplier=1.0, device=ctx.target_device
        )
        diff_weight_prec = diff_weight.to(torch.float64).cpu()
        new_weight = base_weight + diff_weight_prec * multiplier

        new_bias = None
        if diff_bias is not None:
            diff_bias_prec = diff_bias.to(torch.float64).cpu()
            base_bias = ctx.module._lycoris_precise_bias_current
            if base_bias is None:
                base_bias = torch.zeros_like(diff_bias_prec)
            new_bias = base_bias + diff_bias_prec * multiplier
        else:
            new_bias = ctx.module._lycoris_precise_bias_current

        ctx.module._lycoris_precise_weight_current = new_weight.clone()
        if diff_bias is not None:
            ctx.module._lycoris_precise_bias_current = (
                new_bias.clone() if new_bias is not None else None
            )

        return new_weight, new_bias

    def _apply_precise_weights(
        self,
        ctx: _MergeContext,
        weight_prec: torch.Tensor,
        bias_prec: torch.Tensor | None,
    ) -> None:
        ctx.weight_param.data.copy_(weight_prec.to(ctx.target_device, ctx.target_dtype))

        if bias_prec is not None:
            if ctx.bias_param is not None:
                ctx.bias_param.data.copy_(
                    bias_prec.to(ctx.target_device, ctx.target_dtype)
                )
            else:
                ctx.module.bias = nn.Parameter(
                    bias_prec.to(ctx.target_device, ctx.target_dtype)
                )
        elif ctx.bias_param is None:
            ctx.module.bias = None

    @staticmethod
    def _ensure_precise_snapshot(
        module: nn.Module,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> None:
        if not hasattr(module, "_lycoris_precise_weight_base"):
            base = weight.detach().cpu().double()
            module._lycoris_precise_weight_base = base
            module._lycoris_precise_weight_current = base.clone()
        if not hasattr(module, "_lycoris_precise_weight_current"):
            module._lycoris_precise_weight_current = (
                module._lycoris_precise_weight_base.clone()
            )

        if not hasattr(module, "_lycoris_precise_bias_base"):
            if bias is not None:
                base_bias = bias.detach().cpu().double()
            else:
                base_bias = None
            module._lycoris_precise_bias_base = base_bias
            module._lycoris_precise_bias_current = (
                base_bias.clone() if base_bias is not None else None
            )
        if not hasattr(module, "_lycoris_precise_bias_current"):
            module._lycoris_precise_bias_current = (
                module._lycoris_precise_bias_base.clone()
                if module._lycoris_precise_bias_base is not None
                else None
            )

    @staticmethod
    def _load_precise_snapshot(
        module: nn.Module,
        weight_param: torch.Tensor,
        bias_param: torch.Tensor | None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        weight_param.data.copy_(
            module._lycoris_precise_weight_current.to(device=device, dtype=dtype)
        )
        if bias_param is not None:
            bias_snapshot = module._lycoris_precise_bias_current
            if bias_snapshot is None and module._lycoris_precise_bias_base is not None:
                bias_snapshot = module._lycoris_precise_bias_base
                module._lycoris_precise_bias_current = bias_snapshot
            if bias_snapshot is not None:
                bias_param.data.copy_(bias_snapshot.to(device=device, dtype=dtype))
