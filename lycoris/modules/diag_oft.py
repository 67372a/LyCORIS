from functools import cache

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from .base import LycorisBaseModule
from ..functional import factorization
from ..logging import logger
from ..utils.bnb import LinearNF4


@cache
def log_oft_factorize(dim, factor, num, bdim):
    logger.info(
        f"Use OFT(block num: {num}, block dim: {bdim})"
        f" (equivalent to lora_dim={num}) "
        f"for {dim=} and lora_dim={factor=}"
    )


class DiagOFTModule(LycorisBaseModule):
    support_module = {
        "linear",
        "conv1d",
        "conv2d",
        "conv3d",
    }

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
        rank_dropout_scale=False,
        constrain=0,
        rescaled=False,
        bypass_mode=False,
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
            raise ValueError(f"{self.module_type} is not supported in Diag-OFT algo.")

        out_dim = self.dim
        self.block_size, self.block_num = factorization(out_dim, lora_dim)
        # block_num > block_size
        self.rescaled = rescaled
        self.constrain = constrain * out_dim
        self.register_buffer("alpha", torch.tensor(constrain))
        self.oft_blocks = nn.Parameter(
            torch.zeros(self.block_num, self.block_size, self.block_size)
        )
        if rescaled:
            self.rescale = nn.Parameter(
                torch.ones(out_dim, *(1 for _ in range(org_module.weight.dim() - 1)))
            )

        log_oft_factorize(
            dim=out_dim,
            factor=lora_dim,
            num=self.block_num,
            bdim=self.block_size,
        )

    @property
    def I(self):
        return torch.eye(self.block_size, device=self.device)

    def get_r(self):
        I = self.I
        # for Q = -Q^T
        q = self.oft_blocks - self.oft_blocks.transpose(1, 2)
        normed_q = q
        if self.constrain > 0:
            q_norm = torch.norm(q) + 1e-8
            if q_norm > self.constrain:
                normed_q = q * self.constrain / q_norm
        # use float() to prevent unsupported type
        r = (I + normed_q) @ (I - normed_q).float().inverse()
        return r

    def make_weight(self, scale=1, device=None, diff=False):
        r = self.get_r()
        org_weight = self.org_weight.to(device, dtype=r.dtype)
        org_weight = rearrange(
            org_weight,
            "(k n) ... -> k n ...",
            k=self.block_num,
            n=self.block_size,
        )
        # Init R=0, so add I on it to ensure the output of step0 is original model output
        weight = torch.einsum(
            "k n m, k n ... -> k m ...",
            r * scale - scale * self.I + (self.I if diff else 0),
            org_weight,
        )
        weight = rearrange(weight, "k m ... -> (k m) ...")
        if self.rescaled:
            weight = self.rescale * weight
        return weight

    def get_diff_weight(self, multiplier=1, shape=None, device=None):
        diff = self.make_weight(scale=multiplier, device=device, diff=True)
        if shape is not None:
            diff = diff.view(shape)
        return diff

    def get_merged_weight(self, multiplier=1, shape=None, device=None):
        diff = self.make_weight(scale=multiplier, device=device)
        if shape is not None:
            diff = diff.view(shape)
        return diff, None

    @torch.no_grad()
    def apply_max_norm(self, max_norm, device=None):
        orig_norm = self.oft_blocks.to(device).norm()
        norm = torch.clamp(orig_norm, max_norm / 2)
        desired = torch.clamp(norm, max=max_norm)
        ratio = desired / norm

        scaled = norm != desired
        if scaled:
            self.oft_blocks *= ratio

        return scaled, orig_norm * ratio

    def _bypass_forward(self, x, scale=1, diff=False):
        r = self.get_r()
        org_out = self.org_forward(x)
        if self.op == F.conv2d:
            org_out = org_out.transpose(1, -1)
        org_out = rearrange(
            org_out,
            "... (k n) ->... k n",
            k=self.block_num,
            n=self.block_size,
        )
        oft_out = torch.einsum(
            "k n m, ... k n -> ... k m",
            r * scale - scale * self.I + (self.I if diff else 0),
            org_out,
        )
        out = rearrange(oft_out, "... k m -> ... (k m)")
        if self.op == F.conv2d:
            out = out.transpose(1, -1)
        return out

    def bypass_forward_diff(self, x, scale=1):
        return self._bypass_forward(x, scale, diff=True)

    def bypass_forward(self, x, scale=1):
        return self._bypass_forward(x, scale, diff=False)

    def forward(self, x: torch.Tensor, *args, **kwargs):
        if self.module_dropout and self.training:
            if torch.rand(1) < self.module_dropout:
                return self.org_forward(x)
        scale = self.multiplier

        if self.bypass_mode:
            return self.bypass_forward(x, scale)
        else:
            w = self.make_weight(scale, x.device)
            kw_dict = self.kw_dict | {"weight": w, "bias": self.org_module[0].bias}
            return self.op(x, **kw_dict)


if __name__ == "__main__":
    base = nn.Linear(128, 128).cuda()
    doft = DiagOFTModule("test", base, 1, 4, 1, weight_decompose=True, factor=8).cuda()
    print(doft)
    test_input = torch.randn(1, 77, 128).cuda()
    test_output = doft(test_input)
    torch.sum(test_output).backward()
    print(test_output.shape)

    base_4bit = LinearNF4(128, 128)
    base_4bit.load_state_dict(base.state_dict())
    base_4bit.cuda()
    qdoft = DiagOFTModule("test", base_4bit, 1, 4, 1, weight_decompose=False).cuda()
    print(qdoft)
    test_output = qdoft(test_input)
    torch.sum(test_output).backward()
    print(test_output.shape)

    base = nn.Conv2d(128, 128, 3, 1, 1)
    doft = DiagOFTModule("test", base, 1, 4, 1, weight_decompose=True, use_tucker=True)
    print(doft)
    test_input = torch.randn(1, 128, 16, 16)
    test_output = doft(test_input)
    torch.sum(test_output).backward()
    print(test_output.shape)
