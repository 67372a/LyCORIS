import unittest
from itertools import product
from parameterized import parameterized

import torch
import torch.nn as nn

from lycoris.modules import (
    LycorisBaseModule,
    AbbaModule,
    LoConModule,
    LohaModule,
    LokrModule,
    FullModule,
    DiagOFTModule,
    ButterflyOFTModule,
    GLoRAModule,
    DyLoraModule,
    IA3Module,
)


modules: list[LycorisBaseModule] = [
    AbbaModule,
    LoConModule,
    LohaModule,
    LokrModule,
    FullModule,
    DiagOFTModule,
    ButterflyOFTModule,
    GLoRAModule,
    DyLoraModule,
    IA3Module,
]
base_module_and_input = [
    lambda dim: (nn.Linear(dim, dim), torch.randn(1, dim)),
    lambda dim: (nn.Conv1d(dim, dim, 3, 1, 1), torch.randn(1, dim, 16)),
    lambda dim: (nn.Conv2d(dim, dim, (3, 3), 1, 1), torch.randn(1, dim, 16, 16)),
    lambda dim: (nn.Conv3d(dim, dim, (3, 3, 3), 1, 1), torch.randn(1, dim, 16, 16, 16)),
]
device_and_dtype = [
    (torch.device("cpu"), torch.float32),
]
weight_decompose = [False, True]
use_tucker = [False, True]
use_scalar = [False, True]

if torch.cuda.is_available():
    device_and_dtype.append((torch.device("cuda"), torch.float32))
    device_and_dtype.append((torch.device("cuda"), torch.float16))
    device_and_dtype.append((torch.device("cuda"), torch.bfloat16))

if torch.backends.mps.is_available():
    device_and_dtype.append((torch.device("mps"), torch.float32))


patch_forward_param_list = list(
    product(
        modules,
        base_module_and_input,
        device_and_dtype,
        weight_decompose,
        use_tucker,
        use_scalar,
    )
)


class LycorisModuleTests(unittest.TestCase):
    @parameterized.expand(patch_forward_param_list)
    def test_lycoris_modules(self, module, base, device_dtype, wd, tucker, scalar):
        base, test_input = base(16)
        device, dtype = device_dtype
        print(
            f"{module.__name__: <18}",
            f"{base.__class__.__name__: <7}",
            f"device={str(device): <5}",
            f"dtype={str(dtype): <15}",
            f"wd={str(wd): <6}",
            f"tucker={str(tucker): <6}",
            f"scalar={str(scalar): <6}",
            sep="|| ",
        )
        base = base.to(device, dtype)
        test_input = test_input.to(device, dtype)
        net: LycorisBaseModule = module(
            "test",
            base,
            multiplier=1,
            lora_dim=4,
            alpha=1,
            weight_decompose=wd,
            use_tucker=tucker,
            use_scalar=scalar,
        ).to(device, dtype)
        net.apply_to()

        with torch.autocast("cuda", dtype=dtype):
            test_output = base(test_input)
        torch.sum(test_output).backward()
        net.apply_max_norm(1.0)
        state_dict = net.state_dict()
        net.load_state_dict(state_dict)
        net.restore()
        net.merge_to()

        # attr access test
        net.org_weight

    @parameterized.expand(patch_forward_param_list)
    def test_lycoris_modules_bypass_mode(
        self, module, base, device_dtype, wd, tucker, scalar
    ):
        base, test_input = base(16)
        if module == FullModule:
            # Full module not support bypass forward
            return
        device, dtype = device_dtype
        print(
            f"{module.__name__: <18}",
            f"{base.__class__.__name__: <7}",
            f"device={str(device): <5}",
            f"dtype={str(dtype): <15}",
            f"wd={str(wd): <6}",
            f"tucker={str(tucker): <6}",
            f"scalar={str(scalar): <6}",
            sep="|| ",
        )
        base = base.to(device, dtype)
        test_input = test_input.to(device, dtype)
        net: LycorisBaseModule = module(
            "test",
            base,
            multiplier=1,
            lora_dim=4,
            alpha=1,
            weight_decompose=wd,
            use_tucker=tucker,
            use_scalar=scalar,
            bypass_mode=True,
        ).to(device, dtype)
        net.apply_to()

        with torch.autocast("cuda", dtype=dtype):
            test_output = base(test_input)
        torch.sum(test_output).backward()
        state_dict = net.state_dict()
        net.load_state_dict(state_dict)

    @parameterized.expand(patch_forward_param_list)
    def test_lycoris_modules_parametrize(
        self, module, base, device_dtype, wd, tucker, scalar
    ):
        base, test_input = base(16)
        if module == FullModule:
            # Full module not support bypass forward
            return
        device, dtype = device_dtype
        print(
            f"{module.__name__: <18}",
            f"{base.__class__.__name__: <7}",
            f"device={str(device): <5}",
            f"dtype={str(dtype): <15}",
            f"wd={str(wd): <6}",
            f"tucker={str(tucker): <6}",
            f"scalar={str(scalar): <6}",
            sep="|| ",
        )
        base = base.to(device, dtype)
        test_input = test_input.to(device, dtype)
        net = module.parametrize(
            base,
            "weight",
            1,
            4,
            1,
            weight_decompose=wd,
            use_tucker=tucker,
            use_scalar=scalar,
        ).to(device, dtype)

        with torch.autocast("cuda", dtype=dtype):
            test_output = base(test_input)
        torch.sum(test_output).backward()
        state_dict = net.state_dict()
        net.load_state_dict(state_dict)

    # --- Orthogonal init / runtime orthogonalization tests ---
    _orthogonal_modules = [LoConModule, LokrModule, GLoRAModule, LohaModule]

    def test_orthogonal_init_forces_scalar(self):
        """orthogonal_init=True should force use_scalar=True."""
        base = nn.Linear(16, 16)
        for module_cls in self._orthogonal_modules:
            net = module_cls(
                "test",
                base,
                multiplier=1,
                lora_dim=4,
                alpha=1,
                orthogonal_init=True,
                use_scalar=False,
            )
            self.assertIsInstance(
                net.scalar,
                nn.Parameter,
                f"{module_cls.__name__}: orthogonal_init should force scalar to be nn.Parameter",
            )
            net.restore()

    def test_orthogonalize_forces_init_and_scalar(self):
        """orthogonalize=True should force orthogonal_init=True and use_scalar=True."""
        base = nn.Linear(16, 16)
        for module_cls in self._orthogonal_modules:
            net = module_cls(
                "test",
                base,
                multiplier=1,
                lora_dim=4,
                alpha=1,
                orthogonalize=True,
                use_scalar=False,
            )
            self.assertTrue(
                net.use_orthogonal_init,
                f"{module_cls.__name__}: orthogonalize should force orthogonal_init",
            )
            self.assertTrue(
                isinstance(net.scalar, nn.Parameter),
                f"{module_cls.__name__}: orthogonalize should force scalar via orthogonal_init",
            )
            net.restore()

    def test_orthogonal_init_produces_orthogonal_weights(self):
        """Weights initialized with orthogonal_init should have near-orthogonal columns/rows."""
        base = nn.Linear(16, 16)
        for module_cls in self._orthogonal_modules:
            net = module_cls(
                "test",
                base,
                multiplier=1,
                lora_dim=4,
                alpha=1,
                orthogonal_init=True,
            )
            # Check orthogonality of the primary weight matrix for each module type
            weight_to_check = None
            if hasattr(net, "lora_down"):
                weight_to_check = net.lora_down.weight.data.float()
            elif hasattr(net, "hada_w1_a"):
                weight_to_check = net.hada_w1_a.weight.data.float() if isinstance(net.hada_w1_a, nn.Module) else net.hada_w1_a.data.float()
            if weight_to_check is not None:
                w2d = weight_to_check.reshape(weight_to_check.shape[0], -1)
                if w2d.shape[0] <= w2d.shape[1]:
                    gram = w2d @ w2d.T
                else:
                    gram = w2d.T @ w2d
                off_diag = gram - torch.diag(gram.diag())
                diag_scale = gram.diag().abs().mean().clamp(min=1e-8)
                self.assertTrue(
                    off_diag.abs().max() < diag_scale * 0.5,
                    f"{module_cls.__name__}: primary weights not orthogonal after orthogonal_init",
                )
            net.restore()

    def test_orthogonal_init_and_orthogonalize_independent(self):
        """orthogonal_init and orthogonalize should work together; orthogonalize forces orthogonal_init."""
        base = nn.Linear(16, 16)
        for module_cls in self._orthogonal_modules:
            # orthogonal_init only (no runtime orthogonalization)
            net1 = module_cls(
                "test", base, multiplier=1, lora_dim=4, alpha=1,
                orthogonal_init=True, orthogonalize=False,
            )
            self.assertTrue(net1.use_orthogonal_init)
            self.assertFalse(net1.use_orthogonal_weights)
            net1.restore()

            # orthogonalize forces orthogonal_init (and thus use_scalar)
            net2 = module_cls(
                "test", base, multiplier=1, lora_dim=4, alpha=1,
                orthogonal_init=False, orthogonalize=True,
            )
            self.assertTrue(net2.use_orthogonal_init)
            self.assertTrue(net2.use_orthogonal_weights)
            net2.restore()

            # both explicitly set
            net3 = module_cls(
                "test", base, multiplier=1, lora_dim=4, alpha=1,
                orthogonal_init=True, orthogonalize=True,
            )
            self.assertTrue(net3.use_orthogonal_init)
            self.assertTrue(net3.use_orthogonal_weights)
            net3.restore()

    def test_orthogonal_forward_pass(self):
        """Forward/backward should work with each orthogonal configuration."""
        base = nn.Linear(16, 16)
        test_input = torch.randn(1, 16)
        configs = [
            {"orthogonal_init": True, "orthogonalize": False},
            {"orthogonal_init": False, "orthogonalize": True},
            {"orthogonal_init": True, "orthogonalize": True},
        ]
        for module_cls in self._orthogonal_modules:
            for cfg in configs:
                base_copy = nn.Linear(16, 16)
                base_copy.weight.data.copy_(base.weight.data)
                if base.bias is not None:
                    base_copy.bias.data.copy_(base.bias.data)
                net = module_cls(
                    "test", base_copy, multiplier=1, lora_dim=4, alpha=1, **cfg,
                )
                net.apply_to()
                output = base_copy(test_input)
                output.sum().backward()
                # Verify gradient flows through primary weight parameter
                if hasattr(net, "lora_down"):
                    self.assertIsNotNone(net.lora_down.weight.grad)
                elif hasattr(net, "hada_w1_a"):
                    self.assertIsNotNone(net.hada_w1_a.grad)
                net.restore()
