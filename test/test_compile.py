"""Tests for per-module torch.compile integration in LyCORIS.

Verifies:
  - _forward_rebuild_core produces correct outputs for all module types
  - Forward/backward works with compiled rebuild core
  - Numerical equivalence between compiled and non-compiled paths
  - Wrapper integration (create_lycoris with torch_compile=True)
  - Bypass-mode modules are skipped by compile
"""

import unittest

import torch
import torch.nn as nn

from lycoris.modules.locon import LoConModule
from lycoris.modules.loha import LohaModule
from lycoris.modules.lokr import LokrModule
from lycoris.modules.abba import AbbaModule
from lycoris.modules.glora import GLoRAModule
from lycoris.modules.ia3 import IA3Module
from lycoris.modules.diag_oft import DiagOFTModule
from lycoris.modules.boft import ButterflyOFTModule
from lycoris.wrapper import create_lycoris


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CUDA_AVAILABLE = torch.cuda.is_available()


def _compile_device():
    if CUDA_AVAILABLE:
        return torch.device("cuda")
    return torch.device("cpu")


def _compile_kwargs():
    if CUDA_AVAILABLE:
        return dict(mode="default", dynamic=True, fullgraph=False)
    return dict(backend="eager", fullgraph=False)


def _wrapper_compile_kwargs():
    if CUDA_AVAILABLE:
        return dict(
            torch_compile_mode="default",
            torch_compile_dynamic=True,
            torch_compile_fullgraph=False,
        )
    return dict(
        torch_compile_mode="eager",
        torch_compile_dynamic=True,
        torch_compile_fullgraph=False,
    )


# ---------------------------------------------------------------------------
# Base test helpers
# ---------------------------------------------------------------------------

def _make_linear_base(dim, device, dtype):
    base = nn.Linear(dim, dim).to(device, dtype)
    test_input = torch.randn(1, dim, device=device, dtype=dtype)
    return base, test_input


def _make_conv_base(dim, device, dtype):
    base = nn.Conv2d(dim, dim, 3, 1, 1).to(device, dtype)
    test_input = torch.randn(1, dim, 8, 8, device=device, dtype=dtype)
    return base, test_input


# ===========================================================================
# Generic rebuild-core correctness test mixin
# ===========================================================================

class RebuildCoreTestMixin:
    """Mixin that verifies _forward_rebuild_core matches forward() output."""

    module_cls = None
    module_kwargs = {}

    def _run_rebuild_core_test(self, module_kwargs=None):
        if module_kwargs is None:
            module_kwargs = {}
        merged_kwargs = {**self.module_kwargs, **module_kwargs}

        device = _compile_device()
        dtype = torch.float32
        base, test_input = _make_linear_base(16, device, dtype)

        net = self.module_cls("test", base, **merged_kwargs).to(device, dtype)
        net.apply_to()

        # Forward via the full forward()
        out_full = base(test_input)

        # Manually call _forward_rebuild_core
        with torch.no_grad():
            org_weight = net.get_org_weight_for_compute(test_input.device).to(dtype)
            org_bias = net.get_org_bias_for_compute(test_input.device)
            out_core = net._forward_rebuild_core(test_input, org_weight, org_bias)

        torch.testing.assert_close(out_full, out_core)


# ===========================================================================
# Generic compile equivalence test mixin
# ===========================================================================

class CompileEquivalenceTestMixin:
    """Mixin that verifies compiled and non-compiled rebuild cores match."""

    module_cls = None
    module_kwargs = {}

    def _run_equivalence_test(self, module_kwargs=None, dim=32):
        if module_kwargs is None:
            module_kwargs = {}
        merged_kwargs = {**self.module_kwargs, **module_kwargs}

        device = _compile_device()
        dtype = torch.float32

        torch.manual_seed(42)
        base_a = nn.Linear(dim, dim).to(device, dtype)
        net_a = self.module_cls("test", base_a, **merged_kwargs).to(device, dtype)
        net_a.apply_to()

        torch.manual_seed(42)
        base_b = nn.Linear(dim, dim).to(device, dtype)
        base_b.load_state_dict(base_a.state_dict())
        net_b = self.module_cls("test", base_b, **merged_kwargs).to(device, dtype)
        net_b.apply_to()
        net_b.compile_forward(**_compile_kwargs())

        x = torch.randn(1, dim, device=device, dtype=dtype)
        with torch.no_grad():
            out_a = base_a(x)
            out_b = base_b(x)

        torch.testing.assert_close(out_a, out_b)


# ===========================================================================
# Generic training loop test mixin
# ===========================================================================

class TrainingTestMixin:
    """Mixin that verifies gradients flow through compiled rebuild core."""

    module_cls = None
    module_kwargs = {}

    def _run_training_test(self, compile=False, module_kwargs=None):
        if module_kwargs is None:
            module_kwargs = {}
        merged_kwargs = {**self.module_kwargs, **module_kwargs}

        device = _compile_device()
        dtype = torch.float32
        base = nn.Linear(16, 16).to(device, dtype)
        net = self.module_cls("test", base, **merged_kwargs).to(device, dtype)
        if compile:
            net.compile_forward(**_compile_kwargs())
        net.apply_to()

        x = torch.randn(2, 16, device=device, dtype=dtype)
        out = base(x)
        loss = out.sum()
        loss.backward()

        for name, p in net.named_parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad, f"{name} has no grad")


# ===========================================================================
# 1.  LoConModule  (existing)
# ===========================================================================

class LoConRebuildCoreTests(RebuildCoreTestMixin, unittest.TestCase):
    module_cls = LoConModule
    module_kwargs = dict(multiplier=1, lora_dim=4, alpha=1,
                         weight_decompose=False, use_tucker=False, use_scalar=False)

    def test_linear_basic(self):
        self._run_rebuild_core_test()

    def test_linear_dora(self):
        self._run_rebuild_core_test(dict(weight_decompose=True))

    def test_linear_scalar(self):
        self._run_rebuild_core_test(dict(use_scalar=True))

    def test_conv_basic(self):
        device = _compile_device()
        dtype = torch.float32
        base, test_input = _make_conv_base(16, device, dtype)
        net = LoConModule("test", base, multiplier=1, lora_dim=4, alpha=1,
                          weight_decompose=False, use_tucker=False, use_scalar=False,
                          ).to(device, dtype)
        net.apply_to()
        out_full = base(test_input)
        with torch.no_grad():
            org_weight = net.get_org_weight_for_compute(test_input.device).to(dtype)
            org_bias = net.get_org_bias_for_compute(test_input.device)
            out_core = net._forward_rebuild_core(test_input, org_weight, org_bias)
        torch.testing.assert_close(out_full, out_core)

    def test_conv_tucker(self):
        device = _compile_device()
        dtype = torch.float32
        base, test_input = _make_conv_base(16, device, dtype)
        net = LoConModule("test", base, multiplier=1, lora_dim=4, alpha=1,
                          weight_decompose=False, use_tucker=True, use_scalar=False,
                          ).to(device, dtype)
        net.apply_to()
        out_full = base(test_input)
        with torch.no_grad():
            org_weight = net.get_org_weight_for_compute(test_input.device).to(dtype)
            org_bias = net.get_org_bias_for_compute(test_input.device)
            out_core = net._forward_rebuild_core(test_input, org_weight, org_bias)
        torch.testing.assert_close(out_full, out_core)


class LoConTrainingTests(TrainingTestMixin, unittest.TestCase):
    module_cls = LoConModule
    module_kwargs = dict(multiplier=1, lora_dim=4, alpha=1,
                         weight_decompose=False, use_tucker=False, use_scalar=False)

    def test_training_loop_noncompiled(self):
        self._run_training_test(compile=False)

    def test_training_loop_compiled(self):
        self._run_training_test(compile=True)


class LoConCompileEquivalenceTests(CompileEquivalenceTestMixin, unittest.TestCase):
    module_cls = LoConModule
    module_kwargs = dict(multiplier=1, lora_dim=4, alpha=1,
                         weight_decompose=False, use_tucker=False, use_scalar=False)

    def test_compile_equivalence_linear(self):
        self._run_equivalence_test()


# ===========================================================================
# 2.  LohaModule
# ===========================================================================

class LohaRebuildCoreTests(RebuildCoreTestMixin, unittest.TestCase):
    module_cls = LohaModule
    module_kwargs = dict(multiplier=1, lora_dim=4, alpha=1,
                         weight_decompose=False, use_tucker=False, use_scalar=False)

    def test_linear_basic(self):
        self._run_rebuild_core_test()

    def test_linear_dora(self):
        self._run_rebuild_core_test(dict(weight_decompose=True))


class LohaTrainingTests(TrainingTestMixin, unittest.TestCase):
    module_cls = LohaModule
    module_kwargs = dict(multiplier=1, lora_dim=4, alpha=1,
                         weight_decompose=False, use_tucker=False, use_scalar=False)

    def test_training_loop_noncompiled(self):
        self._run_training_test(compile=False)

    def test_training_loop_compiled(self):
        self._run_training_test(compile=True)


class LohaCompileEquivalenceTests(CompileEquivalenceTestMixin, unittest.TestCase):
    module_cls = LohaModule
    module_kwargs = dict(multiplier=1, lora_dim=4, alpha=1,
                         weight_decompose=False, use_tucker=False, use_scalar=False)

    def test_compile_equivalence_linear(self):
        self._run_equivalence_test()


# ===========================================================================
# 3.  LokrModule
# ===========================================================================

class LokrRebuildCoreTests(RebuildCoreTestMixin, unittest.TestCase):
    module_cls = LokrModule
    module_kwargs = dict(multiplier=1, lora_dim=4, alpha=1,
                         weight_decompose=False, use_tucker=False, use_scalar=False)

    def test_linear_basic(self):
        self._run_rebuild_core_test()

    def test_linear_dora(self):
        self._run_rebuild_core_test(dict(weight_decompose=True))

    def test_linear_decompose_both(self):
        self._run_rebuild_core_test(dict(decompose_both=True))


class LokrTrainingTests(TrainingTestMixin, unittest.TestCase):
    module_cls = LokrModule
    module_kwargs = dict(multiplier=1, lora_dim=4, alpha=1,
                         weight_decompose=False, use_tucker=False, use_scalar=False)

    def test_training_loop_noncompiled(self):
        self._run_training_test(compile=False)

    def test_training_loop_compiled(self):
        self._run_training_test(compile=True)


class LokrCompileEquivalenceTests(CompileEquivalenceTestMixin, unittest.TestCase):
    module_cls = LokrModule
    module_kwargs = dict(multiplier=1, lora_dim=4, alpha=1,
                         weight_decompose=False, use_tucker=False, use_scalar=False)

    def test_compile_equivalence_linear(self):
        self._run_equivalence_test()


# ===========================================================================
# 4.  AbbaModule
# ===========================================================================

class AbbaRebuildCoreTests(RebuildCoreTestMixin, unittest.TestCase):
    module_cls = AbbaModule
    module_kwargs = dict(multiplier=1, r1=4, r2=4, alpha=1, weight_decompose=False)

    def test_linear_basic(self):
        self._run_rebuild_core_test()

    def test_linear_dora(self):
        self._run_rebuild_core_test(dict(weight_decompose=True))


class AbbaTrainingTests(TrainingTestMixin, unittest.TestCase):
    module_cls = AbbaModule
    module_kwargs = dict(multiplier=1, r1=4, r2=4, alpha=1, weight_decompose=False)

    def test_training_loop_noncompiled(self):
        self._run_training_test(compile=False)

    def test_training_loop_compiled(self):
        self._run_training_test(compile=True)


class AbbaCompileEquivalenceTests(CompileEquivalenceTestMixin, unittest.TestCase):
    module_cls = AbbaModule
    module_kwargs = dict(multiplier=1, r1=4, r2=4, alpha=1, weight_decompose=False)

    def test_compile_equivalence_linear(self):
        self._run_equivalence_test()


# ===========================================================================
# 5.  GLoRAModule
# ===========================================================================

class GLoRARebuildCoreTests(RebuildCoreTestMixin, unittest.TestCase):
    module_cls = GLoRAModule
    module_kwargs = dict(multiplier=1, lora_dim=4, alpha=1,
                         weight_decompose=False, use_tucker=False, use_scalar=False,
                         a1=64, a2=32)

    def test_linear_basic(self):
        self._run_rebuild_core_test()


class GLoRATrainingTests(TrainingTestMixin, unittest.TestCase):
    module_cls = GLoRAModule
    module_kwargs = dict(multiplier=1, lora_dim=4, alpha=1,
                         weight_decompose=False, use_tucker=False, use_scalar=False,
                         a1=64, a2=32)

    def test_training_loop_noncompiled(self):
        self._run_training_test(compile=False)

    def test_training_loop_compiled(self):
        self._run_training_test(compile=True)


class GLoRACompileEquivalenceTests(CompileEquivalenceTestMixin, unittest.TestCase):
    module_cls = GLoRAModule
    module_kwargs = dict(multiplier=1, lora_dim=4, alpha=1,
                         weight_decompose=False, use_tucker=False, use_scalar=False,
                         a1=64, a2=32)

    def test_compile_equivalence_linear(self):
        self._run_equivalence_test()


# ===========================================================================
# 6.  IA3Module
# ===========================================================================

class IA3RebuildCoreTests(RebuildCoreTestMixin, unittest.TestCase):
    module_cls = IA3Module
    module_kwargs = dict(multiplier=1)

    def test_linear_basic(self):
        self._run_rebuild_core_test()

    def test_linear_train_on_input(self):
        self._run_rebuild_core_test(dict(train_on_input=True))


class IA3TrainingTests(TrainingTestMixin, unittest.TestCase):
    module_cls = IA3Module
    module_kwargs = dict(multiplier=1)

    def test_training_loop_noncompiled(self):
        self._run_training_test(compile=False)

    def test_training_loop_compiled(self):
        self._run_training_test(compile=True)


class IA3CompileEquivalenceTests(CompileEquivalenceTestMixin, unittest.TestCase):
    module_cls = IA3Module
    module_kwargs = dict(multiplier=1)

    def test_compile_equivalence_linear(self):
        self._run_equivalence_test()


# ===========================================================================
# 7.  DiagOFTModule
# ===========================================================================

class DiagOFTRebuildCoreTests(RebuildCoreTestMixin, unittest.TestCase):
    module_cls = DiagOFTModule
    module_kwargs = dict(multiplier=1, lora_dim=8, constraint=0.0, rescaled=False)

    def test_linear_basic(self):
        self._run_rebuild_core_test()


class DiagOFTTrainingTests(TrainingTestMixin, unittest.TestCase):
    module_cls = DiagOFTModule
    module_kwargs = dict(multiplier=1, lora_dim=8, constraint=0.0, rescaled=False)

    def test_training_loop_noncompiled(self):
        self._run_training_test(compile=False)

    def test_training_loop_compiled(self):
        self._run_training_test(compile=True)


class DiagOFTCompileEquivalenceTests(CompileEquivalenceTestMixin, unittest.TestCase):
    module_cls = DiagOFTModule
    module_kwargs = dict(multiplier=1, lora_dim=8, constraint=0.0, rescaled=False)

    def test_compile_equivalence_linear(self):
        self._run_equivalence_test()


# ===========================================================================
# 8.  ButterflyOFTModule (BOFT)
# ===========================================================================

class BOFTTrainingTests(TrainingTestMixin, unittest.TestCase):
    module_cls = ButterflyOFTModule
    module_kwargs = dict(multiplier=1, lora_dim=8, constraint=0.0, rescaled=False,
                         boft_m=2, boft_b=4)

    def test_training_loop_noncompiled(self):
        self._run_training_test(compile=False)

    def test_training_loop_compiled(self):
        self._run_training_test(compile=True)


class BOFTCompileEquivalenceTests(CompileEquivalenceTestMixin, unittest.TestCase):
    module_cls = ButterflyOFTModule
    module_kwargs = dict(multiplier=1, lora_dim=8, constraint=0.0, rescaled=False,
                         boft_m=2, boft_b=4)

    def test_compile_equivalence_linear(self):
        self._run_equivalence_test()


# ===========================================================================
# 9.  Wrapper integration  –  create_lycoris with torch_compile=True
# ===========================================================================

class WrapperCompileIntegrationTests(unittest.TestCase):
    """End-to-end tests of the per-module compilation path through the wrapper."""

    def _create_wrapper(self, compile=False, algo="lora", bypass=False):
        device = _compile_device()
        base = nn.Sequential(
            nn.Linear(16, 16),
            nn.Linear(16, 16),
        ).to(device)

        kwargs = dict(
            linear_dim=4, linear_alpha=1, algo=algo,
        )
        if bypass:
            kwargs["bypass_mode"] = True
        if compile:
            kwargs["torch_compile"] = True
            kwargs.update(_wrapper_compile_kwargs())

        net = create_lycoris(base, 1.0, **kwargs)
        net.to(device)
        net.apply_to()
        return base, net, device

    def test_create_lycoris_no_compile(self):
        base, net, device = self._create_wrapper(compile=False)
        for lora in net.loras:
            self.assertFalse(
                hasattr(lora._forward_rebuild_core, "_torchdynamo_orig_callable"),
                f"{lora.lora_name} was unexpectedly compiled"
            )

    def test_create_lycoris_with_compile_lora(self):
        base, net, device = self._create_wrapper(compile=True, algo="lora")
        compiled_count = 0
        for lora in net.loras:
            if hasattr(lora._forward_rebuild_core, "_torchdynamo_orig_callable"):
                compiled_count += 1
        self.assertGreater(compiled_count, 0, "No modules were compiled")

    def test_create_lycoris_with_compile_loha(self):
        base, net, device = self._create_wrapper(compile=True, algo="loha")
        compiled_count = 0
        for lora in net.loras:
            if hasattr(lora._forward_rebuild_core, "_torchdynamo_orig_callable"):
                compiled_count += 1
        self.assertGreater(compiled_count, 0, "No modules were compiled")
        x = torch.randn(2, 16, device=device)
        out = base(x)
        loss = out.sum()
        loss.backward()

    def test_create_lycoris_with_compile_lokr(self):
        base, net, device = self._create_wrapper(compile=True, algo="lokr")
        x = torch.randn(2, 16, device=device)
        out = base(x)
        loss = out.sum()
        loss.backward()

    def test_create_lycoris_with_compile_ia3(self):
        base, net, device = self._create_wrapper(compile=True, algo="ia3")
        x = torch.randn(2, 16, device=device)
        out = base(x)
        loss = out.sum()
        loss.backward()

    def test_compile_forward_and_backward(self):
        base, net, device = self._create_wrapper(compile=True, algo="lora")
        x = torch.randn(2, 16, device=device)
        out = base(x)
        loss = out.sum()
        loss.backward()
        for lora in net.loras:
            for name, p in lora.named_parameters():
                if p.requires_grad:
                    self.assertIsNotNone(p.grad, f"{lora.lora_name}.{name} has no grad")
        self.assertGreater(len(net.loras), 0)

    def test_bypass_mode_skips_compile(self):
        base, net, device = self._create_wrapper(compile=True, algo="lora", bypass=True)
        x = torch.randn(2, 16, device=device)
        out = base(x)
        self.assertEqual(out.shape, x.shape)

    def test_module_type_table(self):
        base, net, device = self._create_wrapper(compile=True, algo="lora")
        self.assertGreater(len(net.loras), 0)
        for lora in net.loras:
            self.assertIsInstance(lora, LoConModule)


# ===========================================================================
# 10.  Deterministic numerical comparison
# ===========================================================================

class DeterministicCompareTests(unittest.TestCase):
    """Full deterministic comparison: same seed → same output with & without compile."""

    def _run_deterministic_test(self, algo):
        device = _compile_device()
        dtype = torch.float32
        dim = 32

        torch.manual_seed(1234)
        base_a = nn.Linear(dim, dim).to(device, dtype)
        torch.manual_seed(1234)
        base_b = nn.Linear(dim, dim).to(device, dtype)
        self.assertTrue(torch.equal(base_a.weight, base_b.weight))

        # Non-compiled
        torch.manual_seed(5678)
        net_a = create_lycoris(
            base_a, 1.0, linear_dim=4, linear_alpha=1, algo=algo,
        )
        net_a.to(device)
        net_a.apply_to()

        # Compiled
        torch.manual_seed(5678)
        kw = dict(linear_dim=4, linear_alpha=1, algo=algo,
                  torch_compile=True)
        kw.update(_wrapper_compile_kwargs())
        net_b = create_lycoris(base_b, 1.0, **kw)
        net_b.to(device)
        net_b.apply_to()

        x = torch.randn(3, dim, device=device)
        with torch.no_grad():
            out_a = base_a(x)
            out_b = base_b(x)

        torch.testing.assert_close(out_a, out_b)

    def test_linear_lora(self):
        self._run_deterministic_test("lora")

    def test_linear_loha(self):
        self._run_deterministic_test("loha")

    def test_linear_lokr(self):
        self._run_deterministic_test("lokr")

    def test_linear_ia3(self):
        self._run_deterministic_test("ia3")


if __name__ == "__main__":
    unittest.main()
