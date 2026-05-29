"""Tests for Weight Noising feature (inspired by ai-toolkit-perceptual).

Tests both module-level (LycorisBaseModule.inject_weight_noise) and
network-level (LycorisNetwork.inject_weight_noise) behavior.
"""

import unittest
import torch
import torch.nn as nn

from lycoris.modules.locon import LoConModule
from lycoris.modules.loha import LohaModule
from lycoris.modules.lokr import LokrModule
from lycoris.modules.abba import AbbaModule
from lycoris.modules.base import LycorisBaseModule
from lycoris.wrapper import LycorisNetwork, create_lycoris


device_and_dtype = [
    (torch.device("cpu"), torch.float32),
]

if torch.cuda.is_available():
    device_and_dtype.append((torch.device("cuda"), torch.float32))
    device_and_dtype.append((torch.device("cuda"), torch.float16))
    device_and_dtype.append((torch.device("cuda"), torch.bfloat16))


class ModuleWeightNoiseTests(unittest.TestCase):
    """Tests for LycorisBaseModule.inject_weight_noise().

    Direct module creation (without wrapper) requires setting
    weight_noise_sigma and weight_noise_mode manually, since the
    wrapper propagates these in apply_to().
    """

    def _make_module(self, module_cls, base_module, device, dtype, weight_noise_sigma=None, weight_noise_mode="relative"):
        base_module = base_module.to(device, dtype)
        net = module_cls(
            "test",
            base_module,
            multiplier=1,
            lora_dim=4,
            alpha=1,
        ).to(device, dtype)
        # Set weight noise config manually (normally done by wrapper.apply_to())
        net.weight_noise_sigma = weight_noise_sigma
        net.weight_noise_mode = weight_noise_mode
        net.apply_to()
        return net

    def test_disabled_when_sigma_none(self):
        """inject_weight_noise should return 0 when sigma is None."""
        base = nn.Linear(16, 16)
        net = self._make_module(LoConModule, base, torch.device("cpu"), torch.float32, weight_noise_sigma=None)
        result = net.inject_weight_noise()
        self.assertEqual(result, 0.0)

    def test_disabled_when_sigma_zero(self):
        """inject_weight_noise should return 0 when sigma is 0."""
        base = nn.Linear(16, 16)
        net = self._make_module(LoConModule, base, torch.device("cpu"), torch.float32, weight_noise_sigma=0.0)
        result = net.inject_weight_noise()
        self.assertEqual(result, 0.0)

    def test_disabled_when_sigma_negative(self):
        """inject_weight_noise should return 0 when sigma is negative."""
        base = nn.Linear(16, 16)
        net = self._make_module(LoConModule, base, torch.device("cpu"), torch.float32, weight_noise_sigma=-0.001)
        result = net.inject_weight_noise()
        self.assertEqual(result, 0.0)

    def test_absolute_mode_adds_noise(self):
        """In absolute mode, noise with fixed sigma should be added to all trainable params."""
        base = nn.Linear(16, 16)
        net = self._make_module(LoConModule, base, torch.device("cpu"), torch.float32,
                                weight_noise_sigma=0.01, weight_noise_mode="absolute")

        # Snapshot params before
        params_before = {n: p.data.clone() for n, p in net.named_parameters() if p.requires_grad}

        result = net.inject_weight_noise()

        # Should return non-zero noise norm
        self.assertGreater(result, 0.0)

        # At least one param should have changed
        any_changed = False
        for n, p in net.named_parameters():
            if p.requires_grad:
                if not torch.equal(p.data, params_before[n]):
                    any_changed = True
                    break
        self.assertTrue(any_changed, "At least one parameter should have changed after noise injection")

    def test_relative_mode_adds_noise(self):
        """In relative mode, noise should be added based on weight RMS."""
        base = nn.Linear(16, 16)
        net = self._make_module(LoConModule, base, torch.device("cpu"), torch.float32,
                                weight_noise_sigma=0.01, weight_noise_mode="relative")

        params_before = {n: p.data.clone() for n, p in net.named_parameters() if p.requires_grad}

        result = net.inject_weight_noise()
        self.assertGreater(result, 0.0)

        any_changed = False
        for n, p in net.named_parameters():
            if p.requires_grad:
                if not torch.equal(p.data, params_before[n]):
                    any_changed = True
                    break
        self.assertTrue(any_changed, "At least one parameter should have changed after noise injection")

    def test_relative_mode_zero_init_no_noise(self):
        """In relative mode, zero-initialized params (LoRA-up) should get zero noise."""
        base = nn.Linear(16, 16)
        net = self._make_module(LoConModule, base, torch.device("cpu"), torch.float32,
                                weight_noise_sigma=0.1, weight_noise_mode="relative")

        # Manually zero out all params to simulate zero-init state
        for p in net.parameters():
            p.data.zero_()

        result = net.inject_weight_noise()

        # With all-zero weights, RMS≈0 (clamp_min(1e-30) keeps it finite),
        # so sigma ≈ 0 and noise should be negligible.
        self.assertLess(result, 1e-10, "Zero-init params should produce negligible noise in relative mode")

    def test_absolute_mode_zero_init_still_noisy(self):
        """In absolute mode, zero-init params should still get noise."""
        base = nn.Linear(16, 16)
        net = self._make_module(LoConModule, base, torch.device("cpu"), torch.float32,
                                weight_noise_sigma=0.01, weight_noise_mode="absolute")

        # Manually zero out all params
        for p in net.parameters():
            p.data.zero_()

        result = net.inject_weight_noise()

        # Absolute mode should still add noise even when weights are zero
        self.assertGreater(result, 0.0)

    def test_noise_magnitude_scales_with_sigma(self):
        """Larger sigma should produce larger noise on average."""
        base = nn.Linear(16, 16)

        results = []
        for sigma in [0.001, 0.01, 0.1]:
            # Use a fresh module each time with non-zero weights
            net = self._make_module(LoConModule, base, torch.device("cpu"), torch.float32,
                                    weight_noise_sigma=sigma, weight_noise_mode="absolute")
            # Initialize with non-zero weights
            for p in net.parameters():
                if p.dim() >= 2:
                    nn.init.normal_(p.data, std=0.1)
            result = net.inject_weight_noise()
            results.append(result)

        # Noise norms should be monotonically increasing with sigma
        self.assertLess(results[0], results[1])
        self.assertLess(results[1], results[2])


class MultiModuleWeightNoiseTests(unittest.TestCase):
    """Test inject_weight_noise across different module types."""

    module_classes = [LoConModule, LohaModule, LokrModule, AbbaModule]

    def test_all_module_types(self):
        """inject_weight_noise should work for all module types."""
        for module_cls in self.module_classes:
            with self.subTest(module=module_cls.__name__):
                base = nn.Linear(16, 16).to("cpu")
                net = module_cls(
                    "test",
                    base,
                    multiplier=1,
                    lora_dim=4,
                    alpha=1,
                )
                # Set weight noise config manually
                net.weight_noise_sigma = 0.01
                net.weight_noise_mode = "relative"
                net.apply_to()

                result = net.inject_weight_noise()
                # Should return a non-negative float
                self.assertIsInstance(result, float)
                self.assertGreaterEqual(result, 0.0)


class NetworkWeightNoiseTests(unittest.TestCase):
    """Tests for LycorisNetwork.inject_weight_noise()."""

    def test_network_inject_weight_noise_disabled(self):
        """Network should return 0 when weight noise is not configured."""
        model = nn.Sequential(
            nn.Linear(16, 16),
            nn.Linear(16, 16),
        )
        network = create_lycoris(
            model,
            multiplier=1.0,
            linear_dim=4,
            linear_alpha=1,
            algo="lora",
        )
        network.apply_to()

        result = network.inject_weight_noise()
        self.assertEqual(result, 0.0)

    def test_network_inject_weight_noise_enabled(self):
        """Network should inject noise across all modules when configured."""
        model = nn.Sequential(
            nn.Linear(16, 16),
            nn.Linear(16, 16),
        )
        network = create_lycoris(
            model,
            multiplier=1.0,
            linear_dim=4,
            linear_alpha=1,
            algo="lora",
            weight_noise_sigma=0.01,
            weight_noise_mode="absolute",
        )
        network.apply_to()

        # Snapshot before
        params_before = {}
        for lora in network.loras:
            for n, p in lora.named_parameters():
                if p.requires_grad:
                    params_before[f"{lora.lora_name}.{n}"] = p.data.clone()

        result = network.inject_weight_noise()

        # Should return positive noise norm
        self.assertGreater(result, 0.0)

        # At least one param should have changed
        any_changed = False
        for lora in network.loras:
            for n, p in lora.named_parameters():
                if p.requires_grad:
                    key = f"{lora.lora_name}.{n}"
                    if key in params_before and not torch.equal(p.data, params_before[key]):
                        any_changed = True
                        break
            if any_changed:
                break
        self.assertTrue(any_changed, "At least one module parameter should have changed")

    def test_network_weight_noise_passed_to_modules(self):
        """weight_noise_sigma should be passed through to each module via apply_to()."""
        model = nn.Sequential(nn.Linear(16, 16))
        network = create_lycoris(
            model,
            multiplier=1.0,
            linear_dim=4,
            linear_alpha=1,
            algo="lora",
            weight_noise_sigma=0.005,
            weight_noise_mode="absolute",
        )
        network.apply_to()

        for lora in network.loras:
            self.assertEqual(lora.weight_noise_sigma, 0.005)
            self.assertEqual(lora.weight_noise_mode, "absolute")

    def test_network_default_mode_is_relative(self):
        """Default weight_noise_mode should be 'relative'."""
        model = nn.Sequential(nn.Linear(16, 16))
        network = create_lycoris(
            model,
            multiplier=1.0,
            linear_dim=4,
            linear_alpha=1,
            algo="lora",
            weight_noise_sigma=0.01,
        )
        network.apply_to()

        for lora in network.loras:
            self.assertEqual(lora.weight_noise_mode, "relative")

    def test_network_relative_mode_zero_init(self):
        """Network relative mode: zero-init params get zero noise."""
        model = nn.Sequential(nn.Linear(16, 16))
        network = create_lycoris(
            model,
            multiplier=1.0,
            linear_dim=4,
            linear_alpha=1,
            algo="lora",
            weight_noise_sigma=0.1,
            weight_noise_mode="relative",
        )
        network.apply_to()

        # Zero all module params (simulate fresh LoRA-up init)
        for lora in network.loras:
            for p in lora.parameters():
                p.data.zero_()

        result = network.inject_weight_noise()
        # Relative mode with zeroed weights: RMS ≈ 0, noise should be negligible
        self.assertLess(result, 1e-10, "Zero-init params should produce negligible noise in relative mode")


class CUDADeviceWeightNoiseTests(unittest.TestCase):
    """Test weight noising on CUDA device if available."""

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cuda_absolute_mode(self):
        """Weight noising should work on CUDA."""
        device = torch.device("cuda")
        base = nn.Linear(16, 16).to(device)
        net = LoConModule(
            "test",
            base,
            multiplier=1,
            lora_dim=4,
            alpha=1,
        ).to(device)
        net.weight_noise_sigma = 0.01
        net.weight_noise_mode = "absolute"
        net.apply_to()

        result = net.inject_weight_noise()
        self.assertGreater(result, 0.0)

        # Verify params are still on CUDA
        for p in net.parameters():
            self.assertEqual(p.device.type, "cuda")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cuda_relative_mode(self):
        """Relative mode should work on CUDA."""
        device = torch.device("cuda")
        base = nn.Linear(16, 16).to(device)
        net = LoConModule(
            "test",
            base,
            multiplier=1,
            lora_dim=4,
            alpha=1,
        ).to(device)
        net.weight_noise_sigma = 0.01
        net.weight_noise_mode = "relative"
        net.apply_to()

        result = net.inject_weight_noise()
        self.assertGreater(result, 0.0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cuda_network_level(self):
        """Network-level inject_weight_noise should work on CUDA."""
        device = torch.device("cuda")
        model = nn.Sequential(
            nn.Linear(16, 16),
            nn.Linear(16, 16),
        ).to(device)
        network = create_lycoris(
            model,
            multiplier=1.0,
            linear_dim=4,
            linear_alpha=1,
            algo="lora",
            weight_noise_sigma=0.01,
            weight_noise_mode="relative",
        )
        network.to(device)
        network.apply_to()

        result = network.inject_weight_noise()
        self.assertGreater(result, 0.0)


class DynamicSigmaTests(unittest.TestCase):
    """Tests for weight_noise_dynamic_sigma scaling."""

    def _make_module(self, module_cls, base_module, device, dtype,
                     weight_noise_sigma=None, weight_noise_mode="relative",
                     weight_noise_dynamic_sigma=False):
        base_module = base_module.to(device, dtype)
        net = module_cls(
            "test", base_module, multiplier=1, lora_dim=4, alpha=1,
        ).to(device, dtype)
        net.weight_noise_sigma = weight_noise_sigma
        net.weight_noise_mode = weight_noise_mode
        net.weight_noise_dynamic_sigma = weight_noise_dynamic_sigma
        net.apply_to()
        return net

    def test_dynamic_sigma_scaled_by_lr(self):
        """Larger LR should produce larger noise with dynamic sigma."""
        base = nn.Linear(16, 16)
        net = self._make_module(LoConModule, base, torch.device("cpu"), torch.float32,
                                weight_noise_sigma=0.01, weight_noise_mode="absolute",
                                weight_noise_dynamic_sigma=True)

        params_before = {n: p.data.clone() for n, p in net.named_parameters() if p.requires_grad}

        result_small = net.inject_weight_noise(lr=1e-5, effective_batch_size=1)
        params_after_small = {n: p.data.clone() for n, p in net.named_parameters() if p.requires_grad}

        # Reset params
        for n, p in net.named_parameters():
            if p.requires_grad:
                p.data.copy_(params_before[n])

        result_large = net.inject_weight_noise(lr=1e-2, effective_batch_size=1)

        # Larger LR → larger noise norm
        self.assertGreater(result_large, result_small)

    def test_dynamic_sigma_scaled_by_batch_size(self):
        """Larger batch size should produce smaller noise with dynamic sigma."""
        base = nn.Linear(16, 16)
        net = self._make_module(LoConModule, base, torch.device("cpu"), torch.float32,
                                weight_noise_sigma=0.01, weight_noise_mode="absolute",
                                weight_noise_dynamic_sigma=True)

        params_before = {n: p.data.clone() for n, p in net.named_parameters() if p.requires_grad}

        result_small_bs = net.inject_weight_noise(lr=1e-3, effective_batch_size=1)

        # Reset params
        for n, p in net.named_parameters():
            if p.requires_grad:
                p.data.copy_(params_before[n])

        result_large_bs = net.inject_weight_noise(lr=1e-3, effective_batch_size=16)

        # Larger batch → smaller noise (inversely proportional to √batch_size)
        self.assertGreater(result_small_bs, result_large_bs)

    def test_dynamic_sigma_quantitative(self):
        """Verify dynamic sigma produces a different noise level than static.

        inject_weight_noise returns noise_sq (squared Frobenius norm), so
        ratio should be (sigma_a / sigma_b)^2 when sigma scales linearly.
        """
        base = nn.Linear(16, 16)
        net = self._make_module(LoConModule, base, torch.device("cpu"), torch.float32,
                                weight_noise_sigma=1.0, weight_noise_mode="absolute",
                                weight_noise_dynamic_sigma=True)

        # dyn_scale = lr / sqrt(eff_bs)
        # sigma_a = 1.0 * 0.01 / 1 = 0.01
        # sigma_b = 1.0 * 0.001 / 1 = 0.001
        # noise_sq ∝ sigma^2, so ratio ≈ (0.01/0.001)^2 = 100
        result_a = net.inject_weight_noise(lr=0.01, effective_batch_size=1)
        for p in net.parameters():
            if p.requires_grad:
                p.data.zero_()
        result_b = net.inject_weight_noise(lr=0.001, effective_batch_size=1)

        ratio = result_a / result_b
        self.assertAlmostEqual(ratio, 100.0, delta=30.0)

    def test_dynamic_sigma_off_by_default(self):
        """When dynamic_sigma is False, LR and batch_size should not affect noise."""
        base = nn.Linear(16, 16)
        net = self._make_module(LoConModule, base, torch.device("cpu"), torch.float32,
                                weight_noise_sigma=0.01, weight_noise_mode="absolute",
                                weight_noise_dynamic_sigma=False)

        # Use same seed for deterministic random draws
        torch.manual_seed(42)
        for p in net.parameters():
            if p.requires_grad:
                p.data.fill_(1.0)
        result1 = net.inject_weight_noise(lr=1e-5, effective_batch_size=1)

        torch.manual_seed(42)
        for p in net.parameters():
            if p.requires_grad:
                p.data.fill_(1.0)
        result2 = net.inject_weight_noise(lr=1e-2, effective_batch_size=64)

        # The noise norm should be identical since dynamic_sigma is off
        self.assertEqual(result1, result2)

    def test_network_dynamic_sigma_passthrough(self):
        """Network should propagate dynamic_sigma to modules."""
        model = nn.Sequential(nn.Linear(16, 16))
        network = create_lycoris(
            model,
            multiplier=1.0, linear_dim=4, linear_alpha=1, algo="lora",
            weight_noise_sigma=0.01, weight_noise_mode="relative",
            weight_noise_dynamic_sigma=True,
        )
        network.apply_to()

        for lora in network.loras:
            self.assertTrue(lora.weight_noise_dynamic_sigma)

        # Verify it actually affects noise when lr/bs are passed
        params_before = {}
        for lora in network.loras:
            for n, p in lora.named_parameters():
                if p.requires_grad:
                    params_before[f"{lora.lora_name}.{n}"] = p.data.clone()

        result = network.inject_weight_noise(lr=0.01, effective_batch_size=4)
        self.assertGreater(result, 0.0)


class KohyaWeightNoiseTests(unittest.TestCase):
    """Tests for LycorisNetworkKohya weight noise attributes and propagation.

    Regression tests for the bug where LycorisNetworkKohya.__init__ bypasses
    LycorisNetwork.__init__ (calling torch.nn.Module.__init__ directly) and
    omits weight_noise_sigma, weight_noise_mode, and weight_noise_dynamic_sigma
    initialization, causing AttributeError in inject_weight_noise().
    """

    def _make_kohya_network(self, weight_noise_sigma=None, weight_noise_mode="relative",
                            weight_noise_dynamic_sigma=False, algo="locon"):
        """Create a LycorisNetworkKohya with mock models that match target module names.

        LycorisNetworkKohya only creates LoRA modules for specific module class
        names (e.g. Transformer2DModel, CLIPAttention), so we create dummy
        modules with matching names that contain nn.Linear submodules.
        """
        from lycoris.kohya import LycorisNetworkKohya

        # Create dummy modules whose class names match LycorisNetworkKohya targets
        class Transformer2DModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = nn.Linear(16, 16)

        class CLIPAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(16, 16)

        text_encoder = nn.Module()
        text_encoder.encoder = CLIPAttention()

        unet = nn.Module()
        unet.down = Transformer2DModel()

        kwargs = {}
        if weight_noise_sigma is not None:
            kwargs["weight_noise_sigma"] = weight_noise_sigma
        kwargs["weight_noise_mode"] = weight_noise_mode
        kwargs["weight_noise_dynamic_sigma"] = weight_noise_dynamic_sigma

        network = LycorisNetworkKohya(
            text_encoder,
            unet,
            multiplier=1.0,
            lora_dim=4,
            conv_lora_dim=0,
            alpha=1,
            network_module=algo,
            **kwargs,
        )
        network.apply_to(text_encoder, unet, True, True)
        return network

    def test_kohya_has_weight_noise_attributes(self):
        """LycorisNetworkKohya should have weight_noise_* attributes after __init__."""
        network = self._make_kohya_network(weight_noise_sigma=0.01)
        self.assertTrue(hasattr(network, 'weight_noise_sigma'))
        self.assertEqual(network.weight_noise_sigma, 0.01)
        self.assertTrue(hasattr(network, 'weight_noise_mode'))
        self.assertEqual(network.weight_noise_mode, "relative")
        self.assertTrue(hasattr(network, 'weight_noise_dynamic_sigma'))
        self.assertFalse(network.weight_noise_dynamic_sigma)

    def test_kohya_weight_noise_sigma_default_none(self):
        """weight_noise_sigma should default to None when not specified."""
        network = self._make_kohya_network()
        self.assertIsNone(network.weight_noise_sigma)

    def test_kohya_inject_weight_noise_no_error(self):
        """inject_weight_noise should not raise AttributeError.

        This is the exact regression test for the reported bug.
        """
        network = self._make_kohya_network(weight_noise_sigma=0.01)
        # This was the exact call path from the traceback
        result = network.inject_weight_noise(lr=1e-4, effective_batch_size=1, optimizer=None)
        self.assertGreaterEqual(result, 0.0)

    def test_kohya_inject_weight_noise_with_optimizer(self):
        """inject_weight_noise with optimizer should work when dynamic_sigma is False."""
        network = self._make_kohya_network(weight_noise_sigma=0.01)
        # Create a dummy optimizer
        optimizer = torch.optim.Adam(network.parameters(), lr=1e-4)
        result = network.inject_weight_noise(lr=1e-4, effective_batch_size=1, optimizer=optimizer)
        self.assertGreaterEqual(result, 0.0)

    def test_kohya_inject_weight_noise_dynamic_sigma_with_optimizer(self):
        """inject_weight_noise with optimizer and dynamic_sigma should not raise."""
        network = self._make_kohya_network(
            weight_noise_sigma=0.01,
            weight_noise_dynamic_sigma=True,
        )
        optimizer = torch.optim.Adam(network.parameters(), lr=1e-4)
        result = network.inject_weight_noise(lr=1e-4, effective_batch_size=1, optimizer=optimizer)
        self.assertGreater(result, 0.0)

    def test_kohya_weight_noise_propagated_to_modules(self):
        """weight_noise attrs should be propagated to child lora modules via apply_to."""
        network = self._make_kohya_network(
            weight_noise_sigma=0.005,
            weight_noise_mode="absolute",
            weight_noise_dynamic_sigma=True,
        )
        for lora in network.loras:
            self.assertEqual(lora.weight_noise_sigma, 0.005)
            self.assertEqual(lora.weight_noise_mode, "absolute")
            self.assertTrue(lora.weight_noise_dynamic_sigma)

    def test_kohya_weight_noise_disabled_returns_zero(self):
        """inject_weight_noise should return 0 when sigma is None."""
        network = self._make_kohya_network(weight_noise_sigma=None)
        result = network.inject_weight_noise()
        self.assertEqual(result, 0.0)

    def test_kohya_weight_noise_adds_noise(self):
        """With sigma > 0, inject_weight_noise should modify parameters."""
        network = self._make_kohya_network(
            weight_noise_sigma=0.01,
            weight_noise_mode="absolute",
        )

        params_before = {}
        for lora in network.loras:
            for n, p in lora.named_parameters():
                if p.requires_grad:
                    params_before[f"{lora.lora_name}.{n}"] = p.data.clone()

        result = network.inject_weight_noise()
        self.assertGreater(result, 0.0)

        any_changed = False
        for lora in network.loras:
            for n, p in lora.named_parameters():
                if p.requires_grad:
                    key = f"{lora.lora_name}.{n}"
                    if key in params_before and not torch.equal(p.data, params_before[key]):
                        any_changed = True
                        break
            if any_changed:
                break
        self.assertTrue(any_changed, "At least one parameter should have changed after noise injection")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_kohya_weight_noise_cuda(self):
        """Weight noise should work on CUDA for LycorisNetworkKohya."""
        network = self._make_kohya_network(
            weight_noise_sigma=0.01,
            weight_noise_mode="relative",
        )
        result = network.inject_weight_noise()
        self.assertGreater(result, 0.0)


if __name__ == "__main__":
    unittest.main()
