"""Unit tests for TSM (TimeStep Master) module."""

import math
import unittest
from itertools import product

import torch
import torch.nn as nn
import torch.nn.functional as F

from lycoris.modules.tsm import (
    TSMModule,
    set_tsm_timestep,
    get_tsm_timestep,
    clear_tsm_timestep,
)


class TSMModuleTests(unittest.TestCase):
    """Tests for TSMModule core functionality."""

    def setUp(self):
        """Set up common test fixtures."""
        self.dim = 16
        self.lora_dim = 4
        self.linear = nn.Linear(self.dim, self.dim)
        self.conv2d = nn.Conv2d(self.dim, self.dim, (3, 3), 1, 1)
        self.default_kwargs = dict(
            lora_dim=self.lora_dim,
            alpha=4,
            tsm_n_scales=[8, 1],
            tsm_num_timesteps=1000,
            tsm_stage=1,
        )

    def tearDown(self):
        """Clean up global state."""
        clear_tsm_timestep()

    # ------------------------------------------------------------------
    # Module creation
    # ------------------------------------------------------------------

    def test_create_linear(self):
        """TSMModule wraps a Linear layer."""
        mod = TSMModule("test", self.linear, **self.default_kwargs)
        self.assertEqual(mod.module_type, "linear")
        self.assertFalse(mod.isconv)
        self.assertEqual(len(mod.experts), 2)  # 2 scales
        self.assertEqual(len(mod.experts[0]), 8)  # core: 8 experts
        self.assertEqual(len(mod.experts[1]), 1)  # context: 1 expert

    def test_create_conv2d(self):
        """TSMModule wraps a Conv2d layer."""
        mod = TSMModule("test", self.conv2d, **self.default_kwargs)
        self.assertEqual(mod.module_type, "conv2d")
        self.assertTrue(mod.isconv)
        self.assertEqual(len(mod.experts), 2)

    def test_create_default_scales(self):
        """Default n_scales is [8, 1]."""
        mod = TSMModule(
            "test", self.linear, lora_dim=4, alpha=4,
            tsm_num_timesteps=1000,
        )
        self.assertEqual(mod.tsm_n_scales, [8, 1])

    def test_create_custom_scales(self):
        """Custom n_scales [8, 4, 2, 1] creates 4 scales."""
        mod = TSMModule(
            "test", self.linear, lora_dim=4, alpha=4,
            tsm_n_scales=[8, 4, 2, 1], tsm_num_timesteps=1000,
        )
        self.assertEqual(len(mod.experts), 4)
        self.assertEqual(len(mod.experts[0]), 8)
        self.assertEqual(len(mod.experts[1]), 4)
        self.assertEqual(len(mod.experts[2]), 2)
        self.assertEqual(len(mod.experts[3]), 1)

    def test_create_too_few_scales_raises(self):
        """At least 2 scales are required."""
        with self.assertRaises(ValueError):
            TSMModule(
                "test", self.linear, lora_dim=4, alpha=4,
                tsm_n_scales=[8], tsm_num_timesteps=1000,
            )

    # ------------------------------------------------------------------
    # Expert indexing
    # ------------------------------------------------------------------

    def test_expert_indexing_basic(self):
        """Expert index follows ceil(t/T*n_j) formula."""
        mod = TSMModule(
            "test", self.linear, lora_dim=4, alpha=4,
            tsm_n_scales=[8, 1], tsm_num_timesteps=1000,
        )
        # Scale 0 (n=8): t=1 -> ceil(1/1000*8)=1 -> idx=0
        self.assertEqual(mod._get_expert_index(1, 0), 0)
        # t=125 -> ceil(125/1000*8)=1 -> idx=0
        self.assertEqual(mod._get_expert_index(125, 0), 0)
        # t=126 -> ceil(126/1000*8)=2 -> idx=1
        self.assertEqual(mod._get_expert_index(126, 0), 1)
        # t=500 -> ceil(500/1000*8)=4 -> idx=3
        self.assertEqual(mod._get_expert_index(500, 0), 3)
        # t=1000 -> ceil(1000/1000*8)=8 -> idx=7
        self.assertEqual(mod._get_expert_index(1000, 0), 7)

        # Scale 1 (n=1): always expert 0
        self.assertEqual(mod._get_expert_index(1, 1), 0)
        self.assertEqual(mod._get_expert_index(500, 1), 0)
        self.assertEqual(mod._get_expert_index(1000, 1), 0)

    def test_expert_indexing_multi_scale(self):
        """Expert indexing works for multi-scale config [8, 4, 2, 1]."""
        mod = TSMModule(
            "test", self.linear, lora_dim=4, alpha=4,
            tsm_n_scales=[8, 4, 2, 1], tsm_num_timesteps=1000,
        )
        # t=500: scale 1 (n=4) -> ceil(500/1000*4)=2 -> idx=1
        self.assertEqual(mod._get_expert_index(500, 1), 1)
        # t=500: scale 2 (n=2) -> ceil(500/1000*2)=1 -> idx=0
        self.assertEqual(mod._get_expert_index(500, 2), 0)
        # t=500: scale 3 (n=1) -> ceil(500/1000*1)=1 -> idx=0
        self.assertEqual(mod._get_expert_index(500, 3), 0)

    # ------------------------------------------------------------------
    # Zero initialization
    # ------------------------------------------------------------------

    def test_zero_init_output(self):
        """Output delta is zero at initialization (B=zero init)."""
        mod = TSMModule("test", self.linear, **self.default_kwargs)
        x = torch.randn(2, self.dim)
        # Stage 1 forward
        delta = mod._forward_stage1(x)
        self.assertTrue(torch.allclose(delta, torch.zeros_like(delta), atol=1e-6),
                        f"Expected zero delta, got max={delta.abs().max()}")

    def test_zero_init_stage2(self):
        """Stage 2 output delta is zero at init (experts=zero + router=zero gates)."""
        mod = TSMModule("test", self.linear, **self.default_kwargs)
        mod.set_stage(2)
        x = torch.randn(2, self.dim)
        set_tsm_timestep(500)
        delta = mod._forward_stage2(x)
        self.assertTrue(torch.allclose(delta, torch.zeros_like(delta), atol=1e-6),
                        f"Expected zero delta, got max={delta.abs().max()}")

    # ------------------------------------------------------------------
    # Stage switching
    # ------------------------------------------------------------------

    def test_stage_switch_requires_grad(self):
        """Stage switching toggles requires_grad correctly."""
        mod = TSMModule("test", self.linear, **self.default_kwargs)

        # Stage 1: experts trainable, router frozen
        mod.set_stage(1)
        for scale_experts in mod.experts:
            for expert in scale_experts:
                for p in expert.parameters():
                    self.assertTrue(p.requires_grad)
        for p in mod.router_fc.parameters():
            self.assertFalse(p.requires_grad)
        for p in mod.timestep_embed.parameters():
            self.assertFalse(p.requires_grad)

        # Stage 2: experts frozen, router trainable
        mod.set_stage(2)
        for scale_experts in mod.experts:
            for expert in scale_experts:
                for p in expert.parameters():
                    self.assertFalse(p.requires_grad)
        for p in mod.router_fc.parameters():
            self.assertTrue(p.requires_grad)
        for p in mod.timestep_embed.parameters():
            self.assertTrue(p.requires_grad)

    # ------------------------------------------------------------------
    # Forward pass — shapes
    # ------------------------------------------------------------------

    def test_forward_stage1_linear_shape(self):
        """Stage 1 forward produces correct output shape for Linear."""
        mod = TSMModule("test", self.linear, **self.default_kwargs)
        x = torch.randn(2, self.dim)
        out = mod(x)
        self.assertEqual(out.shape, (2, self.dim))

    def test_forward_stage1_conv2d_shape(self):
        """Stage 1 forward produces correct output shape for Conv2d."""
        mod = TSMModule("test", self.conv2d, **self.default_kwargs)
        x = torch.randn(1, self.dim, 16, 16)
        out = mod(x)
        self.assertEqual(out.shape, (1, self.dim, 16, 16))

    def test_forward_stage2_linear_shape(self):
        """Stage 2 forward produces correct output shape for Linear."""
        mod = TSMModule("test", self.linear, **self.default_kwargs)
        mod.set_stage(2)
        set_tsm_timestep(500)
        x = torch.randn(2, self.dim)
        out = mod(x)
        self.assertEqual(out.shape, (2, self.dim))

    def test_forward_stage2_conv2d_shape(self):
        """Stage 2 forward produces correct output shape for Conv2d."""
        mod = TSMModule("test", self.conv2d, **self.default_kwargs)
        mod.set_stage(2)
        set_tsm_timestep(500)
        x = torch.randn(1, self.dim, 16, 16)
        out = mod(x)
        self.assertEqual(out.shape, (1, self.dim, 16, 16))

    # ------------------------------------------------------------------
    # Timestep context
    # ------------------------------------------------------------------

    def test_timestep_global_state(self):
        """Global timestep state works correctly."""
        self.assertIsNone(get_tsm_timestep())
        set_tsm_timestep(500)
        self.assertEqual(get_tsm_timestep(), 500)
        clear_tsm_timestep()
        self.assertIsNone(get_tsm_timestep())

    def test_forward_with_timestep_context(self):
        """Forward pass respects timestep context."""
        mod = TSMModule("test", self.linear, **self.default_kwargs)
        x = torch.randn(2, self.dim)

        # Different timesteps should select different experts
        set_tsm_timestep(100)
        idx_100 = mod._get_expert_index(100, 0)
        set_tsm_timestep(900)
        idx_900 = mod._get_expert_index(900, 0)
        self.assertNotEqual(idx_100, idx_900)

    def test_default_timestep_uses_max(self):
        """When no timestep is set, defaults to T (max timestep)."""
        mod = TSMModule("test", self.linear, **self.default_kwargs)
        t = mod._get_timestep()
        self.assertEqual(t, 1000)

    # ------------------------------------------------------------------
    # Router
    # ------------------------------------------------------------------

    def test_router_zero_init_produces_zero_gates(self):
        """Router with zero init produces 0.0 gates (no sigmoid per paper Eq. 6)."""
        mod = TSMModule("test", self.linear, **self.default_kwargs)
        z = torch.zeros(self.lora_dim)
        gates = mod._compute_gates(z, torch.zeros(self.dim), 500)
        # No sigmoid: FC(0) + embed(0) = 0 + 0 = 0
        self.assertTrue(torch.allclose(gates, torch.zeros_like(gates), atol=1e-6))

    def test_router_bottleneck_mode(self):
        """Router in bottleneck mode uses lora_dim-sized input."""
        mod = TSMModule(
            "test", self.linear, lora_dim=4, alpha=4,
            tsm_n_scales=[8, 1], tsm_num_timesteps=1000,
            tsm_router_input="bottleneck",
        )
        self.assertEqual(mod.router_fc.in_features, self.lora_dim)

    def test_router_input_mode(self):
        """Router in input mode uses in_dim-sized input (paper default)."""
        mod = TSMModule(
            "test", self.linear, lora_dim=4, alpha=4,
            tsm_n_scales=[8, 1], tsm_num_timesteps=1000,
            tsm_router_input="input",
        )
        self.assertEqual(mod.router_fc.in_features, self.dim)

    def test_router_default_is_input_mode(self):
        """Default router input is 'input' per paper (z_t is input feature)."""
        mod = TSMModule(
            "test", self.linear, lora_dim=4, alpha=4,
            tsm_n_scales=[8, 1], tsm_num_timesteps=1000,
        )
        self.assertEqual(mod.tsm_router_input, "input")
        self.assertEqual(mod.router_fc.in_features, self.dim)

    # ------------------------------------------------------------------
    # Bypass mode
    # ------------------------------------------------------------------

    def test_bypass_forward_stage1(self):
        """Bypass mode works in stage 1."""
        mod = TSMModule(
            "test", self.linear, bypass_mode=True, **self.default_kwargs
        )
        x = torch.randn(2, self.dim)
        out = mod(x)
        self.assertEqual(out.shape, (2, self.dim))

    def test_bypass_forward_stage2(self):
        """Bypass mode works in stage 2."""
        mod = TSMModule(
            "test", self.linear, bypass_mode=True, **self.default_kwargs
        )
        mod.set_stage(2)
        set_tsm_timestep(500)
        x = torch.randn(2, self.dim)
        out = mod(x)
        self.assertEqual(out.shape, (2, self.dim))

    # ------------------------------------------------------------------
    # Serialization round-trip
    # ------------------------------------------------------------------

    def test_custom_state_dict_keys(self):
        """custom_state_dict produces expected keys."""
        mod = TSMModule("test", self.linear, **self.default_kwargs)
        sd = mod.custom_state_dict()
        # Check alpha
        self.assertIn("alpha", sd)
        # Check expert keys
        for s in range(len(mod.tsm_n_scales)):
            for e in range(mod.tsm_n_scales[s]):
                self.assertIn(f"experts.{s}.{e}.down.weight", sd)
                self.assertIn(f"experts.{s}.{e}.up.weight", sd)
        # Check router keys
        self.assertIn("router_fc.weight", sd)
        self.assertIn("router_fc.bias", sd)
        self.assertIn("timestep_embed.weight", sd)

    def test_algo_check(self):
        """algo_check identifies TSM state dicts."""
        mod = TSMModule("test", self.linear, **self.default_kwargs)
        sd = mod.custom_state_dict()
        prefixed_sd = {f"test.{k}": v for k, v in sd.items()}
        self.assertTrue(TSMModule.algo_check(prefixed_sd, "test"))

    def test_algo_check_non_tsm(self):
        """algo_check rejects non-TSM state dicts."""
        non_tsm_sd = {"test.lora_up.weight": torch.randn(16, 4)}
        self.assertFalse(TSMModule.algo_check(non_tsm_sd, "test"))

    # ------------------------------------------------------------------
    # Gradient flow
    # ------------------------------------------------------------------

    def test_stage1_gradient_flow(self):
        """In stage 1, active expert parameters receive gradients in ALL scales."""
        mod = TSMModule("test", self.linear, **self.default_kwargs)
        x = torch.randn(2, self.dim, requires_grad=True)
        set_tsm_timestep(500)
        out = mod(x)
        loss = out.sum()
        loss.backward()
        # Each scale's active expert should have non-zero gradients
        for scale_idx, scale_experts in enumerate(mod.experts):
            expert_idx = mod._get_expert_index(500, scale_idx)
            expert = scale_experts[expert_idx]
            has_grad = any(
                p.grad is not None and p.grad.abs().sum() > 0
                for p in expert.parameters()
            )
            self.assertTrue(
                has_grad,
                f"Scale {scale_idx} expert {expert_idx} should have gradients"
            )

    def test_stage1_all_scales_contribute(self):
        """Stage 1 uses experts from ALL scales, not just core."""
        mod = TSMModule(
            "test", self.linear, lora_dim=4, alpha=4,
            tsm_n_scales=[8, 1], tsm_num_timesteps=1000,
        )
        # Perturb a context scale expert's B matrix so it contributes non-zero output
        with torch.no_grad():
            mod.experts[1][0]["up"].weight.fill_(0.01)
        x = torch.randn(2, self.dim)
        set_tsm_timestep(500)
        delta = mod._forward_stage1(x)
        # The output should reflect contributions from both scales
        self.assertGreater(delta.abs().sum().item(), 0,
                           "Context scale expert should contribute to stage 1 output")
        clear_tsm_timestep()

    def test_default_timestep_no_oob(self):
        """Default timestep (T=1000) does not cause embedding index OOB."""
        mod = TSMModule(
            "test", self.linear, lora_dim=4, alpha=4,
            tsm_n_scales=[8, 1], tsm_num_timesteps=1000,
        )
        mod.set_stage(2)
        # Don't set timestep — should default to 1000, clamped to 999
        x = torch.randn(2, self.dim)
        try:
            out = mod(x)
            self.assertEqual(out.shape, (2, self.dim))
        except IndexError:
            self.fail("Default timestep caused embedding index out of bounds")

    def test_stage2_router_gradient_flow(self):
        """In stage 2, router parameters receive gradients."""
        mod = TSMModule("test", self.linear, **self.default_kwargs)
        mod.set_stage(2)
        set_tsm_timestep(500)
        x = torch.randn(2, self.dim, requires_grad=True)
        out = mod(x)
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(mod.router_fc.weight.grad)
        self.assertIsNotNone(mod.timestep_embed.weight.grad)

    # ------------------------------------------------------------------
    # Scalar
    # ------------------------------------------------------------------

    def test_use_scalar(self):
        """use_scalar creates a learnable scalar parameter."""
        mod = TSMModule(
            "test", self.linear, use_scalar=True, **self.default_kwargs
        )
        self.assertIsInstance(mod.scalar, nn.Parameter)

    def test_no_scalar(self):
        """Without use_scalar, scalar is a buffer."""
        mod = TSMModule(
            "test", self.linear, use_scalar=False, **self.default_kwargs
        )
        self.assertFalse(isinstance(mod.scalar, nn.Parameter))


class TSMModuleDeviceTests(unittest.TestCase):
    """Device-specific tests (CUDA if available)."""

    def test_cuda_stage1_forward(self):
        """Stage 1 forward works on CUDA."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        dim = 16
        linear = nn.Linear(dim, dim).cuda()
        mod = TSMModule(
            "test", linear, lora_dim=4, alpha=4,
            tsm_n_scales=[8, 1], tsm_num_timesteps=1000,
        )
        x = torch.randn(2, dim, device="cuda")
        out = mod(x)
        self.assertEqual(out.device.type, "cuda")
        self.assertEqual(out.shape, (2, dim))

    def test_cuda_stage2_forward(self):
        """Stage 2 forward works on CUDA."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        dim = 16
        linear = nn.Linear(dim, dim).cuda()
        mod = TSMModule(
            "test", linear, lora_dim=4, alpha=4,
            tsm_n_scales=[8, 1], tsm_num_timesteps=1000,
        )
        mod.set_stage(2)
        set_tsm_timestep(500)
        x = torch.randn(2, dim, device="cuda")
        out = mod(x)
        self.assertEqual(out.device.type, "cuda")
        self.assertEqual(out.shape, (2, dim))
        clear_tsm_timestep()


class TSMEndToEndTests(unittest.TestCase):
    """End-to-end integration tests using create_lycoris."""

    def test_create_lycoris_tsm(self):
        """create_lycoris with algo='tsm' creates a working network."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        from lycoris import create_lycoris

        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.Linear(128, 64),
        ).cuda()

        network = create_lycoris(
            model,
            algo="tsm",
            linear_dim=4,
            linear_alpha=4,
            tsm_n_scales="8,1",
            tsm_num_timesteps=1000,
            tsm_stage=1,
            tsm_router_input="bottleneck",
        )
        network.apply_to()

        # Verify network structure
        self.assertEqual(len(network.loras), 2)
        for lora in network.loras:
            self.assertIsInstance(lora, TSMModule)
            self.assertEqual(lora.tsm_n_scales, [8, 1])
            self.assertEqual(sum(lora.tsm_n_scales), 9)

        # Stage 1 forward + backward: all scales get gradients
        x = torch.randn(2, 64, device="cuda")
        set_tsm_timestep(500)
        out = model(x)
        self.assertEqual(out.shape, (2, 64))
        loss = out.sum()
        loss.backward()

        for lora in network.loras:
            for s, scale_experts in enumerate(lora.experts):
                eidx = lora._get_expert_index(500, s)
                expert = scale_experts[eidx]
                # Check that at least one parameter has a non-zero gradient.
                # up.weight.grad is non-zero even when up is zero-initialized
                # (gradient of up flows through the matmul with down).
                has_grad = any(
                    p.grad is not None and p.grad.abs().sum() > 0
                    for p in expert.parameters()
                )
                self.assertTrue(
                    has_grad,
                    f"Scale {s} expert {eidx} should have at least one param with grad"
                )
        clear_tsm_timestep()

        # Optimizer step to update context expert weights from stage 1 gradients
        optimizer = torch.optim.Adam(network.parameters(), lr=0.01)
        optimizer.step()
        optimizer.zero_grad()

        # Stage 2 forward + backward: router gets gradients
        network.set_tsm_stage(2)
        set_tsm_timestep(500)
        out2 = model(x)
        self.assertEqual(out2.shape, (2, 64))
        loss2 = out2.sum()
        loss2.backward()

        for lora in network.loras:
            # Router gradient flows through timestep embedding (FC gradient is
            # zero because the bottleneck z is detached zeros in rebuild mode).
            self.assertIsNotNone(lora.timestep_embed.weight.grad)
            self.assertGreater(
                lora.timestep_embed.weight.grad.norm().item(), 0,
                "Timestep embedding should have non-zero gradient in stage 2"
            )
        clear_tsm_timestep()


class TSMPaperAlignmentTests(unittest.TestCase):
    """Tests verifying alignment with the TSM paper (arXiv:2503.07416).

    These tests verify specific equations and design choices from the paper.
    """

    def setUp(self):
        self.dim = 16
        self.lora_dim = 4
        self.linear = nn.Linear(self.dim, self.dim)

    def tearDown(self):
        clear_tsm_timestep()

    # ------------------------------------------------------------------
    # Paper Eq. 6: G(z_t, t) = F(z_t) + ε(t) — NO sigmoid
    # ------------------------------------------------------------------

    def test_gates_are_unbounded_no_sigmoid(self):
        """Paper Eq. 6: gates = F(z_t) + ε(t), no activation function.

        With non-zero inputs, gates should be raw real values (not clamped
        to [0,1] as they would be with sigmoid).
        """
        mod = TSMModule(
            "test", self.linear, lora_dim=4, alpha=4,
            tsm_n_scales=[8, 1], tsm_num_timesteps=1000,
        )
        # Manually set non-zero router weights to produce large values
        with torch.no_grad():
            mod.router_fc.weight.fill_(10.0)
            mod.router_fc.bias.fill_(10.0)
            mod.timestep_embed.weight.fill_(10.0)

        z = torch.ones(self.dim)
        gates = mod._compute_gates(z, torch.ones(self.dim), 500)
        # Without sigmoid, gates can exceed 1.0
        self.assertTrue(
            (gates > 1.0).any(),
            "Gates should be unbounded (>1) without sigmoid per paper Eq. 6"
        )

    def test_gates_can_be_negative(self):
        """Paper Eq. 6: gates can be negative (no sigmoid bound)."""
        mod = TSMModule(
            "test", self.linear, lora_dim=4, alpha=4,
            tsm_n_scales=[8, 4, 2, 1], tsm_num_timesteps=1000,
        )
        with torch.no_grad():
            mod.router_fc.weight.fill_(-5.0)
            mod.router_fc.bias.fill_(-5.0)
            mod.timestep_embed.weight.fill_(-5.0)

        z = torch.ones(self.dim)
        gates = mod._compute_gates(z, torch.ones(self.dim), 500)
        self.assertTrue(
            (gates < 0).any(),
            "Gates should be able to go negative without sigmoid"
        )

    # ------------------------------------------------------------------
    # Paper: z_t is the input feature (dimension k), not bottleneck (r)
    # ------------------------------------------------------------------

    def test_router_default_uses_input_feature(self):
        """Paper Section 3.2: z_t ∈ R^{k×l} is the input feature.

        The default router should use the full input dimension (k=in_dim),
        not the bottleneck dimension (r=lora_dim).
        """
        mod = TSMModule(
            "test", self.linear, lora_dim=4, alpha=4,
            tsm_n_scales=[8, 1], tsm_num_timesteps=1000,
        )
        # Default should be "input" mode
        self.assertEqual(mod.tsm_router_input, "input")
        # FC input dimension should match in_dim, not lora_dim
        self.assertEqual(mod.router_fc.in_features, self.dim)
        self.assertNotEqual(mod.router_fc.in_features, self.lora_dim)

    # ------------------------------------------------------------------
    # Paper: ε(t) extracts the t-th row, t ∈ [1, T] (1-indexed)
    # ------------------------------------------------------------------

    def test_timestep_embedding_1indexed(self):
        """Paper: ε(t) extracts t-th row of T×(m-1) matrix, t ∈ [1,T].

        For t=1, should use embedding index 0 (first row).
        For t=T, should use embedding index T-1 (last row).
        """
        T = 1000
        mod = TSMModule(
            "test", self.linear, lora_dim=4, alpha=4,
            tsm_n_scales=[8, 1], tsm_num_timesteps=T,
        )

        # Set embedding to identity-like: row i has value i
        with torch.no_grad():
            for i in range(T):
                mod.timestep_embed.weight[i] = float(i)

        # Compute gates at t=1 (paper's first timestep)
        # Should use embedding[0] = 0.0
        z = torch.zeros(self.dim)
        gates_t1 = mod._compute_gates(z, z, 1)
        # FC(0) = bias (0 at init), embed(0) = 0.0, so gates ≈ 0
        self.assertAlmostEqual(
            gates_t1.item(), 0.0, places=4,
            msg="t=1 should map to embedding row 0 (1-indexed convention)"
        )

        # Compute gates at t=500
        # Should use embedding[499] = 499.0
        gates_t500 = mod._compute_gates(z, z, 500)
        self.assertAlmostEqual(
            gates_t500.item(), 499.0, places=2,
            msg="t=500 should map to embedding row 499"
        )

    def test_timestep_embedding_boundary(self):
        """Embedding index stays in bounds for t=1 and t=T."""
        T = 1000
        mod = TSMModule(
            "test", self.linear, lora_dim=4, alpha=4,
            tsm_n_scales=[8, 1], tsm_num_timesteps=T,
        )
        z = torch.zeros(self.dim)

        # t=1 should not crash (embedding index 0)
        gates = mod._compute_gates(z, z, 1)
        self.assertEqual(gates.shape, (1,))

        # t=T=1000 should not crash (embedding index 999)
        gates = mod._compute_gates(z, z, T)
        self.assertEqual(gates.shape, (1,))

    # ------------------------------------------------------------------
    # Paper Eq. 7: i_j = ⌈t/T · n_j⌉
    # ------------------------------------------------------------------

    def test_expert_index_formula_eq7(self):
        """Paper Eq. 7: i_j = ceil(t/T * n_j), mapping timesteps to experts.

        With n_j=8 and T=1000:
        t=1   → ceil(0.008) = 1 → expert 0
        t=125 → ceil(1.0)   = 1 → expert 0
        t=126 → ceil(1.008) = 2 → expert 1
        t=250 → ceil(2.0)   = 2 → expert 1
        t=251 → ceil(2.008) = 3 → expert 2
        t=1000→ ceil(8.0)   = 8 → expert 7
        """
        mod = TSMModule(
            "test", self.linear, lora_dim=4, alpha=4,
            tsm_n_scales=[8, 1], tsm_num_timesteps=1000,
        )
        test_cases = [
            (1, 0, 0),      # t=1, scale 0 (n=8) → expert 0
            (125, 0, 0),    # boundary: exactly at interval end
            (126, 0, 1),    # just past boundary
            (250, 0, 1),    # end of interval 2
            (251, 0, 2),    # start of interval 3
            (500, 0, 3),    # middle
            (1000, 0, 7),   # last timestep → last expert
        ]
        for t, scale_idx, expected_idx in test_cases:
            with self.subTest(t=t, scale_idx=scale_idx):
                actual = mod._get_expert_index(t, scale_idx)
                self.assertEqual(
                    actual, expected_idx,
                    f"t={t}, scale={scale_idx}: expected expert {expected_idx}, got {actual}"
                )

    # ------------------------------------------------------------------
    # Paper Eq. 5: ΔW = B_{i1}A_{i1} + Σ_{j=2}^m G_j ⊙ B_{ij}A_{ij}
    # ------------------------------------------------------------------

    def test_stage2_core_expert_ungated(self):
        """Paper Eq. 5: core expert (j=1) contributes without gating.

        When context gates are zero, output should equal core expert only.
        """
        mod = TSMModule(
            "test", self.linear, lora_dim=4, alpha=4,
            tsm_n_scales=[8, 1], tsm_num_timesteps=1000,
        )
        mod.set_stage(2)

        # Make only the core expert produce non-zero output
        for scale_experts in mod.experts:
            for expert in scale_experts:
                nn.init.zeros_(expert["up"].weight)
        # Set one specific core expert to non-zero
        core_expert = mod.experts[0][3]
        nn.init.ones_(core_expert["up"].weight)

        set_tsm_timestep(500)  # → core expert 3
        x = torch.randn(2, self.dim)
        delta = mod._forward_stage2(x)

        # With zero-init router, gates=0, so only core expert contributes
        core_only = mod._apply_expert(core_expert, x, x.dtype) * mod.scalar * mod.scale
        self.assertTrue(
            torch.allclose(delta, core_only, atol=1e-5),
            "Stage 2 output should equal core expert output when context gates are 0"
        )

    def test_stage2_context_experts_are_gated(self):
        """Paper Eq. 5: context experts (j≥2) are scaled by G_j gates.

        Changing the gate value should proportionally change the context expert
        contribution.
        """
        mod = TSMModule(
            "test", self.linear, lora_dim=4, alpha=4,
            tsm_n_scales=[8, 1], tsm_num_timesteps=1000,
        )
        mod.set_stage(2)

        # Set context expert to produce known non-zero output
        ctx_expert = mod.experts[1][0]
        nn.init.ones_(ctx_expert["up"].weight)

        set_tsm_timestep(500)
        x = torch.randn(2, self.dim)

        # Set gate to a specific value via router weights
        with torch.no_grad():
            mod.router_fc.bias.fill_(0.0)
            mod.timestep_embed.weight.fill_(0.0)

        delta_zero_gate = mod._forward_stage2(x)

        # Now set gate to 2.0 via embedding
        with torch.no_grad():
            mod.timestep_embed.weight.fill_(2.0)

        delta_gate2 = mod._forward_stage2(x)

        # The difference should be: gate_diff * ctx_output
        # gate changed from 0 to 2, so context contribution changed by 2 * ctx_out
        ctx_out = mod._apply_expert(ctx_expert, x, x.dtype) * mod.scalar * mod.scale
        expected_diff = 2.0 * ctx_out
        actual_diff = delta_gate2 - delta_zero_gate
        self.assertTrue(
            torch.allclose(actual_diff, expected_diff, atol=1e-4),
            "Context expert contribution should scale linearly with gate value"
        )

    # ------------------------------------------------------------------
    # Paper: multi-scale config [n1=8, n2=4, n3=2, n4=1]
    # ------------------------------------------------------------------

    def test_multi_scale_expert_count(self):
        """Paper Figure 3: 4 scales with n1=8, n2=4, n3=2, n4=1."""
        mod = TSMModule(
            "test", self.linear, lora_dim=4, alpha=4,
            tsm_n_scales=[8, 4, 2, 1], tsm_num_timesteps=1000,
        )
        self.assertEqual(len(mod.experts), 4)
        self.assertEqual(len(mod.experts[0]), 8)
        self.assertEqual(len(mod.experts[1]), 4)
        self.assertEqual(len(mod.experts[2]), 2)
        self.assertEqual(len(mod.experts[3]), 1)
        # Total expert count = 8 + 4 + 2 + 1 = 15
        total = sum(len(s) for s in mod.experts)
        self.assertEqual(total, 15)

    def test_multi_scale_context_experts_count(self):
        """Router outputs m-1 gates (one per context expert scale)."""
        mod = TSMModule(
            "test", self.linear, lora_dim=4, alpha=4,
            tsm_n_scales=[8, 4, 2, 1], tsm_num_timesteps=1000,
        )
        # m=4 scales → m-1=3 context experts → router outputs 3 gates
        self.assertEqual(mod.router_fc.out_features, 3)
        self.assertEqual(mod.timestep_embed.weight.shape[1], 3)

    # ------------------------------------------------------------------
    # Paper: zero initialization of all matrix B
    # ------------------------------------------------------------------

    def test_zero_init_B_matrices(self):
        """Paper: 'employ zero initialization for all matrix B' (up weights)."""
        mod = TSMModule(
            "test", self.linear, lora_dim=4, alpha=4,
            tsm_n_scales=[8, 1], tsm_num_timesteps=1000,
        )
        for scale_experts in mod.experts:
            for expert in scale_experts:
                self.assertTrue(
                    torch.all(expert["up"].weight == 0),
                    "B (up weight) should be zero-initialized per paper"
                )

    # ------------------------------------------------------------------
    # Paper: Stage 2 freezes experts, trains only router
    # ------------------------------------------------------------------

    def test_stage2_only_router_trains(self):
        """Paper: 'freeze them and only learn the parameters of router G(z_t,t)'."""
        mod = TSMModule(
            "test", self.linear, lora_dim=4, alpha=4,
            tsm_n_scales=[8, 1], tsm_num_timesteps=1000,
            tsm_stage=2,
        )
        # All expert params frozen
        for scale_experts in mod.experts:
            for expert in scale_experts:
                for p in expert.parameters():
                    self.assertFalse(p.requires_grad)
        # Router params trainable
        self.assertTrue(mod.router_fc.weight.requires_grad)
        self.assertTrue(mod.router_fc.bias.requires_grad)
        self.assertTrue(mod.timestep_embed.weight.requires_grad)


class TSMNumericalVerificationTests(unittest.TestCase):
    """Numerical verification of paper equations with exact expected values.

    These tests set weights to known values and verify the implementation
    produces exactly the expected outputs per the paper's equations.
    """

    def setUp(self):
        self.dim = 8
        self.lora_dim = 4
        self.linear = nn.Linear(self.dim, self.dim, bias=False)
        # Zero out original weight so only LoRA delta contributes
        nn.init.zeros_(self.linear.weight)
        self.T = 100  # Small T for easy manual computation

    def tearDown(self):
        clear_tsm_timestep()

    # ------------------------------------------------------------------
    # Eq. 5: ΔW = B_{i1}A_{i1} + Σ_{j=2}^m G_j ⊙ B_{ij}A_{ij}
    # ------------------------------------------------------------------

    def test_eq5_stage2_numerical(self):
        """Verify stage 2 output exactly matches Eq. 5 with known weights.

        ΔW*x = B_core A_core x + G_1 * B_ctx A_ctx x
        """
        mod = TSMModule(
            "test", self.linear, lora_dim=self.lora_dim, alpha=self.lora_dim,
            tsm_n_scales=[4, 1], tsm_num_timesteps=self.T,
            tsm_stage=2,
        )

        # Set core expert (scale 0, expert at t=50) to known values
        # t=50, T=100, n=4: i = ceil(50/100*4) = ceil(2.0) = 2 → idx=1
        core_expert = mod.experts[0][1]
        with torch.no_grad():
            # A_core (down): identity-like mapping from dim to lora_dim
            core_expert["down"].weight.copy_(
                torch.eye(self.lora_dim, self.dim)  # (lora_dim, dim)
            )
            # B_core (up): specific values
            core_expert["up"].weight.copy_(
                torch.ones(self.dim, self.lora_dim) * 0.5  # (dim, lora_dim)
            )

        # Set context expert (scale 1, expert 0 since n=1) to known values
        ctx_expert = mod.experts[1][0]
        with torch.no_grad():
            ctx_expert["down"].weight.copy_(
                torch.eye(self.lora_dim, self.dim) * 2.0
            )
            ctx_expert["up"].weight.copy_(
                torch.ones(self.dim, self.lora_dim) * 0.25
            )

        # Set router to produce gate = 3.0
        # G = FC(z_t) + ε(t) = 3.0
        # With z_t = 0 (from input): FC(0) = bias. So set bias = 3.0.
        with torch.no_grad():
            mod.router_fc.weight.fill_(0.0)
            mod.router_fc.bias.fill_(3.0)
            mod.timestep_embed.weight.fill_(0.0)

        set_tsm_timestep(50)
        x = torch.ones(1, self.dim)

        # Expected per Eq. 5:
        # core_out = B_core @ A_core @ x = 0.5 * ones(8,4) @ eye(4,8) @ ones(8)
        # A_core @ x = eye(4,8) @ ones(8) = ones(4)
        # B_core @ ones(4) = 0.5 * ones(8,4) @ ones(4) = 0.5 * 4 * ones(8) = 2*ones(8)
        # ctx_out = B_ctx @ A_ctx @ x = 0.25 * ones(8,4) @ (2*eye(4,8)) @ ones(8)
        # A_ctx @ x = 2*eye(4,8) @ ones(8) = 2*ones(4)
        # B_ctx @ 2*ones(4) = 0.25 * ones(8,4) @ 2*ones(4) = 0.25 * 2 * 4 * ones(8) = 2*ones(8)
        # G = 3.0
        # delta = core_out + G * ctx_out = 2 + 3*2 = 8
        # With scalar=1.0 and scale=alpha/lora_dim = 4/4 = 1.0
        # final = 8 * 1.0 * 1.0 = 8*ones(8)
        expected = torch.ones(1, self.dim) * 8.0

        delta = mod._forward_stage2(x)
        self.assertTrue(
            torch.allclose(delta, expected, atol=1e-4),
            f"Stage 2 output should match Eq. 5.\nExpected:\n{expected}\nGot:\n{delta}"
        )

    def test_eq5_zero_gate_returns_core_only(self):
        """When gate = 0, Eq. 5 reduces to just the core expert output."""
        mod = TSMModule(
            "test", self.linear, lora_dim=self.lora_dim, alpha=self.lora_dim,
            tsm_n_scales=[4, 1], tsm_num_timesteps=self.T,
            tsm_stage=2,
        )

        # Set core expert to non-zero, context to different non-zero
        core_expert = mod.experts[0][1]  # t=50
        ctx_expert = mod.experts[1][0]
        with torch.no_grad():
            core_expert["down"].weight.fill_(1.0)
            core_expert["up"].weight.fill_(1.0)
            ctx_expert["down"].weight.fill_(1.0)
            ctx_expert["up"].weight.fill_(1.0)

        # Gate = 0 (zero router)
        with torch.no_grad():
            mod.router_fc.weight.fill_(0.0)
            mod.router_fc.bias.fill_(0.0)
            mod.timestep_embed.weight.fill_(0.0)

        set_tsm_timestep(50)
        x = torch.randn(2, self.dim)
        delta = mod._forward_stage2(x)

        # Expected: only core expert (scale 0, expert 1)
        core_down_w = core_expert["down"].weight.to(x.dtype)
        core_up_w = core_expert["up"].weight.to(x.dtype)
        mid = F.linear(x, core_down_w, None)
        core_out = F.linear(mid, core_up_w, None)
        expected = core_out * mod.scalar * mod.scale

        self.assertTrue(
            torch.allclose(delta, expected, atol=1e-5),
            "Zero gate should produce core-only output"
        )

    # ------------------------------------------------------------------
    # Eq. 6: G(z_t, t) = F(z_t) + ε(t) — numerical verification
    # ------------------------------------------------------------------

    def test_eq6_gate_numerical(self):
        """Verify gate = FC(z) + embed(t) with exact known values."""
        mod = TSMModule(
            "test", self.linear, lora_dim=self.lora_dim, alpha=self.lora_dim,
            tsm_n_scales=[4, 2], tsm_num_timesteps=self.T,
        )

        # Set FC: weight = eye, bias = ones * 0.5
        # So FC(z) = z + 0.5 for each output
        num_ctx = len(mod.tsm_n_scales) - 1  # 1 context scale
        with torch.no_grad():
            # FC: in_dim=8, out_features=1
            mod.router_fc.weight.copy_(torch.ones(num_ctx, self.dim))
            mod.router_fc.bias.fill_(0.5)
            # Embedding: row t-1 = t * 0.1
            for i in range(self.T):
                mod.timestep_embed.weight[i] = float(i) * 0.1

        # z = ones(dim) → FC(z) = sum(ones(8)) + 0.5 = 8.5
        z = torch.ones(self.dim)
        t = 50
        gates = mod._compute_gates(z, z, t)

        # Expected: FC(z) + embed(49) = 8.5 + 49*0.1 = 8.5 + 4.9 = 13.4
        expected_gate = 8.5 + 4.9
        self.assertAlmostEqual(
            gates[0].item(), expected_gate, places=4,
            msg=f"Gate should be {expected_gate}, got {gates[0].item()}"
        )

    def test_eq6_timestep_only_changes_gate(self):
        """Changing timestep while keeping z fixed changes gate via ε(t)."""
        mod = TSMModule(
            "test", self.linear, lora_dim=self.lora_dim, alpha=self.lora_dim,
            tsm_n_scales=[4, 1], tsm_num_timesteps=self.T,
        )
        with torch.no_grad():
            mod.router_fc.weight.fill_(0.0)
            mod.router_fc.bias.fill_(0.0)
            # Set different embeddings for different timesteps
            for i in range(self.T):
                mod.timestep_embed.weight[i] = float(i)

        z = torch.zeros(self.dim)
        gate_t10 = mod._compute_gates(z, z, 10)[0].item()
        gate_t90 = mod._compute_gates(z, z, 90)[0].item()

        # embed(9) = 9, embed(89) = 89
        self.assertAlmostEqual(gate_t10, 9.0, places=4)
        self.assertAlmostEqual(gate_t90, 89.0, places=4)
        self.assertNotAlmostEqual(gate_t10, gate_t90)

    def test_eq6_feature_changes_gate(self):
        """Changing z_t while keeping timestep fixed changes gate via F(z_t)."""
        mod = TSMModule(
            "test", self.linear, lora_dim=self.lora_dim, alpha=self.lora_dim,
            tsm_n_scales=[4, 1], tsm_num_timesteps=self.T,
        )
        with torch.no_grad():
            mod.router_fc.weight.fill_(1.0)
            mod.router_fc.bias.fill_(0.0)
            mod.timestep_embed.weight.fill_(0.0)

        z1 = torch.zeros(self.dim)
        z2 = torch.ones(self.dim)
        gate_z1 = mod._compute_gates(z1, z1, 50)[0].item()
        gate_z2 = mod._compute_gates(z2, z2, 50)[0].item()

        # FC(zeros) = 0, FC(ones) = sum(ones(8)) = 8
        self.assertAlmostEqual(gate_z1, 0.0, places=4)
        self.assertAlmostEqual(gate_z2, 8.0, places=4)

    # ------------------------------------------------------------------
    # Eq. 7: i_j = ⌈t/T · n_j⌉ — exhaustive boundary verification
    # ------------------------------------------------------------------

    def test_eq7_all_experts_activated(self):
        """Every expert is activated by at least one timestep."""
        n_scales = [8, 4, 2, 1]
        mod = TSMModule(
            "test", self.linear, lora_dim=self.lora_dim, alpha=self.lora_dim,
            tsm_n_scales=n_scales, tsm_num_timesteps=self.T,
        )

        for scale_idx, n_j in enumerate(n_scales):
            activated = set()
            for t in range(1, self.T + 1):
                idx = mod._get_expert_index(t, scale_idx)
                activated.add(idx)
            self.assertEqual(
                len(activated), n_j,
                f"Scale {scale_idx} (n={n_j}): only {len(activated)}/{n_j} experts activated"
            )
            self.assertEqual(activated, set(range(n_j)))

    def test_eq7_interval_coverage(self):
        """Each expert covers exactly its expected interval of timesteps."""
        T = 100
        n_j = 4
        mod = TSMModule(
            "test", self.linear, lora_dim=self.lora_dim, alpha=self.lora_dim,
            tsm_n_scales=[n_j, 1], tsm_num_timesteps=T,
        )

        # Expert 0 should cover t ∈ [1, T/n_j] = [1, 25]
        for t in range(1, 26):
            self.assertEqual(
                mod._get_expert_index(t, 0), 0,
                f"t={t} should map to expert 0"
            )
        # Expert 1 should cover t ∈ [26, 50]
        for t in range(26, 51):
            self.assertEqual(
                mod._get_expert_index(t, 0), 1,
                f"t={t} should map to expert 1"
            )

    # ------------------------------------------------------------------
    # get_diff_weight vs bypass forward consistency
    # ------------------------------------------------------------------

    def test_diff_weight_matches_bypass_linear(self):
        """For linear layers, get_diff_weight merged result matches bypass forward.

        rebuild: org(x) + op(x, W + ΔW) should equal bypass: org(x) + delta(x)
        """
        mod = TSMModule(
            "test", self.linear, lora_dim=self.lora_dim, alpha=self.lora_dim,
            tsm_n_scales=[4, 1], tsm_num_timesteps=self.T,
            bypass_mode=True, tsm_stage=1,
        )

        # Set non-zero weights
        for scale_experts in mod.experts:
            for expert in scale_experts:
                nn.init.normal_(expert["down"].weight, std=0.1)
                nn.init.normal_(expert["up"].weight, std=0.1)

        set_tsm_timestep(50)
        x = torch.randn(2, self.dim)

        # Bypass forward
        bypass_out = mod.bypass_forward_diff(x, scale=1.0)

        # Rebuild via get_diff_weight
        diff_weight, _ = mod.get_diff_weight(multiplier=1.0, device=x.device)
        rebuild_delta = F.linear(x, diff_weight.to(x.dtype), None)

        self.assertTrue(
            torch.allclose(bypass_out, rebuild_delta, atol=1e-5),
            f"Bypass and rebuild deltas should match for linear.\n"
            f"Max diff: {(bypass_out - rebuild_delta).abs().max().item()}"
        )

    def test_diff_weight_matches_bypass_stage2(self):
        """Stage 2 get_diff_weight matches bypass forward for linear."""
        mod = TSMModule(
            "test", self.linear, lora_dim=self.lora_dim, alpha=self.lora_dim,
            tsm_n_scales=[4, 1], tsm_num_timesteps=self.T,
            bypass_mode=True, tsm_stage=2,
        )

        for scale_experts in mod.experts:
            for expert in scale_experts:
                nn.init.normal_(expert["down"].weight, std=0.1)
                nn.init.normal_(expert["up"].weight, std=0.1)

        # Set non-zero router for non-trivial gating
        with torch.no_grad():
            mod.router_fc.bias.fill_(0.5)
            mod.timestep_embed.weight.fill_(0.3)

        set_tsm_timestep(50)
        x = torch.randn(2, self.dim)

        # Bypass forward
        bypass_out = mod.bypass_forward_diff(x, scale=1.0)

        # get_diff_weight uses dummy zeros for z_t, but bypass uses actual input.
        # The gate values will differ because:
        # - bypass: FC(actual_x) + embed(t)
        # - rebuild: FC(zeros) + embed(t)
        # So they WON'T match when router_fc.weight is non-zero.
        # Only when router_fc.weight = 0 will they match.
        # Set router_fc.weight = 0 to make them comparable
        with torch.no_grad():
            mod.router_fc.weight.fill_(0.0)

        bypass_out = mod.bypass_forward_diff(x, scale=1.0)
        diff_weight, _ = mod.get_diff_weight(multiplier=1.0, device=x.device)
        rebuild_delta = F.linear(x, diff_weight.to(x.dtype), None)

        self.assertTrue(
            torch.allclose(bypass_out, rebuild_delta, atol=1e-5),
            f"Stage 2 bypass/rebuild should match when FC weight=0 (gate=embed only).\n"
            f"Max diff: {(bypass_out - rebuild_delta).abs().max().item()}"
        )

    # ------------------------------------------------------------------
    # Multi-scale stage 1 gradient verification
    # ------------------------------------------------------------------

    def test_multi_scale_stage1_all_scales_get_gradients(self):
        """With [8, 4, 2, 1], all 4 scales' active experts get gradients."""
        mod = TSMModule(
            "test", self.linear, lora_dim=self.lora_dim, alpha=4,
            tsm_n_scales=[8, 4, 2, 1], tsm_num_timesteps=1000,
            tsm_stage=1,
        )
        x = torch.randn(2, self.dim, requires_grad=True)
        set_tsm_timestep(500)
        out = mod(x)
        loss = out.sum()
        loss.backward()

        for scale_idx in range(4):
            expert_idx = mod._get_expert_index(500, scale_idx)
            expert = mod.experts[scale_idx][expert_idx]
            has_grad = any(
                p.grad is not None and p.grad.abs().sum() > 0
                for p in expert.parameters()
            )
            self.assertTrue(
                has_grad,
                f"Scale {scale_idx} expert {expert_idx} should have gradients at t=500"
            )

    def test_inactive_experts_no_gradient(self):
        """Experts not matching the current timestep should have zero gradient."""
        mod = TSMModule(
            "test", self.linear, lora_dim=self.lora_dim, alpha=4,
            tsm_n_scales=[4, 1], tsm_num_timesteps=100,
            tsm_stage=1,
        )
        x = torch.randn(2, self.dim, requires_grad=True)
        # t=10: expert_idx = ceil(10/100*4) = ceil(0.4) = 1 → idx=0
        set_tsm_timestep(10)
        out = mod(x)
        loss = out.sum()
        loss.backward()

        # Expert 1, 2, 3 (not active at t=10) should have zero gradient
        for inactive_idx in [1, 2, 3]:
            expert = mod.experts[0][inactive_idx]
            for p in expert.parameters():
                if p.grad is not None:
                    self.assertTrue(
                        torch.all(p.grad == 0),
                        f"Inactive expert {inactive_idx} should have zero gradient"
                    )

    # ------------------------------------------------------------------
    # Stage 2 training: router weights change after optimizer step
    # ------------------------------------------------------------------

    def test_stage2_router_updates_with_optimizer(self):
        """After an optimizer step in stage 2, router weights change."""
        mod = TSMModule(
            "test", self.linear, lora_dim=self.lora_dim, alpha=4,
            tsm_n_scales=[4, 1], tsm_num_timesteps=self.T,
            tsm_stage=2,
        )
        # Set non-zero expert weights so output is non-trivial
        for scale_experts in mod.experts:
            for expert in scale_experts:
                nn.init.normal_(expert["down"].weight, std=0.1)
                nn.init.normal_(expert["up"].weight, std=0.1)

        embed_before = mod.timestep_embed.weight.data.clone()

        optimizer = torch.optim.Adam(
            [p for p in mod.parameters() if p.requires_grad], lr=0.01
        )

        set_tsm_timestep(50)
        x = torch.randn(2, self.dim)
        out = mod(x)
        loss = out.sum()
        loss.backward()
        optimizer.step()

        # Router weights should have changed
        self.assertFalse(
            torch.allclose(mod.timestep_embed.weight.data, embed_before, atol=1e-8),
            "Timestep embedding should change after optimizer step"
        )

        # Expert weights should NOT have changed (frozen)
        for scale_experts in mod.experts:
            for expert in scale_experts:
                for p in expert.parameters():
                    self.assertFalse(
                        p.requires_grad,
                        "Expert params should be frozen in stage 2"
                    )

    # ------------------------------------------------------------------
    # Serialization round-trip
    # ------------------------------------------------------------------

    def test_custom_state_dict_roundtrip(self):
        """custom_state_dict → load preserves expert and router weights.

        custom_state_dict bakes scalar into up weights, so the loaded
        up weights will differ from the original by the scalar factor.
        """
        mod = TSMModule(
            "test", self.linear, lora_dim=self.lora_dim, alpha=4,
            tsm_n_scales=[4, 2], tsm_num_timesteps=self.T,
            tsm_stage=1,
        )

        # Set known weights
        for scale_experts in mod.experts:
            for expert in scale_experts:
                nn.init.normal_(expert["down"].weight, std=0.1)
                nn.init.normal_(expert["up"].weight, std=0.1)
        with torch.no_grad():
            mod.router_fc.weight.fill_(0.5)
            mod.timestep_embed.weight.fill_(0.3)

        # Get state dict (custom_state_dict bakes scalar into up weights)
        sd = mod.custom_state_dict()

        # Create new module and load — state_dict keys match module param names
        mod2 = TSMModule(
            "test", self.linear, lora_dim=self.lora_dim, alpha=4,
            tsm_n_scales=[4, 2], tsm_num_timesteps=self.T,
            tsm_stage=1,
        )

        mod2.load_state_dict(sd, strict=False)

        # Verify down weights match directly (no scalar baking)
        for scale_idx in range(len(mod.tsm_n_scales)):
            for expert_idx in range(mod.tsm_n_scales[scale_idx]):
                orig_down = mod.experts[scale_idx][expert_idx]["down"].weight
                loaded_down = mod2.experts[scale_idx][expert_idx]["down"].weight
                self.assertTrue(
                    torch.allclose(orig_down, loaded_down, atol=1e-6),
                    f"Expert [{scale_idx}][{expert_idx}] down weight mismatch"
                )
                # up weight: custom_state_dict bakes scalar into up.
                # load_weight_hook resets scalar to 1.0, so loaded_up = orig_up * scalar
                orig_up = mod.experts[scale_idx][expert_idx]["up"].weight
                loaded_up = mod2.experts[scale_idx][expert_idx]["up"].weight
                scalar = mod.scalar.item() if isinstance(mod.scalar, nn.Parameter) else mod.scalar
                self.assertTrue(
                    torch.allclose(orig_up * scalar, loaded_up, atol=1e-6),
                    f"Expert [{scale_idx}][{expert_idx}] up weight mismatch"
                )

        # Verify router weights match
        self.assertTrue(
            torch.allclose(mod.router_fc.weight, mod2.router_fc.weight, atol=1e-6)
        )
        self.assertTrue(
            torch.allclose(mod.timestep_embed.weight, mod2.timestep_embed.weight, atol=1e-6)
        )

    def test_extract_state_dict_identifies_tsm(self):
        """extract_state_dict correctly identifies and parses TSM keys."""
        mod = TSMModule(
            "test", self.linear, lora_dim=self.lora_dim, alpha=4,
            tsm_n_scales=[4, 1], tsm_num_timesteps=self.T,
            tsm_stage=1,
        )
        sd = mod.custom_state_dict()
        prefixed = {f"test.{k}": v for k, v in sd.items()}

        result = TSMModule.extract_state_dict(prefixed, "test")
        self.assertIsNotNone(result, "extract_state_dict should return non-None")

    # ------------------------------------------------------------------
    # Conv2d numerical verification
    # ------------------------------------------------------------------

    def test_conv2d_stage2_numerical(self):
        """Stage 2 with Conv2d produces correct output shape and non-zero delta."""
        conv = nn.Conv2d(8, 8, (3, 3), stride=1, padding=1)
        nn.init.zeros_(conv.weight)
        nn.init.zeros_(conv.bias)

        mod = TSMModule(
            "test", conv, lora_dim=4, alpha=4,
            tsm_n_scales=[4, 1], tsm_num_timesteps=100,
            tsm_stage=2,
        )

        # Set core expert to non-zero
        with torch.no_grad():
            nn.init.ones_(mod.experts[0][0]["up"].weight)

        set_tsm_timestep(10)  # expert 0 for scale 0
        x = torch.randn(1, 8, 4, 4)
        out = mod(x)
        self.assertEqual(out.shape, (1, 8, 4, 4))

        # With zero gate, output should reflect core expert only
        self.assertGreater(
            out.abs().sum().item(), 0,
            "Conv2d stage 2 should produce non-zero output with non-zero expert weights"
        )

    def test_conv2d_bypass_matches_org_plus_delta(self):
        """Conv2d bypass forward = org_forward + lora_delta."""
        conv = nn.Conv2d(8, 8, (3, 3), stride=1, padding=1)
        nn.init.normal_(conv.weight, std=0.1)

        mod = TSMModule(
            "test", conv, lora_dim=4, alpha=4,
            tsm_n_scales=[4, 1], tsm_num_timesteps=100,
            bypass_mode=True, tsm_stage=1,
        )

        for scale_experts in mod.experts:
            for expert in scale_experts:
                nn.init.normal_(expert["down"].weight, std=0.1)
                nn.init.normal_(expert["up"].weight, std=0.1)

        set_tsm_timestep(50)
        x = torch.randn(1, 8, 4, 4)
        full_out = mod(x)
        org_out = conv(x)
        delta = mod.bypass_forward_diff(x)

        self.assertTrue(
            torch.allclose(full_out, org_out + delta, atol=1e-5),
            "Bypass forward should equal org_forward + delta"
        )


class TSMMultiScaleForwardTests(unittest.TestCase):
    """Tests for multi-scale forward behavior matching paper Figure 3."""

    def setUp(self):
        self.dim = 16
        self.lora_dim = 4
        self.linear = nn.Linear(self.dim, self.dim)
        self.T = 100

    def tearDown(self):
        clear_tsm_timestep()

    def test_paper_figure3_config(self):
        """Paper Figure 3: n1=8, n2=4, n3=2, n4=1 creates correct structure."""
        mod = TSMModule(
            "test", self.linear, lora_dim=4, alpha=4,
            tsm_n_scales=[8, 4, 2, 1], tsm_num_timesteps=self.T,
        )
        self.assertEqual(len(mod.experts), 4)
        self.assertEqual(sum(len(s) for s in mod.experts), 15)  # 8+4+2+1
        # Router outputs m-1=3 gates
        self.assertEqual(mod.router_fc.out_features, 3)
        self.assertEqual(mod.timestep_embed.weight.shape, (self.T, 3))

    def test_different_timesteps_select_different_core_experts(self):
        """Different timesteps should map to different core experts with n=8."""
        mod = TSMModule(
            "test", self.linear, lora_dim=4, alpha=4,
            tsm_n_scales=[8, 1], tsm_num_timesteps=self.T,
        )
        # t=1 → expert 0, t=50 → expert 4, t=100 → expert 7
        idx_t1 = mod._get_expert_index(1, 0)
        idx_t50 = mod._get_expert_index(50, 0)
        idx_t100 = mod._get_expert_index(100, 0)
        self.assertEqual(idx_t1, 0)
        self.assertNotEqual(idx_t1, idx_t50)
        self.assertNotEqual(idx_t50, idx_t100)

    def test_context_scale_always_same_expert(self):
        """With n=1 context scale, always selects expert 0."""
        mod = TSMModule(
            "test", self.linear, lora_dim=4, alpha=4,
            tsm_n_scales=[8, 1], tsm_num_timesteps=self.T,
        )
        for t in [1, 25, 50, 75, 100]:
            self.assertEqual(
                mod._get_expert_index(t, 1), 0,
                f"Context scale should always select expert 0, got {mod._get_expert_index(t, 1)} at t={t}"
            )

    def test_stage1_sum_all_scales(self):
        """Stage 1 sums contributions from all scales."""
        mod = TSMModule(
            "test", self.linear, lora_dim=4, alpha=4,
            tsm_n_scales=[4, 1], tsm_num_timesteps=self.T,
            tsm_stage=1,
        )
        # Make all experts produce different non-zero output
        for s, scale_experts in enumerate(mod.experts):
            for e, expert in enumerate(scale_experts):
                with torch.no_grad():
                    expert["up"].weight.fill_((s + 1) * (e + 1) * 0.01)

        set_tsm_timestep(50)
        x = torch.randn(2, self.dim)
        delta = mod._forward_stage1(x)

        # Compute expected: sum of all active scales
        expected = torch.zeros(2, self.dim)
        for scale_idx in range(len(mod.tsm_n_scales)):
            expert_idx = mod._get_expert_index(50, scale_idx)
            expert = mod.experts[scale_idx][expert_idx]
            down_w = expert["down"].weight.to(x.dtype)
            up_w = expert["up"].weight.to(x.dtype)
            mid = F.linear(x, down_w, None)
            out = F.linear(mid, up_w, None)
            expected = expected + out
        expected = expected * mod.scalar * mod.scale

        self.assertTrue(
            torch.allclose(delta, expected, atol=1e-5),
            "Stage 1 should sum all scales' expert contributions"
        )


class TSMPaperDefaultConfigTests(unittest.TestCase):
    """Tests verifying the paper's default configuration works correctly."""

    def tearDown(self):
        clear_tsm_timestep()

    def test_default_config_domain_adaptation(self):
        """Paper's domain adaptation config: n1=8, n2=1, r=4, alpha=4."""
        linear = nn.Linear(64, 64)
        mod = TSMModule(
            "test", linear, lora_dim=4, alpha=4,
            tsm_n_scales=[8, 1], tsm_num_timesteps=1000,
            tsm_stage=1,
        )
        self.assertEqual(mod.tsm_n_scales, [8, 1])
        self.assertEqual(mod.lora_dim, 4)
        self.assertEqual(mod.scale, 1.0)  # alpha/lora_dim = 4/4 = 1.0

        # Stage 1 forward
        x = torch.randn(2, 64)
        set_tsm_timestep(500)
        out = mod(x)
        self.assertEqual(out.shape, (2, 64))

        # Switch to stage 2
        mod.set_stage(2)
        out2 = mod(x)
        self.assertEqual(out2.shape, (2, 64))

    def test_default_config_model_distillation(self):
        """Paper's distillation config: n1=4, n2=1, r=64, alpha=8."""
        linear = nn.Linear(320, 320)
        mod = TSMModule(
            "test", linear, lora_dim=64, alpha=8,
            tsm_n_scales=[4, 1], tsm_num_timesteps=1000,
            tsm_stage=1,
        )
        self.assertEqual(mod.tsm_n_scales, [4, 1])
        self.assertEqual(mod.lora_dim, 64)
        self.assertAlmostEqual(mod.scale, 8.0 / 64.0)  # alpha/lora_dim

        x = torch.randn(2, 320)
        set_tsm_timestep(500)
        out = mod(x)
        self.assertEqual(out.shape, (2, 320))

    def test_paper_ablation_n_values(self):
        """Paper Table 8: n=2,4,8,16 all work with r=4."""
        for n in [2, 4, 8, 16]:
            linear = nn.Linear(16, 16)
            mod = TSMModule(
                "test", linear, lora_dim=4, alpha=4,
                tsm_n_scales=[n, 1], tsm_num_timesteps=1000,
                tsm_stage=1,
            )
            self.assertEqual(len(mod.experts[0]), n)
            x = torch.randn(2, 16)
            set_tsm_timestep(500)
            out = mod(x)
            self.assertEqual(out.shape, (2, 16))


if __name__ == "__main__":
    unittest.main()
