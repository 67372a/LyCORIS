"""Unit tests for TSM ComfyUI loader (comfyui_nodes/tsm_loader.py).

Tests the core logic that doesn't require ComfyUI imports:
- Expert index computation
- Checkpoint key parsing and grouping
- TSM adapter forward pass
- Key mapping
"""

import math
import sys
import unittest
from collections import defaultdict
from unittest.mock import MagicMock

import torch
import torch.nn as nn
import torch.nn.functional as F

# Mock ComfyUI dependencies before importing tsm_loader
_mock_modules = [
    "folder_paths",
    "comfy",
    "comfy.lora",
    "comfy.patcher_extension",
    "comfy.utils",
    "comfy.weight_adapter",
]
_original_modules = {}
for mod_name in _mock_modules:
    _original_modules[mod_name] = sys.modules.get(mod_name)
    sys.modules[mod_name] = MagicMock()

try:
    from comfyui_nodes.tsm_loader import (
        _expert_index,
        _TSMAdapter,
        _group_tsm_keys,
        _parse_expert_structure,
        _parse_module_to_model_key,
        _convert_kohya_unet_path,
    )
finally:
    # Restore original modules
    for mod_name, original in _original_modules.items():
        if original is None:
            sys.modules.pop(mod_name, None)
        else:
            sys.modules[mod_name] = original


class ExpertIndexTests(unittest.TestCase):
    """Tests for _expert_index (paper Eq. 7: i_j = ceil(t/T * n_j))."""

    def test_basic_indexing_n8(self):
        """With n=8, T=1000: verify correct expert selection."""
        T = 1000
        n = 8
        # t=1 → ceil(0.008) = 1 → idx=0
        self.assertEqual(_expert_index(1, T, n), 0)
        # t=125 → ceil(1.0) = 1 → idx=0
        self.assertEqual(_expert_index(125, T, n), 0)
        # t=126 → ceil(1.008) = 2 → idx=1
        self.assertEqual(_expert_index(126, T, n), 1)
        # t=500 → ceil(4.0) = 4 → idx=3
        self.assertEqual(_expert_index(500, T, n), 3)
        # t=1000 → ceil(8.0) = 8 → idx=7
        self.assertEqual(_expert_index(1000, T, n), 7)

    def test_single_expert_n1(self):
        """With n=1, always returns expert 0."""
        for t in [1, 500, 1000]:
            self.assertEqual(_expert_index(t, 1000, 1), 0)

    def test_all_experts_activated(self):
        """Every expert is activated by at least one timestep."""
        T = 100
        for n in [2, 4, 8]:
            activated = set()
            for t in range(1, T + 1):
                activated.add(_expert_index(t, T, n))
            self.assertEqual(activated, set(range(n)))

    def test_boundary_values(self):
        """Edge cases: t=0, t=T, T=1."""
        self.assertEqual(_expert_index(0, 1000, 8), 0)  # t=0 → clamped
        self.assertEqual(_expert_index(1000, 1000, 8), 7)
        self.assertEqual(_expert_index(1, 1, 1), 0)


class GroupTSMKeysTests(unittest.TestCase):
    """Tests for _group_tsm_keys checkpoint parsing."""

    def _make_state_dict(self, module_name, n_scales=[4, 1]):
        """Create a mock TSM state dict for testing."""
        sd = {}
        in_dim, lora_dim, out_dim = 8, 4, 8
        T = 100

        for scale_idx, n_experts in enumerate(n_scales):
            for expert_idx in range(n_experts):
                prefix = f"{module_name}.experts.{scale_idx}.{expert_idx}"
                sd[f"{prefix}.down.weight"] = torch.randn(lora_dim, in_dim)
                sd[f"{prefix}.up.weight"] = torch.randn(out_dim, lora_dim)

        num_context = len(n_scales) - 1
        sd[f"{module_name}.router_fc.weight"] = torch.randn(num_context, in_dim)
        sd[f"{module_name}.router_fc.bias"] = torch.randn(num_context)
        sd[f"{module_name}.timestep_embed.weight"] = torch.randn(T, num_context)
        sd[f"{module_name}.alpha"] = torch.tensor(4.0)

        return sd

    def test_group_single_module(self):
        """Groups keys from a single TSM module correctly."""
        sd = self._make_state_dict("lora_unet_blocks_0_attn_to_q")
        groups = _group_tsm_keys(sd)
        self.assertIn("lora_unet_blocks_0_attn_to_q", groups)
        self.assertEqual(len(groups), 1)

    def test_group_multiple_modules(self):
        """Groups keys from multiple TSM modules correctly."""
        sd = {}
        sd.update(self._make_state_dict("lora_unet_blocks_0_attn_to_q"))
        sd.update(self._make_state_dict("lora_unet_blocks_0_attn_to_k"))
        groups = _group_tsm_keys(sd)
        self.assertEqual(len(groups), 2)
        self.assertIn("lora_unet_blocks_0_attn_to_q", groups)
        self.assertIn("lora_unet_blocks_0_attn_to_k", groups)

    def test_group_ignores_non_tsm_keys(self):
        """Non-TSM keys (no .experts. marker) are ignored."""
        sd = self._make_state_dict("lora_unet_blocks_0_attn_to_q")
        sd["lora_unet_blocks_0_attn_to_v.lora_up.weight"] = torch.randn(8, 4)
        groups = _group_tsm_keys(sd)
        self.assertEqual(len(groups), 1)


class ParseExpertStructureTests(unittest.TestCase):
    """Tests for _parse_expert_structure."""

    def test_parse_two_scales(self):
        """Correctly identifies [4, 1] structure."""
        module_data = {}
        for s in range(4):
            module_data[f"experts.0.{s}.down.weight"] = torch.randn(4, 8)
            module_data[f"experts.0.{s}.up.weight"] = torch.randn(8, 4)
        module_data[f"experts.1.0.down.weight"] = torch.randn(4, 8)
        module_data[f"experts.1.0.up.weight"] = torch.randn(8, 4)
        self.assertEqual(_parse_expert_structure(module_data), [4, 1])

    def test_parse_four_scales(self):
        """Correctly identifies [8, 4, 2, 1] structure."""
        module_data = {}
        for s, n in enumerate([8, 4, 2, 1]):
            for e in range(n):
                module_data[f"experts.{s}.{e}.down.weight"] = torch.randn(4, 8)
                module_data[f"experts.{s}.{e}.up.weight"] = torch.randn(8, 4)
        self.assertEqual(_parse_expert_structure(module_data), [8, 4, 2, 1])

    def test_parse_no_experts(self):
        """Returns None when no expert keys present."""
        module_data = {"router_fc.weight": torch.randn(1, 8)}
        self.assertIsNone(_parse_expert_structure(module_data))


class TSMAdapterTests(unittest.TestCase):
    """Tests for _TSMAdapter forward computation."""

    def _make_adapter(self, n_scales=[4, 1], in_dim=8, lora_dim=4, T=100):
        """Create a test adapter with known dimensions."""
        experts = {}
        for scale_idx, n_experts in enumerate(n_scales):
            for expert_idx in range(n_experts):
                down = torch.randn(lora_dim, in_dim) * 0.01
                up = torch.randn(in_dim, lora_dim) * 0.01
                experts[(scale_idx, expert_idx)] = (down, up)

        num_context = len(n_scales) - 1
        router_fc_w = torch.zeros(num_context, in_dim)
        router_fc_b = torch.zeros(num_context)
        timestep_embed_w = torch.zeros(T, num_context)

        return _TSMAdapter(
            experts=experts,
            router_fc_w=router_fc_w,
            router_fc_b=router_fc_b,
            timestep_embed_w=timestep_embed_w,
            scale=1.0,
            n_scales=n_scales,
            num_timesteps=T,
        )

    def test_forward_shape(self):
        """Adapter output has correct shape."""
        adapter = self._make_adapter()
        adapter.current_timestep = 50
        x = torch.randn(2, 8)
        out = adapter.h(x, None)
        self.assertEqual(out.shape, (2, 8))

    def test_zero_init_produces_zero(self):
        """Zero-initialized experts + router produce zero output."""
        adapter = self._make_adapter()
        adapter.current_timestep = 50
        # Zero out ALL experts (not just defaults)
        for key in adapter.experts:
            adapter.experts[key] = (
                torch.zeros_like(adapter.experts[key][0]),
                torch.zeros_like(adapter.experts[key][1]),
            )
        x = torch.randn(2, 8)
        out = adapter.h(x, None)
        self.assertTrue(torch.allclose(out, torch.zeros_like(out), atol=1e-6))

    def test_core_expert_contributes(self):
        """With non-zero core expert and zero gates, output = core only."""
        adapter = self._make_adapter()
        adapter.current_timestep = 50

        # Zero out ALL experts first
        for key in adapter.experts:
            adapter.experts[key] = (
                torch.zeros_like(adapter.experts[key][0]),
                torch.zeros_like(adapter.experts[key][1]),
            )

        # t=50, T=100, n=4: ceil(50/100*4) = ceil(2.0) = 2 → idx=1
        core_expert_idx = _expert_index(50, 100, 4)
        core_down = torch.eye(4, 8)
        core_up = torch.ones(8, 4) * 0.5
        adapter.experts[(0, core_expert_idx)] = (core_down, core_up)

        x = torch.ones(2, 8)
        out = adapter.h(x, None)

        # Expected: core_mid = eye(4,8) @ ones(8) = ones(4)
        # core_out = 0.5 * ones(8,4) @ ones(4) = 2 * ones(8)
        expected_core = F.linear(F.linear(x, core_down), core_up) * adapter.scale
        self.assertTrue(
            torch.allclose(out, expected_core, atol=1e-4),
            f"Output should equal core expert only when gates are zero"
        )

    def test_gate_scales_context_expert(self):
        """Non-zero gate should scale context expert contribution."""
        adapter = self._make_adapter()
        adapter.current_timestep = 50

        # Set context expert to non-zero
        ctx_down = torch.eye(4, 8)
        ctx_up = torch.ones(8, 4) * 0.25
        adapter.experts[(1, 0)] = (ctx_down, ctx_up)

        # Gate = 0 → no context contribution
        x = torch.ones(2, 8)
        out_zero_gate = adapter.h(x, None)

        # Gate = 2.0 → context contribution
        adapter.timestep_embed_w[49] = torch.tensor([2.0])  # t=50 → idx=49
        out_gate2 = adapter.h(x, None)

        ctx_delta = F.linear(F.linear(x, ctx_down), ctx_up) * adapter.scale
        expected_diff = 2.0 * ctx_delta
        actual_diff = out_gate2 - out_zero_gate
        self.assertTrue(
            torch.allclose(actual_diff, expected_diff, atol=1e-4),
            "Context contribution should scale linearly with gate value"
        )

    def test_different_timesteps_select_different_experts(self):
        """Different timesteps should activate different core experts."""
        adapter = self._make_adapter(n_scales=[4, 1], T=100)

        # Set each core expert to produce a distinct output
        for e in range(4):
            down = torch.randn(4, 8) * 0.01
            up = torch.zeros(8, 4)
            up[0, 0] = float(e + 1)  # Unique identifier
            adapter.experts[(0, e)] = (down, up)

        x = torch.randn(1, 8)

        results = {}
        for t in [10, 30, 60, 90]:
            adapter.current_timestep = t
            out = adapter.h(x, None)
            results[t] = out[0, 0].item()

        # At least some timesteps should produce different outputs
        unique_outputs = len(set(round(v, 4) for v in results.values()))
        self.assertGreater(unique_outputs, 1,
                           "Different timesteps should produce different outputs")

    def test_multiplier_affects_output(self):
        """Multiplier linearly scales the output."""
        adapter = self._make_adapter()
        adapter.current_timestep = 50
        adapter.experts[(0, 0)] = (
            torch.eye(4, 8),
            torch.ones(8, 4) * 0.1,
        )
        x = torch.randn(2, 8)

        adapter.multiplier = 1.0
        out1 = adapter.h(x, None)

        adapter.multiplier = 2.0
        out2 = adapter.h(x, None)

        self.assertTrue(
            torch.allclose(out2, out1 * 2.0, atol=1e-5),
            "Multiplier should linearly scale output"
        )

    def test_default_timestep_uses_max(self):
        """When current_timestep is None, defaults to num_timesteps."""
        adapter = self._make_adapter(T=100)
        adapter.current_timestep = None

        # Set last expert to non-zero
        adapter.experts[(0, 3)] = (  # n=4, t=100 → idx=3
            torch.randn(4, 8) * 0.01,
            torch.ones(8, 4) * 0.5,
        )
        x = torch.ones(1, 8)
        out = adapter.h(x, None)

        # Should be non-zero (using expert 3)
        self.assertGreater(out.abs().sum().item(), 0)

    def test_multi_scale_config(self):
        """Works with [8, 4, 2, 1] multi-scale config."""
        adapter = self._make_adapter(n_scales=[8, 4, 2, 1], T=1000)
        adapter.current_timestep = 500
        x = torch.randn(2, 8)
        out = adapter.h(x, None)
        self.assertEqual(out.shape, (2, 8))


class KeyMappingTests(unittest.TestCase):
    """Tests for LyCORIS → ComfyUI key mapping."""

    def test_unet_attention_q(self):
        key = _parse_module_to_model_key(
            "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q"
        )
        self.assertIsNotNone(key)
        self.assertIn("to_q", key)
        self.assertTrue(key.startswith("diffusion_model."))
        self.assertTrue(key.endswith(".weight"))

    def test_unet_attention_out(self):
        key = _parse_module_to_model_key(
            "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_out_0"
        )
        self.assertIsNotNone(key)
        self.assertIn("to_out", key)

    def test_text_encoder_q_proj(self):
        key = _parse_module_to_model_key(
            "lora_te1_text_model_encoder_layers_0_self_attn_q_proj"
        )
        self.assertIsNotNone(key)
        self.assertIn("q_proj", key)
        self.assertTrue(key.startswith("diffusion_model."))

    def test_unrecognized_key_returns_none(self):
        key = _parse_module_to_model_key("random_garbage_key")
        self.assertIsNone(key)

    def test_convert_kohya_unet_path_down_blocks(self):
        result = _convert_kohya_unet_path("down_blocks_0_attentions_0")
        self.assertEqual(result, "down_blocks.0.attentions.0")

    def test_convert_kohya_unet_path_to_q(self):
        result = _convert_kohya_unet_path("down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q")
        self.assertIn("to_q", result)

    def test_convert_kohya_unet_path_to_out(self):
        result = _convert_kohya_unet_path("down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_out_0")
        self.assertIn("to_out", result)


class TSMAdapterVsModuleTests(unittest.TestCase):
    """Cross-validate ComfyUI adapter against TSM module with identical weights.

    This is the definitive correctness test: given the same expert weights,
    router weights, and timestep, the ComfyUI adapter must produce the exact
    same output as the TSMModule._forward_stage2().
    """

    def test_adapter_matches_module_stage2(self):
        """ComfyUI adapter output == TSMModule stage 2 output for same weights."""
        from lycoris.modules.tsm import TSMModule, set_tsm_timestep as _mod_set_ts
        from lycoris.modules.tsm import clear_tsm_timestep as _mod_clear_ts

        in_dim = 16
        lora_dim = 4
        n_scales = [8, 1]
        T = 1000
        linear = nn.Linear(in_dim, in_dim, bias=False)
        nn.init.zeros_(linear.weight)

        # Create TSM module in stage 2
        mod = TSMModule(
            "test", linear, lora_dim=lora_dim, alpha=lora_dim,
            tsm_n_scales=n_scales, tsm_num_timesteps=T, tsm_stage=2,
        )

        # Set known expert weights
        torch.manual_seed(42)
        for scale_experts in mod.experts:
            for expert in scale_experts:
                nn.init.normal_(expert["down"].weight, std=0.1)
                nn.init.normal_(expert["up"].weight, std=0.1)

        # Set known router weights
        with torch.no_grad():
            mod.router_fc.weight.fill_(0.3)
            mod.router_fc.bias.fill_(0.1)
            for i in range(T):
                mod.timestep_embed.weight[i] = float(i) * 0.01

        # Extract weights from module into adapter format
        experts = {}
        for scale_idx, scale_experts in enumerate(mod.experts):
            for expert_idx, expert in enumerate(scale_experts):
                experts[(scale_idx, expert_idx)] = (
                    expert["down"].weight.detach().clone(),
                    expert["up"].weight.detach().clone(),
                )

        adapter = _TSMAdapter(
            experts=experts,
            router_fc_w=mod.router_fc.weight.detach().clone(),
            router_fc_b=mod.router_fc.bias.detach().clone(),
            timestep_embed_w=mod.timestep_embed.weight.detach().clone(),
            scale=lora_dim / lora_dim,  # alpha/lora_dim = 1.0
            n_scales=n_scales,
            num_timesteps=T,
            router_input_mode="input",
        )
        adapter.multiplier = 1.0

        # Test at multiple timesteps
        for t in [1, 100, 500, 1000]:
            x = torch.randn(2, in_dim)

            # Module output
            _mod_set_ts(t)
            mod_out = mod._forward_stage2(x)
            _mod_clear_ts()

            # Adapter output
            adapter.current_timestep = t
            adapter_out = adapter.h(x, None)

            self.assertTrue(
                torch.allclose(mod_out, adapter_out, atol=1e-4),
                f"t={t}: Adapter and module outputs diverge.\n"
                f"  Module max: {mod_out.abs().max():.6f}\n"
                f"  Adapter max: {adapter_out.abs().max():.6f}\n"
                f"  Diff max: {(mod_out - adapter_out).abs().max():.6f}"
            )


if __name__ == "__main__":
    unittest.main()
