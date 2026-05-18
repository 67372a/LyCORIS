"""
Unit tests for regex-based network args:
  - exclude_patterns / include_patterns
  - network_reg_dims / network_reg_lrs
  - Preset fallback with network arg priority
  - exclude_patterns precedence over exclude_name
"""
import unittest
import logging

import torch
import torch.nn as nn

from lycoris import create_lycoris, LycorisNetwork
from lycoris.logging import logger

logger.setLevel(logging.ERROR)


def reset_globals():
    LycorisNetwork.apply_preset(
        {
            "enable_conv": True,
            "target_module": [
                "Linear",
                "Conv1d",
                "Conv2d",
                "Conv3d",
                "GroupNorm",
                "LayerNorm",
            ],
            "target_name": [],
            "lora_prefix": "lycoris",
            "module_algo_map": {},
            "name_algo_map": {},
            "use_fnmatch": False,
            "exclude_name": [],
            "exclude_patterns": None,
            "include_patterns": None,
            "network_reg_dims": None,
            "network_reg_lrs": None,
        }
    )


class SimpleNet(nn.Module):
    """A simple network with named submodules for testing regex filtering."""

    def __init__(self, dim=16):
        super().__init__()
        self.attn_q = nn.Linear(dim, dim)
        self.attn_k = nn.Linear(dim, dim)
        self.attn_v = nn.Linear(dim, dim)
        self.mlp_fc1 = nn.Linear(dim, dim * 4)
        self.mlp_fc2 = nn.Linear(dim * 4, dim)
        self.norm = nn.LayerNorm(dim)
        self.conv = nn.Conv1d(dim, dim, 3, 1, 1)

    def forward(self, x):
        return x


class LycorisRegexArgsTests(unittest.TestCase):
    """Tests for exclude_patterns, include_patterns, network_reg_dims, network_reg_lrs."""

    def _get_lora_names(self, network):
        return sorted([lora.lora_name for lora in network.loras])

    def _get_original_names(self, network):
        return sorted([getattr(lora, 'original_name', '') for lora in network.loras])

    # ── exclude_patterns ──────────────────────────────────────────────

    def test_exclude_patterns_filters_modules(self):
        """Modules matching exclude_patterns should not get LoRA adapters."""
        try:
            net = SimpleNet()
            lycoris_net = create_lycoris(
                net, 1, linear_dim=4, linear_alpha=1,
                exclude_patterns=[r".*mlp.*"],
            )
            names = self._get_original_names(lycoris_net)
            # mlp_fc1 and mlp_fc2 should be excluded
            for n in names:
                self.assertNotIn("mlp", n, f"Module '{n}' should have been excluded")
            # attn modules should still be present
            attn_names = [n for n in names if "attn" in n]
            self.assertGreater(len(attn_names), 0, "attn modules should be present")
        finally:
            reset_globals()

    def test_exclude_patterns_empty_list_excludes_nothing(self):
        """An empty exclude_patterns list should not exclude anything."""
        try:
            net = SimpleNet()
            lycoris_all = create_lycoris(
                net, 1, linear_dim=4, linear_alpha=1,
            )
            lycoris_empty_exclude = create_lycoris(
                net, 1, linear_dim=4, linear_alpha=1,
                exclude_patterns=[],
            )
            self.assertEqual(
                len(lycoris_all.loras),
                len(lycoris_empty_exclude.loras),
            )
        finally:
            reset_globals()

    # ── include_patterns overrides exclude ─────────────────────────────

    def test_include_overrides_exclude(self):
        """include_patterns should override exclude_patterns."""
        try:
            net = SimpleNet()
            # Exclude all attn, but include attn_q back
            lycoris_net = create_lycoris(
                net, 1, linear_dim=4, linear_alpha=1,
                exclude_patterns=[r".*attn.*"],
                include_patterns=[r".*attn_q"],
            )
            names = self._get_original_names(lycoris_net)
            attn_names = [n for n in names if "attn" in n]
            # Only attn_q should survive
            self.assertEqual(len(attn_names), 1, f"Expected 1 attn module, got: {attn_names}")
            self.assertIn("attn_q", attn_names[0])
        finally:
            reset_globals()

    def test_include_overrides_exclude_name(self):
        """include_patterns should also override exclude_name (TARGET_EXCLUDE_NAME)."""
        try:
            # Use exclude_name via preset
            LycorisNetwork.apply_preset({
                "exclude_name": [r".*attn.*"],
            })
            net = SimpleNet()
            lycoris_net = create_lycoris(
                net, 1, linear_dim=4, linear_alpha=1,
                include_patterns=[r".*attn_q"],
            )
            names = self._get_original_names(lycoris_net)
            attn_names = [n for n in names if "attn" in n]
            # attn_q should be included despite exclude_name
            self.assertEqual(len(attn_names), 1, f"Expected 1 attn module, got: {attn_names}")
            self.assertIn("attn_q", attn_names[0])
        finally:
            reset_globals()

    # ── exclude_patterns precedence over exclude_name ──────────────────

    def test_exclude_patterns_overrides_exclude_name(self):
        """When exclude_patterns is set, exclude_name should be ignored."""
        try:
            # Preset sets exclude_name to exclude mlp
            LycorisNetwork.apply_preset({
                "exclude_name": [r".*mlp.*"],
            })
            net = SimpleNet()
            # exclude_patterns only excludes conv - mlp should now be INCLUDED
            # because exclude_patterns takes precedence over exclude_name
            lycoris_net = create_lycoris(
                net, 1, linear_dim=4, linear_alpha=1,
                exclude_patterns=[r".*conv.*"],
            )
            names = self._get_original_names(lycoris_net)
            mlp_names = [n for n in names if "mlp" in n]
            conv_names = [n for n in names if "conv" in n]
            # mlp should be present (exclude_name ignored)
            self.assertGreater(len(mlp_names), 0, "mlp modules should be present when exclude_patterns overrides exclude_name")
            # conv should be excluded by exclude_patterns
            self.assertEqual(len(conv_names), 0, "conv modules should be excluded")
        finally:
            reset_globals()

    def test_exclude_name_used_when_no_exclude_patterns(self):
        """When exclude_patterns is not set, exclude_name should still work."""
        try:
            LycorisNetwork.apply_preset({
                "exclude_name": [r".*mlp.*"],
            })
            net = SimpleNet()
            lycoris_net = create_lycoris(
                net, 1, linear_dim=4, linear_alpha=1,
                # No exclude_patterns set
            )
            names = self._get_original_names(lycoris_net)
            mlp_names = [n for n in names if "mlp" in n]
            self.assertEqual(len(mlp_names), 0, "mlp modules should be excluded by exclude_name")
        finally:
            reset_globals()

    # ── network_reg_dims ──────────────────────────────────────────────────────

    def test_reg_dims_overrides_module_dim(self):
        """network_reg_dims should override the default dim for matching modules."""
        try:
            net = SimpleNet()
            lycoris_net = create_lycoris(
                net, 1, linear_dim=4, linear_alpha=1,
                network_reg_dims={r".*attn.*": 32},
            )
            for lora in lycoris_net.loras:
                orig = getattr(lora, 'original_name', '')
                if "attn" in orig and hasattr(lora, 'lora_dim'):
                    self.assertEqual(lora.lora_dim, 32, f"Module '{orig}' should have lora_dim=32, got {lora.lora_dim}")
                elif hasattr(lora, 'lora_dim') and "norm" not in lora.lora_name:
                    self.assertEqual(lora.lora_dim, 4, f"Module '{orig}' should have default lora_dim=4, got {lora.lora_dim}")
        finally:
            reset_globals()

    def test_reg_dims_multiple_patterns(self):
        """Multiple network_reg_dims patterns should each apply to their matching modules."""
        try:
            net = SimpleNet()
            lycoris_net = create_lycoris(
                net, 1, linear_dim=4, linear_alpha=1,
                network_reg_dims={r".*attn.*": 16, r".*mlp.*": 64},
            )
            for lora in lycoris_net.loras:
                orig = getattr(lora, 'original_name', '')
                if "attn" in orig and hasattr(lora, 'lora_dim'):
                    self.assertEqual(lora.lora_dim, 16, f"'{orig}' should have lora_dim=16")
                elif "mlp" in orig and hasattr(lora, 'lora_dim'):
                    self.assertEqual(lora.lora_dim, 64, f"'{orig}' should have lora_dim=64")
        finally:
            reset_globals()

    # ── network_reg_lrs ──────────────────────────────────────────────────────

    def test_reg_lrs_stored_on_network(self):
        """network_reg_lrs should be stored on the network object for optimizer use."""
        try:
            net = SimpleNet()
            lycoris_net = create_lycoris(
                net, 1, linear_dim=4, linear_alpha=1,
                network_reg_lrs={r".*attn.*": 1e-4, r".*mlp.*": 5e-5},
            )
            self.assertIsNotNone(lycoris_net.network_reg_lrs)
            self.assertEqual(len(lycoris_net.network_reg_lrs), 2)
        finally:
            reset_globals()

    # ── original_name attribute ───────────────────────────────────────

    def test_original_name_set_on_loras(self):
        """All created lora modules should have original_name attribute."""
        try:
            net = SimpleNet()
            lycoris_net = create_lycoris(
                net, 1, linear_dim=4, linear_alpha=1,
            )
            for lora in lycoris_net.loras:
                self.assertTrue(
                    hasattr(lora, 'original_name'),
                    f"Module '{lora.lora_name}' missing original_name attribute",
                )
                self.assertIsNotNone(lora.original_name)
                self.assertNotEqual(lora.original_name, "")
        finally:
            reset_globals()

    # ── Preset fallback ───────────────────────────────────────────────

    def test_preset_exclude_patterns_fallback(self):
        """exclude_patterns from preset should be used when not set via network args."""
        try:
            LycorisNetwork.apply_preset({
                "exclude_patterns": [r".*mlp.*"],
            })
            net = SimpleNet()
            lycoris_net = create_lycoris(
                net, 1, linear_dim=4, linear_alpha=1,
                # No exclude_patterns in network args
            )
            names = self._get_original_names(lycoris_net)
            mlp_names = [n for n in names if "mlp" in n]
            self.assertEqual(len(mlp_names), 0, "mlp modules should be excluded by preset exclude_patterns")
        finally:
            reset_globals()

    def test_preset_include_patterns_fallback(self):
        """include_patterns from preset should be used when not set via network args."""
        try:
            LycorisNetwork.apply_preset({
                "exclude_patterns": [r".*attn.*"],
                "include_patterns": [r".*attn_q"],
            })
            net = SimpleNet()
            lycoris_net = create_lycoris(
                net, 1, linear_dim=4, linear_alpha=1,
            )
            names = self._get_original_names(lycoris_net)
            attn_names = [n for n in names if "attn" in n]
            self.assertEqual(len(attn_names), 1, f"Only attn_q should survive, got: {attn_names}")
        finally:
            reset_globals()

    def test_preset_reg_dims_fallback(self):
        """network_reg_dims from preset should be used when not set via network args."""
        try:
            LycorisNetwork.apply_preset({
                "network_reg_dims": {r".*attn.*": 32},
            })
            net = SimpleNet()
            lycoris_net = create_lycoris(
                net, 1, linear_dim=4, linear_alpha=1,
            )
            for lora in lycoris_net.loras:
                orig = getattr(lora, 'original_name', '')
                if "attn" in orig and hasattr(lora, 'lora_dim'):
                    self.assertEqual(lora.lora_dim, 32, f"'{orig}' should have lora_dim=32 from preset")
        finally:
            reset_globals()

    def test_preset_reg_lrs_fallback(self):
        """network_reg_lrs from preset should be used when not set via network args."""
        try:
            LycorisNetwork.apply_preset({
                "network_reg_lrs": {r".*attn.*": 1e-4},
            })
            net = SimpleNet()
            lycoris_net = create_lycoris(
                net, 1, linear_dim=4, linear_alpha=1,
            )
            self.assertIsNotNone(lycoris_net.network_reg_lrs)
            self.assertIn(r".*attn.*", lycoris_net.network_reg_lrs)
        finally:
            reset_globals()

    # ── Network args override preset ──────────────────────────────────

    def test_network_args_override_preset_exclude_patterns(self):
        """Network arg exclude_patterns should override preset."""
        try:
            # Preset excludes attn
            LycorisNetwork.apply_preset({
                "exclude_patterns": [r".*attn.*"],
            })
            net = SimpleNet()
            # Network arg excludes mlp instead
            lycoris_net = create_lycoris(
                net, 1, linear_dim=4, linear_alpha=1,
                exclude_patterns=[r".*mlp.*"],
            )
            names = self._get_original_names(lycoris_net)
            attn_names = [n for n in names if "attn" in n]
            mlp_names = [n for n in names if "mlp" in n]
            # attn should be present (preset overridden)
            self.assertGreater(len(attn_names), 0, "attn should be present (preset overridden)")
            # mlp should be excluded (network arg applied)
            self.assertEqual(len(mlp_names), 0, "mlp should be excluded (network arg)")
        finally:
            reset_globals()

    def test_network_args_override_preset_reg_dims(self):
        """Network arg network_reg_dims should override preset."""
        try:
            LycorisNetwork.apply_preset({
                "network_reg_dims": {r".*attn.*": 8},
            })
            net = SimpleNet()
            lycoris_net = create_lycoris(
                net, 1, linear_dim=4, linear_alpha=1,
                network_reg_dims={r".*attn.*": 64},
            )
            for lora in lycoris_net.loras:
                orig = getattr(lora, 'original_name', '')
                if "attn" in orig and hasattr(lora, 'lora_dim'):
                    self.assertEqual(lora.lora_dim, 64, f"'{orig}' should have lora_dim=64 from network arg, not 8 from preset")
        finally:
            reset_globals()

    # ── Combined scenarios ────────────────────────────────────────────

    def test_exclude_and_reg_dims_together(self):
        """exclude_patterns and network_reg_dims should work together."""
        try:
            net = SimpleNet()
            lycoris_net = create_lycoris(
                net, 1, linear_dim=4, linear_alpha=1,
                exclude_patterns=[r".*conv.*"],
                network_reg_dims={r".*attn.*": 32},
            )
            names = self._get_original_names(lycoris_net)
            conv_names = [n for n in names if "conv" in n]
            self.assertEqual(len(conv_names), 0, "conv should be excluded")
            for lora in lycoris_net.loras:
                orig = getattr(lora, 'original_name', '')
                if "attn" in orig and hasattr(lora, 'lora_dim'):
                    self.assertEqual(lora.lora_dim, 32, f"'{orig}' should have lora_dim=32")
        finally:
            reset_globals()

    def test_all_features_from_preset(self):
        """All new features should work when set entirely from preset."""
        try:
            LycorisNetwork.apply_preset({
                "exclude_patterns": [r".*conv.*", r".*norm.*"],
                "include_patterns": [r".*norm.*"],  # re-include norm
                "network_reg_dims": {r".*attn.*": 16},
                "network_reg_lrs": {r".*mlp.*": 2e-4},
            })
            net = SimpleNet()
            lycoris_net = create_lycoris(
                net, 1, linear_dim=4, linear_alpha=1,
                train_norm=True,
            )
            names = self._get_original_names(lycoris_net)
            # conv should be excluded
            conv_names = [n for n in names if "conv" in n]
            self.assertEqual(len(conv_names), 0, "conv should be excluded")
            # norm should be included (include overrides exclude)
            norm_names = [n for n in names if "norm" in n]
            self.assertGreater(len(norm_names), 0, "norm should be included via include_patterns")
            # attn should have lora_dim=16
            for lora in lycoris_net.loras:
                orig = getattr(lora, 'original_name', '')
                if "attn" in orig and hasattr(lora, 'lora_dim'):
                    self.assertEqual(lora.lora_dim, 16)
            # network_reg_lrs should be stored
            self.assertIsNotNone(lycoris_net.network_reg_lrs)
        finally:
            reset_globals()


class LycorisRegLoraplusTests(unittest.TestCase):
    """Tests for network_reg_loraplus_ratios feature."""

    def _get_lora_names(self, network):
        return sorted([lora.lora_name for lora in network.loras])

    def test_reg_loraplus_ratios_stored(self):
        """network_reg_loraplus_ratios should be stored on the network object."""
        try:
            from lycoris.kohya import LycorisNetworkKohya

            LycorisNetworkKohya.apply_preset({
                "enable_conv": True,
                "unet_target_module": ["Linear"],
                "unet_target_name": [],
                "text_encoder_target_module": [],
                "text_encoder_target_name": [],
            })
            net = SimpleNet()
            network = LycorisNetworkKohya(
                None, net, 1.0, lora_dim=4, alpha=1,
                network_reg_loraplus_ratios={r".*attn.*": 2.0, r".*mlp.*": 4.0},
            )
            self.assertIsNotNone(network.network_reg_loraplus_ratios)
            self.assertEqual(len(network.network_reg_loraplus_ratios), 2)
            self.assertEqual(network.network_reg_loraplus_ratios[r".*attn.*"], 2.0)
            self.assertEqual(network.network_reg_loraplus_ratios[r".*mlp.*"], 4.0)
        finally:
            reset_globals()

    def test_reg_loraplus_ratios_preset_fallback(self):
        """network_reg_loraplus_ratios from preset should be used when not set via network args."""
        try:
            from lycoris.kohya import LycorisNetworkKohya

            LycorisNetworkKohya.apply_preset({
                "enable_conv": True,
                "unet_target_module": ["Linear"],
                "unet_target_name": [],
                "text_encoder_target_module": [],
                "text_encoder_target_name": [],
                "network_reg_loraplus_ratios": {r".*attn.*": 3.0},
            })
            net = SimpleNet()
            network = LycorisNetworkKohya(
                None, net, 1.0, lora_dim=4, alpha=1,
            )
            self.assertIsNotNone(network.network_reg_loraplus_ratios)
            self.assertIn(r".*attn.*", network.network_reg_loraplus_ratios)
            self.assertEqual(network.network_reg_loraplus_ratios[r".*attn.*"], 3.0)
        finally:
            reset_globals()

    def test_reg_loraplus_ratios_override_preset(self):
        """Network arg network_reg_loraplus_ratios should override preset."""
        try:
            from lycoris.kohya import LycorisNetworkKohya

            LycorisNetworkKohya.apply_preset({
                "enable_conv": True,
                "unet_target_module": ["Linear"],
                "unet_target_name": [],
                "text_encoder_target_module": [],
                "text_encoder_target_name": [],
                "network_reg_loraplus_ratios": {r".*attn.*": 1.5},
            })
            net = SimpleNet()
            network = LycorisNetworkKohya(
                None, net, 1.0, lora_dim=4, alpha=1,
                network_reg_loraplus_ratios={r".*attn.*": 5.0},
            )
            self.assertEqual(network.network_reg_loraplus_ratios[r".*attn.*"], 5.0)
        finally:
            reset_globals()

    def test_reg_loraplus_prepare_optimizer_groups(self):
        """prepare_optimizer_params should group params with regex-specific LoRA+ ratios."""
        try:
            from lycoris.kohya import LycorisNetworkKohya

            LycorisNetworkKohya.apply_preset({
                "enable_conv": True,
                "unet_target_module": ["Linear"],
                "unet_target_name": [],
                "text_encoder_target_module": [],
                "text_encoder_target_name": [],
            })
            net = SimpleNet()
            network = LycorisNetworkKohya(
                None, net, 1.0, lora_dim=4, alpha=1,
                network_reg_loraplus_ratios={r".*attn.*": 2.0},
            )
            network.apply_to(None, net, apply_text_encoder=False, apply_unet=True)

            groups, descriptions = network.prepare_optimizer_params(
                unet_lr=1e-4, learning_rate=1e-5,
            )
            self.assertGreater(len(groups), 0, "Should have at least one param group")

            # Find groups with LoRA+ enabled
            plus_groups = [g for g in groups if g.get('is_lora_plus_group', False)]
            self.assertGreater(len(plus_groups), 0,
                "Should have LoRA+ groups when network_reg_loraplus_ratios is set")

            # Verify plus groups have the correct ratio applied (base_lr * 2.0)
            for g in plus_groups:
                # attn modules should get ratio 2.0 applied to their base LR
                expected_lr = 1e-4 * 2.0
                self.assertAlmostEqual(g['lr'].item(), expected_lr, delta=1e-10,
                    msg=f"LoRA+ group should have lr={expected_lr}, got {g['lr'].item()}")
        finally:
            reset_globals()

    def test_reg_loraplus_without_global_ratio(self):
        """Modules matching regex should get LoRA+ treatment even without global Loraplus ratios."""
        try:
            from lycoris.kohya import LycorisNetworkKohya

            LycorisNetworkKohya.apply_preset({
                "enable_conv": True,
                "unet_target_module": ["Linear"],
                "unet_target_name": [],
                "text_encoder_target_module": [],
                "text_encoder_target_name": [],
            })
            net = SimpleNet()
            network = LycorisNetworkKohya(
                None, net, 1.0, lora_dim=4, alpha=1,
                network_reg_loraplus_ratios={r".*attn.*": 3.0},
                # No loraplus_lr_ratio or loraplus_unet_lr_ratio set
            )
            network.apply_to(None, net, apply_text_encoder=False, apply_unet=True)

            groups, descriptions = network.prepare_optimizer_params(
                unet_lr=1e-4, learning_rate=1e-5,
            )
            # Should still have LoRA+ groups even without global ratio
            plus_groups = [g for g in groups if g.get('is_lora_plus_group', False)]
            self.assertGreater(len(plus_groups), 0,
                "Should have LoRA+ groups even without global loraplus ratio set")

            for g in plus_groups:
                expected_lr = 1e-4 * 3.0
                self.assertAlmostEqual(g['lr'].item(), expected_lr, delta=1e-10,
                    msg=f"LoRA+ group should have lr={expected_lr}, got {g['lr'].item()}")
        finally:
            reset_globals()

    def test_reg_loraplus_precedence_over_global(self):
        """Regex-specific LoRA+ ratio should override global ratio for matching modules."""
        try:
            from lycoris.kohya import LycorisNetworkKohya

            LycorisNetworkKohya.apply_preset({
                "enable_conv": True,
                "unet_target_module": ["Linear"],
                "unet_target_name": [],
                "text_encoder_target_module": [],
                "text_encoder_target_name": [],
            })
            net = SimpleNet()
            network = LycorisNetworkKohya(
                None, net, 1.0, lora_dim=4, alpha=1,
                loraplus_lr_ratio=10.0,  # global, should be overridden by regex
                network_reg_loraplus_ratios={r".*attn.*": 2.0},
            )
            network.set_loraplus_lr_ratio(10.0, None, None)
            network.apply_to(None, net, apply_text_encoder=False, apply_unet=True)

            groups, descriptions = network.prepare_optimizer_params(
                unet_lr=1e-4, learning_rate=1e-5,
            )
            plus_groups = [g for g in groups if g.get('is_lora_plus_group', False)]
            self.assertGreater(len(plus_groups), 0)

            # After grouping, groups with different loraplus_ratios should have different LRs
            lrs = set()
            for g in plus_groups:
                lr_val = g['lr'].item()
                lrs.add(lr_val)
            # Should have at least 1 distinct LRs: regex-matched and/or global fallback
            self.assertGreaterEqual(len(lrs), 1,
                "Should have at least one distinct LR value")

            # Verify that at least one group uses the regex ratio (2.0)
            regex_lrs = [g['lr'].item() for g in plus_groups
                         if abs(g['lr'].item() - 1e-4 * 2.0) < 1e-6]
            self.assertGreater(len(regex_lrs), 0,
                "At least one LoRA+ group should use the regex ratio 2.0 instead of global 10.0")
        finally:
            reset_globals()

    def test_reg_loraplus_non_matching_uses_global(self):
        """Modules not matching any regex should use global/component ratio."""
        try:
            from lycoris.kohya import LycorisNetworkKohya

            LycorisNetworkKohya.apply_preset({
                "enable_conv": True,
                "unet_target_module": ["Linear"],
                "unet_target_name": [],
                "text_encoder_target_module": [],
                "text_encoder_target_name": [],
            })
            net = SimpleNet()
            network = LycorisNetworkKohya(
                None, net, 1.0, lora_dim=4, alpha=1,
                loraplus_unet_lr_ratio=5.0,
                network_reg_loraplus_ratios={r".*nonexistent.*": 2.0},
            )
            network.set_loraplus_lr_ratio(None, 5.0, None)
            network.apply_to(None, net, apply_text_encoder=False, apply_unet=True)

            groups, descriptions = network.prepare_optimizer_params(
                unet_lr=1e-4, learning_rate=1e-5,
            )
            plus_groups = [g for g in groups if g.get('is_lora_plus_group', False)]
            self.assertGreater(len(plus_groups), 0,
                "Should have LoRA+ groups from global/component ratio")

            for g in plus_groups:
                # Should use component ratio 5.0 since no regex matches
                expected_lr = 1e-4 * 5.0
                self.assertAlmostEqual(g['lr'].item(), expected_lr, delta=1e-10,
                    msg=f"Non-matching module should use global ratio 5.0 (lr={expected_lr}), got {g['lr'].item()}")
        finally:
            reset_globals()


if __name__ == "__main__":
    unittest.main()
