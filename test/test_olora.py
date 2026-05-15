"""
Unit tests for O-LoRA integration in LyCORIS LoConModule.

Tests cover:
- Initialization with O-LoRA enabled
- add_task freezing behavior
- Orthogonality loss computation (L1)
- Multi-task weight computation
- Bypass and non-bypass forward passes
- Weight merging into base
- State dict round-trip
- Gradient flow isolation
- Conv module support
- Static loss aggregation
"""

import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Make the LyCORIS package importable from the test directory
# ---------------------------------------------------------------------------
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lycoris.modules.locon import LoConModule


@pytest.fixture(autouse=True)
def _reset_olora_registry():
    """Ensure each test starts with a clean O-LoRA module registry."""
    LoConModule.reset_olora_registry()
    yield
    LoConModule.reset_olora_registry()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def linear_module():
    """A simple linear layer to wrap."""
    return nn.Linear(64, 128, bias=False)


@pytest.fixture
def conv_module():
    """A simple conv2d layer to wrap."""
    return nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)


@pytest.fixture
def olora_linear(linear_module):
    """LoConModule with O-LoRA enabled for linear layer."""
    mod = LoConModule(
        "test_lora",
        linear_module,
        lora_dim=8,
        alpha=8,
        olora=True,
        olora_lambda=0.5,
        olora_task_id=0,
        use_scalar=True,
    )
    return mod


@pytest.fixture
def olora_conv(conv_module):
    """LoConModule with O-LoRA enabled for conv layer."""
    mod = LoConModule(
        "test_conv_lora",
        conv_module,
        lora_dim=4,
        alpha=4,
        olora=True,
        olora_lambda=0.5,
        olora_task_id=0,
        use_scalar=True,
    )
    return mod


def _make_random_lora_weights(down_module, up_module, down_val=0.5, up_val=0.3):
    """Fill LoRA weights with controlled non-orthogonal values."""
    nn.init.constant_(down_module.weight, down_val)
    nn.init.constant_(up_module.weight, up_val)


# ---------------------------------------------------------------------------
# Initialization Tests
# ---------------------------------------------------------------------------

class TestOLoRAInitialization:
    def test_olora_enabled_creates_module_lists(self, olora_linear):
        assert olora_linear.olora is True
        assert len(olora_linear.lora_down_modules) == 1
        assert len(olora_linear.lora_up_modules) == 1
        assert olora_linear.olora_task_id == 0

    def test_olora_default_params_trainable(self, olora_linear):
        for p in olora_linear.lora_down_modules[0].parameters():
            assert p.requires_grad is True
        for p in olora_linear.lora_up_modules[0].parameters():
            assert p.requires_grad is True

    def test_olora_registry(self, olora_linear):
        assert olora_linear in LoConModule._olora_modules

    def test_olora_disabled_no_module_lists(self, linear_module):
        mod = LoConModule("test", linear_module, lora_dim=4)
        assert mod.olora is False
        assert len(mod.lora_down_modules) == 0
        assert len(mod.lora_up_modules) == 0

    def test_backward_compat_no_olora(self, linear_module):
        """Existing (non-OLoRA) creation should still work identically."""
        mod = LoConModule("test", linear_module, lora_dim=4, alpha=4)
        x = torch.randn(2, 64)
        out = mod.bypass_forward(x)
        assert out.shape == (2, 128)
        # Should still have scalar etc.
        assert hasattr(mod, "scalar")


# ---------------------------------------------------------------------------
# add_task Tests
# ---------------------------------------------------------------------------

class TestAddTask:
    def test_add_task_creates_new_modules(self, olora_linear):
        olora_linear.add_task(1)
        assert len(olora_linear.lora_down_modules) == 2
        assert len(olora_linear.lora_up_modules) == 2
        assert olora_linear.olora_task_id == 1

    def test_add_task_freezes_old_modules(self, olora_linear):
        olora_linear.add_task(1)
        # Old modules frozen
        for p in olora_linear.lora_down_modules[0].parameters():
            assert p.requires_grad is False
        for p in olora_linear.lora_up_modules[0].parameters():
            assert p.requires_grad is False
        # New modules trainable
        for p in olora_linear.lora_down_modules[1].parameters():
            assert p.requires_grad is True
        for p in olora_linear.lora_up_modules[1].parameters():
            assert p.requires_grad is True

    def test_add_task_updates_lora_refs(self, olora_linear):
        olora_linear.add_task(1)
        assert olora_linear.lora_down is olora_linear.lora_down_modules[1]
        assert olora_linear.lora_up is olora_linear.lora_up_modules[1]

    def test_add_task_on_non_olora_raises(self, linear_module):
        mod = LoConModule("test", linear_module, lora_dim=4)
        with pytest.raises(RuntimeError):
            mod.add_task(1)

    def test_add_task_creates_scalar(self, olora_linear):
        olora_linear.add_task(1)
        assert len(olora_linear.lora_scalar_list) == 2


# ---------------------------------------------------------------------------
# Orthogonality Loss Tests
# ---------------------------------------------------------------------------

class TestOrthogonalityLoss:
    def test_zero_loss_single_task(self, olora_linear):
        loss = olora_linear.get_olora_orthogonality_loss()
        assert loss.item() == 0.0

    def test_nonzero_loss_non_orthogonal(self, olora_linear):
        _make_random_lora_weights(
            olora_linear.lora_down_modules[0], olora_linear.lora_up_modules[0]
        )
        olora_linear.add_task(1)
        _make_random_lora_weights(
            olora_linear.lora_down_modules[1],
            olora_linear.lora_up_modules[1],
            down_val=0.7,
            up_val=0.2,
        )
        loss = olora_linear.get_olora_orthogonality_loss()
        assert loss.item() > 0.0

    def test_zero_loss_orthogonal_weights(self, olora_linear):
        # Make task 0 A identity-like, task 1 A orthogonal to it
        A0 = torch.eye(8, 64)[:8, :]  # (8, 64) - orthogonal rows
        olora_linear.lora_down_modules[0].weight.data.copy_(A0)
        olora_linear.add_task(1)
        # Task 1 A: rows orthogonal to task 0 rows (use zeros in same space, different rows)
        A1 = torch.zeros(8, 64)
        A1[:, 8:16] = torch.eye(8)  # orthogonal to first 8 dims
        olora_linear.lora_down_modules[1].weight.data.copy_(A1)
        loss = olora_linear.get_olora_orthogonality_loss()
        # A0 @ A1^T should be zero → L1 loss = 0
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_l1_form(self, olora_linear):
        """Verify the loss uses L1 (abs sum) matching reference implementation."""
        olora_linear.lora_down_modules[0].weight.data.fill_(0.1)
        olora_linear.add_task(1)
        olora_linear.lora_down_modules[1].weight.data.fill_(0.2)
        loss = olora_linear.get_olora_orthogonality_loss()
        # Manual: A0 @ A1^T → (8,8) all 0.1*0.2*64 = 1.28 per element
        # L1: sum(abs) over 64 elements = 64 * 1.28 = 81.92
        expected = torch.abs(
            olora_linear.lora_down_modules[0].weight
            @ olora_linear.lora_down_modules[1].weight.T
        ).sum()
        assert loss.item() == pytest.approx(expected.item(), rel=1e-5)

    def test_static_aggregation(self, linear_module):
        mod1 = LoConModule("a", linear_module, lora_dim=4, olora=True, use_scalar=True)
        mod2 = LoConModule("b", linear_module, lora_dim=4, olora=True, use_scalar=True)
        mod1.lora_down_modules[0].weight.data.fill_(0.1)
        mod1.add_task(1)
        mod2.lora_down_modules[0].weight.data.fill_(0.2)
        mod2.add_task(1)
        total = LoConModule.get_total_olora_loss()
        manual = mod1.get_olora_orthogonality_loss() + mod2.get_olora_orthogonality_loss()
        assert total.item() == pytest.approx(manual.item(), rel=1e-5)

    def test_conv_orthogonality_loss(self, olora_conv):
        _make_random_lora_weights(
            olora_conv.lora_down_modules[0], olora_conv.lora_up_modules[0]
        )
        olora_conv.add_task(1)
        _make_random_lora_weights(
            olora_conv.lora_down_modules[1],
            olora_conv.lora_up_modules[1],
            down_val=0.7,
            up_val=0.2,
        )
        loss = olora_conv.get_olora_orthogonality_loss()
        assert loss.item() > 0.0


# ---------------------------------------------------------------------------
# Forward / Weight Tests
# ---------------------------------------------------------------------------

class TestOLoRAForward:
    def test_make_weight_single_task_equals_standard(self, linear_module):
        mod_olora = LoConModule("a", linear_module, lora_dim=4, alpha=4, olora=True)
        mod_std = LoConModule("b", linear_module, lora_dim=4, alpha=4)
        # Copy weights
        mod_std.lora_down.weight.data.copy_(mod_olora.lora_down.weight.data)
        mod_std.lora_up.weight.data.copy_(mod_olora.lora_up.weight.data)
        w_olora = mod_olora.make_weight()
        w_std = mod_std.make_weight()
        assert torch.allclose(w_olora, w_std)
        LoConModule.reset_olora_registry()

    def test_make_weight_two_tasks(self, olora_linear):
        _make_random_lora_weights(
            olora_linear.lora_down_modules[0], olora_linear.lora_up_modules[0]
        )
        olora_linear.add_task(1)
        _make_random_lora_weights(
            olora_linear.lora_down_modules[1],
            olora_linear.lora_up_modules[1],
            down_val=0.2,
            up_val=0.4,
        )
        w = olora_linear.make_weight()
        # Should be the sum of both task weights
        assert w.shape == (128, 64)

    def test_bypass_forward_single_task(self, olora_linear):
        x = torch.randn(4, 64)
        out = olora_linear.bypass_forward(x)
        assert out.shape == (4, 128)

    def test_bypass_forward_two_tasks(self, olora_linear):
        _make_random_lora_weights(
            olora_linear.lora_down_modules[0], olora_linear.lora_up_modules[0]
        )
        olora_linear.add_task(1)
        _make_random_lora_weights(
            olora_linear.lora_down_modules[1],
            olora_linear.lora_up_modules[1],
            down_val=0.2,
            up_val=0.4,
        )
        x = torch.randn(4, 64)
        out = olora_linear.bypass_forward(x)
        assert out.shape == (4, 128)

    def test_forward_nonbypass_single_task(self, olora_linear):
        olora_linear.bypass_mode = False
        x = torch.randn(4, 64)
        out = olora_linear.forward(x)
        assert out.shape == (4, 128)

    def test_forward_nonbypass_two_tasks(self, olora_linear):
        olora_linear.bypass_mode = False
        _make_random_lora_weights(
            olora_linear.lora_down_modules[0], olora_linear.lora_up_modules[0]
        )
        olora_linear.add_task(1)
        _make_random_lora_weights(
            olora_linear.lora_down_modules[1],
            olora_linear.lora_up_modules[1],
            down_val=0.2,
            up_val=0.4,
        )
        x = torch.randn(4, 64)
        out = olora_linear.forward(x)
        assert out.shape == (4, 128)


# ---------------------------------------------------------------------------
# Merge Tests
# ---------------------------------------------------------------------------

class TestOLoRAMerge:
    def test_merge_old_tasks_modifies_base(self, olora_linear):
        _make_random_lora_weights(
            olora_linear.lora_down_modules[0], olora_linear.lora_up_modules[0]
        )
        olora_linear.add_task(1)
        _make_random_lora_weights(
            olora_linear.lora_down_modules[1],
            olora_linear.lora_up_modules[1],
            down_val=0.2,
            up_val=0.4,
        )
        base_before = olora_linear.org_module[0].weight.data.clone()
        olora_linear.merge_old_tasks_to_base()
        base_after = olora_linear.org_module[0].weight.data
        assert not torch.allclose(base_before, base_after)

    def test_merge_old_tasks_removes_modules(self, olora_linear):
        _make_random_lora_weights(
            olora_linear.lora_down_modules[0], olora_linear.lora_up_modules[0]
        )
        olora_linear.add_task(1)
        olora_linear.merge_old_tasks_to_base()
        # Only the current (last) task remains
        assert len(olora_linear.lora_down_modules) == 1
        assert len(olora_linear.lora_up_modules) == 1

    def test_merge_old_tasks_resets_task_id(self, olora_linear):
        olora_linear.add_task(1)
        olora_linear.merge_old_tasks_to_base()
        assert olora_linear.olora_task_id == 0

    def test_merge_single_task_noop(self, olora_linear):
        base_before = olora_linear.org_module[0].weight.data.clone()
        olora_linear.merge_old_tasks_to_base()
        assert torch.allclose(base_before, olora_linear.org_module[0].weight.data)


# ---------------------------------------------------------------------------
# Serialization Tests
# ---------------------------------------------------------------------------

class TestOLoRASerialization:
    def test_state_dict_roundtrip(self, olora_linear):
        _make_random_lora_weights(
            olora_linear.lora_down_modules[0], olora_linear.lora_up_modules[0]
        )
        olora_linear.add_task(1)
        _make_random_lora_weights(
            olora_linear.lora_down_modules[1],
            olora_linear.lora_up_modules[1],
            down_val=0.2,
            up_val=0.4,
        )
        sd = olora_linear.custom_state_dict()
        assert "lora_up_task0.weight" in sd
        assert "lora_down_task0.weight" in sd
        assert "lora_up_task1.weight" in sd
        assert "lora_down_task1.weight" in sd
        assert "olora_task_id" in sd

    def test_non_olora_state_dict_unchanged(self, linear_module):
        mod = LoConModule("test", linear_module, lora_dim=4, alpha=4)
        sd = mod.custom_state_dict()
        assert "lora_up.weight" in sd
        assert "lora_down.weight" in sd
        assert "olora_task_id" not in sd


# ---------------------------------------------------------------------------
# Gradient Isolation Tests
# ---------------------------------------------------------------------------

class TestGradientIsolation:
    def test_gradient_only_to_current_task(self, olora_linear):
        _make_random_lora_weights(
            olora_linear.lora_down_modules[0], olora_linear.lora_up_modules[0]
        )
        olora_linear.add_task(1)
        _make_random_lora_weights(
            olora_linear.lora_down_modules[1],
            olora_linear.lora_up_modules[1],
            down_val=0.2,
            up_val=0.4,
        )
        x = torch.randn(4, 64)
        out = olora_linear.bypass_forward(x)
        loss = out.sum()
        loss.backward()

        # Old task params should have no gradient
        for p in olora_linear.lora_down_modules[0].parameters():
            assert p.grad is None
        for p in olora_linear.lora_up_modules[0].parameters():
            assert p.grad is None
        # New task params should have gradient
        for p in olora_linear.lora_down_modules[1].parameters():
            assert p.grad is not None
        for p in olora_linear.lora_up_modules[1].parameters():
            assert p.grad is not None


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------

class TestOLoRAEdgeCases:
    def test_olora_disabled_orth_loss_returns_zero(self, linear_module):
        mod = LoConModule("test", linear_module, lora_dim=4, olora=False)
        assert mod.get_olora_orthogonality_loss().item() == 0.0

    def test_reset_olora_registry(self, linear_module):
        mod = LoConModule("test", linear_module, lora_dim=4, olora=True)
        assert len(LoConModule._olora_modules) == 1
        LoConModule.reset_olora_registry()
        assert len(LoConModule._olora_modules) == 0
        assert LoConModule.get_total_olora_loss().item() == 0.0
