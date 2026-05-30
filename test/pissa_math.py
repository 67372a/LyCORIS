"""
Mathematical validation of PiSSA implementation against paper equations.

References each equation from:
  Meng et al., "PiSSA: Principal Singular Values and Singular Vectors
  Adaptation of Large Language Models", arXiv:2404.02948, 2024.
"""
import unittest

import torch
import torch.nn as nn

from lycoris.modules.locon import LoConModule
from lycoris.modules.base import LycorisBaseModule


class PiSSAMathValidation(unittest.TestCase):
    """Validate each PiSSA equation against the implementation."""

    def setUp(self):
        torch.manual_seed(42)
        self.m, self.n = 64, 48
        self.r = 6

    # ------------------------------------------------------------------
    # Equation (2): A = U_{[:,:r]} * S_{[:r,:r]}^{1/2}
    # ------------------------------------------------------------------
    def test_equation_2_A_initialization(self):
        """Verify A = U_{[:,:r]} * sqrt(S_{[:r,:r]})."""
        linear = nn.Linear(self.n, self.m, bias=False)
        W = torch.randn(self.m, self.n) * 0.1
        linear.weight.data.copy_(W)

        # Compute ground truth via torch.linalg.svd
        U, S, Vh = torch.linalg.svd(W.float(), full_matrices=False)
        U_r = U[:, :self.r]
        S_r = S[:self.r]
        sqrt_S_r = torch.sqrt(S_r)
        A_expected = U_r * sqrt_S_r.unsqueeze(0)  # (m, r)

        # Compute via PiSSA
        module = LoConModule(
            "test", linear, multiplier=1.0, lora_dim=self.r, alpha=self.r,
            svd_segment="top",
        )
        A_actual = module.lora_up.weight.data.float()

        self.assertTrue(
            torch.allclose(A_expected, A_actual, rtol=1e-4, atol=1e-6),
            f"Eq (2): A mismatch. max diff = {(A_expected - A_actual).abs().max().item():.2e}",
        )

    # ------------------------------------------------------------------
    # Equation (3): B = S_{[:r,:r]}^{1/2} * V_{[:,:r]}^T
    # ------------------------------------------------------------------
    def test_equation_3_B_initialization(self):
        """Verify B = sqrt(S_{[:r,:r]}) * V_{[:,:r]}^T."""
        linear = nn.Linear(self.n, self.m, bias=False)
        W = torch.randn(self.m, self.n) * 0.1
        linear.weight.data.copy_(W)

        # Ground truth
        U, S, Vh = torch.linalg.svd(W.float(), full_matrices=False)
        S_r = S[:self.r]
        Vh_r = Vh[:self.r, :]
        sqrt_S_r = torch.sqrt(S_r)
        B_expected = sqrt_S_r.unsqueeze(1) * Vh_r  # (r, n)

        module = LoConModule(
            "test", linear, multiplier=1.0, lora_dim=self.r, alpha=self.r,
            svd_segment="top",
        )
        B_actual = module.lora_down.weight.data.float()

        self.assertTrue(
            torch.allclose(B_expected, B_actual, rtol=1e-4, atol=1e-6),
            f"Eq (3): B mismatch. max diff = {(B_expected - B_actual).abs().max().item():.2e}",
        )

    # ------------------------------------------------------------------
    # Equation (4): W^res = U_{[:,r:]} S_{[r:,r:]} V_{[:,r:]}^T
    #  or equivalently: W^res = W - A B
    # ------------------------------------------------------------------
    def test_equation_4_residual(self):
        """Verify W^res = W - A B."""
        linear = nn.Linear(self.n, self.m, bias=False)
        W = torch.randn(self.m, self.n) * 0.1
        linear.weight.data.copy_(W)

        module = LoConModule(
            "test", linear, multiplier=1.0, lora_dim=self.r, alpha=self.r,
            svd_segment="top",
        )

        W_res = linear.weight.data.float()
        A = module.lora_up.weight.data.float()
        B = module.lora_down.weight.data.float()

        # Check W_res + A @ B == W
        reconstructed = W_res + A @ B
        self.assertTrue(
            torch.allclose(W, reconstructed, rtol=1e-4, atol=1e-6),
            f"Eq (4): W != W_res + A @ B. max diff = {(W - reconstructed).abs().max().item():.2e}",
        )

    # ------------------------------------------------------------------
    # Equation (5): Y = X (W^res + A B) = X W   at initialization
    # ------------------------------------------------------------------
    def test_equation_5_forward_unchanged_at_init(self):
        """Verify Y = X(W^res+AB) = XW at initialization."""
        linear = nn.Linear(self.n, self.m, bias=False)
        W = torch.randn(self.m, self.n) * 0.1
        linear.weight.data.copy_(W)

        x = torch.randn(3, self.n)

        with torch.no_grad():
            y_original = linear(x).clone()

        module = LoConModule(
            "test", linear, multiplier=1.0, lora_dim=self.r, alpha=self.r,
            svd_segment="top",
        )
        module.apply_to()

        with torch.no_grad():
            y_pissa = linear(x).clone()

        self.assertTrue(
            torch.allclose(y_original, y_pissa, rtol=1e-4, atol=1e-6),
            f"Eq (5): forward mismatch. max diff = {(y_original - y_pissa).abs().max().item():.2e}",
        )
        module.restore()

    # ------------------------------------------------------------------
    # Appendix C / Equation (9): ΔW = A'B' - A₀B₀ = [A' | A₀] [B' | -B₀]^T
    # ------------------------------------------------------------------
    def test_conversion_identity(self):
        """Verify ΔW = A'B' - A₀B₀ = [A'|A₀] @ [B'|-B₀]^T."""
        linear = nn.Linear(self.n, self.m, bias=False)
        W = torch.randn(self.m, self.n) * 0.1
        linear.weight.data.copy_(W)

        module = LoConModule(
            "test", linear, multiplier=1.0, lora_dim=self.r, alpha=self.r,
            svd_segment="top",
        )

        # Simulate training updates
        module.lora_up.weight.data.add_(torch.randn_like(module.lora_up.weight.data) * 0.01)
        module.lora_down.weight.data.add_(torch.randn_like(module.lora_down.weight.data) * 0.01)

        A0 = module.pissa_A_init.float().clone()
        B0 = module.pissa_B_init.float().clone()
        A_trained = module.lora_up.weight.data.float().clone()
        B_trained = module.lora_down.weight.data.float().clone()

        # Paper formula: ΔW = A'B' - A₀B₀
        delta_W_paper = A_trained @ B_trained - A0 @ B0

        # Conversion formula: [A' | A₀] @ [B' | -B₀]
        delta_W_conversion = torch.cat([A_trained, A0], dim=1) @ torch.cat([B_trained, -B0], dim=0)

        self.assertTrue(
            torch.allclose(delta_W_paper, delta_W_conversion, rtol=1e-4, atol=1e-6),
            f"Eq (9): conversion mismatch. max diff = {(delta_W_paper - delta_W_conversion).abs().max().item():.2e}",
        )

        # Now test convert_pissa_to_lora
        module.convert_pissa_to_lora()
        converted_delta = module.lora_up.weight.data.float() @ module.lora_down.weight.data.float()

        self.assertTrue(
            torch.allclose(delta_W_paper, converted_delta, rtol=1e-4, atol=1e-6),
            f"convert_pissa_to_lora: delta mismatch. "
            f"max diff = {(delta_W_paper - converted_delta).abs().max().item():.2e}",
        )

    # ------------------------------------------------------------------
    # Post-conversion forward pass equivalence
    # ------------------------------------------------------------------
    def test_conversion_preserves_forward(self):
        """After conversion, the effective weight must match PiSSA forward."""
        linear = nn.Linear(self.n, self.m, bias=False)
        W = torch.randn(self.m, self.n) * 0.1
        linear.weight.data.copy_(W)

        module = LoConModule(
            "test", linear, multiplier=1.0, lora_dim=self.r, alpha=self.r,
            svd_segment="top",
        )

        # Train a bit
        module.lora_up.weight.data.add_(torch.randn_like(module.lora_up.weight.data) * 0.02)
        module.lora_down.weight.data.add_(torch.randn_like(module.lora_down.weight.data) * 0.02)

        module.apply_to()
        x = torch.randn(3, self.n)
        with torch.no_grad():
            y_before = linear(x).clone()
        module.restore()

        # Convert and check forward
        W_res = linear.weight.data.clone()
        A0 = module.pissa_A_init.clone()
        B0 = module.pissa_B_init.clone()
        A_t = module.lora_up.weight.data.clone()
        B_t = module.lora_down.weight.data.clone()

        module.convert_pissa_to_lora()
        module.apply_to()
        with torch.no_grad():
            y_after = linear(x).clone()
        module.restore()

        # Expected: W_original + [A_t|A0]@[B_t|-B0] = W_res + A0@B0 + A_t@B_t - A0@B0 = W_res + A_t@B_t
        # Which is exactly the PiSSA forward
        self.assertTrue(
            torch.allclose(y_before, y_after, rtol=1e-4, atol=1e-6),
            f"Conversion forward mismatch. max diff = {(y_before - y_after).abs().max().item():.2e}",
        )

        # Verify base weight was restored to original W
        W_reconstructed = W_res.float() + A0.float() @ B0.float()
        self.assertTrue(
            torch.allclose(linear.weight.data.float(), W_reconstructed, rtol=1e-4, atol=1e-6),
            "Base weight not restored to W_original after conversion",
        )

    # ------------------------------------------------------------------
    # Fast SVD vs Exact SVD: the top-r singular values should match closely
    # ------------------------------------------------------------------
    def test_fast_svd_singular_values_accuracy(self):
        """Fast SVD singular values should approximate exact SVD."""
        linear = nn.Linear(128, 96, bias=False)
        W = torch.randn(96, 128) * 0.3
        linear.weight.data.copy_(W)

        # Exact SVD ground truth
        V_exact, S_exact, Uhr_exact = LycorisBaseModule._compute_svd_pissa(
            W.float(), self.r, niter=0
        )

        for niter in [1, 2, 4, 8]:
            V_fast, S_fast, Uhr_fast = LycorisBaseModule._compute_svd_pissa(
                W.float(), self.r, niter=niter
            )

            # Singular values should be close
            sv_error = (S_exact - S_fast).abs().max().item()
            sv_rel_error = sv_error / S_exact[0].item()
            self.assertLess(
                sv_rel_error,
                {1: 0.15, 2: 0.08, 4: 0.04, 8: 0.02}[niter],
                f"niter={niter}: singular value relative error {sv_rel_error:.4f} too high",
            )

    # ------------------------------------------------------------------
    # Scale handling: with alpha=r, scale must be 1.0
    # ------------------------------------------------------------------
    def test_scale_is_one_when_alpha_equals_rank(self):
        """With alpha=lora_dim, self.scale must be 1.0 (no double scaling)."""
        linear = nn.Linear(self.n, self.m, bias=False)
        linear.weight.data.normal_()

        for r in [2, 4, 8, 16]:
            module = LoConModule(
                "test", linear, multiplier=1.0, lora_dim=r, alpha=r,
                svd_segment="top",
            )
            self.assertAlmostEqual(
                module.scale, 1.0, places=6,
                msg=f"r={r}: scale={module.scale} != 1.0",
            )

    # ------------------------------------------------------------------
    # Custom state dict: PiSSA init weights are preserved in export
    # ------------------------------------------------------------------
    def test_custom_state_dict_contains_pissa_init_weights(self):
        """pissa_convert=False preserves pissa_A_init and pissa_B_init in export."""
        linear = nn.Linear(self.n, self.m, bias=False)
        linear.weight.data.normal_()

        module = LoConModule(
            "test", linear, multiplier=1.0, lora_dim=self.r, alpha=self.r,
            svd_segment="top", pissa_convert=False,
        )

        sd = module.custom_state_dict()
        self.assertIn("pissa_A_init", sd,
                      "pissa_A_init must be in custom_state_dict when pissa_convert=False")
        self.assertIn("pissa_B_init", sd,
                      "pissa_B_init must be in custom_state_dict when pissa_convert=False")

        # The exported init weights must match the ground-truth SVD result
        A_exported = sd["pissa_A_init"]
        B_exported = sd["pissa_B_init"]
        self.assertTrue(
            torch.allclose(A_exported, module.lora_up.weight * module.scalar.to(A_exported.device)),
            "Exported pissa_A_init must match lora_up (scaled by scalar)",
        )
        self.assertTrue(
            torch.allclose(B_exported, module.lora_down.weight),
            "Exported pissa_B_init must match lora_down exactly",
        )

    # ------------------------------------------------------------------
    # Custom state dict: converted export produces portable LoRA format
    # ------------------------------------------------------------------
    def test_custom_state_dict_converted_double_rank(self):
        """pissa_convert=True exports doubled-rank LoRA with pissa_converted flag."""
        linear = nn.Linear(self.n, self.m, bias=False)
        linear.weight.data.normal_()

        module = LoConModule(
            "test", linear, multiplier=1.0, lora_dim=self.r, alpha=self.r,
            svd_segment="top", pissa_convert=True,
        )

        sd = module.custom_state_dict()
        self.assertIn("pissa_converted", sd,
                      "pissa_converted flag must be present")
        self.assertEqual(sd["lora_up.weight"].shape, (self.m, 2 * self.r),
                         "Converted lora_up must have doubled rank")
        self.assertEqual(sd["lora_down.weight"].shape, (2 * self.r, self.n),
                         "Converted lora_down must have doubled rank")

        # Verify: portable format delta must be zero at init
        delta = sd["lora_up.weight"].float() @ sd["lora_down.weight"].float()
        self.assertLess(delta.norm().item(), 1e-4,
                        f"Converted LoRA delta must be near-zero at init, got {delta.norm().item():.2e}")


    # ------------------------------------------------------------------
    # Randomized SVD with non-fp32 weights: must upcast internally
    # ------------------------------------------------------------------
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for bf16 test")
    def test_randomized_svd_uses_fp32_for_bf16_weights(self):
        """Randomized SVD must upcast bf16 weights to fp32 internally.

        Before the fix, the randomized SVD path computed W_float but then
        used org_weight_2d (bf16) for the matrix operations, producing
        inaccurate results due to bf16 quantization noise amplification
        in power iterations.  After the fix, all ops use W_float (fp32).

        We verify by seeding the RNG so both bf16 and fp32 paths use
        the same random projection Omega, then comparing singular values.
        If W_float were NOT used, the bf16 matmuls in power iterations
        would produce different (less accurate) singular values.
        """
        torch.manual_seed(42)
        W_bf16 = torch.randn(self.m, self.n, dtype=torch.bfloat16, device="cuda")
        W_fp32 = W_bf16.float()

        # Randomized SVD on bf16 — should internally upcast to fp32
        torch.manual_seed(77)
        _, Sr_bf16, _ = LycorisBaseModule._compute_svd_pissa(
            W_bf16, self.r, niter=4
        )

        # Randomized SVD on fp32 — reference with same seed
        torch.manual_seed(77)
        _, Sr_fp32, _ = LycorisBaseModule._compute_svd_pissa(
            W_fp32, self.r, niter=4
        )

        # With the same seed, both produce the same Omega. Since both now
        # use W_float (fp32) for the matmuls, the singular values must match.
        sv_error = (Sr_fp32 - Sr_bf16).abs().max().item()
        self.assertEqual(
            sv_error,
            0.0,
            f"bf16 and fp32 fast SVD singular values differ by {sv_error:.2e} "
            f"with same seed. This indicates W_float is not being used.",
        )


if __name__ == "__main__":
    unittest.main()
