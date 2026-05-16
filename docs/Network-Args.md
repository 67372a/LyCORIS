# Network Arguments

Arguments to put in `network_args` for kohya sd scripts

### Algo

- Set with `algo=ALGO_NAME`
- Check [List of Implemented Algorithms](Algo-List.md) for algorithms to use

### Preset

- Set with `preset=PRESET/CONFIG_FILE`
- Pre-implemented: `full` (default), `attn-mlp`, `attn-only` etc.
- Valid for all but (IA)^3
- Use `preset=xxx.toml` to choose config file (for LyCORIS module settings)
- More info in [Preset](Preset.md)

### Dimension

- Dimension of the linear layers is set with the _script argument_ `network_dim`
- Dimension of the convolutional layers is set with `conv_dim=INT`
- Valid for all but (IA)^3 and native fine-tuning
- For LoKr, setting dimension to sufficiently large value (>10240/2) prevents the second block from being further decomposed

### Alpha

- Alpha of the linear layers is set with the _script argument_ `network_alpha`
- Alpha of the convolutional layers is set with `conv_alpha=FLOAT`
- Valid for all but (IA)^3 and native fine-tuning, ignored by full dimension LoKr as well
- Merge ratio is alpha/dimension, check Appendix B.1 of our [paper](https://arxiv.org/abs/2309.14859) for relation between alpha and learning rate / initialization

### Dropouts

- Set with `dropout=FLOAT`, `rank_dropout=FLOAT`, `module_dropout=FLOAT`
- Set the dropout rate, the types of dropout that are valid could vary from method to method

### Factor

- Set with `factor=INT`
- Valid for LoKr
- Use `-1` to get the smallest decomposition

### Decompose both

- Enabled with `decompose_both=True`
- Valid for LoKr
- Perform LoRA decomposition of both matrices resulting from LoKr decomposition (by default only the larger matrix is decomposed)

### Block Size

- Set with `block_size=INT`
- Valid for DyLoRA
- Set the "unit" of DyLoRA (i.e. how many rows / columns to update each time)

### Tucker Decomposition

- Enabled with `use_tucker=True`
- Valid for all but (IA)^3 and native fine-tuning
- It was given the wrong name `use_cp=` in older version

### Scalar

- Enabled with `use_scalar=True`
- Valid for LoRA, LoHa, LoKr, GLoRA, ABBA, and T-LoRA.
- Train an additional scalar in front of the weight difference
- Use a different weight initialization strategy

### Scalar Init Value

- Set with `scalar_init_value=FLOAT`
- Valid for all modules that support `use_scalar`
- Set the initial value of the learnable scalar parameter when `use_scalar=True`
- Default: 0.1 for LoRA/LoHa/LoKr/GLoRA, 0.9 for ABBA/T-LoRA

### Weight Decompose

* Enabled with `dora_wd=True`
* Valid for LoRA, LoHa, and LoKr
* Enable the DoRA method for these algorithms.
* Will force `bypass_mode=False`

### Bypass Mode

* Enabled with `bypass_mode=True`
* Valid for LoRA, LoHa, LoKr
* Use $Y = WX + \Delta WX$  instead of $Y=(W+\Delta W)X$
* Designed for bnb 8bit/4bit linear layer. (QLyCORIS)

### Normalization Layers

- Enabled with `train_norm=True`
- Valid for all but (IA)^3

### Rescaled OFT

- Enabled with `rescaled=True`
- Valid for Diag-OFT

### Constrained OFT

- Enabled with `constraint=FLOAT`
- Valid for Diag-OFT

### Singular Vector Type (T-LoRA)

- Set with `sig_type=STRING`
- Valid for T-LoRA
- Options: `principal` (default), `last`, `middle`
- Controls which singular vectors from SVD are used for initialization:
  - `principal`: Top-k singular vectors (largest singular values)
  - `last`: Bottom-k singular vectors (smallest singular values)
  - `middle`: Middle-k singular vectors

### Orthogonal Weight Initialization

- Enabled with `orthogonal_init=True`
- Valid for LoRA, LoHa, LoKr, and GLoRA
- Uses `torch.nn.init.orthogonal_()` instead of `kaiming_uniform_` for initial weight values
- Will force `use_scalar=True` when enabled

### Runtime Orthogonalization

- Enabled with `orthogonalize=True`
- Valid for LoRA, LoHa, LoKr, and GLoRA
- Applies QR decomposition during forward pass to keep weights near-orthogonal during training
- Will force `orthogonal_init=True` (and thus `use_scalar=True`) when enabled
- Can be used together with `orthogonal_init` explicitly, or alone (which implies it)

### Data-Dependent Initialization (T-LoRA)

- Enabled with `use_data_init=True` (default)
- Valid for T-LoRA
- When True, performs SVD on the original layer weights
- When False, performs SVD on a random matrix (data-independent)

### Timestep Mask Group (T-LoRA)

- Set with `mask_group_id=INT`
- Valid for T-LoRA
- Default: 0
- For multi-network scenarios, allows different networks to use different timestep masks

### RaLoRA-Specific Arguments

These arguments control the RaLoRA/RaLoRA-Pro algorithm behavior. They are valid only when `algo=ralora`.

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `ralora_n_max` | int | 32 | Maximum block expansion factor (n_max in paper). Controls the max number of diagonal blocks. Recommended to search over {16, 32, 64} for best results. |
| `ralora_pro` | bool | False | Enable RaLoRA-Pro mode: dual alignment with both intra-layer GID and inter-layer importance. |
| `ralora_ref_rank` | int | lora_dim | Reference rank for RaLoRA-Pro parameter budget calculation. |
| `ralora_min_rank` | int | dim//2 (pro) | Minimum rank per layer (RaLoRA-Pro only). Paper default: r_ref/2. |
| `ralora_max_rank` | int | dim×2 (pro) | Maximum rank per layer (RaLoRA-Pro only). Paper default: 2×r_ref. |
| `ralora_dynamic_scaling` | bool | False | Use per-layer rank (vs. average rank) for the scaling factor α/r. |
| `ralora_rank_stabilize` | bool | False | Apply √r scaling stabilization (rsLoRA-like). |
| `ralora_erank_method` | str | "entropy" | GID estimation method: "entropy" (entropy-based effective rank, recommended), "threshold" (count σ > ε), or "cumulative_variance". |
| `ralora_svd_threshold` | float | 0.0 | Threshold ε for the "threshold" erank method (>0 enables this method). |
| `ralora_cumulative_variance` | float | 0.0 | Variance fraction threshold for "cumulative_variance" method (>0 enables). |
| `ralora_forward_method` | str | "concat" | Block-diagonal forward strategy: "concat" (block_diag assembly) or "einsum" (reshape + batched einsum). |

**Recommended Configuration (from paper):**
- `algo=ralora` with `network_dim=8`, `network_alpha=8` (alpha = dim)
- For RaLoRA: `ralora_n_max=32` (default, search 16-64 for best)
- For RaLoRA-Pro: add `ralora_pro=True` (min/max rank set automatically to dim/2 and dim×2)
- `ralora_erank_method="entropy"` (default, most robust)

**Notes:**
- RaLoRA requires a precomputation phase before training via `RaLoRAModule.precompute_and_init()`
- In kohya scripts, the precomputation must be triggered manually after network creation
- Tucker decomposition (`use_tucker=True`) falls back to n_split=1 for RaLoRA
- For best results, set `alpha` equal to `network_dim` (paper convention)

### GoRA-Specific Arguments

These arguments control the GoRA algorithm behavior. They are valid only when `algo=gora`.

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `gora_ref_rank` | int | lora_dim | Reference rank r^ref for parameter budget calculation (Eq. 7). Allocates similar total params as LoRA with this rank. |
| `gora_min_rank` | int | 1 | Minimum rank per adapter. Paper default: r_ref/2. |
| `gora_max_rank` | int | 32 | Maximum rank per adapter. Paper default: 4×r_ref. |
| `gora_gamma` | float | 0.05 | Scaling factor γ for initialization magnitude. Higher = stronger initialization. Paper: 0.05-0.08. |
| `gora_importance_type` | str | "union_mean" | Importance metric. Default `avg(\|W⊙G\|)` (Eq. 5). Also supports: "union_frobenius_norm", "grad_nuc_norm", "grad_entropy", dual metrics like "union_mean_grad_nuc_norm", etc. |
| `gora_softmax_importance` | bool | False | Use softmax with temperature for importance normalization instead of linear sum. |
| `gora_temperature` | float | 0.5 | Temperature for softmax importance normalization. |
| `gora_scale_importance` | bool | False | Apply sqrt to raw importance scores before normalization. |
| `gora_features_func` | str | "sqrt" | Feature adjustment for budget: "sqrt" (√(m+n), paper default), "log1p", "identity" (m+n). |
| `gora_allocate_strategy` | str | "moderate" | Rounding strategy: "radical" (ceil), "moderate" (round), "conserved" (floor). |
| `gora_adaptive_n` | bool | True | Enable adaptive N: auto-stop gradient accumulation when importance scores converge. |
| `gora_convergence_threshold` | float | 0.01 | Relative change threshold for adaptive N convergence. |
| `gora_min_steps` | int | 3 | Minimum gradient accumulation steps before checking convergence. |
| `gora_adaptive_gamma` | bool | False | Enable adaptive γ: auto-tune scaling factor on first batch to minimize loss. |
| `gora_scale_by_lr` | bool | False | Use learning rate in scaling formula (alternative to gora_gamma). |
| `gora_lr` | float | 1e-3 | Learning rate for lr-based scaling (when gora_scale_by_lr=True). |

**Recommended Configuration (from paper):**
- `algo=gora` with `network_dim=8`, `network_alpha=16`
- `gora_ref_rank=8`, `gora_min_rank=4`, `gora_max_rank=32`
- `gora_gamma=0.05` (for code) or `0.08` (for math)
- `gora_importance_type="union_mean"`
- `gora_adaptive_n=True` (auto-determines N, eliminates manual tuning)

**Notes:**
- GoRA requires a precomputation phase via `LycorisNetwork.prepare_gora()` before training
- After precomputation, the saved state dict is identical to standard LoRA/LoCon
- GoRA always uses rsLoRA scaling (α/√r)
- Does NOT manipulate pre-trained weights — no training-inference gap
- GoRA is compatible with weight_decompose (DoRA), use_scalar, orthogonalize, and QLoRA/bnb layers
- For inference, the checkpoint can be loaded as standard LoRA without special handling
- For adaptively changing γ, set `gora_adaptive_gamma=True` (adds one pre-training forward pass)
