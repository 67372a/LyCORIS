# Change Log

## 2026/08/23 (dev branch)

#### Bug fixes

* Harden LoKr reconstruction across rebuild, bypass, merge, parametrization, and state-dict loading.
  * Apply alpha/rank scaling and scalar baking exactly once.
  * Fix low-rank Conv1d/Conv2d/Conv3d bypass factor views and 1×1 projection parameters.
  * Repair kernel-aware SVD segment initialization, including scalar and unbalanced factorization modes.
  * Preserve mixed decomposition and unbalanced-factorization metadata during checkpoint round trips.
  * Enforce the documented DoRA/bypass incompatibility and align rank-dropout behavior.
  * Add CUDA regression coverage for numerical parity, convolution bypass, SVD initialization, and serialization.

## 2026/07/18 (dev branch)

#### Improvements

* **Optimizer parameter attribute tagging**: LyCORIS modules now set optimizer-relevant attributes (`_is_dora_scale`, `_is_oft`, `_is_lora_A`, `_is_lora_B`, `is_hidden`, `is_vector`) on their `nn.Parameter` objects during `prepare_optimizer_params()` / `prepare_grad_etc()`. This allows Advanced_Optimizers (and compatible optimizers) to accurately identify each parameter's role (DoRA scale, OFT block, LoRA A/B factor, hidden weight, or logical vector) and apply correct spectral normalization, weight decay, and Kourkoutas-β bucketing.
  * `LycorisBaseModule.tag_parameters()` detects known parameter structures via `hasattr` / `isinstance` and sets the appropriate attributes.
  * `LycorisNetwork._tag_all_parameters()` iterates all lora modules and calls `tag_parameters()`. Called from `prepare_optimizer_params()` and `prepare_grad_etc()` to ensure attributes survive device moves (`.to()` / `.cuda()`).
  * LoHa / LoKr / GLoRA factors are **not** tagged as `_is_lora_A` / `_is_lora_B` because they use different factorization geometry.
  * `is_hidden` is determined via an `original_name` heuristic: the module's path into the root model is checked against known non-hidden top-level component prefixes (`time_embedding`, `conv_in`, `conv_out`, `img_in`, `txt_in`, `final_layer`, `x_embedder`, `pos_embedder`, `patch_embed`, `context_embedder`, etc.). Modules targeting these prefixes are marked `is_hidden=False`; all others (transformer/resnet block internals) are `is_hidden=True`. Covers all 7 architectures (SD1/SDXL UNet, Flux, SD3, Anima, Lumina, Hunyuan Image, Chroma).

## 2026/05/16 (dev branch)

#### New Features

* **RaLoRA / RaLoRA-Pro**: Rank-Aligned LoRA with [Gradient Intrinsic Dimensionality Alignment](https://openreview.net/forum?id=kObvnQ6pUx) (ICLR 2026). Adaptively aligns LoRA adapter capacity with per-layer gradient intrinsic dimensionality using block-diagonal decomposition. RaLoRA-Pro adds inter-layer rank reallocation guided by loss sensitivity. Triggered by `algo=ralora`.
  * See [docs/Algo-List.md](docs/Algo-List.md) and [docs/Algo-Details.md](docs/Algo-Details.md) for details.
  * Requires a precomputation phase via `RaLoRAModule.precompute_and_init()` before training.

#### Bug fixes

* Fix orthogonal weights test

## 2026/05/15 (dev branch)

#### New Features

* **GoRA: Gradient-driven Adaptive Low Rank Adaptation**: Adaptively allocates ranks and initializes LoRA weights using gradient information from a brief precomputation phase. The saved checkpoint is identical to standard LoRA/LoCon. Triggered by `algo=gora`.
  * See [docs/Algo-List.md](docs/Algo-List.md) and [docs/Algo-Details.md](docs/Algo-Details.md) for details.
  * Requires a precomputation phase via `LycorisNetwork.prepare_gora()` before training.
  * Ref: [GoRA: Gradient-driven Adaptive Low Rank Adaptation](https://arxiv.org/abs/2502.12171)
* **O-LoRA: Orthogonal Low-Rank Adaptation**: Multi-task LoRA with orthogonality loss between task subspaces. Supports adding tasks dynamically and merging frozen tasks into base weights. Triggered via `olora=True` on LoCon modules.

#### Improvements

* Better logging for GoRA modules
* GoRA config enforcement (alpha = dim, rsLoRA always on)
* Proper recommended defaults for GoRA hyperparameters

#### Bug fixes

* Fix GoRA `org_weight_gpu` reference in forward pass
* Fix logging for non-keyword argument cases
* `use_scalar` enforced only with `use_orthogonal_weights`

## 2026/05/14 (dev branch)

#### New Features

* **PiSSA for LoCon**: Principal Singular Values and Singular Vectors Adaptation initialization. Uses SVD of pre-trained weights to initialize LoRA adapters. Supports fast randomized SVD and conversion to portable LoRA format on save.
* **SVD Segment Initialization**: Flexible SVD-based init supporting "top", "last", or "middle" singular vector segments for LoCon modules.
* **Orthogonal init/weights split**: `use_orthogonal_init` separated from `use_orthogonal_weights`, allowing orthogonal initialization without runtime orthogonalization overhead.

#### Improvements

* Adjust scalar default value from 1.0 to 0.1
* Migrate build system to pyproject.toml and uv
* Align regularization dims and learning rates

## 2026/03/15-18 (3.2.0 series)

#### New Features

* Support for module-level dimension and learning rate assignment
* LoRA-plus learning rate scaling for T-LoRA

#### Improvements

* Default `conv_lora_dim` to zero
* Rename presets for clarity

#### Bug fixes

* Revert incompatible lora-plus on T-LoRA p_layer


## 2025/04/23 update to 3.2.0

#### New Features

* Support lora-plus learning rate scaling
* Support HunYuanVideo model and Wan2.1 model
* LyCORIS now have `onfly_merge` and `onfly_restore` method. Which can be used in inference time to merge the weights of LyCORIS into the original model. This will save the memory and speed up the inference time.

#### Improvements

* [BREAKING CHANGES] Now LyCORIS will use `wd_on_output=True` by default. This will make the weight norm more consistent with the original paper.

#### Bug fixes

* `bypass_mode=False` will turn off the bypass mode correctly now.

## 2024/12/09 update to 3.1.1

#### New Features

* use `wd_on_output=True` can enable "correct" weight-decomposition implementation which use the output dimension of weight to calc the norm. The original implementation in LyCORIS calculate things on input dimension due to ambiguos annotation in paper.

#### Improvements

* BOFT now have more efficient implementation which avoid einops.rearrange.
* `.merge_to()` will automatically match the device and dtype now.

#### Bug fixes

* `scale_weight_norm` working correctly now.

## 2024/10/02 update to 3.1.0

### Highlights

* Support all the quantized linear layer by automatic detecting method
* Support Flux in Kohya-ss/sd-scripts
* Support wildcard matching for select layers in preset

### Full change log

#### New Features

* Support Flux
* Support any quantized linear layer such as torchao
* Refined Functional API to support drop-in replacement between different algorithms
* Support wildcard for name matching in preset

#### Bug fixes

* fix bugs in loading function of BOFT/OFT
* fix bugs in loading function of LoKr
* fix wrong behaviour of weight-decomposition when multiplier != 1

#### Improvements

* Improve the coverage of unit-test

## 2024/06/29 update to 3.0.0 - Brand New Functional API, Parametrize API and Module API

### The reasons of 3.0.0

We reconstruct the whole library with new Class definition and brand new Functional API system.

We also removed lot of redundant/unused modules.

Since the whole library are changed significantly. We decide to call it 3.0.0 as a new major version.

### Major Changes

* New Module API
* Add Parametrize API
* Add Functional API
  * LoCon/LoHa/LoKr/Diag-OFT/BOFT only.
* Remove optional deps from install_requires
* Remove lot of redundant/deprecated modules
* Better testing
* HunYuan DiT Support ([PR](https://github.com/kohya-ss/sd-scripts/pull/1378) in kohya-ss/sd-scripts)

### Full change log

#### New Features

* LyCORIS now have consistent API for different algorithm like `bypass_forward_diff` or `get_diff_weight` method. Developers of other project can utilize these API to do more tricks or integrate LyCORIS into their framework more easily.
* LyCORIS now have parametrize API which utilize `torch.nn.utils.parametrize.register_parametrization` to directly patch individual parameters. Which can be useful for MHA layer or other tricky modules.
  * Currently only support 2~5D tensors. And LyCORIS will pretend these weights are weight of Linear/Conv1,2,3D then send it into LyCORIS modules
  * More native implementation or more detailed control will be added in the future.
* LyCORIS now have functional API. Developers who prefer functional more than Module things can utilize this feature.
  * Functional API also allow developers who don't want to introduce new dependencies. Just copy-paste the source code and utilizing it. (with Apache-2 License, directly copy-paste is totally allowed)
* Add support for Conv1d and Conv3d module on LoCon/LoHa/LoKr/Full/OFT/BOFT/GLoRA (not All algo in LyCORIS support them, you may receive error when apply unsopported algo), support inherited module (for example: `LoRACompatibleConv` or `LoRACompatibleLinear` from [`huggingface/diffusers`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/lora.py))
* HunYuan DiT support.

#### Improvements, Fixes, Slight Changes

* Drop dependencies related to kohya-ss/sd-scripts:
  * We now take kohya-ss/sd-scripts as optional dependency
  * Which means `transformers`, `diffusers` and anything related to kohya are all optional deps now.
* The definition of dropout and rank_dropout in each algorithm are changed. Since some concept of original rank_dropout in the lora of kohya-ss/sd-script is hard to applied to other algorithm. We can only design the dropout for each module seperatedly.
* `apply_max_norm` issue are all fixed.
* DyLoRA, (IA)^3, GLoRA are all rewritten and support Linear/Conv1,2,3d.
* (IA)^3, GLoRA, Diag-OFT, BOFT are supported in `create_lycoris_from_weights`
  * `lycoris.kohya.create_network_from_weights` also support them as well.
* Fix wrong implementation of BOFT.
* `create_lycoris_from_weights` and `create_network_from_weights` now have correct logging infos.
* `get_module` and `make_module` are moved into modules' API.

#### Deprecation

* HCP modules are dropped. We will wait until HCP have better wrapper API.
* HyperNetwork-related modules like `hypernet/`, `attention.py`, `lilora.py` are removed.
* Uncompleted GLoKr are removed.
* code copied from kohya-ss/sd-scripts are removed. The original sd-scripts repo is now an optional dependency.

---

## 2024/03/15 update to 2.2.0 - QLyCORIS and DoRA

#### New Algo

* DoRA
  * Ref: [DoRA: Weight-Decomposed Low-Rank Adaptation](https://github.com/KohakuBlueleaf/LyCORIS)
* Weight decompose for LoHa and LoKr. (A.K.A DoHa/DoKr)
  * DoRA/DoHa/DoKr will require smaller Learning rate!

#### New Features

* Support "bypass" (a.k.a. adapter) mode for LoHa/LoKr/OFT/BOFT
  * LoHa will require 2xFLOPs since we rebuild full diff weight and then do one more forward.
  * LoKr, OFT, BOFT should be more efficient than LoHa in bypass mode.
* Support [bnb 8bit/4bit Linear layer](https://github.com/TimDettmers/bitsandbytes) (a.k.a. QLyCORIS) with LoHa/LoKr/OFT/BOFT.
  * This will force module to enable bypass mode.

#### Fixes, slight changes

* Refine some details about code quality. Based on the report from GitRoll. (Thx you gitroll!)
* Remove redundant calculation in BOFT
* rank_dropout has been removed from OFT/BOFT temporarily untill we ensure how to apply it.
* Fix bugs in lokr when `lokr_w1_a` not exist.
* Fix bugs in conversion scritps.

## 2024/02/18 update to 2.1.0

#### New Algo

* [BOFT (Butterfly OFT)](https://arxiv.org/abs/2311.06243)

#### Improvements

* Faster, better extract script
* support kohya-ss/sd-scripts image gen
* support regex name in kohya-ss/sd-scripts
* support resume on:
  * full
  * loha
  * oft
  * boft
* Add logger into LyCORIS

#### Fixes, slight changes

* Update HCP convert for the case where only UNet or TE is trained.
* Change arg names for conversion scripts.
* Fix wrong TE prefix in merge scripts.
* Fix warnings and confusing logging.

## 2023/12/15 quick fixes of 2.0.2

* Fix bugs in full module.
* Related: Fix bugs in `stable-diffusion-webui/extensions-builtin/Lora`
  * The [PR](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/14300)

## 2023/12/14 quick fixes of 2.0.1

* Support merge sdxl loras which trained on plain diffusers with Kohya's LoRA implementation.
  * Can be found in LECO or other similar projects.
* Refactor the batch convert scripts for pivotal bundle and hcp.
* Change the class name `lycoris.kohya.LycorisNetwork` to `lycoris.kohya.LycorisNetworkKohya` to avoid confusion.
* Fix bugs in merge scripts for Norm module and LoKr module.
* Fix bugs in scaled weight norms of OFT.
* Fix bugs in extract scripts for SDXL.
* Fix bugs in full module which consume 2x vram.
* Fix bugs in `create_network_from_weights` which caused bugs in "resume" feature for SDXL.

## 2023/12/02 update to 2.0.0

* Start supporting [HCP-Diffusion](https://github.com/IrisRainbowNeko/HCP-Diffusion) (The reason to name this version "2.0.0")
  * Now LyCORIS support LoHa/LoKr/Diag-OFT algorithm in HCP-Diffusion
  * Add Pivotal tuning utilities
  * Add hcp convert utilities
  * Have no plan at this time to support full/lora and train_norms since HCP can do them natively
* Add Diag-OFT modules
* Add standalone usage support
  * Can wrap any pytorch module which contains Linear/Conv2d/LayerNorm/GroupNorm modules
  * Will support more module in the future
* Add SDXL support in Merge script
* Add SDXL support in Extract-locon
* More efficient (speed/vram) implementation for full module
* Better implementation of custom state_dict
* Fix errors of dropouts
* Fix errors of apply_max_norms
* Fix errors of resume

---

## 2023/09/27 update to 1.9.0

* Add norm modules (for training LayerNorm and GroupNorm, which should be good for style)
* Add full modules (So you can "native finetune" with lycoris now, should be convinient to try different weight)
* Add preset config system
* Add custom config system
* Support resuming from models
* Merge script support norm and full modules
* Fix errors with optional requirements
* Fix errors with not necessary import
* Fix wrong factorization behaviours

## 2023/07/27 update to 1.8.2

* Update utils in kohya-ss/sd-scripts

## 2023/07/27 update to 1.8.1

* Add config/preset system
* Improve the project structure

## 2023/07/19 update to 1.8.0

* reimplement weight init method
* implement HyperDreamBooth into LyCORIS
* better file structure

## 2023/06/28 update to 1.7.1

* **rearrange the version format, previous 0.1.7 should be 1.7.0**
* fix the bug in scale weight norm

## 2023/06/26 Update to 0.1.7

* Add support for rank_dropout and module_dropout on LoCon/LoHa/LoKr
* Add support for scale_weight_norms on LoCon/LoHa/LoKr
* Will support SDXL on 0.1.8 (you can follow the dev branch)

## 2023/06/04 update to 0.1.6

* add dylora and IA^3 algorithm

## 2023/03/29 Update to 0.1.4

* cp decomposition is default to disable now
* add 4 more layer to train (conv_in/out, time_embedding)

## 2023/03/12 Update to 0.1.0

* Add cp-decomposition implementation for convolution layer
  * Both LoRA(LoCon) and LoHa can use this more parameter-efficient decomposition
* Add sparse bias for extracted LoRA
  * Will add to training in the future (Maybe)
* Change weight initialization method in LoHa
  * Use lower std to avoid loss to go high or NaN when using normal lr (like 0.5 in Dadap)
