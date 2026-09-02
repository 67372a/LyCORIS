from __future__ import annotations

from .config_sdk import PresetConfig, AlgoOverride

FULL_UNET_MODULES = [
    "Transformer2DModel",
    "ResnetBlock2D",
    "Downsample2D",
    "Upsample2D",
    "HunYuanDiTBlock",  # HunYuanDiT
    "DoubleStreamBlock",  # Flux
    "SingleStreamBlock",  # Flux
    "SingleDiTBlock",  # SD3.5
    "MMDoubleStreamBlock",  # HunYuanVideo
    "MMSingleStreamBlock",  # HunYuanVideo
    "WanAttentionBlock",  # Wan
    "HunyuanVideoTransformerBlock",  # FramePack
    "HunyuanVideoSingleTransformerBlock",  # FramePack
    "JointTransformerBlock",  # lumina-image-2
    "FinalLayer",  # lumina-image-2, Anima
    "QwenImageTransformerBlock",  # Qwen
    "LensTransformerBlock",  # Lens
    "Ideogram4TransformerBlock",  # Ideogram 4
    "ZImageTransformerBlock",
    "AceStepEncoderLayer",
    "AceStepDiTLayer",
    "TextFusionBlock",  # Krea 2
    "Block",  # Anima
    "PatchEmbed",  # Anima
    "TimestepEmbedding",  # Anima
    "LLMAdapterTransformerBlock",  # Anima
]

FULL_UNET_NAMES = [
    "conv_in",
    "conv_out",
    "time_embedding.linear_1",
    "time_embedding.linear_2",
]

FULL_TEXT_ENCODER_MODULES = [
    "CLIPAttention",
    "CLIPSdpaAttention",
    "CLIPMLP",
    "MT5Block",
    "BertLayer",
    "Gemma2Attention",
    "Gemma2FlashAttention2",
    "Gemma2SdpaAttention",
    "Gemma2MLP",
    "Qwen3Attention",  # Anima / Qwen3
    "Qwen3FlashAttention2",  # Anima / Qwen3
    "Qwen3SdpaAttention",  # Anima / Qwen3
    "Qwen3MLP",  # Anima / Qwen3
]


ANIMA_QWEN_TEXT_ENCODER_MODULES = [
    "Qwen3Attention",
    "Qwen3MLP",
    "Qwen3SdpaAttention",
    "Qwen3FlashAttention2",
]


BUILTIN_PRESET_CONFIGS = {
    "full": PresetConfig(
        enable_conv=True,
        unet_target_module=FULL_UNET_MODULES,
        unet_target_name=FULL_UNET_NAMES,
        text_encoder_target_module=FULL_TEXT_ENCODER_MODULES,
        text_encoder_target_name=[],
    ),
    "full-lin": PresetConfig(
        enable_conv=False,
        unet_target_module=[
            "Transformer2DModel",
            "ResnetBlock2D",
            "HunYuanDiTBlock",
            "DoubleStreamBlock",
            "SingleStreamBlock",
            "SingleDiTBlock",
            "MMDoubleStreamBlock",  # HunYuanVideo
            "MMSingleStreamBlock",  # HunYuanVideo
            "WanAttentionBlock",  # Wan
            "HunyuanVideoTransformerBlock",  # FramePack
            "HunyuanVideoSingleTransformerBlock",  # FramePack
            "JointTransformerBlock",  # lumina-image-2
            "FinalLayer",  # lumina-image-2, Anima
            "QwenImageTransformerBlock",  # Qwen
            "LensTransformerBlock",  # Lens
            "Ideogram4TransformerBlock",  # Ideogram 4
            "ZImageTransformerBlock",
            "AceStepEncoderLayer",
            "AceStepDiTLayer",
            "TextFusionBlock",  # Krea 2
            "Block",  # Anima
            "PatchEmbed",  # Anima
            "TimestepEmbedding",  # Anima
            "LLMAdapterTransformerBlock",  # Anima
        ],
        unet_target_name=[
            "time_embedding.linear_1",
            "time_embedding.linear_2",
        ],
        text_encoder_target_module=FULL_TEXT_ENCODER_MODULES,
        text_encoder_target_name=[],
    ),
    "attn-mlp": PresetConfig(
        enable_conv=False,
        unet_target_module=[
            "Transformer2DModel",
            "HunYuanDiTBlock",
            "DoubleStreamBlock",
            "SingleStreamBlock",
            "SingleDiTBlock",
            "MMDoubleStreamBlock",  # HunYuanVideo
            "MMSingleStreamBlock",  # HunYuanVideo
            "WanAttentionBlock",  # Wan
            "HunyuanVideoTransformerBlock",  # FramePack
            "HunyuanVideoSingleTransformerBlock",  # FramePack
            "JointTransformerBlock",  # lumina-image-2
            "FinalLayer",  # lumina-image-2, Anima
            "QwenImageTransformerBlock",  # Qwen
            "LensTransformerBlock",  # Lens
            "Ideogram4TransformerBlock",  # Ideogram 4
            "ZImageTransformerBlock",
            "AceStepEncoderLayer",
            "AceStepDiTLayer",
            "TextFusionBlock",  # Krea 2
            "Block",  # Anima
            "PatchEmbed",  # Anima
            "TimestepEmbedding",  # Anima
            "LLMAdapterTransformerBlock",  # Anima
        ],
        unet_target_name=[],
        text_encoder_target_module=FULL_TEXT_ENCODER_MODULES,
        text_encoder_target_name=[],
    ),
    "attn-only": PresetConfig(
        enable_conv=False,
        unet_target_module=[
            "CrossAttention",
            "SelfAttention",
        ],
        unet_target_name=[],
        text_encoder_target_module=[
            "CLIPAttention",
            "CLIPSdpaAttention",
            "BertAttention",
            "MT5LayerSelfAttention",
            "Gemma2Attention",
            "Gemma2FlashAttention2",
            "Gemma2SdpaAttention",
            "Qwen3Attention",
            "Qwen3SdpaAttention",
            "Qwen3FlashAttention2",
        ],
        text_encoder_target_name=[],
    ),
    "unet-only": PresetConfig(
        enable_conv=True,
        unet_target_module=FULL_UNET_MODULES,
        unet_target_name=FULL_UNET_NAMES,
        text_encoder_target_module=[],
        text_encoder_target_name=[],
    ),
    "unet-transformer-only": PresetConfig(
        enable_conv=False,
        unet_target_module=[
            "Transformer2DModel",
            "HunYuanDiTBlock",
            "DoubleStreamBlock",
            "SingleStreamBlock",
            "SingleDiTBlock",
            "MMDoubleStreamBlock",  # HunYuanVideo
            "MMSingleStreamBlock",  # HunYuanVideo
            "WanAttentionBlock",  # Wan
            "HunyuanVideoTransformerBlock",  # FramePack
            "HunyuanVideoSingleTransformerBlock",  # FramePack
            "JointTransformerBlock",  # lumina-image-2
            "FinalLayer",  # lumina-image-2, Anima
            "QwenImageTransformerBlock",  # Qwen
            "LensTransformerBlock",  # Lens
            "Ideogram4TransformerBlock",  # Ideogram 4
            "ZImageTransformerBlock",
            "AceStepEncoderLayer",
            "AceStepDiTLayer",
            "TextFusionBlock",  # Krea 2
            "Block",  # Anima
            "PatchEmbed",  # Anima
            "TimestepEmbedding",  # Anima
            "LLMAdapterTransformerBlock",  # Anima
        ],
        unet_target_name=[],
        text_encoder_target_module=[],
        text_encoder_target_name=[],
    ),
    "unet-convblock-only": PresetConfig(
        enable_conv=True,
        unet_target_module=["ResnetBlock2D", "Downsample2D", "Upsample2D"],
        unet_target_name=[
            "conv_in",
            "conv_out",
        ],
        text_encoder_target_module=[],
        text_encoder_target_name=[],
    ),
    "anima": PresetConfig(
        enable_conv=False,
        unet_target_module=["Block"],
        unet_target_name=[],
        text_encoder_target_module=[],
        text_encoder_target_name=[],
        exclude_name=[r".*(_modulation|_embedder|final_layer).*"],
    ),
    "anima-exp": PresetConfig(
        enable_conv=False,
        unet_target_module=["Block"],
        unet_target_name=[],
        text_encoder_target_module=[],
        text_encoder_target_name=[],
        exclude_name=[r".*(_embedder|final_layer).*"],
    ),
    "anima-qwen": PresetConfig(
        enable_conv=False,
        unet_target_module=["Block"],
        unet_target_name=[],
        text_encoder_target_module=ANIMA_QWEN_TEXT_ENCODER_MODULES,
        text_encoder_target_name=[],
        exclude_name=[r".*(_modulation|_embedder|final_layer).*"],
    ),
    "anima-qwen-exp": PresetConfig(
        enable_conv=False,
        unet_target_module=["Block"],
        unet_target_name=[],
        text_encoder_target_module=ANIMA_QWEN_TEXT_ENCODER_MODULES,
        text_encoder_target_name=[],
        exclude_name=[r".*(_embedder|final_layer).*"],
    ),
    "anima-inpaint": PresetConfig(
        enable_conv=False,
        unet_target_module=["Block", "PatchEmbed", "TimestepEmbedding", "FinalLayer"],
        unet_target_name=[],
        text_encoder_target_module=[],
        text_encoder_target_name=[],
        extra={
            "network_reg_lrs": {
                r".*blocks\.(1[89]|2[0-7])\..*": 6e-05,
                r".*blocks\.([0-9]|1[0-7])\..*": 2.5e-05,
            },
            "network_reg_dims": {r".*blocks\.(1[89]|2[0-7])\..*": 16},
            "include_patterns": [r".*unet_blocks_([0-9]|1[0-9]|2[0-7])\..*"],
            "exclude_patterns": [r".*_te_layers_.*", r".*adaln_modulation.*"],
        },
    ),
    "anima-full-no-excludes": PresetConfig(
        enable_conv=False,
        unet_target_module=[
            "Block",
            "PatchEmbed",
            "TimestepEmbedding",
            "FinalLayer",
        ],
        unet_target_name=[],
        text_encoder_target_module=[],
        text_encoder_target_name=[],
        exclude_name=[],
    ),
    "ia3": PresetConfig(
        enable_conv=False,
        unet_target_module=[],
        unet_target_name=["to_k", "to_v", "ff.net.2"],
        text_encoder_target_module=[],
        text_encoder_target_name=["k_proj", "v_proj", "mlp.fc2"],
        name_algo_map={
            "mlp.fc2": AlgoOverride(options={"train_on_input": True}),
            "ff.net.2": AlgoOverride(options={"train_on_input": True}),
        },
    ),
}


PRESET = {name: cfg.to_dict() for name, cfg in BUILTIN_PRESET_CONFIGS.items()}


def list_builtin_presets() -> dict[str, PresetConfig]:
    return BUILTIN_PRESET_CONFIGS.copy()
