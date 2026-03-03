"""
MLX visual encoder for Qwen3.5-VL models.

Qwen3.5-VL uses a ViT-based visual encoder (patch embed + positional embed +
12 transformer blocks + spatial merger). mlx-lm's qwen3_5.py strips the visual
encoder via sanitize() — this module loads and attaches it.

Architecture (0.8B vision_config):
  hidden_size: 768, depth: 12, num_heads: 12, intermediate_size: 3072
  out_hidden_size: 1024 (matches text hidden_size for 0.8B)
  patch_size: 16 × 16, temporal_patch_size: 2 (video; images get 2-frame dup)
  spatial_merge_size: 2 (2×2 = 4 patches merged per token to LLM)
  deepstack_visual_indexes: [] (disabled for 0.8B — simplifies forward pass)

Weight key mapping (safetensors → MLX module):
  model.visual.patch_embed.proj.weight (768,3,2,16,16) → patch_embed.weight (768,1536)
  model.visual.patch_embed.proj.bias   (768,)          → patch_embed.bias
  model.visual.pos_embed.weight        (2304,768)       → pos_embed.weight
  model.visual.blocks.{i}.norm1.*      (768,)           → blocks[i].norm1.*
  model.visual.blocks.{i}.attn.qkv.*  (2304,768)       → blocks[i].attn.qkv.*
  model.visual.blocks.{i}.attn.proj.* (768,768)        → blocks[i].attn.proj.*
  model.visual.blocks.{i}.norm2.*      (768,)           → blocks[i].norm2.*
  model.visual.blocks.{i}.mlp.linear_fc1.* (3072,768)  → blocks[i].mlp.linear_fc1.*
  model.visual.blocks.{i}.mlp.linear_fc2.* (768,3072)  → blocks[i].mlp.linear_fc2.*
  model.visual.merger.norm.*           (768,)           → merger.norm.*
  model.visual.merger.linear_fc1.*     (3072,3072)      → merger.linear_fc1.*
  model.visual.merger.linear_fc2.*     (1024,3072)      → merger.linear_fc2.*
"""
from __future__ import annotations

import glob
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Qwen35VisionConfig:
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_heads: int = 12
    depth: int = 12
    out_hidden_size: int = 1024
    in_channels: int = 3
    patch_size: int = 16
    temporal_patch_size: int = 2
    spatial_merge_size: int = 2
    num_position_embeddings: int = 2304

    @classmethod
    def from_dict(cls, d: dict) -> "Qwen35VisionConfig":
        fields = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in fields})


# ---------------------------------------------------------------------------
# Vision encoder modules
# ---------------------------------------------------------------------------

class Qwen35VisionAttention(nn.Module):
    """Standard multi-head attention with fused QKV projection (no RoPE)."""

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        B, L, D = x.shape
        # (B, L, 3*H*d) → (B, L, 3, num_heads, head_dim)
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        # → (3, B, num_heads, L, head_dim)
        qkv = qkv.transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scale = 1.0 / math.sqrt(self.head_dim)
        # (B, num_heads, L, L)
        attn = (q @ k.transpose(0, 1, 3, 2)) * scale
        attn = mx.softmax(attn, axis=-1)
        # (B, num_heads, L, head_dim) → (B, L, D)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, L, D)
        return self.proj(out)


class Qwen35VisionMLP(nn.Module):
    """Two-layer MLP: fc1 → GELU → fc2."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.linear_fc1 = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(intermediate_size, hidden_size, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear_fc2(nn.gelu(self.linear_fc1(x)))


class Qwen35VisionBlock(nn.Module):
    """Single ViT block: prenorm attn + prenorm MLP with residuals."""

    def __init__(self, config: Qwen35VisionConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.attn = Qwen35VisionAttention(config.hidden_size, config.num_heads)
        self.norm2 = nn.LayerNorm(config.hidden_size)
        self.mlp = Qwen35VisionMLP(config.hidden_size, config.intermediate_size)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Qwen35VisionMerger(nn.Module):
    """Spatial 2×2 patch merger: 4 adjacent tokens → 1 text token.

    Input:  [N, hidden_size]           (all patch tokens, in grid order)
    Step 1: norm per-token             [N, hidden_size]
    Step 2: spatial merge 2×2         [N/4, hidden_size * 4]  = [N/4, 3072]
    Step 3: linear_fc1 + GELU         [N/4, 3072]
    Step 4: linear_fc2                 [N/4, out_hidden_size]
    """

    def __init__(self, config: Qwen35VisionConfig):
        super().__init__()
        merge_in = config.hidden_size * config.spatial_merge_size ** 2  # 768*4=3072
        self.norm = nn.LayerNorm(config.hidden_size)
        self.linear_fc1 = nn.Linear(merge_in, merge_in, bias=True)
        self.linear_fc2 = nn.Linear(merge_in, config.out_hidden_size, bias=True)
        self._merge_in = merge_in

    def __call__(self, x: mx.array) -> mx.array:
        x = self.norm(x)                        # [N, 768]
        N, D = x.shape
        x = x.reshape(N // 4, self._merge_in)   # [N/4, 3072]
        x = nn.gelu(self.linear_fc1(x))
        return self.linear_fc2(x)               # [N/4, 1024]


class Qwen35VisionEncoder(nn.Module):
    """Full Qwen3.5 visual encoder.

    Processes pre-patchified images (as produced by Qwen3VLImageProcessor) into
    visual token embeddings aligned with the text model's hidden dimension.

    Input:
        pixel_values: [N_patches, in_channels * temporal_patch_size * patch_H * patch_W]
                      = [N, 1536]  for standard images (in_channels=3, T=2, H=W=16)
        position_ids: [N] int32 — position indices into learned pos_embed table

    Output: [N_tokens, out_hidden_size]  where N_tokens = N / 4 after merger
    """

    def __init__(self, config: Qwen35VisionConfig):
        super().__init__()
        patch_in_dim = (
            config.in_channels
            * config.temporal_patch_size
            * config.patch_size
            * config.patch_size
        )
        # Patch embedding: functionally a linear projection on pre-patchified pixels.
        # Weight stored as Conv3d (768, 3, 2, 16, 16) → reshaped to (768, 1536) on load.
        self.patch_embed = nn.Linear(patch_in_dim, config.hidden_size, bias=True)
        # Learned positional embedding (not rotary — 0.8B uses standard embed table).
        self.pos_embed = nn.Embedding(config.num_position_embeddings, config.hidden_size)
        self.blocks = [Qwen35VisionBlock(config) for _ in range(config.depth)]
        self.merger = Qwen35VisionMerger(config)

    def __call__(self, pixel_values: mx.array, position_ids: mx.array) -> mx.array:
        """
        pixel_values: [N, 1536]  (pre-patchified by Qwen3VLImageProcessor)
        position_ids: [N]
        Returns:      [N/4, out_hidden_size]
        """
        x = self.patch_embed(pixel_values)  # [N, 768]
        x = x + self.pos_embed(position_ids)  # [N, 768]
        for block in self.blocks:
            x = block(x)
        return self.merger(x)  # [N/4, out_hidden_size]


# ---------------------------------------------------------------------------
# Weight loading utilities
# ---------------------------------------------------------------------------

def _sanitize_visual_weights(raw_weights: dict) -> dict:
    """Extract and remap model.visual.* weights for Qwen35VisionEncoder.

    Key remapping:
      model.visual.{rest} → {rest}
      patch_embed.proj.weight  (5D Conv3d) → patch_embed.weight  (2D Linear)
      patch_embed.proj.bias    → patch_embed.bias
    """
    visual = {}
    for key, val in raw_weights.items():
        if not key.startswith("model.visual."):
            continue
        rest = key[len("model.visual."):]

        # Patch embed: Conv3d weight (out, in, T, H, W) → Linear weight (out, in*T*H*W)
        if rest == "patch_embed.proj.weight":
            val = val.reshape(val.shape[0], -1)
            rest = "patch_embed.weight"
        elif rest == "patch_embed.proj.bias":
            rest = "patch_embed.bias"

        visual[rest] = val
    return visual


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_qwen35_vl_model(model_path: str):
    """Load Qwen3.5-VL model with visual encoder attached as model.visual.

    This supplements mlx-lm's text-only loading (which strips visual weights)
    by instantiating a Qwen35VisionEncoder and loading its weights from the
    same safetensors files.

    Returns (model, tokenizer) — same interface as mlx_lm.load().
    model.visual is a Qwen35VisionEncoder callable as:
        visual_embeds = model.visual(pixel_values, position_ids)

    Only call this when vision training is needed. For text-only inference/training,
    use mlx_lm.load() directly (faster, lower memory).
    """
    from mlx_lm import load as mlx_load

    model_path = str(Path(model_path).expanduser().resolve())
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {model_path}")

    with config_path.open() as fh:
        full_config = json.load(fh)

    vision_cfg_dict = full_config.get("vision_config")
    if not vision_cfg_dict:
        raise ValueError(
            f"No vision_config in {config_path}. "
            "This model may not be a VL model."
        )

    # 1. Load text model via mlx-lm (strips visual weights — that's fine)
    model, tokenizer = mlx_load(model_path)

    # 2. Load all safetensors weights (includes visual) directly
    weight_files = sorted(glob.glob(str(Path(model_path) / "model*.safetensors")))
    if not weight_files:
        raise FileNotFoundError(f"No model*.safetensors found in {model_path}")

    raw_weights: dict = {}
    for wf in weight_files:
        raw_weights.update(mx.load(wf))

    # 3. Extract and remap visual weights
    visual_weights = _sanitize_visual_weights(raw_weights)
    if not visual_weights:
        raise RuntimeError(
            "No model.visual.* weights found in safetensors. "
            "The model may not have a visual encoder."
        )

    # 4. Instantiate visual encoder
    vision_config = Qwen35VisionConfig.from_dict(vision_cfg_dict)
    visual_encoder = Qwen35VisionEncoder(vision_config)

    # 5. Load weights into visual encoder
    visual_encoder.load_weights(list(visual_weights.items()))
    mx.eval(visual_encoder.parameters())

    # 6. Attach to model
    model.visual = visual_encoder

    return model, tokenizer


def is_qwen35_vl(model_path: str) -> bool:
    """Return True if the model at model_path has a vision_config (is VL)."""
    config_path = Path(model_path).expanduser().resolve() / "config.json"
    if not config_path.exists():
        return False
    try:
        with config_path.open() as fh:
            cfg = json.load(fh)
        return bool(cfg.get("vision_config"))
    except (json.JSONDecodeError, OSError):
        return False
