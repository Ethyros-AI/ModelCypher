"""
Image preprocessing pipeline for Qwen3.5-VL training.

Wraps transformers.Qwen2VLImageProcessor (slow, no torchvision required) and
converts outputs to MLX arrays.

Output contract:
  pixel_values: mx.array [N_patches, 1536]  — pre-patchified pixel data
                where 1536 = in_channels(3) × temporal_patch_size(2) × patch(16) × patch(16)
  position_ids: mx.array [N_patches] int32  — sequential position indices for
                the visual encoder's learned pos_embed table (size 2304×768)

Usage:
    proc = VLPreprocessor.from_model_path(model_path)
    pixel_values, position_ids = proc.preprocess_image("cat.jpg")
    visual_embeds = model.visual(pixel_values, position_ids)  # [N/4, 1024]
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import mlx.core as mx
import numpy as np


class VLPreprocessor:
    """Image preprocessor for Qwen3.5-VL.

    Uses transformers.Qwen2VLImageProcessor (the slow/pure-Python variant —
    does not require torchvision).
    """

    def __init__(self, hf_processor):
        """
        Args:
            hf_processor: transformers.Qwen2VLImageProcessor instance.
        """
        self._proc = hf_processor

    @classmethod
    def from_model_path(cls, model_path: str) -> "VLPreprocessor":
        """Instantiate from a model directory (reads preprocessor_config.json).

        Args:
            model_path: Path to the mlx-community model directory.
        """
        try:
            from transformers import Qwen2VLImageProcessor
        except ImportError as exc:
            raise ImportError(
                "transformers is required for VL image preprocessing. "
                "Install with: pip install transformers"
            ) from exc

        model_path = str(Path(model_path).expanduser().resolve())
        proc = Qwen2VLImageProcessor.from_pretrained(model_path)
        return cls(proc)

    def preprocess_image(self, image_path: str) -> tuple[mx.array, mx.array]:
        """Process a single image file into MLX arrays.

        Args:
            image_path: Absolute or relative path to an image file.

        Returns:
            pixel_values: [N_patches, 1536] bfloat16
                Pre-patchified pixel data. N_patches = T × H × W from grid_thw.
            position_ids: [N_patches] int32
                Sequential position indices (0, 1, ..., N-1) for the visual
                encoder's learned pos_embed table.
        """
        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError(
                "Pillow is required for image loading. Install with: pip install Pillow"
            ) from exc

        img = Image.open(image_path).convert("RGB")
        out = self._proc(images=[img], return_tensors="np")

        pv_np = out["pixel_values"].astype(np.float32)  # (N, 1536)
        pixel_values = mx.array(pv_np, dtype=mx.bfloat16)

        n_patches = pv_np.shape[0]
        position_ids = mx.arange(n_patches, dtype=mx.int32)

        return pixel_values, position_ids

    def preprocess_batch(
        self, image_paths: list[str]
    ) -> list[tuple[mx.array, mx.array]]:
        """Process multiple images. Returns list of (pixel_values, position_ids)."""
        return [self.preprocess_image(p) for p in image_paths]

    def prepare_vl_dataset(
        self,
        samples: list[dict],
        tokenizer,
    ) -> list[dict]:
        """Tokenize VL samples and attach preprocessed image tensors.

        Expects each sample to have:
          - "text": str — the full chat-format text with image placeholder tokens
          - "image_path": str — path to the image file

        Returns list of dicts with keys:
          - "tokens": mx.array [L] int32
          - "pixel_values": mx.array [N, 1536] bfloat16  (or None for text-only)
          - "position_ids": mx.array [N] int32            (or None for text-only)
          - "n_text_tokens": int — total token count (for sequence length info)
        """
        eos_id = getattr(tokenizer, "eos_token_id", None)
        result = []

        for sample in samples:
            text = sample.get("text", "")
            if not isinstance(text, str) or not text:
                continue

            tokens = tokenizer.encode(text)
            if eos_id is not None and (not tokens or tokens[-1] != eos_id):
                tokens.append(eos_id)
            if len(tokens) < 2:
                continue

            tokens_mx = mx.array(tokens, dtype=mx.int32)

            image_path = sample.get("image_path")
            if image_path:
                pixel_values, position_ids = self.preprocess_image(image_path)
            else:
                pixel_values, position_ids = None, None

            result.append({
                "tokens": tokens_mx,
                "pixel_values": pixel_values,
                "position_ids": position_ids,
                "n_text_tokens": len(tokens),
            })

        return result
