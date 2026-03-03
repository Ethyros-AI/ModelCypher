#!/usr/bin/env python3
# Copyright (C) 2026 EthyrosAI LLC / Jason Kempf
#
# VL Training Data Ingestion Script
#
# Converts image files + text into Qwen3.5-VL compatible JSONL training entries.
# Terminal drag-and-drop of image files gives the path directly as CLI argument.
#
# Usage (single image):
#   poetry run python scripts/ingest_vl_images.py \
#     --image /path/to/cat.jpg \
#     --text "A black cat sitting on a wooden table." \
#     --output data/training/vl_train.jsonl
#
# Usage (batch — directory with images + matching .txt caption files):
#   poetry run python scripts/ingest_vl_images.py \
#     --image-dir /path/to/images/ \
#     --output data/training/vl_train.jsonl
#
# VQA-style (custom prompt):
#   poetry run python scripts/ingest_vl_images.py \
#     --image /path/to/product.jpg \
#     --text "ModelCypher logo on a dark background." \
#     --prompt "What brand is shown in this image?" \
#     --output data/training/vl_brand.jsonl
#
# Output format (one JSON per line):
#   {"text": "<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n{text}<|im_end|>",
#    "image_path": "/absolute/path/to/image.jpg"}

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Qwen3.5-VL image token markers (match tokenizer special tokens)
VISION_START = "<|vision_start|>"
IMAGE_PAD = "<|image_pad|>"
VISION_END = "<|vision_end|>"
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"

DEFAULT_PROMPT = "Describe this image in detail."

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff", ".tif"}


# ---------------------------------------------------------------------------
# Format builder
# ---------------------------------------------------------------------------

def make_vl_text(caption: str, prompt: str = DEFAULT_PROMPT) -> str:
    """Build the chat-format text field for a VL training sample.

    Format matches Qwen3.5-VL's expected instruction-tuning template.
    The <|image_pad|> token (id=151655) is the placeholder that gets replaced
    with visual encoder embeddings during the forward pass.
    """
    user_turn = (
        f"{IM_START}user\n"
        f"{VISION_START}{IMAGE_PAD}{VISION_END}\n"
        f"{prompt}{IM_END}\n"
    )
    assistant_turn = f"{IM_START}assistant\n{caption}{IM_END}"
    return user_turn + assistant_turn


def make_entry(image_path: str, caption: str, prompt: str = DEFAULT_PROMPT) -> dict:
    """Create a single VL JSONL entry."""
    return {
        "text": make_vl_text(caption, prompt),
        "image_path": str(Path(image_path).expanduser().resolve()),
    }


# ---------------------------------------------------------------------------
# Batch ingestion from directory
# ---------------------------------------------------------------------------

def _ingest_directory(image_dir: Path, prompt: str) -> list[dict]:
    """Ingest all images in a directory. Caption lookup order:
    1. Matching .txt file (image.jpg → image.txt)
    2. Matching .json file with "caption" key (image.jpg → image.json)
    3. Skip if no caption found (warns and continues)
    """
    entries = []
    image_files = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_files:
        print(f"[warn] No image files found in {image_dir}", file=sys.stderr)
        return entries

    for img_path in image_files:
        caption = None

        txt_path = img_path.with_suffix(".txt")
        if txt_path.exists():
            caption = txt_path.read_text(encoding="utf-8").strip()

        if caption is None:
            json_path = img_path.with_suffix(".json")
            if json_path.exists():
                try:
                    meta = json.loads(json_path.read_text())
                    caption = meta.get("caption") or meta.get("text")
                except json.JSONDecodeError:
                    pass

        if caption is None:
            print(f"[skip] No caption for {img_path.name} (no .txt or .json)", file=sys.stderr)
            continue

        entries.append(make_entry(str(img_path), caption, prompt))

    return entries


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert images + text to Qwen3.5-VL JSONL training entries"
    )

    # Source: single image or directory
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--image",
        metavar="PATH",
        help="Single image file (drag-and-drop the file path here)",
    )
    src.add_argument(
        "--image-dir",
        metavar="DIR",
        help="Directory of images; captions from matching .txt or .json files",
    )

    # Caption for single-image mode
    parser.add_argument(
        "--text",
        metavar="CAPTION",
        help="Caption / answer text (required with --image)",
    )

    # Output
    parser.add_argument(
        "--output",
        metavar="PATH",
        default="data/training/vl_train.jsonl",
        help="Output JSONL file (appended if exists, default: data/training/vl_train.jsonl)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file instead of appending",
    )

    # Optional prompt
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help=f"Instruction prompt for all entries (default: '{DEFAULT_PROMPT}')",
    )

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # Build entries
    entries: list[dict] = []

    if args.image:
        img_path = Path(args.image)
        if not img_path.exists():
            print(f"[error] Image not found: {img_path}", file=sys.stderr)
            sys.exit(1)
        if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
            print(
                f"[warn] Extension {img_path.suffix!r} not a known image type. Proceeding anyway.",
                file=sys.stderr,
            )
        if not args.text:
            print("[error] --text CAPTION is required with --image", file=sys.stderr)
            sys.exit(1)
        entries.append(make_entry(str(img_path), args.text, args.prompt))

    elif args.image_dir:
        image_dir = Path(args.image_dir)
        if not image_dir.is_dir():
            print(f"[error] Not a directory: {image_dir}", file=sys.stderr)
            sys.exit(1)
        entries = _ingest_directory(image_dir, args.prompt)

    if not entries:
        print("[error] No entries produced.", file=sys.stderr)
        sys.exit(1)

    # Write output
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mode = "w" if args.overwrite else "a"
    with out_path.open(mode, encoding="utf-8") as fh:
        for entry in entries:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    action = "Wrote" if args.overwrite else "Appended"
    print(f"{action} {len(entries)} entries → {out_path}")
    if len(entries) == 1:
        print(f"  image: {entries[0]['image_path']}")
        preview = entries[0]["text"][:120].replace("\n", "\\n")
        print(f"  text:  {preview}...")


if __name__ == "__main__":
    main()
