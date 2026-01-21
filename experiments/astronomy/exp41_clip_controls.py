"""
Experiment 41: CLIP Encoding Controls

Validate the exp40 findings by comparing:
1. Wow! signal vs FRB spectrograms (also natural radio bursts)
2. Wow! signal vs shuffled version (same values, destroyed structure)
3. Wow! signal vs Gaussian noise matched to signal statistics

If Wow! is unique, it should differ from FRBs (not just from random noise).
If the result is just about having a "burst" structure, FRBs should look similar.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import h5py
from scipy.io import readsav
from scipy import linalg
import torch
from transformers import CLIPModel, CLIPProcessor
from PIL import Image

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def load_wow_signal():
    """Load the Wow! signal from IDL .sav file."""
    data_path = Path(__file__).parent / "data" / "famous_signals" / "wow_signal.sav"
    data = readsav(str(data_path))
    oseti = data['oseti'][0]
    signal = oseti['SNR'].astype(np.float64)
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
    return signal


def load_frb_signals(n_max=20):
    """Load FRB waterfall plots."""
    frb_dir = Path(__file__).parent / "data" / "raw"
    frb_files = sorted(frb_dir.glob("FRB*.h5"))[:n_max]

    signals = []
    for frb_path in frb_files:
        try:
            with h5py.File(frb_path, 'r') as f:
                wfall = f['frb']['wfall'][:]
                signals.append((frb_path.stem, wfall.astype(np.float64)))
        except Exception as e:
            print(f"Error loading {frb_path.stem}: {e}")

    return signals


def signal_to_image(signal, size=(224, 224)):
    """Convert signal matrix to RGB image for CLIP."""
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
    s_min, s_max = signal.min(), signal.max()
    if s_max - s_min > 0:
        normalized = (signal - s_min) / (s_max - s_min)
    else:
        normalized = np.zeros_like(signal)

    img_array = (normalized * 255).astype(np.uint8)
    img = Image.fromarray(img_array, mode='L')
    img = img.resize(size, Image.Resampling.LANCZOS)
    img = img.convert('RGB')
    return img


def compute_clip_embeddings(processor, model, images=None, texts=None):
    """Compute CLIP embeddings."""
    result = {}

    if images is not None:
        inputs = processor(images=images, return_tensors="pt", padding=True)
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        result['image_embeds'] = image_features.cpu().numpy()

    if texts is not None:
        inputs = processor(text=texts, return_tensors="pt", padding=True)
        with torch.no_grad():
            text_features = model.get_text_features(**inputs)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        result['text_embeds'] = text_features.cpu().numpy()

    return result


def main():
    print("=" * 60)
    print("Experiment 41: CLIP Controls - Validating Signal Uniqueness")
    print("=" * 60)

    # Load signals
    print("\n1. Loading signals...")
    wow = load_wow_signal()
    print(f"   Wow! shape: {wow.shape}")

    frbs = load_frb_signals(n_max=10)
    print(f"   Loaded {len(frbs)} FRB signals")

    # Create control signals
    print("\n2. Creating control signals...")

    # Shuffled Wow! (destroys temporal structure, keeps value distribution)
    wow_flat = wow.flatten()
    np.random.shuffle(wow_flat)
    wow_shuffled = wow_flat.reshape(wow.shape)
    print("   Created shuffled Wow! (destroyed structure)")

    # Gaussian noise matched to Wow! statistics
    wow_matched_noise = np.random.randn(*wow.shape) * wow.std() + wow.mean()
    print(f"   Created matched Gaussian noise (mean={wow.mean():.2f}, std={wow.std():.2f})")

    # Load CLIP
    print("\n3. Loading CLIP model...")
    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)

    # Define concepts
    concepts = [
        "a message", "a signal", "communication", "encoded message",
        "random noise", "chaotic noise", "static", "interference"
    ]

    # Get text embeddings
    text_embeds = compute_clip_embeddings(processor, model, texts=concepts)['text_embeds']

    # Compute embeddings for all signals
    print("\n4. Computing embeddings...")

    all_signals = {
        "Wow!": wow,
        "Wow! (shuffled)": wow_shuffled,
        "Matched noise": wow_matched_noise,
    }

    for name, sig in frbs[:5]:  # First 5 FRBs
        all_signals[name] = sig

    results = {}
    for name, sig in all_signals.items():
        img = signal_to_image(sig)
        embeds = compute_clip_embeddings(processor, model, images=[img])['image_embeds']
        similarities = (embeds @ text_embeds.T).squeeze()
        results[name] = {c: float(similarities[i]) for i, c in enumerate(concepts)}

    # Analysis
    print("\n5. Comparing signals...")

    # Key concepts
    info_concepts = ["a message", "a signal", "communication", "encoded message"]
    noise_concepts = ["random noise", "chaotic noise", "static", "interference"]

    print("\n   Signal                | Info sim | Noise sim | Diff")
    print("   " + "-" * 55)

    summary = {}
    for name, sims in results.items():
        info_sim = np.mean([sims[c] for c in info_concepts])
        noise_sim = np.mean([sims[c] for c in noise_concepts])
        diff = info_sim - noise_sim
        print(f"   {name:20s} | {info_sim:.4f}   | {noise_sim:.4f}   | {diff:+.4f}")
        summary[name] = {
            "info_similarity": float(info_sim),
            "noise_similarity": float(noise_sim),
            "difference": float(diff)
        }

    # Key question: Is Wow! unique among bursts?
    print("\n6. Key findings...")

    wow_diff = summary["Wow!"]["difference"]
    shuffled_diff = summary["Wow! (shuffled)"]["difference"]
    noise_diff = summary["Matched noise"]["difference"]

    print(f"\n   Wow! info-noise diff: {wow_diff:+.4f}")
    print(f"   Shuffled diff:        {shuffled_diff:+.4f}")
    print(f"   Matched noise diff:   {noise_diff:+.4f}")

    frb_diffs = [summary[name]["difference"] for name in summary if name.startswith("FRB")]
    if frb_diffs:
        print(f"   FRB mean diff:        {np.mean(frb_diffs):+.4f} ± {np.std(frb_diffs):.4f}")

        # Z-score of Wow! vs FRBs
        if np.std(frb_diffs) > 0:
            z_vs_frb = (wow_diff - np.mean(frb_diffs)) / np.std(frb_diffs)
            print(f"\n   Wow! vs FRBs z-score: {z_vs_frb:.2f}")

    # Save detailed results
    output = {
        "experiment": "exp41_clip_controls",
        "timestamp": datetime.now().isoformat(),
        "all_similarities": results,
        "summary": summary,
        "wow_info_noise_diff": float(wow_diff),
        "shuffled_diff": float(shuffled_diff),
        "noise_diff": float(noise_diff),
        "frb_diffs": frb_diffs,
    }

    output_path = RESULTS_DIR / "exp41_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n7. Results saved to {output_path}")


if __name__ == "__main__":
    main()
