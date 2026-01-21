"""
Experiment 40: Encode Wow! Signal via CLIP Vision Encoder

The key insight from user:
"why not run a lora training on the data to encode the shape?"

Simpler first approach: use PRETRAINED CLIP without additional training.
- CLIP learns to map images → shared semantic space
- A spectrogram IS an image
- Pass the Wow! signal spectrogram through CLIP's vision encoder
- See where it lands in semantic space relative to concepts

If the signal contains information, it should map to a coherent location
in semantic space, not random scatter.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np
from scipy.io import readsav
from scipy import linalg
import torch
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def load_wow_signal():
    """Load the Wow! signal from IDL .sav file."""
    data_path = Path(__file__).parent / "data" / "famous_signals" / "wow_signal.sav"
    if not data_path.exists():
        raise FileNotFoundError(f"Wow! signal not found at {data_path}")

    data = readsav(str(data_path))
    oseti = data['oseti'][0]
    signal = oseti['SNR']
    signal = signal.astype(np.float64)
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
    return signal


def signal_to_image(signal, size=(224, 224)):
    """Convert signal matrix to RGB image for CLIP.

    Args:
        signal: 2D numpy array (time x frequency)
        size: Target image size (H, W)

    Returns:
        PIL Image in RGB format
    """
    # Normalize to [0, 1]
    s_min, s_max = signal.min(), signal.max()
    if s_max - s_min > 0:
        normalized = (signal - s_min) / (s_max - s_min)
    else:
        normalized = np.zeros_like(signal)

    # Convert to uint8 [0, 255]
    img_array = (normalized * 255).astype(np.uint8)

    # Create PIL image (grayscale first)
    img = Image.fromarray(img_array, mode='L')

    # Resize to CLIP expected size
    img = img.resize(size, Image.Resampling.LANCZOS)

    # Convert to RGB (CLIP expects RGB)
    img = img.convert('RGB')

    return img


def compute_clip_embeddings(processor, model, images=None, texts=None):
    """Compute CLIP embeddings for images and/or texts.

    Returns:
        Dict with 'image_embeds' and/or 'text_embeds'
    """
    result = {}

    if images is not None:
        inputs = processor(images=images, return_tensors="pt", padding=True)
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
        # L2 normalize (CLIP convention)
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
    print("Experiment 40: CLIP Encoding of Wow! Signal")
    print("=" * 60)

    # Load Wow! signal
    print("\n1. Loading Wow! signal...")
    signal = load_wow_signal()
    print(f"   Signal shape: {signal.shape}")

    # Convert to image
    print("\n2. Converting signal to image...")
    wow_image = signal_to_image(signal)
    print(f"   Image size: {wow_image.size}")

    # Save image for inspection
    img_path = RESULTS_DIR / "wow_spectrogram.png"
    wow_image.save(img_path)
    print(f"   Saved to {img_path}")

    # Load CLIP
    print("\n3. Loading CLIP model...")
    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    print(f"   Loaded {model_name}")

    # Define semantic concepts to compare against
    print("\n4. Defining semantic concepts...")
    concepts = [
        # Information/signal concepts
        "a message",
        "a signal",
        "communication",
        "data transmission",
        "information pattern",
        "encoded message",
        "radio signal",

        # Natural phenomena
        "random noise",
        "static",
        "interference",
        "natural phenomenon",
        "cosmic radiation",

        # Abstract patterns
        "mathematical pattern",
        "geometric structure",
        "organized structure",
        "chaotic noise",

        # Physical objects (control)
        "a cat",
        "a house",
        "a person",
        "a mountain",
    ]
    print(f"   {len(concepts)} concepts defined")

    # Get embeddings
    print("\n5. Computing embeddings...")
    image_embeds = compute_clip_embeddings(processor, model, images=[wow_image])['image_embeds']
    text_embeds = compute_clip_embeddings(processor, model, texts=concepts)['text_embeds']
    print(f"   Image embedding shape: {image_embeds.shape}")
    print(f"   Text embeddings shape: {text_embeds.shape}")

    # Compute cosine similarities
    print("\n6. Computing similarities to concepts...")
    similarities = (image_embeds @ text_embeds.T).squeeze()

    # Sort by similarity
    sorted_idx = np.argsort(similarities)[::-1]

    print("\n   Top 10 most similar concepts:")
    for i, idx in enumerate(sorted_idx[:10]):
        print(f"   {i+1}. {concepts[idx]}: {similarities[idx]:.4f}")

    print("\n   Bottom 5 least similar concepts:")
    for idx in sorted_idx[-5:]:
        print(f"      {concepts[idx]}: {similarities[idx]:.4f}")

    # Generate random noise images for comparison
    print("\n7. Generating random noise baselines...")
    random_images = []
    for _ in range(20):
        noise = np.random.randn(82, 50)
        random_images.append(signal_to_image(noise))

    random_embeds = compute_clip_embeddings(processor, model, images=random_images)['image_embeds']

    # Similarity of random noise to concepts
    random_sims = random_embeds @ text_embeds.T  # (20, n_concepts)

    # Compare Wow! to random
    print("\n8. Comparing Wow! signal to random noise baselines...")

    # For each concept, compute z-score of Wow! similarity
    z_scores = {}
    for i, concept in enumerate(concepts):
        wow_sim = similarities[i]
        random_sim_mean = random_sims[:, i].mean()
        random_sim_std = random_sims[:, i].std()
        z = (wow_sim - random_sim_mean) / (random_sim_std + 1e-8)
        z_scores[concept] = {
            'wow_similarity': float(wow_sim),
            'random_mean': float(random_sim_mean),
            'random_std': float(random_sim_std),
            'z_score': float(z)
        }

    print("\n   Concepts where Wow! differs most from random (|z| > 1):")
    significant = [(c, z['z_score']) for c, z in z_scores.items() if abs(z['z_score']) > 1]
    significant.sort(key=lambda x: abs(x[1]), reverse=True)
    for concept, z in significant:
        direction = "MORE similar" if z > 0 else "LESS similar"
        print(f"   {concept}: z={z:.2f} ({direction})")

    # Key question: Is Wow! more similar to "information" concepts vs "noise" concepts?
    print("\n9. Information vs Noise analysis...")

    info_concepts = ["a message", "a signal", "communication", "data transmission",
                     "information pattern", "encoded message", "radio signal"]
    noise_concepts = ["random noise", "static", "interference", "chaotic noise"]

    info_sims = [similarities[concepts.index(c)] for c in info_concepts if c in concepts]
    noise_sims = [similarities[concepts.index(c)] for c in noise_concepts if c in concepts]

    print(f"   Wow! similarity to information concepts: {np.mean(info_sims):.4f}")
    print(f"   Wow! similarity to noise concepts: {np.mean(noise_sims):.4f}")
    print(f"   Difference: {np.mean(info_sims) - np.mean(noise_sims):.4f}")

    # Random baseline for this metric
    random_info_sims = [random_sims[:, concepts.index(c)].mean() for c in info_concepts if c in concepts]
    random_noise_sims = [random_sims[:, concepts.index(c)].mean() for c in noise_concepts if c in concepts]
    print(f"\n   Random similarity to information: {np.mean(random_info_sims):.4f}")
    print(f"   Random similarity to noise: {np.mean(random_noise_sims):.4f}")
    print(f"   Random difference: {np.mean(random_info_sims) - np.mean(random_noise_sims):.4f}")

    # Save results
    results = {
        "experiment": "exp40_clip_encoding",
        "timestamp": datetime.now().isoformat(),
        "signal_shape": list(signal.shape),
        "model": model_name,
        "embedding_dim": int(image_embeds.shape[1]),
        "concept_similarities": {c: float(similarities[i]) for i, c in enumerate(concepts)},
        "top_concepts": [
            {"concept": concepts[idx], "similarity": float(similarities[idx])}
            for idx in sorted_idx[:10]
        ],
        "z_scores": z_scores,
        "significant_deviations": [
            {"concept": c, "z_score": z} for c, z in significant
        ],
        "info_vs_noise": {
            "wow_info_similarity": float(np.mean(info_sims)),
            "wow_noise_similarity": float(np.mean(noise_sims)),
            "wow_difference": float(np.mean(info_sims) - np.mean(noise_sims)),
            "random_info_similarity": float(np.mean(random_info_sims)),
            "random_noise_similarity": float(np.mean(random_noise_sims)),
            "random_difference": float(np.mean(random_info_sims) - np.mean(random_noise_sims))
        }
    }

    output_path = RESULTS_DIR / "exp40_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n10. Results saved to {output_path}")

    return results


if __name__ == "__main__":
    main()
