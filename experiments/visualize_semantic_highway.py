#!/usr/bin/env python3
"""Visualize the 3D Semantic Highway.

The bottleneck compresses meaning to ~3 dimensions. Let's see it.

This script:
1. Collects activations for diverse concepts at the bottleneck
2. Projects to top 3 PCs of the Gram matrix
3. Plots in 3D with semantic category coloring
4. Generates interactive HTML visualization
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import json
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# Comprehensive semantic categories for visualization
SEMANTIC_CATEGORIES = {
    # NSM Primes - the coordinate system
    "PRIMES": {
        "color": "red",
        "symbol": "diamond",  # Valid 3D: circle, circle-open, cross, diamond, diamond-open, square, square-open, x
        "probes": [
            ("I", "I am here."),
            ("YOU", "You are there."),
            ("SOMEONE", "Someone is coming."),
            ("SOMETHING", "Something happened."),
            ("GOOD", "This is good."),
            ("BAD", "This is bad."),
            ("BIG", "It is big."),
            ("SMALL", "It is small."),
            ("THINK", "I think this."),
            ("KNOW", "I know this."),
            ("WANT", "I want this."),
            ("FEEL", "I feel this."),
            ("SEE", "I see this."),
            ("HEAR", "I hear this."),
            ("SAY", "I say this."),
            ("DO", "I do this."),
            ("HAPPEN", "It happened."),
            ("MOVE", "It moves."),
            ("LIVE", "People live."),
            ("DIE", "People die."),
            ("NOW", "It is now."),
            ("BEFORE", "Before this."),
            ("AFTER", "After this."),
            ("HERE", "It is here."),
            ("ABOVE", "It is above."),
            ("BELOW", "It is below."),
            ("NOT", "It is not."),
            ("MAYBE", "Maybe it is."),
            ("CAN", "I can do this."),
            ("BECAUSE", "Because of this."),
        ],
    },

    # Abstract concepts
    "ABSTRACT": {
        "color": "purple",
        "symbol": "circle",
        "probes": [
            ("love", "Love is a powerful emotion."),
            ("justice", "Justice requires fairness."),
            ("freedom", "Freedom means independence."),
            ("truth", "Truth is hard to find."),
            ("beauty", "Beauty is subjective."),
            ("wisdom", "Wisdom comes with experience."),
            ("courage", "Courage overcomes fear."),
            ("hope", "Hope sustains us."),
            ("faith", "Faith guides belief."),
            ("honor", "Honor demands integrity."),
            ("democracy", "Democracy requires participation."),
            ("philosophy", "Philosophy seeks wisdom."),
            ("ethics", "Ethics guides behavior."),
            ("morality", "Morality defines right and wrong."),
            ("consciousness", "Consciousness is awareness."),
        ],
    },

    # Concrete objects
    "CONCRETE": {
        "color": "green",
        "symbol": "square",
        "probes": [
            ("apple", "The red apple sits on the table."),
            ("chair", "A wooden chair has four legs."),
            ("table", "The table is made of oak."),
            ("book", "The book has many pages."),
            ("car", "The car drives fast."),
            ("house", "The house has a roof."),
            ("tree", "The tree grows tall."),
            ("rock", "The rock is heavy."),
            ("water", "Water flows downhill."),
            ("fire", "Fire burns hot."),
            ("mountain", "The mountain is tall."),
            ("river", "The river flows to the sea."),
            ("bird", "The bird flies south."),
            ("dog", "The dog runs fast."),
            ("cat", "The cat sleeps often."),
        ],
    },

    # Actions/Verbs
    "ACTIONS": {
        "color": "orange",
        "symbol": "cross",
        "probes": [
            ("running", "She is running fast."),
            ("jumping", "He is jumping high."),
            ("swimming", "They are swimming in the pool."),
            ("writing", "I am writing a letter."),
            ("reading", "She is reading a book."),
            ("eating", "We are eating dinner."),
            ("sleeping", "The baby is sleeping."),
            ("dancing", "They are dancing together."),
            ("singing", "She is singing beautifully."),
            ("building", "Workers are building a house."),
            ("destroying", "The storm is destroying everything."),
            ("creating", "Artists are creating art."),
            ("learning", "Students are learning math."),
            ("teaching", "Teachers are teaching history."),
            ("healing", "Doctors are healing patients."),
        ],
    },

    # Emotions
    "EMOTIONS": {
        "color": "pink",
        "symbol": "diamond-open",
        "probes": [
            ("happiness", "Happiness fills the room."),
            ("sadness", "Sadness overwhelms her."),
            ("anger", "Anger consumes him."),
            ("fear", "Fear grips the crowd."),
            ("surprise", "Surprise lit up her face."),
            ("disgust", "Disgust showed clearly."),
            ("joy", "Joy spreads easily."),
            ("grief", "Grief takes time."),
            ("anxiety", "Anxiety builds slowly."),
            ("excitement", "Excitement grows."),
            ("jealousy", "Jealousy poisons relationships."),
            ("pride", "Pride swelled in his chest."),
            ("shame", "Shame colored her cheeks."),
            ("guilt", "Guilt weighed heavily."),
            ("contentment", "Contentment settled in."),
        ],
    },

    # Scientific/Technical
    "SCIENTIFIC": {
        "color": "cyan",
        "symbol": "square-open",
        "probes": [
            ("photosynthesis", "Photosynthesis converts light to energy."),
            ("gravity", "Gravity pulls objects down."),
            ("evolution", "Evolution shapes species."),
            ("electricity", "Electricity powers devices."),
            ("magnetism", "Magnetism attracts metals."),
            ("chemistry", "Chemistry studies matter."),
            ("biology", "Biology studies life."),
            ("physics", "Physics explains motion."),
            ("mathematics", "Mathematics is precise."),
            ("algorithm", "The algorithm processes data."),
            ("quantum", "Quantum mechanics is strange."),
            ("relativity", "Relativity bends spacetime."),
            ("entropy", "Entropy always increases."),
            ("thermodynamics", "Thermodynamics governs heat."),
            ("genetics", "Genetics determines traits."),
        ],
    },

    # Social/Relational
    "SOCIAL": {
        "color": "yellow",
        "symbol": "diamond-open",
        "probes": [
            ("family", "Family gathers for dinner."),
            ("friend", "A friend listens carefully."),
            ("enemy", "The enemy approaches."),
            ("stranger", "A stranger walked by."),
            ("community", "The community came together."),
            ("society", "Society shapes behavior."),
            ("culture", "Culture varies widely."),
            ("tradition", "Tradition guides practice."),
            ("marriage", "Marriage joins two people."),
            ("childhood", "Childhood shapes personality."),
            ("leadership", "Leadership requires vision."),
            ("cooperation", "Cooperation achieves more."),
            ("conflict", "Conflict arises often."),
            ("negotiation", "Negotiation seeks compromise."),
            ("celebration", "Celebration brings joy."),
        ],
    },

    # Temporal
    "TEMPORAL": {
        "color": "brown",
        "symbol": "x",
        "probes": [
            ("yesterday", "Yesterday was rainy."),
            ("tomorrow", "Tomorrow will be sunny."),
            ("ancient", "Ancient civilizations built pyramids."),
            ("modern", "Modern technology advances."),
            ("future", "The future is uncertain."),
            ("past", "The past cannot change."),
            ("present", "The present moment matters."),
            ("eternal", "Some things seem eternal."),
            ("temporary", "This is only temporary."),
            ("sudden", "A sudden change occurred."),
            ("gradual", "Gradual change is lasting."),
            ("beginning", "Every story has a beginning."),
            ("ending", "Every story has an ending."),
            ("duration", "The duration was long."),
            ("moment", "A moment of silence."),
        ],
    },

    # Spatial
    "SPATIAL": {
        "color": "blue",
        "symbol": "cross",
        "probes": [
            ("inside", "It is inside the box."),
            ("outside", "Go outside to play."),
            ("near", "Stay near the door."),
            ("far", "The star is far away."),
            ("left", "Turn left at the corner."),
            ("right", "The door is on the right."),
            ("up", "Look up at the sky."),
            ("down", "Climb down the ladder."),
            ("center", "Stand in the center."),
            ("edge", "Walk along the edge."),
            ("surface", "The surface is smooth."),
            ("depth", "The depth is unknown."),
            ("height", "The height is impressive."),
            ("width", "The width is narrow."),
            ("distance", "The distance is great."),
        ],
    },
}


def load_model(path: str):
    """Load a model."""
    from mlx_lm import load
    import mlx.core as mx

    model, tokenizer = load(path)
    mx.eval(model.parameters())

    inner = model.model if hasattr(model, "model") else model
    n_layers = len(inner.layers)

    return model, tokenizer, n_layers


def get_layer_activation(model, tokenizer, text: str, layer_idx: int):
    """Get mean-pooled activation at specific layer."""
    import mlx.core as mx

    inner = model.model if hasattr(model, "model") else model

    tokens = tokenizer.encode(text)
    input_ids = mx.array([tokens])
    mx.eval(input_ids)

    if hasattr(inner, "embed_tokens"):
        h = inner.embed_tokens(input_ids)
    elif hasattr(inner, "wte"):
        h = inner.wte(input_ids)
    else:
        return None

    for idx, layer in enumerate(inner.layers):
        if idx > layer_idx:
            break
        result = layer(h)
        if isinstance(result, tuple):
            h = result[0]
        else:
            h = result

    pooled = mx.mean(h, axis=(0, 1))
    mx.eval(pooled)
    return pooled


def collect_activations(model, tokenizer, layer_idx: int):
    """Collect activations for all semantic categories."""
    import mlx.core as mx

    all_data = []

    for category, info in SEMANTIC_CATEGORIES.items():
        logger.info(f"  Collecting {category}...")
        for label, probe in info["probes"]:
            act = get_layer_activation(model, tokenizer, probe, layer_idx)
            if act is not None:
                act = act.astype(mx.float32)
                mx.eval(act)
                all_data.append({
                    "category": category,
                    "label": label,
                    "probe": probe,
                    "color": info["color"],
                    "symbol": info["symbol"],
                    "activation": np.array(act),
                })

    return all_data


def create_3d_visualization(data: list, model_name: str, output_path: Path):
    """Create interactive 3D visualization."""

    # Stack activations
    activations = np.stack([d["activation"] for d in data])
    n_samples = len(data)

    logger.info(f"Computing PCA on {n_samples} samples...")

    # Compute Gram matrix
    G = activations @ activations.T

    # SVD for PCA in Gram space
    U, S, Vt = np.linalg.svd(G, full_matrices=False)

    # Project onto top 3 PCs
    # Scale by sqrt(eigenvalue) for proper PCA coordinates
    coords = U[:, :3] * np.sqrt(S[:3])

    # Compute effective rank
    threshold = S[0] * 3.45e-4
    effective_rank = int(np.sum(S > threshold))

    # Variance explained
    var_explained = S[:3] / S.sum() * 100

    logger.info(f"Gram effective rank: {effective_rank}")
    logger.info(f"Variance explained: PC1={var_explained[0]:.1f}%, PC2={var_explained[1]:.1f}%, PC3={var_explained[2]:.1f}%")

    # Create figure
    fig = go.Figure()

    # Add traces for each category
    for category, info in SEMANTIC_CATEGORIES.items():
        indices = [i for i, d in enumerate(data) if d["category"] == category]
        if not indices:
            continue

        cat_coords = coords[indices]
        labels = [data[i]["label"] for i in indices]
        probes = [data[i]["probe"] for i in indices]

        hover_text = [f"{l}<br>{p}" for l, p in zip(labels, probes)]

        fig.add_trace(go.Scatter3d(
            x=cat_coords[:, 0],
            y=cat_coords[:, 1],
            z=cat_coords[:, 2],
            mode='markers+text',
            marker=dict(
                size=8 if category == "PRIMES" else 6,
                color=info["color"],
                symbol=info["symbol"],
                opacity=0.9 if category == "PRIMES" else 0.7,
                line=dict(width=1, color='black') if category == "PRIMES" else dict(width=0),
            ),
            text=labels,
            textposition="top center",
            textfont=dict(size=8),
            hovertext=hover_text,
            hoverinfo='text',
            name=category,
        ))

    # Add origin marker
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers',
        marker=dict(size=10, color='black', symbol='x'),
        name='Origin',
        hovertext=['Origin (0,0,0)'],
        hoverinfo='text',
    ))

    # Layout
    fig.update_layout(
        title=dict(
            text=f"The Shape of Meaning: {model_name}<br><sub>3D Semantic Highway at Bottleneck (Gram rank={effective_rank}, Var: {var_explained[0]:.0f}%/{var_explained[1]:.0f}%/{var_explained[2]:.0f}%)</sub>",
            x=0.5,
        ),
        scene=dict(
            xaxis_title=f"PC1 ({var_explained[0]:.1f}%)",
            yaxis_title=f"PC2 ({var_explained[1]:.1f}%)",
            zaxis_title=f"PC3 ({var_explained[2]:.1f}%)",
            aspectmode='cube',
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)",
        ),
        margin=dict(l=0, r=0, t=80, b=0),
        width=1200,
        height=900,
    )

    # Save as HTML
    fig.write_html(str(output_path))
    logger.info(f"Saved interactive visualization to {output_path}")

    # Also create a 2D projection for each PC pair
    fig_2d = make_subplots(
        rows=1, cols=3,
        subplot_titles=(
            f'PC1 vs PC2 ({var_explained[0]:.0f}% vs {var_explained[1]:.0f}%)',
            f'PC1 vs PC3 ({var_explained[0]:.0f}% vs {var_explained[2]:.0f}%)',
            f'PC2 vs PC3 ({var_explained[1]:.0f}% vs {var_explained[2]:.0f}%)'
        ),
        horizontal_spacing=0.08,
    )

    for category, info in SEMANTIC_CATEGORIES.items():
        indices = [i for i, d in enumerate(data) if d["category"] == category]
        if not indices:
            continue

        cat_coords = coords[indices]
        labels = [data[i]["label"] for i in indices]

        # PC1 vs PC2
        fig_2d.add_trace(go.Scatter(
            x=cat_coords[:, 0], y=cat_coords[:, 1],
            mode='markers+text',
            marker=dict(size=10 if category == "PRIMES" else 7, color=info["color"]),
            text=labels, textposition="top center", textfont=dict(size=7),
            name=category, showlegend=True,
        ), row=1, col=1)

        # PC1 vs PC3
        fig_2d.add_trace(go.Scatter(
            x=cat_coords[:, 0], y=cat_coords[:, 2],
            mode='markers+text',
            marker=dict(size=10 if category == "PRIMES" else 7, color=info["color"]),
            text=labels, textposition="top center", textfont=dict(size=7),
            name=category, showlegend=False,
        ), row=1, col=2)

        # PC2 vs PC3
        fig_2d.add_trace(go.Scatter(
            x=cat_coords[:, 1], y=cat_coords[:, 2],
            mode='markers+text',
            marker=dict(size=10 if category == "PRIMES" else 7, color=info["color"]),
            text=labels, textposition="top center", textfont=dict(size=7),
            name=category, showlegend=False,
        ), row=1, col=3)

    fig_2d.update_layout(
        title=f"2D Projections: {model_name} Semantic Highway",
        height=600,
        width=1800,
    )

    fig_2d.update_xaxes(title_text="PC1", row=1, col=1)
    fig_2d.update_yaxes(title_text="PC2", row=1, col=1)
    fig_2d.update_xaxes(title_text="PC1", row=1, col=2)
    fig_2d.update_yaxes(title_text="PC3", row=1, col=2)
    fig_2d.update_xaxes(title_text="PC2", row=1, col=3)
    fig_2d.update_yaxes(title_text="PC3", row=1, col=3)

    output_2d = output_path.parent / f"{output_path.stem}_2d.html"
    fig_2d.write_html(str(output_2d))
    logger.info(f"Saved 2D projections to {output_2d}")

    return {
        "effective_rank": effective_rank,
        "variance_explained": var_explained.tolist(),
        "n_samples": n_samples,
    }


def main():
    models = {
        "SmolLM-135M": str(Path.home() / "ModelCypher/tests/fixtures/.models/HuggingFaceTB--SmolLM-135M"),
        "LFM2-350M": "/path/to/models/mlx-community/LFM2-350M-MLX-bf16",
    }

    output_dir = Path(__file__).parent / "visualizations"
    output_dir.mkdir(exist_ok=True)

    results = {}

    for name, path in models.items():
        logger.info("=" * 60)
        logger.info(f"VISUALIZING {name}")
        logger.info("=" * 60)

        try:
            model, tokenizer, n_layers = load_model(path)
            bottleneck_layer = n_layers // 2

            logger.info(f"Collecting activations at bottleneck layer {bottleneck_layer}/{n_layers}")
            data = collect_activations(model, tokenizer, bottleneck_layer)

            output_path = output_dir / f"semantic_highway_{name.replace('.', '_').replace('-', '_')}.html"
            stats = create_3d_visualization(data, name, output_path)
            results[name] = stats

        except Exception as e:
            logger.error(f"Failed: {e}")
            import traceback
            traceback.print_exc()

    # Save stats
    stats_path = output_dir / "visualization_stats.json"
    with open(stats_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nAll visualizations saved to {output_dir}/")
    logger.info("Open the HTML files in a browser to explore the 3D semantic space!")


if __name__ == "__main__":
    main()
