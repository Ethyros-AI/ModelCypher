# ModelCypher Experiments

Validated experimental results supporting the research papers.

## Structure

Each experiment lives in its own directory with:
- `README.md` or `HYPOTHESIS.md` - Experiment design and methodology
- `analysis/` - Reproducible analysis scripts
- `results/` - Output data (JSON) and summary reports

## Experiments

| Experiment | Paper | Key Finding |
|------------|-------|-------------|
| [Operational Semantics Hypothesis](operational-semantics-hypothesis/) | Paper 0, Claim 4 | Pythagorean theorem encoded as geometric constraint; 88.5% cross-model invariance |

## Reproducibility

Experiments were run on Apple Silicon (M-series) using MLX backend. Models stored on external volume at `/Volumes/CodeCypher/models/`.

To reproduce:
```bash
cd experiments/<experiment-name>/analysis/
poetry run python <script>.py --model /path/to/model --output ../results/
```

## Adding New Experiments

1. Run experiments on external volume (`/Volumes/CodeCypher/experiments/`)
2. Once validated, copy to repository under `experiments/<name>/`
3. Reference from the appropriate paper
