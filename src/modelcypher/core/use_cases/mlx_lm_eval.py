# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ModelCypher is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with ModelCypher.  If not, see <https://www.gnu.org/licenses/>.

"""MLX model wrapper for lm-eval-harness.

Enables running lm-eval benchmarks directly on MLX models without a server.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lm_eval.api.instance import Instance

logger = logging.getLogger(__name__)


def run_benchmark(
    model_path: str,
    tasks: list[str],
    limit: int | None = None,
    num_fewshot: int = 0,
    output_path: str | None = None,
) -> dict[str, Any]:
    """Run lm-eval benchmarks on an MLX model.

    Args:
        model_path: Path to MLX model directory.
        tasks: List of benchmark task names.
        limit: Optional sample limit per task.
        num_fewshot: Number of few-shot examples.
        output_path: Optional path to save results.

    Returns:
        Dictionary with benchmark results.
    """
    import mlx.core as mx
    import mlx.nn as nn
    from lm_eval import evaluator
    from lm_eval.api.model import LM
    from mlx_lm import load

    model_path = Path(model_path).resolve()
    if not model_path.exists():
        raise ValueError(f"Model path does not exist: {model_path}")

    logger.info(f"Loading MLX model from {model_path}")
    model, tokenizer = load(str(model_path))

    class MLXLM(LM):
        """lm-eval wrapper for MLX language models."""

        def __init__(self, mlx_model, mlx_tokenizer, model_path: str):
            super().__init__()
            self._model = mlx_model
            self._tokenizer = mlx_tokenizer
            self._model_path = model_path

        @property
        def eot_token_id(self):
            return self._tokenizer.eos_token_id

        @property
        def max_length(self):
            return getattr(self._model.config, "max_position_embeddings", 2048)

        @property
        def max_gen_toks(self):
            return 256

        @property
        def batch_size(self):
            return 1

        @property
        def device(self):
            return "mlx"

        def tok_encode(self, string: str, left_truncate_len: int | None = None, add_special_tokens: bool = True):
            tokens = self._tokenizer.encode(string, add_special_tokens=add_special_tokens)
            if left_truncate_len:
                tokens = tokens[-left_truncate_len:]
            return tokens

        def tok_decode(self, tokens):
            return self._tokenizer.decode(tokens)

        def _loglikelihood_tokens(
            self,
            requests: list,
            disable_tqdm: bool = False,
        ) -> list[tuple[float, bool]]:
            """Compute log-likelihoods for requests."""
            results = []

            for context, continuation in requests:
                # Encode context and continuation
                ctx_tokens = self.tok_encode(context) if context else []
                cont_tokens = self.tok_encode(continuation, add_special_tokens=False)
                all_tokens = ctx_tokens + cont_tokens

                if not all_tokens:
                    results.append((0.0, False))
                    continue

                # Forward pass
                tokens_mx = mx.array([all_tokens])
                logits = self._model(tokens_mx)
                logits = logits[0]  # Remove batch dim

                # Compute log-probs for continuation tokens
                log_probs = nn.log_softmax(logits, axis=-1)
                mx.eval(log_probs)

                # Get log-probs for the continuation tokens
                ctx_len = len(ctx_tokens) if ctx_tokens else 0
                cont_log_probs = []
                is_greedy = True

                for i, tok in enumerate(cont_tokens):
                    pos = ctx_len + i - 1  # Position in logits (shifted by 1)
                    if pos < 0:
                        pos = 0
                    if pos >= log_probs.shape[0]:
                        break
                    tok_log_prob = float(log_probs[pos, tok])
                    cont_log_probs.append(tok_log_prob)

                    # Check if greedy
                    greedy_tok = int(mx.argmax(log_probs[pos]).item())
                    if greedy_tok != tok:
                        is_greedy = False

                total_log_prob = sum(cont_log_probs)
                results.append((total_log_prob, is_greedy))

            return results

        def loglikelihood(self, requests: list["Instance"]) -> list[tuple[float, bool]]:
            """Compute log-likelihoods for a batch of requests."""
            # Extract context and continuation from requests
            processed = []
            for req in requests:
                context = req.args[0]
                continuation = req.args[1]
                processed.append((context, continuation))
            return self._loglikelihood_tokens(processed)

        def loglikelihood_rolling(self, requests: list["Instance"]) -> list[tuple[float, bool]]:
            """Compute rolling log-likelihoods."""
            processed = []
            for req in requests:
                text = req.args[0]
                processed.append(("", text))
            return self._loglikelihood_tokens(processed)

        def generate_until(self, requests: list["Instance"]) -> list[str]:
            """Generate text until stop sequences."""
            from mlx_lm import generate

            results = []
            for req in requests:
                context = req.args[0]
                gen_kwargs = req.args[1] if len(req.args) > 1 else {}

                until = gen_kwargs.get("until", [])
                max_gen_toks = gen_kwargs.get("max_gen_toks", self.max_gen_toks)

                # Generate using mlx_lm
                response = generate(
                    self._model,
                    self._tokenizer,
                    prompt=context,
                    max_tokens=max_gen_toks,
                    verbose=False,
                )

                # Strip context from response if present
                if response.startswith(context):
                    response = response[len(context):]

                # Truncate at stop sequences
                for stop in until:
                    if stop in response:
                        response = response[:response.index(stop)]

                results.append(response)

            return results

    # Register and create model
    mlx_lm_instance = MLXLM(model, tokenizer, str(model_path))

    # Run evaluation
    logger.info(f"Running benchmarks: {tasks}")
    results = evaluator.simple_evaluate(
        model=mlx_lm_instance,
        tasks=tasks,
        num_fewshot=num_fewshot,
        limit=limit,
        batch_size=1,
        log_samples=False,
    )

    if output_path:
        import json
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to {output_path}")

    return results
