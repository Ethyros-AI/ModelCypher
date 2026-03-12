"""Canonical identity for the shipped geometric LoRA training path."""

from modelcypher.core.domain.training.mass_step_size import (
    OPTIMIZER_MODE_ADAMW_GEOMETRIC,
)

GEOMETRIC_LORA_METHOD = "geometric_lora"
GEOMETRIC_LORA_INIT_METHOD = "pissa"
GEOMETRIC_LORA_INIT_METHOD_CAYLEY = "cayley"
GEOMETRIC_LORA_OPTIMIZER = "adamw_cosine"
GEOMETRIC_LORA_OPTIMIZER_FISHER_MASS = "fisher_mass"
GEOMETRIC_LORA_CONTROLLER = "mass"
GEOMETRIC_LORA_STOPPING = "geometric_certificate"


def resolve_geometric_lora_optimizer_name(optimizer_research_mode: str) -> str:
    """Map optimizer mode to the canonical shipped optimizer identity.

    The canonical identity names the control surface exposed to users and
    artifacts. Any non-default optimizer research mode engages MASS step-size
    control on the geometry-derived LoRA path and therefore reports as
    ``fisher_mass``.
    """
    if optimizer_research_mode == OPTIMIZER_MODE_ADAMW_GEOMETRIC:
        return GEOMETRIC_LORA_OPTIMIZER
    return GEOMETRIC_LORA_OPTIMIZER_FISHER_MASS
