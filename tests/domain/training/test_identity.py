from modelcypher.core.domain.training.identity import (
    GEOMETRIC_LORA_OPTIMIZER,
    GEOMETRIC_LORA_OPTIMIZER_FISHER_MASS,
    resolve_geometric_lora_optimizer_name,
)
from modelcypher.core.domain.training.mass_step_size import (
    OPTIMIZER_MODE_ADAMW_GEOMETRIC,
    OPTIMIZER_MODE_ADAMW_MATCHED_TRACE,
    OPTIMIZER_MODE_CAYLEY_STIEFEL_MASS,
)


def test_resolve_geometric_lora_optimizer_name_default_path():
    assert (
        resolve_geometric_lora_optimizer_name(OPTIMIZER_MODE_ADAMW_GEOMETRIC)
        == GEOMETRIC_LORA_OPTIMIZER
    )


def test_resolve_geometric_lora_optimizer_name_mass_paths():
    assert (
        resolve_geometric_lora_optimizer_name(OPTIMIZER_MODE_ADAMW_MATCHED_TRACE)
        == GEOMETRIC_LORA_OPTIMIZER_FISHER_MASS
    )
    assert (
        resolve_geometric_lora_optimizer_name(OPTIMIZER_MODE_CAYLEY_STIEFEL_MASS)
        == GEOMETRIC_LORA_OPTIMIZER_FISHER_MASS
    )
