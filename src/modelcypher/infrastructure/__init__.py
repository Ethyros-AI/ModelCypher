"""Infrastructure layer for dependency injection.

This module provides the composition root for wiring all adapters to ports.
Following hexagonal architecture, this is the ONLY place where concrete
adapters are instantiated for production use.

USAGE:
======
    from modelcypher.infrastructure import PortRegistry, ServiceFactory

    # Production wiring
    registry = PortRegistry.create_production()
    factory = ServiceFactory(registry)
    service = factory.training_service()
"""
from modelcypher.infrastructure.container import PortRegistry
from modelcypher.infrastructure.service_factory import ServiceFactory

__all__ = ["PortRegistry", "ServiceFactory"]
