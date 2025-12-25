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
