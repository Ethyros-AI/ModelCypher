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

"""Dataset domain modules.

This package contains modules for dataset processing, chat templating,
document chunking, and format conversion.
"""

from modelcypher.core.domain.dataset.chat_message import (
    ChatMessage,
    parse_messages,
)
from modelcypher.core.domain.dataset.chat_template_library import (
    ChatTemplate,
)
from modelcypher.core.domain.dataset.dataset_export_formatter import (
    DatasetExportFormat,
    DatasetExportFormatter,
    DatasetExportFormatterError,
    convert_format,
)
from modelcypher.core.domain.dataset.dataset_slicer import (
    DatasetSlicer,
    DatasetSliceRecipe,
    DatasetSlicingError,
    SliceMode,
)
from modelcypher.core.domain.dataset.document_chunker import (
    DocumentChunker,
    TextChunk,
    TextTokenizer,
)
from modelcypher.core.domain.dataset.jsonl_parser import (
    JSONLParser,
    ParsedJSONLRow,
)
from modelcypher.core.domain.dataset.streaming_shuffler import (
    ShufflerEntry,
    StreamingShuffler,
    shuffle_streaming,
)
from modelcypher.core.domain.dataset.token_counter_service import (
    LRUCache,
    TokenCounterConfig,
    TokenCounterService,
    get_token_counter_service,
    reset_token_counter_service,
    set_token_counter_service,
)

__all__ = [
    # Chat message
    "ChatMessage",
    "parse_messages",
    # Chat templates
    "ChatTemplate",
    # Document chunker
    "DocumentChunker",
    "TextChunk",
    "TextTokenizer",
    # Dataset slicer
    "DatasetSliceRecipe",
    "DatasetSlicer",
    "DatasetSlicingError",
    "SliceMode",
    # Streaming shuffler
    "ShufflerEntry",
    "StreamingShuffler",
    "shuffle_streaming",
    # JSONL parser
    "JSONLParser",
    "ParsedJSONLRow",
    # Export formatter
    "convert_format",
    "DatasetExportFormat",
    "DatasetExportFormatter",
    "DatasetExportFormatterError",
    # Token counter
    "get_token_counter_service",
    "LRUCache",
    "reset_token_counter_service",
    "set_token_counter_service",
    "TokenCounterConfig",
    "TokenCounterService",
]
