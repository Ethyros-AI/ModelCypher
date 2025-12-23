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
from modelcypher.core.domain.dataset.document_chunker import (
    DocumentChunker,
    TextChunk,
    TextTokenizer,
)
from modelcypher.core.domain.dataset.dataset_slicer import (
    DatasetSliceRecipe,
    DatasetSlicer,
    DatasetSlicingError,
    SliceMode,
)
from modelcypher.core.domain.dataset.streaming_shuffler import (
    ShufflerEntry,
    StreamingShuffler,
    shuffle_streaming,
)
from modelcypher.core.domain.dataset.jsonl_parser import (
    JSONLParser,
    ParsedJSONLRow,
)
from modelcypher.core.domain.dataset.dataset_export_formatter import (
    convert_format,
    DatasetExportFormat,
    DatasetExportFormatter,
    DatasetExportFormatterError,
)
from modelcypher.core.domain.dataset.token_counter_service import (
    get_token_counter_service,
    LRUCache,
    reset_token_counter_service,
    set_token_counter_service,
    TokenCounterConfig,
    TokenCounterService,
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
