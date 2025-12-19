from __future__ import annotations

from datetime import datetime, timedelta, timezone
from email.utils import format_datetime

from modelcypher.adapters.hf_model_search import HfModelSearchAdapter
from modelcypher.core.domain.model_search import MemoryFitStatus, ModelSearchError, ModelSearchResult


def test_estimated_size_gb_from_tags() -> None:
    result = ModelSearchResult(
        id="demo/7b",
        downloads=0,
        likes=0,
        tags=["7B"],
        author=None,
        pipeline_tag=None,
        last_modified=None,
        is_private=False,
        is_gated=False,
    )
    assert result.estimated_size_gb == 14.0

    result = ModelSearchResult(
        id="demo/0.5b",
        downloads=0,
        likes=0,
        tags=["0.5B"],
        author=None,
        pipeline_tag=None,
        last_modified=None,
        is_private=False,
        is_gated=False,
    )
    assert result.estimated_size_gb == 1.0

    result = ModelSearchResult(
        id="demo/unknown",
        downloads=0,
        likes=0,
        tags=["mlx"],
        author=None,
        pipeline_tag=None,
        last_modified=None,
        is_private=False,
        is_gated=False,
    )
    assert result.estimated_size_gb is None


def test_is_recommended_requires_mlx_and_quant() -> None:
    result = ModelSearchResult(
        id="demo/recommended",
        downloads=0,
        likes=0,
        tags=["mlx", "4bit"],
        author=None,
        pipeline_tag=None,
        last_modified=None,
        is_private=False,
        is_gated=False,
        memory_fit_status=MemoryFitStatus.fits,
    )
    assert result.is_recommended is True

    result = ModelSearchResult(
        id="demo/no-mlx",
        downloads=0,
        likes=0,
        tags=["4bit"],
        author=None,
        pipeline_tag=None,
        last_modified=None,
        is_private=False,
        is_gated=False,
    )
    assert result.is_recommended is False

    result = ModelSearchResult(
        id="demo/no-quant",
        downloads=0,
        likes=0,
        tags=["mlx"],
        author=None,
        pipeline_tag=None,
        last_modified=None,
        is_private=False,
        is_gated=False,
    )
    assert result.is_recommended is False


def test_extract_cursor_from_link_header() -> None:
    link = '<https://huggingface.co/api/models?cursor=abc123&limit=20>; rel="next", <https://huggingface.co/api/models?cursor=prev&limit=20>; rel="prev"'
    assert HfModelSearchAdapter._extract_cursor(link) == "abc123"


def test_parse_retry_after_seconds() -> None:
    assert HfModelSearchAdapter._parse_retry_after("120") == 120.0


def test_parse_retry_after_date() -> None:
    future = datetime.now(timezone.utc) + timedelta(seconds=60)
    header = format_datetime(future)
    parsed = HfModelSearchAdapter._parse_retry_after(header)
    assert parsed is not None
    assert 0.0 <= parsed <= 120.0


def test_model_search_error_messages() -> None:
    assert str(ModelSearchError.network_error("timeout")) == "Network error: timeout"
    assert str(ModelSearchError.invalid_response("bad json")) == "Invalid response: bad json"
    assert str(ModelSearchError.rate_limited(30)) == "Rate limited. Retry after 30 seconds."
    assert str(ModelSearchError.rate_limited(None)) == "Rate limited. Please try again later."
    assert (
        str(ModelSearchError.authentication_required())
        == "Authentication required. Set HF_TOKEN environment variable."
    )
    assert str(ModelSearchError.search_failed("HTTP 500")) == "Search failed: HTTP 500"
