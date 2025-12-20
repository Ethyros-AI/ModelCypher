"""Property tests for HelpService completions."""
from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from modelcypher.core.use_cases.help_service import HelpService


# **Feature: cli-mcp-parity, Property 10: Completions returns valid shell script**
# **Validates: Requirements 7.2**
@given(
    shell=st.sampled_from(["bash", "zsh", "fish"]),
)
@settings(max_examples=100, deadline=None)
def test_completions_returns_valid_shell_script(shell: str):
    """Property 10: For any shell type in {"bash", "zsh", "fish"},
    completions(shell) returns a non-empty string containing valid shell syntax.
    
    Validates:
    1. The returned script is a non-empty string
    2. The script contains shell-specific syntax markers
    3. The script references the 'tc' command (ModelCypher CLI)
    """
    service = HelpService()
    script = service.completions(shell)

    # Property: script is a non-empty string
    assert isinstance(script, str)
    assert len(script) > 0

    # Property: script contains shell-specific syntax markers
    if shell == "bash":
        # Bash completions should have function definition and complete command
        assert "_tc" in script or "tc" in script
        assert "COMPREPLY" in script or "complete" in script
    elif shell == "zsh":
        # Zsh completions should have compdef or _describe
        assert "#compdef" in script or "_describe" in script or "_tc" in script
    elif shell == "fish":
        # Fish completions should have complete -c command
        assert "complete -c tc" in script or "complete -c" in script

    # Property: script references the tc command
    assert "tc" in script


@given(
    shell=st.sampled_from(["bash", "zsh", "fish"]),
)
@settings(max_examples=100, deadline=None)
def test_completions_case_insensitive(shell: str):
    """Test that completions accepts shell names case-insensitively."""
    service = HelpService()
    
    # Test various case combinations
    variants = [shell.lower(), shell.upper(), shell.capitalize()]
    
    for variant in variants:
        script = service.completions(variant)
        assert isinstance(script, str)
        assert len(script) > 0


@given(
    shell=st.sampled_from(["bash", "zsh", "fish"]),
)
@settings(max_examples=100, deadline=None)
def test_completions_contains_subcommands(shell: str):
    """Test that completions include common subcommands."""
    service = HelpService()
    script = service.completions(shell)

    # Property: script should reference common subcommands
    common_subcommands = ["train", "model", "geometry"]
    
    # At least some common subcommands should be present
    found_subcommands = sum(1 for cmd in common_subcommands if cmd in script)
    assert found_subcommands >= 1, f"Expected at least one common subcommand in {shell} completions"


@given(
    invalid_shell=st.text(min_size=1, max_size=20).filter(
        lambda s: s.lower() not in {"bash", "zsh", "fish"}
    ),
)
@settings(max_examples=50, deadline=None)
def test_completions_rejects_invalid_shell(invalid_shell: str):
    """Test that completions raises ValueError for unsupported shells."""
    service = HelpService()
    
    with pytest.raises(ValueError) as exc_info:
        service.completions(invalid_shell)
    
    # Property: error message mentions the invalid shell
    assert invalid_shell in str(exc_info.value) or "Unsupported" in str(exc_info.value)
