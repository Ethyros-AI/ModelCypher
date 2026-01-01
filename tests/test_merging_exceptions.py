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

"""Tests for merging exceptions module."""

from __future__ import annotations

import pytest

from modelcypher.core.domain.merging.exceptions import MergeError


class TestMergeError:
    """Tests for MergeError exception."""

    def test_inherits_from_exception(self):
        """Test MergeError inherits from Exception."""
        assert issubclass(MergeError, Exception)

    def test_can_be_raised(self):
        """Test MergeError can be raised."""
        with pytest.raises(MergeError):
            raise MergeError("Test error")

    def test_message_accessible(self):
        """Test error message is accessible."""
        try:
            raise MergeError("Test message")
        except MergeError as e:
            assert str(e) == "Test message"

    def test_can_be_caught_as_exception(self):
        """Test MergeError can be caught as Exception."""
        with pytest.raises(Exception):
            raise MergeError("Caught as Exception")

    def test_empty_message(self):
        """Test MergeError with empty message."""
        error = MergeError()
        assert str(error) == ""

    def test_args_preserved(self):
        """Test exception args are preserved."""
        error = MergeError("message", "extra", 123)
        assert error.args == ("message", "extra", 123)
