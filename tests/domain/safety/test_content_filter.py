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


import unittest
from modelcypher.core.domain.safety.regex_content_filter import (
    RegexContentFilter,
    SafetyStatus,
    SafetyCategory
)

class TestRegexContentFilter(unittest.TestCase):

    def setUp(self):
        self.filter = RegexContentFilter.default()

    def test_safe_content(self):
        text = "This is a perfectly safe sentence about kittens."
        result = self.filter.check(text)
        self.assertIsNone(result, "Safe content should return None")

    def test_dangerous_content_shell(self):
        text = "To delete everything, run: sudo rm -rf / --no-preserve-root"
        result = self.filter.check(text)
        self.assertIsNotNone(result)
        self.assertEqual(result.status, SafetyStatus.REJECTED)
        self.assertEqual(result.category, SafetyCategory.DANGEROUS_CODE)
        self.assertIn("Destructive shell command", result.reason)

    def test_pii_detection_email(self):
        text = "Contact me at user@random-domain.xyz for more info."
        # example.com is whitelisted, so we use random-domain.xyz
        result = self.filter.check(text)
        self.assertIsNotNone(result)
        # Check if it was caught as PII
        self.assertTrue(result.category == SafetyCategory.PII or "email" in result.reason.lower())

    def test_whitelist_email(self):
        # Whitelisted domain
        text = "Contact support@example.com for help."
        # example.com IS in default whitelist, so this should pass (return None)
        result = self.filter.check(text)
        self.assertIsNone(result, "Whitelisted email should be ignored")

if __name__ == "__main__":
    unittest.main()
