from unittest import TestCase

from physical_ai_agent import __version__


class CliTest(TestCase):
    def test_version(self) -> None:
        self.assertEqual(__version__, "0.1.0")
