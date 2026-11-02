"""Nonempty real-SDK offline contracts, no unexpected skips or live services."""

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tests import network_guard  # noqa: E402,F401


def ids(suite):
    for test in suite:
        if isinstance(test, unittest.TestSuite):
            yield from ids(test)
        else:
            yield test.id()


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.discover(
        str(ROOT / "tests"), top_level_dir=str(ROOT)
    )
    names = set(ids(suite))
    for group in (
        "test_sniff",
        "test_sheet",
        "test_frames",
        "test_pipeline",
        "test_plugin_contract",
        "test_hardening",
    ):
        if not any(name.startswith("tests." + group + ".") for name in names):
            raise SystemExit("Missing contract group: " + group)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if (
        not names
        or result.testsRun != len(names)
        or not result.wasSuccessful()
        or result.skipped
    ):
        raise SystemExit("Missing, failed or skipped contracts")
