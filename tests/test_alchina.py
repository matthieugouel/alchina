"""Alchina tests."""

from alchina import __version__


# --- Version ---


def test_version():
    """Test package version."""
    assert __version__ == "0.1.1"
