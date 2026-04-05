"""
Root conftest — skip logic for hardware-dependent test levels.

Level 0 (default): mocked, always runs on any machine
Level 2 (workstation1): DWAIL_WS1_IP required; model defaults to facebook/opt-125m
Level 3 (distributed): DWAIL_WS1_IP + DWAIL_WS2_IP required; model defaults to facebook/opt-125m

For levels 2+, set DWAIL_TEST_MODEL to use a larger pre-downloaded model instead.
"""

import os
import pytest

SMALL_MODEL = "facebook/opt-125m"


def pytest_configure(config):
    config.addinivalue_line("markers", "workstation1: requires DWAIL_WS1_IP env var")
    config.addinivalue_line("markers", "distributed: requires DWAIL_WS1_IP and DWAIL_WS2_IP env vars")


def pytest_collection_modifyitems(config, items):
    ws1 = os.environ.get("DWAIL_WS1_IP")
    ws2 = os.environ.get("DWAIL_WS2_IP")

    skip_ws1 = pytest.mark.skip(reason="DWAIL_WS1_IP not set")
    skip_ws2 = pytest.mark.skip(reason="DWAIL_WS1_IP and/or DWAIL_WS2_IP not set")

    for item in items:
        if "workstation1" in item.keywords and not ws1:
            item.add_marker(skip_ws1)
        if "distributed" in item.keywords and not (ws1 and ws2):
            item.add_marker(skip_ws2)


# --- Shared fixtures ---

@pytest.fixture(scope="session")
def ws1_ip():
    return os.environ.get("DWAIL_WS1_IP")


@pytest.fixture(scope="session")
def ws2_ip():
    return os.environ.get("DWAIL_WS2_IP")


@pytest.fixture(scope="session")
def test_model():
    """
    Model to use for levels 2+.
    Defaults to opt-125m; override with DWAIL_TEST_MODEL for a larger pre-downloaded model.
    """
    return os.environ.get("DWAIL_TEST_MODEL", SMALL_MODEL)


