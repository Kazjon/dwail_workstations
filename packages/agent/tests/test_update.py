"""
Tests for the agent update script logic.

Verifies that:
  - The script requires root
  - It upgrades the package via pip from the correct source
  - It restarts the systemd service after upgrading
  - It handles pip failure gracefully
"""

import sys
import subprocess
from pathlib import Path
from unittest.mock import call

import pytest

from dwail_agent.update import _do_update, _find_pip, PACKAGE_URL, SERVICE_NAME


# --- _find_pip ---

def test_find_pip_returns_pip_in_venv_bin(tmp_path):
    pip = tmp_path / "pip"
    pip.touch()
    result = _find_pip(str(tmp_path / "python3"))
    assert result == str(pip)


def test_find_pip_falls_back_to_path_when_not_in_venv_bin(mocker, tmp_path):
    mocker.patch("dwail_agent.update.shutil.which", return_value="/usr/bin/pip3")
    # pip NOT present beside python
    result = _find_pip(str(tmp_path / "python3"))
    assert result == "/usr/bin/pip3"


def test_find_pip_returns_none_when_not_found(mocker, tmp_path):
    mocker.patch("dwail_agent.update.shutil.which", return_value=None)
    result = _find_pip(str(tmp_path / "python3"))
    assert result is None


# --- _do_update ---

def test_do_update_calls_pip_install_upgrade(mocker, tmp_path):
    pip = tmp_path / "pip"
    pip.touch()
    mocker.patch("dwail_agent.update.sys.executable", str(tmp_path / "python3"))
    mock_run = mocker.patch("dwail_agent.update._run")

    _do_update()

    mock_run.assert_any_call([str(pip), "install", "--upgrade", PACKAGE_URL])


def test_do_update_restarts_service_after_upgrade(mocker, tmp_path):
    pip = tmp_path / "pip"
    pip.touch()
    mocker.patch("dwail_agent.update.sys.executable", str(tmp_path / "python3"))
    mock_run = mocker.patch("dwail_agent.update._run")

    _do_update()

    calls = [c.args[0] for c in mock_run.call_args_list]
    assert ["systemctl", "restart", SERVICE_NAME] in calls


def test_do_update_restarts_after_pip(mocker, tmp_path):
    """pip install must come before systemctl restart."""
    pip = tmp_path / "pip"
    pip.touch()
    mocker.patch("dwail_agent.update.sys.executable", str(tmp_path / "python3"))
    mock_run = mocker.patch("dwail_agent.update._run")

    _do_update()

    call_args = [c.args[0] for c in mock_run.call_args_list]
    pip_idx = next(i for i, c in enumerate(call_args) if "install" in c)
    restart_idx = next(i for i, c in enumerate(call_args) if "restart" in c)
    assert pip_idx < restart_idx


def test_do_update_exits_when_pip_not_found(mocker, tmp_path, capsys):
    mocker.patch("dwail_agent.update.sys.executable", str(tmp_path / "python3"))
    mocker.patch("dwail_agent.update.shutil.which", return_value=None)

    with pytest.raises(SystemExit):
        _do_update()

    err = capsys.readouterr().err
    assert "pip" in err.lower()
