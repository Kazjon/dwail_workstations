"""
Tests for the agent install script logic.

Systemd and Ray subprocess calls are mocked — these tests verify
the correct commands are constructed and service file content is correct.
Root check is also bypassed.
"""

import textwrap
from pathlib import Path
from unittest.mock import call, patch, MagicMock

import pytest

from dwail_agent.install import (
    _ensure_model_dir,
    _write_service,
    _setup_ray,
    _uninstall,
    DEFAULT_MODEL_DIR,
    DEFAULT_PORT,
    SERVICE_FILE,
    SERVICE_NAME,
)


# --- _ensure_model_dir ---

def test_ensure_model_dir_creates_directory(tmp_path):
    target = tmp_path / "models" / "subdir"
    _ensure_model_dir(target)
    assert target.exists()


def test_ensure_model_dir_idempotent(tmp_path):
    target = tmp_path / "models"
    _ensure_model_dir(target)
    _ensure_model_dir(target)  # should not raise
    assert target.exists()


# --- _write_service ---

def test_write_service_content(tmp_path, monkeypatch):
    service_path = tmp_path / "dwail-agent.service"
    monkeypatch.setattr("dwail_agent.install.SERVICE_FILE", service_path)

    _write_service(
        python="/usr/bin/python3",
        agent_bin="/usr/local/bin/dwail-agent",
        model_dir=Path("/mnt/models"),
        port=8765,
    )

    content = service_path.read_text()
    assert "ExecStart=/usr/local/bin/dwail-agent" in content
    assert "DWAIL_MODEL_DIR=/mnt/models" in content
    assert "DWAIL_PORT=8765" in content
    assert "Restart=on-failure" in content
    assert "WantedBy=multi-user.target" in content
    # venv bin must be on PATH so systemd finds dwail-agent and ray at boot
    assert "Environment=PATH=/usr/bin" in content


def test_write_service_custom_port(tmp_path, monkeypatch):
    service_path = tmp_path / "dwail-agent.service"
    monkeypatch.setattr("dwail_agent.install.SERVICE_FILE", service_path)

    _write_service("/usr/bin/python3", "/usr/bin/dwail-agent", Path("/data/models"), port=9000)
    assert "DWAIL_PORT=9000" in service_path.read_text()


def test_write_service_is_idempotent(tmp_path, monkeypatch):
    service_path = tmp_path / "dwail-agent.service"
    monkeypatch.setattr("dwail_agent.install.SERVICE_FILE", service_path)

    _write_service("/usr/bin/python3", "/usr/bin/dwail-agent", Path("/mnt/models"), 8765)
    first = service_path.read_text()
    _write_service("/usr/bin/python3", "/usr/bin/dwail-agent", Path("/mnt/models"), 8765)
    second = service_path.read_text()
    assert first == second


# --- _setup_ray ---

def test_setup_ray_head_starts_head_node(mocker, tmp_path):
    # ray exists in the venv bin dir (same dir as sys.executable)
    fake_ray = tmp_path / "ray"
    fake_ray.touch()
    mocker.patch("dwail_agent.install.sys.executable", str(tmp_path / "python3"))
    mock_run = mocker.patch("dwail_agent.install.subprocess.run")
    mock_run_internal = mocker.patch("dwail_agent.install._run")

    _setup_ray(as_head=True, worker_head_ip=None)

    mock_run.assert_called_once_with([str(fake_ray), "stop"], capture_output=True)
    mock_run_internal.assert_called_once_with([str(fake_ray), "start", "--head", "--port=6379"])


def test_setup_ray_worker_connects_to_head(mocker, tmp_path):
    fake_ray = tmp_path / "ray"
    fake_ray.touch()
    mocker.patch("dwail_agent.install.sys.executable", str(tmp_path / "python3"))
    mocker.patch("dwail_agent.install.subprocess.run")
    mock_run_internal = mocker.patch("dwail_agent.install._run")

    _setup_ray(as_head=False, worker_head_ip="10.147.20.5")

    mock_run_internal.assert_called_once_with(
        [str(fake_ray), "start", "--address=10.147.20.5:6379"]
    )


def test_setup_ray_defaults_to_head_when_neither_specified(mocker, tmp_path):
    fake_ray = tmp_path / "ray"
    fake_ray.touch()
    mocker.patch("dwail_agent.install.sys.executable", str(tmp_path / "python3"))
    mocker.patch("dwail_agent.install.subprocess.run")
    mock_run_internal = mocker.patch("dwail_agent.install._run")

    _setup_ray(as_head=False, worker_head_ip=None)

    mock_run_internal.assert_called_once_with([str(fake_ray), "start", "--head", "--port=6379"])


def test_setup_ray_warns_when_ray_not_found(mocker, capsys):
    mocker.patch("dwail_agent.install.shutil.which", return_value=None)

    _setup_ray(as_head=True, worker_head_ip=None)

    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "ray" in out.lower()


# --- _uninstall ---

def test_uninstall_stops_and_removes_service(tmp_path, mocker, monkeypatch):
    service_path = tmp_path / "dwail-agent.service"
    service_path.write_text("[Unit]\nDescription=test\n")
    monkeypatch.setattr("dwail_agent.install.SERVICE_FILE", service_path)

    mock_run = mocker.patch("dwail_agent.install.subprocess.run")
    mock_run_internal = mocker.patch("dwail_agent.install._run")
    mocker.patch("dwail_agent.install.shutil.which", return_value="/usr/bin/ray")

    _uninstall()

    assert not service_path.exists()
    # systemctl stop, disable and daemon-reload should have been called
    calls = [c.args[0] for c in mock_run.call_args_list]
    assert ["systemctl", "stop", SERVICE_NAME] in calls
    assert ["systemctl", "disable", SERVICE_NAME] in calls


def test_uninstall_tolerates_missing_service_file(tmp_path, mocker, monkeypatch):
    service_path = tmp_path / "nonexistent.service"
    monkeypatch.setattr("dwail_agent.install.SERVICE_FILE", service_path)
    mocker.patch("dwail_agent.install.subprocess.run")
    mocker.patch("dwail_agent.install._run")
    mocker.patch("dwail_agent.install.shutil.which", return_value=None)

    _uninstall()  # should not raise
