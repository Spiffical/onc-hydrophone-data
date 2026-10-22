import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "download_hydrophone_data.py"
_SCRIPT_SPEC = importlib.util.spec_from_file_location("download_hydrophone_data", _SCRIPT_PATH)
assert _SCRIPT_SPEC is not None and _SCRIPT_SPEC.loader is not None
cli = importlib.util.module_from_spec(_SCRIPT_SPEC)
_SCRIPT_SPEC.loader.exec_module(cli)


class FakeDownloader:
    """Records the library calls the CLI makes instead of contacting ONC."""

    instances = []

    def __init__(self, onc_token, parent_dir):
        self.calls = []
        FakeDownloader.instances.append(self)

    def download_spectrograms_with_sampling_schedule(self, *args, **kwargs):
        self.calls.append(("sampling", args, kwargs))

    def download_spectrograms_for_range(self, *args, **kwargs):
        self.calls.append(("range", args, kwargs))
        return {"runs_downloaded": 0}


@pytest.fixture
def run_cli(monkeypatch):
    FakeDownloader.instances = []
    monkeypatch.setattr(cli, "load_config", lambda data_dir_override=None: ("token", "data"))
    monkeypatch.setattr(cli, "HydrophoneDownloader", FakeDownloader)

    def prompt(*args, **kwargs):
        raise AssertionError("the CLI prompted for input")

    monkeypatch.setattr("builtins.input", prompt)

    def run(*args):
        monkeypatch.setattr(sys, "argv", ["download_hydrophone_data.py", *args])
        cli.main()
        return FakeDownloader.instances

    return run


def test_overview_mode_prints_the_overview(run_cli, monkeypatch):
    calls = []

    class FakeChecker:
        def __init__(self, onc_token, debug=False):
            pass

        def print_hydrophone_overview(self, *, current_only=False):
            calls.append(current_only)

    monkeypatch.setattr(cli, "HydrophoneDeploymentChecker", FakeChecker)

    downloaders = run_cli("--mode", "overview", "--current-only")

    assert calls == [True]
    assert downloaders == []


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--current-only"], "--current-only requires --mode overview"),
        (["--mode", "specific"], "--mode specific requires --config FILE"),
    ],
)
def test_invalid_mode_arguments_are_reported(run_cli, capsys, args, message):
    with pytest.raises(SystemExit) as excinfo:
        run_cli(*args)

    assert excinfo.value.code == 2
    assert message in capsys.readouterr().err


def test_fully_specified_sampling_runs_without_prompts(run_cli):
    (downloader,) = run_cli(
        "--mode", "sampling", "--device", "ICLISTENHF6324", "--start-date", "2024", "4", "1", "--threshold", "4",
    )

    ((name, args, kwargs),) = downloader.calls
    assert name == "sampling"
    assert args[:3] == ("ICLISTENHF6324", [2024, 4, 1], 4)
    assert kwargs["download_audio"] is False


def test_range_mode_downloads_through_the_end_date(run_cli):
    (downloader,) = run_cli(
        "--mode", "range", "--device", "ICLISTENHF6324",
        "--start-date", "2024", "4", "1", "--end-date", "2024", "4", "2",
        "--spectrograms-per-batch", "12",
    )

    assert downloader.calls == [(
        "range",
        (
            "ICLISTENHF6324",
            datetime(2024, 4, 1, tzinfo=timezone.utc),
            datetime(2024, 4, 3, tzinfo=timezone.utc),
            12,
        ),
        {"download_audio": False},
    )]
