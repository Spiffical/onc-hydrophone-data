import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "generate_spectrograms.py"


@pytest.mark.parametrize(
    ("input_flag", "extra_args", "message"),
    [
        ("--input-dir", [], "--event-time requires --input-file"),
        (
            "--input-file",
            ["--clip-start", "0"],
            "--event-time cannot be combined with --clip-start/--clip-end",
        ),
        (
            "--input-file",
            ["--event-pad-before", "-1"],
            "--event-pad-before must be non-negative",
        ),
    ],
)
def test_event_argument_errors_use_argparse_without_traceback(
    tmp_path: Path,
    input_flag: str,
    extra_args: list[str],
    message: str,
):
    input_path = tmp_path
    if input_flag == "--input-file":
        input_path = tmp_path / "audio.wav"
        input_path.touch()

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            input_flag,
            str(input_path),
            "--event-time",
            "1",
            *extra_args,
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 2
    assert f"error: {message}" in completed.stderr
    assert "Traceback" not in completed.stderr
    assert "Unexpected error" not in completed.stdout + completed.stderr


@pytest.mark.parametrize("flag", ["--clip-pad-seconds", "--edge-pad-seconds"])
@pytest.mark.parametrize(
    "event_args",
    [
        pytest.param([], id="without-event"),
        pytest.param(["--event-time", "1"], id="with-event"),
    ],
)
def test_negative_edge_padding_aliases_use_argparse_without_traceback(
    tmp_path: Path,
    flag: str,
    event_args: list[str],
):
    input_path = tmp_path / "audio.wav"
    input_path.touch()

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--input-file",
            str(input_path),
            flag,
            "-0.1",
            *event_args,
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 2
    assert (
        "error: --clip-pad-seconds/--edge-pad-seconds must be non-negative"
        in completed.stderr
    )
    assert "Traceback" not in completed.stderr
    assert "Unexpected error" not in completed.stdout + completed.stderr
