import logging
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from onc_hydrophone_data.data.downloader.onc_downloads import download_audio_files
from onc_hydrophone_data.data.downloader.parallel_requests import run_parallel_windows


def test_audio_download_skips_existing_nonempty_file(tmp_path: Path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    existing_name = "DEVICE_20240401T120000.000Z.flac"
    (audio_dir / existing_name).write_bytes(b"already downloaded")

    onc = MagicMock()
    onc.outPath = "original"
    onc.getListByDevice.return_value = {"files": [existing_name]}
    downloader = SimpleNamespace(
        onc=onc,
        logger=MagicMock(),
        audio_path=str(audio_dir),
        max_workers=4,
        _parse_timestamp_value=lambda value: value,
        _build_request_windows=lambda start, end: [(start, end)],
    )

    summary = download_audio_files(
        downloader,
        "DEVICE",
        "2024-04-01T12:00:00Z",
        "2024-04-01T12:05:00Z",
    )

    assert summary["files_downloaded"] == 1
    assert summary["files_skipped"] == 1
    onc.getFile.assert_not_called()
    assert onc.outPath == "original"


def test_mat_requests_submit_concurrently_by_default(tmp_path: Path):
    class ConcurrentRequestManager:
        def __init__(self):
            self.active = 0
            self.max_active = 0
            self.lock = threading.Lock()

        def submit_mat_run_no_wait(self, **kwargs):
            with self.lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            time.sleep(0.03)
            with self.lock:
                self.active -= 1
            return {
                "dpRequestId": kwargs["start_dt"].minute,
                "status": "downloaded",
            }

    request_manager = ConcurrentRequestManager()
    downloader = SimpleNamespace(
        max_workers=4,
        request_manager=request_manager,
        logger=logging.getLogger("test-parallel-requests"),
        spectrogram_path=str(tmp_path),
        input_path=str(tmp_path),
        flac_path=str(tmp_path / "audio"),
        setup_directories=lambda *args, **kwargs: None,
    )
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    windows = [
        (start + timedelta(minutes=i), start + timedelta(minutes=i + 1))
        for i in range(4)
    ]

    summary = run_parallel_windows(
        downloader,
        "DEVICE",
        windows,
        spectrograms_per_request=1,
        max_wait_minutes=0,
    )

    assert request_manager.max_active > 1
    assert summary["runs_total"] == 4
    assert summary["runs_downloaded"] == 4
