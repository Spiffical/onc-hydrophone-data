# Quickstart

This is a minimal end‑to‑end example: load config, pick a device, and download a short window of spectrograms.

```python
from datetime import datetime, timezone
from onc_hydrophone_data.onc.common import load_config
from onc_hydrophone_data.data.hydrophone_downloader import HydrophoneDownloader

ONC_TOKEN, DATA_DIR = load_config()
dl = HydrophoneDownloader(ONC_TOKEN, DATA_DIR)

DEVICE = "ICLISTENHF6324"
start = datetime(2024, 4, 1, 12, 0, tzinfo=timezone.utc)
end = datetime(2024, 4, 1, 12, 10, tzinfo=timezone.utc)

result = dl.download_spectrograms_for_range(
    DEVICE,
    start,
    end,
    spectrograms_per_batch=2,
)
print(result)
```

!!! note
    This call submits ONC requests up front and waits for data to become ready.
    ONC executes requests in parallel on their servers for faster throughput.

If you want audio alongside spectrograms, add `download_audio=True`.

```python
result = dl.download_spectrograms_for_range(
    DEVICE,
    start,
    end,
    spectrograms_per_batch=2,
    download_audio=True,
)
```
