# Download Audio

Use this guide after completing the
**[audio-to-spectrogram walkthrough](quickstart.md)**. Downloading audio first is
the most flexible workflow because you can generate many spectrogram variants
without querying ONC again.

## Start with a short range

```python
from datetime import datetime, timezone

from onc_hydrophone_data.data import HydrophoneDownloader
from onc_hydrophone_data.onc.common import load_config

onc_token, data_dir = load_config()
dl = HydrophoneDownloader(onc_token, data_dir)

dl.download_audio_for_range(
    device_code="ICLISTENHF6324",
    start_dt=datetime(2024, 4, 1, 12, 0, tzinfo=timezone.utc),
    end_dt=datetime(2024, 4, 1, 12, 10, tzinfo=timezone.utc),
)

print("Audio directory:", dl.audio_path)
```

The package tries FLAC first and WAV second. It downloads each ONC source file
that overlaps the requested range; source files commonly span five minutes.

!!! warning "Audio can be large"
    File size depends on sample rate, channel count, and compression. Test a
    ten-minute range before requesting hours or days.

## Output location and resume behaviour

The default range layout is:

```text
DATA_DIR/
└── DEVICE_CODE/
    └── audio_range_START_DATE_to_END_DATE/
        ├── audio/
        └── onc_spectrograms/
```

`dl.audio_path` always points to the active `audio/` directory. On a rerun,
existing non-empty files are skipped. This makes interrupted downloads safe to
resume without spending bandwidth on completed files.

## Choose preferred audio formats

```python
dl.download_audio_for_range(
    device_code=DEVICE,
    start_dt=start,
    end_dt=end,
    extensions=("flac", "wav"),
    max_download_workers=4,
)
```

- Put `"wav"` first only when uncompressed WAV is specifically required.
- Reduce `max_download_workers` on a slow or unstable connection.
- Times should be timezone-aware; UTC is the clearest choice.

## Sample a long period

To explore seasonal or long-term variation without downloading every file,
request a uniform sample:

```python
result = dl.download_sampled_audio(
    device_code=DEVICE,
    start_dt=start,
    end_dt=end,
    total_audio_files=24,
    files_per_request=4,
)
```

This chooses request windows across the full range. Confirm deployment and
archive availability first so samples are not wasted on gaps.

## Download around event times

```python
events = [
    datetime(2024, 4, 1, 12, 5, tzinfo=timezone.utc),
    datetime(2024, 4, 1, 13, 25, tzinfo=timezone.utc),
]

result = dl.download_audio_for_events(
    device_code=DEVICE,
    event_times=events,
    window_seconds=300,
)
```

Use event downloads when you have detections, annotations, or field notes and
want each five-minute source window containing a timestamp. For precisely
padded clips, large event lists, or mixed devices, use the
**[Advanced & Batch Downloads](downloads.md)** JSON/CSV workflow.

## Next step

Once files are present under `dl.audio_path`, continue to
**[Generate Local Spectrograms](custom_spectrograms.md)**.
