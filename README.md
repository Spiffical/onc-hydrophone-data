# 🌊 ONC Hydrophone Data Tools

[![PyPI version](https://img.shields.io/pypi/v/onc-hydrophone-data)](https://pypi.org/project/onc-hydrophone-data/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docs](https://img.shields.io/badge/docs-online-brightgreen.svg)](https://spiffical.github.io/onc-hydrophone-data/)

Python tools for downloading Ocean Networks Canada hydrophone audio and
server-generated spectral products, checking deployment availability, and
generating custom spectrograms locally.

## 📦 Installation

```bash
pip install onc-hydrophone-data
```

If you want CPU-only PyTorch (recommended for spectrogram generation on most hosts):
```bash
pip install onc-hydrophone-data \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple
```

For development:
```bash
git clone https://github.com/Spiffical/onc-hydrophone-data.git
cd onc-hydrophone-data
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest
```

Live ONC tests are opt-in because they make API requests and download data:

```bash
pytest -m integration
```

## ⚙️ Configuration

1. Get your ONC API token from: https://data.oceannetworks.ca/Profile

2. Create a `.env` file in your project directory:
```
ONC_TOKEN=your_onc_token_here
DATA_DIR=./data
```

## 🚀 Quick Start

For a guided introduction to both workflows, start with the
**[online documentation](https://spiffical.github.io/onc-hydrophone-data/)**.
An extended [tutorial notebook](notebooks/ONC_Data_Download_Tutorial.ipynb) is
also available for interactive exploration.

### Find a hydrophone

List every hydrophone with its location code, location, deployment period,
depth, and coordinates. Pass `current_only=True` to see only the hydrophones
deployed right now:

```python
from onc_hydrophone_data.data.deployment_checker import HydrophoneDeploymentChecker
from onc_hydrophone_data.onc.common import load_config

onc_token, data_dir = load_config()
checker = HydrophoneDeploymentChecker(onc_token)

checker.print_hydrophone_overview(current_only=True)  # deployed now
checker.print_hydrophone_overview()                   # every deployment, past and present

# The same rows as a pandas DataFrame, for filtering or saving.
overview = checker.get_hydrophone_overview()
overview.to_csv("onc_hydrophone_deployments.csv", index=False)
```

```text
36 hydrophones deployed now at 16 sites (as of 2026-10-22 21:56 UTC)

Location code  Device code                 Begin       End      Depth (m)  Latitude  Longitude  Location
-------------  --------------------------  ----------  -------  ---------  --------  ---------  ----------------------------------------------------------
...
KEMFH.H1       ICLISTENHF6324              2023-09-08  ongoing       2195   47.9493  -129.0982  Main Endeavour Field (Hydrophone A)
...
```

Pass the device code to the download methods and choose times between its
begin and end dates.

### Common workflow: download audio and generate spectrograms locally

```python
from datetime import datetime, timezone
from pathlib import Path

from onc_hydrophone_data.audio import SpectrogramGenerator
from onc_hydrophone_data.data import HydrophoneDownloader
from onc_hydrophone_data.onc.common import load_config

onc_token, data_dir = load_config()
downloader = HydrophoneDownloader(onc_token, data_dir)

device = "ICLISTENHF6324"
start = datetime(2024, 4, 1, 12, 0, tzinfo=timezone.utc)
end = datetime(2024, 4, 1, 12, 10, tzinfo=timezone.utc)

# Download a short, verified ONC audio range.
downloader.download_audio_for_range(
    device_code=device,
    start_dt=start,
    end_dt=end,
)

# Generate PNG and MAT spectrograms locally from the downloaded audio.
audio_dir = Path(downloader.audio_path)
generator = SpectrogramGenerator(
    win_dur=0.5,
    overlap=0.75,
    freq_lims=(20, 10_000),
    crop_freq_lims=True,
)
generator.process_directory(
    audio_dir,
    audio_dir.parent / "custom_spectrograms",
    save_plot=True,
    save_mat=True,
)
```

### Download ONC-generated spectrogram products

Use the same downloader when you want ONC to provide the spectral product
instead of computing it locally:

```python
result = downloader.download_spectrograms_for_range(
    device_code=device,
    start_dt=start,
    end_dt=end,
    spectrograms_per_batch=2,
)
print(result)
```

See the [ONC spectrogram products and server options
guide](https://spiffical.github.io/onc-hydrophone-data/onc_spectrogram_options/)
for one-minute, plot-resolution, and full-resolution products. In the
one-minute product, one minute is the temporal resolution of each spectral
column—not the total duration of the request or output file.

### Generate around a known signal time

When a signal occurs at a known offset in an audio file, event mode retains the
requested signal window and automatically includes extra computation context
to prevent incomplete-window artifacts at its boundaries:

```python
audio_file = next(
    path
    for pattern in ("*.flac", "*.wav")
    for path in audio_dir.glob(pattern)
)
event_result = generator.process_event(
    audio_file,
    audio_dir.parent / "event_spectrograms",
    event_time_seconds=30.0,
    pad_before_seconds=5.0,
    pad_after_seconds=5.0,
    edge_padding_seconds="auto",
)
print(event_result["png_file"])
```

### Command Line

The command-line scripts are in the repository's `scripts/` directory, so run
them from a clone:

```bash
# List hydrophones deployed now (omit --current-only to include past deployments)
python scripts/download_hydrophone_data.py --mode overview --current-only

# Interactive mode (guided setup - recommended)
python scripts/download_hydrophone_data.py

# Download 500 spectrograms sampled from 2024-04-01 to 2024-04-30
python scripts/download_hydrophone_data.py --mode sampling \
    --device ICLISTENHF6324 --start-date 2024 4 1 --end-date 2024 4 30 --threshold 500

# Include FLAC audio files and check the dates against deployments first
python scripts/download_hydrophone_data.py --mode sampling \
    --device ICLISTENHF6324 --start-date 2024 4 1 --end-date 2024 4 30 --threshold 100 \
    --download-audio --check-deployments

# Download every five-minute spectrogram from 2024-04-01 through 2024-04-02
python scripts/download_hydrophone_data.py --mode range \
    --device ICLISTENHF6324 --start-date 2024 4 1 --end-date 2024 4 2

# Generate custom spectrograms from downloaded audio
python scripts/generate_spectrograms.py \
    --input-dir data/ICLISTENHF6324/audio_range_2024-04-01_to_2024-04-01/audio --win-dur 2.0

# Generate around a known signal time with automatic edge context
python scripts/generate_spectrograms.py --input-file audio/example.flac \
    --event-time 30 --event-pad-before 5 --event-pad-after 5 \
    --output-dir event_spectrograms/

# Save only the frequency range needed by the dashboard (much smaller MAT files)
python scripts/generate_spectrograms.py \
    --input-dir data/ICLISTENHF6324/audio_range_2024-04-01_to_2024-04-01/audio \
    --freq-min 10 --freq-max 10000 --crop-freq-lims
```

Commands that include every required option run without prompting; leave
options out to be guided through them. Run either script with `--help` for all
options.

### Deployment Availability Visualization

```python
from onc_hydrophone_data.utils import (
    plot_deployment_availability_timeline,
    plot_availability_calendar,
)

# Reuses `checker` from "Find a hydrophone". Keep the date range modest:
# availability queries list every archived file in the range.
availability = checker.get_device_availability(
    "ICLISTENHF6324",
    start_date="2024-04-01",
    end_date="2024-07-01",
    bin_size="day",
)
plot_deployment_availability_timeline(availability)
plot_availability_calendar(availability)
```

## ✨ Features

- **Hydrophone Overview**: Lists every hydrophone with its location code, location, deployment period, depth, and coordinates, or only those deployed now
- **Smart Sampling**: Intelligently distributes downloads across date ranges
- **Parallel ONC Requests**: Submits many requests at once so ONC processes them in parallel, then downloads when ready (faster than sequential requests)
- **Resumable Audio Downloads**: Downloads FLAC/WAV files in parallel and skips files already present locally
- **Custom Spectrograms**: Generate spectrograms with configurable parameters
- **Event-Centred Spectrograms**: Retain a precise signal window while using automatic STFT context to prevent edge effects
- **JSON Event Workflows**: Download ONC products or generate local event spectrograms with clearly separated request methods
- **Deployment Validation**: Ensures data exists for requested time periods
- **Deployment Availability Visuals**: Timeline/calendar views of data availability by device
- **Interactive Mode**: Guided CLI for easy setup

## 📁 Output Structure

Downloads are organized in a clean, flat structure:

```
data/
└── ICLISTENHF6324/
    └── audio_range_2024-04-01_to_2024-04-01/
        ├── onc_spectrograms/     # ONC-downloaded spectrograms (MAT/PNG)
        │   ├── *.mat             # Spectrogram data files
        │   └── anomaly_report.txt # Any validation issues (if found)
        ├── audio/                # Downloaded audio files
        │   └── *.flac / *.wav
        └── custom_spectrograms/  # Locally-generated spectrograms
            ├── *.mat             # Spectrogram arrays + metadata
            └── *.png             # Spectrogram plots
```

## 🛠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| Invalid ONC Token | Verify token in `.env` file |
| No data found | Run `print_hydrophone_overview()` (or `--mode overview`) and request times inside a deployment; add `--check-deployments` to CLI downloads |
| ONC request timeouts | Reduce `--spectrograms-per-batch` or request a shorter range |
| Local generation is slow or uses too much memory | Use `--max-workers 1` and `--crop-freq-lims` with a focused frequency range |

## 📚 Documentation

Docs site: **https://spiffical.github.io/onc-hydrophone-data/**  
See the **[Tutorial Notebook](notebooks/ONC_Data_Download_Tutorial.ipynb)** for comprehensive examples including:
- Finding hydrophones, their locations, and deployment dates
- Different download modes (sampling, range, specific times)
- Parallel download optimization
- Custom spectrogram generation
- Edge-safe generation around known signal times
- JSON requests for local generation versus ONC-generated products

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.
