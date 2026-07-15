# 🌊 ONC Hydrophone Data Tools

[![PyPI version](https://img.shields.io/pypi/v/onc-hydrophone-data)](https://pypi.org/project/onc-hydrophone-data/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docs](https://img.shields.io/badge/docs-online-brightgreen.svg)](https://spiffical.github.io/onc-hydrophone-data/)

Tools for downloading and processing Ocean Networks Canada hydrophone data, including spectrograms, FLAC audio files, and custom spectrogram generation.

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

For the guided beginner path, start with the
**[online documentation](https://spiffical.github.io/onc-hydrophone-data/)**.
An extended [tutorial notebook](notebooks/ONC_Data_Download_Tutorial.ipynb) is
also available for interactive exploration.

### Python API

```python
from datetime import datetime, timezone
from pathlib import Path

from onc_hydrophone_data.audio import SpectrogramGenerator
from onc_hydrophone_data.data import HydrophoneDownloader
from onc_hydrophone_data.onc.common import load_config

onc_token, data_dir = load_config()
downloader = HydrophoneDownloader(onc_token, data_dir)

# Download a short, verified ONC audio range.
downloader.download_audio_for_range(
    device_code="ICLISTENHF6324",
    start_dt=datetime(2024, 4, 1, 12, 0, tzinfo=timezone.utc),
    end_dt=datetime(2024, 4, 1, 12, 10, tzinfo=timezone.utc),
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

### Command Line

```bash
# Interactive mode (guided setup - recommended)
python scripts/download_hydrophone_data.py

# Download spectrograms with specific parameters
python scripts/download_hydrophone_data.py --mode sampling \
    --device ICLISTENHF6324 --start-date 2024 4 1 --threshold 500

# Include FLAC audio files
python scripts/download_hydrophone_data.py --mode sampling \
    --device ICLISTENHF6324 --start-date 2024 4 1 --threshold 100 --download-audio

# Generate custom spectrograms
python scripts/generate_spectrograms.py --input-dir data/DEVICE/audio/ --win-dur 2.0

# Save only the frequency range needed by the dashboard (much smaller MAT files)
python scripts/generate_spectrograms.py --input-dir data/DEVICE/audio/ \
    --freq-min 10 --freq-max 10000 --crop-freq-lims
```

### Deployment Availability Visualization

```python
from onc_hydrophone_data.data.deployment_checker import HydrophoneDeploymentChecker
from onc_hydrophone_data.utils import (
    plot_deployment_availability_timeline,
    plot_availability_calendar,
)

checker = HydrophoneDeploymentChecker(onc_token)
availability = checker.get_device_availability("ICLISTENHF6324", bin_size="day")
plot_deployment_availability_timeline(availability)
plot_availability_calendar(availability)
```

## ✨ Features

- **Smart Sampling**: Intelligently distributes downloads across date ranges
- **Parallel ONC Requests**: Submits many requests at once so ONC processes them in parallel, then downloads when ready (faster than sequential requests)
- **Resumable Audio Downloads**: Downloads FLAC/WAV files in parallel and skips files already present locally
- **Custom Spectrograms**: Generate spectrograms with configurable parameters
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
        │   └── *.flac
        └── custom_spectrograms/  # Locally-generated spectrograms
            ├── mat/              # Custom MAT files
            └── png/              # Custom PNG plots
```

## 🛠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| Invalid ONC Token | Verify token in `.env` file |
| No data found | Use `--check-deployments` to verify coverage |
| Memory errors | Reduce `--spectrograms-per-batch` |

## 📚 Documentation

Docs site: **https://spiffical.github.io/onc-hydrophone-data/**  
See the **[Tutorial Notebook](notebooks/ONC_Data_Download_Tutorial.ipynb)** for comprehensive examples including:
- Different download modes (sampling, range, specific times)
- Parallel download optimization
- Custom spectrogram generation
- JSON timestamp requests

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.
