# 🌊 ONC Hydrophone Data Tools

[![PyPI version](https://badge.fury.io/py/onc-hydrophone-data.svg)](https://pypi.org/project/onc-hydrophone-data/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Tools for downloading and processing Ocean Networks Canada hydrophone data, including spectrograms, FLAC audio files, and custom spectrogram generation.

## 📦 Installation

```bash
pip install onc-hydrophone-data
```

For development:
```bash
git clone https://github.com/Spiffical/onc-hydrophone-data.git
cd onc-hydrophone-data
pip install -e .
```

## ⚙️ Configuration

1. Get your ONC API token from: https://data.oceannetworks.ca/Profile

2. Create a `.env` file in your project directory:
```
ONC_TOKEN=your_onc_token_here
DATA_DIR=./data
```

## 🚀 Quick Start

**📓 [Tutorial Notebook](notebooks/ONC_Data_Download_Tutorial.ipynb)** - The best way to get started with interactive examples.

### Python API

```python
from onc_hydrophone_data.onc.common import load_config
from onc_hydrophone_data.data import HydrophoneDownloader
from onc_hydrophone_data.audio import SpectrogramGenerator

# Load credentials from .env file
onc_token, data_dir = load_config()

# Download spectrograms using intelligent sampling
downloader = HydrophoneDownloader(onc_token, data_dir)
downloader.download_spectrograms_with_sampling_schedule(
    deviceCode="ICLISTENHF6020",
    start_date=(2021, 1, 1),
    threshold_num=100
)

# Generate custom spectrograms from audio files
generator = SpectrogramGenerator(win_dur=2.0, overlap=0.75)
generator.process_directory("data/DEVICE/flac/", "output/spectrograms/")
```

### Command Line

```bash
# Interactive mode (guided setup - recommended)
python scripts/download_hydrophone_data.py

# Download spectrograms with specific parameters
python scripts/download_hydrophone_data.py --mode sampling \
    --device ICLISTENHF6020 --start-date 2021 1 1 --threshold 500

# Include FLAC audio files
python scripts/download_hydrophone_data.py --mode sampling \
    --device ICLISTENHF6020 --start-date 2021 1 1 --threshold 100 --download-flac

# Generate custom spectrograms
python scripts/generate_spectrograms.py --input-dir data/DEVICE/flac/ --win-dur 2.0
```

## ✨ Features

- **Smart Sampling**: Intelligently distributes downloads across date ranges
- **FLAC Audio**: Download raw audio alongside spectrograms
- **Custom Spectrograms**: Generate spectrograms with configurable parameters
- **Deployment Validation**: Ensures data exists for requested time periods
- **Interactive Mode**: Guided CLI for easy setup

## 📁 Output Structure

```
data/
└── ICLISTENHF6020/
    └── sampling_2021-01-01_to_2021-01-31/
        ├── mat/processed/     # Spectrograms
        └── flac/              # Audio files (optional)
```

## 🛠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| Invalid ONC Token | Verify token in `.env` file |
| No data found | Use `--check-deployments` to verify coverage |
| Memory errors | Reduce `--spectrograms-per-batch` |

## 📚 Documentation

See the **[Tutorial Notebook](notebooks/ONC_Data_Download_Tutorial.ipynb)** for comprehensive examples including:
- Different download modes (sampling, range, specific times)
- Parallel download optimization
- Custom spectrogram generation
- JSON timestamp requests

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.