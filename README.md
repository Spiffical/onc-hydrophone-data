# 🌊 ONC Data Download and Preparation

Complete guide for downloading Ocean Networks Canada spectrograms, FLAC audio files, and preparing ML-ready HDF5 datasets.

## 📋 Table of Contents

- [🚀 Quick Start](#-quick-start)
- [⚙️ Setup](#️-setup)
- [📥 Downloading Spectrograms](#-downloading-spectrograms)
- [🎵 Downloading FLAC Audio Files](#-downloading-flac-audio-files)
- [🎶 Custom Spectrogram Generation](#-custom-spectrogram-generation)
- [🔧 Advanced Options](#-advanced-options)
- [🛠️ Troubleshooting](#️-troubleshooting)

## 🚀 Quick Start

The **easiest way to get started** is using the tutorial notebook, which runs through several real-world examples:
- 📓 **[Tutorial Notebook](notebooks/ONC_Data_Download_Tutorial.ipynb)**

### CLI Quick Start

```bash
# 1. Interactive download (guided cli) - uses sampling strategy
#    Now includes option to download FLAC files!
python scripts/download_hydrophone_data.py

# 2. Direct download with custom batch size
python scripts/download_hydrophone_data.py --mode sampling --device ICLISTENHF6020 --start-date 2021 1 1 --threshold 500 --spectrograms-per-batch 12 --check-deployments

# 3. Download spectrograms WITH corresponding FLAC audio files
python scripts/download_hydrophone_data.py --mode sampling --device ICLISTENHF6020 --start-date 2021 1 1 --threshold 500 --spectrograms-per-batch 6 --download-flac

# 4. Generate custom spectrograms from FLAC files (NEW!)
python scripts/generate_spectrograms.py --input-dir data/ICLISTENHF6020/flac/ --win-dur 2.0
```

## ✨ Key Features

- **🤖 Smart Interactive Mode**: Guided setup that uses the intelligent sampling strategy and includes FLAC audio option
- **🎵 FLAC Audio Download**: Download corresponding raw audio files alongside spectrograms
- **🎶 Custom Spectrogram Generation**: Create spectrograms with any duration/parameters from FLAC files
- **🚀 Deployment Validation**: Ensures hydrophones were deployed during requested periods  
- **📊 Device Discovery**: Browse available hydrophones with deployment information
- **⏰ Date Validation**: Checks dates fall within active deployment periods
- **💾 Efficient Caching**: Minimizes API calls through intelligent caching
- **🔧 Multiple Modes**: Sampling, range, specific times, and deployment checking
- **📁 Universal Folder Support**: Works with enhanced, flat, and nested folder structures

## ⚙️ Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure ONC API token:**
   Create/edit `.env` file:
   ```
   ONC_TOKEN=your_actual_onc_token_here
   DATA_DIR=./data
   ```

## 📥 Downloading Spectrograms

### 🎯 Usage Modes

| Mode | Description | Example |
|------|-------------|---------|
| **Interactive** | Guided setup using **sampling strategy** (recommended) | `python scripts/download_hydrophone_data.py` |
| **Sampling** | Smart sampling from date range | `--mode sampling --threshold 1000` |
| **Range** | All spectrograms in date range | `--mode range --start-date 2021 1 1 --end-date 2021 1 7` |
| **Specific** | Exact timestamps from JSON | `--mode specific --config times.json` |
| **Check** | View deployment info | `--mode check-deployments` |

**📌 Note**: **Interactive mode** is simply a guided way to set up the intelligent sampling strategy. It prompts you for device, dates, threshold, and spectrograms per batch, then uses the same smart sampling algorithm described below.

### 🧠 Intelligent Sampling Strategy

The **sampling mode** (including **interactive mode**) uses a smart algorithm to efficiently distribute downloads across your date range:

**How it works:**
1. **Data Availability Check**: Queries ONC API to find which days have data available
2. **Request Calculation**: Determines number of requests needed based on `spectrograms_per_batch`:
   ```
   total_requests = ceil(threshold_num / spectrograms_per_batch)
   ```
3. **Optimal Day Spacing**: Distributes requests evenly across available days
4. **Random Time Distribution**: Uses random hours (0-23) and minutes (0-59) for maximum temporal diversity
5. **Duplicate Prevention**: Automatically skips dates where files already exist
6. **Adaptive Sampling**: Handles both sparse sampling across many days and multiple requests per day

**Benefits:**
- **Even temporal coverage** across your entire date range
- **Full 24-hour sampling** with random start times for maximum diversity
- **Efficient API usage** by checking availability first
- **Resume-friendly** by skipping existing downloads

### 📊 Spectrograms Per Batch

Control how many 5-minute spectrograms are downloaded per request with `--spectrograms-per-batch`:

| Batch Size | Duration |
|------------|----------|
| `1` | 5 minutes |
| `6` | 30 minutes (default) |
| `12` | 1 hour |
| `36` | 3 hours |

```bash
# Custom batch size example
python scripts/download_hydrophone_data.py --mode sampling --spectrograms-per-batch 12
```

### 🚀 Deployment Validation

Ensures hydrophones were active during requested periods. Add `--check-deployments` to verify:
- ✅ Deployment coverage for your dates
- 📍 Exact locations and coordinates  
- 🔍 Data availability verification
- 💡 Alternative suggestions if needed

```bash
python scripts/download_hydrophone_data.py --check-deployments
```

### 🎛️ Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--mode` | Download mode | Interactive prompt |
| `--device` | Hydrophone device code | Interactive selection |
| `--spectrograms-per-batch` | Number of 5-min spectrograms per request | 6 |
| `--download-flac` | Also download FLAC audio files | False |
| `--check-deployments` | Validate deployment periods | Recommended |
| `--start-date` | Start date (YYYY MM DD) | Prompted |
| `--end-date` | End date (YYYY MM DD) | Prompted |
| `--threshold` | Number of spectrograms | Prompted |

### 📁 File Organization

Downloads are organized by device, method, and date range:

```
data/
└── DEVICE/
    └── sampling_YYYY-MM-DD_to_YYYY-MM-DD/
        ├── mat/
        │   ├── processed/     # Downloaded spectrograms
        │   └── rejects/       # Quality-filtered files
        └── flac/              # FLAC audio files (if --download-flac used)
```

**Example:** `data/ICLISTENHF6020/sampling_2021-01-01_to_2021-01-31/`

### 📝 Specific Times Config

For exact timestamps, create a JSON file:
```json
{
  "ICLISTENHF6020": [
    [2021, 1, 15, 12, 0, 0],
    [2021, 1, 15, 18, 30, 0]
  ]
}
```
Format: `[Year, Month, Day, Hour, Minute, Second]`

## 🎵 Downloading FLAC Audio Files

FLAC files contain raw hydrophone audio recordings. Add `--download-flac` to any command or use interactive mode (which now prompts for FLAC preference):

```bash
# Interactive mode (prompts for FLAC)
python scripts/download_hydrophone_data.py

# Any mode with FLAC
python scripts/download_hydrophone_data.py --mode sampling --download-flac
```

**Use Cases**: Audio analysis, custom spectrograms, ML training on raw audio
**File Organization**: FLAC files saved in `flac/` subdirectory alongside spectrograms  
**Performance**: 10-50x larger than spectrograms; start with small downloads (--threshold 5-10)

## 🎶 Custom Spectrogram Generation

Generate custom spectrograms from your downloaded FLAC/WAV audio files with configurable parameters. This functionality translates MATLAB spectrogram code to Python, allowing you to create spectrograms with different durations, frequency ranges, and analysis parameters.

### ✨ Features
- **Multiple audio formats**: FLAC, WAV, MP3, M4A support
- **Configurable parameters**: Window duration, overlap, frequency limits
- **MATLAB compatibility**: Outputs .mat files with same structure as MATLAB
- **High-quality plots**: PNG visualizations with customizable colormaps
- **Batch processing**: Process entire directories efficiently
- **Project integration**: Works seamlessly with downloaded FLAC files

### 🚀 Quick Start

```bash
# Interactive mode (recommended)
python scripts/generate_spectrograms.py

# Process FLAC files from ONC downloads
python scripts/generate_spectrograms.py --input-dir data/ICLISTENHF6020/flac/

# Custom parameters for longer spectrograms  
python scripts/generate_spectrograms.py \
  --input-dir data/DEVICE/flac/ \
  --win-dur 2.0 \
  --overlap 0.75 \
  --freq-min 5 \
  --freq-max 20000
```

### 🎛️ Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--win-dur` | Window duration in seconds | 1.0 |
| `--overlap` | Overlap ratio (0-1) | 0.5 |
| `--freq-min` | Minimum frequency (Hz) | 10 |
| `--freq-max` | Maximum frequency (Hz) | 10000 |
| `--colormap` | Matplotlib colormap | turbo |
| `--clim-min` | Color scale minimum (dB) | -60 |
| `--clim-max` | Color scale maximum (dB) | 0 |

### 📁 Output Structure

Custom spectrograms are saved parallel to FLAC directories:

```
data/
└── DEVICE/
    └── sampling_YYYY-MM-DD_to_YYYY-MM-DD/
        ├── flac/                    # Downloaded audio files
        └── custom_spectrograms/     # Generated spectrograms
            ├── audio1.mat           # MATLAB data files
            ├── audio1.png           # PNG visualizations
            ├── audio2.mat
            └── audio2.png
```

### ⚙️ Configuration

Parameters can be configured in `config/dataset_config.yaml`:

```yaml
custom_spectrograms:
  window_duration: 1.0     # Window duration in seconds
  overlap: 0.5             # Overlap ratio (0-1)
  frequency_limits:
    min: 10                # Minimum frequency (Hz)
    max: 10000             # Maximum frequency (Hz)
  colormap: "turbo"        # Matplotlib colormap
  color_limits:
    min: -60               # Color scale minimum (dB)
    max: 0                 # Color scale maximum (dB)
  log_frequency: true      # Use log frequency scale
```

### 🔬 Use Cases

**Different Analysis Requirements:**
- **High time resolution**: `--win-dur 0.5 --overlap 0.75` for transient events
- **High frequency resolution**: `--win-dur 4.0 --overlap 0.9` for tonal analysis  
- **Low-frequency focus**: `--freq-min 1 --freq-max 1000` for whale calls
- **Wideband analysis**: `--freq-min 1 --freq-max 50000` for full spectrum

**Custom Duration Spectrograms:**
Unlike ONC's fixed 5-minute spectrograms, you can create any duration by adjusting window parameters to analyze longer or shorter audio segments.

### 💻 Programmatic Usage

```python
from src.audio import SpectrogramGenerator

# Create generator with custom parameters
generator = SpectrogramGenerator(
    win_dur=2.0,           # 2 second windows
    overlap=0.75,          # 75% overlap
    freq_lims=(5, 20000),  # 5 Hz to 20 kHz
    colormap='viridis'
)

# Process a directory
results = generator.process_directory(
    input_dir="data/DEVICE/flac/",
    save_dir="data/DEVICE/custom_spectrograms/",
    save_mat=True,
    save_plot=True
)
```

See `examples/generate_custom_spectrograms_example.py` for complete examples.





## 🔧 Advanced Options

```bash
# Download with custom settings
python scripts/download_hydrophone_data.py --mode sampling --device ICLISTENHF6020 --spectrograms-per-batch 12 --threshold 200 --check-deployments
```



```bash
# 1. Download with custom batch size
python scripts/download_hydrophone_data.py --mode sampling --spectrograms-per-batch 12 --check-deployments
```

## 🛠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| Invalid ONC Token | Check `.env` file |
| No Deployment Coverage | Use `--check-deployments` |
| No .mat files found | Verify folder structure |
| Labels not loading | Check JSON syntax |
| Memory errors | Reduce `--batch-size` |
| FLAC download fails | Check network connection and storage space |
| Large FLAC files | Monitor disk space, start with small downloads |

**💡 Pro Tip**: Always use `--check-deployments` to ensure active deployment periods!