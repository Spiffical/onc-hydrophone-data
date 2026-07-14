# 1. Install and Configure

You need Python, an ONC account, and an ONC API token before the package can
query deployments or download audio.

## Before you begin

- Python 3.9 or newer (Python 3.12 is recommended on macOS)
- An [Oceans 3.0 account](https://data.oceannetworks.ca/)
- Enough disk space for audio: high-sample-rate five-minute FLAC files can be
  much larger than spectrogram images

## Install the package

For normal use:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install onc-hydrophone-data
```

For development from a clone:

```bash
git clone https://github.com/Spiffical/onc-hydrophone-data.git
cd onc-hydrophone-data
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Save your ONC token

1. Sign in to [your ONC profile](https://data.oceannetworks.ca/Profile).
2. Copy your API token.
3. Create a file named `.env` in the directory where you will run Python:

```dotenv
ONC_TOKEN=<your-token>
DATA_DIR=./data
```

!!! danger "Keep the token private"
    Never commit `.env`, paste the token into a notebook that will be shared,
    or include it in screenshots and issue reports. This repository ignores
    `.env` files, but you should still check before committing.

`DATA_DIR` is optional. When omitted, downloads go into a `data/` directory
under the current working directory.

## Verify the configuration

```python
from onc_hydrophone_data.onc.common import load_config

onc_token, data_dir = load_config()
print(f"Data will be saved under: {data_dir}")
```

If that prints a path without raising an error, configuration is ready. The
token itself is intentionally not printed.

## Next step

Continue to **[2. Find a Hydrophone](inventory.md)** to choose a device code and
valid dates before downloading anything.
