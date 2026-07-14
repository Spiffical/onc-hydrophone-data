# ONC Spectrogram Products and Server Options

Ocean Networks Canada (ONC) provides several forms of hydrophone spectral data.
The best choice depends on whether you need a quick visual scan, compact data
for long-term analysis, the values shown in ONC's plots, or the highest available
time and frequency resolution.

This page summarizes the official [ONC Hydrophone Spectral Data product
(HSD)](https://wiki.oceannetworks.ca/spaces/DP/pages/42174569/45) and shows how
its server-side options map to this package. ONC reports the options available
for a particular device and format through its discovery API, so a device may
offer only a subset of the values listed here.

!!! tip "Most beginners should start with audio"
    If your goal is to create spectrograms with your own settings, follow
    **[Download Audio and Make a Spectrogram](quickstart.md)** instead. Use this
    page when you specifically need ONC's server-generated or calibrated
    spectral products.

## Quick chooser

| Goal | ONC format and option | Retrieval and size | Notes |
| --- | --- | --- | --- |
| Browse five-minute periods visually | PNG with default plot options | Usually fastest because default plots are pre-generated | Approximately 1200 × 900 pixels; one plot normally represents one source audio file. |
| Scan a long period visually | PDF, optionally `Daily` or `Weekly` concatenation | Generated on demand; slower than archived PNG | Higher-resolution, multi-page output intended for visual review. |
| Analyze long periods or make probability-density summaries | MAT with `dpo_spectralDataDownsample=1` | Fastest and smallest MAT choice | Pre-generated one-minute ensemble averages; this is ONC's MAT default. |
| Analyze the values represented in the PNG/PDF plot | MAT with `dpo_spectralDataDownsample=2` | Generated on demand | Downsampled in time and frequency to approximately the plot's useful pixel resolution. |
| Preserve the highest server-produced resolution | MAT with `dpo_spectralDataDownsample=0` | Slowest and largest | Usually 0.5 seconds or better in time and roughly 1 Hz frequency bins, depending on the hydrophone and calibration. |
| Work with low-bandwidth or diversion-period spectral files | FFT source or `.fft` files, where available | Device-dependent | Lower resolution than audio-sourced spectra but sometimes available when audio is not; may span a wider frequency range. |

!!! important "Package default versus ONC default"
    ONC's default MAT option is the pre-generated one-minute product
    (`dpo_spectralDataDownsample=1`). `HydrophoneDownloader` currently defaults
    to plot resolution (`2`) because the package was designed around
    spectrogram-level analysis. Choose `1` explicitly when speed and compact
    long-term data are more important than plot-level resolution.

## MAT spectral resolution

The [ONC spectral-data downsampling
documentation](https://wiki.oceannetworks.ca/spaces/DP/pages/114032814/Spectral+Data+Downsampling)
defines three MAT resolutions:

| Value | ONC name | What ONC returns | Concatenation |
| --- | --- | --- | --- |
| `1` | One-minute | Pre-generated one-minute ensemble averages. ONC averages in linear/pressure units and converts back to dB. Frequency bins may also be reduced. | `Concatenate`, `Daily`, or `None` |
| `2` | Spectrogram resolution | The spectral values used at approximately the useful resolution of ONC's PNG/PDF plot. ONC notes these files are commonly around one tenth the size of full resolution. | One MAT file per source audio file (`None`) |
| `0` | Full resolution | Resolution determined by the hydrophone sample rate and calibration. | One MAT file per source audio file (`None`) |

Plot-resolution and full-resolution MAT files are not normally stored in the
archive. ONC generates them from the source data when requested. ONC estimates
about 25 seconds of server processing for each five-minute source file, although
actual queue and processing time varies. Keep non-default requests small; ONC's
specific downsampling guide recommends no more than one month per request.

### Python examples

Set a default for every request made by a downloader:

```python
from onc_hydrophone_data.data import HydrophoneDownloader

# Use ONC's compact, pre-generated one-minute MAT product by default.
dl = HydrophoneDownloader(
    ONC_TOKEN,
    DATA_DIR,
    spectral_downsample=1,
)
```

Or override the ONC data-product options for one workflow:

```python
# Fast, compact one-minute MAT data.
one_minute = dl.download_spectrograms_for_range(
    device_code=DEVICE,
    start_dt=start,
    end_dt=end,
    spectrograms_per_batch=12,
    data_product_options={
        "dpo_spectralDataDownsample": 1,
        "dpo_spectrogramConcatenation": "Concatenate",
    },
)

# Data at approximately ONC's plotted spectrogram resolution.
plot_resolution = dl.download_spectrograms_for_range(
    device_code=DEVICE,
    start_dt=start,
    end_dt=end,
    spectrograms_per_batch=6,
    data_product_options={
        "dpo_spectralDataDownsample": 2,
        "dpo_spectrogramConcatenation": "None",
    },
)

# Highest server-produced resolution. Start with a small time range.
full_resolution = dl.download_spectrograms_for_range(
    device_code=DEVICE,
    start_dt=start,
    end_dt=end,
    spectrograms_per_batch=1,
    data_product_options={
        "dpo_spectralDataDownsample": 0,
        "dpo_spectrogramConcatenation": "None",
    },
)
```

## Concatenation and collation

The official [ONC spectrogram concatenation
guide](https://wiki.oceannetworks.ca/spaces/DP/pages/48695940/Spectrogram+Concatenation)
describes how source files can be grouped:

| Value | Formats | Behaviour |
| --- | --- | --- |
| `None` | MAT, PNG, PDF | One output for each compatible source audio/FFT file. This package uses `None` by default so five-minute windows stay separate. |
| `Concatenate` | One-minute MAT | Combines compatible data until ONC's file-size or frequency-compatibility limit is reached. This is ONC's default for MAT. |
| `Daily` | One-minute MAT, PNG, PDF | Produces daily output. Spectral data is assembled into one-minute non-overlapping averages. |
| `Weekly` | PNG, PDF | Produces weekly plots; the official HSD page describes five-minute averages for weekly plot data. |
| `Adjacent` | PNG, PDF | For a search of five minutes or less, joins adjacent source audio files and calculates one plot covering the requested interval. |

Do not combine `Concatenate` or `Daily` with MAT downsampling values `0` or `2`.
Those resolutions are generated per source file and ONC's user interface hides
the concatenation control when either is selected.

## Source, channel, acquisition, and diversion filters

These options are device-dependent. Values below are the current values exposed
by ONC's discovery API for HSD products.

| Filter | Values | Purpose |
| --- | --- | --- |
| `dpo_spectrogramSource` | `MIX`, `WAVFLAC`, `FFT` | PNG/PDF only. Prefer audio and fill gaps with FFT (`MIX`), require audio (`WAVFLAC`), or require FFT source data (`FFT`). |
| `dpo_hydrophoneChannel` | `H1`, `H2`, `H3`, `All` | Select a channel on multi-channel hydrophone arrays. `H1` is ONC's default. |
| `dpo_hydrophoneAcquisitionMode` | `LF`, `HF`, `All` | Select low- or high-sample-rate periods on duty-cycled hydrophones. |
| `dpo_hydrophoneDataDiversionMode` | `OD`, `LPF`, `HPF`, `All` | Select original, low-pass-filtered, high-pass-filtered, or every available diversion mode. `OD` is ONC's default. |

For example:

```python
result = dl.download_spectrograms_for_range(
    device_code=DEVICE,
    start_dt=start,
    end_dt=end,
    spectrograms_per_batch=6,
    data_product_options={
        "dpo_spectralDataDownsample": 2,
        "dpo_hydrophoneChannel": "H1",
        "dpo_hydrophoneAcquisitionMode": "All",
        "dpo_hydrophoneDataDiversionMode": "OD",
    },
)
```

!!! note
    The ONC wiki uses `WAV` in some older spectrogram-source examples. The
    current discovery API reports `WAVFLAC` for the audio-only value. Query the
    discovery API before building long-lived automation.

## PNG/PDF server plot options

ONC can regenerate PNG/PDF plots with custom rendering options. Changing any
default plot option normally forces on-demand generation instead of returning a
pre-generated plot.

| Filter | Current values |
| --- | --- |
| `dpo_spectrogramColourPalette` | `0` through `5`; `0` is ONC's modified-rainbow default and `1`–`5` are alternate sequential or grayscale palettes. |
| `dpo_lowerColourLimit` | `-1000` for the device default, or a value from `-160` to `140`. |
| `dpo_upperColourLimit` | `-1000` for the device default, or a value from `-160` to `140`. |
| `dpo_spectrogramFrequencyUpperLimit` | `-1` for the device default, presets `1000` or `10000`, or an integer from `100` to `500000` Hz. |

The current high-level `download_spectrograms_*` workflows in this package
request MAT files. The filters above are documented for users working in the
ONC portal or making a lower-level PNG/PDF data-product request; archived PNG
downloads do not apply custom rendering filters.

## How ONC generates spectrograms

ONC describes its server process as a modified Welch calculation:

1. Read source WAV, FLAC, HYD, or (where needed) FFT spectral data.
2. Split audio into Hann-windowed segments with 50% overlap. The window length
   equals the FFT length.
3. Calculate a power spectrum for every segment. Frequency forms the rows and
   segment time forms the columns.
4. Apply hydrophone calibration when calibration data is available. FFT length
   and usable frequency range can depend on sample rate, calibration, and device
   attributes.
5. For PNG/PDF and reduced-resolution MAT products, re-bin in time and frequency
   using linear-domain boxcar averaging, then convert back to dB. This avoids
   averaging logarithmic dB values directly and avoids relying on image-renderer
   resampling.
6. Render a plot or save the `Meta` and `SpectData` structures in a MAT file.

ONC server products and locally generated products are therefore not
interchangeable by default. ONC products may include hydrophone calibration and
absolute sound-level information, while this package's local
`SpectrogramGenerator` currently stores an uncalibrated power array plus a
relative-to-maximum dB array.

!!! warning "Check ONC's current PSD-units notice"
    ONC's HSD wiki currently reports a known normalization issue in
    audio-derived spectral products: values labelled as power spectral density
    were not divided by frequency-bin width and therefore behave as power
    spectrum values. ONC gives a correction of
    `10 * log10(1 / frequency_bin_width_hz)` for existing products and says most
    products use 10 Hz bins. Treat the HSD wiki as the authoritative, changing
    source before interpreting absolute levels.

## Inspect the live options for a device

ONC options can change and are responsive to device, deployment, and format.
Use the discovery API before launching a large request:

```python
from onc.onc import ONC

onc = ONC(ONC_TOKEN, showInfo=False)
products = onc.getDataProducts({
    "deviceCode": DEVICE,
    "dataProductCode": "HSD",
    "extension": "mat",  # or png / pdf
})

for product in products:
    for option in product.get("dataProductOptions", []):
        print(
            option["option"],
            "default=", option.get("defaultValue"),
            "values=", option.get("allowableValues"),
            "range=", option.get("allowableRange"),
        )
```

See also ONC's [Hydrophone Channel
documentation](https://wiki.oceannetworks.ca/spaces/DP/pages/42175229/Hydrophone+Channel)
and [Oceans 3.0 OpenAPI documentation](https://data.oceannetworks.ca/OpenAPI).
