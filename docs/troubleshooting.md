# Troubleshooting

Start with the symptom below. When reporting a problem, include the package
version, operating system, device code, requested UTC range, and full traceback
— but never include your ONC token or `.env` file.

## `Please set your ONC_TOKEN`

- Confirm `.env` is in the directory where Python starts.
- Confirm the variable is named exactly `ONC_TOKEN`.
- Do not add spaces around the name.
- Restart the notebook kernel or terminal after changing environment variables.

Verify without printing the token:

```python
from onc_hydrophone_data.onc.common import load_config

_, data_dir = load_config()
print(data_dir)
```

## The request returns no audio files

1. Check the device code for spelling and case.
2. Confirm the dates fall inside a deployment.
3. Plot archive availability for that range.
4. Use timezone-aware UTC datetimes.
5. Try a ten-minute range known to be green in the availability plot.

See **[Find a Hydrophone](inventory.md)** for the inventory and availability
workflow.

## I cannot find the downloaded files

Print the active paths immediately after a download:

```python
print("Audio:", dl.audio_path)
print("ONC spectrograms:", dl.spectrogram_path)
```

Range downloads are grouped under `DATA_DIR/DEVICE_CODE/METHOD_DATES/`.

## A server spectrogram request takes a long time

ONC generates plot-resolution and full-resolution MAT products on demand. Start
with one to six five-minute windows, or use the pre-generated one-minute product
for long ranges. See **[Choose ONC Server Spectrograms](onc_spectrogram_options.md)**.

## Local spectrogram generation is slow or uses too much memory

- Start with `max_workers=1` or `2`.
- Set `crop_freq_lims=True` and use a focused frequency range.
- Test parameters on one file before processing a directory.
- Avoid retaining full arrays for batch jobs (the directory workflow already
  defaults to releasing them).
- Use `save_mat=False` when only PNG figures are needed.

## Torch or torchaudio fails

Use the SciPy backend to separate backend installation from data problems:

```python
generator = SpectrogramGenerator(backend="scipy")
```

`backend="auto"` falls back to SciPy when the optimized backend cannot handle
the requested window or device.

## The PNG is empty, too dark, or too bright

- Confirm `freq_lims` overlaps frequencies supported by the audio sample rate.
- Try `clim=(-80, 0)` for more low-level detail or `(-40, 0)` for stronger
  contrast.
- Set `log_freq=False` while learning the axes.
- Check that the source audio is non-empty and can be opened by a media player.

## Local and ONC spectrogram values do not match

Local outputs are relative, uncalibrated power by default. ONC server products
may include hydrophone calibration, absolute units, re-binning, and different
FFT settings. They are not expected to be numerically interchangeable without
matching the full processing chain.
