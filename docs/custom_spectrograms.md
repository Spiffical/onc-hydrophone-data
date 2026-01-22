# Custom Spectrograms

Use JSON request files to download audio clips and generate **local** spectrograms with your own parameters.

```python
results = dl.create_custom_spectrograms_from_json(
    "/path/to/custom_requests.json",
    save_mat=True,
    save_png=False,
)
```

You can set `generator_defaults` and per‑request `generator_options` (e.g., `freq_lims`, `win_dur`, `overlap`) in the JSON.

## Example JSON
```json
{
  "defaults": {
    "deviceCode": "ICLISTENHF6324",
    "pad_seconds": 10
  },
  "generator_defaults": {
    "win_dur": 0.5,
    "overlap": 0.5
  },
  "requests": [
    {
      "timestamp": "2024-04-01T12:30:00Z",
      "generator_options": {
        "freq_lims": [10, 1000],
        "log_freq": true
      }
    }
  ]
}
```

!!! tip
    When `freq_lims` is provided, saved outputs are cropped automatically. The
    downloader still pulls extra audio context to avoid edge artifacts.
