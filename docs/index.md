# ONC Hydrophone Data

This site is the user‑focused documentation for **onc-hydrophone-data**: a toolkit to download ONC hydrophone spectrograms/audio and generate custom spectrograms locally.

## What you can do
- Pull ONC spectrograms (MAT/PNG) for time ranges or sampled windows.
- Download FLAC/WAV audio for matching ranges.
- Run event‑based batches via JSON/CSV request files.
- Generate local custom spectrograms from audio with your own FFT parameters.

## Fast download model (ONC‑side parallelism)
This package submits **multiple ONC data requests up front**, then polls and downloads results as they complete. ONC processes those requests in parallel on their servers, which is much faster than submitting one request at a time and waiting for each to finish.

![Parallel ONC request pipeline](assets/parallel_pipeline.svg){: width="100%" }

## Recommended path
1. **Quickstart** for a minimal “download + plot” flow.
2. **Deployments & Inventory** to pick valid dates and devices.
3. **Downloads** for the most common workflows.
4. **Custom Spectrograms** for local generation.

> If you prefer notebooks, see `notebooks/ONC_Data_Download_Tutorial.ipynb` in the repo.
