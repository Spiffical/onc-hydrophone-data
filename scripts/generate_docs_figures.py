#!/usr/bin/env python3
"""Generate deterministic, illustrative figures used by the MkDocs site.

The figures intentionally use synthetic data. They demonstrate the package's
plotting output without requiring an ONC token or presenting simulated values
as real hydrophone observations.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.signal import chirp

from onc_hydrophone_data.audio import SpectrogramGenerator
from onc_hydrophone_data.data.deployment_checker import DeploymentInfo
from onc_hydrophone_data.utils import (
    plot_availability_calendar,
    plot_deployment_availability_timeline,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "docs" / "assets" / "figures"


def _save_webp(
    fig: plt.Figure,
    filename: str,
    *,
    bbox_inches: str | None = "tight",
) -> None:
    """Save a compact WebP while keeping Matplotlib rendering deterministic."""
    temporary_png = OUTPUT_DIR / f".{filename}.png"
    fig.savefig(temporary_png, dpi=170, bbox_inches=bbox_inches)
    with Image.open(temporary_png) as image:
        image.thumbnail((1_800, 1_200), Image.Resampling.LANCZOS)
        image.save(
            OUTPUT_DIR / f"{filename}.webp",
            format="WEBP",
            quality=84,
            method=6,
        )
    temporary_png.unlink()


def _example_availability() -> dict:
    """Build illustrative daily availability with two deployment windows."""
    utc = timezone.utc
    start = datetime(2024, 4, 1, tzinfo=utc)
    end = datetime(2024, 7, 1, tzinfo=utc)
    deployments = [
        DeploymentInfo(
            device_code="EXAMPLE-HYDROPHONE",
            device_id="example-1",
            location_code="SITE-A",
            location_name="Illustrative Site A",
            begin_date=datetime(2024, 4, 8, tzinfo=utc),
            end_date=datetime(2024, 5, 18, tzinfo=utc),
            latitude=48.0,
            longitude=-126.0,
            depth=850.0,
            citation=None,
        ),
        DeploymentInfo(
            device_code="EXAMPLE-HYDROPHONE",
            device_id="example-1",
            location_code="SITE-B",
            location_name="Illustrative Site B",
            begin_date=datetime(2024, 5, 24, tzinfo=utc),
            end_date=datetime(2024, 6, 26, tzinfo=utc),
            latitude=48.5,
            longitude=-127.0,
            depth=1250.0,
            citation=None,
        ),
    ]

    gap_days = {
        datetime(2024, 4, 20, tzinfo=utc).date(),
        datetime(2024, 4, 21, tzinfo=utc).date(),
        datetime(2024, 5, 3, tzinfo=utc).date(),
        datetime(2024, 6, 8, tzinfo=utc).date(),
        datetime(2024, 6, 9, tzinfo=utc).date(),
        datetime(2024, 6, 10, tzinfo=utc).date(),
    }
    partial_coverage = {
        datetime(2024, 4, 15, tzinfo=utc).date(): 0.55,
        datetime(2024, 5, 12, tzinfo=utc).date(): 0.72,
        datetime(2024, 6, 18, tzinfo=utc).date(): 0.35,
    }

    bins = []
    cursor = start
    while cursor < end:
        bin_end = cursor + timedelta(days=1)
        deployment_index = None
        for index, deployment in enumerate(deployments):
            if deployment.begin_date < bin_end and deployment.end_date > cursor:
                deployment_index = index
                break

        if deployment_index is None:
            status = "not_deployed"
            coverage = None
        elif cursor.date() in gap_days:
            status = "no_data"
            coverage = 0.0
        else:
            coverage = partial_coverage.get(cursor.date(), 1.0)
            status = "data"

        bins.append(
            {
                "start": cursor,
                "end": bin_end,
                "coverage": coverage,
                "status": status,
                "deployment_index": deployment_index,
            }
        )
        cursor = bin_end

    summaries = []
    for index, deployment in enumerate(deployments):
        deployment_bins = [
            item for item in bins if item["deployment_index"] == index
        ]
        bins_with_data = sum((item["coverage"] or 0) > 0 for item in deployment_bins)
        summaries.append(
            {
                "deployment_index": index,
                "device_code": deployment.device_code,
                "location_name": deployment.location_name,
                "location_code": deployment.location_code,
                "begin_date": deployment.begin_date,
                "end_date": deployment.end_date,
                "bins_total": len(deployment_bins),
                "bins_with_data": bins_with_data,
                "coverage_ratio": bins_with_data / len(deployment_bins),
            }
        )

    return {
        "device_code": "EXAMPLE-HYDROPHONE",
        "timezone": "UTC",
        "bin_size": "day",
        "start": start,
        "end": end,
        "deployments": deployments,
        "bins": bins,
        "deployment_summary": summaries,
    }


def _synthetic_hydrophone_audio() -> tuple[np.ndarray, int]:
    """Create a deterministic signal with tones, sweeps, clicks, and noise."""
    sample_rate = 16_000
    duration = 24.0
    times = np.arange(int(sample_rate * duration), dtype=np.float64) / sample_rate
    rng = np.random.default_rng(20240714)

    amplitude_modulation = 0.55 + 0.25 * np.sin(2 * np.pi * 0.18 * times)
    signal = amplitude_modulation * (
        0.20 * np.sin(2 * np.pi * 180 * times)
        + 0.10 * np.sin(2 * np.pi * 360 * times)
        + 0.06 * np.sin(2 * np.pi * 540 * times)
    )

    for sweep_start in (5.5, 10.8):
        mask = (times >= sweep_start) & (times < sweep_start + 2.8)
        sweep_times = times[mask] - sweep_start
        envelope = np.sin(np.pi * sweep_times / 2.8) ** 2
        signal[mask] += 0.32 * envelope * chirp(
            sweep_times,
            f0=650,
            f1=2_600,
            t1=2.8,
            method="quadratic",
        )

    for click_time in (15.0, 16.2, 17.4, 18.6):
        relative = times - click_time
        envelope = np.exp(-0.5 * (relative / 0.018) ** 2)
        signal += 0.28 * envelope * np.sin(2 * np.pi * 3_500 * times)

    signal += 0.035 * rng.standard_normal(times.shape)
    signal /= max(1.0, np.max(np.abs(signal)))
    return signal.astype(np.float32), sample_rate


def generate_deployment_figures() -> None:
    availability = _example_availability()

    timeline = plot_deployment_availability_timeline(
        availability,
        title="Illustrative deployment and archive availability",
        show=False,
    )
    if timeline:
        fig, _ = timeline
        _save_webp(fig, "deployment_timeline")
        plt.close(fig)

    calendar = plot_availability_calendar(
        availability,
        title="Illustrative daily audio availability",
        show=False,
    )
    if calendar:
        fig, _ = calendar
        _save_webp(fig, "availability_calendar")
        plt.close(fig)


def generate_spectrogram_figures() -> None:
    audio, sample_rate = _synthetic_hydrophone_audio()
    generator = SpectrogramGenerator(
        win_dur=0.128,
        overlap=0.75,
        freq_lims=(50, 5_000),
        clim=(-50, 0),
        log_freq=False,
        backend="scipy",
        quiet=True,
    )
    frequencies, times, _, power_db = generator.compute_spectrogram(
        audio,
        sample_rate,
    )
    temporary_spectrogram = OUTPUT_DIR / ".example_local_spectrogram.png"
    fig = generator.plot_spectrogram(
        frequencies,
        times,
        power_db,
        title="Illustrative local spectrogram (synthetic audio)",
        save_path=temporary_spectrogram,
    )
    plt.close(fig)
    with Image.open(temporary_spectrogram) as image:
        image.thumbnail((1_800, 1_200), Image.Resampling.LANCZOS)
        image.save(
            OUTPUT_DIR / "example_local_spectrogram.webp",
            format="WEBP",
            quality=84,
            method=6,
        )
    temporary_spectrogram.unlink()

    comparisons = []
    for window_seconds in (0.064, 0.5):
        comparison_generator = SpectrogramGenerator(
            win_dur=window_seconds,
            overlap=0.75,
            freq_lims=(50, 5_000),
            clim=(-50, 0),
            log_freq=False,
            backend="scipy",
            quiet=True,
        )
        freq, time_axis, _, db = comparison_generator.compute_spectrogram(
            audio,
            sample_rate,
        )
        comparisons.append((window_seconds, freq, time_axis, db))

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(11, 7),
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    for axis, (window_seconds, freq, time_axis, db) in zip(axes, comparisons):
        image = axis.pcolormesh(
            time_axis,
            freq,
            db,
            shading="auto",
            cmap="turbo",
            vmin=-50,
            vmax=0,
        )
        axis.set_ylabel("Frequency (Hz)")
        axis.set_title(f"Window duration: {window_seconds:g} s")
        axis.set_ylim(50, 5_000)
    axes[-1].set_xlabel("Time (s)")
    fig.colorbar(
        image,
        ax=axes,
        label="Power relative to maximum (dB)",
        pad=0.02,
        shrink=0.86,
    )
    fig.suptitle("Time–frequency trade-off (synthetic audio example)")
    _save_webp(fig, "spectrogram_window_comparison", bbox_inches=None)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for obsolete_png in OUTPUT_DIR.glob("*.png"):
        obsolete_png.unlink()
    generate_deployment_figures()
    generate_spectrogram_figures()
    print(f"Generated documentation figures in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
