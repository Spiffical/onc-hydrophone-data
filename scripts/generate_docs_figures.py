#!/usr/bin/env python3
"""Generate figures used by the MkDocs site from real ONC records and audio."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import shutil
import tempfile
from urllib.request import Request, urlopen

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from PIL import Image

from onc_hydrophone_data.audio import SpectrogramGenerator
from onc_hydrophone_data.data.deployment_checker import HydrophoneDeploymentChecker
from onc_hydrophone_data.onc.common import load_config
from onc_hydrophone_data.utils import (
    plot_availability_calendar,
    plot_deployment_availability_timeline,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "docs" / "assets" / "figures"
ONC_AUDIO_URL = (
    "https://ibase.oceannetworks.ca/media.php?i=9860&t=4&p=8"
    "&dg=7d61cb4a58b0b01cefa32614d1e047b94914327f"
)
ONC_INVENTORY_SOURCE = (
    "https://wiki.oceannetworks.ca/spaces/O2KB/pages/72548584/"
    "ONC+Hydrophone+Location+Codes+Data+Types"
)
ONC_INVENTORY_SNAPSHOT = datetime(2026, 7, 14)

CURRENT_ENDEAVOUR_ARRAY = [
    ("ICLISTENHF6324", "KEMFH.H1 (A)", "2023-09-08 22:56", "2,195 m"),
    ("ICLISTENHF6327", "KEMFH.H2 (B)", "2023-09-08 22:56", "2,195 m"),
    ("ICLISTENHF6328", "KEMFH.H3 (C)", "2023-09-08 22:56", "2,195 m"),
    ("ICLISTENHF6329", "KEMFH.H4 (D)", "2023-09-08 22:56", "2,195 m"),
]

ENDEAVOUR_DEPLOYMENT_HISTORY = [
    ("ICLISTENHF1561", "KEMF", "2018-07-25T18:03:48", "2020-10-03T02:53:21", False),
    ("ICLISTENHF1950", "KEMF", "2020-06-11T16:25:38", "2023-09-08T23:57:03", False),
    ("ICLISTENHF6324", "KEMFH.H1", "2023-09-08T22:56:13", None, True),
    ("ICLISTENHF6327", "KEMFH.H2", "2023-09-08T22:56:13", None, True),
    ("ICLISTENHF6328", "KEMFH.H3", "2023-09-08T22:56:13", None, True),
    ("ICLISTENHF6329", "KEMFH.H4", "2023-09-08T22:56:13", None, True),
]

AVAILABILITY_DEVICE = "ICLISTENHF6324"
AVAILABILITY_START = "2024-04-01"
AVAILABILITY_END = "2024-07-01"


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


def _download_onc_audio(destination: Path) -> None:
    """Download ONC's public MP3 preview without requiring an ONC token."""
    request = Request(
        ONC_AUDIO_URL,
        headers={"User-Agent": "onc-hydrophone-data documentation figure generator"},
    )
    with urlopen(request, timeout=60) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output)


def generate_deployment_figures() -> None:
    fig, ax = plt.subplots(figsize=(12, 3.8))
    ax.axis("off")
    table = ax.table(
        cellText=CURRENT_ENDEAVOUR_ARRAY,
        colLabels=["Device code", "Array position", "Deployment began (UTC)", "Depth"],
        colWidths=[0.24, 0.24, 0.34, 0.18],
        cellLoc="left",
        colLoc="left",
        bbox=[0, 0.02, 1, 0.78],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#d4dde6")
        cell.set_linewidth(0.8)
        if row == 0:
            cell.set_facecolor("#12324a")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("#f3f7fa" if row % 2 else "white")
            if column == 0:
                cell.get_text().set_weight("bold")
                cell.get_text().set_color("#075985")
    fig.text(
        0.02,
        0.94,
        "Current Main Endeavour Field hydrophone array",
        fontsize=17,
        fontweight="bold",
        color="#102a43",
    )
    fig.text(
        0.02,
        0.865,
        "Official ONC deployment records • positions A–D • 47.949322° N, 129.098209° W",
        fontsize=10.5,
        color="#486581",
    )
    _save_webp(fig, "deployment_inventory", bbox_inches=None)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5.8))
    historical_color = "#7b8794"
    active_color = "#078080"
    for index, (device, location, start_value, end_value, active) in enumerate(
        ENDEAVOUR_DEPLOYMENT_HISTORY
    ):
        start = datetime.fromisoformat(start_value)
        end = (
            ONC_INVENTORY_SNAPSHOT
            if end_value is None
            else datetime.fromisoformat(end_value)
        )
        start_number = mdates.date2num(start)
        width = mdates.date2num(end) - start_number
        ax.barh(
            index,
            width,
            left=start_number,
            height=0.58,
            color=active_color if active else historical_color,
            edgecolor="white",
            linewidth=0.8,
        )

    ax.set_yticks(range(len(ENDEAVOUR_DEPLOYMENT_HISTORY)))
    ax.set_yticklabels(
        [
            f"{device}\n{location}"
            for device, location, *_ in ENDEAVOUR_DEPLOYMENT_HISTORY
        ],
        fontsize=9.5,
    )
    ax.invert_yaxis()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlim(datetime(2018, 1, 1), datetime(2027, 1, 1))
    ax.grid(axis="x", color="#d9e2ec", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=8)
    ax.tick_params(axis="x", colors="#486581")
    ax.set_title(
        "Real hydrophone deployment history at Main Endeavour Field\n"
        "Official ONC records • ongoing deployments shown through 14 July 2026",
        loc="left",
        fontsize=15,
        fontweight="bold",
        color="#102a43",
        pad=16,
    )
    fig.legend(
        handles=[
            Patch(facecolor=historical_color, label="Completed deployment"),
            Patch(facecolor=active_color, label="Ongoing at ONC snapshot"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=2,
        frameon=False,
    )
    fig.tight_layout(rect=(0.04, 0.12, 0.98, 0.98))
    _save_webp(fig, "deployment_timeline", bbox_inches=None)
    plt.close(fig)


def generate_availability_figures() -> None:
    """Query ONC and render the exact availability example used in the docs."""
    onc_token, _ = load_config()
    checker = HydrophoneDeploymentChecker(onc_token)
    availability = checker.get_device_availability(
        AVAILABILITY_DEVICE,
        start_date=AVAILABILITY_START,
        end_date=AVAILABILITY_END,
        timezone_str="UTC",
        bin_size="day",
        quiet=True,
        max_workers=4,
    )

    timeline = plot_deployment_availability_timeline(availability, show=False)
    if timeline:
        fig, _ = timeline
        _save_webp(fig, "availability_timeline")
        plt.close(fig)

    calendar = plot_availability_calendar(availability, show=False)
    if calendar:
        fig, _ = calendar
        _save_webp(fig, "availability_calendar")
        plt.close(fig)

    days_with_data = sum(
        (entry.get("coverage") or 0) > 0 for entry in availability["bins"]
    )
    print(
        "Generated authenticated availability figures for "
        f"{AVAILABILITY_DEVICE}: {days_with_data}/{len(availability['bins'])} "
        "daily bins contain archived data"
    )


def generate_spectrogram_figures(audio_path: Path) -> None:
    generator = SpectrogramGenerator(
        win_dur=0.128,
        overlap=0.75,
        freq_lims=(20, 2_000),
        clim=(-80, -25),
        log_freq=False,
        backend="scipy",
        quiet=True,
    )
    audio, sample_rate, _ = generator.load_audio(audio_path)
    frequencies, times, _, power_db = generator.compute_spectrogram(
        audio,
        sample_rate,
    )
    temporary_spectrogram = OUTPUT_DIR / ".example_local_spectrogram.png"
    fig = generator.plot_spectrogram(
        frequencies,
        times,
        power_db,
        title="Humpback whale calls — Folger Passage — 2012-08-01 12:24 UTC",
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
    for window_seconds in (0.032, 0.5):
        comparison_generator = SpectrogramGenerator(
            win_dur=window_seconds,
            overlap=0.75,
            freq_lims=(20, 2_000),
            clim=(-80, -25),
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
            vmin=-80,
            vmax=-25,
        )
        axis.set_ylabel("Frequency (Hz)")
        axis.set_title(f"Window duration: {window_seconds:g} s")
        axis.set_ylim(20, 2_000)
    axes[-1].set_xlabel("Time (s)")
    fig.colorbar(
        image,
        ax=axes,
        label="Power relative to maximum (dB)",
        pad=0.02,
        shrink=0.86,
    )
    fig.suptitle(
        "Time–frequency trade-off — ONC humpback calls, Folger Passage"
    )
    _save_webp(fig, "spectrogram_window_comparison", bbox_inches=None)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audio-file",
        type=Path,
        help=(
            "Use this ONC audio file for spectrogram figures instead of "
            "downloading the public ONC preview"
        ),
    )
    parser.add_argument(
        "--with-live-availability",
        action="store_true",
        help=(
            "Query ONC using ONC_TOKEN from .env and regenerate the real "
            "availability timeline and calendar"
        ),
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for obsolete_png in OUTPUT_DIR.glob("*.png"):
        obsolete_png.unlink()
    generate_deployment_figures()
    if args.with_live_availability:
        generate_availability_figures()
    if args.audio_file:
        generate_spectrogram_figures(args.audio_file)
    else:
        with tempfile.TemporaryDirectory(prefix="onc-docs-audio-") as temp_dir:
            audio_path = Path(temp_dir) / "folger_humpback.mp3"
            _download_onc_audio(audio_path)
            generate_spectrogram_figures(audio_path)
    print(f"Generated documentation figures in {OUTPUT_DIR}")
    print(f"Deployment source: {ONC_INVENTORY_SOURCE}")


if __name__ == "__main__":
    main()
