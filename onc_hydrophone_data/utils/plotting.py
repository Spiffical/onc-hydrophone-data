"""
Plotting helpers for ONC spectrogram MAT files and audio waveforms.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import json
from pathlib import Path
from typing import Iterable, Optional, Any

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import scipy.io

try:
    import soundfile as sf
except ImportError:
    sf = None

try:
    import librosa
except ImportError:
    librosa = None


def _matlab_datenum_to_datetime(datenums: Iterable[float]) -> list[datetime]:
    dates: list[datetime] = []
    for dn in np.atleast_1d(datenums):
        dates.append(
            datetime.fromordinal(int(dn))
            + timedelta(days=float(dn) % 1)
            - timedelta(days=366)
        )
    return dates


def find_first_file(directory: str | Path, patterns: Iterable[str]) -> Optional[Path]:
    dir_path = Path(directory)
    for pattern in patterns:
        matches = sorted(dir_path.glob(pattern))
        if matches:
            return matches[0]
    return None


def plot_first_spectrogram(
    downloader: Any,
    *,
    title: Optional[str] = None,
    patterns: Iterable[str] = ("*.mat",),
) -> Optional[Path]:
    """Find and plot the first spectrogram file in the downloader output path."""
    spectrogram_path = getattr(downloader, "spectrogram_path", None)
    if not spectrogram_path:
        print("No spectrogram path available to plot.")
        return None
    mat_path = find_first_file(spectrogram_path, patterns)
    if not mat_path:
        print("No spectrogram files found to plot.")
        return None
    plot_onc_mat_spectrogram(mat_path, title=title or mat_path.name)
    return mat_path


def plot_first_audio(
    downloader: Any,
    *,
    max_seconds: Optional[float] = 10.0,
    patterns: Iterable[str] = ("*.flac", "*.wav"),
) -> Optional[Path]:
    """Find and plot the first audio file in the downloader output path."""
    audio_path = getattr(downloader, "audio_path", None)
    if not audio_path:
        print("No audio path available to plot.")
        return None
    audio_file = find_first_file(audio_path, patterns)
    if not audio_file:
        print("No audio files found to plot.")
        return None
    plot_audio_waveform(audio_file, max_seconds=max_seconds)
    return audio_file


def plot_onc_mat_spectrogram(
    mat_path: str | Path,
    title: Optional[str] = None,
    cmap: str = "turbo",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    freq_lims: Optional[tuple[float, float]] = None,
    log_freq: Optional[bool] = None,
) -> None:
    """Plot a MAT spectrogram file, optionally limiting frequency range."""
    mat_path = Path(mat_path)
    mat = scipy.io.loadmat(mat_path)
    psd = None
    freq = None
    time_vals = None

    if "SpectData" in mat:
        entry = mat["SpectData"][0, 0]
        psd = entry["PSD"]
        freq = entry["frequency"]
        time_vals = entry["time"]
    else:
        psd = mat.get("P")
        if psd is None:
            psd = mat.get("PSD")
        if psd is None:
            psd = mat.get("spectrogram")
        freq = mat.get("F")
        if freq is None:
            freq = mat.get("frequency")
        time_vals = mat.get("T")
        if time_vals is None:
            time_vals = mat.get("time")

    if psd is None:
        print(f"No spectrogram data found in {mat_path}")
        return

    psd = np.asarray(psd)
    if freq is None:
        freq = np.arange(psd.shape[0])
    else:
        freq = np.asarray(freq).squeeze()
    if freq.size != psd.shape[0]:
        freq = np.arange(psd.shape[0])

    if time_vals is None:
        x_vals = np.arange(psd.shape[1])
        x_dates = False
        time_label = "Time (index)"
    else:
        time_vals = np.asarray(time_vals).squeeze()
        if time_vals.size != psd.shape[1]:
            time_vals = np.arange(psd.shape[1])
        if np.nanmax(time_vals) > 1e5:
            dt = _matlab_datenum_to_datetime(time_vals)
            x_vals = mdates.date2num(dt)
            x_dates = True
            time_label = "Time (UTC)"
        else:
            x_vals = time_vals - time_vals[0]
            x_dates = False
            time_label = "Time (s)"

    finite = psd[np.isfinite(psd)]
    if finite.size and (vmin is None or vmax is None):
        vmin, vmax = np.percentile(finite, [5, 95])

    fig, ax = plt.subplots(figsize=(10, 4))
    mesh = ax.pcolormesh(x_vals, freq, psd, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    if log_freq:
        ax.set_yscale('log')
    if freq_lims is not None:
        ax.set_ylim(freq_lims)
    if x_dates:
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        fig.autofmt_xdate()
    ax.set_xlabel(time_label)
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title or mat_path.name)
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("PSD (dB)")
    plt.show()


def plot_audio_waveform(audio_path: str | Path, max_seconds: float = 10.0) -> None:
    audio_path = Path(audio_path)
    audio_data, sr = _load_audio_data(audio_path)
    if audio_data is None or sr is None:
        return

    if max_seconds is not None:
        max_samples = int(max_seconds * sr)
        audio_data = audio_data[:max_samples]

    times = np.arange(len(audio_data)) / sr
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(times, audio_data, linewidth=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(audio_path.name)
    ax.grid(True, alpha=0.3)
    plt.show()


def plot_clip_pair(
    spectrogram_clip_path: str | Path,
    audio_clip_path: str | Path,
    title: Optional[str] = None,
    cmap: str = "turbo",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    spec_data = _load_spectrogram_clip_data(spectrogram_clip_path)
    if spec_data is None:
        return
    spec, freq, seconds_per_col, spec_meta_duration = spec_data

    audio_data, sr = _load_audio_data(audio_clip_path)
    if audio_data is None or sr is None:
        return

    audio_time = np.arange(len(audio_data)) / sr
    spec_duration = (
        spec_meta_duration
        if spec_meta_duration is not None
        else _clip_duration_from_seconds_per_col(spec.shape[1], seconds_per_col)
    )
    audio_meta_duration = _audio_clip_duration_from_meta(audio_clip_path)
    audio_duration = audio_meta_duration if audio_meta_duration is not None else _duration_from_audio(audio_time)
    if spec_duration and audio_duration and abs(spec_duration - audio_duration) > 0.5:
        print(
            "Note: spectrogram clips use fixed time bins, so their plotted duration can differ "
            f"slightly from the sample-accurate audio ({spec_duration:.2f}s vs {audio_duration:.2f}s)."
        )
    clip_duration = spec_meta_duration or audio_meta_duration
    if clip_duration is None:
        clip_duration = max(spec_duration or 0.0, audio_duration or 0.0)
    spec_time = _spectrogram_time_axis(spec.shape[1], seconds_per_col, clip_duration)

    fig, (ax_spec, ax_audio) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(10, 6),
        gridspec_kw={'height_ratios': [2, 1]},
    )
    mesh = ax_spec.pcolormesh(
        spec_time,
        freq,
        spec,
        shading="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax_spec.set_ylabel("Frequency (Hz)")
    ax_spec.set_title(title or "Clipped spectrogram + audio")
    cbar = fig.colorbar(mesh, ax=[ax_spec, ax_audio], pad=0.02, fraction=0.04)
    cbar.set_label("PSD (dB)")

    ax_audio.plot(audio_time, audio_data, linewidth=0.8)
    ax_audio.set_xlabel("Time (s)")
    ax_audio.set_ylabel("Amplitude")
    if clip_duration > 0:
        ax_audio.set_xlim(0, clip_duration)
    ax_audio.grid(True, alpha=0.3)
    plt.show()


def plot_spectrogram_clip(
    clip_path: str | Path,
    title: Optional[str] = None,
    cmap: str = "turbo",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    clip_path = Path(clip_path)
    spec_data = _load_spectrogram_clip_data(clip_path)
    if spec_data is None:
        return
    spec, freq, seconds_per_col, clip_duration = spec_data
    x_vals = _spectrogram_time_axis(spec.shape[1], seconds_per_col, clip_duration)
    time_label = "Time (s)" if seconds_per_col or clip_duration else "Time (index)"

    finite = spec[np.isfinite(spec)]
    if finite.size and (vmin is None or vmax is None):
        vmin, vmax = np.percentile(finite, [5, 95])

    fig, ax = plt.subplots(figsize=(10, 4))
    mesh = ax.pcolormesh(x_vals, freq, spec, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xlabel(time_label)
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title or clip_path.name)
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("PSD (dB)")
    plt.show()


def plot_deployment_availability_timeline(
    availability: dict,
    *,
    title: Optional[str] = None,
    data_color: str = "#2ca02c",
    missing_color: str = "#d62728",
    deployment_color: str = "#e6e6e6",
    show_coverage: bool = True,
    show: bool = True,
):
    """
    Plot a deployment-aware availability timeline.

    Args:
        availability: Output from HydrophoneDeploymentChecker.get_device_availability.
        title: Optional plot title.
        data_color: Color for bins with data.
        missing_color: Color for bins without data.
        deployment_color: Background color for deployment windows.
        show_coverage: Annotate per-deployment coverage percentage.
        show: If True, call plt.show().
    """
    bins = availability.get('bins') or []
    deployments = availability.get('deployments') or []
    if not bins or not deployments:
        print("No availability data to plot.")
        return None

    bins_by_dep = {}
    for b in bins:
        dep_idx = b.get('deployment_index')
        if dep_idx is None:
            continue
        bins_by_dep.setdefault(dep_idx, []).append(b)

    for dep_idx in bins_by_dep:
        bins_by_dep[dep_idx] = sorted(bins_by_dep[dep_idx], key=lambda x: x['start'])

    row_height = 0.6
    row_gap = 0.3
    fig_height = max(2.0, len(deployments) * (row_height + row_gap) + 1.0)
    fig, ax = plt.subplots(figsize=(12, fig_height))

    y_ticks = []
    y_labels = []
    range_start = availability.get('start')
    range_end = availability.get('end')
    summary_by_idx = {
        entry.get('deployment_index'): entry
        for entry in (availability.get('deployment_summary') or [])
    }

    for i, dep in enumerate(deployments):
        y = i * (row_height + row_gap)
        y_ticks.append(y + row_height / 2)
        y_labels.append(_deployment_label(dep))

        dep_start = getattr(dep, "begin_date", None)
        dep_end = getattr(dep, "end_date", None) or range_end
        if dep_start is None or dep_end is None:
            continue
        if range_start is not None:
            dep_start = max(dep_start, range_start)
        if range_end is not None:
            dep_end = min(dep_end, range_end)
        if dep_end <= dep_start:
            continue

        start_num = mdates.date2num(_strip_tz(dep_start))
        end_num = mdates.date2num(_strip_tz(dep_end))
        ax.broken_barh(
            [(start_num, end_num - start_num)],
            (y, row_height),
            facecolor=deployment_color,
            edgecolor='none',
        )

        for status, seg_start, seg_end in _segments_from_bins(bins_by_dep.get(i, [])):
            seg_start_num = mdates.date2num(_strip_tz(seg_start))
            seg_end_num = mdates.date2num(_strip_tz(seg_end))
            width = seg_end_num - seg_start_num
            if width <= 0:
                continue
            color = data_color if status == 'data' else missing_color
            ax.broken_barh(
                [(seg_start_num, width)],
                (y, row_height),
                facecolor=color,
                edgecolor='none',
            )

        if show_coverage and i in summary_by_idx:
            coverage = summary_by_idx[i].get('coverage_ratio', 0.0) * 100.0
            label_x = mdates.date2num(_strip_tz(range_end or dep_end)) + 2
            ax.text(label_x, y + row_height / 2, f"{coverage:.0f}%", va='center', fontsize=9)

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax.grid(axis='x', alpha=0.2)
    ax.set_ylim(-0.2, len(deployments) * (row_height + row_gap))

    if title is None:
        device_code = availability.get('device_code', 'device')
        title = f"Deployment availability for {device_code}"
    ax.set_title(title)

    from matplotlib.patches import Patch
    legend_items = [
        Patch(facecolor=data_color, label="Data available"),
        Patch(facecolor=missing_color, label="No data"),
        Patch(facecolor=deployment_color, label="Deployment window"),
    ]
    ax.legend(handles=legend_items, loc='upper right', frameon=False)

    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def plot_availability_calendar(
    availability: dict,
    *,
    title: Optional[str] = None,
    data_cmap: str = "Greens",
    missing_color: str = "#d62728",
    not_deployed_color: str = "#eeeeee",
    show: bool = True,
):
    """
    Plot a calendar-style daily availability heatmap.

    Args:
        availability: Output from HydrophoneDeploymentChecker.get_device_availability.
        title: Optional plot title.
        data_cmap: Colormap for data coverage (0-1).
        missing_color: Color for days with no data during deployment.
        not_deployed_color: Color for days outside deployments.
        show: If True, call plt.show().
    """
    bins = availability.get('bins') or []
    if not bins:
        print("No availability data to plot.")
        return None
    if availability.get('bin_size') != 'day':
        raise ValueError("plot_availability_calendar requires bin_size='day'")

    date_to_bin = {b['start'].date(): b for b in bins if b.get('start') is not None}
    start_dt = availability.get('start') or bins[0]['start']
    end_dt = availability.get('end') or bins[-1]['end']
    if start_dt is None or end_dt is None:
        print("Availability window is empty.")
        return None

    start_date = start_dt.date()
    end_date = end_dt.date()
    week0_start = start_date - timedelta(days=start_date.weekday())
    total_days = (end_date - start_date).days
    num_weeks = ((end_date - week0_start).days // 7) + 1

    grid = np.zeros((7, num_weeks, 4))
    grid[:, :, 3] = 0.0
    cmap = plt.get_cmap(data_cmap)
    missing_rgba = mcolors.to_rgba(missing_color)
    not_deployed_rgba = mcolors.to_rgba(not_deployed_color)

    for offset in range(total_days + 1):
        day = start_date + timedelta(days=offset)
        week_idx = (day - week0_start).days // 7
        dow = day.weekday()
        bin_entry = date_to_bin.get(day)
        if bin_entry is None or bin_entry.get('status') == 'not_deployed':
            color = not_deployed_rgba
        else:
            coverage = bin_entry.get('coverage') or 0.0
            if coverage <= 0:
                color = missing_rgba
            else:
                color = cmap(coverage)
        grid[dow, week_idx] = color

    width = min(18, max(8, num_weeks * 0.35))
    fig, ax = plt.subplots(figsize=(width, 3.2))
    ax.imshow(grid, aspect='auto', interpolation='none', origin='upper')

    ax.set_yticks(range(7))
    ax.set_yticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])

    month_ticks = []
    month_labels = []
    cursor = datetime(start_date.year, start_date.month, 1).date()
    if cursor < start_date:
        if cursor.month == 12:
            cursor = datetime(cursor.year + 1, 1, 1).date()
        else:
            cursor = datetime(cursor.year, cursor.month + 1, 1).date()
    while cursor <= end_date:
        week_idx = (cursor - week0_start).days // 7
        month_ticks.append(week_idx)
        label = cursor.strftime('%b %Y') if cursor.month == 1 else cursor.strftime('%b')
        month_labels.append(label)
        if cursor.month == 12:
            cursor = datetime(cursor.year + 1, 1, 1).date()
        else:
            cursor = datetime(cursor.year, cursor.month + 1, 1).date()

    ax.set_xticks(month_ticks)
    ax.set_xticklabels(month_labels)
    ax.set_xlabel("Week")
    ax.set_ylabel("Day")

    norm = mcolors.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label("Data coverage")

    from matplotlib.patches import Patch
    legend_items = [
        Patch(facecolor=missing_color, label="No data"),
        Patch(facecolor=not_deployed_color, label="Not deployed"),
    ]
    ax.legend(handles=legend_items, loc='upper right', frameon=False)

    if title is None:
        device_code = availability.get('device_code', 'device')
        title = f"Daily availability calendar for {device_code}"
    ax.set_title(title)

    ax.set_xlim(-0.5, num_weeks - 0.5)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def describe_spec_clip(clip_path: str | Path) -> Optional[dict]:
    """Print spectrogram clip timing metadata and return parsed values."""
    clip_path = Path(clip_path)
    try:
        data = np.load(clip_path, allow_pickle=True)
    except Exception as exc:
        print(f"Could not read spectrogram clip metadata: {exc}")
        return None

    seconds_per_col = data["seconds_per_column"] if "seconds_per_column" in data else None
    if seconds_per_col is not None:
        try:
            seconds_per_col = float(seconds_per_col)
        except Exception:
            seconds_per_col = None
    spec = data["spectrogram"] if "spectrogram" in data else None
    clip_start = data["clip_start"] if "clip_start" in data else None
    clip_end = data["clip_end"] if "clip_end" in data else None

    clip_duration = _clip_duration_from_meta(clip_start, clip_end)
    approx_duration = None
    if spec is not None and seconds_per_col:
        approx_duration = spec.shape[1] * seconds_per_col
    if seconds_per_col:
        msg = f"Spectrogram bin width: {seconds_per_col:.3f}s"
        if approx_duration is not None:
            msg += f"; approx duration {approx_duration:.2f}s"
        if clip_duration is not None:
            msg += f"; target clip {clip_duration:.2f}s"
        print(msg)
    return {
        'seconds_per_column': seconds_per_col,
        'approx_duration': approx_duration,
        'clip_duration': clip_duration,
    }


def plot_request_results(
    results: Iterable[dict],
    *,
    downloader: Optional[Any] = None,
    max_audio_seconds: Optional[float] = 10.0,
) -> None:
    """Plot first downloaded files and any request-level clips."""
    if downloader is not None:
        plot_first_spectrogram(downloader, title="Request spectrogram")
        plot_first_audio(downloader, max_seconds=max_audio_seconds)

    for result in results:
        spec_clip = (result.get('spectrogram') or {}).get('clip_path')
        audio_clip = (result.get('audio') or {}).get('clip_path')
        if spec_clip:
            describe_spec_clip(spec_clip)
        if spec_clip and audio_clip:
            plot_clip_pair(
                spec_clip,
                audio_clip,
                title=f"Clipped spectrogram + audio {result.get('timestamp')}",
            )
        else:
            if spec_clip:
                plot_spectrogram_clip(spec_clip, title=f"Spectrogram clip {result.get('timestamp')}")
            if audio_clip:
                plot_audio_waveform(audio_clip, max_seconds=None)


def _strip_tz(dt_obj: datetime) -> datetime:
    if dt_obj.tzinfo is None:
        return dt_obj
    return dt_obj.replace(tzinfo=None)


def _segments_from_bins(bins: list[dict]) -> list[tuple[str, datetime, datetime]]:
    if not bins:
        return []
    segments: list[tuple[str, datetime, datetime]] = []
    current_status = None
    current_start = None
    current_end = None
    for b in sorted(bins, key=lambda x: x['start']):
        status = 'data' if (b.get('coverage') or 0) > 0 else 'no_data'
        if current_status is None:
            current_status = status
            current_start = b['start']
            current_end = b['end']
            continue
        if status == current_status and b['start'] == current_end:
            current_end = b['end']
            continue
        segments.append((current_status, current_start, current_end))
        current_status = status
        current_start = b['start']
        current_end = b['end']
    if current_status is not None:
        segments.append((current_status, current_start, current_end))
    return segments


def _deployment_label(dep: Any) -> str:
    location = getattr(dep, "location_name", None) or getattr(dep, "location_code", None) or ""
    position = getattr(dep, "position_name", None)
    if position and position != location:
        location = f"{location} ({position})" if location else position
    begin = getattr(dep, "begin_date", None)
    end = getattr(dep, "end_date", None)
    begin_str = begin.strftime('%Y-%m-%d') if isinstance(begin, datetime) else "?"
    end_str = end.strftime('%Y-%m-%d') if isinstance(end, datetime) else "ongoing"
    label = location or getattr(dep, "device_code", "") or "deployment"
    return f"{label} ({begin_str} to {end_str})"


def availability_widget(
    checker: Any,
    *,
    inventory: Optional[dict] = None,
    device_codes: Optional[list[str]] = None,
    default_device: Optional[str] = None,
    timezone_str: str = "UTC",
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    bin_size: str = "day",
    max_days_per_request: int = 60,
    auto_run: bool = True,
    max_workers: int = 4,
    request_delay_seconds: float = 0.0,
):
    """Build an interactive Plotly + ipywidgets availability explorer."""
    try:
        import ipywidgets as widgets  # type: ignore
    except Exception as exc:
        raise ImportError("ipywidgets is required for availability_widget") from exc
    try:
        import pandas as pd  # type: ignore
        import plotly.express as px  # type: ignore
        import plotly.graph_objects as go  # type: ignore
    except Exception as exc:
        raise ImportError("plotly and pandas are required for availability_widget") from exc
    try:
        from IPython.display import display  # type: ignore
    except Exception as exc:
        raise ImportError("IPython is required for availability_widget") from exc

    if device_codes is None:
        if inventory is None:
            inventory = checker.collect_hydrophone_inventory()
        device_codes = sorted({
            row.get('device_code')
            for row in (inventory.get('history') or [])
            if row.get('device_code')
        })

    if not device_codes:
        raise ValueError("No device codes available for widget")

    if default_device is None:
        default_device = device_codes[0]

    def _coerce_date(value, include_full_day: bool = False):
        if value is None:
            return None
        if isinstance(value, datetime):
            dt = value
        else:
            dt = datetime(value.year, value.month, value.day)
        if include_full_day:
            dt = dt + timedelta(days=1)
        return dt

    def _build_timeline(availability: dict) -> "go.Figure":
        bins = availability.get('bins') or []
        deployments = availability.get('deployments') or []
        if not bins or not deployments:
            fig = go.Figure()
            fig.add_annotation(text="No availability data to plot.", showarrow=False)
            return fig

        bins_by_dep: dict[int, list[dict]] = {}
        for b in bins:
            dep_idx = b.get('deployment_index')
            if dep_idx is None:
                continue
            bins_by_dep.setdefault(dep_idx, []).append(b)

        segments = []
        for dep_idx, dep in enumerate(deployments):
            label = _deployment_label(dep)
            for status, seg_start, seg_end in _segments_from_bins(bins_by_dep.get(dep_idx, [])):
                segments.append({
                    'deployment': label,
                    'start': seg_start,
                    'end': seg_end,
                    'status': status,
                })

        if not segments:
            fig = go.Figure()
            fig.add_annotation(text="No availability data to plot.", showarrow=False)
            return fig

        df = pd.DataFrame(segments)
        color_map = {
            'data': "#2ca02c",
            'no_data': "#d62728",
        }
        fig = px.timeline(
            df,
            x_start="start",
            x_end="end",
            y="deployment",
            color="status",
            color_discrete_map=color_map,
        )
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(
            height=max(350, len(deployments) * 40 + 120),
            legend_orientation="h",
            legend_y=-0.15,
            margin=dict(l=40, r=20, t=40, b=40),
        )
        return fig

    def _build_calendar(availability: dict) -> "go.Figure":
        bins = availability.get('bins') or []
        if not bins:
            fig = go.Figure()
            fig.add_annotation(text="No availability data to plot.", showarrow=False)
            return fig
        if availability.get('bin_size') != 'day':
            fig = go.Figure()
            fig.add_annotation(text="Calendar view requires daily bins.", showarrow=False)
            return fig

        date_to_bin = {b['start'].date(): b for b in bins if b.get('start') is not None}
        start_dt = availability.get('start') or bins[0]['start']
        end_dt = availability.get('end') or bins[-1]['end']
        if start_dt is None or end_dt is None:
            fig = go.Figure()
            fig.add_annotation(text="Availability window is empty.", showarrow=False)
            return fig

        start_date = start_dt.date()
        end_date = end_dt.date()
        week0_start = start_date - timedelta(days=start_date.weekday())
        total_days = (end_date - start_date).days
        num_weeks = ((end_date - week0_start).days // 7) + 1
        week_starts = [week0_start + timedelta(days=7 * i) for i in range(num_weeks)]

        z = [[None for _ in range(num_weeks)] for _ in range(7)]
        text = [[None for _ in range(num_weeks)] for _ in range(7)]
        for offset in range(total_days + 1):
            day = start_date + timedelta(days=offset)
            week_idx = (day - week0_start).days // 7
            dow = day.weekday()
            bin_entry = date_to_bin.get(day)
            if bin_entry is None or bin_entry.get('status') == 'not_deployed':
                value = -1.0
                label = "not deployed"
                coverage_pct = 0
            else:
                coverage = bin_entry.get('coverage') or 0.0
                if coverage <= 0:
                    value = 0.0
                    label = "no data"
                    coverage_pct = 0
                else:
                    value = coverage
                    label = "data"
                    coverage_pct = int(round(coverage * 100))
            z[dow][week_idx] = value
            text[dow][week_idx] = f"{day} • {label} • {coverage_pct}%"

        colorscale = [
            [0.0, "#eeeeee"],
            [0.49, "#eeeeee"],
            [0.5, "#d62728"],
            [0.52, "#b7e1cd"],
            [1.0, "#2ca02c"],
        ]
        fig = go.Figure(
            data=go.Heatmap(
                z=z,
                x=week_starts,
                y=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
                colorscale=colorscale,
                zmin=-1,
                zmax=1,
                hoverinfo="text",
                text=text,
                xgap=2,
                ygap=2,
                colorbar=dict(title="Coverage"),
            )
        )
        fig.update_layout(
            height=320,
            margin=dict(l=40, r=20, t=40, b=40),
            xaxis=dict(
                tickformat="%b %Y",
                showgrid=False,
                ticklabelmode="period",
            ),
            yaxis=dict(autorange="reversed"),
        )
        return fig

    device_dropdown = widgets.Dropdown(
        options=device_codes,
        value=default_device if default_device in device_codes else device_codes[0],
        description="Device",
        layout=widgets.Layout(width="300px"),
    )
    start_picker = widgets.DatePicker(
        description="Start",
        value=start_date.date() if isinstance(start_date, datetime) else None,
    )
    end_picker = widgets.DatePicker(
        description="End",
        value=end_date.date() if isinstance(end_date, datetime) else None,
    )
    tz_input = widgets.Text(
        value=timezone_str,
        description="Timezone",
        placeholder="UTC",
        layout=widgets.Layout(width="220px"),
    )
    bin_dropdown = widgets.Dropdown(
        options=["day", "hour"],
        value=bin_size if bin_size in ("day", "hour") else "day",
        description="Bins",
        layout=widgets.Layout(width="180px"),
    )
    view_toggle = widgets.ToggleButtons(
        options=[("Timeline", "timeline"), ("Calendar", "calendar")],
        value="timeline",
        description="View",
    )
    update_button = widgets.Button(
        description="Update",
        button_style="primary",
        icon="refresh",
    )
    status = widgets.HTML("")
    output = widgets.Output()
    fig_widget = go.FigureWidget()

    def _on_view_change(change):
        if change["new"] == "calendar":
            bin_dropdown.value = "day"
            bin_dropdown.disabled = True
        else:
            bin_dropdown.disabled = False

    view_toggle.observe(_on_view_change, names="value")

    def _update(_=None):
        status.value = ""
        try:
            start_dt = _coerce_date(start_picker.value)
            end_dt = _coerce_date(end_picker.value, include_full_day=True)
            effective_bin = "day" if view_toggle.value == "calendar" else bin_dropdown.value
            try:
                from tqdm.auto import tqdm  # type: ignore
            except Exception:
                tqdm = None

            progress_bar = None
            if tqdm is not None:
                progress_bar = tqdm(total=0, desc="Querying archive", leave=False)

            def _progress(total=0, advance=0):
                if progress_bar is None:
                    return
                if total and progress_bar.total != total:
                    progress_bar.reset(total=total)
                if advance:
                    progress_bar.update(advance)

            availability = checker.get_device_availability(
                device_dropdown.value,
                start_date=start_dt,
                end_date=end_dt,
                timezone_str=tz_input.value or "UTC",
                bin_size=effective_bin,
                max_days_per_request=max_days_per_request,
                progress=_progress,
                quiet=True,
                max_workers=max_workers,
                request_delay_seconds=request_delay_seconds,
            )
            if progress_bar is not None:
                progress_bar.close()
            fig = _build_calendar(availability) if view_toggle.value == "calendar" else _build_timeline(availability)
            fig_widget.data = []
            fig_widget.layout = go.Layout()
            for trace in fig.data:
                fig_widget.add_trace(trace)
            fig_widget.update_layout(fig.layout)
        except Exception as exc:
            status.value = f"<b>Error:</b> {exc}"

    update_button.on_click(_update)
    if auto_run:
        _update()

    controls = widgets.VBox([
        widgets.HBox([device_dropdown, view_toggle]),
        widgets.HBox([start_picker, end_picker, tz_input]),
        widgets.HBox([bin_dropdown, update_button]),
        status,
    ])
    return widgets.VBox([controls, fig_widget])


def _load_audio_data(audio_path: str | Path) -> tuple[Optional[np.ndarray], Optional[float]]:
    if sf is None and librosa is None:
        print("Install soundfile or librosa to plot audio waveforms.")
        return None, None

    audio_path = Path(audio_path)
    if sf is not None:
        audio_data, sr = sf.read(audio_path)
    else:
        audio_data, sr = librosa.load(audio_path, sr=None, mono=False)

    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)
    return audio_data, sr


def _load_spectrogram_clip_data(
    clip_path: str | Path,
) -> Optional[tuple[np.ndarray, np.ndarray, Optional[float], Optional[float]]]:
    clip_path = Path(clip_path)
    try:
        data = np.load(clip_path, allow_pickle=True)
    except Exception as exc:
        print(f"Failed to load spectrogram clip {clip_path}: {exc}")
        return None

    if "spectrogram" not in data:
        print(f"No spectrogram array found in {clip_path}")
        return None

    spec = np.asarray(data["spectrogram"])
    freq = data["frequency"] if "frequency" in data else None
    seconds_per_col = data["seconds_per_column"] if "seconds_per_column" in data else None
    clip_start = data["clip_start"] if "clip_start" in data else None
    clip_end = data["clip_end"] if "clip_end" in data else None

    if freq is None:
        freq = np.arange(spec.shape[0])
    else:
        freq = np.asarray(freq).squeeze()
    if freq.size != spec.shape[0]:
        freq = np.arange(spec.shape[0])

    if seconds_per_col is not None:
        try:
            seconds_per_col = float(seconds_per_col)
        except Exception:
            seconds_per_col = None

    clip_duration = _clip_duration_from_meta(clip_start, clip_end)
    return spec, freq, seconds_per_col, clip_duration


def _spectrogram_time_axis(
    num_cols: int,
    seconds_per_col: Optional[float],
    clip_duration: Optional[float],
) -> np.ndarray:
    if clip_duration and num_cols:
        return np.linspace(0.0, clip_duration, num=num_cols, endpoint=False)
    if seconds_per_col:
        return np.arange(num_cols) * seconds_per_col
    return np.arange(num_cols)


def _clip_duration_from_meta(start_value: Any, end_value: Any) -> Optional[float]:
    start_dt = _parse_iso_datetime(start_value)
    end_dt = _parse_iso_datetime(end_value)
    if start_dt is None or end_dt is None:
        return None
    duration = (end_dt - start_dt).total_seconds()
    return max(0.0, duration)


def _parse_iso_datetime(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, bytes):
        try:
            value = value.decode()
        except Exception:
            return None
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        try:
            value = value.item()
        except Exception:
            return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except Exception:
            return None
    return None


def _clip_duration_from_seconds_per_col(
    num_cols: int,
    seconds_per_col: Optional[float],
) -> Optional[float]:
    if seconds_per_col is None:
        return None
    return float(num_cols) * float(seconds_per_col)


def _duration_from_audio(audio_time: np.ndarray) -> Optional[float]:
    if audio_time.size == 0:
        return None
    return float(audio_time[-1])


def _audio_clip_duration_from_meta(audio_path: str | Path) -> Optional[float]:
    meta_path = Path(audio_path).with_suffix(".json")
    if not meta_path.exists():
        return None
    try:
        with meta_path.open("r", encoding="utf-8") as handle:
            meta = json.load(handle)
    except Exception:
        return None
    return _clip_duration_from_meta(meta.get("clip_start"), meta.get("clip_end"))
