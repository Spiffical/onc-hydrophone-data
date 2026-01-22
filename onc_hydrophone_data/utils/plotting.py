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
