import math
from copy import deepcopy
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from onc_hydrophone_data.data.hydrophone_downloader import HydrophoneDownloader
from onc_hydrophone_data.onc.common import ensure_timezone_aware, format_iso_utc


DEFAULT_PARALLEL_CONFIG = {
    'stagger_seconds': 3.0,
    'max_wait_minutes': 45,
    'poll_interval_seconds': 30,
    'max_download_workers': 4,
    'max_attempts': 6,
}


def _extract_windows(
    device_code: str,
    request_windows: Union[
        Dict[str, Sequence[Tuple[datetime, datetime]]],
        Sequence[Tuple[datetime, datetime]],
    ],
) -> List[Tuple[datetime, datetime]]:
    if isinstance(request_windows, dict):
        windows = request_windows.get(device_code) or []
    else:
        windows = list(request_windows)
    return windows


def run_parallel_for_device(
    downloader: HydrophoneDownloader,
    device_code: str,
    request_windows: Union[
        Dict[str, Sequence[Tuple[datetime, datetime]]],
        Sequence[Tuple[datetime, datetime]],
    ],
    spectrograms_per_request: int,
    *,
    tag: str = 'tutorial',
    download_audio: Optional[bool] = None,
    download_flac: Optional[bool] = None,
    parallel_config: Optional[Dict[str, float]] = None,
    data_product_options: Optional[Dict[str, Union[str, int, float]]] = None,
    **overrides,
):
    """Shim around the default download workflow.

    This calls the parallel submission/polling implementation under the hood.
    Prefer `HydrophoneDownloader.download_spectrogram_windows` for user-facing code.
    Use data_product_options to pass ONC dpo_* filters (optional).
    """
    windows = _extract_windows(device_code, request_windows)
    if not windows:
        raise ValueError(f"No windows defined for {device_code}")
    if download_audio is None:
        download_audio = bool(download_flac)
    elif download_flac is not None and download_audio != download_flac:
        raise ValueError("download_audio and download_flac provide conflicting values")

    config = {**DEFAULT_PARALLEL_CONFIG}
    if parallel_config:
        config.update(parallel_config)
    config.update(overrides)
    if 'data_product_options' in config:
        data_product_options = config.pop('data_product_options') or data_product_options
    spectral_downsample = config.pop('spectral_downsample', None)
    if spectral_downsample is not None:
        data_product_options = dict(data_product_options or {})
        data_product_options.setdefault('dpo_spectralDataDownsample', spectral_downsample)

    return downloader.run_parallel_windows(
        device_code,
        windows,
        spectrograms_per_request=spectrograms_per_request,
        tag=tag,
        download_audio=download_audio,
        data_product_options=data_product_options,
        **config,
    )


def build_sampling_windows(
    device_code: str,
    start_dt: datetime,
    end_dt: datetime,
    total_spectrograms: int,
    spectrograms_per_request: int,
) -> Dict[str, List[Tuple[datetime, datetime]]]:
    """Spread a target number of five-minute windows between two datetimes."""
    if total_spectrograms <= 0:
        raise ValueError("total_spectrograms must be positive")
    if spectrograms_per_request <= 0:
        raise ValueError("spectrograms_per_request must be positive")
    if end_dt <= start_dt:
        raise ValueError("end_dt must be after start_dt")

    duration_per_request = max(0, (spectrograms_per_request - 1) * 300)
    total_requests = max(1, math.ceil(total_spectrograms / spectrograms_per_request))
    usable_seconds = max(0, (end_dt - start_dt).total_seconds() - duration_per_request)

    if total_requests == 1:
        starts = [start_dt]
    else:
        step = usable_seconds / (total_requests - 1) if total_requests > 1 else 0
        starts = [start_dt + timedelta(seconds=step * i) for i in range(total_requests)]

    windows = [(start, start + timedelta(seconds=duration_per_request)) for start in starts]
    return {device_code: windows}


def plan_audio_window(
    center_time: datetime,
    duration_seconds: float,
    *,
    window_seconds: int = 300,
) -> Dict[str, Any]:
    """Plan the 5-minute audio windows needed for a center time + duration."""
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")

    center_time = ensure_timezone_aware(center_time)
    half = duration_seconds / 2.0
    clip_start = center_time - timedelta(seconds=half)
    clip_end = center_time + timedelta(seconds=half)

    file_start = clip_start.replace(minute=(clip_start.minute // 5) * 5, second=0, microsecond=0)
    file_end = clip_end.replace(minute=(clip_end.minute // 5) * 5, second=0, microsecond=0)
    file_end += timedelta(seconds=window_seconds)

    window_count = max(1, int((file_end - file_start).total_seconds() / window_seconds))
    crosses_boundary = window_count > 1

    return {
        'center_time': center_time,
        'duration_seconds': duration_seconds,
        'clip_start': clip_start,
        'clip_end': clip_end,
        'file_start': file_start,
        'file_end': file_end,
        'window_count': window_count,
        'crosses_boundary': crosses_boundary,
        'request_start': format_iso_utc(file_start),
        'request_end': format_iso_utc(file_end),
    }


def describe_audio_window(
    center_time: datetime,
    duration_seconds: float,
    *,
    window_seconds: int = 300,
) -> Dict[str, Any]:
    """Print a summary of the audio window plan and return the plan."""
    plan = plan_audio_window(
        center_time,
        duration_seconds,
        window_seconds=window_seconds,
    )
    print(f"Center: {plan['center_time']}")
    print(f"Clip range: {plan['clip_start']} to {plan['clip_end']}")
    print(f"Files needed: {plan['file_start']} to {plan['file_end']} ({plan['window_count']} file(s))")
    if plan['crosses_boundary']:
        print("This clip crosses a 5-minute boundary, so two adjacent audio files are required.")
    return plan


def print_audio_download_range(plan: Dict[str, Any]) -> None:
    """Print the request range for a planned audio download."""
    print(f"Would download audio from {plan['request_start']} to {plan['request_end']}")


def download_audio_for_center_time(
    downloader: HydrophoneDownloader,
    device_code: str,
    center_time: datetime,
    duration_seconds: float,
    *,
    extensions: Sequence[str] = ('flac', 'wav'),
    max_download_workers: Optional[int] = None,
) -> Dict[str, Any]:
    """Download audio files that cover a centered clip window."""
    plan = plan_audio_window(center_time, duration_seconds)
    downloader.download_audio_files(
        device_code,
        plan['request_start'],
        plan['request_end'],
        extensions=extensions,
        max_download_workers=max_download_workers,
    )
    return plan


def format_audio_range(
    start_dt: datetime,
    end_dt: datetime,
) -> Dict[str, Any]:
    """Format a start/end audio range into ONC-compatible strings."""
    start_dt = ensure_timezone_aware(start_dt)
    end_dt = ensure_timezone_aware(end_dt)
    if end_dt <= start_dt:
        raise ValueError("end_dt must be after start_dt")
    return {
        'start_dt': start_dt,
        'end_dt': end_dt,
        'request_start': format_iso_utc(start_dt),
        'request_end': format_iso_utc(end_dt),
    }


def describe_audio_range(
    start_dt: datetime,
    end_dt: datetime,
) -> Dict[str, Any]:
    """Print a summary of a start/end audio range."""
    plan = format_audio_range(start_dt, end_dt)
    print(f"Downloading audio from {plan['request_start']} to {plan['request_end']}")
    return plan


def download_audio_range(
    downloader: HydrophoneDownloader,
    device_code: str,
    start_dt: datetime,
    end_dt: datetime,
    *,
    tag: str = 'audio_range',
    filetype: str = 'mat',
    extensions: Sequence[str] = ('flac', 'wav'),
    max_download_workers: Optional[int] = None,
    setup_dirs: bool = True,
) -> Dict[str, Any]:
    """Download audio for a start/end range and return the request plan."""
    plan = format_audio_range(start_dt, end_dt)
    if setup_dirs:
        downloader.setup_directories(
            filetype,
            device_code,
            tag,
            plan['start_dt'],
            plan['end_dt'],
        )
    downloader.download_audio_files(
        device_code,
        plan['request_start'],
        plan['request_end'],
        extensions=extensions,
        max_download_workers=max_download_workers,
    )
    return plan


def build_event_audio_windows(
    event_times: Iterable[datetime],
    *,
    window_seconds: int = 300,
) -> List[Tuple[datetime, datetime]]:
    """Floor timestamps to 5-minute windows and return (start, end) pairs."""
    windows: List[Tuple[datetime, datetime]] = []
    for ts in event_times:
        ts = ensure_timezone_aware(ts)
        start = ts.replace(minute=(ts.minute // 5) * 5, second=0, microsecond=0)
        end = start + timedelta(seconds=window_seconds)
        windows.append((start, end))
    return windows


def describe_event_audio_windows(
    event_times: Iterable[datetime],
    *,
    window_seconds: int = 300,
) -> List[Tuple[datetime, datetime]]:
    """Print event-to-window mappings and return the windows."""
    windows = build_event_audio_windows(event_times, window_seconds=window_seconds)
    for ts, window in zip(event_times, windows):
        start, end = window
        print(f"Event {ensure_timezone_aware(ts)} → Window: {format_iso_utc(start)} to {format_iso_utc(end)}")
    return windows


def download_audio_for_events(
    downloader: HydrophoneDownloader,
    device_code: str,
    event_times: Iterable[datetime],
    *,
    window_seconds: int = 300,
    extensions: Sequence[str] = ('flac', 'wav'),
    max_download_workers: Optional[int] = None,
) -> List[Tuple[datetime, datetime]]:
    """Download audio for the 5-minute windows containing each event."""
    windows = build_event_audio_windows(event_times, window_seconds=window_seconds)
    for start, end in windows:
        downloader.download_audio_files(
            device_code,
            format_iso_utc(start),
            format_iso_utc(end),
            extensions=extensions,
            max_download_workers=max_download_workers,
        )
    return windows


HSD_BASE_FILTERS = {
    'dataProductCode': 'HSD',
    'extension': 'mat',
}


def build_hsd_filters(
    device_code: str,
    start: datetime,
    end: datetime,
    *,
    downsample: Optional[int] = 2,
    hydrophone_data_diversion_mode: Optional[str] = 'OD',
    hydrophone_acquisition_mode: Optional[str] = None,
    spectrogram_source: Optional[str] = None,
    spectrogram_concatenation: Optional[str] = "None",
    spectrogram_colour_palette: Optional[int] = None,
    upper_colour_limit: Optional[float] = None,
    lower_colour_limit: Optional[float] = None,
    spectrogram_frequency_upper_limit: Optional[int] = None,
    spectrogram_upper_frequency_limit: Optional[int] = None,
    window_sec: Optional[float] = None,
    overlap: Optional[float] = None,
    extension: Optional[str] = None,
) -> Dict[str, Union[str, int, float]]:
    """Return a filter dict for HSD data product requests.

    Parameters map directly to ONC data product options (dpo_*). Only set the
    options you need; unsupported combinations may be ignored by ONC.
    """
    if end <= start:
        raise ValueError("end must be after start")

    def fmt(dt: datetime) -> str:
        return dt.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'

    filters = deepcopy(HSD_BASE_FILTERS)
    filters.update({
        'deviceCode': device_code,
        'dateFrom': fmt(start),
        'dateTo': fmt(end),
    })
    if downsample is not None:
        filters['dpo_spectralDataDownsample'] = downsample
    if hydrophone_data_diversion_mode is not None:
        filters['dpo_hydrophoneDataDiversionMode'] = hydrophone_data_diversion_mode
    if hydrophone_acquisition_mode is not None:
        filters['dpo_hydrophoneAcquisitionMode'] = hydrophone_acquisition_mode
    if spectrogram_source is not None:
        filters['dpo_spectrogramSource'] = spectrogram_source
    if spectrogram_concatenation is not None:
        filters['dpo_spectrogramConcatenation'] = spectrogram_concatenation
    if spectrogram_colour_palette is not None:
        filters['dpo_spectrogramColourPalette'] = spectrogram_colour_palette
    if upper_colour_limit is not None:
        filters['dpo_upperColourLimit'] = upper_colour_limit
    if lower_colour_limit is not None:
        filters['dpo_lowerColourLimit'] = lower_colour_limit
    if spectrogram_frequency_upper_limit is not None:
        filters['dpo_spectrogramFrequencyUpperLimit'] = spectrogram_frequency_upper_limit
    if spectrogram_upper_frequency_limit is not None:
        filters['dpo_spectrogramUpperFrequencyLimit'] = spectrogram_upper_frequency_limit
    if window_sec is not None:
        filters['dpo_spectrogramWindowLengthSec'] = window_sec
    if overlap is not None:
        filters['dpo_spectrogramOverlap'] = overlap
    if extension is not None:
        filters['extension'] = extension
    return filters
