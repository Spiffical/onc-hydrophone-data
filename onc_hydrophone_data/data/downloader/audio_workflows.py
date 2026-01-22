import math
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ...onc.common import ensure_timezone_aware, format_iso_utc
from .constants import FIVE_MINUTES_SECONDS


def download_audio_for_range(
    self,
    device_code: str,
    start_dt: datetime,
    end_dt: datetime,
    *,
    tag: str = 'audio_range',
    filetype: str = 'mat',
    extensions: Sequence[str] = ('flac', 'wav'),
    max_download_workers: Optional[int] = None,
) -> Dict[str, Any]:
    """Download all 5-minute audio files that overlap a time window.

    Args:
        device_code: ONC hydrophone device code.
        start_dt: Range start (timezone‑aware recommended).
        end_dt: Range end (timezone‑aware recommended).
        tag: Output folder tag for this run.
        filetype: Used for directory naming (kept for compatibility).
        extensions: Audio extensions to try in order (default: FLAC, then WAV).
        max_download_workers: Override parallel audio download workers.

    Returns:
        Dict with normalized start/end datetimes and request strings.

    Raises:
        ValueError: If the time range is invalid.
    """
    start_dt = ensure_timezone_aware(start_dt)
    end_dt = ensure_timezone_aware(end_dt)
    if end_dt <= start_dt:
        raise ValueError("end_dt must be after start_dt")

    self.setup_directories(filetype, device_code, tag, start_dt, end_dt)
    request_start = format_iso_utc(start_dt)
    request_end = format_iso_utc(end_dt)
    self.download_audio_files(
        device_code,
        request_start,
        request_end,
        extensions=extensions,
        max_download_workers=max_download_workers,
    )
    return {
        'start_dt': start_dt,
        'end_dt': end_dt,
        'request_start': request_start,
        'request_end': request_end,
    }


def download_sampled_audio(
    self,
    device_code: str,
    start_dt: datetime,
    end_dt: datetime,
    total_audio_files: int,
    files_per_request: int,
    *,
    tag: str = 'sampled_audio',
    filetype: str = 'mat',
    extensions: Sequence[str] = ('flac', 'wav'),
    max_download_workers: Optional[int] = None,
) -> Dict[str, Any]:
    """Download a uniform sample of 5-minute audio files across a date range.

    Args:
        device_code: ONC hydrophone device code.
        start_dt: Range start (timezone‑aware recommended).
        end_dt: Range end (timezone‑aware recommended).
        total_audio_files: Target number of files to sample.
        files_per_request: Number of 5‑minute files per request.
        tag: Output folder tag for this run.
        filetype: Used for directory naming (kept for compatibility).
        extensions: Audio extensions to try in order.
        max_download_workers: Override parallel audio download workers.

    Returns:
        Dict including request windows used for sampling.

    Raises:
        ValueError: If inputs are invalid.
    """
    if total_audio_files <= 0:
        raise ValueError("total_audio_files must be positive")
    if files_per_request <= 0:
        raise ValueError("files_per_request must be positive")
    start_dt = ensure_timezone_aware(start_dt)
    end_dt = ensure_timezone_aware(end_dt)
    if end_dt <= start_dt:
        raise ValueError("end_dt must be after start_dt")

    windows = self._build_request_windows(start_dt, end_dt)
    if not windows:
        raise ValueError("No windows found for the requested range")

    max_start_index = max(0, len(windows) - files_per_request)
    total_requests = max(1, math.ceil(total_audio_files / files_per_request))
    if total_requests == 1:
        start_indices = [0]
    else:
        step = max_start_index / (total_requests - 1) if total_requests > 1 else 0
        start_indices = [int(round(step * i)) for i in range(total_requests)]

    start_indices = sorted({min(max(idx, 0), max_start_index) for idx in start_indices})

    request_windows: List[Tuple[datetime, datetime]] = []
    for idx in start_indices:
        chunk = windows[idx:idx + files_per_request]
        if not chunk:
            continue
        request_windows.append((chunk[0][0], chunk[-1][1]))

    if not request_windows:
        raise ValueError("No request windows built for the sampled range")

    range_start = min(start for start, _ in request_windows)
    range_end = max(end for _, end in request_windows)
    self.setup_directories(filetype, device_code, tag, range_start, range_end)

    for start, end in request_windows:
        self.download_audio_files(
            device_code,
            format_iso_utc(start),
            format_iso_utc(end),
            extensions=extensions,
            max_download_workers=max_download_workers,
        )

    return {
        'start_dt': start_dt,
        'end_dt': end_dt,
        'total_audio_files': total_audio_files,
        'files_per_request': files_per_request,
        'request_windows': request_windows,
    }


def _build_event_audio_windows(
    self,
    event_times: Iterable[datetime],
    *,
    window_seconds: int = FIVE_MINUTES_SECONDS,
) -> List[Tuple[datetime, datetime]]:
    """Convert event timestamps into aligned audio windows.

    Args:
        event_times: Iterable of event timestamps.
        window_seconds: Window size (default: 300 seconds).

    Returns:
        List of ``(start_dt, end_dt)`` windows aligned to the 5‑minute grid.
    """
    windows: List[Tuple[datetime, datetime]] = []
    for ts in event_times:
        ts = ensure_timezone_aware(ts)
        start = ts.replace(minute=(ts.minute // 5) * 5, second=0, microsecond=0)
        end = start + timedelta(seconds=window_seconds)
        windows.append((start, end))
    return windows


def describe_event_audio_windows(
    self,
    event_times: Iterable[datetime],
    *,
    window_seconds: int = FIVE_MINUTES_SECONDS,
) -> List[Tuple[datetime, datetime]]:
    """Print event-to-window mappings and return the windows.

    Args:
        event_times: Iterable of event timestamps.
        window_seconds: Window size (default: 300 seconds).

    Returns:
        List of ``(start_dt, end_dt)`` windows aligned to the 5‑minute grid.
    """
    times = list(event_times)
    windows = self._build_event_audio_windows(times, window_seconds=window_seconds)
    for ts, window in zip(times, windows):
        start, end = window
        print(f"Event {ensure_timezone_aware(ts)} → Window: {format_iso_utc(start)} to {format_iso_utc(end)}")
    return windows


def download_audio_for_events(
    self,
    device_code: str,
    event_times: Iterable[datetime],
    *,
    tag: str = 'audio_events',
    filetype: str = 'mat',
    window_seconds: int = FIVE_MINUTES_SECONDS,
    extensions: Sequence[str] = ('flac', 'wav'),
    max_download_workers: Optional[int] = None,
) -> List[Tuple[datetime, datetime]]:
    """Download audio for the 5-minute windows containing each event.

    Args:
        device_code: ONC hydrophone device code.
        event_times: Iterable of event timestamps.
        tag: Output folder tag for this run.
        filetype: Used for directory naming (kept for compatibility).
        window_seconds: Window size (default: 300 seconds).
        extensions: Audio extensions to try in order.
        max_download_workers: Override parallel audio download workers.

    Returns:
        List of ``(start_dt, end_dt)`` windows requested.
    """
    windows = self._build_event_audio_windows(event_times, window_seconds=window_seconds)
    if not windows:
        return []
    range_start = min(start for start, _ in windows)
    range_end = max(end for _, end in windows)
    self.setup_directories(filetype, device_code, tag, range_start, range_end)
    for start, end in windows:
        self.download_audio_files(
            device_code,
            format_iso_utc(start),
            format_iso_utc(end),
            extensions=extensions,
            max_download_workers=max_download_workers,
        )
    return windows


def plan_audio_window(
    self,
    center_time: datetime,
    duration_seconds: float,
    *,
    window_seconds: int = FIVE_MINUTES_SECONDS,
) -> Dict[str, Any]:
    """Plan the 5-minute audio windows needed for a center time + duration.

    Args:
        center_time: Center timestamp of the desired clip.
        duration_seconds: Total clip duration (seconds).
        window_seconds: Window size (default: 300 seconds).

    Returns:
        Dict describing clip bounds and request window range.
    """
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
    self,
    center_time: datetime,
    duration_seconds: float,
    *,
    window_seconds: int = FIVE_MINUTES_SECONDS,
) -> Dict[str, Any]:
    """Print a summary of the audio window plan and return the plan.

    Args:
        center_time: Center timestamp of the desired clip.
        duration_seconds: Total clip duration (seconds).
        window_seconds: Window size (default: 300 seconds).

    Returns:
        Dict describing clip bounds and request window range.
    """
    plan = self.plan_audio_window(
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


def download_audio_for_center_time(
    self,
    device_code: str,
    center_time: datetime,
    duration_seconds: float,
    *,
    tag: str = 'audio_window',
    filetype: str = 'mat',
    window_seconds: int = FIVE_MINUTES_SECONDS,
    extensions: Sequence[str] = ('flac', 'wav'),
    max_download_workers: Optional[int] = None,
) -> Dict[str, Any]:
    """Download audio files that cover a centered clip window.

    Args:
        device_code: ONC hydrophone device code.
        center_time: Center timestamp of the desired clip.
        duration_seconds: Total clip duration (seconds).
        tag: Output folder tag for this run.
        filetype: Used for directory naming (kept for compatibility).
        window_seconds: Window size (default: 300 seconds).
        extensions: Audio extensions to try in order.
        max_download_workers: Override parallel audio download workers.

    Returns:
        Dict describing clip bounds and request window range.
    """
    plan = self.plan_audio_window(
        center_time,
        duration_seconds,
        window_seconds=window_seconds,
    )
    self.setup_directories(filetype, device_code, tag, plan['file_start'], plan['file_end'])
    self.download_audio_files(
        device_code,
        plan['request_start'],
        plan['request_end'],
        extensions=extensions,
        max_download_workers=max_download_workers,
    )
    return plan
