import math
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from ...onc.common import ensure_timezone_aware
from .constants import FIVE_MINUTES_SECONDS
from .flags import _resolve_download_audio


def download_spectrogram_windows(
    self,
    device_code: str,
    request_windows: Union[
        Dict[str, Sequence[Tuple[datetime, datetime]]],
        Sequence[Tuple[datetime, datetime]],
    ],
    spectrograms_per_request: int,
    *,
    tag: str = 'download',
    download_audio: Optional[bool] = None,
    download_flac: Optional[bool] = None,
    audio_download_workers: Optional[int] = None,
    data_product_options: Optional[Dict[str, Any]] = None,
    **download_options,
) -> Dict[str, Any]:
    """Download spectrogram windows using the default request workflow.

    This is the standard entry point: it submits all requests immediately,
    then polls and downloads them in parallel. Use keyword arguments to tune
    pacing, retries, or polling behavior when needed.

    Args:
        device_code: ONC hydrophone device code (e.g., ``ICLISTENHF6324``).
        request_windows: Sequence of ``(start_dt, end_dt)`` windows, or a
            dict mapping device codes to windows.
        spectrograms_per_request: Number of 5‑minute windows bundled per request.
        tag: Output folder tag for this run.
        download_audio: If True, download matching audio for each request.
        download_flac: Legacy alias for ``download_audio``.
        audio_download_workers: Override parallelism for audio downloads.
        data_product_options: Optional ONC ``dpo_*`` overrides for HSD requests.
        **download_options: Extra controls forwarded to ``run_parallel_windows``
            (e.g., ``stagger_seconds``, ``max_wait_minutes``, ``poll_interval_seconds``).

    Returns:
        Summary dict with counts, output paths, and timing information.

    Raises:
        ValueError: If no windows are provided for the device.
    """
    if isinstance(request_windows, dict):
        windows = list(request_windows.get(device_code) or [])
    else:
        windows = list(request_windows)
    if not windows:
        raise ValueError(f"No windows defined for {device_code}")
    download_audio = _resolve_download_audio(download_audio, download_flac)

    return self.run_parallel_windows(
        device_code,
        windows,
        spectrograms_per_request=spectrograms_per_request,
        tag=tag,
        download_audio=download_audio,
        audio_download_workers=audio_download_workers,
        data_product_options=data_product_options,
        **download_options,
    )


def download_spectrograms_for_range(
    self,
    device_code: str,
    start_dt: datetime,
    end_dt: datetime,
    spectrograms_per_batch: int,
    *,
    tag: str = 'date_range',
    download_audio: Optional[bool] = None,
    download_flac: Optional[bool] = None,
    audio_download_workers: Optional[int] = None,
    data_product_options: Optional[Dict[str, Any]] = None,
    **download_options,
) -> Dict[str, Any]:
    """Download all 5-minute spectrograms between two datetimes.

    This helper batches contiguous 5-minute windows into requests of
    size ``spectrograms_per_batch``, then submits them in parallel.

    Args:
        device_code: ONC hydrophone device code.
        start_dt: Range start (timezone‑aware recommended).
        end_dt: Range end (timezone‑aware recommended).
        spectrograms_per_batch: Number of 5‑minute windows per request.
        tag: Output folder tag for this run.
        download_audio: If True, download matching audio.
        download_flac: Legacy alias for ``download_audio``.
        audio_download_workers: Override parallelism for audio downloads.
        data_product_options: Optional ONC ``dpo_*`` overrides for HSD requests.
        **download_options: Extra controls forwarded to ``download_spectrogram_windows``.

    Returns:
        Summary dict with counts, output paths, and timing information.

    Raises:
        ValueError: If inputs are invalid or no windows are found.
    """
    if spectrograms_per_batch <= 0:
        raise ValueError("spectrograms_per_batch must be positive")
    download_audio = _resolve_download_audio(download_audio, download_flac)
    windows = self._build_request_windows(start_dt, end_dt)
    if not windows:
        raise ValueError("No windows found for the requested range")
    request_windows: List[Tuple[datetime, datetime]] = []
    for i in range(0, len(windows), spectrograms_per_batch):
        chunk = windows[i:i + spectrograms_per_batch]
        request_windows.append((chunk[0][0], chunk[-1][1]))
    return self.download_spectrogram_windows(
        device_code,
        request_windows,
        spectrograms_per_request=spectrograms_per_batch,
        tag=tag,
        download_audio=download_audio,
        audio_download_workers=audio_download_workers,
        data_product_options=data_product_options,
        **download_options,
    )


def download_sampled_spectrograms(
    self,
    device_code: str,
    start_dt: datetime,
    end_dt: datetime,
    total_spectrograms: int,
    spectrograms_per_request: int,
    *,
    tag: str = 'sampled',
    download_audio: Optional[bool] = None,
    download_flac: Optional[bool] = None,
    audio_download_workers: Optional[int] = None,
    data_product_options: Optional[Dict[str, Any]] = None,
    **download_options,
) -> Dict[str, Any]:
    """Download a uniform sample of spectrogram windows across a date range.

    The total number of requests is derived from ``total_spectrograms`` and
    ``spectrograms_per_request``, then spaced evenly across the range.

    Args:
        device_code: ONC hydrophone device code.
        start_dt: Range start (timezone‑aware recommended).
        end_dt: Range end (timezone‑aware recommended).
        total_spectrograms: Total spectrograms to sample across the range.
        spectrograms_per_request: Number of windows per request.
        tag: Output folder tag for this run.
        download_audio: If True, download matching audio.
        download_flac: Legacy alias for ``download_audio``.
        audio_download_workers: Override parallelism for audio downloads.
        data_product_options: Optional ONC ``dpo_*`` overrides for HSD requests.
        **download_options: Extra controls forwarded to ``download_spectrogram_windows``.

    Returns:
        Summary dict with counts, output paths, and timing information.

    Raises:
        ValueError: If inputs are invalid.
    """
    if total_spectrograms <= 0:
        raise ValueError("total_spectrograms must be positive")
    if spectrograms_per_request <= 0:
        raise ValueError("spectrograms_per_request must be positive")
    download_audio = _resolve_download_audio(download_audio, download_flac)
    start_dt = ensure_timezone_aware(start_dt)
    end_dt = ensure_timezone_aware(end_dt)
    if end_dt <= start_dt:
        raise ValueError("end_dt must be after start_dt")

    duration_per_request = max(0, (spectrograms_per_request - 1) * FIVE_MINUTES_SECONDS)
    total_requests = max(1, math.ceil(total_spectrograms / spectrograms_per_request))
    usable_seconds = max(0, (end_dt - start_dt).total_seconds() - duration_per_request)

    if total_requests == 1:
        starts = [start_dt]
    else:
        step = usable_seconds / (total_requests - 1) if total_requests > 1 else 0
        starts = [start_dt + timedelta(seconds=step * i) for i in range(total_requests)]

    windows = [(start, start + timedelta(seconds=duration_per_request)) for start in starts]
    return self.download_spectrogram_windows(
        device_code,
        windows,
        spectrograms_per_request=spectrograms_per_request,
        tag=tag,
        download_audio=download_audio,
        audio_download_workers=audio_download_workers,
        data_product_options=data_product_options,
        **download_options,
    )


def download_spectrograms_for_events(
    self,
    device_code: str,
    event_times: Iterable[datetime],
    spectrograms_per_request: int,
    *,
    tag: str = 'event_times',
    download_audio: Optional[bool] = None,
    download_flac: Optional[bool] = None,
    audio_download_workers: Optional[int] = None,
    data_product_options: Optional[Dict[str, Any]] = None,
    **download_options,
) -> Dict[str, Any]:
    """Download spectrogram windows containing each event timestamp.

    Event timestamps are mapped to their containing 5‑minute windows, then
    contiguous windows are grouped into requests of size ``spectrograms_per_request``.

    Args:
        device_code: ONC hydrophone device code.
        event_times: Iterable of event timestamps.
        spectrograms_per_request: Number of windows per request.
        tag: Output folder tag for this run.
        download_audio: If True, download matching audio.
        download_flac: Legacy alias for ``download_audio``.
        audio_download_workers: Override parallelism for audio downloads.
        data_product_options: Optional ONC ``dpo_*`` overrides for HSD requests.
        **download_options: Extra controls forwarded to ``download_spectrogram_windows``.

    Returns:
        Summary dict with counts, output paths, and timing information.

    Raises:
        ValueError: If event_times is empty or inputs are invalid.
    """
    if spectrograms_per_request <= 0:
        raise ValueError("spectrograms_per_request must be positive")
    download_audio = _resolve_download_audio(download_audio, download_flac)
    times = list(event_times)
    if not times:
        raise ValueError("event_times must not be empty")

    windows = self._build_event_audio_windows(times)
    if not windows:
        raise ValueError("No windows derived from event_times")

    unique_windows = {}
    for start, end in windows:
        existing = unique_windows.get(start)
        if existing is None or end > existing:
            unique_windows[start] = end

    ordered = sorted(unique_windows.items(), key=lambda pair: pair[0])
    ordered_windows = [(start, end) for start, end in ordered]

    request_windows: List[Tuple[datetime, datetime]] = []
    current_group: List[Tuple[datetime, datetime]] = []

    def flush_group(group: List[Tuple[datetime, datetime]]) -> None:
        for i in range(0, len(group), spectrograms_per_request):
            chunk = group[i:i + spectrograms_per_request]
            request_windows.append((chunk[0][0], chunk[-1][1]))

    for start, end in ordered_windows:
        if not current_group:
            current_group = [(start, end)]
            continue
        if start == current_group[-1][1]:
            current_group.append((start, end))
        else:
            flush_group(current_group)
            current_group = [(start, end)]

    if current_group:
        flush_group(current_group)

    return self.download_spectrogram_windows(
        device_code,
        request_windows,
        spectrograms_per_request=spectrograms_per_request,
        tag=tag,
        download_audio=download_audio,
        audio_download_workers=audio_download_workers,
        data_product_options=data_product_options,
        **download_options,
    )
