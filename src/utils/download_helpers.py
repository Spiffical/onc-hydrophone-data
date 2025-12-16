import math
from copy import deepcopy
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

from src.data.hydrophone_downloader import HydrophoneDownloader


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
    download_flac: bool = False,
    parallel_config: Optional[Dict[str, float]] = None,
    **overrides,
):
    """Canonical shim around HydrophoneDownloader.run_parallel_windows."""
    windows = _extract_windows(device_code, request_windows)
    if not windows:
        raise ValueError(f"No windows defined for {device_code}")

    config = {**DEFAULT_PARALLEL_CONFIG}
    if parallel_config:
        config.update(parallel_config)
    config.update(overrides)
    spectral_downsample = config.pop('spectral_downsample', None)

    return downloader.run_parallel_windows(
        device_code,
        windows,
        spectrograms_per_request=spectrograms_per_request,
        tag=tag,
        download_flac=download_flac,
        spectral_downsample=spectral_downsample,
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


HSD_BASE_FILTERS = {
    'dataProductCode': 'HSD',
    'dpo_hydrophoneDataDiversionMode': 'OD',
    'dpo_spectralDataDownsample': 2,
    'extension': 'mat',
}


def build_hsd_filters(
    device_code: str,
    start: datetime,
    end: datetime,
    *,
    downsample: int = 2,
    window_sec: float = 1.0,
    overlap: float = 0.5,
) -> Dict[str, Union[str, int, float]]:
    """Return a filter dict for plotRes-level HSD pulls."""
    if end <= start:
        raise ValueError("end must be after start")

    def fmt(dt: datetime) -> str:
        return dt.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'

    filters = deepcopy(HSD_BASE_FILTERS)
    filters.update({
        'deviceCode': device_code,
        'dateFrom': fmt(start),
        'dateTo': fmt(end),
        'dpo_spectralDataDownsample': downsample,
        'dpo_spectrogramWindowLengthSec': window_sec,
        'dpo_spectrogramOverlap': overlap,
    })
    return filters


