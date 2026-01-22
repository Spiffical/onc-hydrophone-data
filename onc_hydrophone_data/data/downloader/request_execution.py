import json
from datetime import timedelta
from typing import Any, Dict, List, Optional

import numpy as np

from ...onc.common import ensure_timezone_aware
from .audio_backends import sf, torch, torchaudio
from .constants import FIVE_MINUTES_SECONDS
from .models import TimestampRequest


def _execute_timestamp_request(
    self,
    req: TimestampRequest,
    *,
    audio_download_workers: Optional[int] = None,
) -> Dict[str, Any]:
    """Download files for a timestamp-centric request and optionally clip outputs."""
    base_timestamp = ensure_timezone_aware(req.timestamp)
    pad_before = float(req.pad_before or 0)
    pad_after = float(req.pad_after or 0)
    clip_start = base_timestamp - timedelta(seconds=pad_before)
    clip_end = base_timestamp + timedelta(seconds=pad_after)
    if clip_end <= clip_start:
        clip_end = clip_start + timedelta(seconds=1)
    use_clip_window = req.clip_outputs or pad_before > 0 or pad_after > 0
    coverage_start = clip_start if use_clip_window else base_timestamp
    coverage_end = clip_end if use_clip_window else coverage_start + timedelta(seconds=1)
    windows = self._build_request_windows(coverage_start, coverage_end)
    coverage_start = windows[0][0]
    coverage_end = windows[-1][1]
    duration_seconds = int((len(windows) or 1) * FIVE_MINUTES_SECONDS) - 1
    self.setup_directories(
        req.spectrogram_format,
        req.device_code,
        req.tag,
        coverage_start,
        coverage_end,
        duration_seconds=duration_seconds,
    )

    summary: Dict[str, Any] = {
        'deviceCode': req.device_code,
        'timestamp': req.timestamp.isoformat(),
        'tag': req.tag,
        'clip': req.clip_outputs,
        'spectrogram': None,
        'audio': None,
    }

    if req.want_spectrogram:
        self.download_MAT_or_PNG(
            req.device_code,
            coverage_start,
            filetype=req.spectrogram_format,
            spectrograms_per_batch=len(windows),
            download_audio=False,
            spectral_downsample=req.spectral_downsample,
            data_product_options=req.data_product_options,
        )
        spec_files = self._collect_files_for_range(
            req.device_code,
            coverage_start,
            coverage_end,
            req.spectrogram_format,
        )
        clip_path = None
        if req.clip_outputs:
            segments: List[np.ndarray] = []
            freq_axis: Optional[np.ndarray] = None
            seconds_per_col: Optional[float] = None
            for file_start, path in spec_files:
                chunk_start = max(clip_start, file_start)
                chunk_end = min(clip_end, file_start + timedelta(seconds=FIVE_MINUTES_SECONDS))
                if chunk_end <= chunk_start:
                    continue
                chunk, freq_axis_candidate, secs = self._load_spectrogram_chunk(path, file_start, chunk_start, chunk_end)
                if chunk.size == 0:
                    continue
                segments.append(chunk)
                if freq_axis is None:
                    freq_axis = freq_axis_candidate
                if seconds_per_col is None:
                    seconds_per_col = secs
            if segments and seconds_per_col is not None:
                clip_matrix = np.concatenate(segments, axis=1)
                clip_path = self._write_spectrogram_clip(
                    req,
                    clip_matrix,
                    freq_axis,
                    seconds_per_col,
                    clip_start,
                    clip_end,
                )
            else:
                self.logger.warning(f"No spectrogram data overlapped requested clip for {req.device_code}")
        summary['spectrogram'] = {
            'files': [path for _, path in spec_files],
            'windows': len(windows),
            'clip_path': clip_path,
        }

    if req.want_audio:
        start_str = self._format_iso_utc(coverage_start)
        end_str = self._format_iso_utc(coverage_end)
        self.download_flac_files(
            req.device_code,
            start_str,
            end_str,
            max_download_workers=audio_download_workers,
        )
        audio_files = self._collect_files_for_range(
            req.device_code,
            coverage_start,
            coverage_end,
            req.audio_extension,
            search_dirs=[self.flac_path],
        )
        audio_clip_path = None
        has_audio_writer = torchaudio is not None or sf is not None
        if req.clip_outputs and torch is not None and has_audio_writer:
            chunks: List['torch.Tensor'] = []
            sample_rate: Optional[int] = None
            for file_start, path in audio_files:
                chunk_start = max(clip_start, file_start)
                chunk_end = min(clip_end, file_start + timedelta(seconds=FIVE_MINUTES_SECONDS))
                if chunk_end <= chunk_start:
                    continue
                waveform, sr = self._load_audio_chunk(path, file_start, chunk_start, chunk_end)
                if waveform is None or sr is None:
                    continue
                if sample_rate is None:
                    sample_rate = sr
                elif sr != sample_rate:
                    self.logger.warning(f"Sample rate mismatch for {path}; skipping chunk")
                    continue
                chunks.append(waveform)
            if chunks and sample_rate is not None:
                clip_waveform = torch.cat(chunks, dim=1)
                audio_clip_path = self._write_audio_clip(
                    req,
                    clip_waveform,
                    sample_rate,
                    clip_start,
                    clip_end,
                    suffix='audio',
                )
            elif not chunks:
                self.logger.warning(f"No audio overlap for {req.device_code} request at {req.timestamp.isoformat()}")
        elif req.clip_outputs and (torch is None or not has_audio_writer):
            self.logger.warning("Audio clip export requires torch and an audio writer (torchaudio/soundfile)")
        summary['audio'] = {
            'files': [path for _, path in audio_files],
            'clip_path': audio_clip_path,
        }

    return summary
    

def download_requests_from_json(
    self,
    json_path: str,
    *,
    default_pad_seconds: float = 0.0,
    default_tag: str = 'timestamp_requests',
    clip_outputs: Optional[bool] = None,
    spectrogram_format: str = 'mat',
    download_audio: Optional[bool] = None,
    download_spectrogram: Optional[bool] = None,
    audio_download_workers: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Execute timestamp-centric downloads defined in a JSON file.

    Args:
        json_path: Path to the JSON request file.
        default_pad_seconds: Default symmetric padding if not provided per request.
        default_tag: Output folder tag if not provided per request.
        clip_outputs: Force clip behavior (overrides request values).
        spectrogram_format: ``mat`` or ``png``.
        download_audio: Force audio download for all requests.
        download_spectrogram: Force spectrogram download for all requests.
        audio_download_workers: Override parallel audio download workers for this call.

    Returns:
        List of per‑request summary dicts (files, clip paths, windows).

    Notes:
        The JSON may follow the legacy schema ``{device: [[Y, M, D, H, M, S], ...]}``
        or the richer schema::

            {
              \"defaults\": {...},
              \"requests\": [
                 {
                    \"deviceCode\": \"ICLISTENHF6324\",
                    \"timestamp\": \"2024-04-01T04:25:00Z\",
                    \"pad_seconds\": 60,
                    \"download_audio\": true,
                    \"download_spectrogram\": true,
                    \"data_product_options\": {\"dpo_spectralDataDownsample\": 2}
                 }
              ]
            }
    """
    with open(json_path, 'r') as f:
        payload = json.load(f)
    requests = self._coerce_timestamp_requests(
        payload,
        default_pad_seconds=default_pad_seconds,
        default_tag=default_tag,
        clip_outputs=clip_outputs,
        spectrogram_format=spectrogram_format,
        download_audio=download_audio,
        download_spectrogram=download_spectrogram,
    )
    summaries = []
    for request in requests:
        summaries.append(
            self._execute_timestamp_request(
                request,
                audio_download_workers=audio_download_workers,
            )
        )
    return summaries


def download_requests_from_csv(
    self,
    csv_path: str,
    *,
    default_pad_seconds: float = 0.0,
    default_tag: str = 'timestamp_requests',
    clip_outputs: Optional[bool] = None,
    spectrogram_format: str = 'mat',
    download_audio: Optional[bool] = None,
    download_spectrogram: Optional[bool] = None,
    audio_download_workers: Optional[int] = None,
    defaults: Optional[Dict[str, Any]] = None,
    data_product_options_key: str = 'data_product_options',
) -> List[Dict[str, Any]]:
    """Execute timestamp-centric downloads defined in a CSV file.

    Args:
        csv_path: Path to the CSV request file.
        default_pad_seconds: Default symmetric padding if not provided per row.
        default_tag: Output folder tag if not provided per row.
        clip_outputs: Force clip behavior (overrides row values).
        spectrogram_format: ``mat`` or ``png``.
        download_audio: Force audio download for all rows.
        download_spectrogram: Force spectrogram download for all rows.
        audio_download_workers: Override parallel audio download workers for this call.
        defaults: Optional defaults dict to apply to every row.
        data_product_options_key: CSV column name containing JSON for ``dpo_*`` options.

    Returns:
        List of per‑request summary dicts (files, clip paths, windows).
    """
    payload = self._load_request_payload_from_csv(
        csv_path,
        defaults=defaults,
        data_product_options_key=data_product_options_key,
    )
    requests = self._coerce_timestamp_requests(
        payload,
        default_pad_seconds=default_pad_seconds,
        default_tag=default_tag,
        clip_outputs=clip_outputs,
        spectrogram_format=spectrogram_format,
        download_audio=download_audio,
        download_spectrogram=download_spectrogram,
    )
    summaries = []
    for request in requests:
        summaries.append(
            self._execute_timestamp_request(
                request,
                audio_download_workers=audio_download_workers,
            )
        )
    return summaries
