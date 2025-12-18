import os
import json
import math
import re
import numpy as np
import datetime as dt
from datetime import date, datetime, timedelta, timezone
from dataclasses import dataclass
from onc.onc import ONC
import random
from .trim_image import crop_image
from .segment import segment2
from .deployment_checker import HydrophoneDeploymentChecker
from PIL import Image
import glob
import shutil
import scipy.io
import concurrent.futures
from typing import List, Dict, Any, Tuple, Optional
import time
import logging
from threading import Lock
import requests

try:
    import torchaudio  # type: ignore
except Exception:
    torchaudio = None  # type: ignore

try:
    import torch  # type: ignore
except Exception:
    torch = None  # type: ignore
from ..onc.common import start_and_end_strings, format_iso_utc, ensure_timezone_aware
from .onc_requests import ONCRequestManager

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


FIVE_MINUTES_SECONDS = 300
FILENAME_TS_PATTERN = re.compile(r'(\d{8}T\d{6})')


@dataclass
class TimestampRequest:
    device_code: str
    timestamp: datetime
    pad_before: float = 0.0
    pad_after: float = 0.0
    want_spectrogram: bool = True
    want_audio: bool = False
    clip_outputs: bool = False
    tag: str = 'timestamp_requests'
    spectrogram_format: str = 'mat'
    audio_extension: str = 'flac'
    spectral_downsample: Optional[int] = None
    description: Optional[str] = None
    output_name: Optional[str] = None


class PrintLogger:
    def info(self, msg, *args, **kwargs):
        print(msg % args if args else msg)
    def warning(self, msg, *args, **kwargs):
        print('WARNING:', msg % args if args else msg)
    def error(self, msg, *args, **kwargs):
        print('ERROR:', msg % args if args else msg)
    def debug(self, msg, *args, **kwargs):
        print('DEBUG:', msg % args if args else msg)


# Helper function to ensure datetime is timezone-aware
def ensure_timezone_aware(dt_obj, tz=timezone.utc):
    """Convert timezone-naive datetime to timezone-aware datetime."""
    if dt_obj.tzinfo is None:
        return dt_obj.replace(tzinfo=tz)
    return dt_obj

class HydrophoneDownloader:
    def __init__(
        self,
        ONC_token,
        parent_dir,
        use_logging: bool = True,
        *,
        spectral_downsample: int = 2,
        **kwargs,
    ):
        self.onc = ONC(ONC_token)
        self._onc_token = ONC_token  # Stored for thread-safe client creation
        self.parent_dir = parent_dir
        self.logger = logger if use_logging else PrintLogger()
        self.max_workers = 4
        
        # Initialize Request Manager
        self.request_manager = ONCRequestManager(ONC_token, parent_dir, self.logger)

        # Default spectral downsample for data product requests:
        # 0 = full Res
        # 1 = one-minute average
        # 2 = plot resolution (default)
        # 3 = 1 hour average
        # 4 = 1 day average
        self.spectral_downsample = spectral_downsample

        # Legacy/Flat structure paths (defaults)
        self.input_path = os.path.join(self.parent_dir, 'mat', '')
        self.processed_path = os.path.join(self.input_path, 'processed', '')
        self.anom_path = os.path.join(self.input_path, 'rejects', '')
        self.flac_path = os.path.join(self.parent_dir, 'flac', '')
        
        # Lock for thread-safe path modification
        self._path_lock = Lock()

        # Build deployment checker
        from .deployment_checker import DeploymentChecker
        self.deployment_checker = DeploymentChecker(self._onc_token)
    
    def set_spectral_downsample(self, value: int) -> None:
        """Override default downsample option (0=fullRes, 1=one-minute, 2=plotRes, etc.)."""
        self.spectral_downsample = value
    
    @staticmethod
    def _parse_timestamp_value(value: Any) -> datetime:
        """Normalize various timestamp formats to timezone-aware UTC datetimes."""
        if isinstance(value, datetime):
            return ensure_timezone_aware(value)
        if isinstance(value, str):
            cleaned = value.strip()
            if cleaned.endswith('Z'):
                cleaned = cleaned[:-1] + '+00:00'
            try:
                dt_obj = datetime.fromisoformat(cleaned)
            except ValueError:
                dt_obj = datetime.strptime(value, '%Y-%m-%d %H:%M:%S')
            return ensure_timezone_aware(dt_obj)
        if isinstance(value, (list, tuple)) and len(value) >= 6:
            dt_obj = datetime(
                int(value[0]), int(value[1]), int(value[2]),
                int(value[3]), int(value[4]), int(value[5])
            )
            return ensure_timezone_aware(dt_obj)
        raise ValueError(f"Unsupported timestamp format: {value}")

    @staticmethod
    def _floor_to_window(dt_obj: datetime, seconds: int = FIVE_MINUTES_SECONDS) -> datetime:
        epoch = ensure_timezone_aware(dt_obj).timestamp()
        floored = math.floor(epoch / seconds) * seconds
        return datetime.fromtimestamp(floored, tz=timezone.utc)

    @staticmethod
    def _ceil_to_window(dt_obj: datetime, seconds: int = FIVE_MINUTES_SECONDS) -> datetime:
        epoch = ensure_timezone_aware(dt_obj).timestamp()
        ceiled = math.ceil(epoch / seconds) * seconds
        return datetime.fromtimestamp(ceiled, tz=timezone.utc)

    @staticmethod
    def _timestamp_from_filename(path: str) -> Optional[datetime]:
        """Extract start timestamp from standard ONC filename."""
        match = FILENAME_TS_PATTERN.search(os.path.basename(path))
        if not match:
            return None
        try:
            ts = datetime.strptime(match.group(1), '%Y%m%dT%H%M%S')
            return ts.replace(tzinfo=timezone.utc)
        except ValueError:
            return None

    def _collect_files_for_range(
        self,
        device_code: str,
        start_dt: datetime,
        end_dt: datetime,
        extension: str,
        search_dirs: Optional[List[str]] = None,
    ) -> List[Tuple[datetime, str]]:
        """Return sorted list of (start_time, path) overlapping the requested range."""
        dirs = search_dirs or [self.processed_path, self.input_path, self.anom_path]
        matches: List[Tuple[datetime, str]] = []
        for base in dirs:
            if not base or not os.path.isdir(base):
                continue
            pattern = os.path.join(base, f"{device_code}_*.{extension}")
            for path in glob.glob(pattern):
                ts = self._timestamp_from_filename(path)
                if not ts:
                    continue
                file_end = ts + timedelta(seconds=FIVE_MINUTES_SECONDS)
                if file_end <= start_dt or ts >= end_dt:
                    continue
                matches.append((ts, path))
        matches.sort(key=lambda pair: pair[0])
        return matches
    
    @staticmethod
    def _extract_frequency_axis(spect_struct: Any) -> Optional[np.ndarray]:
        """Attempt to pull frequency axis from SpectData struct."""
        if not hasattr(spect_struct, 'dtype') or not spect_struct.dtype.names:
            return None
        for key in ('freq', 'freqs', 'frequency', 'Frequency', 'freqHz', 'f'):
            if key in spect_struct.dtype.names:
                try:
                    arr = spect_struct[key]
                    if isinstance(arr, np.ndarray):
                        return np.squeeze(arr)
                except Exception:
                    continue
        return None

    def _load_spectrogram_chunk(
        self,
        file_path: str,
        file_start: datetime,
        clip_start: datetime,
        clip_end: datetime,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], float]:
        """Load the portion of a MAT file that overlaps the clip range."""
        try:
            mat_data = scipy.io.loadmat(file_path)
        except Exception as exc:
            self.logger.warning(f"Failed to read {file_path}: {exc}")
            return np.empty((0, 0)), None, float(FIVE_MINUTES_SECONDS)
        spect = mat_data.get('SpectData')
        if spect is None:
            return np.empty((0, 0)), None, float(FIVE_MINUTES_SECONDS)
        # ONC MAT structs are 1x1 arrays of custom dtype
        try:
            spect_struct = spect[0, 0]
            psd = spect_struct['PSD'][0, 0]
        except Exception:
            return np.empty((0, 0)), None, float(FIVE_MINUTES_SECONDS)
        total_cols = psd.shape[1]
        seconds_per_col = FIVE_MINUTES_SECONDS / max(1, total_cols)
        # Determine column indices
        start_offset = max(0.0, (clip_start - file_start).total_seconds())
        end_offset = max(0.0, (clip_end - file_start).total_seconds())
        start_idx = int(max(0, math.floor(start_offset / seconds_per_col)))
        end_idx = int(min(total_cols, math.ceil(end_offset / seconds_per_col)))
        if end_idx <= start_idx:
            return np.empty((psd.shape[0], 0)), None, seconds_per_col
        freq_axis = self._extract_frequency_axis(spect_struct)
        return psd[:, start_idx:end_idx], freq_axis, seconds_per_col

    def _clip_basename(self, req: TimestampRequest, suffix: str) -> str:
        base = req.output_name or f"{req.device_code}_{req.timestamp.strftime('%Y%m%dT%H%M%S')}"
        if req.clip_outputs:
            base = f"{base}_{suffix}_{int(req.pad_before)}s_{int(req.pad_after)}s"
        else:
            base = f"{base}_{suffix}"
        return base

    def _write_spectrogram_clip(
        self,
        req: TimestampRequest,
        clip_data: np.ndarray,
        freq_axis: Optional[np.ndarray],
        seconds_per_col: float,
        clip_start: datetime,
        clip_end: datetime,
    ) -> str:
        clip_dir = os.path.join(self.parent_dir, req.device_code, req.tag, 'clips', 'spectrograms')
        os.makedirs(clip_dir, exist_ok=True)
        out_name = self._clip_basename(req, 'spec')
        out_path = os.path.join(clip_dir, f"{out_name}.npz")
        np.savez_compressed(
            out_path,
            spectrogram=clip_data,
            frequency=freq_axis,
            seconds_per_column=seconds_per_col,
            clip_start=clip_start.isoformat(),
            clip_end=clip_end.isoformat(),
            device=req.device_code,
            description=req.description,
        )
        return out_path

    def _load_audio_chunk(
        self,
        file_path: str,
        file_start: datetime,
        clip_start: datetime,
        clip_end: datetime,
    ) -> Tuple[Optional['torch.Tensor'], Optional[int]]:
        if torchaudio is None or torch is None:
            self.logger.warning("torchaudio/torch not available; skipping audio clipping")
            return None, None
        try:
            waveform, sample_rate = torchaudio.load(file_path)
        except Exception as exc:
            self.logger.warning(f"Failed to read audio {file_path}: {exc}")
            return None, None
        total_samples = waveform.shape[1]
        start_offset = max(0.0, (clip_start - file_start).total_seconds())
        end_offset = max(0.0, (clip_end - file_start).total_seconds())
        start_sample = int(max(0, math.floor(start_offset * sample_rate)))
        end_sample = int(min(total_samples, math.ceil(end_offset * sample_rate)))
        if end_sample <= start_sample:
            return None, sample_rate
        return waveform[:, start_sample:end_sample], sample_rate

    def _write_audio_clip(
        self,
        req: TimestampRequest,
        waveform: 'torch.Tensor',
        sample_rate: int,
        clip_start: datetime,
        clip_end: datetime,
    ) -> str:
        clip_dir = os.path.join(self.parent_dir, req.device_code, req.tag, 'clips', 'audio')
        os.makedirs(clip_dir, exist_ok=True)
        out_name = self._clip_basename(req, 'audio')
        out_path = os.path.join(clip_dir, f"{out_name}.flac")
        torchaudio.save(out_path, waveform, sample_rate, format='FLAC')
        meta = {
            'device': req.device_code,
            'clip_start': clip_start.isoformat(),
            'clip_end': clip_end.isoformat(),
            'sample_rate': sample_rate,
            'description': req.description,
        }
        with open(os.path.join(clip_dir, f"{out_name}.json"), 'w') as jf:
            json.dump(meta, jf, indent=2)
        return out_path

    def _build_request_windows(self, start_dt: datetime, end_dt: datetime) -> List[Tuple[datetime, datetime]]:
        """Build contiguous five-minute windows that cover [start_dt, end_dt)."""
        start_dt = ensure_timezone_aware(start_dt)
        end_dt = ensure_timezone_aware(end_dt)
        floor_start = self._floor_to_window(start_dt)
        ceil_end = self._ceil_to_window(end_dt)
        if ceil_end <= floor_start:
            ceil_end = floor_start + timedelta(seconds=FIVE_MINUTES_SECONDS)
        windows: List[Tuple[datetime, datetime]] = []
        cursor = floor_start
        while cursor < ceil_end:
            windows.append((cursor, cursor + timedelta(seconds=FIVE_MINUTES_SECONDS)))
            cursor += timedelta(seconds=FIVE_MINUTES_SECONDS)
        if not windows:
            windows.append((floor_start, floor_start + timedelta(seconds=FIVE_MINUTES_SECONDS)))
        return windows

    def _build_request_from_dict(
        self,
        data: Dict[str, Any],
        *,
        defaults: Dict[str, Any],
        default_pad_seconds: float,
        default_tag: str,
        clip_outputs: Optional[bool],
        spectrogram_format: str,
        download_audio: Optional[bool],
        download_spectrogram: Optional[bool],
    ) -> TimestampRequest:
        base = {**defaults, **(data or {})}
        device_code = base.get('deviceCode') or base.get('device')
        if not device_code:
            raise ValueError("Each request requires a deviceCode/device field")
        start_value = (
            base.get('start')
            or base.get('start_time')
            or base.get('begin')
            or base.get('begin_time')
        )
        end_value = (
            base.get('end')
            or base.get('end_time')
            or base.get('stop')
            or base.get('stop_time')
        )
        timestamp_value = (
            base.get('timestamp')
            or base.get('time')
            or base.get('datetime')
            or start_value
        )
        if timestamp_value is None:
            raise ValueError(f"Missing timestamp/start for device {device_code}")
        ts = self._parse_timestamp_value(timestamp_value)
        clip_range_start = self._parse_timestamp_value(start_value) if start_value else ts
        clip_range_end: Optional[datetime] = None
        span_seconds = 0.0
        if end_value:
            clip_range_end = self._parse_timestamp_value(end_value)
            if clip_range_end <= clip_range_start:
                raise ValueError("end time must be after start time in request")
            span_seconds = (clip_range_end - clip_range_start).total_seconds()
        duration_seconds = base.get('duration_seconds')
        if duration_seconds and not span_seconds:
            try:
                span_seconds = max(span_seconds, float(duration_seconds))
            except (TypeError, ValueError):
                pass
        if start_value:
            ts = clip_range_start
        sym_pad = base.get('pad_seconds', default_pad_seconds)
        sym_pad_value = float(sym_pad) if sym_pad is not None else 0.0
        pad_before = float(base.get('pad_before_seconds', sym_pad_value))
        pad_after = float(base.get('pad_after_seconds', sym_pad_value))
        if span_seconds > 0:
            pad_after = max(pad_after, span_seconds)
        want_spec = base.get('download_spectrogram')
        if want_spec is None:
            want_spec = base.get('spectrogram')
        if want_spec is None:
            want_spec = download_spectrogram if download_spectrogram is not None else True
        want_audio = base.get('download_audio')
        if want_audio is None:
            want_audio = base.get('audio')
        if want_audio is None:
            want_audio = download_audio if download_audio is not None else False
        clip_flag = base.get('clip')
        if clip_flag is None:
            if clip_outputs is not None:
                clip_flag = clip_outputs
            else:
                clip_flag = (pad_before > 0) or (pad_after > 0)
        tag = base.get('output_tag') or default_tag
        spectral_downsample = base.get('spectral_downsample')
        req = TimestampRequest(
            device_code=device_code,
            timestamp=ts,
            pad_before=pad_before,
            pad_after=pad_after,
            want_spectrogram=bool(want_spec),
            want_audio=bool(want_audio),
            clip_outputs=bool(clip_flag),
            tag=tag,
            spectrogram_format=base.get('spectrogram_format', spectrogram_format),
            audio_extension=base.get('audio_extension', 'flac'),
            spectral_downsample=spectral_downsample,
            description=base.get('label') or base.get('description'),
            output_name=base.get('output_name'),
        )
        return req

    def _coerce_timestamp_requests(
        self,
        payload: Any,
        *,
        default_pad_seconds: float,
        default_tag: str,
        clip_outputs: Optional[bool],
        spectrogram_format: str,
        download_audio: Optional[bool],
        download_spectrogram: Optional[bool],
    ) -> List[TimestampRequest]:
        """Normalize any payload structure to a list of TimestampRequest objects."""
        requests: List[TimestampRequest] = []
        if isinstance(payload, dict) and 'requests' in payload:
            defaults = payload.get('defaults', {})
            for entry in payload.get('requests', []):
                requests.append(
                    self._build_request_from_dict(
                        entry,
                        defaults=defaults,
                        default_pad_seconds=default_pad_seconds,
                        default_tag=default_tag,
                        clip_outputs=clip_outputs,
                        spectrogram_format=spectrogram_format,
                        download_audio=download_audio,
                        download_spectrogram=download_spectrogram,
                    )
                )
            return requests

        if isinstance(payload, dict):
            for device_code, entries in payload.items():
                if not isinstance(entries, list):
                    raise ValueError("Legacy device mapping must map to a list of timestamps")
                for entry in entries:
                    if isinstance(entry, dict):
                        entry = {**entry, 'device': device_code}
                        requests.append(
                            self._build_request_from_dict(
                                entry,
                                defaults={},
                                default_pad_seconds=default_pad_seconds,
                                default_tag=default_tag,
                                clip_outputs=clip_outputs,
                                spectrogram_format=spectrogram_format,
                                download_audio=download_audio,
                                download_spectrogram=download_spectrogram,
                            )
                        )
                    else:
                        ts = self._parse_timestamp_value(entry)
                        requests.append(
                            TimestampRequest(
                                device_code=device_code,
                                timestamp=ts,
                                pad_before=float(default_pad_seconds or 0),
                                pad_after=float(default_pad_seconds or 0),
                                want_spectrogram=download_spectrogram if download_spectrogram is not None else True,
                                want_audio=download_audio if download_audio is not None else False,
                                clip_outputs=bool(clip_outputs if clip_outputs is not None else (default_pad_seconds > 0)),
                                tag=default_tag,
                                spectrogram_format=spectrogram_format,
                            )
                        )
            return requests

        if isinstance(payload, list):
            for entry in payload:
                requests.extend(
                    self._coerce_timestamp_requests(
                        entry,
                        default_pad_seconds=default_pad_seconds,
                        default_tag=default_tag,
                        clip_outputs=clip_outputs,
                        spectrogram_format=spectrogram_format,
                        download_audio=download_audio,
                        download_spectrogram=download_spectrogram,
                    )
                )
            return requests

        raise ValueError("Unsupported request configuration structure")

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
    ) -> List[Dict[str, Any]]:
        """
        Execute timestamp-centric downloads defined in a JSON file.

        The JSON may follow the legacy schema ({device: [[Y, M, D, H, M, S], ...]})
        or the richer schema:
        {
          "defaults": {...},
          "requests": [
             {
                "deviceCode": "ICLISTENHF6324",
                "timestamp": "2024-04-01T04:25:00Z",
                "pad_seconds": 60,
                "download_audio": true,
                "download_spectrogram": true
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
            summaries.append(self._execute_timestamp_request(request))
        return summaries

    def _execute_timestamp_request(self, req: TimestampRequest) -> Dict[str, Any]:
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
                download_flac=False,
                spectral_downsample=req.spectral_downsample,
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
                    freq_axis = freq_axis or freq_axis_candidate
                    seconds_per_col = seconds_per_col or secs
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
            self.download_flac_files(req.device_code, start_str, end_str)
            audio_files = self._collect_files_for_range(
                req.device_code,
                coverage_start,
                coverage_end,
                req.audio_extension,
                search_dirs=[self.flac_path],
            )
            audio_clip_path = None
            if req.clip_outputs and torchaudio is not None and torch is not None:
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
                    audio_clip_path = self._write_audio_clip(req, clip_waveform, sample_rate, clip_start, clip_end)
                elif not chunks:
                    self.logger.warning(f"No audio overlap for {req.device_code} request at {req.timestamp.isoformat()}")
            elif req.clip_outputs and (torchaudio is None or torch is None):
                self.logger.warning("torchaudio not available; skipping audio clip export")
            summary['audio'] = {
                'files': [path for _, path in audio_files],
                'clip_path': audio_clip_path,
            }

        return summary
        
    def setup_directories(self, filetype, device_code=None, download_method=None, start_date=None, end_date=None, duration_seconds=None):
        """Setup directory structure with optional device, method, and date organization"""
        if device_code and download_method:
            # Create method folder name with date information
            method_folder = self._create_method_folder_name(download_method, start_date, end_date, duration_seconds)
            
            # New organized structure: data/DEVICE/METHOD_DATES/
            base_path = os.path.join(self.parent_dir, device_code, method_folder)
            
            # Create mat directory with processed and rejects subdirectories
            self.input_path = os.path.join(base_path, filetype, '')
            self.processed_path = os.path.join(self.input_path, 'processed', '')
            self.anom_path = os.path.join(self.input_path, 'rejects', '')
            
            # Create flac directory at the same level as mat
            self.flac_path = os.path.join(base_path, 'flac', '')
        else:
            # Legacy structure for backwards compatibility
            self.input_path = os.path.join(self.parent_dir, filetype, '')
            self.processed_path = os.path.join(self.input_path, 'processed', '')
            self.anom_path = os.path.join(self.input_path, 'rejects', '')
            self.flac_path = os.path.join(self.parent_dir, 'flac', '')
        
        self.onc.outPath = self.input_path

        # Create all necessary directories
        for folder_path in [self.parent_dir, self.input_path, self.processed_path, self.anom_path, self.flac_path]:
            os.makedirs(folder_path, exist_ok=True)
    
    def _create_method_folder_name(self, download_method, start_date=None, end_date=None, duration_seconds=None):
        """Create a descriptive folder name including method and dates"""
        folder_name = download_method
        
        # Add date range information
        if start_date:
            if isinstance(start_date, (list, tuple)):
                # Handle tuple format (year, month, day)
                start_str = f"{start_date[0]}-{start_date[1]:02d}-{start_date[2]:02d}"
            elif hasattr(start_date, 'strftime'):
                # Handle datetime object
                start_str = start_date.strftime('%Y-%m-%d')
            else:
                start_str = str(start_date)
            
            folder_name += f"_{start_str}"
            
            if end_date:
                if isinstance(end_date, (list, tuple)):
                    # Handle tuple format (year, month, day)
                    end_str = f"{end_date[0]}-{end_date[1]:02d}-{end_date[2]:02d}"
                elif hasattr(end_date, 'strftime'):
                    # Handle datetime object
                    end_str = end_date.strftime('%Y-%m-%d')
                else:
                    end_str = str(end_date)
                
                folder_name += f"_to_{end_str}"
        
        return folder_name



    def try_download_run(
        self,
        rec: Dict[str, Any],
        allow_rerun: bool = True,
        download_flac: bool = False,
        max_attempts: int = 6,
        wait_for_complete: bool = False,
        poll_interval_seconds: int = 15,
        max_wait_seconds: int = 900,
    ) -> (str, Dict[str, Any]):
        """Attempt to download a previously submitted run. Returns (status, updated_rec).
        status: 'pending' | 'downloaded' | 'error'"""
        # Prepare a per-call ONC client to avoid cross-thread outPath races
        local_out_path = rec.get('outPath')
        if local_out_path:
            onc_client = ONC(self._onc_token, showInfo=False)
            onc_client.outPath = local_out_path
            # Ensure directories exist prior to download
            os.makedirs(local_out_path, exist_ok=True)
            os.makedirs(os.path.join(local_out_path, 'processed'), exist_ok=True)
            os.makedirs(os.path.join(local_out_path, 'rejects'), exist_ok=True)
        else:
            onc_client = self.onc

        # Ensure we have a runId
        run_id = None
        if isinstance(rec.get('runIds'), list) and rec['runIds']:
            run_id = rec['runIds'][0]
        if run_id is None and allow_rerun and rec.get('dpRequestId'):
            try:
                run_data = onc_client.runDataProduct(rec['dpRequestId'], waitComplete=False)
                if isinstance(run_data, dict) and run_data.get('runIds'):
                    rec['runIds'] = run_data['runIds']
                    run_id = rec['runIds'][0]
            except Exception as e:
                self.logger.debug(f"runDataProduct (no-wait) not ready for dpRequestId={rec.get('dpRequestId')}: {e}")

        if run_id is None:
            # Not ready yet
            rec['status'] = 'pending'
            return 'pending', rec

        should_wait = wait_for_complete and rec.get('dpRequestId') and not rec.get('readyAt')
        if should_wait:
            # Use request_manager for polling
            ready, reason, payload = self.request_manager.wait_for_data_product_ready(
                rec['dpRequestId'],
                max_wait_seconds=max_wait_seconds,
                poll_interval_seconds=poll_interval_seconds,
            )
            rec['latestStatus'] = payload
            if not ready:
                if reason in ('cancelled', 'error', 'failed'):
                    rec['status'] = 'error'
                    rec['error'] = f'Data product status={reason}'
                    rec['lastDownloadError'] = f'data product status={reason}'
                    return 'error', rec
                rec['status'] = 'pending'
                rec['pendingReason'] = reason
                rec['lastDownloadError'] = reason
                return 'pending', rec
            rec['readyAt'] = rec.get('readyAt') or format_iso_utc(datetime.now(timezone.utc))

        # Try download
        attempt_limit = max_attempts if (max_attempts or 0) > 0 else None
        if wait_for_complete:
            attempt_limit = None

        try:
            rec['attempts'] = rec.get('attempts', 0) + 1
            file_infos = onc_client.downloadDataProduct(
                run_id,
                maxRetries=10,
                downloadResultsOnly=False,
                includeMetadataFile=False,
                overwrite=True,
            )

            rec['lastDownloadError'] = None
            # Determine if any MAT files actually arrived
            mat_downloaded = False
            target_prefix = f"{rec['deviceCode']}_"
            target_suffix = rec.get('start')
            if target_suffix:
                target_suffix = target_suffix.replace(':', '').replace('-', '').replace('.', '')
            for info in file_infos or []:
                fname = info.get('file') or ''
                if fname.lower().endswith('.mat') and os.path.basename(fname).startswith(target_prefix):
                    mat_downloaded = True
                    break
            if not mat_downloaded:
                # Also check filesystem in case getInfo lacks names
                pattern = f"{rec['deviceCode']}_*.mat"
                mat_glob = glob.glob(os.path.join(local_out_path or self.input_path, pattern))
                mat_downloaded = bool(mat_glob)

            if not mat_downloaded:
                if attempt_limit is not None and rec['attempts'] >= attempt_limit:
                    rec['status'] = 'error'
                    rec['error'] = 'No MAT files downloaded after max attempts'
                    rec['lastDownloadError'] = 'No MAT files downloaded after max attempts'
                    return 'error', rec
                rec['status'] = 'pending'
                rec['pendingReason'] = 'waiting-for-mat-files'
                rec['lastDownloadError'] = 'waiting-for-mat-files'
                return 'pending', rec
            # Process downloaded files under a lock to avoid path interference
            if local_out_path:
                with self._path_lock:
                    self.input_path = local_out_path
                    self.processed_path = os.path.join(local_out_path, 'processed', '')
                    self.anom_path = os.path.join(local_out_path, 'rejects', '')
                    self.process_spectrograms('mat')
            else:
                self.process_spectrograms('mat')
            if download_flac and rec.get('start') and rec.get('end'):
                try:
                    flac_client = ONC(self._onc_token, showInfo=False)
                    flac_client.outPath = self.flac_path
                    self.download_flac_files(rec['deviceCode'], rec['start'], rec['end'], onc_client=flac_client)
                    rec['flac_status'] = 'downloaded'
                except Exception as e:
                    rec['flac_status'] = f'error: {e}'
                    self.logger.warning(f"FLAC download failed for runId={run_id}: {e}")
            rec['status'] = 'downloaded'
            rec['completedAt'] = format_iso_utc(datetime.now(timezone.utc))
            return 'downloaded', rec
        except requests.HTTPError as http_err:
            status_code = getattr(http_err.response, 'status_code', None)
            msg = f"http-{status_code}" if status_code else str(http_err)
            rec['lastDownloadError'] = msg
            self.logger.debug(f"download HTTP error for runId={run_id}: {http_err}")
            transient_codes = {500, 502, 503, 504}
            if status_code in transient_codes or (status_code is None and 'HTTP status 500' in str(http_err)):
                if attempt_limit is not None and rec.get('attempts', 0) >= attempt_limit:
                    rec['status'] = 'error'
                    rec['error'] = msg
                    return 'error', rec
                rec['status'] = 'pending'
                rec['pendingReason'] = msg
                return 'pending', rec
            if attempt_limit is not None and rec.get('attempts', 0) >= attempt_limit:
                rec['status'] = 'error'
                rec['error'] = msg
                return 'error', rec
            rec['status'] = 'pending'
            rec['pendingReason'] = msg
            return 'pending', rec
        except Exception as e:
            # Likely not yet ready; keep pending
            rec['lastDownloadError'] = str(e)
            self.logger.debug(f"download not ready for runId={run_id}: {e}")
            if attempt_limit is not None and rec.get('attempts', 0) >= attempt_limit:
                rec['status'] = 'error'
                rec['error'] = str(e)
                rec['lastDownloadError'] = str(e)
                return 'error', rec
            rec['status'] = 'pending'
            return 'pending', rec

    def run_parallel_windows(
        self,
        device_code: str,
        windows: List[Tuple[datetime, datetime]],
        *,
        spectrograms_per_request: int,
        tag: str = 'parallel',
        download_flac: bool = False,
        stagger_seconds: float = 3.0,
        max_wait_minutes: int = 45,
        poll_interval_seconds: int = 30,
        max_attempts: int = 6,
        max_download_workers: int = 4,
        spectral_downsample: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Fire off a batch of MAT runs, poll until they're ready, then download them."""
        if not windows:
            raise ValueError("No windows provided for parallel run")

        ordered_windows = sorted(windows, key=lambda pair: pair[0])
        range_start = ordered_windows[0][0]
        range_end = ordered_windows[-1][1]
        self.setup_directories('mat', device_code, tag, range_start, range_end)
        downsample = spectral_downsample if spectral_downsample is not None else self.spectral_downsample

        total_windows = len(ordered_windows)
        print(f"Submitting {total_windows} requests for {device_code}...")

        wall_start = time.time()
        run_records: List[Dict[str, Any]] = []
        for i, (start_dt, end_dt) in enumerate(ordered_windows, 1):
            if i % 5 == 1 or i == total_windows:
                 print(f"Submitting request {i}/{total_windows}...")
            
            rec = self.request_manager.submit_mat_run_no_wait(
                device_code=device_code,
                start_dt=start_dt,
                end_dt=end_dt,
                out_path=self.input_path,
                spectrograms_per_batch=spectrograms_per_request,
                spectral_downsample=downsample,
            )
            run_records.append(rec)
            if stagger_seconds > 0 and i < total_windows:
                time.sleep(stagger_seconds)

        def replace_record(updated: Dict[str, Any]) -> None:
            for idx, existing in enumerate(run_records):
                if existing.get('dpRequestId') == updated.get('dpRequestId'):
                    run_records[idx] = updated
                    return

        def pending_records() -> List[Dict[str, Any]]:
            return [r for r in run_records if r.get('status') not in ('downloaded', 'error')]

        total_wait_seconds = max_wait_minutes * 60 if max_wait_minutes > 0 else 0

        def attempt_batch(records: List[Dict[str, Any]], *, wait_for_complete: bool, allow_rerun: bool) -> None:
            """Download helper that optionally fans out via thread pool."""
            if max_download_workers and max_download_workers > 1:
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_download_workers) as executor:
                    future_map = {
                        executor.submit(
                            self.try_download_run,
                            rec,
                            allow_rerun=allow_rerun,
                            download_flac=download_flac,
                            max_attempts=max_attempts,
                            wait_for_complete=wait_for_complete,
                            poll_interval_seconds=poll_interval_seconds,
                            max_wait_seconds=(total_wait_seconds if wait_for_complete else 0),
                        ): rec
                        for rec in records
                    }
                    for future in concurrent.futures.as_completed(future_map):
                        base = future_map[future]
                        try:
                            status, updated = future.result()
                        except Exception as exc:
                            updated = {**base, 'status': 'error', 'error': str(exc), 'lastDownloadError': str(exc)}
                        replace_record(updated)
            else:
                for rec in records:
                    try:
                        status, updated = self.try_download_run(
                            rec,
                            allow_rerun=allow_rerun,
                            download_flac=download_flac,
                            max_attempts=max_attempts,
                            wait_for_complete=wait_for_complete,
                            poll_interval_seconds=poll_interval_seconds,
                            max_wait_seconds=(total_wait_seconds if wait_for_complete else 0),
                        )
                    except Exception as exc:
                        updated = {**rec, 'status': 'error', 'error': str(exc), 'lastDownloadError': str(exc)}
                    replace_record(updated)

        deadline = time.time() + (total_wait_seconds if total_wait_seconds > 0 else float('inf'))
        pass_counter = 0
        while pending_records() and time.time() < deadline:
            batch = pending_records()
            pass_counter += 1
            self.logger.info(f"Parallel poll pass {pass_counter}: {len(batch)} pending")
            attempt_batch(batch, wait_for_complete=False, allow_rerun=True)
            if pending_records():
                time.sleep(max(1, poll_interval_seconds))

        leftovers = [r for r in run_records if r.get('status') != 'downloaded']
        if leftovers:
            for rec in leftovers:
                if rec.get('dpRequestId'):
                    try:
                        run_info = self.onc.runDataProduct(rec['dpRequestId'], waitComplete=True)
                        if isinstance(run_info, dict) and run_info.get('runIds'):
                            rec['runIds'] = run_info['runIds']
                    except Exception as exc:
                        rec['status'] = 'error'
                        rec['error'] = str(exc)
                        rec['lastDownloadError'] = str(exc)
                        replace_record(rec)
                        continue
                status, updated = self.try_download_run(
                    rec,
                    allow_rerun=False,
                    download_flac=download_flac,
                    max_attempts=max_attempts,
                    wait_for_complete=False,
                )
                replace_record(updated)

        runs_downloaded = len([r for r in run_records if r.get('status') == 'downloaded'])
        runs_errors = len([r for r in run_records if r.get('status') == 'error'])
        processed_files = len(glob.glob(os.path.join(self.processed_path, '*.mat')))

        return {
            'device': device_code,
            'runs_total': len(run_records),
            'runs_downloaded': runs_downloaded,
            'runs_errors': runs_errors,
            'processed_mat': processed_files,
            'input_path': self.input_path,
            'flac_path': self.flac_path,
            'wall_seconds': time.time() - wall_start,
        }
    
    def filter_existing_requests(self, device_code, request_times, extension='mat'):
        """
        Skip requests that already have a matching file downloaded.
        Matching is done on the exact request start timestamp prefix, not just the day.
        """
        search_paths = [
            os.path.join(self.processed_path, f"{device_code}_*.{extension}"),
            os.path.join(self.input_path, f"{device_code}_*.{extension}"),
        ]
        existing_prefixes = set()
        for pattern in search_paths:
            for file in glob.glob(pattern):
                filename = os.path.basename(file)
                parts = filename.split('_')
                if len(parts) > 1:
                    prefix = f"{parts[0]}_{parts[1].split('.')[0]}"  # device_YYYYMMDDTHHMMSS
                    existing_prefixes.add(prefix)

        filtered = []
        for ts in request_times:
            if hasattr(ts, 'strftime'):
                prefix = f"{device_code}_{ts.strftime('%Y%m%dT%H%M%S')}"
                if prefix in existing_prefixes:
                    continue
            filtered.append(ts)
        skipped = len(request_times) - len(filtered)
        if skipped:
            self.logger.info(f"Skipping {skipped} already-downloaded requests based on timestamp match")
        return filtered

    def sampling_schedule(self, deviceCode, threshold_num, year, month, day, day_interval=None, num_days=None, spectrograms_per_batch=6):
        spect_length = 300
        sample_time_per_day = 1799
        min_per_day = (sample_time_per_day + 1) / spect_length

        start_date = date(year, month, day)
        if num_days is None:
            today = date.today()
            num_days = (today - start_date).days

        time_delta = dt.timedelta(num_days)
        start_time_str, end_time_str = start_and_end_strings(start_date, time_delta)

        filters = {
            'deviceCode': deviceCode,
            'dateFrom': start_time_str,
            'dateTo': end_time_str,
            'extension': 'png'
        }

        result = self.onc.getListByDevice(filters, allPages=True)
        result_files = result.get('files', []) if isinstance(result, dict) else []
        spect_png_files = [s for s in result_files if "Z-spect.png" in s]

        day_strings = [spect_png_file.split('_')[1] for spect_png_file in spect_png_files]
        days_int = [int(day_str[0:8]) for day_str in day_strings]
        unique_days = np.unique(days_int)
        num_days_available = len(unique_days)
        print(f'Number of days available: {num_days_available}')

        if num_days_available == 0:
            self.logger.warning("No spectrogram files found in the requested date range—nothing to sample.")
            return [], sample_time_per_day

        if day_interval == 1:
            sample_time_per_day = 86400 - 1
            num_per_day = 86400 / spect_length
        else:
            if day_interval is None:
                day_interval = num_days_available / (threshold_num * 1.1 / min_per_day)
                if day_interval > 1:
                    day_interval = int(np.round(day_interval))
                else:
                    day_interval = 1

            if len(np.arange(0, num_days_available, day_interval)) * min_per_day < threshold_num:
                num_per_day = int(np.ceil(threshold_num * 1.1 / len(np.arange(0, num_days_available, day_interval))))
                sample_time_per_day = spect_length * num_per_day - 1
            else:
                num_per_day = int(min_per_day)

        print(f'Plan is to retrieve {num_per_day} spectrograms per day')

        # Calculate how many requests we need (each request gets exactly spectrograms_per_batch spectrograms)
        total_requests_needed = int(np.ceil(threshold_num / spectrograms_per_batch))
        actual_spectrograms_to_download = total_requests_needed * spectrograms_per_batch
        
        print(f'Target: {threshold_num} spectrograms')
        print(f'Each request gets {spectrograms_per_batch} spectrograms')
        print(f'Therefore need {total_requests_needed} requests')
        print(f'This will download {actual_spectrograms_to_download} spectrograms total')
        
        # Generate sampling schedule - distribute requests evenly across the FULL requested time range
        date_list = []
        
        # Calculate how many days we'll sample from
        requests_per_day = max(1, int(np.ceil(total_requests_needed / min(total_requests_needed, num_days_available))))
        num_sampling_days = int(np.ceil(total_requests_needed / requests_per_day))
        
        print(f'Will make {requests_per_day} requests per day across {num_sampling_days} days')
        print(f'Sampling across full requested range of {num_days} days')
        
        for day_idx in range(num_sampling_days):
            if len(date_list) >= total_requests_needed:
                break
                
            # Calculate day offset - spread across the FULL requested date range (num_days)
            # This ensures we sample from start to end of the requested period
            if num_sampling_days > 1:
                day_offset = day_idx * (num_days - 1) // (num_sampling_days - 1)
            else:
                day_offset = 0
                
            # Ensure we don't exceed the requested date range
            if day_offset >= num_days:
                day_offset = num_days - 1
                
            # Calculate the actual date for this day offset
            sample_date = start_date + timedelta(days=day_offset)
            
            # Add the specified number of requests for this day
            for request_in_day in range(requests_per_day):
                if len(date_list) >= total_requests_needed:
                    break
                    
                # Distribute hours across the day for multiple requests
                # OR use random sampling for better temporal diversity
                if requests_per_day > 1:
                    # Multiple requests per day - distribute hours evenly within the day
                    hour_offset = request_in_day * (24 // requests_per_day)
                else:
                    # One request per day - use random hour for maximum temporal diversity
                    # Use day_idx as seed for reproducible but varied sampling
                    random.seed(day_idx + hash(str(sample_date)))  # Reproducible but varied
                    hour_offset = random.randint(0, 23)
                    
                # Convert date to datetime and add random minutes for even better diversity
                # Use same seed for reproducible minute selection
                minute_offset = random.randint(0, 59)
                sample_datetime = datetime.combine(sample_date, datetime.min.time()) + timedelta(hours=hour_offset, minutes=minute_offset)
                
                date_list.append(sample_datetime)
                
        print(f'✅ Generated {len(date_list)} requests across {num_sampling_days} days')
        print(f'This will download exactly {len(date_list) * spectrograms_per_batch} spectrograms total')

        return date_list, sample_time_per_day

    def download_MAT_or_PNG(
        self,
        deviceCode,
        start_date_object,
        filetype='png',
        spectrograms_per_batch=6,
        download_flac=False,
        spectral_downsample: Optional[int] = None,
    ):
        """
        Download MAT or PNG files for a given time period.
        
        :param deviceCode: ONC device code
        :param start_date_object: Start date and time
        :param filetype: Type of file to download ('png' or 'mat')
        :param spectrograms_per_batch: Number of 5-minute spectrograms to download per batch
        :param download_flac: Whether to download corresponding FLAC files
        """
        # Calculate duration based on number of spectrograms (each is 5 minutes = 300 seconds)
        # Use exact duration to get precisely the requested number of spectrograms
        data_length_seconds = (spectrograms_per_batch - 1) * 300
        
        time_delta = dt.timedelta(0, data_length_seconds)
        start_time, end_time = start_and_end_strings(start_date_object, time_delta)

        downsample = spectral_downsample if spectral_downsample is not None else self.spectral_downsample

        if filetype == 'mat':
            # Format the date nicely for logging
            date_str = start_date_object.strftime('%Y-%m-%d')
            time_str = start_date_object.strftime('%H:%M:%S')
            day_name = start_date_object.strftime('%A')
            print(f'📅 Downloading data for {day_name}, {date_str} at {time_str} (requesting {spectrograms_per_batch} spectrograms)')
            dataProductCode = 'HSD'
            filters = {
                'dataProductCode': dataProductCode,
                'deviceCode': deviceCode,
                'dateFrom': start_time,
                'dateTo': end_time,
                'extension': 'mat',
                'dpo_hydrophoneDataDiversionMode': 'OD',
                'dpo_spectralDataDownsample': downsample
            }
            
            # Request data product
            result = self.onc.requestDataProduct(filters)
            self.logger.info(f"Request Id: {result['dpRequestId']}")
            self.logger.info(f"Estimated files: {spectrograms_per_batch} spectrograms + 1 metadata = {spectrograms_per_batch + 1} files")
            
            # Run data product and wait for completion
            run_start = time.time()
            run_data = self.onc.runDataProduct(result['dpRequestId'], waitComplete=True)
            self.logger.info(f"Data product run completed in {time.time() - run_start:.2f}s")
            
            # Download all files from the run
            if 'runIds' in run_data and run_data['runIds']:
                self.logger.info("Downloading files...")
                download_start = time.time()
                self.onc.downloadDataProduct(run_data['runIds'][0])
                self.logger.info(f"Files downloaded successfully in {time.time() - download_start:.2f}s")
                
                # Download FLAC files if requested
                if download_flac:
                    flac_start = time.time()
                    self.download_flac_files(deviceCode, start_time, end_time)
                    self.logger.info(f"FLAC files downloaded in {time.time() - flac_start:.2f}s")
                
                # Process downloaded files
                process_start = time.time()
                self.process_spectrograms(filetype)
                self.logger.info(f"Files processed in {time.time() - process_start:.2f}s")
                
                # Log progress
                num_files = len(glob.glob(os.path.join(self.processed_path, f'*.{filetype}')))
                self.logger.info(f"Progress: {num_files} files downloaded")
                
        elif filetype == 'png':
            # Format the date nicely for logging
            date_str = start_date_object.strftime('%Y-%m-%d')
            time_str = start_date_object.strftime('%H:%M:%S')
            day_name = start_date_object.strftime('%A')
            print(f'📅 Downloading data for {day_name}, {date_str} at {time_str} (requesting {spectrograms_per_batch} spectrograms)')
            filters = {
                'deviceCode': deviceCode,
                'dateFrom': start_time,
                'dateTo': end_time,
                'extension': 'png'
            }
            result = self.onc.getListByDevice(filters, allPages=True)
            spect_png_files = [s for s in result['files'] if "Z-spect.png" in s]
            
            self.logger.info(f"Found {len(spect_png_files)} PNG files")
            
            # Download all PNG files in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [executor.submit(self.onc.getFile, png_file) for png_file in spect_png_files]
                for future in concurrent.futures.as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        self.logger.error(f"Error downloading PNG file: {e}")
            
            # Download FLAC files if requested
            if download_flac:
                flac_start = time.time()
                self.download_flac_files(deviceCode, start_time, end_time)
                self.logger.info(f"FLAC files downloaded in {time.time() - flac_start:.2f}s")
            
            # Process downloaded files
            self.process_spectrograms(filetype)

    def download_flac_files(self, deviceCode, start_time, end_time, onc_client=None):
        """
        Download FLAC audio files corresponding to the same time window as spectrograms.
        Uses parallel downloads for better performance.
        
        :param deviceCode: ONC device code
        :param start_time: Start time string in ISO format
        :param end_time: End time string in ISO format
        """
        self.logger.info(f'Finding FLAC audio files for {deviceCode} from {start_time} to {end_time}')
        
        client = onc_client or self.onc
        # Store original path to ensure it's always restored
        original_output_path = client.outPath
        
        try:
            # Search for FLAC files using archive file API
            search_start = time.time()
            filters = {
                'deviceCode': deviceCode,
                'dateFrom': start_time,
                'dateTo': end_time,
                'extension': 'flac'
            }
            
            result = client.getListByDevice(filters, allPages=True)
            self.logger.info(f"FLAC file search completed in {time.time() - search_start:.2f}s")
            
            if 'files' in result and result['files']:
                flac_files = [f for f in result['files'] if f.lower().endswith('.flac')]
                
                if flac_files:
                    # Temporarily set output path to flac directory
                    client.outPath = self.flac_path
                    
                    self.logger.info(f'Found {len(flac_files)} FLAC file(s)')
                    
                    # Poll-download loop: try until all succeed or attempts exhausted
                    download_start = time.time()
                    pending = set(flac_files)
                    attempts = {f: 0 for f in flac_files}
                    max_attempts = 6
                    while pending:
                        to_retry = list(pending)
                        for flac_file in to_retry:
                            attempts[flac_file] += 1
                            try:
                                self.logger.info(f"Downloading FLAC (attempt {attempts[flac_file]}/{max_attempts}): {flac_file}")
                                client.getFile(flac_file, overwrite=True)
                                pending.discard(flac_file)
                            except Exception as e:
                                if attempts[flac_file] >= max_attempts:
                                    self.logger.warning(f"Failed to download {flac_file} after {attempts[flac_file]} attempts: {e}")
                                    pending.discard(flac_file)
                                else:
                                    self.logger.debug(f"FLAC not ready yet ({flac_file}): {e}")
                        if pending:
                            time.sleep(5)
                    self.logger.info(f"FLAC files downloaded in {time.time() - download_start:.2f}s")
                else:
                    self.logger.info('No FLAC files found in the specified time range')
            else:
                self.logger.info('No files found or error in API response for FLAC search')
                
        except Exception as e:
            self.logger.error(f'Error searching for FLAC files: {e}')
        finally:
            # Always restore original output path
            client.outPath = original_output_path

    def check_for_anomalies(self, file_path, file1, file2):
        try:
            image_obj = None
            if file_path.lower().endswith('.png'):
                image_obj = Image.open(file_path)
                image_obj = np.transpose(image_obj, [1, 0, 2])
            elif file_path.lower().endswith('.mat'):
                mat_data = scipy.io.loadmat(file_path)
                if 'SpectData' in mat_data:
                    image_obj = mat_data['SpectData']['PSD'][0,0]
                else:
                    raise ValueError('No "SpectData" key found in .mat file')

            if image_obj is not None:
                s = np.zeros([np.shape(image_obj)[0], 1])

                anomaly_found = 0
                anom_indices_black = []
                anom_indices_white = []
                for ii in np.arange(0, np.shape(image_obj)[0]):
                    s[ii] = np.sum(image_obj[ii])
                    if s[ii] < 500:
                        anomaly_found = 1
                        anom_indices_black.append(ii)
                    elif s[ii] > 568000:
                        anomaly_found = 2
                        anom_indices_white.append(ii)

                if anomaly_found > 0:
                    file1.write(file_path + "\n")

                    if len(anom_indices_black) > 0:
                        seg = segment2(anom_indices_black)
                        num_segments = seg.shape[0]
                        summary_string = f"{num_segments} black segment(s) found, with entries [{', '.join(' to '.join(map(str, row)) for row in seg)}]"
                        print(f'{summary_string}: {os.path.basename(file_path)}')
                        file2.write(f'{summary_string}: {os.path.basename(file_path)}\n')

                    if len(anom_indices_white) > 0:
                        seg = segment2(anom_indices_white)
                        num_segments = seg.shape[0]
                        summary_string = f"{num_segments} white segment(s) found, with entries [{', '.join(' to '.join(map(str, row)) for row in seg)}]"
                        print(f'{summary_string}: {os.path.basename(file_path)}')
                        file2.write(f'{summary_string}: {os.path.basename(file_path)}\n')

                    # Move file to the rejects folder, check if it's already there
                    if os.path.exists(os.path.join(self.anom_path, os.path.basename(file_path))):
                        # Remove the file from the input folder
                        print(f'File {file_path} already exists in the rejects folder. Removing from the input folder.')
                        os.remove(file_path)
                    else:
                        shutil.move(file_path, self.anom_path)

        except Exception as e:
            err_msg = str(e)
            # Allow truncated MAT files to remain for downstream inspection rather than sending to rejects
            if 'truncated' in err_msg.lower():
                self.logger.warning(f'Truncated MAT detected; keeping file for inspection: {file_path}')
                return

            print(f'Error encountered for: {file_path}, {err_msg}')
            file1.write(file_path + "\n")
            file2.write(f'Error encountered for: {file_path}\n')
            # Move file to the rejects folder, check if it's already there
            if os.path.exists(os.path.join(self.anom_path, os.path.basename(file_path))):
                # Remove the file from the input folder
                print(f'File {file_path} already exists in the rejects folder. Removing from the input folder.')
                os.remove(file_path)
            else:
                shutil.move(file_path, self.anom_path)

    def process_spectrograms(self, filetype='png'):
        process_start = time.time()
        self.logger.info("Starting spectrogram processing")
        
        with open(os.path.join(self.processed_path, 'anomalous_files.txt'), 'w') as file1, \
            open(os.path.join(self.processed_path, 'anomalous_file_summary.txt'), 'w') as file2:

            if filetype == 'png':
                input_image_paths = glob.glob(os.path.join(self.input_path, '*.png'))
                self.logger.info(f"Found {len(input_image_paths)} PNG files to process")

                for input_image in input_image_paths:
                    image_area = (107, 67, 1042, 810)
                    crop_image(input_image, self.processed_path, image_area)

                    image_name = os.path.basename(input_image)
                    trimmed_path = os.path.join(self.processed_path, image_name)

                    self.check_for_anomalies(trimmed_path, file1, file2)

                [os.remove(os.path.join(self.input_path, file_name)) for file_name in os.listdir(self.input_path) if file_name.lower().endswith('.png')]
            elif filetype == 'mat':
                mat_paths = glob.glob(os.path.join(self.input_path, '*.mat'))
                self.logger.info(f"Found {len(mat_paths)} MAT files to process")
                
                for mat_path in mat_paths:
                    self.check_for_anomalies(mat_path, file1, file2)
                
                # Move files to the processed folder, check if they're already there
                for file_name in os.listdir(self.input_path):
                    if file_name.lower().endswith('.mat'):
                        if os.path.exists(os.path.join(self.processed_path, file_name)):
                            # Remove the file from the input folder
                            self.logger.info(f'File {file_name} already exists in the processed folder. Removing from the input folder.')
                            os.remove(os.path.join(self.input_path, file_name))
                        else:
                            shutil.move(os.path.join(self.input_path, file_name), os.path.join(self.processed_path, file_name))

        self.logger.info(f"Spectrogram processing completed in {time.time() - process_start:.2f}s")

    def download_spectrograms_with_sampling_schedule(self, deviceCode, start_date, threshold_num, num_days=None, filetype='png', spectrograms_per_batch=6, download_flac=False):
        """
        Download spectrograms based on a sampling schedule.
        
        :param deviceCode: ONC device code
        :param start_date: Start date for sampling (tuple: year, month, day)
        :param threshold_num: Number of samples to take
        :param num_days: Number of days to sample (optional)
        :param filetype: Type of file to download ('png' or 'mat')
        :param spectrograms_per_batch: Number of 5-minute spectrograms to download per batch
        :param download_flac: Whether to download corresponding FLAC files
        """
        schedule_start = time.time()
        self.logger.info(f"Starting sampling schedule download for {deviceCode} from {start_date}")
        self.logger.info(f"Batch size: {spectrograms_per_batch} spectrograms per request")
        
        # Generate sampling schedule first to determine actual date range
        schedule_start_time = time.time()
        year, month, day = start_date
        date_object_list, sample_time_per_day = self.sampling_schedule(
            deviceCode, threshold_num, year, month, day, num_days=num_days, spectrograms_per_batch=spectrograms_per_batch
        )
        self.logger.info(f"Generated sampling schedule in {time.time() - schedule_start_time:.2f}s")
        
        if not date_object_list:
            self.logger.error("Failed to generate sampling schedule")
            return

        # Calculate actual date range from the sampling schedule
        actual_start_date = min(date_object_list).date()
        actual_end_date = max(date_object_list).date()
        
        # Convert to tuple format for directory setup
        start_date_tuple = (actual_start_date.year, actual_start_date.month, actual_start_date.day)
        end_date_tuple = (actual_end_date.year, actual_end_date.month, actual_end_date.day)
        
        # Calculate duration for directory setup (used for folder naming)
        duration_seconds = (spectrograms_per_batch * 300) - 1

        # Set up directories with the actual date range
        self.setup_directories(filetype, deviceCode, 'sampling', start_date_tuple, end_date_tuple, duration_seconds)

        # Check for existing files and filter the dates (match exact request timestamps)
        date_object_list = self.filter_existing_requests(deviceCode, date_object_list, extension='mat' if filetype == 'mat' else filetype)

        # Download files for each request
        total_requests = len(date_object_list)
        self.logger.info(f"Starting download of {total_requests} requests")
        
        # Show summary of days being downloaded
        unique_dates = sorted(set(ts.date() for ts in date_object_list))
        print(f"📅 Will download data from {len(unique_dates)} unique days:")
        for date in unique_dates:
            day_name = date.strftime('%A')
            date_str = date.strftime('%Y-%m-%d')
            requests_on_day = len([ts for ts in date_object_list if ts.date() == date])
            spectrograms_on_day = requests_on_day * spectrograms_per_batch
            print(f"   • {day_name}, {date_str} ({requests_on_day} requests = {spectrograms_on_day} spectrograms)")
        
        if filetype == 'mat':
            # Submit all requests concurrently (no wait) then poll/download in parallel
            self.logger.info("Submitting MAT runs without waiting for completion...")
            submit_start = time.time()
            run_records = []
            data_length_seconds = (spectrograms_per_batch - 1) * 300
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [
                    executor.submit(
                        self.submit_mat_run_no_wait,
                        deviceCode,
                        ts,
                        ts + timedelta(seconds=data_length_seconds),
                        spectrograms_per_batch,
                    )
                    for ts in date_object_list
                ]
                for future in concurrent.futures.as_completed(futures):
                    try:
                        rec = future.result()
                        run_records.append(rec)
                    except Exception as e:
                        self.logger.error(f"Error submitting run: {e}")
            self.logger.info(f"Submitted {len(run_records)} runs in {time.time() - submit_start:.2f}s")

            # Poll + download in parallel
            pending = run_records
            downloaded = []
            attempt = 0
            while pending:
                attempt += 1
                self.logger.info(f"Polling attempt {attempt}: {len(pending)} runs pending")
                next_pending = []
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = [executor.submit(self.try_download_run, rec, True, download_flac) for rec in pending]
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            status, updated_rec = future.result()
                        except Exception as e:
                            self.logger.error(f"Polling error: {e}")
                            continue
                        if status == 'downloaded':
                            downloaded.append(updated_rec)
                        elif status == 'error':
                            next_pending.append(updated_rec)
                        else:
                            next_pending.append(updated_rec)
                pending = next_pending
                if pending:
                    time.sleep(5)
            total_time = time.time() - schedule_start
            self.logger.info(f"Downloaded {len(downloaded)} MAT batches in {total_time:.2f}s")
        else:
            # Existing sequential path for PNG
            for i, request_time in enumerate(date_object_list, 1):
                request_start = time.time()
                self.logger.info(f"Processing request {i}/{total_requests}: {request_time}")
                
                # Download files for this request (this will get spectrograms_per_batch + 1 files)
                self.download_MAT_or_PNG(deviceCode, request_time, filetype, spectrograms_per_batch, download_flac)
                
                self.logger.info(f"Completed request {i}/{total_requests} in {time.time() - request_start:.2f}s")
                self.logger.info(f"Overall progress: {i}/{total_requests} requests completed")
            
            total_time = time.time() - schedule_start
            self.logger.info(f"Completed all downloads in {total_time:.2f}s")
            self.logger.info(f"Average time per request: {total_time/total_requests:.2f}s")

    def download_spectrograms_with_deployment_check(self, deviceCode, start_date, threshold_num, num_days=None, filetype='png', auto_select_deployment=False, spectrograms_per_batch=6, download_flac=False):
        """
        Download spectrograms with deployment checking enabled.
        
        :param deviceCode: ONC device code
        :param start_date: Start date for sampling
        :param threshold_num: Number of samples to take
        :param num_days: Number of days to sample (optional)
        :param filetype: Type of file to download ('png' or 'mat')
        :param auto_select_deployment: Whether to automatically select the best deployment
        :param spectrograms_per_batch: Number of 5-minute spectrograms to download per batch
        :param download_flac: Whether to download corresponding FLAC files
        """
        self.logger.info(f"Starting deployment-aware download for {deviceCode}")
        self.logger.info(f"Batch size: {spectrograms_per_batch} spectrograms per request")
        
        # Get deployment information
        deployment_info = self.deployment_checker.get_deployment_info(deviceCode)
        if not deployment_info:
            self.logger.error(f"Could not get deployment information for {deviceCode}")
            return

        # Generate sampling schedule
        sampling_schedule = self.deployment_checker.generate_sampling_schedule(
            deployment_info, 
            start_date, 
            threshold_num, 
            num_days
        )
        
        if not sampling_schedule:
            self.logger.error("Failed to generate sampling schedule")
            return

        # Calculate duration for directory setup
        duration_seconds = (spectrograms_per_batch * 300) - 1

        # Set up directories for the download
        self.setup_directories(deviceCode, filetype, start_date, sampling_schedule[-1], duration_seconds)

        # Download files for each time slot with deployment checking
        total_slots = len(sampling_schedule)
        self.logger.info(f"Starting download of {total_slots} time slots with deployment checking")
        
        for i, time_slot in enumerate(sampling_schedule, 1):
            slot_start = time.time()
            self.logger.info(f"Processing slot {i}/{total_slots}: {time_slot}")
            
            # Download with deployment check
            success, deployment = self.download_with_deployment_check(
                deviceCode, time_slot, filetype, spectrograms_per_batch, auto_select_deployment, download_flac
            )
            
            if success:
                self.logger.info(f"Successfully downloaded slot {i}/{total_slots}")
            else:
                self.logger.warning(f"Failed to download slot {i}/{total_slots}")
            
            self.logger.info(f"Completed slot {i}/{total_slots} in {time.time() - slot_start:.2f}s")
            self.logger.info(f"Overall progress: {i}/{total_slots} slots completed")
        
        self.logger.info("Deployment-aware download completed")

    def download_with_deployment_check(self, deviceCode, start_date_object, filetype='png', data_length_seconds=1799, auto_select_deployment=False, download_flac=False):
        """
        Download spectrograms with deployment validation.
        
        :param deviceCode: Device code
        :param start_date_object: Start date (datetime object)
        :param filetype: File type ('png' or 'mat')
        :param data_length_seconds: Length of data to download in seconds
        :param auto_select_deployment: If True, automatically select best deployment
        :param download_flac: Whether to also download corresponding FLAC audio files
        :return: Success status and deployment info
        """
        # Ensure timezone-aware datetimes
        start_date_object = ensure_timezone_aware(start_date_object)
        end_date_object = start_date_object + timedelta(seconds=data_length_seconds)
        
        print(f"\nValidating deployment coverage for {deviceCode}...")
        has_coverage, deployments = self.validate_deployment_coverage(
            deviceCode, start_date_object, end_date_object
        )
        
        if not has_coverage:
            print(f"❌ No deployment coverage for {deviceCode} from {start_date_object.strftime('%Y-%m-%d %H:%M:%S')} to {end_date_object.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Suggest alternative dates - get all deployments for this device
            all_deployments = self._get_cached_deployments()
            device_deployments = [dep for dep in all_deployments if dep.device_code == deviceCode]
            
            if device_deployments:
                print("\nAvailable deployment periods:")
                for deployment in device_deployments:
                    end_str = deployment.end_date.strftime('%Y-%m-%d') if deployment.end_date else 'ongoing'
                    print(f"  • {deployment.begin_date.strftime('%Y-%m-%d')} to {end_str} at {deployment.location_name}")
            return False, None
        
        if auto_select_deployment:
            # Use the first available deployment for now
            selected_deployment = deployments[0]
        else:
            # Interactive selection if multiple deployments
            if len(deployments) > 1:
                print(f"\nMultiple deployments found for the requested time range.")
                selected_deployment = self.interactive_deployment_selection(
                    deviceCode, start_date_object, end_date_object
                )
                if not selected_deployment:
                    print("No deployment selected. Aborting download.")
                    return False, None
            else:
                selected_deployment = deployments[0]
        
        end_str = selected_deployment.end_date.strftime('%Y-%m-%d') if selected_deployment.end_date else 'ongoing'
        print(f"✅ Using deployment: {selected_deployment.begin_date.strftime('%Y-%m-%d')} to {end_str} at {selected_deployment.location_name}")
        
        # Proceed with download
        try:
            self.download_MAT_or_PNG(deviceCode, start_date_object, filetype=filetype, spectrograms_per_batch=6, download_flac=download_flac)
            return True, selected_deployment
        except Exception as e:
            print(f"❌ Download failed: {e}")
            return False, selected_deployment

    def show_available_deployments(self, device_code, start_date, end_date, check_data_availability=True):
        """
        Show available deployments for a device within a date range.
        
        :param device_code: Device code to check deployments for
        :param start_date: Start date (datetime object)
        :param end_date: End date (datetime object)
        :param check_data_availability: Whether to check data availability
        :return: List of deployment info objects
        """
        print(f"\nChecking deployments for {device_code} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Ensure timezone-aware datetimes
        start_date = ensure_timezone_aware(start_date)
        end_date = ensure_timezone_aware(end_date)
        
        # Use cached deployments to avoid redundant API calls
        all_deployments = self._get_cached_deployments()
        device_deployments = [dep for dep in all_deployments if dep.device_code == device_code]
        
        # Filter to deployments that overlap with the date range
        overlapping_deployments = []
        for deployment in device_deployments:
            dep_start = ensure_timezone_aware(deployment.begin_date)
            dep_end = ensure_timezone_aware(deployment.end_date) if deployment.end_date else datetime.now(timezone.utc)
            
            # Check if deployment overlaps with requested time range
            if dep_start <= end_date and dep_end >= start_date:
                overlapping_deployments.append(deployment)
        
        if check_data_availability and overlapping_deployments:
            overlapping_deployments = self.deployment_checker.check_data_availability(
                overlapping_deployments, start_date, end_date
            )
        
        if not overlapping_deployments:
            print(f"No deployments found for {device_code} in the specified date range.")
            return []
        
        print(f"\nFound {len(overlapping_deployments)} deployment(s):")
        for i, deployment in enumerate(overlapping_deployments, 1):
            print(f"  {i}. {deployment.begin_date.strftime('%Y-%m-%d')} to {deployment.end_date.strftime('%Y-%m-%d') if deployment.end_date else 'ongoing'}")
            print(f"     Location: {deployment.location_name}")
            if hasattr(deployment, 'has_data'):
                print(f"     Data Available: {deployment.has_data}")
        
        return overlapping_deployments

    def interactive_deployment_selection(self, device_code, start_date, end_date):
        """
        Interactive deployment selection for a device within a date range.
        
        :param device_code: Device code to check deployments for
        :param start_date: Start date (datetime object)
        :param end_date: End date (datetime object)
        :return: Selected deployment info object or None
        """
        from .deployment_checker import interactive_deployment_selector
        return interactive_deployment_selector(self.deployment_checker, start_date, end_date)

    def validate_deployment_coverage(self, device_code, start_date, end_date):
        """
        Validate that the requested date range has deployment coverage.
        
        :param device_code: Device code to validate
        :param start_date: Start date (datetime object)
        :param end_date: End date (datetime object)
        :return: (bool, list) - (has_coverage, list_of_deployments)
        """
        # Ensure input dates are timezone-aware
        start_date = ensure_timezone_aware(start_date)
        end_date = ensure_timezone_aware(end_date)
        
        # Use cached deployments to avoid redundant API calls
        all_deployments = self._get_cached_deployments()
        device_deployments = [dep for dep in all_deployments if dep.device_code == device_code]
        
        # Filter to deployments that overlap with the date range
        overlapping_deployments = []
        for deployment in device_deployments:
            dep_start = ensure_timezone_aware(deployment.begin_date)
            dep_end = ensure_timezone_aware(deployment.end_date) if deployment.end_date else datetime.now(timezone.utc)
            
            # Check if deployment overlaps with requested time range
            if dep_start <= end_date and dep_end >= start_date:
                overlapping_deployments.append(deployment)
        
        if not overlapping_deployments:
            return False, []
        
        # Check if any deployment covers the entire requested range
        for deployment in overlapping_deployments:
            dep_start = ensure_timezone_aware(deployment.begin_date)
            dep_end = ensure_timezone_aware(deployment.end_date) if deployment.end_date else datetime.now(timezone.utc)
            if dep_start <= start_date and dep_end >= end_date:
                return True, [deployment]
        
        # Check if multiple deployments together cover the range
        deployments_sorted = sorted(overlapping_deployments, key=lambda x: x.begin_date)
        coverage_start = ensure_timezone_aware(deployments_sorted[0].begin_date)
        coverage_end = ensure_timezone_aware(deployments_sorted[-1].end_date) if deployments_sorted[-1].end_date else datetime.now(timezone.utc)
        
        if coverage_start <= start_date and coverage_end >= end_date:
            # Check for gaps
            for i in range(len(deployments_sorted) - 1):
                curr_end = ensure_timezone_aware(deployments_sorted[i].end_date) if deployments_sorted[i].end_date else datetime.now(timezone.utc)
                next_start = ensure_timezone_aware(deployments_sorted[i + 1].begin_date)
                if curr_end < next_start:
                    gap_start = curr_end
                    gap_end = next_start
                    if gap_start < end_date and gap_end > start_date:
                        print(f"Warning: Gap in deployment coverage from {gap_start.strftime('%Y-%m-%d')} to {gap_end.strftime('%Y-%m-%d')}")
            return True, deployments_sorted
        
        return False, overlapping_deployments

    def download_specific_spectrograms(self, device_times_dict, filetype='png', duration_seconds=300, download_flac=False):
        """
        Downloads spectrograms for specific device IDs and timestamps.
        
        :param device_times_dict: Dictionary where keys are device IDs, and values are lists of tuples (year, month, day, hour, minute, second).
        :param filetype: File type to download ('png' or 'mat').
        :param duration_seconds: Duration of each spectrogram in seconds (default: 300 for 5 minutes).
        :param download_flac: Whether to also download corresponding FLAC audio files.
        """
        
        for device_id, times in device_times_dict.items():
            # Calculate date range for this device
            if times:
                # Get min and max dates from the time list
                dates = [datetime(t[0], t[1], t[2]) for t in times]
                start_date = min(dates)
                end_date = max(dates)
                
                start_date_tuple = (start_date.year, start_date.month, start_date.day)
                end_date_tuple = (end_date.year, end_date.month, end_date.day) if start_date.date() != end_date.date() else None
                
                # Setup directories once per device with date range
                self.setup_directories(filetype, device_id, 'specific_times', start_date_tuple, end_date_tuple, duration_seconds)

            for time_tuple in times:
                year, month, day, hour, minute, second = time_tuple
                start_date_object = datetime(year, month, day, hour, minute, second)

                # Download specific spectrogram with custom duration
                self.download_MAT_or_PNG(device_id, start_date_object, filetype=filetype, spectrograms_per_batch=6, download_flac=download_flac)

                # Process the spectrograms
                # self.process_spectrograms(filetype)

    def quick_deployment_check(self, device_code, start_date, end_date):
        """
        Quick check for deployment availability in a date range.
        
        :param device_code: Device code to check
        :param start_date: Start date (datetime object)
        :param end_date: End date (datetime object)
        :return: Boolean indicating if deployments are available
        """
        # Ensure timezone-aware datetimes
        start_date = ensure_timezone_aware(start_date)
        end_date = ensure_timezone_aware(end_date)
        
        # Use cached deployments to avoid redundant API calls
        all_deployments = self._get_cached_deployments()
        device_deployments = [dep for dep in all_deployments if dep.device_code == device_code]
        
        # Check for overlapping deployments
        for deployment in device_deployments:
            dep_start = ensure_timezone_aware(deployment.begin_date)
            dep_end = ensure_timezone_aware(deployment.end_date) if deployment.end_date else datetime.now(timezone.utc)
            
            # Check if deployment overlaps with requested time range
            if dep_start <= end_date and dep_end >= start_date:
                return True
        
        return False

    def interactive_download_with_deployments(self, device_code, filetype='png'):
        """
        Interactive download process with deployment guidance.
        
        :param device_code: Device code
        :param filetype: File type ('png' or 'mat')
        """
        print(f"\n🎯 Interactive Hydrophone Data Download for {device_code}")
        print("=" * 60)
        
        # Get all deployments for this device (using cache to avoid redundant API calls)
        all_deployments = self._get_cached_deployments()
        device_deployments = [dep for dep in all_deployments if dep.device_code == device_code]
        
        if not device_deployments:
            print(f"❌ No deployments found for device {device_code}")
            return
        
        print(f"\nAvailable deployments for {device_code}:")
        for i, deployment in enumerate(device_deployments, 1):
            end_str = deployment.end_date.strftime('%Y-%m-%d') if deployment.end_date else 'ongoing'
            print(f"  {i}. {deployment.begin_date.strftime('%Y-%m-%d')} to {end_str}")
            print(f"     Location: {deployment.location_name}")
        
        # Get user input for date range
        try:
            start_input = input("\nEnter start date (YYYY-MM-DD): ").strip()
            end_input = input("Enter end date (YYYY-MM-DD): ").strip()
            
            # Create timezone-aware datetimes
            start_date = ensure_timezone_aware(datetime.strptime(start_input, '%Y-%m-%d'))
            end_date = ensure_timezone_aware(datetime.strptime(end_input, '%Y-%m-%d'))
            
            if start_date >= end_date:
                print("❌ Start date must be before end date")
                return
            
        except ValueError:
            print("❌ Invalid date format. Please use YYYY-MM-DD")
            return
        
        # Check deployment coverage using already fetched data
        has_coverage, deployments = self._validate_deployment_coverage_with_data(
            device_deployments, start_date, end_date
        )
        
        if not has_coverage:
            print(f"❌ No deployment coverage for the requested date range")
            print("\nWould you like to see alternative date ranges? (y/n): ", end="")
            if input().strip().lower() == 'y':
                # Show deployments within an expanded range
                expanded_start = start_date - timedelta(days=30)
                expanded_end = end_date + timedelta(days=30)
                self._show_deployments_with_data(device_deployments, expanded_start, expanded_end)
            return
        
        # Check data availability for the deployments we found
        print("Checking data availability...")
        available_deployments = self.deployment_checker.check_data_availability(
            deployments, start_date, end_date
        )
        
        if not available_deployments:
            print("❌ No data available for the deployment periods covering your date range")
            return
        
        print(f"✅ Found {len(available_deployments)} deployment(s) with available data")
        
        # Get sampling parameters
        try:
            threshold_num = int(input("\nHow many spectrograms do you want to download? "))
            if threshold_num <= 0:
                print("❌ Number of spectrograms must be positive")
                return
        except ValueError:
            print("❌ Invalid number")
            return
        
        print(f"\nProceeding with download:")
        print(f"  Device: {device_code}")
        print(f"  Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print(f"  Target: {threshold_num} spectrograms")
        print(f"  File type: {filetype}")
        
        # Convert to the format expected by download_spectrograms_with_deployment_check
        start_date_tuple = (start_date.year, start_date.month, start_date.day)
        end_date_tuple = (end_date.year, end_date.month, end_date.day)
        num_days = (end_date - start_date).days
        
        # Setup directories with date range info (using default 5-minute duration)
        self.setup_directories(filetype, device_code, 'sampling', start_date_tuple, end_date_tuple, 300)
        
        self.download_spectrograms_with_deployment_check(
            device_code, start_date_tuple, threshold_num, num_days=num_days, 
            filetype=filetype, auto_select_deployment=True
        )
    
    def _validate_deployment_coverage_with_data(self, device_deployments, start_date, end_date):
        """
        Validate deployment coverage using pre-fetched deployment data.
        
        :param device_deployments: List of deployment objects for the device
        :param start_date: Start date (timezone-aware datetime)
        :param end_date: End date (timezone-aware datetime)
        :return: (bool, list) - (has_coverage, list_of_covering_deployments)
        """
        # Filter deployments that overlap with the date range
        overlapping_deployments = []
        for deployment in device_deployments:
            dep_start = ensure_timezone_aware(deployment.begin_date)
            dep_end = ensure_timezone_aware(deployment.end_date) if deployment.end_date else datetime.now(timezone.utc)
            
            # Check if deployment overlaps with requested time range
            if dep_start <= end_date and dep_end >= start_date:
                overlapping_deployments.append(deployment)
        
        if not overlapping_deployments:
            return False, []
        
        # Check if any deployment covers the entire requested range
        for deployment in overlapping_deployments:
            dep_start = ensure_timezone_aware(deployment.begin_date)
            dep_end = ensure_timezone_aware(deployment.end_date) if deployment.end_date else datetime.now(timezone.utc)
            if dep_start <= start_date and dep_end >= end_date:
                return True, [deployment]
        
        # Check if multiple deployments together cover the range
        deployments_sorted = sorted(overlapping_deployments, key=lambda x: x.begin_date)
        coverage_start = ensure_timezone_aware(deployments_sorted[0].begin_date)
        coverage_end = ensure_timezone_aware(deployments_sorted[-1].end_date) if deployments_sorted[-1].end_date else datetime.now(timezone.utc)
        
        if coverage_start <= start_date and coverage_end >= end_date:
            # Check for gaps
            for i in range(len(deployments_sorted) - 1):
                curr_end = ensure_timezone_aware(deployments_sorted[i].end_date) if deployments_sorted[i].end_date else datetime.now(timezone.utc)
                next_start = ensure_timezone_aware(deployments_sorted[i + 1].begin_date)
                if curr_end < next_start:
                    gap_start = curr_end
                    gap_end = next_start
                    if gap_start < end_date and gap_end > start_date:
                        print(f"Warning: Gap in deployment coverage from {gap_start.strftime('%Y-%m-%d')} to {gap_end.strftime('%Y-%m-%d')}")
            return True, deployments_sorted
        
        return False, overlapping_deployments
    
    def _show_deployments_with_data(self, device_deployments, start_date, end_date):
        """
        Show deployments using pre-fetched data instead of making new API calls.
        """
        overlapping = []
        for deployment in device_deployments:
            dep_start = ensure_timezone_aware(deployment.begin_date)
            dep_end = ensure_timezone_aware(deployment.end_date) if deployment.end_date else datetime.now(timezone.utc)
            
            # Check if deployment overlaps with requested time range
            if dep_start <= end_date and dep_end >= start_date:
                overlapping.append(deployment)
        
        if overlapping:
            print(f"\nDeployments overlapping with {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}:")
            for i, deployment in enumerate(overlapping, 1):
                end_str = deployment.end_date.strftime('%Y-%m-%d') if deployment.end_date else 'ongoing'
                print(f"  {i}. {deployment.begin_date.strftime('%Y-%m-%d')} to {end_str}")
                print(f"     Location: {deployment.location_name}")
        else:
            print(f"\nNo deployments found overlapping with {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

    def _get_cached_deployments(self, max_age_minutes=30):
        """
        Get deployments from cache or fetch new ones if cache is stale.
        
        :param max_age_minutes: Maximum age of cache in minutes
        :return: List of all deployment objects
        """
        now = datetime.now()
        
        # Check if cache is valid
        if (self._deployment_cache is not None and 
            self._cache_timestamp is not None and 
            (now - self._cache_timestamp).total_seconds() < max_age_minutes * 60):
            return self._deployment_cache
        
        # Cache is stale or doesn't exist, fetch fresh data
        print("Fetching deployment information...")
        self._deployment_cache = self.deployment_checker.get_all_hydrophone_deployments()
        self._cache_timestamp = now
        
        return self._deployment_cache
