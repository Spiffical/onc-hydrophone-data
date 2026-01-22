import glob
import json
import math
import os
from datetime import datetime, timedelta
from typing import Any, List, Optional, Tuple

import numpy as np
import scipy.io

from .audio_backends import librosa, sf, torch, torchaudio
from .constants import FIVE_MINUTES_SECONDS
from .models import TimestampRequest


def _collect_files_for_range(
    self,
    device_code: str,
    start_dt: datetime,
    end_dt: datetime,
    extension: str,
    search_dirs: Optional[List[str]] = None,
) -> List[Tuple[datetime, str]]:
    """Return sorted list of (start_time, path) overlapping the requested range."""
    # Use flat directory structure - just search spectrogram_path
    dirs = search_dirs or [self.spectrogram_path]
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
        psd_field = spect_struct['PSD']
    except Exception:
        return np.empty((0, 0)), None, float(FIVE_MINUTES_SECONDS)
    try:
        psd = psd_field
        if isinstance(psd, np.ndarray) and psd.dtype == object and psd.size == 1:
            psd = psd.flat[0]
        psd = np.asarray(psd)
    except Exception:
        return np.empty((0, 0)), None, float(FIVE_MINUTES_SECONDS)
    if psd.ndim == 0:
        return np.empty((0, 0)), None, float(FIVE_MINUTES_SECONDS)
    if psd.ndim == 1:
        psd = psd[:, None]
    elif psd.ndim > 2:
        psd = np.squeeze(psd)
        if psd.ndim == 1:
            psd = psd[:, None]
        elif psd.ndim != 2:
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
    if torch is None:
        self.logger.warning("torch not available; skipping audio clipping")
        return None, None
    waveform = None
    sample_rate = None
    last_error = None
    if torchaudio is not None:
        try:
            waveform, sample_rate = torchaudio.load(file_path)
        except Exception as exc:
            last_error = exc
            waveform = None
    if waveform is None:
        if sf is not None:
            try:
                audio_data, sample_rate = sf.read(file_path, always_2d=True)
                audio_data = audio_data.T
                waveform = torch.from_numpy(audio_data.astype(np.float32, copy=False))
            except Exception as exc:
                last_error = exc
        elif librosa is not None:
            try:
                audio_data, sample_rate = librosa.load(file_path, sr=None, mono=False)
                if audio_data.ndim == 1:
                    audio_data = audio_data[None, :]
                waveform = torch.from_numpy(audio_data.astype(np.float32, copy=False))
            except Exception as exc:
                last_error = exc
    if waveform is None or sample_rate is None:
        if last_error is not None:
            self.logger.warning(f"Failed to read audio {file_path}: {last_error}")
        else:
            self.logger.warning(f"Failed to read audio {file_path}: no available decoder")
        return None, None
    total_samples = waveform.shape[1]
    start_offset = max(0.0, (clip_start - file_start).total_seconds())
    end_offset = max(0.0, (clip_end - file_start).total_seconds())
    start_sample = int(max(0, math.floor(start_offset * sample_rate)))
    end_sample = int(min(total_samples, math.ceil(end_offset * sample_rate)))
    if end_sample <= start_sample:
        return None, sample_rate
    return waveform[:, start_sample:end_sample], sample_rate


def _load_audio_file(self, file_path: str) -> Tuple[Optional['torch.Tensor'], Optional[int]]:
    if torch is None:
        self.logger.warning("torch not available; skipping audio trim")
        return None, None
    waveform = None
    sample_rate = None
    last_error = None
    if torchaudio is not None:
        try:
            waveform, sample_rate = torchaudio.load(file_path)
        except Exception as exc:
            last_error = exc
            waveform = None
    if waveform is None:
        if sf is not None:
            try:
                audio_data, sample_rate = sf.read(file_path, always_2d=True)
                audio_data = audio_data.T
                waveform = torch.from_numpy(audio_data.astype(np.float32, copy=False))
            except Exception as exc:
                last_error = exc
        elif librosa is not None:
            try:
                audio_data, sample_rate = librosa.load(file_path, sr=None, mono=False)
                if audio_data.ndim == 1:
                    audio_data = audio_data[None, :]
                waveform = torch.from_numpy(audio_data.astype(np.float32, copy=False))
            except Exception as exc:
                last_error = exc
    if waveform is None or sample_rate is None:
        if last_error is not None:
            self.logger.warning(f"Failed to read audio {file_path}: {last_error}")
        else:
            self.logger.warning(f"Failed to read audio {file_path}: no available decoder")
        return None, None
    return waveform, sample_rate


def _write_audio_clip(
    self,
    req: TimestampRequest,
    waveform: 'torch.Tensor',
    sample_rate: int,
    clip_start: datetime,
    clip_end: datetime,
    *,
    suffix: str = 'audio',
) -> str:
    clip_dir = os.path.join(self.parent_dir, req.device_code, req.tag, 'clips', 'audio')
    os.makedirs(clip_dir, exist_ok=True)
    out_name = self._clip_basename(req, suffix)
    out_path = os.path.join(clip_dir, f"{out_name}.flac")
    if torchaudio is not None:
        try:
            torchaudio.save(out_path, waveform, sample_rate)
        except Exception as exc:
            if sf is None:
                raise
            self.logger.warning(f"torchaudio save failed; falling back to soundfile: {exc}")
            audio_np = waveform.detach().cpu().numpy().T
            sf.write(out_path, audio_np, sample_rate, format='FLAC')
    elif sf is not None:
        audio_np = waveform.detach().cpu().numpy().T
        sf.write(out_path, audio_np, sample_rate, format='FLAC')
    else:
        raise RuntimeError("No audio backend available to save clips (torchaudio/soundfile)")
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


def _trim_audio_clip(
    self,
    audio_path: str,
    req: TimestampRequest,
    *,
    clip_start: datetime,
    clip_end: datetime,
    offset_seconds: float,
    duration_seconds: float,
) -> Optional[str]:
    waveform, sample_rate = self._load_audio_file(audio_path)
    if waveform is None or sample_rate is None:
        return None
    total_samples = waveform.shape[1]
    start_sample = int(max(0, math.floor(offset_seconds * sample_rate)))
    end_sample = int(math.ceil((offset_seconds + duration_seconds) * sample_rate))
    end_sample = min(total_samples, end_sample)
    if end_sample <= start_sample:
        self.logger.warning(f"Trim window produced empty audio for {audio_path}")
        return None
    trimmed = waveform[:, start_sample:end_sample]
    if trimmed.shape[1] == 0:
        self.logger.warning(f"Trim window produced empty audio for {audio_path}")
        return None
    return self._write_audio_clip(
        req,
        trimmed,
        sample_rate,
        clip_start,
        clip_end,
        suffix='audio_exact',
    )
