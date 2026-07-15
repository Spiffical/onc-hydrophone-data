#!/usr/bin/env python3
"""
Custom spectrogram generation from audio files.
Translates MATLAB spectrogram functionality to Python.
"""

import copy
import os
import json
from datetime import datetime, timezone
import numpy as np
import scipy.io
import scipy.signal
import matplotlib

def _should_use_agg() -> bool:
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            return False
    except Exception:
        pass
    return True

if _should_use_agg():
    matplotlib.use('Agg')  # Use non-interactive backend for thread safety in scripts.

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import soundfile as sf
import librosa
from pathlib import Path
from typing import Union, Tuple, List, Optional, Any
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
from functools import lru_cache

# Optional imports for acceleration
try:
    import torch
    import torchaudio
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    torchaudio = None

# Thread lock for matplotlib operations (shared across instances)
_plot_lock = threading.Lock()


@lru_cache(maxsize=32)
def _cached_scipy_window(window_type: Union[str, Tuple[str, float]], win_length: int) -> np.ndarray:
    """Build immutable, reusable FFT windows for the common hashable specs."""
    window = np.asarray(
        scipy.signal.get_window(window_type, win_length, fftbins=True),
        dtype=np.float32,
    )
    window.setflags(write=False)
    return window

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PrintLogger:
    def info(self, msg, *args, **kwargs):
        print(msg % args if args else msg)
    def warning(self, msg, *args, **kwargs):
        print('WARNING:', msg % args if args else msg)
    def error(self, msg, *args, **kwargs):
        print('ERROR:', msg % args if args else msg)

class SpectrogramGenerator:
    """
    Generate spectrograms from audio files with configurable parameters.
    Based on MATLAB spectrogram computation with normalization and dB conversion.
    """
    
    def __init__(self,
                 win_dur: float = 1.0,
                 overlap: float = 0.5,
                 window_type: Union[str, Tuple[str, float], np.ndarray] = 'hann',
                 nfft: Optional[int] = None,
                 win_length: Optional[int] = None,
                 hop_length: Optional[int] = None,
                 freq_lims: Tuple[float, float] = (10, 10000),
                 colormap: str = 'turbo',
                 clim: Tuple[float, float] = (-60, 0),
                 log_freq: bool = True,
                 crop_freq_lims: bool = False,
                 max_duration: Optional[float] = None,
                 clip_start: Optional[float] = None,
                 clip_end: Optional[float] = None,
                 clip_pad_seconds: Union[float, str, None] = 'auto',
                 backend: str = 'auto',
                 torch_device: str = 'cpu',
                 scaling: str = 'density',
                 quiet: bool = False,
                 use_logging: bool = True):
        """
        Initialize spectrogram generator with parameters from MATLAB code.
        
        Args:
            win_dur: Window duration in seconds (controls FFT size: NFFT = win_dur * fs)
            overlap: Overlap ratio between adjacent windows (0-1), higher = smoother time axis
            window_type: Window function name/tuple for scipy.signal.get_window (e.g., 'hann', ('kaiser', 14))
                         Custom arrays or unsupported window types fall back to the SciPy backend.
            nfft: FFT size in samples (None = derived from win_dur/sample_rate)
            win_length: Window length in samples (None = use nfft)
            hop_length: Step size in samples (None = derived from overlap ratio)
            freq_lims: Frequency limits for plotting [Hz] (and cropping if crop_freq_lims=True)
            colormap: Matplotlib colormap name
            clim: Color axis limits [dB]
            log_freq: Whether to use logarithmic frequency scale
            crop_freq_lims: If True, crop saved outputs to freq_lims
            max_duration: Maximum duration to process in seconds (None = full file)
            clip_start: Optional start time (seconds) to trim from beginning of audio
            clip_end: Optional end time (seconds) to stop processing; must be > clip_start
            clip_pad_seconds: Extra context (seconds) to include on each side of the clip
                before the STFT; the spectrogram is trimmed back to the target window.
                Use 'auto' to pad by half the window length (helps reduce edge artifacts).
            backend: 'auto' (default), 'torch', or 'scipy' backend for spectrogram computation
            torch_device: Torch device for spectrogram computation ('cpu', 'cuda', or 'auto')
            scaling: 'density' (default) or 'spectrum' scaling for PSD normalization
            quiet: If True, suppress logger noise (only minimal prints for progress bar)
            use_logging: If False, fall back to stdout printing (avoids notebook logging friction)
        """
        self.win_dur = win_dur
        self.overlap = overlap
        self.window_type = window_type
        self.nfft = nfft
        self.win_length = win_length
        self.hop_length = hop_length
        self.freq_lims = freq_lims
        self.colormap = colormap
        self.clim = clim
        self.log_freq = log_freq
        self.crop_freq_lims = crop_freq_lims
        self.max_duration = max_duration
        self.clip_start = clip_start
        self.clip_end = clip_end
        self.clip_pad_seconds = clip_pad_seconds
        self.backend = backend
        self.torch_device = torch_device
        self.scaling = scaling
        self.quiet = quiet
        self.log = logger if use_logging else PrintLogger()
        self._torch_transform_cache = {}
        self._cache_lock = threading.Lock()
        
    def load_audio(self, audio_path: Union[str, Path]) -> Tuple[np.ndarray, int, Optional[dict]]:
        """
        Load audio file supporting multiple formats.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (audio_data, sample_rate, clip_meta)
        """
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
        try:
            # Try soundfile first (supports FLAC, WAV, etc.)
            audio_data, sample_rate = sf.read(str(audio_path))
            
            # Convert to mono if stereo
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
                
        except Exception as e:
            try:
                # Fallback to librosa for other formats
                audio_data, sample_rate = librosa.load(str(audio_path), sr=None)
            except Exception as e2:
                raise RuntimeError(f"Could not load audio file {audio_path}: {e}, {e2}")
        
        # Optional clipping window with context padding
        total_samples = len(audio_data)
        original_duration = total_samples / sample_rate
        clip_meta = None
        start_idx = 0
        end_idx = total_samples
        if self.clip_start is not None or self.clip_end is not None:
            clip_start = self.clip_start or 0.0
            clip_end = self.clip_end or original_duration
            if clip_end <= clip_start:
                raise ValueError(f"Invalid clip window: start={clip_start}s end={clip_end}s for {audio_path.name}")
            clip_pad_seconds = self._resolve_clip_pad_seconds(sample_rate)
            if clip_pad_seconds < 0:
                raise ValueError(f"clip_pad_seconds must be >= 0; got {clip_pad_seconds}")
            extended_start = max(0.0, clip_start - clip_pad_seconds)
            extended_end = min(original_duration, clip_end + clip_pad_seconds)
            start_idx = max(0, int(extended_start * sample_rate))
            end_idx = min(total_samples, int(extended_end * sample_rate))
            if end_idx <= start_idx:
                raise ValueError(f"Invalid padded clip window: {extended_start:.2f}s–{extended_end:.2f}s for {audio_path.name}")
            audio_data = audio_data[start_idx:end_idx]
            original_duration = len(audio_data) / sample_rate
            clip_meta = {
                'clip_offset_seconds': max(0.0, clip_start - extended_start),
                'clip_duration_seconds': max(0.0, clip_end - clip_start),
                'clip_pad_seconds': clip_pad_seconds,
            }
            if not self.quiet:
                self.log.info(
                    f"Clipped audio to {extended_start:.2f}s–{extended_end:.2f}s "
                    f"(context pad {clip_pad_seconds:.2f}s, {audio_path.name})"
                )

        # Truncate to max_duration if specified
        if self.max_duration is not None and original_duration > self.max_duration:
            max_samples = int(self.max_duration * sample_rate)
            audio_data = audio_data[:max_samples]
            if not self.quiet:
                self.log.info(f"Truncated audio from {original_duration:.2f}s to {self.max_duration:.2f}s")

        if not self.quiet:
            self.log.info(f"Loaded audio: {audio_path.name}, duration: {len(audio_data)/sample_rate:.2f}s, sr: {sample_rate}Hz")
        return audio_data, sample_rate, clip_meta

    def _resolve_clip_pad_seconds(self, sample_rate: int) -> float:
        """Resolve clip_pad_seconds, supporting an 'auto' mode."""
        if self.clip_pad_seconds is None:
            auto_mode = True
        elif isinstance(self.clip_pad_seconds, str):
            auto_mode = self.clip_pad_seconds.strip().lower() == 'auto'
        else:
            auto_mode = False

        if auto_mode:
            win_length, _, hop_length, _ = self._resolve_fft_params(sample_rate)
            win_seconds = win_length / float(sample_rate)
            hop_seconds = hop_length / float(sample_rate)
            return max(win_seconds * 0.5, hop_seconds * 0.5)

        try:
            return float(self.clip_pad_seconds)
        except (TypeError, ValueError) as exc:
            raise ValueError("clip_pad_seconds must be a non-negative float or 'auto'") from exc

    def _resolve_fft_params(self, sample_rate: int) -> Tuple[int, int, int, int]:
        win_length = self.win_length or int(self.win_dur * sample_rate)
        if win_length <= 0:
            raise ValueError(f"Invalid win_length={win_length}; check win_dur or win_length settings.")
        nfft = self.nfft or win_length
        if nfft < win_length:
            self.log.warning(f"nfft={nfft} < win_length={win_length}; using win_length for nfft.")
            nfft = win_length
        if self.hop_length is not None:
            hop_length = int(self.hop_length)
            if hop_length <= 0 or hop_length > win_length:
                raise ValueError(f"Invalid hop_length={hop_length}; must be in (0, win_length].")
        else:
            noverlap = int(self.overlap * win_length)
            hop_length = win_length - noverlap
            if hop_length <= 0:
                raise ValueError(f"Invalid overlap={self.overlap}; results in non-positive hop_length.")
        noverlap = win_length - hop_length
        return win_length, nfft, hop_length, noverlap

    @staticmethod
    def _sanitize_metadata_for_mat(value: Any) -> Any:
        if value is None:
            return 'null'
        if isinstance(value, dict):
            return {key: SpectrogramGenerator._sanitize_metadata_for_mat(val) for key, val in value.items()}
        if isinstance(value, (list, tuple)):
            return [SpectrogramGenerator._sanitize_metadata_for_mat(val) for val in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, (str, int, float, bool)):
            return value
        return str(value)

    def _describe_window_type(self) -> Any:
        window_type = self.window_type
        if isinstance(window_type, np.ndarray):
            return {'type': 'array', 'shape': list(window_type.shape)}
        if isinstance(window_type, (list, tuple)):
            if (
                isinstance(window_type, tuple)
                and window_type
                and isinstance(window_type[0], str)
            ):
                return list(window_type)
            return {'type': 'sequence', 'length': len(window_type)}
        return window_type

    def _apply_freq_lims(
        self,
        frequencies: np.ndarray,
        power_spectrogram: np.ndarray,
        power_db_norm: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.crop_freq_lims:
            return frequencies, power_spectrogram, power_db_norm

        try:
            fmin, fmax = self.freq_lims
        except (TypeError, ValueError):
            if not self.quiet:
                self.log.warning("freq_lims must be a (min, max) tuple; skipping crop.")
            return frequencies, power_spectrogram, power_db_norm

        try:
            fmin = float(fmin)
            fmax = float(fmax)
        except (TypeError, ValueError):
            if not self.quiet:
                self.log.warning("freq_lims must contain numeric values; skipping crop.")
            return frequencies, power_spectrogram, power_db_norm

        if fmax <= fmin:
            if not self.quiet:
                self.log.warning("freq_lims max must be greater than min; skipping crop.")
            return frequencies, power_spectrogram, power_db_norm

        mask = (frequencies >= fmin) & (frequencies <= fmax)
        if not np.any(mask):
            if not self.quiet:
                self.log.warning("freq_lims are outside the available frequency range; skipping crop.")
            return frequencies, power_spectrogram, power_db_norm

        return (
            frequencies[mask],
            power_spectrogram[mask, :],
            power_db_norm[mask, :],
        )

    def _resolve_window(self, win_length: int) -> np.ndarray:
        window_type = self.window_type
        is_window_spec = isinstance(window_type, tuple) and window_type and isinstance(window_type[0], str)
        if isinstance(window_type, np.ndarray):
            window = window_type
        elif not isinstance(window_type, str) and not is_window_spec:
            window = np.asarray(window_type)
        else:
            window = _cached_scipy_window(window_type, win_length)
        if len(window) != win_length:
            raise ValueError(f"Window length mismatch: expected {win_length}, got {len(window)}")
        return np.asarray(window, dtype=np.float32)

    def _get_torch_spectrogram_transform(
        self,
        *,
        nfft: int,
        win_length: int,
        hop_length: int,
        device: str,
        torch_window: Tuple[Any, dict],
    ):
        """Return a cached, stateless torchaudio transform."""
        window_fn, wkwargs = torch_window
        cache_key = (
            nfft,
            win_length,
            hop_length,
            str(device),
            getattr(window_fn, '__name__', repr(window_fn)),
            tuple(sorted(wkwargs.items())),
        )
        transform = self._torch_transform_cache.get(cache_key)
        if transform is not None:
            return transform

        with self._cache_lock:
            transform = self._torch_transform_cache.get(cache_key)
            if transform is None:
                transform = torchaudio.transforms.Spectrogram(
                    n_fft=nfft,
                    win_length=win_length,
                    hop_length=hop_length,
                    power=2.0,
                    center=False,
                    pad=0,
                    normalized=False,
                    window_fn=window_fn,
                    wkwargs=wkwargs,
                ).to(device)
                self._torch_transform_cache[cache_key] = transform
        return transform

    def _torch_window_spec(self) -> Optional[Tuple[Any, dict]]:
        if isinstance(self.window_type, np.ndarray):
            return None
        if isinstance(self.window_type, (list, tuple)) and not (
            isinstance(self.window_type, tuple) and self.window_type and isinstance(self.window_type[0], str)
        ):
            return None

        name = self.window_type
        param = None
        if isinstance(name, tuple):
            name = name[0]
            if len(self.window_type) > 1:
                param = self.window_type[1]
        if not isinstance(name, str):
            return None

        name = name.lower()
        if name == 'hanning':
            name = 'hann'

        if name == 'hann':
            return (torch.hann_window, {})
        if name == 'hamming':
            return (torch.hamming_window, {})
        if name == 'blackman':
            return (torch.blackman_window, {})
        if name == 'bartlett':
            return (torch.bartlett_window, {})
        if name == 'kaiser':
            beta = float(param) if param is not None else 14.0
            return (torch.kaiser_window, {'beta': beta})
        return None
    
    def compute_spectrogram(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        clip_meta: Optional[dict] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute spectrogram following MATLAB implementation.
        
        Args:
            audio_data: Audio signal
            sample_rate: Sample rate in Hz
            clip_meta: Optional clip metadata to trim spectrogram to target window
            
        Returns:
            Tuple of (frequencies, times, power_spectrogram, normalized_db)
        """
        win_length, nfft, hop_length, noverlap = self._resolve_fft_params(sample_rate)
        window = self._resolve_window(win_length)

        # Determine backend
        use_torch_backend = False
        device = 'cpu'
        backend = (self.backend or 'auto').lower()
        if backend not in {'auto', 'torch', 'scipy'}:
            self.log.warning(f"Unknown backend '{self.backend}'; falling back to auto.")
            backend = 'auto'
        
        if backend != 'scipy' and HAS_TORCH:
            requested_device = 'cpu'
            if self.torch_device is not None:
                requested_device = str(self.torch_device).strip().lower() or 'cpu'
            if requested_device == 'auto':
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            elif requested_device.startswith('cuda'):
                if torch.cuda.is_available():
                    device = requested_device
                else:
                    self.log.warning(f"Torch device '{requested_device}' requested but CUDA is unavailable; using CPU.")
                    device = 'cpu'
            elif requested_device == 'cpu':
                device = 'cpu'
            else:
                self.log.warning(f"Unknown torch_device '{self.torch_device}'; using CPU.")
                device = 'cpu'
            use_torch_backend = True
        elif backend == 'torch' and not HAS_TORCH:
            self.log.warning("Torch backend requested but torch/torchaudio is unavailable; using SciPy.")

        if use_torch_backend:
            torch_window = self._torch_window_spec()
            if torch_window is None:
                if backend == 'torch':
                    self.log.warning("Torch backend does not support the requested window type; using SciPy.")
                use_torch_backend = False

        scaling = self.scaling
        if scaling not in {'density', 'spectrum'}:
            self.log.warning(f"Unknown scaling '{self.scaling}'; using 'density'.")
            scaling = 'density'

        scale = 1.0
        if scaling == 'density':
            scale = 1.0 / (sample_rate * np.sum(window ** 2))
        elif scaling == 'spectrum':
            scale = 1.0 / (np.sum(window) ** 2)

        if use_torch_backend:
            try:
                spec_transform = self._get_torch_spectrogram_transform(
                    nfft=nfft,
                    win_length=win_length,
                    hop_length=hop_length,
                    device=device,
                    torch_window=torch_window,
                )
                
                # Prepare data
                audio_t = torch.from_numpy(audio_data.astype(np.float32, copy=False)).to(device)
                
                # Compute
                spec = spec_transform(audio_t)
                spec *= scale

                # Match scipy.signal.spectrogram's one-sided PSD convention.
                # Interior bins represent both positive and negative frequencies.
                if nfft % 2 == 0:
                    spec[1:-1] *= 2.0
                else:
                    spec[1:] *= 2.0
                
                if str(device).startswith('cuda'):
                    spec = spec.cpu()
                
                Sxx = spec.numpy()
                
                # Construct axes
                frequencies = np.linspace(0, sample_rate/2, Sxx.shape[0])
                # Report window centres, matching SciPy/MATLAB rather than frame starts.
                times = (
                    np.arange(Sxx.shape[1]) * hop_length + (win_length / 2.0)
                ) / sample_rate
                
            except Exception as e:
                self.log.warning(f"Torchaudio computation failed, falling back to Scipy: {e}")
                use_torch_backend = False

        if not use_torch_backend:
            # CPU path using scipy (equivalent to MATLAB spectrogram with 'psd')
            frequencies, times, Sxx = scipy.signal.spectrogram(
                audio_data,
                fs=sample_rate,
                window=window,
                nperseg=win_length,
                noverlap=noverlap,
                nfft=nfft,
                scaling=scaling,  # Power spectral density
                mode='psd',
                detrend=False,
            )
        
        backend_used = 'torch' if use_torch_backend else 'scipy'
        self._last_backend = backend_used
        self._last_scaling = scaling
        self._last_device = device if backend_used == 'torch' else None

        if clip_meta and clip_meta.get('clip_duration_seconds') is not None:
            offset = float(clip_meta.get('clip_offset_seconds', 0.0))
            duration = float(clip_meta.get('clip_duration_seconds', 0.0))
            if duration > 0:
                end_time = offset + duration
                keep_mask = (times >= offset) & (times <= end_time)
                if keep_mask.any():
                    times = times[keep_mask] - offset
                    Sxx = Sxx[:, keep_mask]
                elif not self.quiet:
                    self.log.warning("Clip trimming removed all spectrogram frames; check clip window settings.")

        # Normalize after time trimming so computation-only edge context cannot
        # change the relative dB values in the retained target interval.
        max_power = np.max(Sxx)
        if max_power > 0:
            # Reuse one working array rather than allocating abs(), division,
            # clipping, log, and multiplication intermediates.
            power_db_norm = np.asarray(Sxx / max_power)
            np.maximum(power_db_norm, 1e-10, out=power_db_norm)
            np.log10(power_db_norm, out=power_db_norm)
            power_db_norm *= 10.0
        else:
            power_db_norm = np.full_like(Sxx, -100.0)  # Very low dB value

        if not self.quiet:
            self.log.info(f"Spectrogram computed: {frequencies.shape[0]} freq bins, {times.shape[0]} time frames")
        return frequencies, times, Sxx, power_db_norm
    
    def plot_spectrogram(self, frequencies: np.ndarray, times: np.ndarray, 
                        power_db_norm: np.ndarray, title: str = "Spectrogram",
                        save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
        """
        Plot spectrogram following MATLAB visualization.
        
        Args:
            frequencies: Frequency array
            times: Time array
            power_db_norm: Normalized power in dB
            title: Plot title
            save_path: Optional path to save plot
            
        Returns:
            matplotlib Figure object
        """
        # Use thread lock for all matplotlib operations
        with _plot_lock:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # pcolormesh accepts 1-D axes directly; avoiding meshgrid saves two
            # full-size temporary arrays for large hydrophone spectrograms.
            pcm = ax.pcolormesh(times, frequencies, power_db_norm,
                           cmap=self.colormap, 
                           shading='auto',
                           vmin=self.clim[0], 
                           vmax=self.clim[1])
            
            # Set frequency limits
            ax.set_ylim(self.freq_lims)
            
            # Set log scale if requested
            if self.log_freq:
                ax.set_yscale('log')
            
            # Labels and title
            ax.set_xlabel('Time [s]')
            ax.set_ylabel('Frequency [Hz]')
            ax.set_title(title)
            
            # Colorbar
            cbar = plt.colorbar(pcm, ax=ax)
            cbar.set_label('PSD re max [dB]')
            
            # Save if requested
            if save_path:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
            if not self.quiet:
                self.log.info(f"Spectrogram plot saved: {save_path}")
        
        return fig
    
    def save_matlab_format(self, frequencies: np.ndarray, times: np.ndarray,
                          power_spectrogram: np.ndarray, power_db_norm: np.ndarray,
                          save_path: Union[str, Path],
                          metadata: Optional[dict] = None) -> None:
        """
        Save spectrogram data in MATLAB format.
        
        Args:
            frequencies: Frequency array
            times: Time array  
            power_spectrogram: Raw power spectrogram
            power_db_norm: Normalized power in dB
            save_path: Path to save .mat file
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save in MATLAB format (following original MATLAB code)
        payload = {
            'F': frequencies,
            'T': times,
            'P': power_spectrogram,
            'PdB_norm': power_db_norm
        }
        if metadata:
            payload['metadata'] = self._sanitize_metadata_for_mat(metadata)
            payload['metadata_json'] = json.dumps(metadata, default=str)
        scipy.io.savemat(save_path, payload)
        if not self.quiet:
            self.log.info(f"MATLAB data saved: {save_path}")
    
    def process_single_file(self, audio_path: Union[str, Path],
                           save_dir: Union[str, Path],
                           save_plot: bool = True,
                           save_mat: bool = True,
                           save_npy: bool = False,
                           extra_metadata: Optional[dict] = None,
                           retain_arrays: bool = True,
                           output_stem: Optional[str] = None) -> dict:
        """Process a single audio file and generate a spectrogram.

        Args:
            audio_path: Path to the input audio file.
            save_dir: Output directory for generated files.
            save_plot: Save a PNG plot (default: True).
            save_mat: Save MATLAB ``.mat`` output (default: True).
            save_npy: Save NumPy ``.npy`` output (default: False). The payload
                is a dict with ``F``, ``T``, ``P``, ``PdB_norm``, and metadata.
            extra_metadata: Optional extra metadata to store in outputs.
            retain_arrays: Keep full spectrogram arrays in the returned dict.
                Disable this for batch jobs to release memory after each file.
            output_stem: Optional filename stem for saved outputs. Defaults to
                the input audio filename stem.

        Returns:
            Dict with file paths, arrays, and metadata. Keys include:
            ``audio_file``, ``frequencies``, ``times``, ``power_spectrogram``,
            ``power_db_norm``, ``sample_rate``, ``duration``, ``metadata``,
            and any saved file paths (``mat_file``, ``png_file``, ``npy_file``).

        Raises:
            FileNotFoundError: If the audio file does not exist.

        Example:
            ```python
            generator = SpectrogramGenerator(win_dur=0.5, overlap=0.5)
            result = generator.process_single_file(
                \"example.flac\",
                \"./out\",
                save_mat=True,
                save_plot=False,
            )
            print(result[\"mat_file\"])
            ```
        """
        audio_path = Path(audio_path)
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Load audio
        audio_data, sample_rate, clip_meta = self.load_audio(audio_path)
        
        # Compute spectrogram
        frequencies, times, power_spec, power_db_norm = self.compute_spectrogram(
            audio_data,
            sample_rate,
            clip_meta=clip_meta,
        )
        frequencies, power_spec, power_db_norm = self._apply_freq_lims(
            frequencies,
            power_spec,
            power_db_norm,
        )
        
        # Create output filenames
        base_name = Path(output_stem).name if output_stem else audio_path.stem
        if not base_name or base_name in {'.', '..'}:
            raise ValueError("output_stem must contain a valid filename stem")
        mat_path = save_dir / f"{base_name}.mat"
        png_path = save_dir / f"{base_name}.png"
        npy_path = save_dir / f"{base_name}.npy"
        
        # Save outputs
        duration_seconds = len(audio_data) / sample_rate
        if clip_meta and clip_meta.get('clip_duration_seconds') is not None:
            duration_seconds = float(clip_meta.get('clip_duration_seconds', duration_seconds))
        win_length, nfft, hop_length, noverlap = self._resolve_fft_params(sample_rate)
        metadata = {
            'generated_utc': datetime.now(timezone.utc).isoformat(),
            'audio_file': str(audio_path),
            'sample_rate_hz': sample_rate,
            'duration_seconds': duration_seconds,
            'clip_meta': clip_meta,
            'fft_params': {
                'win_length': win_length,
                'nfft': nfft,
                'hop_length': hop_length,
                'noverlap': noverlap,
                'win_dur': float(self.win_dur),
                'overlap': float(self.overlap),
            },
            'spectrogram_settings': {
                'window_type': self._describe_window_type(),
                'nfft': self.nfft,
                'win_length': self.win_length,
                'hop_length': self.hop_length,
                'freq_lims': list(self.freq_lims) if self.freq_lims is not None else None,
                'colormap': self.colormap,
                'clim': list(self.clim) if self.clim is not None else None,
                'log_freq': self.log_freq,
                'crop_freq_lims': self.crop_freq_lims,
                'max_duration': self.max_duration,
                'clip_start': self.clip_start,
                'clip_end': self.clip_end,
                'clip_pad_seconds': self.clip_pad_seconds,
                'backend_requested': self.backend,
                'backend_used': getattr(self, '_last_backend', None),
                'scaling_requested': self.scaling,
                'scaling_used': getattr(self, '_last_scaling', None),
            },
        }
        if extra_metadata:
            metadata['extra_metadata'] = extra_metadata
        results = {
            'audio_file': str(audio_path),
            'frequencies': frequencies,
            'times': times,
            'power_spectrogram': power_spec,
            'power_db_norm': power_db_norm,
            'sample_rate': sample_rate,
            'duration': duration_seconds,
            'metadata': metadata,
        }
        
        if save_mat:
            self.save_matlab_format(frequencies, times, power_spec, power_db_norm, mat_path, metadata=metadata)
            results['mat_file'] = str(mat_path)

        if save_npy:
            self.save_numpy_format(frequencies, times, power_spec, power_db_norm, npy_path, metadata=metadata)
            results['npy_file'] = str(npy_path)
        
        if save_plot:
            fig = self.plot_spectrogram(frequencies, times, power_db_norm, 
                                       title=f"Spectrogram - {base_name}",
                                       save_path=png_path)
            plt.close(fig)  # Close to free memory
            results['png_file'] = str(png_path)

        if not retain_arrays:
            for key in ('frequencies', 'times', 'power_spectrogram', 'power_db_norm'):
                results.pop(key, None)
        
        return results

    def process_event(
        self,
        audio_path: Union[str, Path],
        save_dir: Union[str, Path],
        event_time_seconds: float,
        *,
        pad_before_seconds: float = 5.0,
        pad_after_seconds: Optional[float] = None,
        edge_padding_seconds: Union[float, str, None] = 'auto',
        save_plot: bool = True,
        save_mat: bool = True,
        save_npy: bool = False,
        extra_metadata: Optional[dict] = None,
        retain_arrays: bool = True,
        output_stem: Optional[str] = None,
    ) -> dict:
        """Generate an edge-safe spectrogram around an event in an audio file.

        The retained target interval extends before and after
        ``event_time_seconds``. Additional audio context is included only while
        computing the STFT, then frames are trimmed back to the target interval.
        The default ``'auto'`` context is half the resolved STFT window on each
        side, so every retained frame has a complete analysis window.

        Args:
            audio_path: Source audio file.
            save_dir: Directory for generated outputs.
            event_time_seconds: Event offset from the start of the audio file.
            pad_before_seconds: Target data retained before the event. Defaults
                to five seconds.
            pad_after_seconds: Target data retained after the event. Defaults to
                ``pad_before_seconds``.
            edge_padding_seconds: Extra computation-only context on each side.
                Use ``'auto'`` (default) to derive half the STFT window length.
            save_plot: Save a PNG plot.
            save_mat: Save MATLAB output.
            save_npy: Save NumPy output.
            extra_metadata: Optional additional output metadata.
            retain_arrays: Keep arrays in the returned result.
            output_stem: Optional saved-output filename stem.

        Returns:
            The usual single-file result plus event and target-window metadata.
        """
        try:
            event_time_seconds = float(event_time_seconds)
            pad_before_seconds = float(pad_before_seconds)
            if pad_after_seconds is None:
                pad_after_seconds = pad_before_seconds
            pad_after_seconds = float(pad_after_seconds)
        except (TypeError, ValueError) as exc:
            raise ValueError("Event time and padding values must be numeric") from exc

        if not all(np.isfinite(value) for value in (
            event_time_seconds,
            pad_before_seconds,
            pad_after_seconds,
        )):
            raise ValueError("Event time and padding values must be finite")
        if event_time_seconds < 0:
            raise ValueError("event_time_seconds must be >= 0")
        if pad_before_seconds < 0 or pad_after_seconds < 0:
            raise ValueError("Event padding values must be >= 0")
        if pad_before_seconds + pad_after_seconds <= 0:
            raise ValueError("At least one event padding value must be > 0")

        target_start = max(0.0, event_time_seconds - pad_before_seconds)
        target_end = event_time_seconds + pad_after_seconds
        if target_end <= target_start:
            raise ValueError("The event target interval is empty")

        if edge_padding_seconds is not None and not (
            isinstance(edge_padding_seconds, str)
            and edge_padding_seconds.strip().lower() == 'auto'
        ):
            try:
                explicit_edge_padding = float(edge_padding_seconds)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "edge_padding_seconds must be a non-negative float or 'auto'"
                ) from exc
            if not np.isfinite(explicit_edge_padding) or explicit_edge_padding < 0:
                raise ValueError(
                    "edge_padding_seconds must be a non-negative finite value"
                )
            edge_padding_seconds = explicit_edge_padding

        # A shallow copy keeps the thread-safe FFT cache but isolates clip
        # settings so this convenience method never mutates the caller's
        # generator, including when it is reused for directory processing.
        event_generator = copy.copy(self)
        event_generator.clip_start = target_start
        event_generator.clip_end = target_end
        event_generator.clip_pad_seconds = edge_padding_seconds
        event_generator.max_duration = None

        event_metadata = {
            **(extra_metadata or {}),
            'event_spectrogram': {
                'event_time_seconds': event_time_seconds,
                'target_start_seconds': target_start,
                'target_end_seconds': target_end,
                'pad_before_seconds': event_time_seconds - target_start,
                'pad_after_seconds': pad_after_seconds,
                'edge_padding_requested': edge_padding_seconds,
            },
        }
        if output_stem is None:
            event_milliseconds = int(round(event_time_seconds * 1000.0))
            output_stem = f"{Path(audio_path).stem}_event_{event_milliseconds}ms"

        result = event_generator.process_single_file(
            audio_path,
            save_dir,
            save_plot=save_plot,
            save_mat=save_mat,
            save_npy=save_npy,
            extra_metadata=event_metadata,
            retain_arrays=retain_arrays,
            output_stem=output_stem,
        )
        clip_meta = result.get('metadata', {}).get('clip_meta') or {}
        result.update({
            'event_time_seconds': event_time_seconds,
            'target_start_seconds': target_start,
            'target_end_seconds': target_end,
            'pad_before_seconds': event_time_seconds - target_start,
            'pad_after_seconds': pad_after_seconds,
            'edge_padding_seconds': clip_meta.get('clip_pad_seconds'),
        })
        return result

    def save_numpy_format(
        self,
        frequencies: np.ndarray,
        times: np.ndarray,
        power_spectrogram: np.ndarray,
        power_db_norm: np.ndarray,
        save_path: Union[str, Path],
        metadata: Optional[dict] = None,
    ) -> None:
        """
        Save spectrogram data in numpy format.

        Notes:
            This uses np.save with a dict payload (requires allow_pickle on load).
            Metadata is stored under the "metadata" key when provided.
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'F': frequencies,
            'T': times,
            'P': power_spectrogram,
            'PdB_norm': power_db_norm,
        }
        if metadata:
            payload['metadata'] = metadata
        np.save(save_path, payload, allow_pickle=True)
        if not self.quiet:
            self.log.info(f"Numpy data saved: {save_path}")
    
    def process_directory(self, input_dir: Union[str, Path], 
                         save_dir: Union[str, Path],
                         file_extensions: Optional[List[str]] = None,
                         save_plot: bool = True,
                         save_mat: bool = True,
                         save_npy: bool = False,
                         max_workers: Optional[int] = None,
                         retain_arrays: bool = False) -> List[dict]:
        """Process all audio files in a directory.

        Args:
            input_dir: Directory containing audio files.
            save_dir: Directory to save outputs.
            file_extensions: Audio file extensions to include.
            save_plot: Save PNG plots (default: True).
            save_mat: Save MATLAB ``.mat`` files (default: True).
            save_npy: Save NumPy ``.npy`` files (default: False).
            max_workers: Batch worker count. Defaults to at most four to limit
                peak memory; raise it explicitly for many short files.
            retain_arrays: Keep full arrays in each result. Defaults to False
                so long batch jobs do not retain every spectrogram in memory.

        Returns:
            List of processing result dicts (one per file).

        Raises:
            FileNotFoundError: If the input directory does not exist.
        """
        input_dir = Path(input_dir)
        save_dir = Path(save_dir)
        
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        # Find all audio files
        file_extensions = file_extensions or ['.wav', '.flac', '.mp3', '.m4a']
        audio_files = []
        for ext in file_extensions:
            audio_files.extend(input_dir.glob(f"*{ext}"))
            audio_files.extend(input_dir.glob(f"*{ext.upper()}"))
        
        audio_files = sorted(set(audio_files))
        if not audio_files:
            logger.warning(f"No audio files found in {input_dir} with extensions {file_extensions}")
            return []
        
        if not self.quiet:
            self.log.info(f"Processing {len(audio_files)} audio files from {input_dir}")
        
        indexed_results = [None] * len(audio_files)
        if max_workers is None:
            # Four workers are within a few percent of eight for Torch on a
            # typical Apple Silicon Mac while roughly halving worst-case RAM.
            max_workers = min(4, os.cpu_count() or 4, len(audio_files))
        else:
            max_workers = max(1, min(int(max_workers), len(audio_files)))
        total = len(audio_files)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(
                    self.process_single_file,
                    audio_file,
                    save_dir,
                    save_plot=save_plot,
                    save_mat=save_mat,
                    save_npy=save_npy,
                    retain_arrays=retain_arrays,
                ): (index, audio_file)
                for index, audio_file in enumerate(audio_files)
            }
            completed = 0
            for future in as_completed(future_to_file):
                index, audio_file = future_to_file[future]
                completed += 1
                try:
                    result = future.result()
                    indexed_results[index] = result
                    if not self.quiet:
                        self.log.info(f"Processed {completed}/{total}: {audio_file.name}")
                except Exception as e:
                    if not self.quiet:
                        self.log.error(f"Error processing {audio_file}: {e}")
                    indexed_results[index] = {
                        'audio_file': str(audio_file),
                        'error': str(e)
                    }
                if not self.quiet:
                    bar_len = 20
                    filled = int(bar_len * completed / total)
                    bar = '#' * filled + '-' * (bar_len - filled)
                    sys.stdout.write(f"\rSpectrograms: [{bar}] {completed}/{total}")
                    sys.stdout.flush()
            if not self.quiet:
                sys.stdout.write("\n")
        
        if not self.quiet:
            self.log.info(f"Completed processing {len(indexed_results)} files")
        return indexed_results
