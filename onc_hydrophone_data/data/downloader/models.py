from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional


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
    data_product_options: Optional[Dict[str, Any]] = None
    description: Optional[str] = None
    output_name: Optional[str] = None
