import ast
import csv
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from ...onc.common import ensure_timezone_aware
from .constants import FIVE_MINUTES_SECONDS
from .models import TimestampRequest


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
    timezone_str = base.get('timezone') or base.get('timezone_str') or base.get('tz')
    if timestamp_value is None:
        raise ValueError(f"Missing timestamp/start for device {device_code}")
    ts = self._parse_timestamp_value(timestamp_value, timezone_str)
    clip_range_start = self._parse_timestamp_value(start_value, timezone_str) if start_value else ts
    clip_range_end: Optional[datetime] = None
    span_seconds = 0.0
    if end_value:
        clip_range_end = self._parse_timestamp_value(end_value, timezone_str)
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
    data_product_options = base.get('data_product_options') or base.get('hsd_options')
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
        data_product_options=data_product_options,
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


def _parse_csv_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(int(value))
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ('true', 't', 'yes', 'y', '1'):
            return True
        if text in ('false', 'f', 'no', 'n', '0'):
            return False
    return None


def _parse_csv_number(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _parse_csv_data_product_options(value: Any) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(text)
            except (ValueError, SyntaxError):
                return None
    return None


def _normalize_csv_request_row(
    self,
    row: Dict[str, Any],
    *,
    data_product_options_key: str,
) -> Dict[str, Any]:
    cleaned: Dict[str, Any] = {}
    for key, value in row.items():
        if key is None:
            continue
        key = key.strip()
        if value is None:
            continue
        if isinstance(value, str):
            value = value.strip()
            if not value:
                continue
        cleaned[key] = value

    if 'deviceCode' not in cleaned:
        for alias in ('device_code', 'device'):
            if alias in cleaned:
                cleaned['deviceCode'] = cleaned.pop(alias)
                break

    for alias, target in (('spectrogram', 'download_spectrogram'), ('audio', 'download_audio')):
        if alias in cleaned and target not in cleaned:
            cleaned[target] = cleaned.pop(alias)

    bool_fields = ('download_audio', 'download_spectrogram', 'clip')
    for key in bool_fields:
        if key in cleaned:
            parsed = self._parse_csv_bool(cleaned[key])
            if parsed is None:
                self.logger.warning(f"CSV field {key} has invalid boolean: {cleaned[key]!r}; skipping")
                cleaned.pop(key, None)
            else:
                cleaned[key] = parsed

    number_fields = ('pad_seconds', 'pad_before_seconds', 'pad_after_seconds', 'duration_seconds')
    for key in number_fields:
        if key in cleaned:
            parsed = self._parse_csv_number(cleaned[key])
            if parsed is None:
                self.logger.warning(f"CSV field {key} has invalid number: {cleaned[key]!r}; skipping")
                cleaned.pop(key, None)
            else:
                cleaned[key] = parsed

    if 'spectral_downsample' in cleaned:
        parsed = self._parse_csv_number(cleaned['spectral_downsample'])
        if parsed is None:
            self.logger.warning(
                f"CSV field spectral_downsample has invalid number: {cleaned['spectral_downsample']!r}; skipping"
            )
            cleaned.pop('spectral_downsample', None)
        else:
            cleaned['spectral_downsample'] = int(parsed)

    if 'spectrogram_format' in cleaned and isinstance(cleaned['spectrogram_format'], str):
        cleaned['spectrogram_format'] = cleaned['spectrogram_format'].lower()
    if 'audio_extension' in cleaned and isinstance(cleaned['audio_extension'], str):
        cleaned['audio_extension'] = cleaned['audio_extension'].lower()

    dpo_value = cleaned.pop(data_product_options_key, None)
    dpo = self._parse_csv_data_product_options(dpo_value)
    if dpo_value is not None and dpo is None:
        self.logger.warning(
            f"CSV field {data_product_options_key} has invalid JSON: {dpo_value!r}; skipping"
        )
    if dpo:
        cleaned['data_product_options'] = dpo

    return cleaned


def _load_request_payload_from_csv(
    self,
    csv_path: str,
    *,
    defaults: Optional[Dict[str, Any]] = None,
    data_product_options_key: str = 'data_product_options',
) -> Dict[str, Any]:
    """Load timestamp requests from a CSV file into the JSON payload shape."""
    requests: List[Dict[str, Any]] = []
    with open(csv_path, newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("CSV file is missing a header row")
        for row in reader:
            normalized = self._normalize_csv_request_row(
                row,
                data_product_options_key=data_product_options_key,
            )
            if normalized:
                requests.append(normalized)

    payload: Dict[str, Any] = {'requests': requests}
    if defaults:
        payload['defaults'] = defaults
    return payload
