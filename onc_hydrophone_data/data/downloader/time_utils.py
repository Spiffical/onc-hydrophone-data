import math
import os
import re
from datetime import datetime, timedelta, timezone, tzinfo
from typing import Any, Optional

from ...onc.common import ensure_timezone_aware, format_iso_utc
from .constants import FIVE_MINUTES_SECONDS, FILENAME_TS_PATTERN


def _resolve_timezone(timezone_str: Optional[Any]) -> Optional[tzinfo]:
    if timezone_str is None:
        return None
    if isinstance(timezone_str, tzinfo):
        return timezone_str
    if not isinstance(timezone_str, str):
        raise ValueError(f"Timezone must be a string or tzinfo, got {type(timezone_str).__name__}")
    cleaned = timezone_str.strip()
    if not cleaned:
        return None
    upper = cleaned.upper()
    if upper in ('UTC', 'Z'):
        return timezone.utc
    offset_match = re.match(r'^([+-])(\d{2}):?(\d{2})$', cleaned)
    if offset_match:
        sign = 1 if offset_match.group(1) == '+' else -1
        hours = int(offset_match.group(2))
        minutes = int(offset_match.group(3))
        return timezone(sign * timedelta(hours=hours, minutes=minutes))
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo(cleaned)
    except Exception as exc:
        raise ValueError(f"Unknown timezone '{timezone_str}'. Use IANA names (e.g., America/Vancouver).") from exc


def _parse_timestamp_value(value: Any, timezone_str: Optional[Any] = None) -> datetime:
    """Normalize various timestamp formats to timezone-aware UTC datetimes."""
    tz = _resolve_timezone(timezone_str) if timezone_str else None
    if isinstance(value, datetime):
        dt_obj = value
    elif isinstance(value, str):
        cleaned = value.strip()
        if cleaned.endswith('Z'):
            cleaned = cleaned[:-1] + '+00:00'
        try:
            dt_obj = datetime.fromisoformat(cleaned)
        except ValueError:
            dt_obj = datetime.strptime(value, '%Y-%m-%d %H:%M:%S')
    elif isinstance(value, (list, tuple)) and len(value) >= 6:
        dt_obj = datetime(
            int(value[0]), int(value[1]), int(value[2]),
            int(value[3]), int(value[4]), int(value[5])
        )
    else:
        raise ValueError(f"Unsupported timestamp format: {value}")

    if dt_obj.tzinfo is None:
        dt_obj = dt_obj.replace(tzinfo=tz or timezone.utc)
    return dt_obj.astimezone(timezone.utc)


def _floor_to_window(dt_obj: datetime, seconds: int = FIVE_MINUTES_SECONDS) -> datetime:
    epoch = ensure_timezone_aware(dt_obj).timestamp()
    floored = math.floor(epoch / seconds) * seconds
    return datetime.fromtimestamp(floored, tz=timezone.utc)


def _format_iso_utc(dt_obj: datetime) -> str:
    return format_iso_utc(ensure_timezone_aware(dt_obj))


def _ceil_to_window(dt_obj: datetime, seconds: int = FIVE_MINUTES_SECONDS) -> datetime:
    epoch = ensure_timezone_aware(dt_obj).timestamp()
    ceiled = math.ceil(epoch / seconds) * seconds
    return datetime.fromtimestamp(ceiled, tz=timezone.utc)


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
