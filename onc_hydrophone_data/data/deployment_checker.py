"""
ONC Hydrophone Deployment Date Checker

This module provides functionality to check deployment dates for hydrophones
from Ocean Networks Canada (ONC) and helps users choose appropriate dates
for downloading hydrophone data.

Based on functionality from: https://github.com/Spiffical/hydrophonedatarequests
"""
import logging
import time
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional, Union
import concurrent.futures
from dataclasses import dataclass
from .location_mappings import (
    build_friendly_mapping_names,
    build_reverse_location_mapping,
    get_system_for_location,
)
from ..onc.common import ensure_timezone_aware, format_iso_utc

try:
    from dateutil import parser as dtparse
    from dateutil.tz import gettz, UTC
except ImportError:
    raise ImportError("ERROR: 'python-dateutil' library not found. Please install it: pip install python-dateutil")

try:
    from onc import ONC
    from requests.exceptions import HTTPError
except ImportError:
    raise ImportError("ERROR: 'onc-python' library not found. Please install it: pip install onc-python")


@dataclass
class DeploymentInfo:
    """Information about a hydrophone deployment."""
    device_code: str
    device_id: Optional[str]
    location_code: str
    location_name: str
    begin_date: datetime
    end_date: Optional[datetime]
    latitude: float
    longitude: float
    depth: Optional[float]
    citation: Optional[str]
    # Optional granular info for nested locations (e.g., Hydrophone A/B/C within an array)
    position_name: Optional[str] = None
    location_path: Optional[Tuple[str, ...]] = None
    has_data: bool = False


class HydrophoneDeploymentChecker:
    """Check deployment dates and data availability for ONC hydrophones."""
    
    def __init__(self, onc_token: str, debug: bool = False):
        """
        Initialize the deployment checker.
        
        Args:
            onc_token: ONC API token
            debug: Enable debug logging
        """
        self.onc = ONC(onc_token, showInfo=debug, showWarning=debug)
        # Best-effort: quiet the ONC client if it supports these toggles
        for attr, value in (
            ('showInfo', False),
            ('showWarning', False),
            ('showWarnings', False),
            ('showErrors', False),
        ):
            try:
                setattr(self.onc, attr, value)
            except Exception:
                pass
        self.debug = debug
        self._location_cache = {}
        self._location_paths = {}
        self._location_cache_built = False
        self._location_paths_built = False
        self._deployments_cache = None
        self._deployments_cache_at = None
        self._archive_cache = {}
        self._device_deployments_cache = {}
        self._device_deployments_cache_at = {}
        
        # Setup logging
        log_level = logging.DEBUG if debug else logging.INFO
        logging.basicConfig(level=log_level, 
                          format='%(asctime)s - %(levelname)s - %(message)s')
        
    def get_all_hydrophone_deployments(self) -> List[DeploymentInfo]:
        """
        Get all hydrophone deployments from ONC.
        
        Returns:
            List of ``DeploymentInfo`` objects for all deployments, including
            location metadata (name/code, lat/lon, depth) and deployment dates.
        """
        logging.info("Fetching hydrophone devices from ONC...")
        
        # Get all hydrophone devices
        hydrophones = self.onc.getDevices({"deviceCategoryCode": "HYDROPHONE"})
        if not isinstance(hydrophones, list):
            raise ValueError("Unexpected response format from getDevices")
        
        if not hydrophones:
            raise ValueError("No hydrophone devices found")
        
        logging.info(f"Found {len(hydrophones)} hydrophone device(s)")
        
        # Get location information for mapping
        self._build_location_cache()
        
        # Get deployments for all hydrophones in parallel
        all_deployments = self._get_deployments_parallel(hydrophones)
        
        # Convert to DeploymentInfo objects
        deployment_infos = []
        for dep in all_deployments:
            try:
                deployment_info = self._parse_deployment(dep)
                if deployment_info:
                    deployment_infos.append(deployment_info)
            except Exception as e:
                if self.debug:
                    logging.warning(f"Error parsing deployment: {e}")
                continue
        
        logging.info(f"Found {len(deployment_infos)} valid deployment(s)")
        return deployment_infos
    
    def find_deployments_by_time_range(self, 
                                     start_date: Union[str, datetime], 
                                     end_date: Union[str, datetime],
                                     timezone_str: str = 'UTC') -> List[DeploymentInfo]:
        """
        Find deployments that overlap with a specific time range.
        
        Args:
            start_date: Start date (string or datetime)
            end_date: End date (string or datetime)
            timezone_str: Timezone for date interpretation
            
        Returns:
            List of deployments that overlap with the time range
        """
        # Parse dates
        if isinstance(start_date, str):
            start_dt = dtparse.parse(start_date)
        else:
            start_dt = start_date
            
        if isinstance(end_date, str):
            end_dt = dtparse.parse(end_date)
        else:
            end_dt = end_date
        
        # Handle timezone
        if timezone_str != 'UTC':
            tz = gettz(timezone_str)
            if start_dt.tzinfo is None:
                start_dt = start_dt.replace(tzinfo=tz)
            if end_dt.tzinfo is None:
                end_dt = end_dt.replace(tzinfo=tz)
        
        # Convert to UTC
        start_utc = start_dt.astimezone(UTC) if start_dt.tzinfo else start_dt.replace(tzinfo=UTC)
        end_utc = end_dt.astimezone(UTC) if end_dt.tzinfo else end_dt.replace(tzinfo=UTC)
        
        logging.info(f"Searching for deployments overlapping: {start_utc} to {end_utc}")
        
        # Get all deployments
        all_deployments = self._get_cached_deployments()
        
        # Filter by time overlap
        overlapping = []
        for dep in all_deployments:
            # Check if deployment overlaps with requested time range
            if (dep.begin_date <= end_utc) and (not dep.end_date or dep.end_date >= start_utc):
                overlapping.append(dep)
        
        logging.info(f"Found {len(overlapping)} overlapping deployment(s)")
        return overlapping
    
    def check_data_availability(self, 
                              deployments: List[DeploymentInfo], 
                              start_date: datetime, 
                              end_date: datetime,
                              check_archive: bool = False) -> List[DeploymentInfo]:
        """
        Check data availability for deployments in the specified time range.
        
        Args:
            deployments: List of deployments to check
            start_date: Start date for data availability check
            end_date: End date for data availability check
            check_archive: If True, check for archive files, otherwise check data products
            
        Returns:
            List of deployments with data availability marked
        """
        logging.info(f"Checking data availability for {len(deployments)} deployment(s)...")
        
        # Group by device code for efficiency
        device_to_deployments = defaultdict(list)
        for dep in deployments:
            device_to_deployments[dep.device_code].append(dep)
        
        # Check availability for each device
        if check_archive:
            device_availability = self._check_archive_availability_parallel(
                list(device_to_deployments.keys()), start_date, end_date)
        else:
            device_availability = self._check_product_availability_parallel(
                list(device_to_deployments.keys()))
        
        # Update deployment info with availability
        available_deployments = []
        for device_code, has_data in device_availability.items():
            for dep in device_to_deployments[device_code]:
                dep.has_data = has_data
                if has_data:
                    available_deployments.append(dep)
        
        logging.info(f"Found {len(available_deployments)} deployment(s) with available data")
        return available_deployments
    
    def get_deployment_date_ranges(self, device_codes: Optional[List[str]] = None) -> Dict[str, List[Tuple[datetime, Optional[datetime]]]]:
        """
        Get deployment date ranges for specific device codes or all hydrophones.
        
        Args:
            device_codes: Optional list of device codes to check. If None, checks all.
            
        Returns:
            Dict mapping device codes to lists of ``(start_date, end_date)`` tuples.
        """
        all_deployments = self._get_cached_deployments()
        
        # Filter by device codes if specified
        if device_codes:
            all_deployments = [dep for dep in all_deployments if dep.device_code in device_codes]
        
        # Group by device code
        date_ranges = defaultdict(list)
        for dep in all_deployments:
            date_ranges[dep.device_code].append((dep.begin_date, dep.end_date))
        
        # Sort date ranges
        for device_code in date_ranges:
            date_ranges[device_code].sort(key=lambda x: x[0])
        
        return dict(date_ranges)
    
    def print_deployment_summary(self, deployments: List[DeploymentInfo], show_data_availability: bool = True):
        """
        Print a formatted summary of deployments.
        
        Args:
            deployments: List of deployments to summarize
            show_data_availability: Whether to show data availability status
        """
        if not deployments:
            print("No deployments found.")
            return
        
        print(f"\n{'='*80}")
        print(f"HYDROPHONE DEPLOYMENT SUMMARY ({len(deployments)} deployments)")
        print(f"{'='*80}")
        
        # Group by location for better organization
        by_location = defaultdict(list)
        for dep in deployments:
            by_location[dep.location_name or dep.location_code].append(dep)
        
        for location, deps in sorted(by_location.items()):
            print(f"\n📍 Location: {location}")
            print("-" * 60)
            
            for dep in sorted(deps, key=lambda x: x.begin_date):
                end_str = dep.end_date.strftime('%Y-%m-%d') if dep.end_date else "ongoing"
                data_status = " ✅ Has Data" if show_data_availability and dep.has_data else " ❌ No Data" if show_data_availability else ""
                
                print(f"  🔹 {dep.device_code}")
                print(f"     Period: {dep.begin_date.strftime('%Y-%m-%d')} to {end_str}{data_status}")
                if getattr(dep, "position_name", None) and dep.position_name != dep.location_name:
                    print(f"     Position: {dep.position_name}")
                if dep.depth:
                    print(f"     Depth: {dep.depth}m")
                if dep.latitude and dep.longitude:
                    print(f"     Location: {dep.latitude:.4f}°N, {dep.longitude:.4f}°W")
                print()

    def collect_hydrophone_inventory(
        self,
        *,
        include_inactive: bool = True,
    ) -> Dict[str, Any]:
        """
        Collect hydrophone deployment inventory with current and history views.

        Args:
            include_inactive: If True, include inactive devices in the
                per‑device summaries when building the inventory.

        Returns:
            Dict with ``current`` and ``history`` lists of records.
        """
        deployments = self.get_all_hydrophone_deployments()
        code_to_mapping_names = build_reverse_location_mapping()
        now = datetime.now(timezone.utc)

        def format_dep(dep: DeploymentInfo) -> Dict[str, Any]:
            raw_names = code_to_mapping_names.get(dep.location_code, [])
            friendly_names = build_friendly_mapping_names(raw_names)
            systems = sorted({
                get_system_for_location(name)
                for name in raw_names
                if get_system_for_location(name) != 'Unknown'
            })
            location_path = " > ".join(dep.location_path) if dep.location_path else None
            active = dep.end_date is None or dep.end_date >= now
            return {
                'device_code': dep.device_code,
                'device_id': dep.device_id,
                'location_code': dep.location_code,
                'location_name': dep.location_name,
                'mapped_location_names': ", ".join(friendly_names) if friendly_names else None,
                'mapped_systems': ", ".join(systems) if systems else None,
                'begin_date': dep.begin_date,
                'end_date': dep.end_date,
                'depth_m': dep.depth,
                'latitude': dep.latitude,
                'longitude': dep.longitude,
                'position_name': dep.position_name,
                'location_path': location_path,
                'has_data': dep.has_data,
                'active': active,
            }

        history_rows: List[Dict[str, Any]] = []
        current_rows: List[Dict[str, Any]] = []

        by_device: Dict[str, List[DeploymentInfo]] = defaultdict(list)
        for dep in deployments:
            by_device[dep.device_code].append(dep)

        for device_code, deps in by_device.items():
            deps_sorted = sorted(deps, key=lambda d: d.begin_date)
            deployment_count = len(deps_sorted)
            history_start = deps_sorted[0].begin_date if deps_sorted else None
            history_end: Optional[datetime]
            if any(dep.end_date is None for dep in deps_sorted):
                history_end = None
            else:
                history_end = max((dep.end_date for dep in deps_sorted if dep.end_date), default=None)

            for dep in deps_sorted:
                history_rows.append(format_dep(dep))

            for dep in deps_sorted:
                active = dep.end_date is None or dep.end_date >= now
                if not active and not include_inactive:
                    continue
                if active:
                    row = format_dep(dep)
                    row['deployment_count'] = deployment_count
                    row['history_start'] = history_start
                    row['history_end'] = history_end
                    current_rows.append(row)

        return {
            'generated_at': now,
            'current': current_rows,
            'history': history_rows,
        }

    def render_hydrophone_inventory_table(
        self,
        inventory: Dict[str, Any],
        *,
        view: str = 'current',
        max_rows: Optional[int] = 50,
    ):
        """
        Render the inventory into a table (DataFrame if pandas is available).

        Args:
            inventory: Output of ``collect_hydrophone_inventory``.
            view: ``current`` or ``history``.
            max_rows: Maximum number of rows to display (None = no limit).

        Returns:
            A pandas DataFrame if pandas is installed, otherwise a Markdown string.
        """
        rows = list(inventory.get(view, []))
        if max_rows is not None:
            rows = rows[:max_rows]
        if not rows:
            return "No rows to display."

        def fmt_value(value: Any) -> Any:
            if isinstance(value, datetime):
                return value.strftime('%Y-%m-%d %H:%M')
            if value is None:
                return ''
            if isinstance(value, float):
                return round(value, 4)
            return value

        columns_current = [
            'device_code',
            'device_id',
            'location_code',
            'location_name',
            'mapped_location_names',
            'mapped_systems',
            'begin_date',
            'end_date',
            'depth_m',
            'latitude',
            'longitude',
            'deployment_count',
            'history_start',
            'history_end',
        ]
        columns_history = [
            'device_code',
            'device_id',
            'location_code',
            'location_name',
            'mapped_location_names',
            'mapped_systems',
            'begin_date',
            'end_date',
            'depth_m',
            'latitude',
            'longitude',
            'position_name',
            'location_path',
        ]
        columns = columns_current if view == 'current' else columns_history

        table_rows = [
            {col: fmt_value(row.get(col)) for col in columns}
            for row in rows
        ]

        try:
            import pandas as pd  # type: ignore
        except Exception:
            pd = None  # type: ignore

        if pd is not None:
            return pd.DataFrame(table_rows, columns=columns)

        header = '| ' + ' | '.join(columns) + ' |'
        separator = '| ' + ' | '.join(['---'] * len(columns)) + ' |'
        lines = [header, separator]
        for row in table_rows:
            line = '| ' + ' | '.join(str(row.get(col, '')) for col in columns) + ' |'
            lines.append(line)
        return '\n'.join(lines)

    def show_hydrophone_inventory_table(
        self,
        inventory: Dict[str, Any],
        *,
        view: str = 'current',
        max_rows: Optional[int] = 50,
    ):
        """
        Display the inventory table in notebooks or print a Markdown fallback.

        Args:
            inventory: Output of ``collect_hydrophone_inventory``.
            view: ``current`` or ``history``.
            max_rows: Maximum number of rows to display (None = no limit).

        Returns:
            The rendered table (DataFrame or Markdown string).
        """
        table = self.render_hydrophone_inventory_table(
            inventory,
            view=view,
            max_rows=max_rows,
        )
        try:
            from IPython.display import Markdown, display  # type: ignore
        except Exception:
            display = None  # type: ignore
            Markdown = None  # type: ignore

        if hasattr(table, 'to_string'):
            if display is not None:
                display(table)
            else:
                print(table.to_string(index=False))
        else:
            if display is not None and Markdown is not None and isinstance(table, str):
                display(Markdown(table))
            else:
                print(table)
        return table

    def _filter_inventory_by_devices(
        self,
        inventory: Dict[str, Any],
        *,
        device_codes: Optional[List[str]] = None,
        device_ids: Optional[List[Union[str, int]]] = None,
    ) -> Dict[str, Any]:
        rows = list(inventory.get('history', []))
        if not rows:
            return {'history': []}
        codes_set = {code for code in (device_codes or []) if code}
        ids_set = {str(device_id) for device_id in (device_ids or []) if device_id is not None}

        if not codes_set and not ids_set:
            return {'history': rows}

        filtered = []
        for row in rows:
            if codes_set and row.get('device_code') in codes_set:
                filtered.append(row)
                continue
            if ids_set and row.get('device_id') in ids_set:
                filtered.append(row)
                continue
        return {'history': filtered}

    def show_device_deployments(
        self,
        *,
        device_codes: Optional[List[str]] = None,
        device_ids: Optional[List[Union[str, int]]] = None,
        max_rows: Optional[int] = 50,
        inventory: Optional[Dict[str, Any]] = None,
    ):
        """
        Show deployment history for specific device codes or device IDs.

        Args:
            device_codes: List of ONC device codes to filter by.
            device_ids: List of numeric device IDs to filter by.
            max_rows: Maximum number of rows to display (None = no limit).
            inventory: Optional inventory output to reuse (avoids re-fetch).

        Returns:
            The rendered table (DataFrame or Markdown string).
        """
        inventory = inventory or self.collect_hydrophone_inventory()
        filtered = self._filter_inventory_by_devices(
            inventory,
            device_codes=device_codes,
            device_ids=device_ids,
        )
        return self.show_hydrophone_inventory_table(
            filtered,
            view='history',
            max_rows=max_rows,
        )

    def get_device_availability(
        self,
        device_code: str,
        *,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        timezone_str: str = 'UTC',
        bin_size: str = 'day',
        max_days_per_request: int = 60,
        progress: Optional[Any] = None,
        quiet: bool = True,
        max_workers: int = 4,
        request_delay_seconds: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Build deployment-aware availability bins for a specific device.

        Args:
            device_code: ONC device code to evaluate.
            start_date: Optional start date for availability window.
            end_date: Optional end date for availability window.
            timezone_str: Timezone for binning and display.
            bin_size: "day" or "hour" binning for availability.
            max_days_per_request: Chunk size for archive queries.

        Returns:
            Dict with device metadata, deployments, bins, and deployment_summary.
            Bin records include: start, end, coverage (0-1), status, deployment_index.
            Bins align to day/hour boundaries in the requested timezone.
        """
        if not device_code:
            raise ValueError("device_code is required")

        bin_size = (bin_size or 'day').lower().strip()
        if bin_size in ('days',):
            bin_size = 'day'
        if bin_size in ('hours',):
            bin_size = 'hour'
        if bin_size not in ('day', 'hour'):
            raise ValueError("bin_size must be 'day' or 'hour'")

        tz = gettz(timezone_str) if timezone_str else UTC
        if tz is None:
            raise ValueError(f"Unknown timezone: {timezone_str}")

        device_deployments = self._get_device_deployments(device_code)
        device_deployments.sort(key=lambda d: d.begin_date)
        deduped: List[DeploymentInfo] = []
        seen = set()
        for dep in device_deployments:
            key = (
                dep.device_code,
                dep.location_code,
                dep.begin_date,
                dep.end_date,
                dep.position_name,
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(dep)
        device_deployments = deduped

        if not device_deployments:
            return {
                'device_code': device_code,
                'timezone': timezone_str,
                'bin_size': bin_size,
                'start': None,
                'end': None,
                'deployments': [],
                'bins': [],
                'deployment_summary': [],
            }

        now_utc = datetime.now(timezone.utc)
        dep_starts = [
            ensure_timezone_aware(dep.begin_date, tz=timezone.utc).astimezone(tz)
            for dep in device_deployments
        ]
        dep_ends = [
            ensure_timezone_aware(dep.end_date, tz=timezone.utc).astimezone(tz)
            for dep in device_deployments
            if dep.end_date is not None
        ]

        default_start = min(dep_starts)
        default_end = max(dep_ends) if dep_ends else now_utc.astimezone(tz)
        if any(dep.end_date is None for dep in device_deployments):
            default_end = max(default_end, now_utc.astimezone(tz))

        if start_date is None:
            start_local = default_start
        else:
            start_local = self._coerce_datetime(start_date, tz)
        if end_date is None:
            end_local = default_end
        else:
            end_local = self._coerce_datetime(end_date, tz)

        if end_local <= start_local:
            raise ValueError("end_date must be after start_date")

        start_local = self._align_to_bin_start(start_local, bin_size)

        bins: List[Dict[str, Any]] = []
        for bin_start, bin_end in self._iter_bins(start_local, end_local, bin_size):
            bins.append({
                'start': bin_start,
                'end': bin_end,
                'start_utc': bin_start.astimezone(UTC),
                'end_utc': bin_end.astimezone(UTC),
                'coverage': None,
                'status': 'not_deployed',
                'deployment_index': None,
            })

        if not bins:
            return {
                'device_code': device_code,
                'timezone': timezone_str,
                'bin_size': bin_size,
                'start': start_local,
                'end': end_local,
                'deployments': device_deployments,
                'bins': [],
                'deployment_summary': [],
            }

        range_start_utc = bins[0]['start_utc']
        range_end_utc = bins[-1]['end_utc']

        deployments_in_range: List[DeploymentInfo] = []
        dep_ranges: List[Tuple[datetime, datetime]] = []
        for dep in device_deployments:
            dep_start = ensure_timezone_aware(dep.begin_date, tz=timezone.utc)
            dep_end = ensure_timezone_aware(dep.end_date, tz=timezone.utc) if dep.end_date else range_end_utc
            if dep_end <= range_start_utc or dep_start >= range_end_utc:
                continue
            deployments_in_range.append(dep)
            dep_ranges.append((dep_start, dep_end))

        if not deployments_in_range:
            return {
                'device_code': device_code,
                'timezone': timezone_str,
                'bin_size': bin_size,
                'start': start_local,
                'end': end_local,
                'deployments': [],
                'bins': bins,
                'deployment_summary': [],
            }

        archive_intervals = self._fetch_archive_intervals(
            device_code,
            range_start_utc,
            range_end_utc,
            max_days_per_request=max_days_per_request,
            progress=progress,
            quiet=quiet,
            max_workers=max_workers,
            request_delay_seconds=request_delay_seconds,
        )
        merged_intervals = self._merge_intervals(archive_intervals)

        dep_idx = 0
        interval_idx = 0
        for b in bins:
            b_start = b['start_utc']
            b_end = b['end_utc']
            while interval_idx < len(merged_intervals) and merged_intervals[interval_idx][1] <= b_start:
                interval_idx += 1

            while dep_idx < len(dep_ranges) and dep_ranges[dep_idx][1] <= b_start:
                dep_idx += 1

            active_dep_idx = None
            if dep_idx < len(dep_ranges):
                dep_start, dep_end = dep_ranges[dep_idx]
                if dep_start < b_end and dep_end > b_start:
                    active_dep_idx = dep_idx

            if active_dep_idx is None:
                continue

            coverage_seconds = 0.0
            j = interval_idx
            while j < len(merged_intervals) and merged_intervals[j][0] < b_end:
                overlap_start = max(b_start, merged_intervals[j][0])
                overlap_end = min(b_end, merged_intervals[j][1])
                if overlap_end > overlap_start:
                    coverage_seconds += (overlap_end - overlap_start).total_seconds()
                if merged_intervals[j][1] <= b_end:
                    j += 1
                else:
                    break

            bin_seconds = (b_end - b_start).total_seconds()
            coverage_ratio = (coverage_seconds / bin_seconds) if bin_seconds > 0 else 0.0

            b['deployment_index'] = active_dep_idx
            b['coverage'] = max(0.0, min(1.0, coverage_ratio))
            b['status'] = 'data' if coverage_seconds > 0 else 'no_data'

        deployment_summary: List[Dict[str, Any]] = []
        for idx, dep in enumerate(deployments_in_range):
            dep_bins = [b for b in bins if b.get('deployment_index') == idx]
            total_bins = len(dep_bins)
            bins_with_data = sum(1 for b in dep_bins if (b.get('coverage') or 0) > 0)
            coverage_ratio = (bins_with_data / total_bins) if total_bins else 0.0
            deployment_summary.append({
                'deployment_index': idx,
                'device_code': dep.device_code,
                'location_name': dep.location_name,
                'location_code': dep.location_code,
                'begin_date': dep.begin_date,
                'end_date': dep.end_date,
                'bins_total': total_bins,
                'bins_with_data': bins_with_data,
                'coverage_ratio': coverage_ratio,
            })

        return {
            'device_code': device_code,
            'timezone': timezone_str,
            'bin_size': bin_size,
            'start': start_local,
            'end': end_local,
            'deployments': deployments_in_range,
            'bins': bins,
            'deployment_summary': deployment_summary,
        }

    def plot_device_availability(
        self,
        device_code: str,
        *,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        timezone_str: str = 'UTC',
        bin_size: str = 'day',
        max_days_per_request: int = 60,
        style: str = 'timeline',
        **plot_kwargs: Any,
    ):
        """
        Plot availability for a device using timeline or calendar styles.

        Args:
            device_code: ONC device code to evaluate.
            start_date: Optional start date for availability window.
            end_date: Optional end date for availability window.
            timezone_str: Timezone for binning and display.
            bin_size: "day" or "hour" binning for availability.
            max_days_per_request: Chunk size for archive queries.
            style: "timeline" or "calendar" (calendar requires bin_size='day').
            plot_kwargs: Passed to plotting helper.

        Returns:
            Matplotlib (fig, ax) tuple.
        """
        availability = self.get_device_availability(
            device_code,
            start_date=start_date,
            end_date=end_date,
            timezone_str=timezone_str,
            bin_size=bin_size,
            max_days_per_request=max_days_per_request,
        )
        if style == 'calendar':
            from ..utils.plotting import plot_availability_calendar
            return plot_availability_calendar(availability, **plot_kwargs)
        if style == 'timeline':
            from ..utils.plotting import plot_deployment_availability_timeline
            return plot_deployment_availability_timeline(availability, **plot_kwargs)
        raise ValueError("style must be 'timeline' or 'calendar'")
    
    def find_best_deployments_for_date_range(self, 
                                           start_date: Union[str, datetime], 
                                           end_date: Union[str, datetime],
                                           timezone_str: str = 'UTC',
                                           min_coverage_days: int = 1) -> List[DeploymentInfo]:
        """
        Find the best deployments that cover a specific date range.
        
        Args:
            start_date: Desired start date
            end_date: Desired end date
            timezone_str: Timezone for date interpretation
            min_coverage_days: Minimum days of coverage required
            
        Returns:
            List of best deployments sorted by coverage quality
        """
        # Parse and convert dates
        if isinstance(start_date, str):
            start_dt = dtparse.parse(start_date)
        else:
            start_dt = start_date
            
        if isinstance(end_date, str):
            end_dt = dtparse.parse(end_date)
        else:
            end_dt = end_date
        
        # Handle timezone
        if timezone_str != 'UTC':
            tz = gettz(timezone_str)
            if start_dt.tzinfo is None:
                start_dt = start_dt.replace(tzinfo=tz)
            if end_dt.tzinfo is None:
                end_dt = end_dt.replace(tzinfo=tz)
        
        # Convert to UTC
        start_utc = start_dt.astimezone(UTC) if start_dt.tzinfo else start_dt.replace(tzinfo=UTC)
        end_utc = end_dt.astimezone(UTC) if end_dt.tzinfo else end_dt.replace(tzinfo=UTC)
        
        # Find overlapping deployments
        overlapping = self.find_deployments_by_time_range(start_utc, end_utc, 'UTC')
        
        # Check data availability
        available = self.check_data_availability(overlapping, start_utc, end_utc)
        
        # Calculate coverage quality for each deployment
        scored_deployments = []
        for dep in available:
            # Calculate overlap period
            overlap_start = max(start_utc, dep.begin_date)
            overlap_end = min(end_utc, dep.end_date) if dep.end_date else end_utc
            
            if overlap_end > overlap_start:
                coverage_days = (overlap_end - overlap_start).days
                total_requested_days = (end_utc - start_utc).days
                
                if coverage_days >= min_coverage_days:
                    coverage_ratio = coverage_days / max(total_requested_days, 1)
                    scored_deployments.append((dep, coverage_ratio, coverage_days))
        
        # Sort by coverage ratio (best first)
        scored_deployments.sort(key=lambda x: x[1], reverse=True)
        
        return [dep for dep, _, _ in scored_deployments]
    
    def _build_location_cache(self):
        """Build cache of location information and hierarchy paths."""
        try:
            if self._location_cache_built:
                return
            locations = self.onc.getLocations({})
            for loc in locations:
                if isinstance(loc, dict):
                    code = loc.get('locationCode')
                    if code:
                        self._location_cache[code] = loc
            self._location_cache_built = True
        except Exception as e:
            logging.warning(f"Failed to build location cache: {e}")
        
        # Build a code -> path mapping so we can show parent locations for array elements (Hydrophone A/B/C)
        try:
            if self._location_paths_built:
                return
            tree = self.onc.getLocationHierarchy({})
            self._location_paths = {}
            
            def _walk(nodes: List[Dict], trail: List[str]):
                for node in nodes or []:
                    code = node.get("locationCode")
                    name = node.get("locationName", "")
                    path = trail + ([name] if name else [])
                    if code:
                        self._location_paths[code] = tuple(path)
                    children = node.get("children") or []
                    if children:
                        _walk(children, path)
            
            _walk(tree, [])
            self._location_paths_built = True
        except Exception as e:
            logging.warning(f"Failed to build location hierarchy: {e}")

    def _get_cached_deployments(self, max_age_minutes: int = 30) -> List[DeploymentInfo]:
        """Return cached deployments if fresh, otherwise fetch new ones."""
        now = datetime.now(timezone.utc)
        if (
            self._deployments_cache is not None
            and self._deployments_cache_at is not None
            and (now - self._deployments_cache_at).total_seconds() < max_age_minutes * 60
        ):
            return self._deployments_cache

        self._deployments_cache = self.get_all_hydrophone_deployments()
        self._deployments_cache_at = now
        return self._deployments_cache

    def _get_device_deployments(self, device_code: str, max_age_minutes: int = 30) -> List[DeploymentInfo]:
        """Fetch deployments for a single device (cached) for faster queries."""
        now = datetime.now(timezone.utc)
        cached = self._device_deployments_cache.get(device_code)
        cached_at = self._device_deployments_cache_at.get(device_code)
        if cached is not None and cached_at is not None:
            if (now - cached_at).total_seconds() < max_age_minutes * 60:
                return cached

        self._build_location_cache()
        try:
            deployments = self.onc.getDeployments({"deviceCode": device_code})
        except Exception as exc:
            if self.debug:
                logging.warning(f"Failed to fetch deployments for {device_code}: {exc}")
            deployments = []

        parsed: List[DeploymentInfo] = []
        if isinstance(deployments, list):
            for dep in deployments:
                if not isinstance(dep, dict):
                    continue
                # Ensure deviceCode is present for parsing
                dep.setdefault("deviceCode", device_code)
                deployment_info = self._parse_deployment(dep)
                if deployment_info:
                    parsed.append(deployment_info)

        self._device_deployments_cache[device_code] = parsed
        self._device_deployments_cache_at[device_code] = now
        return parsed
    
    def _get_deployments_parallel(self, hydrophones: List[Dict], max_workers: int = 10) -> List[Dict]:
        """Fetch deployments for multiple hydrophones in parallel."""
        def fetch_device_deployments(device: Dict) -> List[Dict]:
            device_code = device.get("deviceCode")
            if not device_code:
                return []
            
            try:
                device_deployments = self.onc.getDeployments({"deviceCode": device_code})
                if not isinstance(device_deployments, list):
                    logging.warning(f"Unexpected response type for {device_code} deployments")
                    return []
                
                # Add device info to each deployment
                for dep in device_deployments:
                    if isinstance(dep, dict):
                        dep.update(device)
                
                return device_deployments
            except HTTPError as http_err:
                if http_err.response is not None and http_err.response.status_code == 404:
                    if self.debug:
                        logging.debug(f"No deployments found (404) for device {device_code}")
                    return []
                raise
        
        all_deployments = []
        completed = 0
        total = len(hydrophones)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_device = {executor.submit(fetch_device_deployments, device): device 
                              for device in hydrophones}
            
            for future in concurrent.futures.as_completed(future_to_device):
                completed += 1
                print(f"\rFetching deployments: {completed}/{total}", end="", flush=True)
                
                try:
                    deployments = future.result()
                    if deployments:
                        all_deployments.extend(deployments)
                except Exception as e:
                    if self.debug:
                        logging.error(f"Error fetching deployments: {e}")
        
        print()  # New line after progress
        return all_deployments
    
    def _parse_deployment(self, deployment_dict: Dict) -> Optional[DeploymentInfo]:
        """Parse a deployment dictionary into a DeploymentInfo object."""
        try:
            device_code = deployment_dict.get('deviceCode')
            if not device_code:
                return None
            device_id = deployment_dict.get('deviceId') or deployment_dict.get('deviceID') or deployment_dict.get('device_id')
            if device_id is not None:
                device_id = str(device_id)
            
            # Parse dates
            begin_str = deployment_dict.get('begin')
            if not begin_str:
                return None
            begin_date = dtparse.parse(begin_str)
            
            end_str = deployment_dict.get('end')
            end_date = dtparse.parse(end_str) if end_str else None
            
            # Get location info
            location_code = deployment_dict.get('locationCode', '')
            location_info = self._location_cache.get(location_code, {})
            raw_location_name = location_info.get('locationName', '') or deployment_dict.get('locationName', '') or location_code
            path = self._location_paths.get(location_code, tuple())
            location_name, position_name = self._resolve_display_location(raw_location_name, path)
            
            # Get coordinates
            latitude = deployment_dict.get('lat') or location_info.get('lat', 0.0)
            longitude = deployment_dict.get('lon') or location_info.get('lon', 0.0)
            depth = deployment_dict.get('depth')
            
            citation = deployment_dict.get('citation')
            
            return DeploymentInfo(
                device_code=device_code,
                device_id=device_id,
                location_code=location_code,
                location_name=location_name,
                begin_date=begin_date,
                end_date=end_date,
                latitude=float(latitude) if latitude else 0.0,
                longitude=float(longitude) if longitude else 0.0,
                depth=float(depth) if depth else None,
                citation=citation,
                position_name=position_name,
                location_path=path or None
            )
        except Exception as e:
            if self.debug:
                logging.warning(f"Error parsing deployment: {e}")
            return None
    
    def _check_archive_availability_parallel(self, device_codes: List[str], 
                                           start_date: datetime, end_date: datetime,
                                           max_workers: int = 10) -> Dict[str, bool]:
        """Check archive file availability for multiple devices in parallel."""
        def check_device_files(device_code: str) -> Tuple[str, bool]:
            archive_filters = {
                'deviceCode': device_code,
                'dateFrom': start_date.strftime('%Y-%m-%dT%H:%M:%S.000Z'),
                'dateTo': end_date.strftime('%Y-%m-%dT%H:%M:%S.000Z'),
                'returnOptions': 'all'
            }
            try:
                # Avoid redirecting stdout/stderr in threads; redirect_stdout is not thread-safe
                # under concurrent use and can clobber notebook printing. We rely on the ONC
                # client verbosity flags set in __init__ to keep output minimal.
                list_result = self.onc.getArchivefile(filters=archive_filters, allPages=True)
                has_files = bool(list_result.get("files", []))
                return device_code, has_files
            except Exception as e:
                if self.debug:
                    logging.warning(f"Error checking files for device {device_code}: {e}")
                return device_code, False
        
        results = {}
        completed = 0
        total = len(device_codes)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_device = {executor.submit(check_device_files, device_code): device_code 
                              for device_code in device_codes}
            
            for future in concurrent.futures.as_completed(future_to_device):
                completed += 1
                print(f"\rChecking data availability: {completed}/{total}", end="", flush=True)
                
                try:
                    device_code, has_files = future.result()
                    results[device_code] = has_files
                except Exception as e:
                    if self.debug:
                        logging.error(f"Error checking device: {e}")
        
        print()  # New line after progress
        return results

    def _resolve_display_location(self, leaf_name: str, path: Tuple[str, ...]) -> Tuple[str, Optional[str]]:
        """
        Decide which human-friendly location name to show.
        For array elements where the leaf is \"Hydrophone A/B/C...\", prefer the parent site name.
        """
        position_name = None
        if path:
            leaf = path[-1]
            parent = path[-2] if len(path) > 1 else ''
            grandparent = path[-3] if len(path) > 2 else ''
        else:
            leaf = leaf_name
            parent = ''
            grandparent = ''
        
        # Identify hydrophone array leaf nodes
        if leaf and leaf.lower().startswith("hydrophone"):
            position_name = leaf
            # Prefer the site above the array container (grandparent) if available
            display_name = grandparent or parent or leaf
        else:
            display_name = leaf or parent or leaf_name
        
        return display_name, position_name
    
    def _check_product_availability_parallel(self, device_codes: List[str], 
                                           max_workers: int = 10) -> Dict[str, bool]:
        """Check data product availability for multiple devices in parallel."""
        def check_device_products(device_code: str) -> Tuple[str, bool]:
            try:
                prod_opts = self.onc.getDataProducts({"deviceCode": device_code})
                has_products = bool(prod_opts and isinstance(prod_opts, list) and prod_opts)
                return device_code, has_products
            except Exception as e:
                if self.debug:
                    logging.warning(f"Error checking products for device {device_code}: {e}")
                return device_code, False
        
        results = {}
        completed = 0
        total = len(device_codes)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_device = {executor.submit(check_device_products, device_code): device_code 
                              for device_code in device_codes}
            
            for future in concurrent.futures.as_completed(future_to_device):
                completed += 1
                print(f"\rChecking data products: {completed}/{total}", end="", flush=True)
                
                try:
                    device_code, has_products = future.result()
                    results[device_code] = has_products
                except Exception as e:
                    if self.debug:
                        logging.error(f"Error checking device: {e}")
        
        print()  # New line after progress
        return results

    def _coerce_datetime(self, value: Union[str, datetime], tz) -> datetime:
        if isinstance(value, str):
            dt = dtparse.parse(value)
        else:
            dt = ensure_timezone_aware(value, tz=tz)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=tz)
        return dt.astimezone(tz)

    def _align_to_bin_start(self, dt_obj: datetime, bin_size: str) -> datetime:
        if bin_size == 'day':
            return dt_obj.replace(hour=0, minute=0, second=0, microsecond=0)
        if bin_size == 'hour':
            return dt_obj.replace(minute=0, second=0, microsecond=0)
        raise ValueError("bin_size must be 'day' or 'hour'")

    def _iter_bins(self, start_local: datetime, end_local: datetime, bin_size: str):
        step = timedelta(days=1) if bin_size == 'day' else timedelta(hours=1)
        current = start_local
        while current < end_local:
            nxt = current + step
            yield current, min(nxt, end_local)
            current = nxt

    def _fetch_archive_intervals(
        self,
        device_code: str,
        start_utc: datetime,
        end_utc: datetime,
        *,
        max_days_per_request: int = 60,
        progress: Optional[Any] = None,
        quiet: bool = True,
        max_workers: int = 4,
        request_delay_seconds: float = 0.0,
    ) -> List[Tuple[datetime, datetime]]:
        if end_utc <= start_utc:
            return []
        intervals: List[Tuple[datetime, datetime]] = []
        if max_days_per_request and max_days_per_request > 0:
            step = timedelta(days=max_days_per_request)
        else:
            step = end_utc - start_utc
        cache_key = (
            device_code,
            format_iso_utc(start_utc),
            format_iso_utc(end_utc),
            max_days_per_request,
        )
        cached = self._archive_cache.get(cache_key)
        if cached is not None:
            return list(cached)

        chunks: List[Tuple[datetime, datetime]] = []
        current = start_utc
        while current < end_utc:
            chunk_end = min(end_utc, current + step)
            chunks.append((current, chunk_end))
            current = chunk_end

        total_chunks = len(chunks)
        if progress is not None:
            try:
                progress(total=total_chunks, advance=0)
            except Exception:
                pass

        def fetch_chunk(chunk: Tuple[datetime, datetime]):
            chunk_start, chunk_end = chunk
            archive_filters = {
                'deviceCode': device_code,
                'dateFrom': format_iso_utc(chunk_start),
                'dateTo': format_iso_utc(chunk_end),
                'returnOptions': 'all',
            }
            try:
                if request_delay_seconds and request_delay_seconds > 0:
                    import time as _time
                    _time.sleep(request_delay_seconds)
                if quiet:
                    import io
                    import contextlib
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                            list_result = self.onc.getArchivefile(filters=archive_filters, allPages=True)
                else:
                    list_result = self.onc.getArchivefile(filters=archive_filters, allPages=True)
            except Exception as exc:
                if self.debug:
                    logging.warning(f"Archive query failed for {device_code}: {exc}")
                list_result = None
            return self._extract_archive_file_intervals(list_result)

        if max_workers and max_workers > 1 and len(chunks) > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(fetch_chunk, chunk) for chunk in chunks]
                for future in concurrent.futures.as_completed(futures):
                    try:
                        intervals.extend(future.result())
                    except Exception as exc:
                        if self.debug:
                            logging.warning(f"Archive chunk failed for {device_code}: {exc}")
                    if progress is not None:
                        try:
                            progress(total=total_chunks, advance=1)
                        except Exception:
                            pass
        else:
            for chunk in chunks:
                intervals.extend(fetch_chunk(chunk))
                if progress is not None:
                    try:
                        progress(total=total_chunks, advance=1)
                    except Exception:
                        pass
        self._archive_cache[cache_key] = list(intervals)
        return intervals

    def _extract_archive_file_intervals(self, archive_response: Any) -> List[Tuple[datetime, datetime]]:
        files: List[Any] = []
        if isinstance(archive_response, dict):
            files = archive_response.get('files') or archive_response.get('data') or archive_response.get('results') or []
        elif isinstance(archive_response, list):
            files = archive_response
        if not files:
            return []

        intervals: List[Tuple[datetime, datetime]] = []
        for record in files:
            if not isinstance(record, dict):
                continue
            start = self._parse_timestamp(
                record.get('dateFrom') or record.get('begin') or record.get('start') or
                record.get('startTime') or record.get('fileStart') or record.get('timeFrom') or
                record.get('timeStart') or record.get('timestamp')
            )
            end = self._parse_timestamp(
                record.get('dateTo') or record.get('end') or record.get('stop') or
                record.get('endTime') or record.get('fileEnd') or record.get('timeTo') or
                record.get('timeEnd')
            )
            if start is not None and end is None:
                duration = self._parse_duration_seconds(record)
                if duration is not None:
                    end = start + timedelta(seconds=duration)
            if start is None or end is None:
                continue
            if end < start:
                start, end = end, start
            intervals.append((start, end))
        return intervals

    def _parse_timestamp(self, value: Any) -> Optional[datetime]:
        if value is None:
            return None
        if isinstance(value, datetime):
            dt = value
        elif isinstance(value, bytes):
            try:
                dt = dtparse.parse(value.decode())
            except Exception:
                return None
        elif isinstance(value, (int, float)):
            try:
                if value > 1e12:
                    dt = datetime.fromtimestamp(value / 1000.0, tz=timezone.utc)
                elif value > 1e10:
                    dt = datetime.fromtimestamp(value / 1000.0, tz=timezone.utc)
                else:
                    dt = datetime.fromtimestamp(value, tz=timezone.utc)
            except Exception:
                return None
        elif isinstance(value, str):
            try:
                dt = dtparse.parse(value)
            except Exception:
                return None
        else:
            return None

        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    def _parse_duration_seconds(self, record: Dict[str, Any]) -> Optional[float]:
        for key in (
            'duration',
            'durationSeconds',
            'fileDuration',
            'duration_sec',
            'durationSecs',
            'seconds',
        ):
            value = record.get(key)
            if value is None:
                continue
            try:
                return float(value)
            except Exception:
                continue
        return None

    def _merge_intervals(self, intervals: List[Tuple[datetime, datetime]]) -> List[Tuple[datetime, datetime]]:
        if not intervals:
            return []
        intervals_sorted = sorted(intervals, key=lambda x: x[0])
        merged: List[Tuple[datetime, datetime]] = []
        current_start, current_end = intervals_sorted[0]
        for start, end in intervals_sorted[1:]:
            if start > current_end:
                merged.append((current_start, current_end))
                current_start, current_end = start, end
            else:
                if end > current_end:
                    current_end = end
        merged.append((current_start, current_end))
        return merged


def interactive_deployment_selector(checker: HydrophoneDeploymentChecker,
                                  start_date: Optional[Union[str, datetime]] = None,
                                  end_date: Optional[Union[str, datetime]] = None,
                                  timezone_str: str = 'UTC') -> List[DeploymentInfo]:
    """
    Interactive function to help users select deployments with data availability.
    
    Args:
        checker: HydrophoneDeploymentChecker instance
        start_date: Optional start date
        end_date: Optional end date
        timezone_str: Timezone for date interpretation
        
    Returns:
        List of selected deployments
    """
    print("🌊 ONC Hydrophone Deployment Selector")
    print("=" * 50)
    
    # Get date range if not provided
    if not start_date or not end_date:
        print("\nFirst, let's see what deployments are available...")
        all_deployments = checker.get_all_hydrophone_deployments()
        
        if not all_deployments:
            print("❌ No deployments found!")
            return []
        
        # Show summary of all deployments
        checker.print_deployment_summary(all_deployments, show_data_availability=False)
        
        # Get date range from user
        if not start_date:
            start_date = input("\n📅 Enter start date (YYYY-MM-DD or YYYY-MM-DD HH:MM): ").strip()
        if not end_date:
            end_date = input("📅 Enter end date (YYYY-MM-DD or YYYY-MM-DD HH:MM): ").strip()
    
    # Find best deployments for the date range
    print(f"\n🔍 Finding deployments for: {start_date} to {end_date} ({timezone_str})")
    best_deployments = checker.find_best_deployments_for_date_range(
        start_date, end_date, timezone_str)
    
    if not best_deployments:
        print("❌ No deployments with data found for the specified date range!")
        return []
    
    # Show available deployments with data
    print("\n✅ Found deployments with available data:")
    checker.print_deployment_summary(best_deployments, show_data_availability=True)
    
    # Let user select deployments
    print("\nSelect deployments to download (enter numbers separated by commas, or 'all'):")
    for i, dep in enumerate(best_deployments):
        end_str = dep.end_date.strftime('%Y-%m-%d') if dep.end_date else "ongoing"
        print(f"  [{i}] {dep.device_code} at {dep.location_name} ({dep.begin_date.strftime('%Y-%m-%d')} to {end_str})")
    
    selection = input("\nYour selection: ").strip().lower()
    
    if selection == 'all':
        return best_deployments
    
    try:
        indices = [int(x.strip()) for x in selection.split(',')]
        selected = [best_deployments[i] for i in indices if 0 <= i < len(best_deployments)]
        
        if selected:
            print(f"\n✅ Selected {len(selected)} deployment(s):")
            for dep in selected:
                print(f"  • {dep.device_code} at {dep.location_name}")
        else:
            print("❌ No valid selections made!")
        
        return selected
    except (ValueError, IndexError) as e:
        print(f"❌ Invalid selection: {e}")
        return []


# Example usage functions
def example_basic_usage():
    """Example of basic deployment checking."""
    # Initialize checker (you need to provide your ONC token)
    onc_token = "YOUR_ONC_TOKEN_HERE"
    checker = HydrophoneDeploymentChecker(onc_token, debug=True)
    
    # Get all deployments
    all_deployments = checker.get_all_hydrophone_deployments()
    checker.print_deployment_summary(all_deployments)
    
    # Find deployments for a specific time range
    deployments = checker.find_deployments_by_time_range(
        start_date="2020-01-01",
        end_date="2020-12-31",
        timezone_str="America/Vancouver"
    )
    
    # Check data availability
    available_deployments = checker.check_data_availability(
        deployments, 
        dtparse.parse("2020-01-01"), 
        dtparse.parse("2020-12-31")
    )
    
    print("\nDeployments with available data:")
    checker.print_deployment_summary(available_deployments)


def example_interactive_usage():
    """Example of interactive deployment selection."""
    # Initialize checker
    onc_token = "YOUR_ONC_TOKEN_HERE"
    checker = HydrophoneDeploymentChecker(onc_token)
    
    # Interactive selection
    selected_deployments = interactive_deployment_selector(checker)
    
    if selected_deployments:
        print(f"You selected {len(selected_deployments)} deployment(s)")
        # Now you can use these deployments for downloading data
    else:
        print("No deployments selected")


if __name__ == "__main__":
    # Run example
    print("Example usage:")
    print("1. Basic usage - example_basic_usage()")
    print("2. Interactive usage - example_interactive_usage()")
    print("\nMake sure to set your ONC_TOKEN in the examples!") 

# Alias for compatibility
DeploymentChecker = HydrophoneDeploymentChecker
