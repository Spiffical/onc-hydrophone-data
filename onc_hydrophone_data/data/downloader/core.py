import logging
import os
from threading import Lock

from ..onc_requests import ONCRequestManager
from ..deployment_checker import DeploymentChecker

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PrintLogger:
    def info(self, msg, *args, **kwargs):
        print(msg % args if args else msg)

    def warning(self, msg, *args, **kwargs):
        print('WARNING:', msg % args if args else msg)

    def error(self, msg, *args, **kwargs):
        print('ERROR:', msg % args if args else msg)

    def debug(self, msg, *args, **kwargs):
        print('DEBUG:', msg % args if args else msg)


class HydrophoneDownloader:
    """User-facing downloader for ONC hydrophone data.

    This class bundles the most common workflows:
    - Range downloads for spectrograms and audio
    - Sampling and event-based downloads
    - JSON/CSV request files
    - Custom spectrogram generation
    """
    def __init__(
        self,
        ONC_token,
        parent_dir,
        use_logging: bool = True,
        *,
        spectral_downsample: int = 2,
        **kwargs,
    ):
        """Initialize the downloader.

        Args:
            ONC_token: ONC API token.
            parent_dir: Root directory for downloads.
            use_logging: If True, use module logger; otherwise print to stdout.
            spectral_downsample: Default HSD downsample setting
                (0=full, 1=1‑min, 2=plotRes).
            **kwargs: Reserved for future options.
        """
        from .. import hydrophone_downloader as hd

        self.onc = hd.ONC(ONC_token)
        self._onc_token = ONC_token  # Stored for thread-safe client creation
        self.parent_dir = parent_dir
        self.logger = logger if use_logging else PrintLogger()
        self.max_workers = 4

        # Initialize Request Manager with default HSD options to avoid API warnings
        self.request_manager = ONCRequestManager(
            ONC_token,
            parent_dir,
            self.logger,
            spectral_downsample=spectral_downsample,
            spectrogram_concatenation="None",
        )

        # Default spectral downsample for data product requests:
        # 0 = full Res
        # 1 = one-minute average
        # 2 = plot resolution (default)
        # 3 = 1 hour average
        # 4 = 1 day average
        self.spectral_downsample = spectral_downsample

        # Clean flat structure paths (defaults)
        # ONC spectrograms go to onc_spectrograms/ (no subdirs)
        self.spectrogram_path = os.path.join(self.parent_dir, 'onc_spectrograms', '')
        # Keep input_path as alias for backwards compatibility
        self.input_path = self.spectrogram_path
        # Audio files go to audio/ (renamed from flac/)
        self.audio_path = os.path.join(self.parent_dir, 'audio', '')
        # Keep flac_path as alias for backwards compatibility
        self.flac_path = self.audio_path

        # Lock for thread-safe path modification
        self._path_lock = Lock()

        # Build deployment checker
        self.deployment_checker = DeploymentChecker(self._onc_token)
        self._deployment_cache = None
        self._cache_timestamp = None

    def set_spectral_downsample(self, value: int) -> None:
        """Override default downsample option (0=fullRes, 1=one-minute, 2=plotRes, etc.)."""
        self.spectral_downsample = value


from .audio_workflows import (
    _build_event_audio_windows,
    describe_audio_window,
    describe_event_audio_windows,
    download_audio_for_center_time,
    download_audio_for_events,
    download_audio_for_range,
    download_sampled_audio,
    plan_audio_window,
)
from .clip_io import (
    _clip_basename,
    _collect_files_for_range,
    _extract_frequency_axis,
    _load_audio_chunk,
    _load_audio_file,
    _load_spectrogram_chunk,
    _trim_audio_clip,
    _write_audio_clip,
    _write_spectrogram_clip,
)
from .custom_spectrograms import _resolve_custom_clip_pad_seconds, create_custom_spectrograms_from_json
from .deployment_workflows import (
    _get_cached_deployments,
    _show_deployments_with_data,
    _validate_deployment_coverage_with_data,
    download_specific_spectrograms,
    download_spectrograms_with_deployment_check,
    download_with_deployment_check,
    interactive_deployment_selection,
    interactive_download_with_deployments,
    quick_deployment_check,
    show_available_deployments,
    validate_deployment_coverage,
)
from .directories import _create_method_folder_name, setup_directories
from .onc_downloads import download_MAT_or_PNG, download_audio_files, download_flac_files
from .parallel_requests import filter_existing_requests, run_parallel_windows, try_download_run
from .request_execution import _execute_timestamp_request, download_requests_from_csv, download_requests_from_json
from .request_parsing import (
    _build_request_from_dict,
    _build_request_windows,
    _coerce_timestamp_requests,
    _load_request_payload_from_csv,
    _normalize_csv_request_row,
    _parse_csv_bool,
    _parse_csv_data_product_options,
    _parse_csv_number,
)
from .sampling_workflows import download_spectrograms_with_sampling_schedule, sampling_schedule
from .spectrogram_workflows import (
    download_spectrogram_windows,
    download_spectrograms_for_events,
    download_spectrograms_for_range,
    download_sampled_spectrograms,
)
from .time_utils import (
    _ceil_to_window,
    _floor_to_window,
    _format_iso_utc,
    _parse_timestamp_value,
    _resolve_timezone,
    _timestamp_from_filename,
)
from .validation import check_for_anomalies, process_spectrograms, validate_spectrograms


HydrophoneDownloader._resolve_timezone = staticmethod(_resolve_timezone)
HydrophoneDownloader._parse_timestamp_value = staticmethod(_parse_timestamp_value)
HydrophoneDownloader._floor_to_window = staticmethod(_floor_to_window)
HydrophoneDownloader._format_iso_utc = staticmethod(_format_iso_utc)
HydrophoneDownloader._ceil_to_window = staticmethod(_ceil_to_window)
HydrophoneDownloader._timestamp_from_filename = staticmethod(_timestamp_from_filename)
HydrophoneDownloader._extract_frequency_axis = staticmethod(_extract_frequency_axis)
HydrophoneDownloader._parse_csv_bool = staticmethod(_parse_csv_bool)
HydrophoneDownloader._parse_csv_number = staticmethod(_parse_csv_number)
HydrophoneDownloader._parse_csv_data_product_options = staticmethod(_parse_csv_data_product_options)
HydrophoneDownloader._resolve_custom_clip_pad_seconds = staticmethod(_resolve_custom_clip_pad_seconds)

HydrophoneDownloader._collect_files_for_range = _collect_files_for_range
HydrophoneDownloader._load_spectrogram_chunk = _load_spectrogram_chunk
HydrophoneDownloader._clip_basename = _clip_basename
HydrophoneDownloader._write_spectrogram_clip = _write_spectrogram_clip
HydrophoneDownloader._load_audio_chunk = _load_audio_chunk
HydrophoneDownloader._load_audio_file = _load_audio_file
HydrophoneDownloader._write_audio_clip = _write_audio_clip
HydrophoneDownloader._trim_audio_clip = _trim_audio_clip
HydrophoneDownloader._build_request_windows = _build_request_windows
HydrophoneDownloader._build_request_from_dict = _build_request_from_dict
HydrophoneDownloader._coerce_timestamp_requests = _coerce_timestamp_requests
HydrophoneDownloader._normalize_csv_request_row = _normalize_csv_request_row
HydrophoneDownloader._load_request_payload_from_csv = _load_request_payload_from_csv
HydrophoneDownloader._execute_timestamp_request = _execute_timestamp_request
HydrophoneDownloader.download_requests_from_json = download_requests_from_json
HydrophoneDownloader.download_requests_from_csv = download_requests_from_csv
HydrophoneDownloader.create_custom_spectrograms_from_json = create_custom_spectrograms_from_json
HydrophoneDownloader.setup_directories = setup_directories
HydrophoneDownloader._create_method_folder_name = _create_method_folder_name
HydrophoneDownloader.try_download_run = try_download_run
HydrophoneDownloader.run_parallel_windows = run_parallel_windows
HydrophoneDownloader.filter_existing_requests = filter_existing_requests
HydrophoneDownloader.download_spectrogram_windows = download_spectrogram_windows
HydrophoneDownloader.download_spectrograms_for_range = download_spectrograms_for_range
HydrophoneDownloader.download_sampled_spectrograms = download_sampled_spectrograms
HydrophoneDownloader.download_spectrograms_for_events = download_spectrograms_for_events
HydrophoneDownloader.download_audio_for_range = download_audio_for_range
HydrophoneDownloader.download_sampled_audio = download_sampled_audio
HydrophoneDownloader._build_event_audio_windows = _build_event_audio_windows
HydrophoneDownloader.describe_event_audio_windows = describe_event_audio_windows
HydrophoneDownloader.download_audio_for_events = download_audio_for_events
HydrophoneDownloader.plan_audio_window = plan_audio_window
HydrophoneDownloader.describe_audio_window = describe_audio_window
HydrophoneDownloader.download_audio_for_center_time = download_audio_for_center_time
HydrophoneDownloader.sampling_schedule = sampling_schedule
HydrophoneDownloader.download_MAT_or_PNG = download_MAT_or_PNG
HydrophoneDownloader.download_audio_files = download_audio_files
HydrophoneDownloader.download_flac_files = download_flac_files
HydrophoneDownloader.check_for_anomalies = check_for_anomalies
HydrophoneDownloader.validate_spectrograms = validate_spectrograms
HydrophoneDownloader.process_spectrograms = process_spectrograms
HydrophoneDownloader.download_spectrograms_with_sampling_schedule = download_spectrograms_with_sampling_schedule
HydrophoneDownloader.download_spectrograms_with_deployment_check = download_spectrograms_with_deployment_check
HydrophoneDownloader.download_with_deployment_check = download_with_deployment_check
HydrophoneDownloader.show_available_deployments = show_available_deployments
HydrophoneDownloader.interactive_deployment_selection = interactive_deployment_selection
HydrophoneDownloader.validate_deployment_coverage = validate_deployment_coverage
HydrophoneDownloader.download_specific_spectrograms = download_specific_spectrograms
HydrophoneDownloader.quick_deployment_check = quick_deployment_check
HydrophoneDownloader.interactive_download_with_deployments = interactive_download_with_deployments
HydrophoneDownloader._validate_deployment_coverage_with_data = _validate_deployment_coverage_with_data
HydrophoneDownloader._show_deployments_with_data = _show_deployments_with_data
HydrophoneDownloader._get_cached_deployments = _get_cached_deployments
