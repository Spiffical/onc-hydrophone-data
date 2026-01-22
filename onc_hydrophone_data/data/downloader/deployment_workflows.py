import time
from datetime import datetime, timedelta, timezone
from typing import Optional

from ...onc.common import ensure_timezone_aware
from .flags import _resolve_download_audio


def download_spectrograms_with_deployment_check(
    self,
    deviceCode,
    start_date,
    threshold_num,
    num_days=None,
    filetype='png',
    auto_select_deployment=False,
    spectrograms_per_batch=6,
    download_audio: Optional[bool] = None,
    download_flac: Optional[bool] = None,
):
    """
    Download spectrograms with deployment checking enabled.
    
    :param deviceCode: ONC device code
    :param start_date: Start date for sampling
    :param threshold_num: Number of samples to take
    :param num_days: Number of days to sample (optional)
    :param filetype: Type of file to download ('png' or 'mat')
    :param auto_select_deployment: Whether to automatically select the best deployment
    :param spectrograms_per_batch: Number of 5-minute spectrograms to download per batch
    :param download_audio: Whether to download corresponding audio files (FLAC/WAV fallback)
    """
    self.logger.info(f"Starting deployment-aware download for {deviceCode}")
    self.logger.info(f"Batch size: {spectrograms_per_batch} spectrograms per request")
    download_audio = _resolve_download_audio(download_audio, download_flac)
    
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
            deviceCode,
            time_slot,
            filetype,
            spectrograms_per_batch,
            auto_select_deployment,
            download_audio=download_audio,
        )
        
        if success:
            self.logger.info(f"Successfully downloaded slot {i}/{total_slots}")
        else:
            self.logger.warning(f"Failed to download slot {i}/{total_slots}")
        
        self.logger.info(f"Completed slot {i}/{total_slots} in {time.time() - slot_start:.2f}s")
        self.logger.info(f"Overall progress: {i}/{total_slots} slots completed")
    
    self.logger.info("Deployment-aware download completed")


def download_with_deployment_check(
    self,
    deviceCode,
    start_date_object,
    filetype='png',
    data_length_seconds=1799,
    auto_select_deployment=False,
    download_audio: Optional[bool] = None,
    download_flac: Optional[bool] = None,
):
    """
    Download spectrograms with deployment validation.
    
    :param deviceCode: Device code
    :param start_date_object: Start date (datetime object)
    :param filetype: File type ('png' or 'mat')
    :param data_length_seconds: Length of data to download in seconds
    :param auto_select_deployment: If True, automatically select best deployment
    :param download_audio: Whether to also download corresponding audio files (FLAC/WAV fallback)
    :return: Success status and deployment info
    """
    download_audio = _resolve_download_audio(download_audio, download_flac)
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
        self.download_MAT_or_PNG(
            deviceCode,
            start_date_object,
            filetype=filetype,
            spectrograms_per_batch=6,
            download_audio=download_audio,
        )
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
    from ..deployment_checker import interactive_deployment_selector
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


def download_specific_spectrograms(
    self,
    device_times_dict,
    filetype='png',
    duration_seconds=300,
    download_audio: Optional[bool] = None,
    download_flac: Optional[bool] = None,
):
    """
    Downloads spectrograms for specific device IDs and timestamps.
    
    :param device_times_dict: Dictionary where keys are device IDs, and values are lists of tuples (year, month, day, hour, minute, second).
    :param filetype: File type to download ('png' or 'mat').
    :param duration_seconds: Duration of each spectrogram in seconds (default: 300 for 5 minutes).
    :param download_audio: Whether to also download corresponding audio files (FLAC/WAV fallback).
    """
    download_audio = _resolve_download_audio(download_audio, download_flac)
    
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
            self.download_MAT_or_PNG(
                device_id,
                start_date_object,
                filetype=filetype,
                spectrograms_per_batch=6,
                download_audio=download_audio,
            )

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
