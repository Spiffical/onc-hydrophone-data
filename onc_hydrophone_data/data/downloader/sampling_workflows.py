import datetime as dt
import random
import time
from datetime import date, datetime, timedelta
from typing import Optional

import numpy as np

from ...onc.common import start_and_end_strings
from .flags import _resolve_download_audio


def sampling_schedule(self, deviceCode, threshold_num, year, month, day, day_interval=None, num_days=None, spectrograms_per_batch=6):
    """Generate a sampling schedule of datetimes across a date range.

    Args:
        deviceCode: ONC hydrophone device code.
        threshold_num: Target number of spectrograms to sample.
        year: Start year.
        month: Start month.
        day: Start day.
        day_interval: Optional day step (auto‑computed if None).
        num_days: Optional number of days to sample (defaults to today‑start).
        spectrograms_per_batch: Number of spectrograms per request.

    Returns:
        Tuple of ``(date_list, sample_time_per_day)`` where ``date_list`` is a list
        of datetime start times for requests.
    """
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


def download_spectrograms_with_sampling_schedule(
    self,
    deviceCode,
    start_date,
    threshold_num,
    num_days=None,
    filetype='png',
    spectrograms_per_batch=6,
    download_audio: Optional[bool] = None,
    download_flac: Optional[bool] = None,
):
    """Download spectrograms based on a sampling schedule.

    Args:
        deviceCode: ONC device code.
        start_date: Start date for sampling (tuple: year, month, day).
        threshold_num: Number of samples to take.
        num_days: Number of days to sample (optional).
        filetype: Type of file to download ('png' or 'mat').
        spectrograms_per_batch: Number of 5-minute spectrograms per batch.
        download_audio: Whether to download corresponding audio files.
        download_flac: Legacy alias for ``download_audio``.
    """
    schedule_start = time.time()
    self.logger.info(f"Starting sampling schedule download for {deviceCode} from {start_date}")
    self.logger.info(f"Batch size: {spectrograms_per_batch} spectrograms per request")
    download_audio = _resolve_download_audio(download_audio, download_flac)
    
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
                    self.request_manager.submit_mat_run_no_wait,
                    deviceCode,
                    ts,
                    ts + timedelta(seconds=data_length_seconds),
                    self.spectrogram_path,  # out_path
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
                futures = [
                    executor.submit(self.try_download_run, rec, True, download_audio=download_audio)
                    for rec in pending
                ]
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
            self.download_MAT_or_PNG(
                deviceCode,
                request_time,
                filetype,
                spectrograms_per_batch,
                download_audio=download_audio,
            )
            
            self.logger.info(f"Completed request {i}/{total_requests} in {time.time() - request_start:.2f}s")
            self.logger.info(f"Overall progress: {i}/{total_requests} requests completed")
        
        total_time = time.time() - schedule_start
        self.logger.info(f"Completed all downloads in {total_time:.2f}s")
        self.logger.info(f"Average time per request: {total_time/total_requests:.2f}s")
