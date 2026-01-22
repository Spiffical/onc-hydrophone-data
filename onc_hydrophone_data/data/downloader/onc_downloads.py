import concurrent.futures
import datetime as dt
import glob
import os
import time
from typing import Any, Dict, Optional, Sequence

from ...onc.common import start_and_end_strings
from .flags import _resolve_download_audio


def download_MAT_or_PNG(
    self,
    deviceCode,
    start_date_object,
    filetype='png',
    spectrograms_per_batch=6,
    download_audio: Optional[bool] = None,
    download_flac: Optional[bool] = None,
    spectral_downsample: Optional[int] = None,
    data_product_options: Optional[Dict[str, Any]] = None,
    audio_download_workers: Optional[int] = None,
):
    """Download MAT or PNG files for a given time period.

    Args:
        deviceCode: ONC device code.
        start_date_object: Start datetime for the request.
        filetype: ``png`` or ``mat``.
        spectrograms_per_batch: Number of 5‑minute spectrograms per request.
        download_audio: Whether to download matching audio files.
        download_flac: Legacy alias for ``download_audio``.
        spectral_downsample: Optional override for HSD downsample option.
        data_product_options: Optional ONC ``dpo_*`` overrides for HSD requests.
        audio_download_workers: Optional override for parallel audio downloads.
    """
    download_audio = _resolve_download_audio(download_audio, download_flac)
    # Calculate duration based on number of spectrograms (each is 5 minutes = 300 seconds)
    # Use exact duration to get precisely the requested number of spectrograms
    data_length_seconds = (spectrograms_per_batch - 1) * 300
    
    time_delta = dt.timedelta(0, data_length_seconds)
    start_time, end_time = start_and_end_strings(start_date_object, time_delta)

    downsample = spectral_downsample if spectral_downsample is not None else self.spectral_downsample
    data_product_options = dict(data_product_options or {})

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
        }
        if 'dpo_spectralDataDownsample' not in data_product_options:
            filters['dpo_spectralDataDownsample'] = downsample
        if 'dpo_spectrogramConcatenation' not in data_product_options:
            filters['dpo_spectrogramConcatenation'] = 'None'
        filters.update(data_product_options)
        
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
            
            # Download audio files if requested (FLAC, fallback to WAV)
            if download_audio:
                flac_start = time.time()
                self.download_flac_files(
                    deviceCode,
                    start_time,
                    end_time,
                    max_download_workers=audio_download_workers,
                )
                self.logger.info(f"Audio files downloaded in {time.time() - flac_start:.2f}s")
            
            # Process downloaded files
            process_start = time.time()
            self.process_spectrograms(filetype)
            self.logger.info(f"Files processed in {time.time() - process_start:.2f}s")
            
            # Log progress
            num_files = len(glob.glob(os.path.join(self.spectrogram_path, f'*.{filetype}')))
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
        
        # Download audio files if requested (FLAC, fallback to WAV)
        if download_audio:
            flac_start = time.time()
            self.download_flac_files(
                deviceCode,
                start_time,
                end_time,
                max_download_workers=audio_download_workers,
            )
            self.logger.info(f"Audio files downloaded in {time.time() - flac_start:.2f}s")
        
        # Process downloaded files
        self.process_spectrograms(filetype)


def download_audio_files(
    self,
    deviceCode,
    start_time,
    end_time,
    *,
    extensions: Sequence[str] = ('flac', 'wav'),
    max_download_workers: Optional[int] = None,
    onc_client=None,
):
    """Download audio files for the requested time window.

    Tries extensions in order and stops after any successful download.

    Args:
        deviceCode: ONC device code.
        start_time: ISO start timestamp (UTC).
        end_time: ISO end timestamp (UTC).
        extensions: Audio extensions to try in order.
        max_download_workers: Override parallel download workers.
        onc_client: Optional ONC client instance (used for thread safety).

    Returns:
        Summary dict with files found/downloaded and extension used.
    """
    self.logger.info(f'Finding audio files for {deviceCode} from {start_time} to {end_time}')
    try:
        start_dt = self._parse_timestamp_value(start_time)
        end_dt = self._parse_timestamp_value(end_time)
        windows = self._build_request_windows(start_dt, end_dt)
        if len(windows) > 1:
            self.logger.info(
                f"Time range spans {len(windows)} adjacent 5-minute audio files; "
                "downloading adjacent file(s)."
            )
    except Exception:
        pass

    client = onc_client or self.onc
    original_output_path = client.outPath
    max_workers = max_download_workers or self.max_workers or 1

    summary = {
        'extension_used': None,
        'files_found': 0,
        'files_downloaded': 0,
        'errors': 0,
    }

    try:
        for extension in extensions:
            search_start = time.time()
            filters = {
                'deviceCode': deviceCode,
                'dateFrom': start_time,
                'dateTo': end_time,
                'extension': extension,
            }
            try:
                result = client.getListByDevice(filters, allPages=True)
            except Exception as e:
                self.logger.error(f"Error searching for {extension.upper()} files: {e}")
                continue

            self.logger.info(
                f"{extension.upper()} file search completed in {time.time() - search_start:.2f}s"
            )
            files = result.get('files', []) if isinstance(result, dict) else []
            audio_files = [f for f in files if f.lower().endswith(f".{extension}")]
            if not audio_files:
                self.logger.info(f'No {extension.upper()} files found in the specified time range')
                continue

            summary['files_found'] = len(audio_files)
            client.outPath = self.audio_path
            self.logger.info(f'Found {len(audio_files)} {extension.upper()} file(s)')

            download_start = time.time()
            pending = set(audio_files)
            attempts = {f: 0 for f in audio_files}
            max_attempts = 6
            attempt_round = 0
            downloaded = 0
            errors = 0

            while pending:
                attempt_round += 1
                pending_list = list(pending)
                for f in pending_list:
                    attempts[f] += 1

                self.logger.info(
                    f"Downloading {extension.upper()} batch (attempt {attempt_round}/{max_attempts})..."
                )

                pending_next = set()
                if max_workers > 1 and len(pending_list) > 1:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                        future_map = {
                            executor.submit(client.getFile, f, overwrite=True): f for f in pending_list
                        }
                        for future in concurrent.futures.as_completed(future_map):
                            audio_file = future_map[future]
                            try:
                                future.result()
                                downloaded += 1
                            except Exception as e:
                                if attempts[audio_file] >= max_attempts:
                                    errors += 1
                                    self.logger.warning(
                                        f"Failed to download {audio_file} after {attempts[audio_file]} attempts: {e}"
                                    )
                                else:
                                    self.logger.debug(
                                        f"{extension.upper()} not ready yet ({audio_file}): {e}"
                                    )
                                    pending_next.add(audio_file)
                else:
                    for audio_file in pending_list:
                        try:
                            client.getFile(audio_file, overwrite=True)
                            downloaded += 1
                        except Exception as e:
                            if attempts[audio_file] >= max_attempts:
                                errors += 1
                                self.logger.warning(
                                    f"Failed to download {audio_file} after {attempts[audio_file]} attempts: {e}"
                                )
                            else:
                                self.logger.debug(
                                    f"{extension.upper()} not ready yet ({audio_file}): {e}"
                                )
                                pending_next.add(audio_file)

                pending = pending_next
                if pending:
                    time.sleep(5)

            self.logger.info(
                f"{extension.upper()} files downloaded in {time.time() - download_start:.2f}s"
            )
            summary.update({
                'extension_used': extension if downloaded else None,
                'files_downloaded': downloaded,
                'errors': errors,
            })
            if downloaded:
                return summary

            self.logger.warning(
                f"No {extension.upper()} files downloaded; trying next format if available"
            )

        self.logger.info('No audio files downloaded in the specified time range')
        return summary
    except Exception as e:
        self.logger.error(f'Error searching for audio files: {e}')
        return summary
    finally:
        client.outPath = original_output_path


def download_flac_files(
    self,
    deviceCode,
    start_time,
    end_time,
    onc_client=None,
    max_download_workers: Optional[int] = None,
):
    """Backwards-compatible wrapper for audio downloads (FLAC with WAV fallback).

    Args:
        deviceCode: ONC device code.
        start_time: ISO start timestamp (UTC).
        end_time: ISO end timestamp (UTC).
        onc_client: Optional ONC client instance.
        max_download_workers: Override parallel download workers.
    """
    return self.download_audio_files(
        deviceCode,
        start_time,
        end_time,
        extensions=('flac', 'wav'),
        max_download_workers=max_download_workers,
        onc_client=onc_client,
    )
