import concurrent.futures
import glob
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests
from onc.onc import ONC

from ...onc.common import format_iso_utc
from .flags import _resolve_download_audio


def try_download_run(
    self,
    rec: Dict[str, Any],
    allow_rerun: bool = True,
    download_audio: Optional[bool] = None,
    download_flac: Optional[bool] = None,
    audio_download_workers: Optional[int] = None,
    max_attempts: int = 6,
    wait_for_complete: bool = False,
    poll_interval_seconds: int = 15,
    max_wait_seconds: int = 900,
) -> (str, Dict[str, Any]):
    """Attempt to download a previously submitted run. Returns (status, updated_rec).
    status: 'pending' | 'downloaded' | 'error'"""
    download_audio = _resolve_download_audio(download_audio, download_flac)
    # Prepare a per-call ONC client to avoid cross-thread outPath races
    local_out_path = rec.get('outPath')
    if local_out_path:
        onc_client = ONC(self._onc_token, showInfo=False)
        onc_client.outPath = local_out_path
        # Ensure directory exists prior to download (flat structure, no subdirs)
        os.makedirs(local_out_path, exist_ok=True)
    else:
        onc_client = self.onc

    # Ensure we have a runId
    run_id = None
    if isinstance(rec.get('runIds'), list) and rec['runIds']:
        run_id = rec['runIds'][0]
    if run_id is None and allow_rerun and rec.get('dpRequestId'):
        try:
            run_data = onc_client.runDataProduct(rec['dpRequestId'], waitComplete=False)
            if isinstance(run_data, dict) and run_data.get('runIds'):
                rec['runIds'] = run_data['runIds']
                run_id = rec['runIds'][0]
        except Exception as e:
            self.logger.debug(f"runDataProduct (no-wait) not ready for dpRequestId={rec.get('dpRequestId')}: {e}")

    if run_id is None:
        # Not ready yet
        rec['status'] = 'pending'
        return 'pending', rec

    should_wait = wait_for_complete and rec.get('dpRequestId') and not rec.get('readyAt')
    if should_wait:
        # Use request_manager for polling
        ready, reason, payload = self.request_manager.wait_for_data_product_ready(
            rec['dpRequestId'],
            max_wait_seconds=max_wait_seconds,
            poll_interval_seconds=poll_interval_seconds,
        )
        rec['latestStatus'] = payload
        if not ready:
            if reason in ('cancelled', 'error', 'failed'):
                rec['status'] = 'error'
                rec['error'] = f'Data product status={reason}'
                rec['lastDownloadError'] = f'data product status={reason}'
                return 'error', rec
            rec['status'] = 'pending'
            rec['pendingReason'] = reason
            rec['lastDownloadError'] = reason
            return 'pending', rec
        rec['readyAt'] = rec.get('readyAt') or format_iso_utc(datetime.now(timezone.utc))

    # Try download
    attempt_limit = max_attempts if (max_attempts or 0) > 0 else None
    if wait_for_complete:
        attempt_limit = None

    try:
        rec['attempts'] = rec.get('attempts', 0) + 1
        file_infos = onc_client.downloadDataProduct(
            run_id,
            maxRetries=10,
            downloadResultsOnly=False,
            includeMetadataFile=False,
            overwrite=True,
        )

        rec['lastDownloadError'] = None
        # Determine if any MAT files actually arrived
        mat_downloaded = False
        target_prefix = f"{rec['deviceCode']}_"
        target_suffix = rec.get('start')
        if target_suffix:
            target_suffix = target_suffix.replace(':', '').replace('-', '').replace('.', '')
        for info in file_infos or []:
            fname = info.get('file') or ''
            if fname.lower().endswith('.mat') and os.path.basename(fname).startswith(target_prefix):
                mat_downloaded = True
                break
        if not mat_downloaded:
            # Also check filesystem in case getInfo lacks names
            pattern = f"{rec['deviceCode']}_*.mat"
            mat_glob = glob.glob(os.path.join(local_out_path or self.input_path, pattern))
            mat_downloaded = bool(mat_glob)

        if not mat_downloaded:
            if attempt_limit is not None and rec['attempts'] >= attempt_limit:
                rec['status'] = 'error'
                rec['error'] = 'No MAT files downloaded after max attempts'
                rec['lastDownloadError'] = 'No MAT files downloaded after max attempts'
                return 'error', rec
            rec['status'] = 'pending'
            rec['pendingReason'] = 'waiting-for-mat-files'
            rec['lastDownloadError'] = 'waiting-for-mat-files'
            return 'pending', rec
        # Validate downloaded files (no longer moving to subdirectories)
        if local_out_path:
            with self._path_lock:
                self.spectrogram_path = local_out_path
                self.input_path = local_out_path
                self.validate_spectrograms('mat')
        else:
            self.validate_spectrograms('mat')
        if download_audio and rec.get('start') and rec.get('end'):
            try:
                flac_client = ONC(self._onc_token, showInfo=False)
                flac_client.outPath = self.flac_path
                self.download_flac_files(
                    rec['deviceCode'],
                    rec['start'],
                    rec['end'],
                    onc_client=flac_client,
                    max_download_workers=audio_download_workers,
                )
                rec['flac_status'] = 'downloaded'
            except Exception as e:
                rec['flac_status'] = f'error: {e}'
                self.logger.warning(f"Audio download failed for runId={run_id}: {e}")
        rec['status'] = 'downloaded'
        rec['completedAt'] = format_iso_utc(datetime.now(timezone.utc))
        return 'downloaded', rec
    except requests.HTTPError as http_err:
        status_code = getattr(http_err.response, 'status_code', None)
        msg = f"http-{status_code}" if status_code else str(http_err)
        rec['lastDownloadError'] = msg
        self.logger.debug(f"download HTTP error for runId={run_id}: {http_err}")
        transient_codes = {500, 502, 503, 504}
        if status_code in transient_codes or (status_code is None and 'HTTP status 500' in str(http_err)):
            if attempt_limit is not None and rec.get('attempts', 0) >= attempt_limit:
                rec['status'] = 'error'
                rec['error'] = msg
                return 'error', rec
            rec['status'] = 'pending'
            rec['pendingReason'] = msg
            return 'pending', rec
        if attempt_limit is not None and rec.get('attempts', 0) >= attempt_limit:
            rec['status'] = 'error'
            rec['error'] = msg
            return 'error', rec
        rec['status'] = 'pending'
        rec['pendingReason'] = msg
        return 'pending', rec
    except Exception as e:
        # Likely not yet ready; keep pending
        rec['lastDownloadError'] = str(e)
        self.logger.debug(f"download not ready for runId={run_id}: {e}")
        if attempt_limit is not None and rec.get('attempts', 0) >= attempt_limit:
            rec['status'] = 'error'
            rec['error'] = str(e)
            rec['lastDownloadError'] = str(e)
            return 'error', rec
        rec['status'] = 'pending'
        return 'pending', rec


def run_parallel_windows(
    self,
    device_code: str,
    windows: List[Tuple[datetime, datetime]],
    *,
    spectrograms_per_request: int,
    tag: str = 'parallel',
    download_audio: Optional[bool] = None,
    download_flac: Optional[bool] = None,
    audio_download_workers: Optional[int] = None,
    stagger_seconds: float = 3.0,
    max_wait_minutes: int = 45,
    poll_interval_seconds: int = 30,
    max_attempts: int = 6,
    max_download_workers: int = 4,
    data_product_options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Fire off a batch of MAT runs, poll until they're ready, then download them in parallel.

    This is the internal implementation used by the default download API.
    """
    if not windows:
        raise ValueError("No windows provided for parallel run")
    download_audio = _resolve_download_audio(download_audio, download_flac)

    ordered_windows = sorted(windows, key=lambda pair: pair[0])
    range_start = ordered_windows[0][0]
    range_end = ordered_windows[-1][1]
    self.setup_directories('mat', device_code, tag, range_start, range_end)

    total_windows = len(ordered_windows)
    print(f"Submitting {total_windows} requests for {device_code}...")

    wall_start = time.time()
    run_records: List[Dict[str, Any]] = []
    for i, (start_dt, end_dt) in enumerate(ordered_windows, 1):
        if i % 5 == 1 or i == total_windows:
             print(f"Submitting request {i}/{total_windows}...")
        
        rec = self.request_manager.submit_mat_run_no_wait(
            device_code=device_code,
            start_dt=start_dt,
            end_dt=end_dt,
            out_path=self.input_path,
            spectrograms_per_batch=spectrograms_per_request,
            data_product_options=data_product_options,
        )
        run_records.append(rec)
        if stagger_seconds > 0 and i < total_windows:
            time.sleep(stagger_seconds)

    def replace_record(updated: Dict[str, Any]) -> None:
        for idx, existing in enumerate(run_records):
            if existing.get('dpRequestId') == updated.get('dpRequestId'):
                run_records[idx] = updated
                return

    def pending_records() -> List[Dict[str, Any]]:
        return [r for r in run_records if r.get('status') not in ('downloaded', 'error')]

    total_wait_seconds = max_wait_minutes * 60 if max_wait_minutes > 0 else 0

    def attempt_batch(records: List[Dict[str, Any]], *, wait_for_complete: bool, allow_rerun: bool) -> None:
        """Download helper that optionally fans out via thread pool."""
        if max_download_workers and max_download_workers > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_download_workers) as executor:
                future_map = {
                    executor.submit(
                        self.try_download_run,
                        rec,
                        allow_rerun=allow_rerun,
                        download_audio=download_audio,
                        audio_download_workers=audio_download_workers,
                        max_attempts=max_attempts,
                        wait_for_complete=wait_for_complete,
                        poll_interval_seconds=poll_interval_seconds,
                        max_wait_seconds=(total_wait_seconds if wait_for_complete else 0),
                    ): rec
                    for rec in records
                }
                for future in concurrent.futures.as_completed(future_map):
                    base = future_map[future]
                    try:
                        status, updated = future.result()
                    except Exception as exc:
                        updated = {**base, 'status': 'error', 'error': str(exc), 'lastDownloadError': str(exc)}
                    replace_record(updated)
        else:
            for rec in records:
                try:
                    status, updated = self.try_download_run(
                        rec,
                        allow_rerun=allow_rerun,
                        download_audio=download_audio,
                        audio_download_workers=audio_download_workers,
                        max_attempts=max_attempts,
                        wait_for_complete=wait_for_complete,
                        poll_interval_seconds=poll_interval_seconds,
                        max_wait_seconds=(total_wait_seconds if wait_for_complete else 0),
                    )
                except Exception as exc:
                    updated = {**rec, 'status': 'error', 'error': str(exc), 'lastDownloadError': str(exc)}
                replace_record(updated)

    deadline = time.time() + (total_wait_seconds if total_wait_seconds > 0 else float('inf'))
    pass_counter = 0
    while pending_records() and time.time() < deadline:
        batch = pending_records()
        pass_counter += 1
        self.logger.info(f"Parallel poll pass {pass_counter}: {len(batch)} pending")
        attempt_batch(batch, wait_for_complete=False, allow_rerun=True)
        if pending_records():
            time.sleep(max(1, poll_interval_seconds))

    leftovers = [r for r in run_records if r.get('status') != 'downloaded']
    if leftovers:
        for rec in leftovers:
            if rec.get('dpRequestId'):
                try:
                    run_info = self.onc.runDataProduct(rec['dpRequestId'], waitComplete=True)
                    if isinstance(run_info, dict) and run_info.get('runIds'):
                        rec['runIds'] = run_info['runIds']
                except Exception as exc:
                    rec['status'] = 'error'
                    rec['error'] = str(exc)
                    rec['lastDownloadError'] = str(exc)
                    replace_record(rec)
                    continue
            status, updated = self.try_download_run(
                rec,
                allow_rerun=False,
                download_audio=download_audio,
                audio_download_workers=audio_download_workers,
                max_attempts=max_attempts,
                wait_for_complete=False,
            )
            replace_record(updated)

    runs_downloaded = len([r for r in run_records if r.get('status') == 'downloaded'])
    runs_errors = len([r for r in run_records if r.get('status') == 'error'])
    spectrogram_files = len(glob.glob(os.path.join(self.spectrogram_path, '*.mat')))

    return {
        'device': device_code,
        'runs_total': len(run_records),
        'runs_downloaded': runs_downloaded,
        'runs_errors': runs_errors,
        'spectrogram_files': spectrogram_files,
        'input_path': self.input_path,
        'flac_path': self.flac_path,
        'wall_seconds': time.time() - wall_start,
    }


def filter_existing_requests(self, device_code, request_times, extension='mat'):
    """
    Skip requests that already have a matching file downloaded.
    Matching is done on the exact request start timestamp prefix, not just the day.
    """
    # Use flat spectrogram_path (no subdirectories)
    search_paths = [
        os.path.join(self.spectrogram_path, f"{device_code}_*.{extension}"),
    ]
    existing_prefixes = set()
    for pattern in search_paths:
        for file in glob.glob(pattern):
            filename = os.path.basename(file)
            parts = filename.split('_')
            if len(parts) > 1:
                prefix = f"{parts[0]}_{parts[1].split('.')[0]}"  # device_YYYYMMDDTHHMMSS
                existing_prefixes.add(prefix)

    filtered = []
    for ts in request_times:
        if hasattr(ts, 'strftime'):
            prefix = f"{device_code}_{ts.strftime('%Y%m%dT%H%M%S')}"
            if prefix in existing_prefixes:
                continue
        filtered.append(ts)
    skipped = len(request_times) - len(filtered)
    if skipped:
        self.logger.info(f"Skipping {skipped} already-downloaded requests based on timestamp match")
    return filtered
