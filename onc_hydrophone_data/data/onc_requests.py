import os
import contextlib
import time
import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

from onc.onc import ONC
from onc_hydrophone_data.onc.common import start_and_end_strings, format_iso_utc


class ONCRequestManager:
    """
    Manages the lifecycle of ONC data product requests.
    
    Responsibilities:
    1. Submitting new data product requests (specifically Hydrophone Spectrogram Data - HSD).
    2. Polling the status of submitted requests until completion or timeout.
    3. Persisting the state of active/recent runs to disk (JSON queue) to support resumption.
    
    This class handles the interaction with the ONC API for ordering and tracking data products,
    abstracting these details away from the file downloading logic.
    """
    
    def __init__(self, onc_token: str, parent_dir: str, logger: logging.Logger):
        """
        Initialize the Request Manager.

        Args:
            onc_token: The API token for authenticating with Ocean Networks Canada.
            parent_dir: Root directory for the project. Used to store the run queue in .onc_runs/.
            logger: Logger instance for status updates.
        """
        self._onc_token = onc_token
        # Initialize ONC client with showInfo=False to rely on our own progress logging
        self.onc = ONC(onc_token, showInfo=False)
        self.parent_dir = parent_dir
        self.logger = logger
        
        # Directory to store persistent state of ongoing downloads
        self.queue_dir = os.path.join(self.parent_dir, '.onc_runs')
        os.makedirs(self.queue_dir, exist_ok=True)

    def runs_file_path(self, kind: str = 'last24h') -> str:
        """
        Get the absolute path to the JSON file storing run records.

        Args:
            kind: Identifier for the queue type (e.g., 'last24h', 'archived').

        Returns:
            Absolute path to the JSON file.
        """
        return os.path.join(self.queue_dir, f'{kind}_runs.json')

    def load_runs(self, kind: str = 'last24h') -> List[Dict[str, Any]]:
        """
        Load the list of run records from the persistent JSON queue.

        Args:
            kind: Identifier for the queue type.

        Returns:
            A list of dictionary records representing submitted runs.
            Returns an empty list if the file doesn't exist or is corrupt.
        """
        path = self.runs_file_path(kind)
        if not os.path.exists(path):
            return []
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
        except Exception:
            # If the file is corrupt/empty, return empty list rather than crashing
            pass
        return []

    def save_runs(self, kind: str, runs: List[Dict[str, Any]]) -> None:
        """
        Atomically save the list of run records to the persistent JSON queue.
        
        Uses a write-then-rename strategy to prevent data corruption if the process crashes
        during writing.

        Args:
            kind: Identifier for the queue type.
            runs: The list of run records to save.
        """
        path = self.runs_file_path(kind)
        tmp_path = f"{path}.tmp"
        with open(tmp_path, 'w') as f:
            json.dump(runs, f, indent=2)
        os.replace(tmp_path, path)

    def submit_mat_run_no_wait(
        self,
        device_code: str,
        start_dt: datetime,
        end_dt: datetime,
        out_path: str,
        spectrograms_per_batch: int = 6,
        spectral_downsample: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Submit a request for Hydrophone Spectrogram Data (MAT files) to the ONC API asynchronously.

        This method initiates the generation of the data product but does NOT wait for it to complete.
        It returns a record containing the request IDs which can be polled later.

        Args:
            device_code: The unique identifier for the hydrophone (e.g., 'ICLISTENHF6020').
            start_dt: Start datetime for the data range.
            end_dt: End datetime for the data range.
            out_path: Local directory where files should eventually be downloaded.
            spectrograms_per_batch: (Unused parameter retained for interface compatibility).
            spectral_downsample: Quality/Resolution setting. 
                                 None/2 = Plot Resolution (standard), 0 = Full Resolution.

        Returns:
            A dictionary record describing the submitted run, including:
            - dpRequestId: The Request ID.
            - runIds: List of Run IDs (initially empty or populated if immediate).
            - status: Initial status ('submitted').
        """
        # Calculate time strings for the API request
        time_delta = end_dt - start_dt
        start_time, end_time = start_and_end_strings(start_dt, time_delta)
        
        # Configure filters for HSD (Hydrophone Spectrogram Data)
        filters = {
            'dataProductCode': 'HSD',
            'deviceCode': device_code,
            'dateFrom': start_time,
            'dateTo': end_time,
            'extension': 'mat',
            'dpo_hydrophoneDataDiversionMode': 'OD',  # 'OD' = On Demand
            'dpo_spectralDataDownsample': spectral_downsample if spectral_downsample is not None else 0,
        }

        # Step 1: Request the data product
        # This registers the request with ONC
        result = self.onc.requestDataProduct(filters)
        
        dp_request_id = result['dpRequestId'] if isinstance(result, dict) else result
        self.logger.info(f"Submitted request (no-wait) dpRequestId={dp_request_id} for {device_code}")

        # Step 2: Trigger the run
        # This actually starts the processing job. We do not wait for completion here.
        run_data = self.onc.runDataProduct(dp_request_id, waitComplete=False)
                 
        run_ids = None
        if isinstance(run_data, dict) and 'runIds' in run_data:
            run_ids = run_data['runIds']

        # Step 3: Create a tracking record
        rec: Dict[str, Any] = {
            'deviceCode': device_code,
            'dpRequestId': dp_request_id,
            'runIds': run_ids,
            'start': start_time,
            'end': end_time,
            'outPath': out_path,
            'status': 'submitted',
            'createdAt': format_iso_utc(datetime.now(timezone.utc)),
            'attempts': 0,
        }
        return rec

    def wait_for_data_product_ready(
        self,
        dp_request_id: int,
        *,
        max_wait_seconds: int = 900,
        poll_interval_seconds: int = 15,
    ) -> Tuple[bool, str, List[Dict[str, Any]]]:
        """
        Poll ONC's status endpoint until a run is complete or the timeout is reached.

        Args:
            dp_request_id: The Data Product Request ID to poll.
            max_wait_seconds: Maximum time to wait in seconds before giving up.
            poll_interval_seconds: Time to sleep between status checks.

        Returns:
            tuple: (is_ready: bool, status_string: str, full_status_payload: list)
            
            - is_ready: True if status is 'complete', False otherwise.
            - status_string: The final status (e.g., 'complete', 'running', 'error', 'timeout').
            - full_status_payload: The raw status dictionaries returned by the API.
        """
        if not dp_request_id:
            return True, 'missing-request-id', []

        # Create a thread-safe client instance for polling
        # This ensures that if this method is called from multiple threads,
        # we don't share the same client state (though the underlying library might handle it).
        status_client = ONC(self._onc_token, showInfo=False)
        
        deadline = time.time() + max(0, max_wait_seconds)
        last_payload: List[Dict[str, Any]] = []
        
        while True:
            try:
                response = status_client.checkDataProduct(dp_request_id)
                if isinstance(response, list):
                    statuses = response
                elif response:
                    statuses = [response]
                else:
                    statuses = []
            except Exception as exc:
                statuses = [{'status': 'error', 'message': str(exc)}]

            last_payload = statuses
            normalized = [(row.get('status') or '').lower() for row in statuses]

            # Success condition: All runs for this request are complete
            if statuses and all(state == 'complete' for state in normalized):
                return True, 'complete', last_payload
            
            # Failure condition: Any run has failed or been cancelled
            if any(state in ('cancelled', 'error', 'failed') for state in normalized):
                # Return the first failure status found
                return False, next(s for s in normalized if s in ('cancelled', 'error', 'failed')), last_payload

            # Timeout condition
            if max_wait_seconds <= 0 or time.time() >= deadline:
                return False, 'timeout', last_payload

            time.sleep(max(1, poll_interval_seconds))
