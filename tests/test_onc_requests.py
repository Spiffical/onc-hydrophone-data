
import unittest
from unittest.mock import MagicMock, patch
import os
import json
from datetime import datetime, timezone, timedelta
import tempfile
import shutil

# Mock the entire onc package before importing ours to avoid real API calls or import errors
sys_modules_patch = patch.dict('sys.modules', {'onc': MagicMock(), 'onc.onc': MagicMock()})
sys_modules_patch.start()

from src.data.onc_requests import ONCRequestManager
from src.onc.common import format_iso_utc

class TestONCRequestManager(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.logger = MagicMock()
        self.token = "FAKE_TOKEN"
        self.manager = ONCRequestManager(self.token, self.test_dir, self.logger)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_run_persistence(self):
        # Create a fake run record
        run_record = {
            'deviceCode': 'TEST_DEVICE',
            'dpRequestId': 12345,
            'status': 'submitted'
        }
        
        # Save it
        self.manager.save_runs('test', [run_record])
        
        # Load it back
        loaded_runs = self.manager.load_runs('test')
        self.assertEqual(len(loaded_runs), 1)
        self.assertEqual(loaded_runs[0]['deviceCode'], 'TEST_DEVICE')
        self.assertEqual(loaded_runs[0]['dpRequestId'], 12345)

    def test_submit_mat_run_no_wait(self):
        # Mock the ONC client inside the manager
        self.manager.onc.requestDataProduct.return_value = {'dpRequestId': 999}
        self.manager.onc.runDataProduct.return_value = {'runIds': [1001]}

        start_dt = datetime(2021, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        end_dt = start_dt + timedelta(minutes=5)
        
        rec = self.manager.submit_mat_run_no_wait(
            device_code="TEST_DEVICE",
            start_dt=start_dt,
            end_dt=end_dt,
            out_path="/tmp/out",
            spectrograms_per_batch=1
        )

        # precise check of args passed to requestDataProduct
        # Ensure our date string logic calls are correct
        # start: 2021-01-01T12:00:00.000Z
        # end: 2021-01-01T12:05:00.000Z
        calls = self.manager.onc.requestDataProduct.call_args
        self.assertIsNotNone(calls)
        args, kwargs = calls
        filters = args[0]
        
        self.assertEqual(filters['deviceCode'], "TEST_DEVICE")
        self.assertEqual(filters['dateFrom'], "2021-01-01T12:00:00.000Z")
        self.assertEqual(filters['dateTo'], "2021-01-01T12:05:00.000Z")
        self.assertEqual(filters['dataProductCode'], 'HSD')
        
        # Check returned record
        self.assertEqual(rec['dpRequestId'], 999)
        self.assertEqual(rec['runIds'], [1001])
        self.assertEqual(rec['status'], 'submitted')


    @patch('src.data.onc_requests.ONC')
    def test_wait_for_data_product_ready_complete(self, mock_onc_cls):
        # Mock status client
        mock_client = MagicMock()
        mock_onc_cls.return_value = mock_client
        
        # Simulate 'complete' response
        mock_client.checkDataProduct.return_value = [{'status': 'complete'}]
        
        ready, status, payload = self.manager.wait_for_data_product_ready(
            dp_request_id=123,
            poll_interval_seconds=0  # fast test
        )
        
        self.assertTrue(ready)
        self.assertEqual(status, 'complete')

    @patch('src.data.onc_requests.ONC')
    def test_wait_for_data_product_ready_timeout(self, mock_onc_cls):
        # Mock status client
        mock_client = MagicMock()
        mock_onc_cls.return_value = mock_client
        
        # Simulate 'running' response forever
        mock_client.checkDataProduct.return_value = [{'status': 'running'}]
        
        ready, status, payload = self.manager.wait_for_data_product_ready(
            dp_request_id=123,
            max_wait_seconds=1, # short timeout
            poll_interval_seconds=0.1
        )
        
        self.assertFalse(ready)
        self.assertEqual(status, 'timeout')


if __name__ == '__main__':
    unittest.main()
