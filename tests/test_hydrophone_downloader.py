#!/usr/bin/env python3
"""
Comprehensive test suite for the HydrophoneDownloader class.

Tests cover:
1. Directory setup and folder structure
2. Timestamp parsing and window calculations
3. Request building and JSON parsing
4. Sampling schedule generation
5. File filtering
6. Anomaly detection
7. Download workflows (mocked)
"""

import unittest
from unittest.mock import MagicMock, patch, PropertyMock
import os
import sys
import json
import tempfile
import shutil
from datetime import datetime, date, timedelta, timezone
from pathlib import Path
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Mock the ONC package before importing our modules
sys.modules['onc'] = MagicMock()
sys.modules['onc.onc'] = MagicMock()

from onc_hydrophone_data.data.hydrophone_downloader import (
    HydrophoneDownloader,
    TimestampRequest,
    ensure_timezone_aware,
    FIVE_MINUTES_SECONDS,
)
from onc_hydrophone_data.onc.common import format_iso_utc, start_and_end_strings


class TestEnsureTimezoneAware(unittest.TestCase):
    """Test timezone handling utilities."""
    
    def test_naive_datetime(self):
        """Naive datetime should become UTC."""
        naive = datetime(2024, 1, 15, 12, 30, 0)
        aware = ensure_timezone_aware(naive)
        self.assertEqual(aware.tzinfo, timezone.utc)
        self.assertEqual(aware.hour, 12)
    
    def test_already_aware(self):
        """Already aware datetime should be unchanged."""
        aware_input = datetime(2024, 1, 15, 12, 30, 0, tzinfo=timezone.utc)
        result = ensure_timezone_aware(aware_input)
        self.assertEqual(result, aware_input)
    
    def test_date_object(self):
        """Date objects should be converted to datetime at midnight."""
        d = date(2024, 1, 15)
        result = ensure_timezone_aware(d)
        self.assertIsInstance(result, datetime)
        self.assertEqual(result.hour, 0)
        self.assertEqual(result.minute, 0)
        self.assertEqual(result.tzinfo, timezone.utc)


class TestFormatIsoUtc(unittest.TestCase):
    """Test ISO UTC string formatting."""
    
    def test_format_datetime(self):
        dt = datetime(2024, 4, 1, 10, 30, 45, 123456, tzinfo=timezone.utc)
        result = format_iso_utc(dt)
        self.assertEqual(result, "2024-04-01T10:30:45.123Z")
    
    def test_format_date(self):
        d = date(2024, 4, 1)
        result = format_iso_utc(d)
        self.assertEqual(result, "2024-04-01T00:00:00.000Z")


class TestStartAndEndStrings(unittest.TestCase):
    """Test start/end string calculation."""
    
    def test_timedelta(self):
        start = datetime(2024, 4, 1, 0, 0, 0, tzinfo=timezone.utc)
        delta = timedelta(days=2)
        start_str, end_str = start_and_end_strings(start, delta)
        self.assertEqual(start_str, "2024-04-01T00:00:00.000Z")
        self.assertEqual(end_str, "2024-04-03T00:00:00.000Z")


class TestDirectorySetup(unittest.TestCase):
    """Test directory structure creation."""
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        # Create mock ONC
        with patch('onc_hydrophone_data.data.hydrophone_downloader.ONC'):
            self.downloader = HydrophoneDownloader("FAKE_TOKEN", self.test_dir)
    
    def tearDown(self):
        shutil.rmtree(self.test_dir)
    
    def test_setup_with_device_and_method(self):
        """Test organized directory structure."""
        self.downloader.setup_directories(
            filetype='mat',
            device_code='TESTDEVICE',
            download_method='sampling',
            start_date=datetime(2024, 4, 1),
            end_date=datetime(2024, 4, 5)
        )
        
        # Check paths are correct
        self.assertIn('onc_spectrograms', self.downloader.spectrogram_path)
        self.assertIn('audio', self.downloader.audio_path)
        self.assertIn('TESTDEVICE', self.downloader.spectrogram_path)
        self.assertIn('sampling', self.downloader.spectrogram_path)
        
        # Check directories exist
        self.assertTrue(os.path.exists(self.downloader.spectrogram_path))
        self.assertTrue(os.path.exists(self.downloader.audio_path))
    
    def test_setup_legacy_structure(self):
        """Test legacy directory structure without device/method."""
        self.downloader.setup_directories(filetype='mat')
        
        self.assertIn('onc_spectrograms', self.downloader.spectrogram_path)
        self.assertTrue(os.path.exists(self.downloader.spectrogram_path))
    
    def test_backwards_compatible_aliases(self):
        """Test that legacy aliases work."""
        self.downloader.setup_directories(
            filetype='mat',
            device_code='TEST',
            download_method='test'
        )
        
        # input_path should equal spectrogram_path
        self.assertEqual(self.downloader.input_path, self.downloader.spectrogram_path)
        # flac_path should equal audio_path
        self.assertEqual(self.downloader.flac_path, self.downloader.audio_path)
    
    def test_no_old_style_folders(self):
        """Ensure no processed/rejects folders are created."""
        self.downloader.setup_directories(
            filetype='mat',
            device_code='TEST',
            download_method='test'
        )
        
        # Search for old-style folders
        test_path = Path(self.test_dir)
        old_folders = (
            list(test_path.rglob('processed')) +
            list(test_path.rglob('rejects')) +
            list(test_path.rglob('mat'))
        )
        self.assertEqual(len(old_folders), 0, f"Found old-style folders: {old_folders}")


class TestMethodFolderName(unittest.TestCase):
    """Test folder name generation."""
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        with patch('onc_hydrophone_data.data.hydrophone_downloader.ONC'):
            self.downloader = HydrophoneDownloader("FAKE_TOKEN", self.test_dir)
    
    def tearDown(self):
        shutil.rmtree(self.test_dir)
    
    def test_with_datetime(self):
        result = self.downloader._create_method_folder_name(
            'sampling',
            start_date=datetime(2024, 4, 1),
            end_date=datetime(2024, 4, 5)
        )
        self.assertEqual(result, "sampling_2024-04-01_to_2024-04-05")
    
    def test_with_tuple(self):
        result = self.downloader._create_method_folder_name(
            'range',
            start_date=(2024, 5, 10),
            end_date=(2024, 5, 15)
        )
        self.assertEqual(result, "range_2024-05-10_to_2024-05-15")
    
    def test_start_only(self):
        result = self.downloader._create_method_folder_name(
            'test',
            start_date=datetime(2024, 6, 1)
        )
        self.assertEqual(result, "test_2024-06-01")


class TestTimestampParsing(unittest.TestCase):
    """Test timestamp parsing from various formats."""
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        with patch('onc_hydrophone_data.data.hydrophone_downloader.ONC'):
            self.downloader = HydrophoneDownloader("FAKE_TOKEN", self.test_dir)
    
    def tearDown(self):
        shutil.rmtree(self.test_dir)
    
    def test_parse_datetime(self):
        dt = datetime(2024, 4, 1, 12, 30, 0)
        result = self.downloader._parse_timestamp_value(dt)
        self.assertEqual(result.tzinfo, timezone.utc)
        self.assertEqual(result.hour, 12)
    
    def test_parse_iso_string(self):
        result = self.downloader._parse_timestamp_value("2024-04-01T12:30:00Z")
        self.assertEqual(result.year, 2024)
        self.assertEqual(result.month, 4)
        self.assertEqual(result.hour, 12)
    
    def test_parse_tuple(self):
        result = self.downloader._parse_timestamp_value([2024, 4, 1, 12, 30, 0])
        self.assertEqual(result.year, 2024)
        self.assertEqual(result.hour, 12)


class TestWindowCalculations(unittest.TestCase):
    """Test 5-minute window calculations."""
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        with patch('onc_hydrophone_data.data.hydrophone_downloader.ONC'):
            self.downloader = HydrophoneDownloader("FAKE_TOKEN", self.test_dir)
    
    def tearDown(self):
        shutil.rmtree(self.test_dir)
    
    def test_floor_to_window(self):
        # 12:33:45 should floor to 12:30:00
        dt = datetime(2024, 4, 1, 12, 33, 45, tzinfo=timezone.utc)
        result = self.downloader._floor_to_window(dt)
        self.assertEqual(result.minute, 30)
        self.assertEqual(result.second, 0)
    
    def test_ceil_to_window(self):
        # 12:31:00 should ceil to 12:35:00
        dt = datetime(2024, 4, 1, 12, 31, 0, tzinfo=timezone.utc)
        result = self.downloader._ceil_to_window(dt)
        self.assertEqual(result.minute, 35)
    
    def test_build_request_windows(self):
        start = datetime(2024, 4, 1, 12, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 4, 1, 12, 15, 0, tzinfo=timezone.utc)
        
        windows = self.downloader._build_request_windows(start, end)
        
        # Should get 3 windows of 5 minutes each
        self.assertEqual(len(windows), 3)
        
        # Each window should be 5 minutes
        for w_start, w_end in windows:
            self.assertEqual((w_end - w_start).seconds, 300)


class TestTimestampFromFilename(unittest.TestCase):
    """Test timestamp extraction from ONC filenames."""
    
    def test_standard_filename(self):
        path = "/data/ICLISTENHF6324_20240401T123000.mat"
        result = HydrophoneDownloader._timestamp_from_filename(path)
        
        self.assertEqual(result.year, 2024)
        self.assertEqual(result.month, 4)
        self.assertEqual(result.day, 1)
        self.assertEqual(result.hour, 12)
        self.assertEqual(result.minute, 30)
    
    def test_no_timestamp(self):
        path = "/data/random_file.mat"
        result = HydrophoneDownloader._timestamp_from_filename(path)
        self.assertIsNone(result)


class TestFilterExistingRequests(unittest.TestCase):
    """Test filtering of already-downloaded files."""
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        with patch('onc_hydrophone_data.data.hydrophone_downloader.ONC'):
            self.downloader = HydrophoneDownloader("FAKE_TOKEN", self.test_dir)
        self.downloader.setup_directories(filetype='mat', device_code='TEST', download_method='test')
        
        # Create some fake existing files
        existing_file = os.path.join(self.downloader.spectrogram_path, "TEST_20240401T120000.mat")
        Path(existing_file).touch()
    
    def tearDown(self):
        shutil.rmtree(self.test_dir)
    
    def test_filters_existing(self):
        request_times = [
            datetime(2024, 4, 1, 12, 0, 0, tzinfo=timezone.utc),  # Exists
            datetime(2024, 4, 1, 12, 5, 0, tzinfo=timezone.utc),  # New
            datetime(2024, 4, 1, 12, 10, 0, tzinfo=timezone.utc), # New
        ]
        
        filtered = self.downloader.filter_existing_requests("TEST", request_times)
        
        # Should only have 2 (the existing one filtered out)
        self.assertEqual(len(filtered), 2)


class TestJSONRequestParsing(unittest.TestCase):
    """Test JSON request file parsing."""
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        with patch('onc_hydrophone_data.data.hydrophone_downloader.ONC'):
            self.downloader = HydrophoneDownloader("FAKE_TOKEN", self.test_dir)
    
    def tearDown(self):
        shutil.rmtree(self.test_dir)
    
    def test_parse_new_format(self):
        payload = {
            "defaults": {"pad_seconds": 60},
            "requests": [
                {
                    "deviceCode": "TESTDEVICE",
                    "timestamp": "2024-04-01T12:00:00Z"
                }
            ]
        }
        
        requests = self.downloader._coerce_timestamp_requests(
            payload,
            default_pad_seconds=0,
            default_tag='test',
            clip_outputs=None,
            spectrogram_format='mat',
            download_audio=None,
            download_spectrogram=None,
        )
        
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].device_code, "TESTDEVICE")
        # Pad should come from defaults
        self.assertEqual(requests[0].pad_before, 60)
    
    def test_parse_legacy_format(self):
        """Test legacy format: {device: [[Y,M,D,H,M,S], ...]}"""
        payload = {
            "TESTDEVICE": [
                [2024, 4, 1, 12, 0, 0],
                [2024, 4, 1, 12, 5, 0]
            ]
        }
        
        requests = self.downloader._coerce_timestamp_requests(
            payload,
            default_pad_seconds=30,
            default_tag='legacy',
            clip_outputs=None,
            spectrogram_format='mat',
            download_audio=None,
            download_spectrogram=None,
        )
        
        self.assertEqual(len(requests), 2)
        self.assertEqual(requests[0].device_code, "TESTDEVICE")
        self.assertEqual(requests[0].tag, 'legacy')


class TestTimestampRequestBuilding(unittest.TestCase):
    """Test building TimestampRequest objects from dicts."""
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        with patch('onc_hydrophone_data.data.hydrophone_downloader.ONC'):
            self.downloader = HydrophoneDownloader("FAKE_TOKEN", self.test_dir)
    
    def tearDown(self):
        shutil.rmtree(self.test_dir)
    
    def test_build_basic_request(self):
        data = {
            "deviceCode": "MYDEVICE",
            "timestamp": "2024-04-01T10:00:00Z",
            "download_audio": True,
            "download_spectrogram": True,
            "label": "Test whale call"
        }
        
        req = self.downloader._build_request_from_dict(
            data,
            defaults={},
            default_pad_seconds=0,
            default_tag='test',
            clip_outputs=False,
            spectrogram_format='mat',
            download_audio=None,
            download_spectrogram=None,
        )
        
        self.assertEqual(req.device_code, "MYDEVICE")
        self.assertTrue(req.want_audio)
        self.assertTrue(req.want_spectrogram)
        self.assertEqual(req.description, "Test whale call")
    
    def test_build_with_time_range(self):
        data = {
            "deviceCode": "MYDEVICE",
            "start": "2024-04-01T10:00:00Z",
            "end": "2024-04-01T10:30:00Z",
        }
        
        req = self.downloader._build_request_from_dict(
            data,
            defaults={},
            default_pad_seconds=0,
            default_tag='test',
            clip_outputs=True,
            spectrogram_format='mat',
            download_audio=None,
            download_spectrogram=None,
        )
        
        # pad_after should be set to cover the range
        self.assertGreater(req.pad_after, 0)


class TestAnomalyDetection(unittest.TestCase):
    """Test spectrogram anomaly detection."""
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        with patch('onc_hydrophone_data.data.hydrophone_downloader.ONC'):
            self.downloader = HydrophoneDownloader("FAKE_TOKEN", self.test_dir)
        self.downloader.setup_directories(filetype='mat', device_code='TEST', download_method='test')
    
    def tearDown(self):
        shutil.rmtree(self.test_dir)
    
    def test_no_file(self):
        """Test handling of non-existent file."""
        result = self.downloader.check_for_anomalies("/nonexistent/file.mat")
        self.assertTrue(result['has_anomaly'])
        self.assertIn('Error', result['issues'][0])


class TestValidateSpectrograms(unittest.TestCase):
    """Test spectrogram validation method."""
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        with patch('onc_hydrophone_data.data.hydrophone_downloader.ONC'):
            self.downloader = HydrophoneDownloader("FAKE_TOKEN", self.test_dir)
        self.downloader.setup_directories(filetype='mat', device_code='TEST', download_method='test')
    
    def tearDown(self):
        shutil.rmtree(self.test_dir)
    
    def test_empty_directory(self):
        """Test validation of empty directory."""
        result = self.downloader.validate_spectrograms('mat')
        
        self.assertEqual(result['total_files'], 0)
        self.assertEqual(result['anomalies'], 0)


class TestSamplingSchedule(unittest.TestCase):
    """Test sampling schedule generation."""
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        with patch('onc_hydrophone_data.data.hydrophone_downloader.ONC') as mock_onc:
            # Mock the getListByDevice call
            mock_instance = MagicMock()
            mock_instance.getListByDevice.return_value = {
                'files': [
                    {'dateFrom': '2024-04-01T00:00:00.000Z'},
                    {'dateFrom': '2024-04-01T12:00:00.000Z'},
                ]
            }
            mock_onc.return_value = mock_instance
            self.downloader = HydrophoneDownloader("FAKE_TOKEN", self.test_dir)
    
    def tearDown(self):
        shutil.rmtree(self.test_dir)
    
    def test_schedule_parameters(self):
        """Test that schedule parameters are validated."""
        # This tests the basic structure - actual schedule depends on API data
        self.assertIsNotNone(self.downloader.sampling_schedule)


class TestDownloadHelpers(unittest.TestCase):
    """Test download helper functions."""
    
    def test_build_sampling_windows(self):
        from onc_hydrophone_data.utils.download_helpers import build_sampling_windows
        
        result = build_sampling_windows(
            device_code="TEST",
            start_dt=datetime(2024, 4, 1, tzinfo=timezone.utc),
            end_dt=datetime(2024, 4, 2, tzinfo=timezone.utc),
            total_spectrograms=10,
            spectrograms_per_request=5
        )
        
        self.assertIn("TEST", result)
        self.assertEqual(len(result["TEST"]), 2)  # 10 specs / 5 per request = 2 requests
    
    def test_build_hsd_filters(self):
        from onc_hydrophone_data.utils.download_helpers import build_hsd_filters
        
        result = build_hsd_filters(
            device_code="TEST",
            start=datetime(2024, 4, 1, tzinfo=timezone.utc),
            end=datetime(2024, 4, 1, 0, 5, 0, tzinfo=timezone.utc),
            downsample=2
        )
        
        self.assertEqual(result['deviceCode'], "TEST")
        self.assertEqual(result['dataProductCode'], 'HSD')
        self.assertEqual(result['extension'], 'mat')


class TestDownloadWorkflow(unittest.TestCase):
    """Test download workflow with mocked ONC API."""
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.mock_onc_patcher = patch('onc_hydrophone_data.data.hydrophone_downloader.ONC')
        self.mock_onc = self.mock_onc_patcher.start()
        
        # Configure mock
        self.mock_onc_instance = MagicMock()
        self.mock_onc.return_value = self.mock_onc_instance
        
        self.downloader = HydrophoneDownloader("FAKE_TOKEN", self.test_dir)
    
    def tearDown(self):
        self.mock_onc_patcher.stop()
        shutil.rmtree(self.test_dir)
    
    def test_download_mat_setup(self):
        """Test that download properly sets up directories."""
        self.downloader.setup_directories(
            filetype='mat',
            device_code='TESTDEVICE',
            download_method='test',
            start_date=datetime(2024, 4, 1)
        )
        
        # ONC output path should be set to spectrogram_path
        self.assertEqual(
            self.downloader.onc.outPath,
            self.downloader.spectrogram_path
        )


class TestIntegrationScenarios(unittest.TestCase):
    """Integration tests for common usage scenarios."""
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.mock_onc_patcher = patch('onc_hydrophone_data.data.hydrophone_downloader.ONC')
        self.mock_onc = self.mock_onc_patcher.start()
        self.mock_onc_instance = MagicMock()
        self.mock_onc.return_value = self.mock_onc_instance
        
        self.downloader = HydrophoneDownloader("FAKE_TOKEN", self.test_dir)
    
    def tearDown(self):
        self.mock_onc_patcher.stop()
        shutil.rmtree(self.test_dir)
    
    def test_json_file_workflow(self):
        """Test complete JSON file processing workflow."""
        # Create a test JSON file
        json_content = {
            "defaults": {"pad_seconds": 30},
            "requests": [
                {
                    "deviceCode": "TESTDEVICE",
                    "timestamp": "2024-04-01T12:00:00Z",
                    "label": "Test event"
                }
            ]
        }
        
        json_path = os.path.join(self.test_dir, "test_requests.json")
        with open(json_path, 'w') as f:
            json.dump(json_content, f)
        
        # Mock the API responses
        self.mock_onc_instance.requestDataProduct.return_value = {'dpRequestId': 123}
        self.mock_onc_instance.runDataProduct.return_value = {'runIds': [456]}
        
        # This should not raise
        # Note: actual download would require more complex mocking
        with open(json_path, 'r') as f:
            payload = json.load(f)
        
        requests = self.downloader._coerce_timestamp_requests(
            payload,
            default_pad_seconds=0,
            default_tag='test',
            clip_outputs=None,
            spectrogram_format='mat',
            download_audio=None,
            download_spectrogram=None,
        )
        
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].pad_before, 30)
        self.assertEqual(requests[0].description, "Test event")


def run_tests():
    """Run all tests and return results."""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
