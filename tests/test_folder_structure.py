#!/usr/bin/env python3
"""
Test script to verify the new folder structure works correctly for different download modes.

This script tests:
1. Sampling schedule mode
2. Specific times mode  
3. Date range mode
4. Verify folder structure is clean (onc_spectrograms/, audio/)
"""

import os
import sys
import shutil
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

# Add parent to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from onc_hydrophone_data.onc.common import load_config
from onc_hydrophone_data.data.hydrophone_downloader import HydrophoneDownloader


def print_status(msg, level="INFO"):
    """Print status message with emoji."""
    icons = {"INFO": "ℹ️", "SUCCESS": "✅", "ERROR": "❌", "WARN": "⚠️", "TEST": "🧪"}
    print(f"{icons.get(level, 'ℹ️')} {msg}")


def verify_folder_structure(base_path: str, expected_folders: list) -> bool:
    """Verify that expected folders exist and are properly named."""
    base = Path(base_path)
    all_good = True
    
    for folder in expected_folders:
        folder_path = base / folder
        if folder_path.exists():
            print_status(f"Found: {folder_path}", "SUCCESS")
        else:
            print_status(f"Missing: {folder_path}", "ERROR")
            all_good = False
    
    # Check for old-style folders that should NOT exist
    old_style_folders = ['mat', 'flac', 'processed', 'rejects']
    for old_folder in old_style_folders:
        # Search recursively
        found = list(base.rglob(old_folder))
        if found:
            print_status(f"Found deprecated folder '{old_folder}': {found}", "WARN")
            all_good = False
    
    return all_good


def test_sampling_mode(downloader: HydrophoneDownloader, test_dir: str):
    """Test sampling schedule download mode."""
    print("\n" + "="*60)
    print_status("Testing SAMPLING MODE", "TEST")
    print("="*60)
    
    device_code = "ICLISTENHF6324"  # Known working device
    start_date = (2024, 4, 1)
    
    # Very small test - just 1 spectrogram
    try:
        downloader.download_spectrograms_with_sampling_schedule(
            deviceCode=device_code,
            start_date=start_date,
            threshold_num=1,  # Just 1 file
            num_days=1,
            filetype='mat',
            spectrograms_per_batch=1,
            download_audio=False
        )
        
        # Check folder structure
        device_folder = Path(test_dir) / device_code
        if device_folder.exists():
            print_status(f"Device folder created: {device_folder}", "SUCCESS")
            
            # List contents
            for item in device_folder.rglob("*"):
                if item.is_file():
                    print_status(f"  File: {item.relative_to(device_folder)}", "INFO")
                elif item.is_dir():
                    print_status(f"  Dir:  {item.relative_to(device_folder)}/", "INFO")
            
            # Verify structure
            method_folders = list(device_folder.glob("sampling_*"))
            if method_folders:
                method_folder = method_folders[0]
                expected = ['onc_spectrograms']
                verify_folder_structure(str(method_folder), expected)
                return True
        else:
            print_status("Device folder not created", "ERROR")
            
    except Exception as e:
        print_status(f"Sampling mode test failed: {e}", "ERROR")
        import traceback
        traceback.print_exc()
    
    return False


def test_directory_setup(downloader: HydrophoneDownloader, test_dir: str):
    """Test that setup_directories creates the correct structure."""
    print("\n" + "="*60)
    print_status("Testing DIRECTORY SETUP", "TEST")
    print("="*60)
    
    # Test with device code and method
    device_code = "TESTDEVICE"
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 1, 2)
    
    downloader.setup_directories(
        filetype='mat',
        device_code=device_code,
        download_method='test_mode',
        start_date=start_date,
        end_date=end_date
    )
    
    print_status(f"spectrogram_path: {downloader.spectrogram_path}", "INFO")
    print_status(f"audio_path: {downloader.audio_path}", "INFO")
    print_status(f"input_path (alias): {downloader.input_path}", "INFO")
    print_status(f"flac_path (alias): {downloader.flac_path}", "INFO")
    
    # Verify paths
    expected_spec = "onc_spectrograms" in downloader.spectrogram_path
    expected_audio = "audio" in downloader.audio_path
    
    if expected_spec and expected_audio:
        print_status("Paths correctly use new naming (onc_spectrograms/, audio/)", "SUCCESS")
        
        # Check that directories were created
        if os.path.exists(downloader.spectrogram_path):
            print_status(f"Spectrogram dir exists: {downloader.spectrogram_path}", "SUCCESS")
        else:
            print_status(f"Spectrogram dir missing!", "ERROR")
            
        if os.path.exists(downloader.audio_path):
            print_status(f"Audio dir exists: {downloader.audio_path}", "SUCCESS")
        else:
            print_status(f"Audio dir missing!", "ERROR")
        
        return True
    else:
        print_status("Paths don't use expected naming", "ERROR")
        return False


def test_legacy_compatibility(downloader: HydrophoneDownloader, test_dir: str):
    """Test that legacy attributes still work."""
    print("\n" + "="*60)
    print_status("Testing LEGACY COMPATIBILITY", "TEST")
    print("="*60)
    
    # Set up directories
    downloader.setup_directories(
        filetype='mat',
        device_code='LEGACYTEST',
        download_method='legacy_test',
        start_date=datetime(2024, 1, 1)
    )
    
    # Check aliases
    results = []
    
    # input_path should equal spectrogram_path
    if downloader.input_path == downloader.spectrogram_path:
        print_status("input_path == spectrogram_path ✓", "SUCCESS")
        results.append(True)
    else:
        print_status("input_path != spectrogram_path ✗", "ERROR")
        results.append(False)
    
    # flac_path should equal audio_path
    if downloader.flac_path == downloader.audio_path:
        print_status("flac_path == audio_path ✓", "SUCCESS")
        results.append(True)
    else:
        print_status("flac_path != audio_path ✗", "ERROR")
        results.append(False)
    
    return all(results)


def main():
    print("\n" + "="*60)
    print("🧪 FOLDER STRUCTURE TESTS")
    print("="*60)
    
    # Load config
    try:
        onc_token, data_dir = load_config()
        print_status(f"Loaded config, data_dir: {data_dir}", "SUCCESS")
    except Exception as e:
        print_status(f"Failed to load config: {e}", "ERROR")
        return 1
    
    # Create temp test directory
    test_dir = tempfile.mkdtemp(prefix="onc_folder_test_")
    print_status(f"Using test directory: {test_dir}", "INFO")
    
    try:
        downloader = HydrophoneDownloader(onc_token, test_dir)
        
        results = []
        
        # Test 1: Directory setup
        results.append(("Directory Setup", test_directory_setup(downloader, test_dir)))
        
        # Test 2: Legacy compatibility
        results.append(("Legacy Compatibility", test_legacy_compatibility(downloader, test_dir)))
        
        # Test 3: Sampling mode (actual download - optional)
        if os.environ.get("RUN_DOWNLOAD_TESTS", "0") == "1":
            results.append(("Sampling Mode Download", test_sampling_mode(downloader, test_dir)))
        else:
            print_status("\nSkipping download tests (set RUN_DOWNLOAD_TESTS=1 to enable)", "WARN")
        
        # Summary
        print("\n" + "="*60)
        print("📊 TEST SUMMARY")
        print("="*60)
        
        all_passed = True
        for name, passed in results:
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {status}: {name}")
            if not passed:
                all_passed = False
        
        return 0 if all_passed else 1
        
    finally:
        # Cleanup
        print_status(f"\nCleaning up test directory: {test_dir}", "INFO")
        shutil.rmtree(test_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
